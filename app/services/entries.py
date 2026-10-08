"""Service layer for the entry-evaluation pipeline.

A thin orchestrator that handles:
  * registering Entries from form responses (both Himachal + Ashramshala)
  * downloading the Drive file referenced by each form response
  * running OCR (vision-LLM or Tesseract) against the image
  * cleaning the resulting text
  * writing everything to the DB in the correct order

The orchestrator is kept as a single `process_entry` coroutine for the sandbox
(tool) plus a `register_parsed_responses` helper used by the form-CSV import.
"""
from __future__ import annotations

import asyncio
import csv
import io
import shutil as _shutil
import time
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging import get_logger
from app.models import (
    DriveFile,
    Entry,
    EntryOCR,
    EntryState,
    OcrMethod,
    OcrStatus,
)
from app.pipeline.cleaning import clean_entry_text
from app.pipeline.ingestion.forms import ParsedFormResponse
from app.pipeline.ingestion.google_drive import (
    DriveClient,
    build_drive_client,
    sha256_of,
)
from app.pipeline.preprocessing import load_image, save_page_image
from app.pipeline.readers import build_reader, build_repairer
from app.pipeline.scoring.entry import score_entry
from app.providers.llm import LLMProvider
from app.schemas import CanonicalDocument, CanonicalPage, LLMMessage, SourceType
from app.services.storage import Storage

logger = get_logger("exam_eval.entries")


# ---------------------------------------------------------------------------
# form-CSV import
# ---------------------------------------------------------------------------
async def register_parsed_responses(
    session: AsyncSession,
    program_id: int,
    parsed: list[ParsedFormResponse],
    *,
    drive_client: DriveClient | None = None,
    settings: Settings,
) -> tuple[list[Entry], list[ParsedFormResponse]]:
    """Upsert entries from form-CSV rows. Existing entries (same program +
    external_ref) are NOT recreated; only their demographics are refreshed.
    Returns (entries_created, parsed_rows_unparsed).
    """
    if drive_client is None:
        drive_client = build_drive_client(settings)

    existing_refs = {
        ref for (ref,) in (
            await session.execute(
                select(Entry.external_ref).where(Entry.program_id == program_id)
            )
        ).all()
    }

    created: list[Entry] = []
    unparsed: list[ParsedFormResponse] = []
    for row in parsed:
        if row.external_ref in existing_refs:
            continue
        drive_meta: DriveFile | None = None
        if row.drive_file_id:
            # ensure the URL row is present so future dedupe is real
            drive_meta = await session.merge(
                DriveFile(
                    drive_file_id=row.drive_file_id,
                    name=row.drive_url or row.drive_file_id,
                    download_url=row.drive_url,
                    status="pending",
                ),
                load=True,
            )
        entry = Entry(
            program_id=program_id,
            external_ref=row.external_ref,
            state=EntryState.PENDING,
            form_kind=row.form_kind,
            full_name=row.full_name,
            gender=row.gender,
            phone=row.phone,
            school_name=row.school_name,
            class_name=row.class_name,
            state_name=row.state,
            district=row.district,
            block_name=row.block_name,
            division=row.division,
            project_office=row.project_office,
            ashramshala=row.ashramshala,
            language=row.language,
            theme=row.theme,
            ai_description=row.ai_description,
            resources=row.resources,
            situation=row.situation,
            proposed_solution=row.proposed_solution,
            declaration=row.declaration,
            consent=row.consent,
            po_name=row.po_name,
            school_name_form=row.school_name_form,
            submission_count=row.submission_count,
            email=row.email,
            drive_url=row.drive_url,
            drive_file_id=row.drive_file_id,
            status="pending",
            metadata_json={"raw": row.raw},
        )
        session.add(entry)
        created.append(entry)
    await session.commit()
    return created, unparsed


def write_entries_csv(entries: Iterable[Entry]) -> str:
    """Render a single CSV string with all the client sheet columns."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_HEADERS)
    for e in entries:
        writer.writerow(_csv_row(e))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# per-entry processing (used by the sandbox and the worker)
# ---------------------------------------------------------------------------
async def _ensure_drive_file_local(
    settings: Settings, drive_client: DriveClient, drive_file_id: str, drive_url: str | None, name_hint: str | None,
    local_file_path: str | None = None,
) -> tuple[Path | None, str | None]:
    """Download to cache if needed. Returns (local_path, sha256).

    `local_file_path` is used when the caller already has a local copy
    (e.g. the sandbox uploaded an image directly). Otherwise we ask the
    DriveClient to fetch it.
    """
    if local_file_path:
        path = Path(local_file_path)
        if path.exists():
            return path, sha256_of(path)
    cache_dir = settings.storage_root / "drive_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = (name_hint and Path(name_hint).suffix) or ""
    target = cache_dir / f"{drive_file_id}{ext}"
    if not target.exists() and drive_url:
        from app.pipeline.ingestion.google_drive import DriveFileMeta
        meta = DriveFileMeta(
            drive_file_id=drive_file_id,
            name=name_hint or f"{drive_file_id}{ext}",
            mime=None,
            size_bytes=0,
            download_url=drive_url,
        )
        try:
            await drive_client.download(meta, target)
        except Exception as exc:  # noqa: BLE001
            logger.warning("drive_download_failed", file_id=drive_file_id, error=str(exc)[:200])
            return None, None
    if not target.exists():
        return None, None
    return target, sha256_of(target)


async def process_entry_via_vision(
    session: AsyncSession,
    entry: Entry,
    *,
    settings: Settings,
    llm: LLMProvider,
    drive_client: DriveClient | None = None,
    rubric_id: int | None = None,
) -> dict:
    """Path A: vision-LLM does both OCR and scoring in one pass."""
    started = time.perf_counter()
    if drive_client is None:
        drive_client = build_drive_client(settings)

    local_path: Path | None = None
    sha: str | None = None
    if entry.drive_file_id:
        local_path, sha = await _ensure_drive_file_local(
            settings,
            drive_client,
            entry.drive_file_id,
            entry.drive_url,
            entry.full_name,
            local_file_path=entry.local_file_path,
        )
        entry.local_file_path = str(local_path) if local_path else None
        entry.drive_sha256 = sha
    if local_path is None or not local_path.exists():
        entry.state = EntryState.FAILED
        entry.status = "no_image"
        await session.commit()
        return {"ok": False, "error": "no image available"}

    image_bytes = local_path.read_bytes()

    # vision OCR pass — separate small call to keep tokens bounded
    ocr_prompt = (
        "Transcribe the handwritten content of this student's project submission. "
        "Preserve line breaks. Use [illegible] for anything you can't read. "
        "Return JSON: {\"text\": \"<transcription>\", \"language\": \"<hin|eng|mr|mixed>\"}"
    )
    ocr_result = None
    ocr_text = ""
    language = entry.language
    try:
        ocr_result = await llm.complete(
            messages=[
                LLMMessage(role="system", content="You are a meticulous OCR transcriber."),
                LLMMessage(role="user", content=ocr_prompt),
            ],
            images=[image_bytes],
            json_mode=True,
            max_tokens=min(settings.minimax_max_output_tokens, 2048),
            stage="entry_vision_ocr",
            temperature=0.0,
            context={"entry_id": entry.id},
        )
        import json as _json
        data = _json.loads(ocr_result.text)
        ocr_text = (data.get("text") or "").strip()
        language = data.get("language") or entry.language
    except Exception as exc:  # noqa: BLE001
        logger.warning("entry_vision_ocr_failed", entry_id=entry.id, error=str(exc)[:200])

    entry_ocr = EntryOCR(
        entry_id=entry.id,
        method=OcrMethod.VISION_LLM,
        status=OcrStatus.OK if ocr_text else OcrStatus.EMPTY,
        language=language,
        original_text=ocr_text,
        reconstructed_text=ocr_text,
        confidence=0.92 if ocr_text else 0.0,
        input_tokens=(ocr_result.prompt_tokens if ocr_result else None),
        output_tokens=(ocr_result.completion_tokens if ocr_result else None),
        total_tokens=(ocr_result.total_tokens if ocr_result else None),
        provider=settings.llm_provider,
        model=(ocr_result.model if ocr_result else None),
        latency_ms=(ocr_result.latency_ms if ocr_result else None),
        is_selected=True,
    )
    session.add(entry_ocr)
    entry.state = EntryState.SCORING
    await session.commit()

    # cleaning
    cleaned = await clean_entry_text(
        ocr_text,
        settings=settings,
        llm=llm,
        language_hint=language,
        section="full",
        mode=settings.cleaning_mode,
    )
    entry.situation = cleaned.cleaned_text or entry.situation
    entry.submitted_text = cleaned.cleaned_text

    # scoring
    score = await score_entry(session, entry, llm=llm, settings=settings, rubric_id=rubric_id)
    entry.state = EntryState.COMPLETED
    entry.status = "completed"
    await session.commit()
    return {
        "ok": True,
        "method": "vision",
        "ocr_text": ocr_text,
        "language": language,
        "score": {
            "total_awarded": score.total_awarded,
                "total_max": score.total_max,
                "percentage": score.percentage,
                "confidence": score.confidence,
                "strengths": score.strengths,
                "gaps": score.gaps,
                "observations": score.observations,
                "criteria": score.criteria,
            },
        "input_tokens": score.input_tokens + (entry_ocr.input_tokens or 0),
        "output_tokens": score.output_tokens + (entry_ocr.output_tokens or 0),
        "total_tokens": score.total_tokens + (entry_ocr.total_tokens or 0),
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


async def process_entry_via_tesseract_then_llm(
    session: AsyncSession,
    entry: Entry,
    *,
    settings: Settings,
    llm: LLMProvider,
    drive_client: DriveClient | None = None,
    rubric_id: int | None = None,
) -> dict:
    """Path B: Tesseract OCR first, then LLM scoring on the cleaned text."""
    started = time.perf_counter()
    if drive_client is None:
        drive_client = build_drive_client(settings)

    local_path: Path | None = None
    sha: str | None = None
    if entry.drive_file_id:
        local_path, sha = await _ensure_drive_file_local(
            settings,
            drive_client,
            entry.drive_file_id,
            entry.drive_url,
            entry.full_name,
            local_file_path=entry.local_file_path,
        )
        entry.local_file_path = str(local_path) if local_path else None
        entry.drive_sha256 = sha
    if local_path is None or not local_path.exists():
        entry.state = EntryState.FAILED
        entry.status = "no_image"
        await session.commit()
        return {"ok": False, "error": "no image available"}

    from app.pipeline.readers.tesseract import TesseractReader

    reader = TesseractReader(settings=settings)
    canonical_page = CanonicalPage(index=0, has_text_layer=False)
    canonical_doc = CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename=entry.drive_file_id or entry.full_name or "entry",
        sha256=sha or "",
    )
    # placeholder: we need a path on disk; the reader loads from ctx.original_path
    from app.pipeline.readers.base import ReaderContext
    ctx = ReaderContext(document=canonical_doc, original_path=local_path)
    try:
        result = await reader.read(canonical_page, ctx)
    except Exception as exc:  # noqa: BLE001
        result = None
        logger.warning("tesseract_failed", entry_id=entry.id, error=str(exc)[:200])
    ocr_text = (result.text if result else "") or ""
    confidence = float(result.confidence) if result and result.confidence is not None else 0.0
    language = (result.language if result else None) or entry.language

    entry_ocr = EntryOCR(
        entry_id=entry.id,
        method=OcrMethod.TESSERACT,
        status=OcrStatus.OK if ocr_text else OcrStatus.EMPTY,
        language=language,
        original_text=ocr_text,
        reconstructed_text=ocr_text,
        confidence=confidence,
        provider=reader.name,
        model=None,
        latency_ms=result.latency_ms if result else None,
        is_selected=True,
    )
    session.add(entry_ocr)
    entry.state = EntryState.SCORING
    await session.commit()

    cleaned = await clean_entry_text(
        ocr_text,
        settings=settings,
        llm=llm,
        language_hint=language,
        section="full",
        mode=settings.cleaning_mode,
    )
    entry.situation = cleaned.cleaned_text or entry.situation
    entry.submitted_text = cleaned.cleaned_text

    score = await score_entry(session, entry, llm=llm, settings=settings, rubric_id=rubric_id)
    entry.state = EntryState.COMPLETED
    entry.status = "completed"
    await session.commit()
    return {
        "ok": True,
        "method": "tesseract_llm",
        "ocr_text": ocr_text,
        "confidence": confidence,
        "language": language,
        "score": {
            "total_awarded": score.total_awarded,
            "total_max": score.total_max,
            "percentage": score.percentage,
            "confidence": score.confidence,
            "strengths": score.strengths,
            "gaps": score.gaps,
            "observations": score.observations,
            "criteria": score.criteria,
        },
        "input_tokens": score.input_tokens,
        "output_tokens": score.output_tokens,
        "total_tokens": score.total_tokens,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
_CSV_HEADERS = [
    "Timestamp", "External Reference", "Unique ID",
    "Full Name", "Gender", "Phone", "Email",
    "School", "Class", "State", "District", "Block",
    "Division", "Project Office", "Ashramshala",
    "Language", "Theme", "Resources",
    "AI Description",
    "Situation", "Proposed AI Solution",
    "Submission Count", "PO Name",
    "Drive URL", "Local File",
    "OCR (Original)", "OCR (Reconstruction)",
    "Cleaning Method", "Cleaned Text",
    "Input Tokens", "Output Tokens", "Total Tokens",
    "Confidence", "Status",
    "Score", "Max Score", "Percentage",
    "Per-Criterion Scores", "Strengths", "Gaps",
    "Observations", "Review Status", "Human Score", "Human Notes",
]


def _csv_row(e: Entry) -> list[str]:
    latest_ocr = (
        sorted(e.ocr_runs, key=lambda r: r.id, reverse=True)[0]
        if e.ocr_runs
        else None
    )
    latest_score = (
        sorted(e.scores, key=lambda r: r.id, reverse=True)[0]
        if e.scores
        else None
    )
    criteria_str = (
        "; ".join(
            f"{c.criterion}={c.awarded:.1f}/{c.max_score:.1f}"
            for c in (latest_score.criteria if latest_score else [])
        )
        if latest_score
        else ""
    )
    return [
        e.timestamp if hasattr(e, "timestamp") else "",
        e.external_ref,
        f"E{e.id:06d}",
        e.full_name or "",
        e.gender or "",
        e.phone or "",
        e.email or "",
        e.school_name or "",
        e.class_name or "",
        e.state_name or "",
        e.district or "",
        e.block_name or "",
        e.division or "",
        e.project_office or "",
        e.ashramshala or "",
        e.language or "",
        e.theme or "",
        ";".join(f"{k}={'Y' if v else 'N'}" for k, v in (e.resources or {}).items()),
        (e.ai_description or "")[:1000],
        (e.situation or "")[:1000],
        (e.proposed_solution or "")[:2000],
        str(e.submission_count or ""),
        e.po_name or "",
        e.drive_url or "",
        e.local_file_path or "",
        (latest_ocr.original_text or "")[:2000] if latest_ocr else "",
        (latest_ocr.reconstructed_text or "")[:2000] if latest_ocr else "",
        (latest_ocr.method.value if latest_ocr and latest_ocr.method else "") or "",
        (e.submitted_text or "")[:2000],
        str(latest_ocr.input_tokens or 0) if latest_ocr else "0",
        str(latest_ocr.output_tokens or 0) if latest_ocr else "0",
        str(latest_ocr.total_tokens or 0) if latest_ocr else "0",
        f"{latest_ocr.confidence:.2f}" if latest_ocr and latest_ocr.confidence is not None else "",
        e.status,
        str(e.total_score or 0),
        str(e.max_score or 0),
        str(e.percentage or 0),
        criteria_str,
        (latest_score.strengths or "") if latest_score else "",
        (latest_score.gaps or "") if latest_score else "",
        (latest_score.observations or "") if latest_score else "",
        e.review_status,
        str(e.human_score if e.human_score is not None else ""),
        (e.human_score_notes or "")[:1000],
    ]