"""HTML sandbox backend — 3-way OCR comparison for testing point of view.

The page (`app/templates/sandbox.html` served at `/sandbox`) lets a developer
or reviewer:

    * drop in a single page image (or pick from the bundled `data/samples/`
      Kagaaz handwritten pages)
    * run Path A: strict (zero-hallucination) vision-LLM OCR + scoring
    * run Path B: Tesseract does OCR, then the LLM scores the cleaned text
    * run Path C: Tesseract + vision-LLM transcripts merged by a vision LLM
      (the page image arbitrates) into one human-like transcript, then scored
    * compare all paths side-by-side against the human-verified golden
      transcript stored per bundled page (CER shown when available)

The compare endpoint takes a multipart image and runs the selected paths
against the running app's configured MiniMax provider. Results are returned
as JSON; the HTML page renders the comparison.
"""
from __future__ import annotations

import asyncio
import shutil as _shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.logging import get_logger
from app.models import (
    DriveFile,
    Entry,
    EntryOCR,
    EntryState,
    OcrMethod,
    OcrStatus,
    Program,
    ProgramState,
    Rubric,
    RubricCriterion,
)
from app.pipeline.ingestion.google_drive import sha256_of
from app.pipeline.scoring.entry import score_entry
from app.providers.llm import build_provider
from app.routers.common import render
from app.services.entries import (
    process_entry_via_fusion,
    process_entry_via_tesseract_then_llm,
    process_entry_via_vision,
)

router = APIRouter(tags=["sandbox"])
logger = get_logger("entry_eval.sandbox")


@router.get("/sandbox")
async def sandbox_page(request: Request):
    """The single-file HTML sandbox lives at docs/sandbox.html. The FastAPI
    app serves it from /sandbox (overwrites path) so the user gets a clean URL.
    """
    return render(request, "sandbox.html", {"active": "sandbox"})


async def _ensure_sandbox_program(session: AsyncSession) -> Program:
    """Make sure there is at least one Program we can attach sandbox entries
    to. Created on demand, named "Sandbox".
    """
    program = (
        await session.execute(select(Program).where(Program.name == "Sandbox"))
    ).scalar_one_or_none()
    if program is not None:
        return program
    rubric = (
        await session.execute(select(Rubric).where(Rubric.name == "Sandbox default rubric"))
    ).scalar_one_or_none()
    if rubric is None:
        rubric = Rubric(name="Sandbox default rubric", description="auto-created")
        session.add(rubric)
        await session.flush()
        for nm, desc in [
            ("relevance", "How relevant the proposal is to the chosen theme."),
            ("feasibility", "How realistic it is to build in the student's context."),
            ("originality", "How original / non-trivially-copied the idea is."),
            ("ai_understanding", "How clearly the student explains how AI is used."),
            ("communication", "Clarity, structure, and respect for the word-count guidance."),
        ]:
            session.add(
                RubricCriterion(
                    rubric_id=rubric.id, name=nm, weight=1.0, description=desc,
                    knockout=False, knockout_below_fraction=0.0,
                )
            )
        await session.flush()
    program = Program(
        name="Sandbox",
        description="Auto-created for 3-way sandbox OCR comparison (vision / tesseract+LLM / vision+OCR fusion).",
        state=ProgramState.ACTIVE,
        rubric_id=rubric.id,
        batch_size=250,
    )
    session.add(program)
    await session.commit()
    await session.refresh(program)
    return program


async def _materialise_entry(
    session: AsyncSession,
    *,
    program: Program,
    image_bytes: bytes,
    file_name: str,
    language_hint: str = "auto",
    theme: str = "ai_education",
    situation: str = "",
    proposed_solution: str = "",
    ai_description: str = "",
) -> tuple[Entry, Image]:
    from PIL import Image as _Image
    import secrets

    settings = get_settings()
    sha = _sha256_of_bytes(image_bytes)

    cache_dir = settings.storage_root / "sandbox_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sha}_{file_name}"
    cache_path.write_bytes(image_bytes)

    ext = Path(file_name).suffix.lower() or ".jpg"
    suffix = secrets.token_hex(4)
    drive_file_id = f"sandbox-{sha[:12]}-{suffix}"
    external_ref = drive_file_id

    entry = Entry(
        program_id=program.id,
        external_ref=external_ref,
        state=EntryState.PENDING,
        form_kind="sandbox",
        full_name=f"Sandbox {sha[:6]}",
        language=language_hint or None,
        theme=theme,
        ai_description=ai_description or None,
        situation=situation or None,
        proposed_solution=proposed_solution or None,
        declaration=True,
        consent=True,
        drive_url="local:sandbox",
        drive_file_id=drive_file_id,
        local_file_path=str(cache_path),
        drive_sha256=sha,
        status="pending",
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry, _Image.open(cache_path)


def _sha256_of_bytes(b: bytes) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(b).hexdigest()


def _char_error_rate(reference: str, hypothesis: str) -> float | None:
    """Character error rate between the golden transcript and a path's OCR
    output (0.0 = perfect, 1.0 = total mismatch). Normalizes case and
    whitespace; returns None when there is no reference."""
    import re as _re

    def norm(t: str) -> str:
        return _re.sub(r"\s+", " ", t.lower()).strip()

    ref, hyp = norm(reference), norm(hypothesis)
    if not ref:
        return None
    # classic DP Levenshtein over characters
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i]
        for j, hc in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if rc == hc else 1)))
        prev = cur
    return prev[-1] / len(ref)


def _golden_transcripts(settings) -> dict[str, str]:
    """Map sha256(page image) -> human-verified golden transcript. Golden
    files live in `data/samples/golden/<page-stem>.txt` next to the bundled
    `data/samples/pages/` images."""
    samples_root = settings.storage_root / "samples"
    golden_dir = samples_root / "golden"
    out: dict[str, str] = {}
    if not golden_dir.exists():
        return out
    for page in sorted(samples_root.glob("pages/page-*.png")):
        gt = golden_dir / f"{page.stem}.txt"
        if gt.exists():
            out[sha256_of(page)] = gt.read_text(encoding="utf-8").strip()
    return out


async def _run_path(
    label: str,
    coro_factory,
) -> dict:
    """Run a path sequentially. Each path opens its own SQLAlchemy session to
    avoid sharing a session across coroutines (which closes the underlying
    transaction)."""
    started = time.perf_counter()
    try:
        result = await coro_factory()
        result = dict(result or {})
        result["path"] = label
        result["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "path": label,
            "error": str(exc)[:500],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }


@router.post("/sandbox/compare")
async def sandbox_compare(
    request: Request,
    session: AsyncSession = Depends(get_session),
    image: UploadFile = File(...),
    language: str = Form("auto"),
    theme: str = Form("ai_education"),
    situation: str = Form(""),
    proposed_solution: str = Form(""),
    ai_description: str = Form(""),
    rubric_id: str = Form(""),
    run_vision: str = Form("true"),
    run_tesseract: str = Form("true"),
    run_fusion: str = Form("true"),
):
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(
            400,
            "MINIMAX_API_KEY (or LLM_API_KEY) is missing. Set it in .env.",
        )
    program = await _ensure_sandbox_program(session)
    image_bytes = await image.read()
    file_name = image.filename or "sandbox.jpg"
    entry, _img = await _materialise_entry(
        session,
        program=program,
        image_bytes=image_bytes,
        file_name=file_name,
        language_hint=language or "auto",
        theme=theme or "ai_education",
        situation=situation,
        proposed_solution=proposed_solution,
        ai_description=ai_description,
    )
    await session.commit()

    rubric = rubric_id and int(rubric_id) or None
    if rubric is None:
        rubric = program.rubric_id

    # Each path gets its own session; we use the get_session factory directly
    # so a failed path doesn't poison the others.
    from app.db import get_sessionmaker

    sm = get_sessionmaker()

    # The two paths share one Entry row, and each path writes its OCR output
    # into entry.situation before scoring. Without a reset between paths the
    # second path would score the FIRST path's text (dishonest comparison).
    pristine = {
        "situation": entry.situation,
        "proposed_solution": entry.proposed_solution,
        "submitted_text": entry.submitted_text,
        "ai_description": entry.ai_description,
    }

    async def _restore_entry_fields() -> None:
        async with sm() as restore_session:
            e = await restore_session.get(Entry, entry.id)
            for col, val in pristine.items():
                setattr(e, col, val)
            await restore_session.commit()

    results: list[dict] = []

    if run_vision.lower() == "true":
        await _restore_entry_fields()

        async def _vision():
            async with sm() as path_session:
                local_entry = await path_session.get(Entry, entry.id)
                return await process_entry_via_vision(
                    path_session, local_entry,
                    settings=settings,
                    llm=build_provider(settings),
                    rubric_id=rubric,
                )

        results.append(await _run_path("vision", _vision))

    if run_tesseract.lower() == "true":
        await _restore_entry_fields()

        async def _tess():
            async with sm() as path_session:
                local_entry = await path_session.get(Entry, entry.id)
                return await process_entry_via_tesseract_then_llm(
                    path_session, local_entry,
                    settings=settings,
                    llm=build_provider(settings),
                    rubric_id=rubric,
                )

        results.append(await _run_path("tesseract_llm", _tess))

    if run_fusion.lower() == "true":
        await _restore_entry_fields()

        async def _fusion():
            async with sm() as path_session:
                local_entry = await path_session.get(Entry, entry.id)
                return await process_entry_via_fusion(
                    path_session, local_entry,
                    settings=settings,
                    llm=build_provider(settings),
                    rubric_id=rubric,
                )

        results.append(await _run_path("vision_ocr_fusion", _fusion))

    if not results:
        raise HTTPException(400, "Choose at least one path to run")

    # Golden transcript + per-path character error rate, when the uploaded
    # image is one of the bundled sample pages and a human-verified transcript
    # exists for it.
    golden_text = _golden_transcripts(settings).get(_sha256_of_bytes(image_bytes))
    if golden_text:
        for r in results:
            if r.get("ok") and r.get("ocr_text"):
                cer = _char_error_rate(golden_text, r["ocr_text"])
                if cer is not None:
                    r["cer"] = round(100 * cer, 1)

    # Leave the shared entry row in its pristine form too.
    await _restore_entry_fields()

    # reload entry so we have the latest state (paths snapshot/restore the
    # shared row, so it is back to its pristine form here)
    final = await session.get(Entry, entry.id)
    return {
        "entry_id": entry.id,
        "external_ref": entry.external_ref,
        "theme": entry.theme,
        "language": entry.language,
        "image_path": final.local_file_path if final else None,
        "results": results,
        "rubric_id": rubric,
        "golden": golden_text,
    }


@router.get("/sandbox/samples")
async def list_sandbox_samples():
    """List the bundled handwritten sample pages (from data/samples/pages/)
    so the HTML sandbox can let the user pick one as a quick fixture. Each
    sample carries its human-verified golden transcript (from
    data/samples/golden/<page>.txt) when one exists."""
    settings = get_settings()
    samples_root = settings.storage_root / "samples"
    pages_root = samples_root / "pages"
    golden_dir = samples_root / "golden"
    samples: list[dict] = []
    if pages_root.exists():
        for p in sorted(pages_root.glob("page-*.png")):
            gt = golden_dir / f"{p.stem}.txt"
            samples.append(
                {
                    "name": p.name,
                    "path": f"/files/{p}",
                    "golden_text": gt.read_text(encoding="utf-8").strip() if gt.exists() else None,
                }
            )
    return {"samples": samples}


@router.post("/sandbox/reset")
async def sandbox_reset(session: AsyncSession = Depends(get_session)):
    """Delete sandbox-created programs + entries so a comparison round can start
    clean. Real-program entries are not touched."""
    from sqlalchemy import delete
    program = (
        await session.execute(select(Program).where(Program.name == "Sandbox"))
    ).scalar_one_or_none()
    if program is None:
        return {"deleted_entries": 0}
    entries = (
        await session.execute(
            select(Entry).where(Entry.program_id == program.id)
        )
    ).scalars().all()
    ids = [e.id for e in entries]
    await session.commit()
    return {"deleted_entries": len(ids)}