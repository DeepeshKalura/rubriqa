"""Import wizard: scan -> dry-run diff -> commit -> queue.

Idempotency is visible in the dry-run: discovered / already imported / new.
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Exam, ImportBatch, Job, JobType, Submission
from app.pipeline.ingestion import IngestionError
from app.routers.common import render
from app.services import jobs as jobs_service
from app.services.importer import (
    DiscoveryReport,
    extract_archive,
    scan_folder,
)
from app.services.storage import Storage

router = APIRouter(tags=["imports"])
settings = get_settings()


@router.get("/imports")
async def imports_list(request: Request, session: AsyncSession = Depends(get_session)):
    batches = (
        (await session.execute(select(ImportBatch).order_by(ImportBatch.id.desc()).limit(50)))
        .scalars()
        .all()
    )
    exams = {
        exam.id: exam
        for exam in (await session.execute(select(Exam))).scalars().all()
    }
    return render(
        request, "imports.html", {"active": "imports", "batches": batches, "exams": exams}
    )


@router.get("/imports/new")
async def import_new(request: Request, session: AsyncSession = Depends(get_session)):
    exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
    return render(request, "import_new.html", {"active": "imports", "exams": exams})


@router.post("/imports/upload")
async def import_upload(
    request: Request,
    session: AsyncSession = Depends(get_session),
    exam_id: int = Form(...),
    files: list[UploadFile] = File(default=[]),
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    if not files:
        exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
        return render(
            request,
            "import_new.html",
            {"active": "imports", "exams": exams, "flash": "No files selected", "flash_kind": "error"},
        )

    storage = Storage()
    storage.ensure_dirs()
    batch_dir = storage.uploads / f"batch_{exam_id}_{len(files)}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    saved = 0
    for upload in files:
        filename = Path(upload.filename or "upload.bin").name
        target = batch_dir / filename
        # Fail fast on Content-Length before buffering the body. Headers can
        # be spoofed so we also re-check while streaming.
        declared = upload.headers.get("content-length") if upload.headers else None
        if declared and declared.isdigit() and int(declared) > max_bytes:
            return _upload_error(request, exams, f"{filename} exceeds MAX_UPLOAD_MB")
        sink = io.BytesIO()
        written = 0
        while True:
            chunk = await upload.read(1 << 20)  # 1 MB
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                return _upload_error(request, exams, f"{filename} exceeds MAX_UPLOAD_MB")
            sink.write(chunk)
        storage.atomic_write_bytes(target, sink.getvalue())
        saved += 1

    # zip archives are expanded (zip-slip protected)
    for archive in list(batch_dir.glob("*.zip")):
        try:
            extract_archive(archive, batch_dir / archive.stem)
        except IngestionError as exc:
            return render(
                request,
                "import_new.html",
                {"active": "imports", "exams": [], "flash": str(exc), "flash_kind": "error"},
            )
        archive.unlink()

    plan = scan_folder(batch_dir)
    existing = await _existing_hashes(session, exam_id)
    report = DiscoveryReport.from_plan(plan, existing)
    exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
    return render(
        request,
        "import_confirm.html",
        {
            "active": "imports",
            "exams": exams,
            "exam": exam,
            "report": report,
            "root": str(batch_dir),
            "saved": saved,
        },
    )


@router.post("/imports/commit")
async def import_commit(
    request: Request,
    session: AsyncSession = Depends(get_session),
    exam_id: int = Form(...),
    root: str = Form(...),
    queue: str = Form("1"),
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    root_path = Storage().safe_resolve(root)
    if root_path is None or not root_path.exists():
        return RedirectResponse("/imports/new", status_code=303)

    plan = scan_folder(root_path)
    existing = await _existing_hashes(session, exam_id)
    report = DiscoveryReport.from_plan(plan, existing)

    from app.services.importer import register_documents

    import_batch = await register_documents(
        session,
        exam_id=exam_id,
        plan=plan,
        storage=Storage(),
        existing_hashes=existing,
    )
    if queue == "1":
        from app.services.importer import enqueue_submissions

        await enqueue_submissions(session, exam_id)
    return RedirectResponse(f"/imports?committed={import_batch.id}", status_code=303)


@router.post("/imports/scan-path")
async def scan_path(
    request: Request,
    session: AsyncSession = Depends(get_session),
    exam_id: int = Form(...),
    path: str = Form(...),
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    # Containment check: refuse absolute paths outside the storage root and
    # any traversal that escapes it. Without this, a request with
    # path=../../etc would scan arbitrary host directories.
    root = Storage().safe_resolve(path)
    if root is None or not root.exists():
        exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
        return render(
            request,
            "import_new.html",
            {"active": "imports", "exams": exams, "flash": f"Path not found: {path}", "flash_kind": "error"},
        )
    plan = scan_folder(root)
    existing = await _existing_hashes(session, exam_id)
    report = DiscoveryReport.from_plan(plan, existing)
    exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
    return render(
        request,
        "import_confirm.html",
        {"active": "imports", "exams": exams, "exam": exam, "report": report, "root": str(root), "saved": 0},
    )


async def _existing_hashes(session: AsyncSession, exam_id: int) -> set[str]:
    from app.models import Document

    rows = (
        await session.execute(
            select(Document.sha256)
            .join(Submission, Submission.id == Document.submission_id)
            .where(Submission.exam_id == exam_id)
        )
    ).scalars().all()
    return set(rows)
