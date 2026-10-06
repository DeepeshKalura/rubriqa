"""Import service: folder upload -> submissions/documents with idempotency.

The filename convention maps files to students:
    <anything>/STU001/....jpg   or   STU001_Q1.jpg
The student external_id is extracted with a configurable regex; unmatched
files are reported, never guessed.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging import get_logger
from app.models import (
    ACTIVE_STATES,
    Document,
    ImportBatch,
    Page,
    Student,
    Submission,
    SubmissionState,
)
from app.pipeline.ingestion import IngestionError, detect_source_type, ingest
from app.services.storage import Storage

logger = get_logger("exam_eval.import")

DEFAULT_STUDENT_PATTERN = re.compile(r"(?:^|[\\/])(STU\d+|[A-Z]{2,}\d{2,})(?:[\\/_\-.]|$)", re.I)


@dataclass
class DiscoveredFile:
    path: Path
    student_external_id: str | None
    sha256: str | None = None
    already_imported: bool = False


@dataclass
class ImportPlan:
    exam_id: int
    root: Path
    discovered: int = 0
    already_imported: int = 0
    new_documents: int = 0
    unmatched: int = 0
    per_student: dict[str, list[DiscoveredFile]] = field(default_factory=dict)
    unmatched_files: list[str] = field(default_factory=list)


def extract_archive(archive: Path, target_dir: Path) -> Path:
    """Safely extract a zip into target_dir (zip-slip protected)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            member_path = Path(member)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise IngestionError(f"Unsafe path in archive: {member}", code="unsafe_archive")
            destination = (target_dir / member_path).resolve()
            if not str(destination).startswith(str(target_dir.resolve())):
                raise IngestionError(f"Unsafe path in archive: {member}", code="unsafe_archive")
        zf.extractall(target_dir)
    return target_dir


@dataclass
class DiscoveryReport:
    """Dry-run result shown to the user before committing an import."""

    discovered: int = 0
    already_imported: int = 0
    new_documents: int = 0
    unmatched: int = 0
    per_student: dict[str, int] = field(default_factory=dict)
    unmatched_files: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: ImportPlan, existing_hashes: set[str]) -> "DiscoveryReport":
        store = Storage()
        report = cls(
            discovered=plan.discovered,
            unmatched=plan.unmatched,
            unmatched_files=list(plan.unmatched_files),
        )
        for external_id, files in plan.per_student.items():
            new = 0
            for file in files:
                digest = Storage.sha256_file(file.path)
                if digest in existing_hashes:
                    report.already_imported += 1
                else:
                    new += 1
                    report.new_documents += 1
                    existing_hashes = existing_hashes | {digest}
            report.per_student[external_id] = new
        return report

    def summary(self) -> str:
        return (
            f"{self.discovered:,} files discovered · {self.already_imported:,} already "
            f"imported · {self.new_documents:,} new · {self.unmatched:,} unmatched"
        )


def scan_folder(root: Path, pattern: re.Pattern[str] = DEFAULT_STUDENT_PATTERN) -> ImportPlan:
    store = Storage()
    plan = ImportPlan(exam_id=0, root=root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        try:
            detect_source_type(path)
        except IngestionError:
            continue
        match = pattern.search(str(path))
        student_id = match.group(1) if match else None
        file = DiscoveredFile(path=path, student_external_id=student_id)
        plan.discovered += 1
        if student_id is None:
            plan.unmatched += 1
            plan.unmatched_files.append(str(path.relative_to(root)))
            continue
        plan.per_student.setdefault(student_id.upper(), []).append(file)
    return plan


async def register_documents(
    session: AsyncSession,
    *,
    exam_id: int,
    plan: ImportPlan,
    storage: Storage | None = None,
    existing_hashes: set[str] | None = None,
) -> ImportBatch:
    """Create submissions/documents for a scanned plan. Idempotent by sha256."""
    store = storage or Storage()
    store.ensure_dirs()
    batch = ImportBatch(
        exam_id=exam_id,
        source="path",
        root_path=str(plan.root),
        status="committed",
        discovered=plan.discovered,
        already_imported=0,
        new_documents=0,
        submissions_created=0,
        unmatched=plan.unmatched,
    )
    session.add(batch)
    await session.flush()

    for external_id, files in plan.per_student.items():
        student = await _find_student(session, external_id)
        if student is None:
            plan.unmatched += len(files)
            plan.unmatched_files.extend(str(f.path) for f in files)
            batch.unmatched += len(files)
            continue
        submission = (
            await session.execute(
                select(Submission).where(
                    Submission.exam_id == exam_id, Submission.student_id == student.id
                )
            )
        ).scalar_one_or_none()
        if submission is None:
            submission = Submission(
                exam_id=exam_id, student_id=student.id, state=SubmissionState.PENDING
            )
            session.add(submission)
            await session.flush()
            batch.submissions_created += 1

        for file in files:
            sha = Storage.sha256_file(file.path)
            duplicate = (
                await session.execute(
                    select(Document).where(
                        Document.submission_id == submission.id, Document.sha256 == sha
                    )
                )
            ).scalar_one_or_none()
            if duplicate is not None or (existing_hashes is not None and sha in existing_hashes):
                batch.already_imported += 1
                continue
            stored = store.store_original(file.path, sha, file.path.suffix)
            canonical = ingest(file.path)
            document = Document(
                submission_id=submission.id,
                sha256=sha,
                kind=canonical.source_type,
                original_filename=file.path.name,
                storage_path=store.relative(stored),
                mime=canonical.mime,
                size_bytes=canonical.size_bytes,
                page_count=len(canonical.pages),
                source="import",
                metadata_json={"source_path": str(file.path)},
            )
            session.add(document)
            await session.flush()
            for page in canonical.pages:
                session.add(
                    Page(
                        document_id=document.id,
                        index=page.index,
                        width=page.width,
                        height=page.height,
                        has_text_layer=page.has_text_layer,
                        direct_text=page.text if page.has_text_layer else None,
                    )
                )
            batch.new_documents += 1
    await session.commit()
    return batch


async def _find_student(session: AsyncSession, external_id: str) -> Student | None:
    return (
        await session.execute(
            select(Student).where(Student.external_id == external_id.upper())
        )
    ).scalar_one_or_none()


async def enqueue_submissions(
    session: AsyncSession, exam_id: int, *, only_pending: bool = True
) -> int:
    """Queue PROCESS_SUBMISSION for every pending submission of the exam."""
    from app.models import Job, JobType
    from app.services import jobs as jobs_service

    query = select(Submission).where(Submission.exam_id == exam_id)
    if only_pending:
        query = query.where(
            Submission.state.in_([SubmissionState.PENDING, SubmissionState.NEEDS_REVIEW])
        )
    submissions = (await session.execute(query)).scalars().all()
    queued = 0
    for submission in submissions:
        active = (
            await session.execute(
                select(Job).where(
                    Job.submission_id == submission.id, Job.state.in_(list(ACTIVE_STATES))
                )
            )
        ).scalar_one_or_none()
        if active is not None:
            continue
        await jobs_service.enqueue(
            session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
        )
        queued += 1
    await session.commit()
    return queued
