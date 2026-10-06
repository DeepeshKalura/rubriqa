"""Import service tests: scan, match, idempotency, archive safety, unmatched."""
from __future__ import annotations

import zipfile

import pytest
from sqlalchemy import select

from app.models import Document, Exam, Klass, Organization, Student, Submission
from app.pipeline.ingestion import IngestionError
from app.services.importer import (
    DiscoveryReport,
    extract_archive,
    register_documents,
    scan_folder,
)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "incoming"
    (root / "STU001").mkdir(parents=True)
    (root / "STU002").mkdir(parents=True)
    (root / "STU001" / "a.txt").write_text("Q1. 56", encoding="utf-8")
    (root / "STU002" / "b.txt").write_text("Q1. 54", encoding="utf-8")
    (root / "random.txt").write_text("no roll number here", encoding="utf-8")
    return root


async def seed(session):
    org = Organization(name="Import School")
    session.add(org)
    await session.flush()
    klass = Klass(organization_id=org.id, name="C1")
    session.add(klass)
    await session.flush()
    for external_id in ("STU001", "STU002"):
        session.add(Student(organization_id=org.id, external_id=external_id, name=external_id))
    exam = Exam(class_id=klass.id, name="Import Exam", exam_date="2026-01-01")
    session.add(exam)
    await session.commit()
    return exam


def test_scan_matches_students_and_reports_unmatched(folder):
    plan = scan_folder(folder)
    assert plan.discovered == 3
    assert plan.unmatched == 1
    assert set(plan.per_student.keys()) == {"STU001", "STU002"}


def test_scan_ignores_unsupported_files(tmp_path):
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "STU001").mkdir()
    (root / "STU001" / "a.txt").write_text("x", encoding="utf-8")
    (root / "STU001" / "video.mp4").write_bytes(b"not supported")
    plan = scan_folder(root)
    assert plan.discovered == 1


def test_discovery_report_counts(tmp_path):
    root = tmp_path / "r"
    (root / "STU001").mkdir(parents=True)
    (root / "STU001" / "a.txt").write_text("same", encoding="utf-8")
    plan = scan_folder(root)
    report = DiscoveryReport.from_plan(plan, set())
    assert report.new_documents == 1
    assert report.already_imported == 0


def test_zip_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "boom")
    with pytest.raises(IngestionError) as exc:
        extract_archive(archive, tmp_path / "out")
    assert exc.value.code == "unsafe_archive"


def test_zip_extraction_works(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("STU001/a.txt", "Q1. 56")
    target = extract_archive(archive, tmp_path / "out")
    assert (target / "STU001" / "a.txt").exists()


async def test_register_documents_idempotent(session, clean_db, folder):
    exam = await seed(session)
    plan = scan_folder(folder)
    first = await register_documents(session, exam_id=exam.id, plan=plan)
    assert first.new_documents == 2
    assert first.submissions_created == 2

    hashes = (
        await session.execute(
            select(Document.sha256)
            .join(Submission, Submission.id == Document.submission_id)
            .where(Submission.exam_id == exam.id)
        )
    ).scalars().all()
    second = await register_documents(
        session, exam_id=exam.id, plan=scan_folder(folder), existing_hashes=set(hashes)
    )
    assert second.new_documents == 0
    assert second.already_imported == 2

    documents = (await session.execute(select(Document))).scalars().all()
    assert len(documents) == 2
    submissions = (await session.execute(select(Submission))).scalars().all()
    assert len(submissions) == 2


async def test_register_skips_unknown_students(session, clean_db, folder):
    exam = await seed(session)
    (folder / "STU999").mkdir()
    (folder / "STU999" / "x.txt").write_text("Q1. 1", encoding="utf-8")
    plan = scan_folder(folder)
    batch = await register_documents(session, exam_id=exam.id, plan=plan)
    assert batch.unmatched == 2  # random.txt + STU999 file
    assert batch.new_documents == 2


# ---------------------------------------------------------------------------
# Security regressions
# ---------------------------------------------------------------------------

async def _ensure_exam_one(session):
    from app.models import Exam, Klass, Organization

    org = (await session.execute(select(Organization).where(Organization.name == "_sec"))).scalar_one_or_none()
    if org is None:
        org = Organization(name="_sec"); session.add(org); await session.flush()
    klass = (await session.execute(select(Klass).where(Klass.organization_id == org.id, Klass.name == "_k"))).scalar_one_or_none()
    if klass is None:
        klass = Klass(organization_id=org.id, name="_k"); session.add(klass); await session.flush()
    exam = (await session.execute(select(Exam).where(Exam.class_id == klass.id, Exam.name == "_e"))).scalar_one_or_none()
    if exam is None:
        exam = Exam(class_id=klass.id, name="_e"); session.add(exam); await session.flush()
    return exam.id


async def test_scan_path_rejects_traversal_outside_storage_root(client, session, test_settings):
    """A request with path=../../etc must NOT resolve; scan-path must return 400."""
    exam_id = str(await _ensure_exam_one(session))
    await session.commit()
    response = await client.post(
        "/imports/scan-path",
        data={"exam_id": exam_id, "path": "../../../etc"},
    )
    assert response.status_code == 200  # renders the wizard with an error
    body = response.text
    assert "Path not found" in body or "not found" in body.lower()


async def test_scan_path_rejects_absolute_path_outside_root(client, session, tmp_path, test_settings):
    """An absolute path outside the storage root must also be rejected."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "a.txt").write_text("x", encoding="utf-8")
    exam_id = str(await _ensure_exam_one(session))
    await session.commit()
    response = await client.post(
        "/imports/scan-path",
        data={"exam_id": exam_id, "path": str(outside)},
    )
    assert response.status_code == 200
    assert "not found" in response.text.lower()
