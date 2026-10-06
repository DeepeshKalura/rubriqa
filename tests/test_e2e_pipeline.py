"""E2E pipeline test with the fake provider: ingestion through scoring in DB.

This is the deterministic counterpart to `make e2e` (which uses MiniMax).
No network access.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models import (
    Answer,
    Document,
    Exam,
    Job,
    JobState,
    JobType,
    Klass,
    Organization,
    Page,
    Question,
    ReadingRun,
    Rubric,
    RubricCriterion,
    Score,
    ScoringRun,
    Student,
    Submission,
    SubmissionState,
)
from app.pipeline.readers import build_reader
from app.providers.llm.fake import FakeProvider
from app.services import jobs as jobs_service
from app.services.importer import (
    DiscoveryReport,
    enqueue_submissions,
    register_documents,
    scan_folder,
)
from app.services.pipeline import SubmissionPipeline


@pytest.fixture
def dataset(tmp_path):
    """Two students, one text file + one image each, plus a duplicate file."""
    root = tmp_path / "dataset"
    for external_id in ("STU001", "STU002"):
        folder = root / "Class 1" / "Batch A" / external_id
        folder.mkdir(parents=True)
        (folder / "answers.txt").write_text(
            f"Name: {external_id}\nQ1. What is 7 x 8?\nAns: 56\n"
            "Q2. State Newton's first law.\nAns: An object stays at rest unless a force acts.\n",
            encoding="utf-8",
        )
    # duplicate content under a different name (idempotency check)
    duplicate = root / "Class 1" / "Batch A" / "STU001" / "answers_copy.txt"
    duplicate.write_text(
        (root / "Class 1" / "Batch A" / "STU001" / "answers.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return root


async def seed_exam(session) -> dict:
    org = Organization(name="E2E School")
    session.add(org)
    await session.flush()
    klass = Klass(organization_id=org.id, name="Class 1")
    session.add(klass)
    await session.flush()
    students = {}
    for external_id, name in (("STU001", "Aarav"), ("STU002", "Priya")):
        student = Student(organization_id=org.id, external_id=external_id, name=name)
        session.add(student)
        students[external_id] = student
    await session.flush()
    exam = Exam(class_id=klass.id, name="E2E Exam", exam_date="2026-01-01")
    session.add(exam)
    await session.flush()
    rubric = Rubric(name="E2E rubric", description="three criteria")
    session.add(rubric)
    await session.flush()
    session.add_all(
        [
            RubricCriterion(rubric_id=rubric.id, name="correctness", weight=0.5, knockout=True),
            RubricCriterion(rubric_id=rubric.id, name="completeness", weight=0.3),
            RubricCriterion(rubric_id=rubric.id, name="reasoning", weight=0.2),
        ]
    )
    questions = [
        Question(
            exam_id=exam.id, q_no="Q1", text="What is 7 x 8?", max_score=1.0,
            strategy="numeric", answer_key="56", rubric_id=rubric.id,
        ),
        Question(
            exam_id=exam.id, q_no="Q2", text="State Newton's first law.", max_score=2.0,
            strategy="rubric_llm", answer_key="Objects stay at rest or in motion unless a force acts.",
            rubric_id=rubric.id,
        ),
    ]
    session.add_all(questions)
    await session.commit()
    return {
        "org": org, "class": klass, "students": students, "exam": exam,
        "rubric": rubric, "questions": questions,
    }


async def run_pipeline_for(session, settings, llm, submission: Submission, reader):
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    pipeline = SubmissionPipeline(session, settings, reader=reader, repairer=None, llm=llm)
    result = await pipeline.run(claimed)
    return claimed, result


async def test_full_pipeline_fake_provider(session, clean_db, dataset, test_settings, fake_llm):
    data = await seed_exam(session)
    plan = scan_folder(dataset)
    report = DiscoveryReport.from_plan(plan, set())
    # 3 files discovered; the duplicate copy shares STU001's hash
    assert report.discovered == 3
    assert report.new_documents == 2
    batch = await register_documents(session, exam_id=data["exam"].id, plan=plan)
    assert batch.new_documents == 2
    assert batch.submissions_created == 2

    # idempotent re-import
    plan2 = scan_folder(dataset)
    hashes = (
        await session.execute(
            select(Document.sha256)
            .join(Submission, Submission.id == Document.submission_id)
            .where(Submission.exam_id == data["exam"].id)
        )
    ).scalars().all()
    batch2 = await register_documents(
        session, exam_id=data["exam"].id, plan=plan2, existing_hashes=set(hashes)
    )
    assert batch2.new_documents == 0
    assert batch2.already_imported == 3

    reader = build_reader("hybrid", llm=fake_llm, settings=test_settings)
    # fake vision reads a canned text that includes the numeric answer for numeric Q
    fake_llm.queue(
        "extract_answers",
        '{"answers": [{"question_no": "Q1", "answer_text": "56", "confidence": 0.9}, '
        '{"question_no": "Q2", "answer_text": "An object stays at rest unless a force acts.", "confidence": 0.9}]}',
    )
    submissions = (
        (await session.execute(select(Submission).order_by(Submission.id))).scalars().all()
    )
    for submission in submissions:
        # reset for each submission so both get answers
        fake_llm.queue(
            "extract_answers",
            '{"answers": [{"question_no": "Q1", "answer_text": "56", "confidence": 0.9}, '
            '{"question_no": "Q2", "answer_text": "An object stays at rest unless a force acts.", "confidence": 0.9}]}',
        )
        job, result = await run_pipeline_for(
            session, test_settings, fake_llm, submission, reader
        )
        assert result.blocking_issues == 0

    # verify
    submissions = (
        (await session.execute(select(Submission).order_by(Submission.id))).scalars().all()
    )
    assert all(sub.state is SubmissionState.COMPLETED for sub in submissions)
    assert all(sub.total_score is not None for sub in submissions)

    answer_count = (await session.execute(select(func.count(Answer.id)))).scalar_one()
    assert answer_count == 4

    runs = (await session.execute(select(ScoringRun))).scalars().all()
    assert len(runs) == 2
    assert all(run.is_current for run in runs)

    scores = (await session.execute(select(Score))).scalars().all()
    # per submission: 1 numeric criterion (Q1) + 1 rubric criterion (fake provider
    # returns only "correctness") = 2
    assert len(scores) == 4

    # 2 distinct documents: the copy has the same hash and was skipped
    pages = (await session.execute(select(Page))).scalars().all()
    assert len(pages) == 2

    readings = (await session.execute(select(ReadingRun))).scalars().all()
    assert len(readings) == 2
    assert all(run.is_selected for run in readings)


async def test_pipeline_marks_needs_review_on_empty_document(
    session, clean_db, test_settings, fake_llm, tmp_path
):
    data = await seed_exam(session)
    # a document whose page produces no text: text file with only whitespace is
    # rejected at ingestion, so simulate via an image page read as empty
    fake_llm.queue(
        "vision_read",
        '{"text": "", "legibility": 0.0, "language": "en", "notes": "blank"}',
    )
    fake_llm.queue("extract_answers", '{"answers": []}')
    root = tmp_path / "blank"
    folder = root / "STU001"
    folder.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (400, 600), "white").save(folder / "blank.png")

    plan = scan_folder(root)
    await register_documents(session, exam_id=data["exam"].id, plan=plan)
    submission = (await session.execute(select(Submission))).scalar_one()

    reader = build_reader("minimax_vision", llm=fake_llm, settings=test_settings)
    settings = test_settings.model_copy(
        update={"reader_mode": "minimax_vision", "repair_enabled": False}
    )
    job, result = await run_pipeline_for(session, settings, fake_llm, submission, reader)
    refreshed = await session.get(Submission, submission.id)
    # empty page -> warning, not blocking error -> proceeds but flagged
    assert result.answers == 0
    assert refreshed.state in (SubmissionState.NEEDS_REVIEW, SubmissionState.COMPLETED)
    assert refreshed.needs_review or result.warning_issues > 0
