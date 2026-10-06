"""Database tests: uniqueness constraints, idempotent imports, rollback,
job transitions, active-job uniqueness."""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import (
    Answer,
    Batch,
    Document,
    Exam,
    Job,
    JobState,
    JobType,
    Klass,
    Organization,
    Question,
    Student,
    Submission,
    SubmissionState,
)
from app.services import jobs as jobs_service


async def make_hierarchy(session) -> dict:
    org = Organization(name="T School")
    session.add(org)
    await session.flush()
    klass = Klass(organization_id=org.id, name="Class 1")
    session.add(klass)
    await session.flush()
    student = Student(organization_id=org.id, external_id="STU001", name="Test Student")
    session.add(student)
    await session.flush()
    exam = Exam(class_id=klass.id, name="Exam 1", exam_date="2026-01-01")
    session.add(exam)
    await session.flush()
    question = Question(exam_id=exam.id, q_no="Q1", text="2+2?", max_score=1.0)
    session.add(question)
    await session.flush()
    return {"org": org, "class": klass, "student": student, "exam": exam, "question": question}


async def test_submission_unique_exam_student(session, clean_db):
    data = await make_hierarchy(session)
    session.add(Submission(exam_id=data["exam"].id, student_id=data["student"].id))
    await session.commit()
    session.add(Submission(exam_id=data["exam"].id, student_id=data["student"].id))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_document_unique_hash_per_submission(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    for _ in range(2):
        session.add(
            Document(
                submission_id=submission.id,
                sha256="a" * 64,
                kind="image",
                original_filename="x.jpg",
                storage_path="originals/aa/x.jpg",
            )
        )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_answer_unique_submission_question(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    for _ in range(2):
        session.add(
            Answer(
                submission_id=submission.id,
                question_id=data["question"].id,
                text="4",
            )
        )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_duplicate_question_number_rejected(session, clean_db):
    data = await make_hierarchy(session)
    session.add(Question(exam_id=data["exam"].id, q_no="Q1", text="dup", max_score=1.0))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_duplicate_student_external_id_rejected(session, clean_db):
    data = await make_hierarchy(session)
    session.add(Student(organization_id=data["org"].id, external_id="STU001", name="Other"))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_transaction_rollback_leaves_no_partial_data(session, clean_db):
    data = await make_hierarchy(session)
    await session.commit()  # persist the good state first
    session.add(
        Student(organization_id=data["org"].id, external_id="STU999", name="Doomed")
    )
    await session.flush()
    await session.rollback()
    count = len((await session.execute(select(Student))).scalars().all())
    assert count == 1  # only STU001


async def test_job_enqueue_is_idempotent_for_submission(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    first = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    second = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    assert first.id == second.id
    jobs = (await session.execute(select(Job))).scalars().all()
    assert len(jobs) == 1


async def test_job_state_transitions_and_events(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    assert job.state is JobState.QUEUED

    claimed = await jobs_service.claim_next_job(session)
    assert claimed is not None
    assert claimed.state is JobState.RUNNING
    assert claimed.attempts == 1

    await jobs_service.transition(session, claimed, JobState.READING, "reading")
    await jobs_service.transition(session, claimed, JobState.VALIDATING, "validating")
    await jobs_service.complete(session, claimed)

    refreshed = await session.get(Job, job.id)
    assert refreshed.state is JobState.COMPLETED
    from app.models import JobEvent

    events = [
        row[0]
        for row in (
            await session.execute(select(JobEvent.stage).where(JobEvent.job_id == job.id))
        ).all()
    ]
    assert "queued" in events and "reading" in events and "completed" in events


async def test_job_failure_marks_submission_failed(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    await jobs_service.fail(session, claimed, "boom", code="test")
    refreshed = await session.get(Submission, submission.id)
    assert refreshed.state is SubmissionState.FAILED


async def test_retry_preserves_attempts_and_checkpoint(session, clean_db):
    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    claimed.checkpoint = {"read": True}
    await session.commit()
    await jobs_service.fail(session, claimed, "boom")
    await jobs_service.retry_job(session, claimed)
    assert claimed.state is JobState.QUEUED
    assert claimed.attempts == 1  # preserved
    assert claimed.checkpoint == {"read": True}


async def test_sweeper_requeues_stale_job(session, clean_db):
    from datetime import datetime, timedelta, timezone

    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    claimed.heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=1)
    await session.commit()

    requeued = await jobs_service.sweep_stale_jobs(session)
    assert requeued == 1
    refreshed = await session.get(Job, job.id)
    assert refreshed.state is JobState.QUEUED
    assert refreshed.locked_by is None


async def test_sweeper_fails_after_max_attempts(session, clean_db):
    from datetime import datetime, timedelta, timezone

    data = await make_hierarchy(session)
    submission = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    session.add(submission)
    await session.flush()
    job = await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    claimed.attempts = claimed.max_attempts
    claimed.heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=1)
    await session.commit()

    requeued = await jobs_service.sweep_stale_jobs(session)
    assert requeued == 1
    refreshed = await session.get(Job, job.id)
    assert refreshed.state is JobState.FAILED
    assert refreshed.error_code == "stale_worker"
    assert "retry" in refreshed.error.lower()


async def test_job_priority_ordering(session, clean_db):
    data = await make_hierarchy(session)
    low = Student(organization_id=data["org"].id, external_id="STU002", name="Low")
    session.add(low)
    await session.flush()
    submission_a = Submission(exam_id=data["exam"].id, student_id=data["student"].id)
    submission_b = Submission(exam_id=data["exam"].id, student_id=low.id)
    session.add_all([submission_a, submission_b])
    await session.flush()
    await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission_a.id, priority=100
    )
    await jobs_service.enqueue(
        session, JobType.PROCESS_SUBMISSION, submission_id=submission_b.id, priority=10
    )
    await session.commit()
    claimed = await jobs_service.claim_next_job(session)
    assert claimed.submission_id == submission_b.id
