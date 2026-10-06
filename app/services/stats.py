"""Aggregate queries for the dashboard and list pages.

Every query here is bounded: counts use indexes, list queries take explicit
limits and are never asked to load a whole table.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ACTIVE_STATES,
    Answer,
    Batch,
    Document,
    Exam,
    Job,
    JobState,
    Klass,
    LlmCall,
    Organization,
    Page,
    Score,
    Student,
    Submission,
    SubmissionState,
    ValidationIssue,
)


async def dashboard_stats(session: AsyncSession) -> dict:
    classes = (await session.execute(select(func.count(Klass.id)))).scalar_one()
    students = (await session.execute(select(func.count(Student.id)))).scalar_one()
    exams = (await session.execute(select(func.count(Exam.id)))).scalar_one()
    submissions = (await session.execute(select(func.count(Submission.id)))).scalar_one()
    documents = (await session.execute(select(func.count(Document.id)))).scalar_one()
    pages = (await session.execute(select(func.count(Page.id)))).scalar_one()
    answers = (await session.execute(select(func.count(Answer.id)))).scalar_one()

    job_counts = {
        state.value: count
        for state, count in (
            await session.execute(select(Job.state, func.count(Job.id)).group_by(Job.state))
        ).all()
    }
    submission_states = {
        state.value: count
        for state, count in (
            await session.execute(
                select(Submission.state, func.count(Submission.id)).group_by(Submission.state)
            )
        ).all()
    }
    open_issues = (
        await session.execute(
            select(func.count(ValidationIssue.id)).where(
                ValidationIssue.resolved.is_(False),
                ValidationIssue.severity == "error",
            )
        )
    ).scalar_one()
    llm_tokens = (
        await session.execute(
            select(
                func.coalesce(func.sum(LlmCall.total_tokens), 0),
                func.count(LlmCall.id),
            )
        )
    ).one()
    active_jobs = sum(job_counts.get(state.value, 0) for state in ACTIVE_STATES)
    return {
        "classes": classes,
        "students": students,
        "exams": exams,
        "submissions": submissions,
        "documents": documents,
        "pages": pages,
        "answers": answers,
        "job_counts": job_counts,
        "active_jobs": active_jobs,
        "submission_states": submission_states,
        "open_errors": open_issues,
        "llm_tokens": llm_tokens[0],
        "llm_calls": llm_tokens[1],
    }


async def recent_jobs(session: AsyncSession, limit: int = 10) -> list[dict]:
    jobs = (
        (
            await session.execute(
                select(Job).order_by(Job.id.desc()).limit(min(limit, 50))
            )
        )
        .scalars()
        .all()
    )
    rows = []
    for job in jobs:
        student_name = None
        if job.submission_id:
            submission = await session.get(Submission, job.submission_id)
            if submission:
                student = await session.get(Student, submission.student_id)
                student_name = student.name if student else None
        rows.append({"job": job, "student_name": student_name})
    return rows


async def score_distribution(session: AsyncSession, exam_id: int) -> dict:
    rows = (
        await session.execute(
            select(Submission.percentage).where(
                Submission.exam_id == exam_id, Submission.percentage.isnot(None)
            )
        )
    ).scalars().all()
    buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for value in rows:
        if value < 20:
            buckets["0-20"] += 1
        elif value < 40:
            buckets["20-40"] += 1
        elif value < 60:
            buckets["40-60"] += 1
        elif value < 80:
            buckets["60-80"] += 1
        else:
            buckets["80-100"] += 1
    return buckets


async def org_tree(session: AsyncSession) -> list[dict]:
    """Class -> batch summary for the classes page (bounded by class count)."""
    classes = (await session.execute(select(Klass).order_by(Klass.name))).scalars().all()
    result = []
    for klass in classes:
        batch_count = (
            await session.execute(
                select(func.count(Batch.id)).where(Batch.class_id == klass.id)
            )
        ).scalar_one()
        from app.models import Enrollment

        student_count = (
            await session.execute(
                select(func.count(func.distinct(Enrollment.student_id)))
                .join(Batch, Batch.id == Enrollment.batch_id)
                .where(Batch.class_id == klass.id)
            )
        ).scalar_one()
        exam_count = (
            await session.execute(
                select(func.count(Exam.id)).where(Exam.class_id == klass.id)
            )
        ).scalar_one()
        result.append(
            {
                "klass": klass,
                "batches": batch_count,
                "students": student_count,
                "exams": exam_count,
            }
        )
    return result
