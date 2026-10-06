"""Students list + profile (paginated, searchable)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    Answer,
    Batch,
    Enrollment,
    Exam,
    Klass,
    Question,
    Score,
    ScoringRun,
    Student,
    Submission,
)
from app.routers.common import pagination, render

router = APIRouter(tags=["students"])

PER_PAGE_CHOICES = (25, 50, 100)


@router.get("/students")
async def students(
    request: Request,
    session: AsyncSession = Depends(get_session),
    q: str = "",
    page: int = 1,
    per_page: int = 25,
):
    per_page = per_page if per_page in PER_PAGE_CHOICES else 25
    query = select(Student)
    count_query = select(func.count(Student.id))
    if q.strip():
        pattern = f"%{q.strip()}%"
        condition = or_(Student.name.ilike(pattern), Student.external_id.ilike(pattern))
        query = query.where(condition)
        count_query = count_query.where(condition)
    total = (await session.execute(count_query)).scalar_one()
    page_meta = pagination(page, per_page, total)
    rows = (
        (
            await session.execute(
                query.order_by(Student.external_id)
                .limit(per_page)
                .offset((page_meta["page"] - 1) * per_page)
            )
        )
        .scalars()
        .all()
    )
    items = []
    for student in rows:
        enrollment = (
            await session.execute(
                select(Enrollment, Batch)
                .join(Batch, Batch.id == Enrollment.batch_id)
                .where(Enrollment.student_id == student.id)
                .limit(1)
            )
        ).first()
        submission_count = (
            await session.execute(
                select(func.count(Submission.id)).where(Submission.student_id == student.id)
            )
        ).scalar_one()
        avg = (
            await session.execute(
                select(func.avg(Submission.percentage)).where(
                    Submission.student_id == student.id, Submission.percentage.isnot(None)
                )
            )
        ).scalar_one()
        items.append(
            {
                "student": student,
                "batch": enrollment[1] if enrollment else None,
                "submissions": submission_count,
                "avg": round(float(avg), 1) if avg is not None else None,
            }
        )
    context = {
        "active": "students",
        "items": items,
        "page_meta": page_meta,
        "q": q,
        "per_page": per_page,
        "per_page_choices": PER_PAGE_CHOICES,
    }
    if request.headers.get("HX-Request"):
        return render(request, "partials/students_table.html", context)
    return render(request, "students.html", context)


@router.get("/students/{student_id}")
async def student_detail(
    student_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    student = await session.get(Student, student_id)
    if student is None:
        raise HTTPException(404, "Student not found")
    enrollment = (
        await session.execute(
            select(Enrollment, Batch, Klass)
            .join(Batch, Batch.id == Enrollment.batch_id)
            .join(Klass, Klass.id == Batch.class_id)
            .where(Enrollment.student_id == student_id)
            .limit(1)
        )
    ).first()

    submissions = (
        (
            await session.execute(
                select(Submission, Exam)
                .join(Exam, Exam.id == Submission.exam_id)
                .where(Submission.student_id == student_id)
                .order_by(Submission.id.desc())
            )
        )
        .all()
    )
    submission_rows = []
    for submission, exam in submissions:
        current_run = (
            await session.execute(
                select(ScoringRun)
                .where(
                    ScoringRun.submission_id == submission.id,
                    ScoringRun.is_current.is_(True),
                )
                .order_by(ScoringRun.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        question_rows = await _question_rows(session, submission, current_run)
        submission_rows.append(
            {
                "submission": submission,
                "exam": exam,
                "questions": question_rows,
                "run": current_run,
            }
        )
    return render(
        request,
        "student_detail.html",
        {
            "active": "students",
            "student": student,
            "enrollment": enrollment,
            "submissions": submission_rows,
        },
    )


async def _question_rows(session: AsyncSession, submission: Submission, run: ScoringRun | None):
    if run is None:
        return []
    questions = (
        (
            await session.execute(
                select(Question).where(Question.exam_id == submission.exam_id).order_by(Question.id)
            )
        )
        .scalars()
        .all()
    )
    rows = []
    for question in questions:
        answer = (
            await session.execute(
                select(Answer).where(
                    Answer.submission_id == submission.id, Answer.question_id == question.id
                )
            )
        ).scalar_one_or_none()
        scores = []
        if answer is not None:
            scores = (
                (
                    await session.execute(
                        select(Score).where(
                            Score.scoring_run_id == run.id, Score.answer_id == answer.id
                        )
                    )
                )
                .scalars()
                .all()
            )
        awarded = sum(s.awarded for s in scores)
        max_score = sum(s.max_score for s in scores) or question.max_score
        rows.append(
            {
                "question": question,
                "answer": answer,
                "scores": scores,
                "awarded": round(awarded, 2),
                "max_score": round(max_score, 2),
            }
        )
    return rows
