"""Results overview + Google Forms OAuth routes are added in their own phase.
This module keeps the results list global."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Exam, Klass, Student, Submission
from app.routers.common import pagination, render

router = APIRouter(tags=["results"])

PER_PAGE = 25


@router.get("/results")
async def results_overview(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = 1,
    exam_id: int | None = None,
):
    query = (
        select(Submission, Student, Exam)
        .join(Student, Student.id == Submission.student_id)
        .join(Exam, Exam.id == Submission.exam_id)
        .order_by(Submission.id.desc())
    )
    count_query = select(func.count(Submission.id))
    if exam_id:
        query = query.where(Submission.exam_id == exam_id)
        count_query = count_query.where(Submission.exam_id == exam_id)
    total = (await session.execute(count_query)).scalar_one()
    page_meta = pagination(page, PER_PAGE, total)
    rows = (
        (
            await session.execute(
                query.limit(PER_PAGE).offset((page_meta["page"] - 1) * PER_PAGE)
            )
        )
        .all()
    )
    exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
    return render(
        request,
        "results.html",
        {
            "active": "results",
            "items": rows,
            "page_meta": page_meta,
            "exams": exams,
            "exam_id": exam_id,
        },
    )
