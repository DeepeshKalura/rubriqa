"""Review queue: blocking validation issues + repair decisions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    Answer,
    Document,
    Exam,
    Page,
    Question,
    ReadingRun,
    Repair,
    Student,
    Submission,
    ValidationIssue,
)
from app.routers.common import pagination, render

router = APIRouter(tags=["review"])

PER_PAGE = 25


@router.get("/review")
async def review_queue(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = 1,
    severity: str = "error",
):
    query = (
        select(ValidationIssue, Submission, Student, Exam)
        .join(Submission, Submission.id == ValidationIssue.submission_id)
        .join(Student, Student.id == Submission.student_id)
        .join(Exam, Exam.id == Submission.exam_id)
        .where(ValidationIssue.resolved.is_(False))
    )
    count_query = (
        select(func.count(ValidationIssue.id))
        .where(ValidationIssue.resolved.is_(False))
    )
    if severity:
        query = query.where(ValidationIssue.severity == severity)
        count_query = count_query.where(ValidationIssue.severity == severity)
    total = (await session.execute(count_query)).scalar_one()
    page_meta = pagination(page, PER_PAGE, total)
    rows = (
        (
            await session.execute(
                query.order_by(ValidationIssue.id.desc())
                .limit(PER_PAGE)
                .offset((page_meta["page"] - 1) * PER_PAGE)
            )
        )
        .all()
    )
    items = [
        {"issue": issue, "submission": submission, "student": student, "exam": exam}
        for issue, submission, student, exam in rows
    ]
    pending_repairs = (
        (
            await session.execute(
                select(Repair)
                .where(Repair.accepted.is_(False), Repair.accepted_by.is_(None))
                .order_by(Repair.id.desc())
                .limit(20)
            )
        )
        .scalars()
        .all()
    )
    return render(
        request,
        "review.html",
        {
            "active": "review",
            "items": items,
            "page_meta": page_meta,
            "severity": severity,
            "pending_repairs": pending_repairs,
        },
    )


@router.get("/review/repair/{repair_id}")
async def repair_detail(
    repair_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    repair = await session.get(Repair, repair_id)
    if repair is None:
        raise HTTPException(404, "Repair not found")
    run = await session.get(ReadingRun, repair.reading_run_id)
    page = await session.get(Page, run.page_id) if run else None
    document = await session.get(Document, page.document_id) if page else None
    submission = await session.get(Submission, document.submission_id) if document else None
    student = await session.get(Student, submission.student_id) if submission else None
    exam = await session.get(Exam, submission.exam_id) if submission else None
    return render(
        request,
        "repair_detail.html",
        {
            "active": "review",
            "repair": repair,
            "run": run,
            "page": page,
            "document": document,
            "submission": submission,
            "student": student,
            "exam": exam,
        },
    )


@router.post("/review/{issue_id}/resolve")
async def resolve_issue(
    issue_id: int,
    session: AsyncSession = Depends(get_session),
):
    issue = await session.get(ValidationIssue, issue_id)
    if issue is None:
        raise HTTPException(404, "Issue not found")
    issue.resolved = True
    issue.resolution = "manually marked resolved"
    await session.commit()
    return RedirectResponse("/review", status_code=303)
