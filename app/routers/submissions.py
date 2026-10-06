"""Submission detail: documents, pages, readings/repairs, answers, scores.

Also hosts page-level actions (accept/override repair, reprocess page) and
submission-level actions (retry, rescore).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    ACTIVE_STATES,
    Answer,
    Document,
    Exam,
    Job,
    JobType,
    Page,
    Question,
    ReadingRun,
    Repair,
    Score,
    ScoringRun,
    Student,
    Submission,
    ValidationIssue,
)
from app.routers.common import human_bytes, render
from app.services import jobs as jobs_service

router = APIRouter(tags=["submissions"])


@router.get("/submissions/{submission_id}")
async def submission_detail(
    submission_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    submission = await session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404, "Submission not found")
    student = await session.get(Student, submission.student_id)
    exam = await session.get(Exam, submission.exam_id)

    documents = (
        (
            await session.execute(
                select(Document)
                .where(Document.submission_id == submission_id)
                .order_by(Document.id)
            )
        )
        .scalars()
        .all()
    )
    # One round-trip per page, and one for all repairs across the submission.
    document_rows = []
    all_pages: list[Page] = []
    pages_by_document: dict[int, list[Page]] = {}
    for document in documents:
        pages = (
            (
                await session.execute(
                    select(Page).where(Page.document_id == document.id).order_by(Page.index)
                )
            )
            .scalars()
            .all()
        )
        pages_by_document[document.id] = pages
        all_pages.extend(pages)

    runs_by_page: dict[int, list] = {}
    if all_pages:
        reading_runs = (
            (
                await session.execute(
                    select(ReadingRun)
                    .where(ReadingRun.page_id.in_([p.id for p in all_pages]))
                    .order_by(ReadingRun.id.desc())
                )
            )
            .scalars()
            .all()
        )
        for run in reading_runs:
            runs_by_page.setdefault(run.page_id, []).append(run)
    repairs_by_run: dict[int, list] = {}
    if runs_by_page:
        all_run_ids = [r.id for runs in runs_by_page.values() for r in runs]
        if all_run_ids:
            repairs = (
                (
                    await session.execute(
                        select(Repair)
                        .where(Repair.reading_run_id.in_(all_run_ids))
                        .order_by(Repair.id.desc())
                    )
                )
                .scalars()
                .all()
            )
            for repair in repairs:
                repairs_by_run.setdefault(repair.reading_run_id, []).append(repair)

    for document in documents:
        page_rows = []
        for page in pages_by_document.get(document.id, []):
            runs = runs_by_page.get(page.id, [])
            selected = next((r for r in runs if r.is_selected), runs[0] if runs else None)
            repairs: list = []
            for run in runs:
                repairs.extend(repairs_by_run.get(run.id, []))
            page_rows.append(
                {"page": page, "runs": runs, "selected": selected, "repairs": repairs}
            )
        document_rows.append({"document": document, "pages": page_rows})

    questions = (
        (
            await session.execute(
                select(Question).where(Question.exam_id == submission.exam_id).order_by(Question.id)
            )
        )
        .scalars()
        .all()
    )
    current_run = (
        await session.execute(
            select(ScoringRun)
            .where(ScoringRun.submission_id == submission_id, ScoringRun.is_current.is_(True))
            .order_by(ScoringRun.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    runs_count = (
        (
            await session.execute(
                select(ScoringRun).where(ScoringRun.submission_id == submission_id)
            )
        )
        .scalars()
        .all()
    )
    question_rows = []
    for question in questions:
        answer = (
            await session.execute(
                select(Answer).where(
                    Answer.submission_id == submission_id, Answer.question_id == question.id
                )
            )
        ).scalar_one_or_none()
        scores = []
        if answer and current_run:
            scores = (
                (
                    await session.execute(
                        select(Score).where(
                            Score.scoring_run_id == current_run.id, Score.answer_id == answer.id
                        )
                    )
                )
                .scalars()
                .all()
            )
        question_rows.append(
            {
                "question": question,
                "answer": answer,
                "scores": scores,
                "awarded": round(sum(s.awarded for s in scores), 2),
                "max_score": round(sum(s.max_score for s in scores), 2) or question.max_score,
            }
        )

    issues = (
        (
            await session.execute(
                select(ValidationIssue)
                .where(ValidationIssue.submission_id == submission_id)
                .order_by(ValidationIssue.severity, ValidationIssue.id)
            )
        )
        .scalars()
        .all()
    )
    active_job = (
        await session.execute(
            select(Job)
            .where(Job.submission_id == submission_id, Job.state.in_(list(ACTIVE_STATES)))
            .order_by(Job.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    last_job = (
        await session.execute(
            select(Job).where(Job.submission_id == submission_id).order_by(Job.id.desc()).limit(1)
        )
    ).scalar_one_or_none()

    return render(
        request,
        "submission_detail.html",
        {
            "active": "results",
            "submission": submission,
            "student": student,
            "exam": exam,
            "documents": document_rows,
            "questions": question_rows,
            "issues": issues,
            "current_run": current_run,
            "runs_count": len(runs_count),
            "active_job": active_job,
            "last_job": last_job,
            "human_bytes": human_bytes,
        },
    )


@router.post("/repairs/{repair_id}/accept")
async def accept_repair(
    repair_id: int,
    session: AsyncSession = Depends(get_session),
    submission_id: int = Form(...),
):
    repair = await session.get(Repair, repair_id)
    if repair is None:
        raise HTTPException(404, "Repair not found")
    run = await session.get(ReadingRun, repair.reading_run_id)
    repair.accepted = True
    repair.accepted_by = "human"
    if run is not None:
        run.is_selected = False
        session.add(
            ReadingRun(
                page_id=run.page_id,
                provider=repair.provider,
                model=repair.model,
                mode="repair_manual",
                version=run.version,
                status="ok",
                text=repair.repaired_text,
                confidence=repair.repaired_confidence,
                warnings="accepted_by_human",
                is_selected=True,
            )
        )
        page = await session.get(Page, run.page_id)
        if page is not None:
            page.direct_text = repair.repaired_text
    await session.commit()
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)


@router.post("/repairs/{repair_id}/reject")
async def reject_repair(
    repair_id: int,
    session: AsyncSession = Depends(get_session),
    submission_id: int = Form(...),
):
    repair = await session.get(Repair, repair_id)
    if repair is None:
        raise HTTPException(404, "Repair not found")
    repair.accepted = False
    repair.accepted_by = "human"
    await session.commit()
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)


@router.post("/submissions/{submission_id}/retry")
async def retry_submission(
    submission_id: int, session: AsyncSession = Depends(get_session)
):
    submission = await session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404, "Submission not found")
    active = (
        await session.execute(
            select(Job)
            .where(Job.submission_id == submission_id, Job.state.in_(list(ACTIVE_STATES)))
            .order_by(Job.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if active is not None:
        return RedirectResponse(f"/submissions/{submission_id}", status_code=303)
    last = (
        await session.execute(
            select(Job).where(Job.submission_id == submission_id).order_by(Job.id.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if last is not None and last.state.value == "failed":
        await jobs_service.retry_job(session, last)
    else:
        await jobs_service.enqueue(
            session, JobType.PROCESS_SUBMISSION, submission_id=submission_id, priority=50
        )
        await session.commit()
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)


@router.post("/submissions/{submission_id}/rescore")
async def rescore_submission(
    submission_id: int, session: AsyncSession = Depends(get_session)
):
    submission = await session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404, "Submission not found")
    await jobs_service.enqueue(
        session, JobType.RESCORE_SUBMISSION, submission_id=submission_id, priority=50
    )
    await session.commit()
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)


@router.post("/pages/{page_id}/reprocess")
async def reprocess_page(
    page_id: int, session: AsyncSession = Depends(get_session)
):
    page = await session.get(Page, page_id)
    if page is None:
        raise HTTPException(404, "Page not found")
    document = await session.get(Document, page.document_id)
    submission_id = document.submission_id if document else None
    await jobs_service.enqueue(
        session,
        JobType.REPROCESS_PAGE,
        page_id=page_id,
        document_id=page.document_id,
        submission_id=submission_id,
        params={"force_pages": True},
        priority=40,
    )
    await session.commit()
    return RedirectResponse(f"/submissions/{submission_id}", status_code=303)
