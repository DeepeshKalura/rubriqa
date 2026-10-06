"""Jobs list + detail with live HTMX polling and retry."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Job, JobEvent, JobState, Student, Submission
from app.routers.common import pagination, render
from app.services import jobs as jobs_service

router = APIRouter(tags=["jobs"])

PER_PAGE = 25


@router.get("/jobs")
async def jobs_list(
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = 1,
    state: str = "",
):
    query = select(Job)
    count_query = select(func.count(Job.id))
    if state:
        try:
            state_enum = JobState(state)
            query = query.where(Job.state == state_enum)
            count_query = count_query.where(Job.state == state_enum)
        except ValueError:
            state = ""
    total = (await session.execute(count_query)).scalar_one()
    page_meta = pagination(page, PER_PAGE, total)
    rows = (
        (
            await session.execute(
                query.order_by(Job.id.desc()).limit(PER_PAGE).offset((page_meta["page"] - 1) * PER_PAGE)
            )
        )
        .scalars()
        .all()
    )
    items = []
    for job in rows:
        student_name = None
        if job.submission_id:
            result = (
                await session.execute(
                    select(Student.name)
                    .join(Submission, Submission.student_id == Student.id)
                    .where(Submission.id == job.submission_id)
                )
            ).first()
            student_name = result[0] if result else None
        items.append({"job": job, "student_name": student_name})
    context = {
        "active": "jobs",
        "items": items,
        "page_meta": page_meta,
        "state": state,
        "states": [s.value for s in JobState],
    }
    if request.headers.get("HX-Request") and request.headers.get("HX-Target") == "jobs-table":
        return render(request, "partials/jobs_table.html", context)
    return render(request, "jobs.html", context)


@router.get("/jobs/{job_id}")
async def job_detail(
    job_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    events = (
        (
            await session.execute(
                select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id.desc())
            )
        )
        .scalars()
        .all()
    )
    student_name = None
    if job.submission_id:
        result = (
            await session.execute(
                select(Student.name)
                .join(Submission, Submission.student_id == Student.id)
                .where(Submission.id == job.submission_id)
            )
        ).first()
        student_name = result[0] if result else None
    context = {"active": "jobs", "job": job, "events": events, "student_name": student_name}
    if request.headers.get("HX-Request"):
        return render(request, "partials/job_detail_body.html", context)
    return render(request, "job_detail.html", context)


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: int, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.state is JobState.COMPLETED:
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)
    if job.state is JobState.FAILED:
        await jobs_service.retry_job(session, job)
    else:
        await jobs_service.retry_job(session, job)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    job.state = JobState.CANCELLED
    await session.commit()
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)
