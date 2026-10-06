"""Classes and batches."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    Batch,
    Enrollment,
    Exam,
    Klass,
    Student,
    Submission,
)
from app.routers.common import pagination, render
from app.services import stats

router = APIRouter(tags=["classes"])


@router.get("/classes")
async def classes(request: Request, session: AsyncSession = Depends(get_session)):
    tree = await stats.org_tree(session)
    return render(request, "classes.html", {"active": "classes", "tree": tree})


@router.get("/classes/{class_id}")
async def class_detail(
    class_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    klass = await session.get(Klass, class_id)
    if klass is None:
        raise HTTPException(404, "Class not found")
    rows = []
    batches = (
        (await session.execute(select(Batch).where(Batch.class_id == class_id).order_by(Batch.name)))
        .scalars()
        .all()
    )
    for batch in batches:
        student_count = (
            await session.execute(
                select(func.count(Enrollment.id)).where(Enrollment.batch_id == batch.id)
            )
        ).scalar_one()
        rows.append({"batch": batch, "students": student_count})
    exams = (
        (await session.execute(select(Exam).where(Exam.class_id == class_id).order_by(Exam.name)))
        .scalars()
        .all()
    )
    exam_rows = []
    for exam in exams:
        counts = dict(
            (
                await session.execute(
                    select(Submission.state, func.count(Submission.id))
                    .where(Submission.exam_id == exam.id)
                    .group_by(Submission.state)
                )
            ).all()
        )
        exam_rows.append({"exam": exam, "states": counts})
    return render(
        request,
        "class_detail.html",
        {"active": "classes", "klass": klass, "rows": rows, "exams": exam_rows},
    )


@router.get("/batches/{batch_id}")
async def batch_detail(
    batch_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    batch = await session.get(Batch, batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    klass = await session.get(Klass, batch.class_id)
    enrollments = (
        (
            await session.execute(
                select(Enrollment, Student)
                .join(Student, Student.id == Enrollment.student_id)
                .where(Enrollment.batch_id == batch_id)
                .order_by(Student.external_id)
            )
        )
        .all()
    )
    students = []
    for enrollment, student in enrollments:
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
        students.append(
            {
                "student": student,
                "roll": enrollment.roll_no_in_batch,
                "submissions": submission_count,
                "avg": round(float(avg), 1) if avg is not None else None,
            }
        )
    return render(
        request,
        "batch_detail.html",
        {"active": "classes", "batch": batch, "klass": klass, "students": students},
    )
