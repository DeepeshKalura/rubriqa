"""Exams: list, detail (questions + rubrics), results table, CSV export."""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    Answer,
    Exam,
    Job,
    JobType,
    Klass,
    Question,
    Rubric,
    RubricCriterion,
    Score,
    ScoringRun,
    Student,
    Submission,
)
from app.routers.common import pagination, render
from app.services import jobs as jobs_service
from app.services import stats
from app.models import ACTIVE_STATES

router = APIRouter(tags=["exams"])

PER_PAGE_CHOICES = (25, 50, 100)


@router.get("/exams")
async def exams(request: Request, session: AsyncSession = Depends(get_session)):
    rows = (
        (
            await session.execute(
                select(Exam, Klass)
                .join(Klass, Klass.id == Exam.class_id)
                .order_by(Exam.id.desc())
            )
        )
        .all()
    )
    # One round-trip per aggregation via LEFT JOIN + GROUP BY.
    exam_ids = [exam.id for exam, _ in rows]
    question_counts = dict(
        (
            await session.execute(
                select(Question.exam_id, func.count(Question.id))
                .where(Question.exam_id.in_(exam_ids))
                .group_by(Question.exam_id)
            )
        ).all()
    )
    submission_states_rows = (
        (
            await session.execute(
                select(Submission.exam_id, Submission.state, func.count(Submission.id))
                .where(Submission.exam_id.in_(exam_ids))
                .group_by(Submission.exam_id, Submission.state)
            )
        ).all()
    )
    states_by_exam: dict[int, dict[str, int]] = {}
    for exam_id, state, count in submission_states_rows:
        states_by_exam.setdefault(exam_id, {})[state.value] = count

    items = [
        {
            "exam": exam,
            "klass": klass,
            "questions": question_counts.get(exam.id, 0),
            "states": states_by_exam.get(exam.id, {}),
        }
        for exam, klass in rows
    ]
    return render(request, "exams.html", {"active": "exams", "items": items})


@router.get("/exams/{exam_id}")
async def exam_detail(
    exam_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    klass = await session.get(Klass, exam.class_id)
    questions = (
        (
            await session.execute(
                select(Question).where(Question.exam_id == exam_id).order_by(Question.id)
            )
        )
        .scalars()
        .all()
    )
    rows = []
    for question in questions:
        rubric = None
        criteria = []
        if question.rubric_id:
            rubric = await session.get(Rubric, question.rubric_id)
            criteria = (
                (
                    await session.execute(
                        select(RubricCriterion)
                        .where(RubricCriterion.rubric_id == question.rubric_id)
                        .order_by(RubricCriterion.id)
                    )
                )
                .scalars()
                .all()
            )
        rows.append({"question": question, "rubric": rubric, "criteria": criteria})
    counts = dict(
        (
            await session.execute(
                select(Submission.state, func.count(Submission.id))
                .where(Submission.exam_id == exam_id)
                .group_by(Submission.state)
            )
        ).all()
    )
    return render(
        request,
        "exam_detail.html",
        {"active": "exams", "exam": exam, "klass": klass, "rows": rows, "states": counts},
    )


@router.get("/exams/{exam_id}/results")
async def exam_results(
    exam_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
    page: int = 1,
    per_page: int = 25,
    state: str = "",
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    klass = await session.get(Klass, exam.class_id)
    per_page = per_page if per_page in PER_PAGE_CHOICES else 25

    base = select(Submission, Student).join(Student, Student.id == Submission.student_id).where(
        Submission.exam_id == exam_id
    )
    count_query = (
        select(func.count(Submission.id)).where(Submission.exam_id == exam_id)
    )
    if state:
        from app.models import SubmissionState

        try:
            state_enum = SubmissionState(state)
            base = base.where(Submission.state == state_enum)
            count_query = count_query.where(Submission.state == state_enum)
        except ValueError:
            state = ""
    total = (await session.execute(count_query)).scalar_one()
    page_meta = pagination(page, per_page, total)
    rows = (
        (
            await session.execute(
                base.order_by(Student.external_id)
                .limit(per_page)
                .offset((page_meta["page"] - 1) * per_page)
            )
        )
        .all()
    )
    items = []
    for submission, student in rows:
        items.append({"submission": submission, "student": student})
    distribution = await stats.score_distribution(session, exam_id)
    context = {
        "active": "exams",
        "exam": exam,
        "klass": klass,
        "items": items,
        "page_meta": page_meta,
        "state": state,
        "per_page": per_page,
        "per_page_choices": PER_PAGE_CHOICES,
        "distribution": distribution,
    }
    if request.headers.get("HX-Request"):
        return render(request, "partials/results_table.html", context)
    return render(request, "exam_results.html", context)


@router.get("/exams/{exam_id}/results.csv")
async def exam_results_csv(
    exam_id: int, session: AsyncSession = Depends(get_session)
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    questions = (
        (
            await session.execute(
                select(Question).where(Question.exam_id == exam_id).order_by(Question.id)
            )
        )
        .scalars()
        .all()
    )
    rows = (
        (
            await session.execute(
                select(Submission, Student)
                .join(Student, Student.id == Submission.student_id)
                .where(Submission.exam_id == exam_id)
                .order_by(Student.external_id)
            )
        )
        .all()
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["student_id", "student_name", "state", "total_score", "max_score", "percentage"]
        + [q.q_no for q in questions]
    )
    for submission, student in rows:
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
        per_question = []
        for question in questions:
            answer = (
                await session.execute(
                    select(Answer).where(
                        Answer.submission_id == submission.id,
                        Answer.question_id == question.id,
                    )
                )
            ).scalar_one_or_none()
            awarded = 0.0
            if answer and current_run:
                scores = (
                    (
                        await session.execute(
                            select(Score).where(
                                Score.scoring_run_id == current_run.id,
                                Score.answer_id == answer.id,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                awarded = round(sum(s.awarded for s in scores), 2)
            per_question.append(awarded)
        writer.writerow(
            [
                student.external_id,
                student.name,
                submission.state.value,
                submission.total_score if submission.total_score is not None else "",
                submission.max_score if submission.max_score is not None else "",
                submission.percentage if submission.percentage is not None else "",
            ]
            + per_question
        )
    buffer.seek(0)
    filename = f"exam_{exam_id}_results.csv"
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/exams/{exam_id}/process")
async def process_exam(
    exam_id: int,
    session: AsyncSession = Depends(get_session),
):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    from app.models import SubmissionState
    from app.services.importer import enqueue_submissions

    queued = await enqueue_submissions(session, exam_id, only_pending=True)
    return {"queued": queued}


@router.post("/exams/{exam_id}/rescore")
async def rescore_exam(exam_id: int, session: AsyncSession = Depends(get_session)):
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")
    submissions = (
        (await session.execute(select(Submission).where(Submission.exam_id == exam_id)))
        .scalars()
        .all()
    )
    queued = 0
    for submission in submissions:
        active = (
            await session.execute(
                select(Job).where(
                    Job.submission_id == submission.id, Job.state.in_(list(ACTIVE_STATES))
                )
            )
        ).scalar_one_or_none()
        if active is not None:
            continue
        await jobs_service.enqueue(
            session, JobType.RESCORE_SUBMISSION, submission_id=submission.id
        )
        queued += 1
    await session.commit()
    return {"queued": queued}
