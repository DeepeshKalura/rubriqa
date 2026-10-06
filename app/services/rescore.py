"""Targeted rescore: re-run scoring for one submission under the current flags.

Keeps the old scoring run (is_current=False) so history is auditable.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging import get_logger
from app.models import Answer, Job, Question, Score, ScoringRun, Submission
from app.pipeline.scoring.factory import build_scoring
from app.services.textutil import sanitize_text

logger = get_logger("exam_eval.rescore")


async def rescore_submission(
    session: AsyncSession, settings: Settings, llm, job: Job
) -> None:
    from app.services import jobs as jobs_service
    from app.services.pipeline import SubmissionPipeline

    submission = await session.get(Submission, job.submission_id)
    if submission is None:
        raise ValueError("Submission not found for rescore job")

    pipeline = SubmissionPipeline(session, settings, reader=None, llm=llm)
    questions = (
        (
            await session.execute(
                select(Question).where(Question.exam_id == submission.exam_id).order_by(Question.id)
            )
        )
        .scalars()
        .all()
    )
    answers = (
        (await session.execute(select(Answer).where(Answer.submission_id == submission.id)))
        .scalars()
        .all()
    )
    answers_by_question = {a.question_id: a for a in answers}

    previous_runs = (
        (
            await session.execute(
                select(ScoringRun).where(
                    ScoringRun.submission_id == submission.id, ScoringRun.is_current.is_(True)
                )
            )
        )
        .scalars()
        .all()
    )
    for previous in previous_runs:
        previous.is_current = False

    from app.models import Exam

    exam = await session.get(Exam, submission.exam_id)
    strategy_name = (exam.scoring_strategy if exam else None) or settings.scoring_strategy
    run = ScoringRun(
        submission_id=submission.id,
        strategy=strategy_name,
        provider=settings.llm_provider,
        model=settings.minimax_text_model if settings.llm_provider == "minimax" else "fake",
        is_current=True,
    )
    session.add(run)
    await session.commit()

    total_awarded = total_max = 0.0
    for question in questions:
        answer = answers_by_question.get(question.id)
        rubric = await pipeline._rubric_spec(question)
        strategy = build_scoring(
            question.strategy or strategy_name, llm=llm, settings=settings
        )
        result = await strategy.score(
            {
                "id": question.id,
                "q_no": question.q_no,
                "text": question.text,
                "max_score": question.max_score,
                "answer_key": question.answer_key,
                "answer_key_alternatives": question.answer_key_alternatives,
                "strategy": question.strategy,
            },
            answer.text if answer else "",
            rubric,
            context={"job_id": job.id, "submission_id": submission.id, "question_id": question.id},
        )
        total_max += float(question.max_score)
        total_awarded += result.total_awarded
        for criterion in result.criteria or []:
            if answer is not None:
                session.add(
                    Score(
                        scoring_run_id=run.id,
                        answer_id=answer.id,
                        criterion=criterion.criterion,
                        awarded=criterion.awarded,
                        max_score=criterion.max_score,
                        rationale=sanitize_text(criterion.rationale) or None,
                        feedback=sanitize_text(criterion.feedback) or None,
                        confidence=criterion.confidence,
                        strategy=result.strategy,
                        provider=result.provider,
                        model=result.model,
                    )
                )
    run.total_awarded = round(total_awarded, 2)
    run.total_max = round(total_max, 2)
    submission.total_score = round(total_awarded, 2)
    submission.max_score = round(total_max, 2)
    submission.percentage = round(100.0 * total_awarded / total_max, 1) if total_max else 0.0
    await session.commit()
    await jobs_service.complete(session, job, "Rescored under current strategy")
