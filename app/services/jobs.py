"""Job helpers: enqueue, claim, heartbeat, stage transitions, retry, sweep."""
from __future__ import annotations

import socket
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.logging import get_logger
from app.models import ACTIVE_STATES, Job, JobEvent, JobState, JobType

logger = get_logger("exam_eval.jobs")

WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def enqueue(
    session: AsyncSession,
    job_type: JobType,
    *,
    submission_id: int | None = None,
    document_id: int | None = None,
    page_id: int | None = None,
    import_batch_id: int | None = None,
    params: dict | None = None,
    priority: int = 100,
) -> Job:
    """Idempotent enqueue: a submission can only have one active job."""
    if submission_id is not None:
        existing = (
            await session.execute(
                select(Job).where(
                    Job.submission_id == submission_id, Job.state.in_(list(ACTIVE_STATES))
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
    job = Job(
        type=job_type,
        state=JobState.QUEUED,
        submission_id=submission_id,
        document_id=document_id,
        page_id=page_id,
        import_batch_id=import_batch_id,
        params_json=params or {},
        priority=priority,
        max_attempts=get_settings().job_max_attempts,
        checkpoint={},
    )
    session.add(job)
    await session.flush()
    await add_event(session, job, "queued", "info", "Job queued")
    return job


async def add_event(
    session: AsyncSession,
    job: Job,
    stage: str,
    status: str,
    message: str | None = None,
    data: dict | None = None,
    duration_ms: int | None = None,
) -> None:
    from app.services.textutil import sanitize_optional

    session.add(
        JobEvent(
            job_id=job.id,
            stage=stage,
            status=status,
            message=sanitize_optional(message),
            data_json=data,
            duration_ms=duration_ms,
        )
    )


async def claim_next_job(session: AsyncSession) -> Job | None:
    """Claim one queued job with FOR UPDATE SKIP LOCKED. Returns None if empty."""
    result = await session.execute(
        text(
            """
            SELECT id FROM jobs
            WHERE state = 'queued'
            ORDER BY priority ASC, created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
    )
    row = result.first()
    if row is None:
        return None
    job = await session.get(Job, row[0], with_for_update=True)
    assert job is not None
    job.state = JobState.RUNNING
    job.locked_by = WORKER_ID
    job.locked_at = utcnow()
    job.heartbeat_at = utcnow()
    job.started_at = job.started_at or utcnow()
    job.attempts += 1
    await add_event(
        session, job, "claim", "info", f"Claimed by {WORKER_ID} (attempt {job.attempts})"
    )
    await session.commit()
    return job


async def heartbeat(session: AsyncSession, job_id: int) -> None:
    await session.execute(
        update(Job).where(Job.id == job_id).values(heartbeat_at=utcnow())
    )
    await session.commit()


async def transition(
    session: AsyncSession,
    job: Job,
    state: JobState,
    message: str | None = None,
    *,
    progress_current: int | None = None,
    progress_total: int | None = None,
) -> None:
    job.state = state
    if progress_current is not None:
        job.progress_current = progress_current
    if progress_total is not None:
        job.progress_total = progress_total
    job.heartbeat_at = utcnow()
    await add_event(session, job, state.value, "info", message)
    await session.commit()


async def complete(session: AsyncSession, job: Job, message: str = "Job completed") -> None:
    job.state = JobState.COMPLETED
    job.finished_at = utcnow()
    job.locked_by = None
    job.error = None
    job.error_code = None
    await add_event(session, job, "completed", "ok", message)
    await session.commit()


async def fail(
    session: AsyncSession, job: Job, error: str, *, code: str = "job_failed"
) -> None:
    job.state = JobState.FAILED
    job.finished_at = utcnow()
    job.locked_by = None
    job.error = error[:1500]
    job.error_code = code
    await add_event(session, job, "failed", "error", error[:1500])
    # keep the submission state honest: a failed job must not leave it "processing"
    if job.submission_id is not None:
        from app.models import Submission, SubmissionState

        submission = await session.get(Submission, job.submission_id)
        if submission is not None and submission.state not in (
            SubmissionState.COMPLETED,
            SubmissionState.NEEDS_REVIEW,
        ):
            submission.state = SubmissionState.FAILED
    await session.commit()


async def retry_job(session: AsyncSession, job: Job, *, reset_attempts: bool = False) -> Job:
    """Manual retry from the UI. Keeps checkpoint so completed stages are skipped."""
    if reset_attempts:
        job.attempts = 0
    job.state = JobState.QUEUED
    job.locked_by = None
    job.locked_at = None
    job.finished_at = None
    job.error = None
    job.error_code = None
    await add_event(session, job, "requeued", "info", "Requeued by user")
    await session.commit()
    return job


async def sweep_stale_jobs(session: AsyncSession) -> int:
    """Requeue jobs whose worker died. Returns number of jobs requeued/failed.

    This is the crash-recovery path: the sweeper does not need the original
    process to cooperate.
    """
    settings = get_settings()
    cutoff = utcnow() - timedelta(minutes=settings.job_stuck_minutes)
    result = await session.execute(
        select(Job).where(Job.state.in_(list(ACTIVE_STATES)), Job.heartbeat_at < cutoff)
    )
    stale = result.scalars().all()
    processed = 0
    for job in stale:
        # A job that never heartbeated at all (RUNNING but heartbeat null) is
        # also stale: treat locked_at/created_at as the reference.
        if job.attempts >= job.max_attempts:
            job.state = JobState.FAILED
            job.error_code = "stale_worker"
            job.error = (
                f"No heartbeat for over {settings.job_stuck_minutes} minutes after "
                f"{job.attempts} attempts. The worker likely stopped; check logs and retry."
            )
            job.finished_at = utcnow()
            job.locked_by = None
            await add_event(session, job, "failed", "error", job.error)
        else:
            job.state = JobState.QUEUED
            job.locked_by = None
            job.locked_at = None
            await add_event(
                session,
                job,
                "stale_requeue",
                "info",
                f"Requeued after stale heartbeat (attempt {job.attempts}/{job.max_attempts})",
            )
        processed += 1
    await session.commit()
    return processed
