"""Requeue failed/needs-review submissions and wait for the worker to finish."""
from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import func, select

from app.config import get_settings
from app.db import get_sessionmaker, init_engine
from app.logging import configure_logging
from app.models import (
    ACTIVE_STATES,
    Job,
    Submission,
    SubmissionState,
)
from app.services import jobs as jobs_service


async def requeue_all() -> int:
    maker = get_sessionmaker()
    async with maker() as session:
        submissions = (
            (
                await session.execute(
                    select(Submission).where(
                        Submission.state.in_(
                            [
                                SubmissionState.PENDING,
                                SubmissionState.PROCESSING,
                                SubmissionState.NEEDS_REVIEW,
                                SubmissionState.FAILED,
                            ]
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        from app.models import JobType

        queued = 0
        for submission in submissions:
            active = (
                await session.execute(
                    select(Job).where(
                        Job.submission_id == submission.id,
                        Job.state.in_(list(ACTIVE_STATES)),
                    )
                )
            ).scalar_one_or_none()
            if active:
                continue
            await jobs_service.enqueue(
                session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
            )
            if submission.state is not SubmissionState.PENDING:
                submission.state = SubmissionState.PENDING
                submission.needs_review = False
                submission.notes = None
            queued += 1
        await session.commit()
        return queued


async def wait_for_idle(timeout: int = 1800, poll: int = 3) -> bool:
    maker = get_sessionmaker()
    started = time.time()
    while time.time() - started < timeout:
        async with maker() as session:
            active = (
                await session.execute(
                    select(func.count(Job.id)).where(Job.state.in_(list(ACTIVE_STATES)))
                )
            ).scalar_one()
        if active == 0:
            return True
        await asyncio.sleep(poll)
    return False


async def summary() -> None:
    maker = get_sessionmaker()
    async with maker() as session:
        states = (
            await session.execute(
                select(Submission.state, func.count(Submission.id)).group_by(Submission.state)
            )
        ).all()
        jobs = (
            await session.execute(select(Job.state, func.count(Job.id)).group_by(Job.state))
        ).all()
    print("submissions:", {state.value: count for state, count in states})
    print("jobs:", {state.value: count for state, count in jobs})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requeue", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings)

    async def run():
        if args.requeue:
            queued = await requeue_all()
            print(f"requeued {queued} submission(s)")
        if args.wait:
            ok = await wait_for_idle(timeout=args.timeout)
            print("idle" if ok else "TIMEOUT waiting for jobs")
        await summary()

    asyncio.run(run())


if __name__ == "__main__":
    main()
