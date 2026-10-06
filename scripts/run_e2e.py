"""End-to-end run over the demo dataset with DB verification.

Flow: reseed -> import -> process -> verify scores/pages/traceability in DB.

Usage:
    python -m scripts.run_e2e [--expect-fake] [--timeout 1800]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from sqlalchemy import func, select

from app.config import get_settings
from app.db import get_sessionmaker, init_engine
from app.logging import configure_logging, get_logger
from app.models import (
    ACTIVE_STATES,
    Answer,
    Document,
    Exam,
    Job,
    JobType,
    Page,
    ReadingRun,
    Repair,
    Score,
    ScoringRun,
    Student,
    Submission,
    SubmissionState,
)
from app.services import jobs as jobs_service
from app.services.importer import (
    DiscoveryReport,
    enqueue_submissions,
    register_documents,
    scan_folder,
)
from app.services.storage import Storage

logger = get_logger("scripts.e2e")

DATA = Path("/data/demo")


async def main(expect: str, timeout: int) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings)
    maker = get_sessionmaker()
    started = time.time()

    async with maker() as session:
        exam = (
            await session.execute(select(Exam).order_by(Exam.id).limit(1))
        ).scalar_one_or_none()
        if exam is None:
            print("No exam found. Run `make seed` first.")
            return 2
        print(f"Exam: #{exam.id} {exam.name}")
        print(f"Provider: {settings.llm_provider} · Reader: {settings.reader_mode}")

        # 1. Import with dry-run report
        plan = scan_folder(DATA)
        existing: set[str] = set()
        report = DiscoveryReport.from_plan(plan, existing)
        print(f"Scan: {report.summary()}")
        batch = await register_documents(session, exam_id=exam.id, plan=plan)
        print(
            f"Registered: new={batch.new_documents} duplicates={batch.already_imported} "
            f"submissions_created={batch.submissions_created}"
        )

        # 2. Re-import immediately: must be a no-op (idempotency proof)
        plan2 = scan_folder(DATA)
        existing_rows = (
            await session.execute(
                select(Document.sha256)
                .join(Submission, Submission.id == Document.submission_id)
                .where(Submission.exam_id == exam.id)
            )
        ).scalars().all()
        report2 = DiscoveryReport.from_plan(plan2, set(existing_rows))
        print(f"Re-import dry-run: {report2.summary()}")
        batch2 = await register_documents(session, exam_id=exam.id, plan=plan2)
        assert batch2.new_documents == 0, "re-import must not create documents!"
        print("Idempotency: PASS (0 new documents on re-import)")

        # 3. Queue and wait
        queued = await enqueue_submissions(session, exam.id)
        print(f"Queued {queued} submission job(s)")
        await session.commit()

    # 4. Wait for completion
    while time.time() - started < timeout:
        async with maker() as session:
            active = (
                await session.execute(
                    select(func.count(Job.id)).where(Job.state.in_(list(ACTIVE_STATES)))
                )
            ).scalar_one()
        if active == 0:
            break
        await asyncio.sleep(3)
    else:
        print("TIMEOUT waiting for jobs")
        return 1
    elapsed = time.time() - started

    # 5. Verify
    async with maker() as session:
        submissions = (
            (
                await session.execute(
                    select(Submission).where(Submission.exam_id == exam.id).order_by(Submission.id)
                )
            )
            .scalars()
            .all()
        )
        documents = (
            await session.execute(
                select(func.count(Document.id))
                .join(Submission, Submission.id == Document.submission_id)
                .where(Submission.exam_id == exam.id)
            )
        ).scalar_one()
        pages = (
            await session.execute(
                select(func.count(Page.id))
                .join(Document, Document.id == Page.document_id)
                .join(Submission, Submission.id == Document.submission_id)
                .where(Submission.exam_id == exam.id)
            )
        ).scalar_one()
        readings = (
            await session.execute(
                select(func.count(ReadingRun.id))
                .join(Page, Page.id == ReadingRun.page_id)
                .join(Document, Document.id == Page.document_id)
                .join(Submission, Submission.id == Document.submission_id)
                .where(Submission.exam_id == exam.id)
            )
        ).scalar_one()
        answers = (
            await session.execute(
                select(func.count(Answer.id)).join(
                    Submission, Submission.id == Answer.submission_id
                ).where(Submission.exam_id == exam.id)
            )
        ).scalar_one()
        scores = (
            await session.execute(
                select(func.count(Score.id))
                .join(ScoringRun, ScoringRun.id == Score.scoring_run_id)
                .where(ScoringRun.is_current.is_(True))
            )
        ).scalar_one()
        repairs = (await session.execute(select(func.count(Repair.id)))).scalar_one()
        scored = [
            s for s in submissions if s.state is SubmissionState.COMPLETED
        ]
        review = [s for s in submissions if s.state is SubmissionState.NEEDS_REVIEW]
        failed = [s for s in submissions if s.state is SubmissionState.FAILED]

        print()
        print("=" * 62)
        print("E2E RESULT")
        print("=" * 62)
        print(f"  duration          : {elapsed:.1f}s")
        print(f"  documents         : {documents}")
        print(f"  pages             : {pages}")
        print(f"  reading runs      : {readings}")
        print(f"  repairs           : {repairs}")
        print(f"  answers extracted : {answers}")
        print(f"  scores (current)  : {scores}")
        print(
            f"  submissions       : completed={len(scored)} needs_review={len(review)} "
            f"failed={len(failed)}"
        )
        for submission in submissions:
            student = await session.get(Student, submission.student_id)
            print(
                f"    {student.external_id:8s} {student.name:16s} "
                f"{submission.state.value:12s} "
                f"{submission.total_score}/{submission.max_score} "
                f"({submission.percentage}%)"
            )

        checks = {
            "documents imported": documents >= 25,
            "pages registered": pages >= 30,
            "answers extracted": answers >= 15,
            "scores recorded": scores >= 15,
            "at least 4 completed": len(scored) >= 4,
            "no failed submissions": len(failed) == 0,
        }
        print()
        ok = True
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            ok = ok and passed

        evidence = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "provider": expect,
            "reader_mode": settings.reader_mode,
            "duration_seconds": round(elapsed, 1),
            "documents": documents,
            "pages": pages,
            "reading_runs": readings,
            "repairs": repairs,
            "answers": answers,
            "scores": scores,
            "submissions": {
                "completed": len(scored),
                "needs_review": len(review),
                "failed": len(failed),
            },
            "per_student": [
                {
                    "external_id": student.external_id,
                    "name": student.name,
                    "state": submission.state.value,
                    "score": submission.total_score,
                    "max": submission.max_score,
                    "percentage": submission.percentage,
                }
                for submission in submissions
                for student in [await session.get(Student, submission.student_id)]
            ],
        }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = reports / f"e2e_{stamp}.json"
    out.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"\nEvidence written to {out}")
    return 0 if ok else 1


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-fake", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    code = asyncio.run(main("fake" if args.expect_fake else "minimax", args.timeout))
    raise SystemExit(code)


if __name__ == "__main__":
    run()
