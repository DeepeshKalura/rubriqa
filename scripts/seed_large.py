"""Seed a large synthetic dataset (default 20,000 submissions) directly into
the DB with bulk inserts, to test pagination/index behaviour at scale.

This is a DATA-LAYER simulation: it does not run OCR/LLM on 20k exams.

Usage:
    python -m scripts.seed_large --submissions 20000 [--answers 10]
"""
from __future__ import annotations

import argparse
import asyncio
import random
import time

from sqlalchemy import func, insert, select

from app.config import get_settings
from app.db import get_sessionmaker, init_engine
from app.logging import configure_logging, get_logger
from app.models import (
    Answer,
    Batch,
    Enrollment,
    Exam,
    Klass,
    Organization,
    Question,
    Score,
    ScoringRun,
    Student,
    Submission,
    SubmissionState,
)

logger = get_logger("scripts.scale")

FIRST_NAMES = ["Aarav", "Priya", "Rahul", "Sneha", "Imran", "Divya", "Arjun", "Kavya", "Rohan", "Ananya"]
LAST_NAMES = ["Sharma", "Verma", "Mehta", "Iyer", "Khan", "Patel", "Reddy", "Singh", "Gupta", "Nair"]


async def seed(submissions_target: int, answers_per: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings)
    maker = get_sessionmaker()
    started = time.perf_counter()

    async with maker() as session:
        org = (
            await session.execute(
                select(Organization).where(Organization.name == "Scale Test Org")
            )
        ).scalar_one_or_none()
        if org is None:
            org = Organization(name="Scale Test Org", code="SCALE")
            session.add(org)
            await session.flush()
        klass = (
            await session.execute(
                select(Klass).where(Klass.organization_id == org.id, Klass.name == "Scale Class")
            )
        ).scalar_one_or_none()
        if klass is None:
            klass = Klass(organization_id=org.id, name="Scale Class")
            session.add(klass)
            await session.flush()
        exam = (
            await session.execute(
                select(Exam).where(Exam.class_id == klass.id, Exam.name == "Scale Exam")
            )
        ).scalar_one_or_none()
        if exam is None:
            exam = Exam(class_id=klass.id, name="Scale Exam", exam_date="2026-01-01", subject="Scale")
            session.add(exam)
            await session.flush()
            for index in range(answers_per):
                session.add(
                    Question(
                        exam_id=exam.id,
                        q_no=f"Q{index + 1}",
                        text=f"Scale question {index + 1}",
                        max_score=1.0,
                        strategy="exact",
                        answer_key="yes",
                    )
                )
            await session.commit()

        questions = (
            await session.execute(select(Question).where(Question.exam_id == exam.id))
        ).scalars().all()
        existing = (
            await session.execute(
                select(func.count(Student.id)).where(Student.organization_id == org.id)
            )
        ).scalar_one()
        print(f"Existing students in scale org: {existing:,}")

        # Batches: 40 batches x ~500 students (reuse if a previous run made them)
        batches = []
        for index in range(40):
            name = f"Scale Batch {index + 1:02d}"
            batch = (
                await session.execute(
                    select(Batch).where(Batch.class_id == klass.id, Batch.name == name)
                )
            ).scalar_one_or_none()
            if batch is None:
                batch = Batch(class_id=klass.id, name=name, year=2026)
                session.add(batch)
            batches.append(batch)
        await session.flush()

        remaining = submissions_target - existing if existing < submissions_target else 0
        if remaining <= 0:
            print("Target already satisfied; nothing to seed.")
            return

        # Bulk insert students in chunks
        chunk = 1000
        all_student_ids: list[int] = []
        rng = random.Random(42)
        print(f"Seeding {remaining:,} students + submissions + answers...")
        student_rows: list[dict] = []
        for index in range(remaining):
            external_id = f"SCALE{existing + index + 1:06d}"
            student_rows.append(
                {
                    "organization_id": org.id,
                    "external_id": external_id,
                    "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                }
            )
        for start in range(0, len(student_rows), chunk):
            await session.execute(insert(Student), student_rows[start : start + chunk])
            await session.commit()
            print(f"  students {start + len(student_rows[start:start+chunk]):,}/{len(student_rows):,}", end="\r")
        print()

        result = await session.execute(
            select(Student.id).where(Student.organization_id == org.id).order_by(Student.id)
        )
        all_student_ids = [row[0] for row in result.all()][existing:]

        # Enrollments
        enrollment_rows = []
        for index, student_id in enumerate(all_student_ids):
            enrollment_rows.append(
                {"batch_id": batches[index % len(batches)].id, "student_id": student_id}
            )
        for start in range(0, len(enrollment_rows), chunk):
            await session.execute(insert(Enrollment), enrollment_rows[start : start + chunk])
        await session.commit()

        # Submissions
        submission_rows = [
            {"exam_id": exam.id, "student_id": student_id} for student_id in all_student_ids
        ]
        for start in range(0, len(submission_rows), chunk):
            await session.execute(insert(Submission), submission_rows[start : start + chunk])
        await session.commit()
        print(f"  submissions: {len(submission_rows):,}")

        submission_ids = [
            row[0]
            for row in (
                await session.execute(
                    select(Submission.id).where(Submission.exam_id == exam.id).order_by(Submission.id)
                )
            ).all()
        ]
        # Only answers/scores for the new submissions (skip previously seeded)
        new_submission_ids = submission_ids[-len(all_student_ids):] if all_student_ids else []

        # Answers
        answer_rows = []
        for submission_id in new_submission_ids:
            for question in questions:
                answer_rows.append(
                    {
                        "submission_id": submission_id,
                        "question_id": question.id,
                        "text": "yes" if rng.random() > 0.3 else "no",
                        "method": "scale_seed",
                        "confidence": 0.9,
                    }
                )
        for start in range(0, len(answer_rows), 5000):
            await session.execute(insert(Answer), answer_rows[start : start + 5000])
        await session.commit()
        print(f"  answers: {len(answer_rows):,}")

        # Scoring runs + scores
        run_rows = [
            {
                "submission_id": submission_id,
                "strategy": "exact",
                "provider": "deterministic",
                "is_current": True,
                "total_awarded": 0,
                "total_max": float(answers_per),
            }
            for submission_id in new_submission_ids
        ]
        for start in range(0, len(run_rows), chunk):
            await session.execute(insert(ScoringRun), run_rows[start : start + chunk])
        await session.commit()

        runs = {
            row[0]: row[1]
            for row in (
                await session.execute(
                    select(ScoringRun.submission_id, ScoringRun.id).where(
                        ScoringRun.submission_id.in_(new_submission_ids)
                    )
                )
            ).all()
        }
        answer_pairs = (
            await session.execute(
                select(Answer.id, Answer.submission_id)
                .where(Answer.submission_id.in_(new_submission_ids))
            )
        ).all()
        score_rows = []
        for answer_id, submission_id in answer_pairs:
            awarded = 1.0 if rng.random() > 0.35 else 0.0
            score_rows.append(
                {
                    "scoring_run_id": runs[submission_id],
                    "answer_id": answer_id,
                    "criterion": "correctness",
                    "awarded": awarded,
                    "max_score": 1.0,
                    "strategy": "exact",
                    "provider": "deterministic",
                }
            )
        for start in range(0, len(score_rows), 5000):
            await session.execute(insert(Score), score_rows[start : start + 5000])
        await session.commit()
        print(f"  scores: {len(score_rows):,}")

        # Final: compute totals + states in one pass (set-based, not row-by-row)
        from sqlalchemy import text as sql_text

        await session.execute(
            sql_text(
                """
                UPDATE submissions s
                SET total_score = agg.total,
                    max_score = agg.max_total,
                    percentage = CASE WHEN agg.max_total > 0
                                      THEN ROUND((agg.total / agg.max_total * 100)::numeric, 1)
                                      ELSE 0 END,
                    state = 'completed',
                    updated_at = now()
                FROM (
                    SELECT a.submission_id,
                           SUM(sc.awarded) AS total,
                           SUM(sc.max_score) AS max_total
                    FROM scores sc
                    JOIN scoring_runs sr ON sr.id = sc.scoring_run_id AND sr.is_current = true
                    JOIN answers a ON a.id = sc.answer_id
                    GROUP BY a.submission_id
                ) agg
                WHERE s.id = agg.submission_id AND s.id = ANY(:ids)
                """
            ),
            {"ids": new_submission_ids},
        )
        await session.commit()

    elapsed = time.perf_counter() - started
    print(f"\nSeeded in {elapsed:.1f}s")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", type=int, default=20000)
    parser.add_argument("--answers", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(seed(args.submissions, args.answers))


if __name__ == "__main__":
    run()
