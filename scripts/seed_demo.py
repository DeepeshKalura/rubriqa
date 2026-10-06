"""Seed demo data: org, class, batches, students, exam, rubric, and import.

Usage:
    python -m scripts.seed_demo [--docs PATH] [--queue]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db import get_sessionmaker, init_engine
from app.logging import configure_logging, get_logger
from app.models import (
    Batch,
    Enrollment,
    Exam,
    Klass,
    Organization,
    Question,
    Rubric,
    RubricCriterion,
    Student,
)
from app.services import jobs as jobs_service
from app.services.importer import enqueue_submissions, register_documents, scan_folder
from app.services.storage import Storage

from scripts.gen_synthetic import EXAM, RUBRIC, STUDENTS

logger = get_logger("scripts.seed")


async def seed(docs: Path | None, queue: bool) -> None:
    maker = get_sessionmaker()
    async with maker() as session:
        org = (
            await session.execute(
                select(Organization).where(Organization.name == "Demo Public School")
            )
        ).scalar_one_or_none()
        if org is None:
            org = Organization(name="Demo Public School", code="DPS")
            session.add(org)
            await session.flush()

        klass = (
            await session.execute(
                select(Klass).where(Klass.organization_id == org.id, Klass.name == "Class 8")
            )
        ).scalar_one_or_none()
        if klass is None:
            klass = Klass(organization_id=org.id, name="Class 8", section="A")
            session.add(klass)
            await session.flush()

        batches: dict[str, Batch] = {}
        for student in STUDENTS:
            if student["batch"] not in batches:
                batch = (
                    await session.execute(
                        select(Batch).where(
                            Batch.class_id == klass.id, Batch.name == student["batch"]
                        )
                    )
                ).scalar_one_or_none()
                if batch is None:
                    batch = Batch(class_id=klass.id, name=student["batch"], year=2026)
                    session.add(batch)
                    await session.flush()
                batches[student["batch"]] = batch
            db_student = (
                await session.execute(
                    select(Student).where(
                        Student.organization_id == org.id,
                        Student.external_id == student["external_id"],
                    )
                )
            ).scalar_one_or_none()
            if db_student is None:
                db_student = Student(
                    organization_id=org.id,
                    external_id=student["external_id"],
                    name=student["name"],
                )
                session.add(db_student)
                await session.flush()
            enrollment = (
                await session.execute(
                    select(Enrollment).where(
                        Enrollment.batch_id == batches[student["batch"]].id,
                        Enrollment.student_id == db_student.id,
                    )
                )
            ).scalar_one_or_none()
            if enrollment is None:
                session.add(
                    Enrollment(
                        batch_id=batches[student["batch"]].id,
                        student_id=db_student.id,
                        roll_no_in_batch=student["external_id"],
                    )
                )
        await session.commit()

        exam = (
            await session.execute(
                select(Exam).where(Exam.class_id == klass.id, Exam.name == EXAM["name"])
            )
        ).scalar_one_or_none()
        if exam is None:
            exam = Exam(
                class_id=klass.id,
                name=EXAM["name"],
                exam_date=EXAM["exam_date"],
                subject=EXAM["subject"],
                language=EXAM["language"],
            )
            session.add(exam)
            await session.flush()
            rubric = (
                await session.execute(select(Rubric).where(Rubric.name == RUBRIC["name"]))
            ).scalar_one_or_none()
            if rubric is None:
                rubric = Rubric(name=RUBRIC["name"], description=RUBRIC["description"])
                session.add(rubric)
                await session.flush()
                for criterion in RUBRIC["criteria"]:
                    session.add(
                        RubricCriterion(
                            rubric_id=rubric.id,
                            name=criterion["name"],
                            weight=criterion["weight"],
                            description=criterion.get("description"),
                            knockout=bool(criterion.get("knockout")),
                            knockout_below_fraction=float(
                                criterion.get("knockout_below_fraction") or 0
                            ),
                        )
                    )
            for question in EXAM["questions"]:
                session.add(
                    Question(
                        exam_id=exam.id,
                        q_no=question["q_no"],
                        text=question["text"],
                        text_hindi=question.get("text_hindi"),
                        max_score=question["max_score"],
                        strategy=question.get("strategy"),
                        answer_key=question.get("answer_key"),
                        rubric_id=rubric.id,
                    )
                )
            await session.commit()

        print(f"Seeded org={org.id} class={klass.id} exam={exam.id}")

        if docs:
            plan = scan_folder(Storage().resolve(str(docs)))
            batch = await register_documents(session, exam_id=exam.id, plan=plan)
            print(
                f"Import scan: discovered={plan.discovered} new={batch.new_documents} "
                f"duplicates={batch.already_imported} unmatched={plan.unmatched} "
                f"submissions_created={batch.submissions_created}"
            )
            if queue:
                from app.models import JobType

                await jobs_service.enqueue(
                    session,
                    JobType.INGEST_IMPORT,
                    params={"root_path": str(docs), "exam_id": exam.id},
                    priority=10,
                )
                await session.commit()
                print("Queued import job + submission processing jobs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=None)
    parser.add_argument("--queue", action="store_true", default=True)
    parser.add_argument("--no-queue", dest="queue", action="store_false")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine(settings)
    asyncio.run(seed(args.docs, args.queue))


if __name__ == "__main__":
    main()
