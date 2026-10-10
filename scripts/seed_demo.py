"""Seed a demo entry-evaluation program + a small synthetic dataset.

Creates one Program, one 4-criterion Rubric, and a handful of Entries. The
batches and review samples can then be created from the UI.

Usage:
    python -m scripts.seed_demo
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
    Entry,
    Program,
    ProgramState,
    Rubric,
    RubricCriterion,
)

logger = get_logger("scripts.seed_demo")


RUBRIC_NAME = "AI Innovation Project (demo)"
RUBRIC_CRITERIA = [
    ("Problem clarity", 10.0, "Is the real-life situation clearly described?"),
    ("AI fit", 10.0, "Is the proposed solution a credible use of AI?"),
    ("Feasibility", 10.0, "Could a student team actually build this?"),
    ("Originality", 10.0, "How novel is the idea in their context?"),
]


SAMPLE_ENTRIES = [
    {
        "full_name": "Asha Kumari",
        "student_external_id": "STU-DEMO-01",
        "school_name": "Govt. Senior Secondary School, Shimla",
        "class_name": "Class 9",
        "gender": "F",
        "state": "Himachal Pradesh",
        "district": "Shimla",
        "language": "Hindi",
        "theme": "smart_agriculture",
        "situation": (
            "हमारे गाँव में किसान अक्सर मौसम की सही जानकारी न मिलने से फसल बर्बाद कर देते हैं।"
        ),
        "proposed_ai_solution": (
            "एक ऐसा मोबाइल ऐप जो स्थानीय मौसम, मिट्टी और फसल के आधार पर किसान को हफ्ते भर की सलाह दे।"
        ),
    },
    {
        "full_name": "Ravi Patil",
        "student_external_id": "STU-DEMO-02",
        "school_name": "Ashramshala, Satara",
        "class_name": "Class 8",
        "gender": "M",
        "state": "Maharashtra",
        "district": "Satara",
        "language": "Marathi",
        "theme": "health_wellbeing",
        "situation": (
            "आश्रमशाळेतील विद्यार्थ्यांना ताप, सर्दी झाल्यावर लगेच डॉक्टरांकडे जाता येत नाही."
        ),
        "proposed_ai_solution": (
            "एक AI चॅटबॉट जो लक्षणांवरून प्राथमिक सल्ला देतो आणि गंभीर असेल तर आश्रमशाळेतील शिक्षकांना सूचित करतो."
        ),
    },
    {
        "full_name": "Priya Sharma",
        "student_external_id": "STU-DEMO-03",
        "school_name": "Govt. High School, Dharamshala",
        "class_name": "Class 10",
        "gender": "F",
        "state": "Himachal Pradesh",
        "district": "Kangra",
        "language": "English",
        "theme": "ai_education",
        "situation": "Many students in our village cannot afford private tutors for maths.",
        "proposed_ai_solution": (
            "An AI tutor that explains maths problems in Hindi and English, "
            "with practice questions matched to the student's level."
        ),
    },
]


async def seed() -> None:
    maker = get_sessionmaker()
    async with maker() as session:
        program = (
            await session.execute(
                select(Program).where(Program.name == "Demo Program")
            )
        ).scalar_one_or_none()
        if program is None:
            program = Program(
                name="Demo Program",
                description="Seed program for local UI testing",
                batch_size=250,
                state=ProgramState.ACTIVE,
            )
            session.add(program)
            await session.flush()
            logger.info("seed_program_created", program_id=program.id)

        rubric = (
            await session.execute(
                select(Rubric).where(Rubric.name == RUBRIC_NAME)
            )
        ).scalar_one_or_none()
        if rubric is None:
            rubric = Rubric(name=RUBRIC_NAME, description="Seed rubric for demo program")
            session.add(rubric)
            await session.flush()
            for name, weight, desc in RUBRIC_CRITERIA:
                session.add(
                    RubricCriterion(
                        rubric_id=rubric.id,
                        name=name,
                        weight=weight,
                        description=desc,
                    )
                )
            await session.flush()
            program.rubric_id = rubric.id
            logger.info("seed_rubric_created", rubric_id=rubric.id, criteria=len(RUBRIC_CRITERIA))

        for sample in SAMPLE_ENTRIES:
            existing = (
                await session.execute(
                    select(Entry).where(
                        Entry.program_id == program.id,
                        Entry.external_ref == sample["student_external_id"],
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                continue
            payload = dict(sample)
            payload["external_ref"] = payload.pop("student_external_id")
            payload["state_name"] = payload.pop("state")
            payload["proposed_solution"] = payload.pop("proposed_ai_solution")
            session.add(Entry(program_id=program.id, **payload))
        await session.commit()
        logger.info("seed_done", program_id=program.id, entries=len(SAMPLE_ENTRIES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a demo program + entries")
    parser.add_argument("--docs", type=Path, default=None, help="unused; kept for CLI compat")
    args = parser.parse_args()
    settings = get_settings()
    init_engine(settings)
    configure_logging(settings.log_level, settings.log_json)
    asyncio.run(seed())


if __name__ == "__main__":
    main()
