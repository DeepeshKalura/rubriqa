"""drop legacy exam pipeline (ADR-0001)

Drops every table that served the Q&A exam pipeline (organization -> class ->
batch -> student -> submission -> page -> reading -> repair -> score, plus the
job queue and the per-call LLM log). The only data left is the entry-eval
side: programs, entries, ocr, cleaning, scores, batches, samples, drive files,
rubrics, rubric_criteria, settings, google_credentials, and alembic_version.

Revision ID: 9a1f2c3b4d5e
Revises: 8a3e1c1f0a01
Create Date: 2026-10-08 18:30:00.000000
"""
from typing import Sequence, Union

from alembic import op


revision: str = "9a1f2c3b4d5e"
down_revision: Union[str, None] = "8a3e1c1f0a01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop in FK-safe order. Each op is a no-op if the table is already gone
    # (e.g. on a fresh DB where the initial migration did not include these).
    tables = (
        "repairs",
        "scores",
        "answers",
        "scoring_runs",
        "validation_issues",
        "reading_runs",
        "pages",
        "documents",
        "job_events",
        "jobs",
        "import_batches",
        "submissions",
        "questions",
        "exams",
        "enrollments",
        "batches",
        "students",
        "classes",
        "organizations",
        "llm_calls",
        "google_credentials",
    )
    for table in tables:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def downgrade() -> None:
    # No downgrade path. The legacy pipeline is intentionally not reconstructible.
    raise NotImplementedError(
        "ADR-0001: the exam pipeline was removed in the no-legacy cleanup. "
        "There is no downgrade path; restore from a backup if you need it back."
    )
