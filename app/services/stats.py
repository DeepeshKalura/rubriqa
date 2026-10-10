"""Aggregations for the dashboard.

Counts programs, entries, batches, review samples, and surfaces the most
recently updated entries. Kept as plain SQL so the dashboard is fast at the
configured batch size (200-300 entries per program).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    BatchState,
    Entry,
    EntryScore,
    EvaluationBatch,
    Program,
    ReviewSample,
    ReviewStatus,
)


async def dashboard_stats(session: AsyncSession) -> dict[str, Any]:
    programs = (await session.execute(select(func.count(Program.id)))).scalar_one()
    entries = (await session.execute(select(func.count(Entry.id)))).scalar_one()
    scored = (
        await session.execute(
            select(func.count(EntryScore.id)).where(EntryScore.error.is_(None))
        )
    ).scalar_one()
    batches = (await session.execute(select(func.count(EvaluationBatch.id)))).scalar_one()
    awaiting = (
        await session.execute(
            select(func.count(EvaluationBatch.id)).where(
                EvaluationBatch.state == BatchState.AWAITING_REVIEW
            )
        )
    ).scalar_one()
    samples = (await session.execute(select(func.count(ReviewSample.id)))).scalar_one()
    samples_done = (
        await session.execute(
            select(func.count(ReviewSample.id)).where(
                ReviewSample.status.in_(
                    [ReviewStatus.APPROVED, ReviewStatus.CORRECTED]
                )
            )
        )
    ).scalar_one()

    state_rows = (
        await session.execute(select(Entry.state, func.count(Entry.id)).group_by(Entry.state))
    ).all()
    entry_states = {row[0].value if hasattr(row[0], "value") else str(row[0]): row[1] for row in state_rows}

    return {
        "programs": programs,
        "entries": entries,
        "scored": scored,
        "batches": batches,
        "batches_awaiting": awaiting,
        "review_samples": samples,
        "review_samples_done": samples_done,
        "entry_states": entry_states,
    }


async def recent_entries(session: AsyncSession, limit: int = 8) -> list[dict[str, Any]]:
    rows = (
        (
            await session.execute(
                select(Entry, EntryScore.percentage)
                .outerjoin(EntryScore, EntryScore.entry_id == Entry.id)
                .order_by(Entry.id.desc())
                .limit(limit)
            )
        )
        .all()
    )
    return [
        {
            "id": entry.id,
            "full_name": entry.full_name,
            "external_ref": entry.external_ref,
            "theme": entry.theme,
            "state": entry.state.value if hasattr(entry.state, "value") else str(entry.state),
            "percentage": percentage,
        }
        for entry, percentage in rows
    ]


def _entry_state_counts(states: list[str]) -> dict[str, int]:
    return dict(Counter(states))
