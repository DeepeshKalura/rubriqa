"""Golden-set helper used by the scoring iteration loop.

The "golden set" is the small set of entries that already have a real
`human_score` populated from a real review. It is the only ground truth we
have right now: we use it both to replay scoring against (so we can measure
how much the prompt would have produced different per-criterion numbers)
and to compute the entry-level kappa / MAE between the LLM and a human.

This module is intentionally read-only: it never writes back to the DB.
Scoring itself still runs through `app.pipeline.scoring.entry.score_entry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Entry, EntryScore, EntryScoreCriterion


@dataclass
class GoldenEntry:
    """One golden entry, with the latest LLM scoring snapshot attached."""

    entry: Entry
    score_id: int | None
    human_score: float
    # criterion name -> awarded score from the latest is_current run
    criteria: dict[str, float] = field(default_factory=dict)
    # total awarded from the latest is_current run (None if no run yet)
    llm_total: float | None = None
    llm_max: float | None = None

    @property
    def theme(self) -> str | None:
        return self.entry.theme

    @property
    def external_ref(self) -> str:
        return self.entry.external_ref


async def golden_set(
    session: AsyncSession,
    *,
    program_id: int | None = None,
    theme: str | None = None,
    limit: int | None = None,
) -> list[GoldenEntry]:
    """Return entries that have a `human_score` populated.

    Optionally filtered by program_id / theme. `limit` is a guardrail; the
    roadmap item calls for "all" but the dev phase keeps the cap explicit so
    the bench cannot accidentally spin on a 20k-entries table.
    """
    stmt = select(Entry).where(Entry.human_score.is_not(None))
    if program_id is not None:
        stmt = stmt.where(Entry.program_id == program_id)
    if theme is not None:
        stmt = stmt.where(Entry.theme == theme)
    stmt = stmt.order_by(Entry.id)
    if limit is not None:
        stmt = stmt.limit(limit)

    rows = (await session.execute(stmt)).scalars().all()
    out: list[GoldenEntry] = []
    for entry in rows:
        # Latest is_current EntryScore (may be None if LLM never scored this
        # entry yet — replay still scores it from scratch).
        latest = (
            await session.execute(
                select(EntryScore)
                .where(EntryScore.entry_id == entry.id, EntryScore.is_current.is_(True))
                .order_by(EntryScore.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        criteria: dict[str, float] = {}
        score_id = None
        llm_total = None
        llm_max = None
        if latest is not None:
            score_id = latest.id
            llm_total = float(latest.total_awarded or 0)
            llm_max = float(latest.total_max or 0)
            crit_rows = (
                await session.execute(
                    select(EntryScoreCriterion).where(EntryScoreCriterion.score_id == latest.id)
                )
            ).scalars().all()
            criteria = {c.criterion: float(c.awarded or 0) for c in crit_rows}
        out.append(
            GoldenEntry(
                entry=entry,
                score_id=score_id,
                human_score=float(entry.human_score or 0),
                criteria=criteria,
                llm_total=llm_total,
                llm_max=llm_max,
            )
        )
    return out
