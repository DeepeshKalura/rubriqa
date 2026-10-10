"""Promote an entry from a resolved review sample.

Roadmap item 3. Pure async function:

  * reads the latest ReviewSample for the entry,
  * sets the entry's verdict via `entry.review_status` (the existing
    `pending|approved|corrected|rejected` string column) — NOT `Entry.state`,
    which is a pipeline-progress enum and has no `APPROVED`/`REJECTED` values,
  * copies `sample.final_score` onto `entry.human_score`,
  * logs the transition via structlog (`entry_promoted` / `entry_rejected` /
    `entry_corrected`),
  * is idempotent: a second call returns `(ok=True, reason="already_promoted")`
    and writes nothing.

Returns a small `PromoteResult` so the route layer can map reasons to HTTP
status codes without touching SQLAlchemy state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.logging import get_logger
from app.models import Entry, ReviewSample, ReviewStatus

logger = get_logger("entry_eval.promotion")


@dataclass
class PromoteResult:
    ok: bool
    reason: Optional[str]
    new_state: Optional[str]
    promoted_score: Optional[float]


_VERDICT_FOR_STATUS = {
    ReviewStatus.APPROVED: "approved",
    ReviewStatus.CORRECTED: "corrected",
    ReviewStatus.REJECTED: "rejected",
}


async def promote_entry(session: AsyncSession, entry_id: int) -> PromoteResult:
    entry = await session.get(Entry, entry_id)
    if entry is None:
        return PromoteResult(ok=False, reason="not_found", new_state=None, promoted_score=None)

    sample = (
        await session.execute(
            select(ReviewSample)
            .where(ReviewSample.entry_id == entry_id)
            .order_by(ReviewSample.created_at.desc(), ReviewSample.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if sample is None:
        return PromoteResult(ok=False, reason="no_review_sample", new_state=None, promoted_score=None)

    if sample.status is ReviewStatus.PENDING:
        return PromoteResult(
            ok=False, reason="review_pending", new_state=entry.review_status, promoted_score=entry.human_score,
        )

    target_verdict = _VERDICT_FOR_STATUS[sample.status]
    target_score = sample.final_score

    if entry.review_status == target_verdict and entry.human_score == target_score:
        return PromoteResult(
            ok=True, reason="already_promoted", new_state=entry.review_status, promoted_score=entry.human_score,
        )

    from_state = entry.review_status
    entry.review_status = target_verdict
    entry.human_score = target_score
    if sample.status is ReviewStatus.CORRECTED and sample.notes:
        entry.review_notes = sample.notes
        entry.human_score_notes = sample.notes
    await session.commit()

    event = {
        ReviewStatus.APPROVED: "entry_promoted",
        ReviewStatus.CORRECTED: "entry_corrected",
        ReviewStatus.REJECTED: "entry_rejected",
    }[sample.status]
    logger.info(
        event,
        entry_id=entry.id,
        review_sample_id=sample.id,
        from_state=from_state,
        to_state=target_verdict,
        promoted_score=target_score,
    )

    return PromoteResult(
        ok=True, reason=None, new_state=entry.review_status, promoted_score=entry.human_score,
    )