"""Batch orchestration for entry evaluation.

Each Program is split into `EvaluationBatch` rows of configurable size
(200-300 by default, clamped via `EVAL_BATCH_SIZE_MIN/MAX`). Within a batch:
  * the worker reads, cleans, and scores each Entry
  * samples a fraction of the batch for human review (random + AI-flagged +
    low-confidence)
  * pauses for human review before continuing to the next one

The pause is a server-side flag (`BatchState.AWAITING_REVIEW`); the worker
respects it and does not start the next batch until the user signals continue.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    BatchState,
    Entry,
    EntryScore,
    EntryState,
    EvaluationBatch,
    Program,
    ReviewSample,
    ReviewStatus,
    SampleKind,
)


async def _clamp_batch_size(settings: Settings, requested: int | None) -> int:
    size = requested or settings.eval_batch_size
    return max(settings.eval_batch_size_min, min(settings.eval_batch_size_max, size))


async def create_batches_for_program(
    session: AsyncSession,
    program: Program,
    *,
    settings: Settings,
    batch_size: int | None = None,
) -> list[EvaluationBatch]:
    """Split all PENDING entries in the program into batches.

    Idempotent: existing pending/active batches for the program are left alone;
    only entries not yet in a batch will be assigned.
    """
    size = await _clamp_batch_size(settings, batch_size)
    open_states = (BatchState.PENDING, BatchState.PROCESSING, BatchState.AWAITING_REVIEW)
    existing = (
        await session.execute(
            select(EvaluationBatch).where(
                EvaluationBatch.program_id == program.id,
                EvaluationBatch.state.in_(open_states),
            )
        )
    ).scalars().all()

    assigned_ids: set[int] = set()
    for batch in existing:
        assigned_ids.add(batch.id)

    pending_entries = (
        await session.execute(
            select(Entry)
            .where(
                Entry.program_id == program.id,
                Entry.batch_id.is_(None),
                Entry.state.in_((EntryState.PENDING, EntryState.FAILED)),
            )
            .order_by(Entry.id)
        )
    ).scalars().all()

    if not pending_entries:
        return list(existing)

    chunks: list[list[Entry]] = []
    current: list[Entry] = []
    for entry in pending_entries:
        current.append(entry)
        if len(current) >= size:
            chunks.append(current)
            current = []
    if current:
        # If the tail is smaller than the minimum batch size, fold it back into
        # the previous batch so we never ship a batch below the spec range.
        if chunks and len(current) < settings.eval_batch_size_min:
            chunks[-1].extend(current)
        else:
            chunks.append(current)
    next_idx = len(existing) + 1
    new_batches: list[EvaluationBatch] = []
    for chunk in chunks:
        batch = EvaluationBatch(
            program_id=program.id,
            name=f"B{next_idx:03d}",
            state=BatchState.PENDING,
            target_size=size,
            size=len(chunk),
            completed=0,
            failed=0,
            flagged=0,
        )
        session.add(batch)
        await session.flush()
        next_idx += 1
        for entry in chunk:
            entry.batch_id = batch.id
        new_batches.append(batch)
    await session.commit()
    return list(existing) + new_batches


async def start_batch(session: AsyncSession, batch: EvaluationBatch) -> None:
    if batch.state is not BatchState.PENDING:
        return
    batch.state = BatchState.PROCESSING
    batch.started_at = datetime.now(timezone.utc)
    await session.commit()


async def pause_batch_for_review(
    session: AsyncSession, batch: EvaluationBatch, settings: Settings
) -> int:
    """Mark batch as awaiting review and create review rows for the sample
    (random + AI-flagged + low-confidence).

    Returns the count of review samples created. Accepts both PENDING and
    PROCESSING — devs frequently want to gate a batch before work begins.
    """
    if batch.state not in (BatchState.PENDING, BatchState.PROCESSING):
        return 0
    entries = (
        await session.execute(
            select(Entry)
            .where(Entry.batch_id == batch.id)
            .order_by(Entry.id)
        )
    ).scalars().all()
    # score-based flagging. We only treat them as flagged if the worker has
    # already produced a score; until then nothing is "AI-flagged" (the
    # system has not made a judgement yet).
    scored = [e for e in entries if e.total_score is not None]
    ai_flagged: set[int] = {
        e.id for e in scored
        if (e.confidence is not None and e.confidence < 0.55)
        or e.needs_review
    }
    low_conf: set[int] = {
        e.id for e in scored
        if e.confidence is not None and e.confidence < 0.45
    }
    # random sample
    import random

    rng_obj = random.Random(int(batch.id) * 7919)
    pool = [e for e in entries if e.id not in ai_flagged and e.id not in low_conf]
    rng_obj.shuffle(pool)
    sample_count = max(1, int(len(entries) * settings.eval_review_sample_fraction))
    random_picks = pool[:sample_count]
    random_ids = {e.id for e in random_picks}

    seen_ids: set[int] = set()
    created = 0
    sample_groups: list[tuple[SampleKind, set[int]]] = [
        (SampleKind.AI_FLAGGED, ai_flagged),
        (SampleKind.LOW_CONFIDENCE, low_conf - ai_flagged),
        (SampleKind.RANDOM, random_ids),
    ]
    for kind, ids in sample_groups:
        for entry_id in ids:
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            session.add(
                ReviewSample(
                    batch_id=batch.id,
                    entry_id=entry_id,
                    kind=kind,
                    status=ReviewStatus.PENDING,
                )
            )
            created += 1
    batch.state = BatchState.AWAITING_REVIEW
    batch.paused_at = datetime.now(timezone.utc)
    batch.flagged = len(ai_flagged) + len(low_conf)
    await session.commit()
    return created


async def resume_batch(session: AsyncSession, batch: EvaluationBatch) -> None:
    if batch.state is not BatchState.AWAITING_REVIEW:
        return
    batch.state = BatchState.PROCESSING
    batch.paused_at = None
    await session.commit()


async def complete_batch(session: AsyncSession, batch: EvaluationBatch) -> None:
    completed = (
        await session.execute(
            select(func.count())
            .select_from(Entry)
            .where(Entry.batch_id == batch.id, Entry.state == EntryState.COMPLETED)
        )
    ).scalar_one()
    failed = (
        await session.execute(
            select(func.count())
            .select_from(Entry)
            .where(Entry.batch_id == batch.id, Entry.state == EntryState.FAILED)
        )
    ).scalar_one()
    batch.completed = completed or 0
    batch.failed = failed or 0
    batch.state = BatchState.COMPLETED
    batch.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def batches_for_program(session: AsyncSession, program_id: int) -> list[EvaluationBatch]:
    return (
        await session.execute(
            select(EvaluationBatch)
            .where(EvaluationBatch.program_id == program_id)
            .order_by(EvaluationBatch.id)
        )
    ).scalars().all()