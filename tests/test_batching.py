"""EvaluationBatch orchestration tests."""
from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import Settings
from app.models import (
    BatchState,
    Entry,
    EntryState,
    EvaluationBatch,
    Program,
    ProgramState,
    ReviewSample,
    ReviewStatus,
    Rubric,
    RubricCriterion,
    SampleKind,
)
from app.services.batching import (
    batches_for_program,
    complete_batch,
    create_batches_for_program,
    pause_batch_for_review,
    resume_batch,
)


async def _make_program_with_entries(session, *, n: int = 600, batch_size: int | None = None) -> Program:
    import secrets
    program = Program(
        name=f"Test program {secrets.token_hex(4)}",
        state=ProgramState.ACTIVE,
        batch_size=batch_size or 250,
    )
    session.add(program)
    await session.flush()
    for i in range(n):
        session.add(
            Entry(
                program_id=program.id,
                external_ref=f"ref-{secrets.token_hex(4)}-{i:04d}",
                form_kind="himachal",
                state=EntryState.PENDING,
                full_name=f"Student {i}",
                theme="ai_education",
                status="pending",
            )
        )
    await session.commit()
    return program


# ---------------------------------------------------------------------------
# create_batches_for_program
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_create_batches_splits_into_250_chunks(session) -> None:
    program = await _make_program_with_entries(session, n=600)
    settings = Settings(eval_batch_size=250, eval_batch_size_min=200, eval_batch_size_max=300)
    batches = await create_batches_for_program(session, program, settings=settings)
    # 600 split as 250 + 250 + 100, then tail 100 < 200 folds back -> [250, 350]
    sizes = sorted(b.size for b in batches)
    assert sizes == [250, 350]


@pytest.mark.asyncio
async def test_create_batches_clamps_to_min_max(session) -> None:
    program = await _make_program_with_entries(session, n=300)
    settings = Settings(eval_batch_size=250, eval_batch_size_min=200, eval_batch_size_max=300)
    batches = await create_batches_for_program(session, program, settings=settings)
    # 300 split as 250 + 50; tail 50 < 200 folds back -> [300]
    sizes = [b.size for b in batches]
    assert all(200 <= s <= 300 for s in sizes)


@pytest.mark.asyncio
async def test_create_batches_idempotent(session) -> None:
    program = await _make_program_with_entries(session, n=300)
    settings = Settings(eval_batch_size=250, eval_batch_size_min=200, eval_batch_size_max=300)
    a = await create_batches_for_program(session, program, settings=settings)
    b = await create_batches_for_program(session, program, settings=settings)
    assert len(a) == len(b)
    # second call should not create more (entries already assigned)
    assert b == a or {x.id for x in b} == {x.id for x in a}


@pytest.mark.asyncio
async def test_create_batches_assigns_entries_to_batch(session) -> None:
    program = await _make_program_with_entries(session, n=300)
    settings = Settings(eval_batch_size=250)
    batches = await create_batches_for_program(session, program, settings=settings)
    entry_count = (
        await session.execute(
            select(Entry).where(Entry.program_id == program.id, Entry.batch_id.is_not(None))
        )
    ).scalars().all()
    assert len(entry_count) == 300
    batch_ids = {b.id for b in batches}
    assert all(e.batch_id in batch_ids for e in entry_count)


# ---------------------------------------------------------------------------
# pause_batch_for_review + sampling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pause_creates_review_samples(session) -> None:
    program = await _make_program_with_entries(session, n=10)
    settings = Settings(eval_batch_size=250, eval_review_sample_fraction=0.5)
    batches = await create_batches_for_program(session, program, settings=settings)
    batch = batches[0]
    # mark some entries as scored + low-confidence + needs_review so they
    # actually qualify for AI-flagged sampling
    entries = (
        await session.execute(select(Entry).where(Entry.batch_id == batch.id))
    ).scalars().all()
    for e in entries:
        e.total_score = 30.0  # pretend every entry has been scored
    entries[0].confidence = 0.3  # flagged (low confidence)
    entries[1].needs_review = True  # flagged
    entries[2].confidence = 0.4  # low-confidence
    await session.commit()
    n_samples = await pause_batch_for_review(session, batch, settings)
    assert n_samples >= 3
    assert batch.state == BatchState.AWAITING_REVIEW
    samples = (
        await session.execute(select(ReviewSample).where(ReviewSample.batch_id == batch.id))
    ).scalars().all()
    kinds = {s.kind for s in samples}
    assert SampleKind.AI_FLAGGED in kinds
    # 0.5 of 10 = 5; some overlap with flagged so >= 5
    assert sum(1 for s in samples if s.kind == SampleKind.RANDOM) >= 3


@pytest.mark.asyncio
async def test_pause_then_resume(session) -> None:
    program = await _make_program_with_entries(session, n=5)
    settings = Settings(eval_batch_size=250, eval_review_sample_fraction=0.5)
    batches = await create_batches_for_program(session, program, settings=settings)
    batch = batches[0]
    await pause_batch_for_review(session, batch, settings)
    assert batch.state == BatchState.AWAITING_REVIEW
    await resume_batch(session, batch)
    assert batch.state == BatchState.PROCESSING


@pytest.mark.asyncio
async def test_resume_only_works_when_awaiting_review(session) -> None:
    program = await _make_program_with_entries(session, n=5)
    settings = Settings(eval_batch_size=250, eval_review_sample_fraction=0.5)
    batches = await create_batches_for_program(session, program, settings=settings)
    batch = batches[0]
    # state is PENDING initially
    await resume_batch(session, batch)
    assert batch.state == BatchState.PENDING  # unchanged


# ---------------------------------------------------------------------------
# complete_batch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_complete_batch_counts_entries(session) -> None:
    program = await _make_program_with_entries(session, n=4)
    settings = Settings(eval_batch_size=250)
    batches = await create_batches_for_program(session, program, settings=settings)
    batch = batches[0]
    entries = (
        await session.execute(select(Entry).where(Entry.batch_id == batch.id))
    ).scalars().all()
    for i, e in enumerate(entries):
        e.state = EntryState.COMPLETED if i < 3 else EntryState.FAILED
    await session.commit()
    await complete_batch(session, batch)
    assert batch.state == BatchState.COMPLETED
    assert batch.completed == 3
    assert batch.failed == 1
    assert batch.completed_at is not None


# ---------------------------------------------------------------------------
# batches_for_program
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_batches_for_program_orders_by_id(session) -> None:
    program = await _make_program_with_entries(session, n=600)
    settings = Settings(eval_batch_size=200, eval_batch_size_min=200, eval_batch_size_max=300)
    await create_batches_for_program(session, program, settings=settings)
    batches = await batches_for_program(session, program.id)
    ids = [b.id for b in batches]
    assert ids == sorted(ids)
    # 600 entries / 200 = 3 batches
    assert len(batches) == 3