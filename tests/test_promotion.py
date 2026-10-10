"""Promote route + service tests (roadmap item 3).

Covers the four contracts:
  * POST `/entries/{id}/promote` with an approved sample -> 200, entry verdict
    set to `approved`, `human_score` copied from `sample.final_score`.
  * Second POST -> 200, `reason="already_promoted"`, no extra commit.
  * Pending sample -> 409, `reason="review_pending"`.
  * No sample at all -> 409, `reason="no_review_sample"`.

The service writes `entry.review_status` (the existing string column with
values `pending|approved|corrected|rejected`) -- `Entry.state` is the pipeline-
progress enum and intentionally not touched here.
"""
from __future__ import annotations

import secrets

import pytest
from httpx import AsyncClient

from app.db import get_sessionmaker
from app.models import (
    Entry,
    EntryState,
    EvaluationBatch,
    Program,
    ProgramState,
    ReviewSample,
    ReviewStatus,
    SampleKind,
)


async def _seed_one_approved_sample(review_status: ReviewStatus, final_score: float | None) -> tuple[int, int]:
    """Create program + entry + batch + sample. Returns (program_id, entry_id)."""
    sm = get_sessionmaker()
    async with sm() as s:
        program = Program(
            name=f"Promote test {secrets.token_hex(4)}",
            state=ProgramState.ACTIVE,
            batch_size=250,
        )
        s.add(program)
        await s.flush()
        entry = Entry(
            program_id=program.id,
            external_ref=f"ref-{secrets.token_hex(4)}",
            form_kind="himachal",
            state=EntryState.PENDING,
            full_name="Promote Target",
            theme="ai_education",
            status="pending",
        )
        s.add(entry)
        await s.flush()
        batch = EvaluationBatch(
            program_id=program.id,
            name="b1",
            state=EntryState.PENDING,
            target_size=250,
        )
        s.add(batch)
        await s.flush()
        sample = ReviewSample(
            batch_id=batch.id,
            entry_id=entry.id,
            kind=SampleKind.RANDOM,
            status=review_status,
            reviewer="tester",
            final_score=final_score,
        )
        s.add(sample)
        await s.commit()
        return program.id, entry.id


@pytest.mark.asyncio
async def test_promote_approved_sample_sets_verdict_and_score(client: AsyncClient) -> None:
    _, entry_id = await _seed_one_approved_sample(ReviewStatus.APPROVED, 88.5)

    r = await client.post(f"/entries/{entry_id}/promote", follow_redirects=False)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["reason"] is None
    assert body["new_state"] == "approved"
    assert body["promoted_score"] == 88.5

    sm = get_sessionmaker()
    async with sm() as s:
        e = await s.get(Entry, entry_id)
        assert e.review_status == "approved"
        assert e.human_score == 88.5


@pytest.mark.asyncio
async def test_promote_is_idempotent_no_op(client: AsyncClient) -> None:
    _, entry_id = await _seed_one_approved_sample(ReviewStatus.APPROVED, 77.0)

    r1 = await client.post(f"/entries/{entry_id}/promote", follow_redirects=False)
    assert r1.status_code == 200
    assert r1.json()["new_state"] == "approved"

    r2 = await client.post(f"/entries/{entry_id}/promote", follow_redirects=False)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["ok"] is True
    assert body2["reason"] == "already_promoted"
    assert body2["new_state"] == "approved"
    assert body2["promoted_score"] == 77.0


@pytest.mark.asyncio
async def test_promote_pending_sample_returns_409(client: AsyncClient) -> None:
    _, entry_id = await _seed_one_approved_sample(ReviewStatus.PENDING, None)

    r = await client.post(f"/entries/{entry_id}/promote", follow_redirects=False)
    assert r.status_code == 409
    assert "review_pending" in r.text

    sm = get_sessionmaker()
    async with sm() as s:
        e = await s.get(Entry, entry_id)
        assert e.review_status == "pending"
        assert e.human_score is None


@pytest.mark.asyncio
async def test_promote_entry_with_no_sample_returns_409(client: AsyncClient) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        program = Program(
            name=f"NoSample {secrets.token_hex(4)}",
            state=ProgramState.ACTIVE,
            batch_size=250,
        )
        s.add(program)
        await s.flush()
        entry = Entry(
            program_id=program.id,
            external_ref=f"ref-{secrets.token_hex(4)}",
            form_kind="himachal",
            state=EntryState.PENDING,
            full_name="No Sample Target",
            theme="ai_education",
            status="pending",
        )
        s.add(entry)
        await s.commit()
        entry_id = entry.id

    r = await client.post(f"/entries/{entry_id}/promote", follow_redirects=False)
    assert r.status_code == 409
    assert "no_review_sample" in r.text


@pytest.mark.asyncio
async def test_promote_missing_entry_returns_404(client: AsyncClient) -> None:
    r = await client.post("/entries/9999999/promote", follow_redirects=False)
    assert r.status_code == 404