"""Entry-level rubric scoring tests (with the fake LLM provider)."""
from __future__ import annotations

import json
import secrets

import pytest

from app.config import Settings
from app.models import Entry, EntryState, Program, ProgramState
from app.pipeline.scoring.entry import score_entry
from app.providers.llm.fake import FakeProvider


def _payload(criteria: list[dict], **rest) -> str:
    return json.dumps({"criteria": criteria, "strengths": [], "gaps": [], "observations": "", **rest})


async def _make_entry(session) -> Entry:
    program = Program(name=f"Score test {secrets.token_hex(3)}", state=ProgramState.ACTIVE)
    session.add(program)
    await session.commit()
    entry = Entry(
        program_id=program.id,
        external_ref=secrets.token_hex(4),
        form_kind="himachal",
        state=EntryState.SCORING,
        full_name="S",
        theme="ai_education",
        language="hin",
        ai_description="AI is software that learns.",
        situation="Class 7 students struggle with math word problems.",
        proposed_solution=(
            "An AI tutor that explains word problems step by step in Hindi, "
            "gives practice questions, and tracks progress over time."
        ),
        status="scoring",
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


@pytest.mark.asyncio
async def test_score_entry_uses_default_rubric_when_none(session) -> None:
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FakeProvider()
    llm.queue(stage="entry_rubric_llm", text=_payload([
        {"name": "relevance", "awarded": 8, "max_score": 10, "rationale": "On theme.", "confidence": 0.9},
        {"name": "feasibility", "awarded": 7, "max_score": 10, "rationale": "Realistic.", "confidence": 0.8},
        {"name": "originality", "awarded": 6, "max_score": 10, "rationale": "Common.", "confidence": 0.7},
        {"name": "ai_understanding", "awarded": 5, "max_score": 10, "rationale": "Basic.", "confidence": 0.8},
        {"name": "communication", "awarded": 7, "max_score": 10, "rationale": "Clear.", "confidence": 0.9},
    ]))
    result = await score_entry(session, entry, llm=llm, settings=settings)
    assert result.score_id is not None
    assert result.total_awarded == 33.0
    assert result.total_max == 50.0
    assert abs(result.percentage - 66.0) < 0.5
    assert len(result.criteria) == 5
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_score_entry_clamps_to_max(session) -> None:
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FakeProvider()
    llm.queue(stage="entry_rubric_llm", text=_payload([
        {"name": "relevance", "awarded": 999, "max_score": 10, "rationale": "R", "confidence": 0.9},
        {"name": "feasibility", "awarded": 50, "max_score": 10, "rationale": "F", "confidence": 0.9},
        {"name": "originality", "awarded": 50, "max_score": 10, "rationale": "O", "confidence": 0.9},
        {"name": "ai_understanding", "awarded": 50, "max_score": 10, "rationale": "A", "confidence": 0.9},
        {"name": "communication", "awarded": 50, "max_score": 10, "rationale": "C", "confidence": 0.9},
    ]))
    result = await score_entry(session, entry, llm=llm, settings=settings)
    # All awards clamped to 10; total = 50
    assert result.total_awarded == 50.0
    assert result.total_max == 50.0


@pytest.mark.asyncio
async def test_score_entry_handles_empty_criteria(session) -> None:
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FakeProvider()
    llm.queue(stage="entry_rubric_llm", text=_payload([]))
    result = await score_entry(session, entry, llm=llm, settings=settings)
    assert result.total_awarded == 0
    assert result.total_max > 0  # default rubric still has max


@pytest.mark.asyncio
async def test_score_entry_records_provider_and_model(session) -> None:
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FakeProvider()
    llm.queue(stage="entry_rubric_llm", text=_payload([
        {"name": "relevance", "awarded": 5, "max_score": 10, "rationale": "r", "confidence": 0.5},
    ]))
    result = await score_entry(session, entry, llm=llm, settings=settings)
    assert result.provider == "fake"
    assert result.strategy == "entry_rubric_llm"


@pytest.mark.asyncio
async def test_score_entry_supersedes_previous_run(session) -> None:
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FakeProvider()
    llm.queue(stage="entry_rubric_llm", text=_payload([
        {"name": "relevance", "awarded": 5, "max_score": 10, "rationale": "r", "confidence": 0.5},
    ]))
    await score_entry(session, entry, llm=llm, settings=settings)
    llm.queue(stage="entry_rubric_llm", text=_payload([
        {"name": "relevance", "awarded": 7, "max_score": 10, "rationale": "r2", "confidence": 0.6},
    ]))
    await score_entry(session, entry, llm=llm, settings=settings)
    from sqlalchemy import select
    from app.models import EntryScore
    scores = (
        await session.execute(
            select(EntryScore).where(EntryScore.entry_id == entry.id).order_by(EntryScore.id)
        )
    ).scalars().all()
    assert len(scores) == 2
    current = [s for s in scores if s.is_current]
    assert len(current) == 1
    assert current[0].total_awarded == 7.0


@pytest.mark.asyncio
async def test_score_entry_handles_llm_error_gracefully(session) -> None:
    from app.providers.llm.fake import FailingProvider
    entry = await _make_entry(session)
    settings = Settings(llm_provider="fake", minimax_text_model="MiniMax-M3")
    llm = FailingProvider()
    result = await score_entry(session, entry, llm=llm, settings=settings)
    assert result.error is not None
    assert result.score_id is None
    assert result.total_awarded == 0