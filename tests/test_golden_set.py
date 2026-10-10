"""Tests for the scoring iteration loop:

  * `app.services.golden_set.golden_set` filters correctly.
  * `scripts.bench_scoring` reads a replay JSON and writes a bench JSON
    that has the documented keys.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest

from app.models import (
    Entry,
    EntryState,
    Program,
    ProgramState,
)
from app.services.golden_set import golden_set


async def _make_program_and_entries(
    session,
    *,
    with_human: int,
    without_human: int,
    theme: str = "ai_education",
) -> tuple[Program, list[Entry], list[Entry]]:
    program = Program(
        name=f"Golden test {secrets.token_hex(3)}",
        state=ProgramState.ACTIVE,
    )
    session.add(program)
    await session.commit()

    gold: list[Entry] = []
    plain: list[Entry] = []
    for i in range(with_human):
        e = Entry(
            program_id=program.id,
            external_ref=f"GOLD-{secrets.token_hex(3)}-{i}",
            form_kind="himachal",
            state=EntryState.COMPLETED,
            full_name=f"Gold {i}",
            theme=theme,
            language="hin",
            situation="hamare gaon mein",
            proposed_solution="ek AI app jo madad kare",
            human_score=35.0 + i,
            review_status="approved",
        )
        session.add(e)
        gold.append(e)
    for i in range(without_human):
        e = Entry(
            program_id=program.id,
            external_ref=f"PLAIN-{secrets.token_hex(3)}-{i}",
            form_kind="himachal",
            state=EntryState.COMPLETED,
            full_name=f"Plain {i}",
            theme=theme,
            language="hin",
            situation="hamare gaon mein",
            proposed_solution="ek AI app jo madad kare",
        )
        session.add(e)
        plain.append(e)
    await session.commit()
    return program, gold, plain


@pytest.mark.asyncio
async def test_golden_set_returns_only_entries_with_human_score(session) -> None:
    program, gold, plain = await _make_program_and_entries(
        session, with_human=2, without_human=2
    )
    rows = await golden_set(session, program_id=program.id)
    ids = {r.entry.id for r in rows}
    expected = {e.id for e in gold}
    assert ids == expected
    # sorted by id, deterministic
    assert [r.entry.id for r in rows] == sorted(r.entry.id for r in rows)


@pytest.mark.asyncio
async def test_golden_set_attach_current_criteria_when_present(session) -> None:
    program, gold, _ = await _make_program_and_entries(session, with_human=1, without_human=0)
    from sqlalchemy import select

    from app.models import EntryScore, EntryScoreCriterion

    entry = gold[0]
    run = EntryScore(
        entry_id=entry.id,
        strategy="entry_rubric_llm",
        provider="fake",
        model="fake-1",
        total_awarded=20.0,
        total_max=50.0,
        percentage=40.0,
        confidence=0.5,
        is_current=True,
    )
    session.add(run)
    await session.commit()

    # re-read the row from the DB so we have a fresh, definitely-persisted id.
    persisted = (
        await session.execute(
            select(EntryScore).where(EntryScore.entry_id == entry.id)
        )
    ).scalar_one()
    score_id = persisted.id

    for name, awarded in [("relevance", 5.0), ("feasibility", 7.0), ("originality", 8.0)]:
        session.add(
            EntryScoreCriterion(score_id=score_id, criterion=name, awarded=awarded, max_score=10.0)
        )
    await session.commit()

    rows = await golden_set(session, program_id=program.id)
    assert len(rows) == 1
    g = rows[0]
    assert g.human_score == pytest.approx(35.0)
    assert g.criteria == {"relevance": 5.0, "feasibility": 7.0, "originality": 8.0}


@pytest.mark.asyncio
async def test_golden_set_filters_by_theme(session) -> None:
    program, gold, _ = await _make_program_and_entries(
        session, with_human=2, without_human=0, theme="smart_agriculture"
    )
    rows = await golden_set(session, program_id=program.id, theme="health_wellbeing")
    assert rows == []
    rows = await golden_set(session, program_id=program.id, theme="smart_agriculture")
    assert {r.entry.id for r in rows} == {e.id for e in gold}


def test_bench_scoring_writes_json_with_expected_keys(tmp_path, monkeypatch) -> None:
    """Drive `bench_scoring._summarise` directly on an in-memory replay and
    assert the produced JSON has the documented top-level keys + the
    entry-level kappa slot."""
    import scripts.bench_scoring as bench_mod

    # bench module reads `reports/` from cwd via Path("reports"); make a
    # sandboxed CWD so we don't pollute the repo.
    cwd = tmp_path / "run"
    cwd.mkdir()
    (cwd / "reports").mkdir()
    monkeypatch.chdir(cwd)

    replay = {
        "tag": "2030-01-01",
        "records": [
            {
                "entry_id": 1,
                "external_ref": "A",
                "theme": "ai_education",
                "criterion": "relevance",
                "old_score": 5.0,
                "new_score": 7.0,
                "human_total": 30.0,
            },
            {
                "entry_id": 1,
                "external_ref": "A",
                "theme": "ai_education",
                "criterion": "feasibility",
                "old_score": 6.0,
                "new_score": 6.0,
                "human_total": 30.0,
            },
            {
                "entry_id": 2,
                "external_ref": "B",
                "theme": "ai_education",
                "criterion": "relevance",
                "old_score": 4.0,
                "new_score": 8.0,
                "human_total": 30.0,
            },
            {
                "entry_id": 2,
                "external_ref": "B",
                "theme": "ai_education",
                "criterion": "feasibility",
                "old_score": 5.0,
                "new_score": 7.0,
                "human_total": 30.0,
            },
            {
                "entry_id": 3,
                "external_ref": "C",
                "theme": "health_wellbeing",
                "criterion": "relevance",
                "old_score": None,
                "new_score": 9.0,
                "human_total": 35.0,
            },
        ],
    }
    (cwd / "reports" / "replay_2030-01-01.json").write_text(json.dumps(replay))

    # drive main() with an explicit --tag
    monkeypatch.setattr("sys.argv", ["bench_scoring", "--tag", "2030-01-01"])
    bench_mod.main()

    out = json.loads((cwd / "reports" / "scoring_bench_2030-01-01.json").read_text())
    # documented top-level keys
    for key in (
        "tag",
        "generated_at",
        "record_count",
        "entry_count",
        "criteria",
        "themes",
        "entry_level",
        "notes",
    ):
        assert key in out, f"missing key: {key}"
    # per-criterion breakdown
    assert "relevance" in out["criteria"]
    assert "feasibility" in out["criteria"]
    crit = out["criteria"]["relevance"]
    assert crit["n"] == 3
    # entry-level MAE + kappa exist (numeric or null)
    assert "mae_new_vs_human" in out["entry_level"]
    assert "kappa_new_vs_human_binned" in out["entry_level"]
    assert out["record_count"] == 5
    assert out["entry_count"] == 3
