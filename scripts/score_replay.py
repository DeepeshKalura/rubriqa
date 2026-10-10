"""Score-replay script (roadmap item 1 / "scoring iteration loop").

Reads the golden set (entries with `human_score` set) and re-runs the rubric
LLM scoring path on every one of them. Captures the per-criterion scores
BEFORE the replay (the "old" values from the latest is_current EntryScore)
and AFTER (the new criteria written by `score_entry`).

Writes one JSON file at `reports/replay_<YYYY-MM-DD>.json` with the per-row
records plus a small summary. Re-running the script on the same date
overwrites that day's file.

Honest limitations we do NOT yet hit:
- The replay always uses the existing single-prompt (no N-shot examples
  drawn from past human decisions per theme). That's the next sub-step of
  this roadmap item; the bench will still be meaningful without it.
- The replay does not persist an `is_replay` flag — new EntryScore rows
  simply supersede the prior `is_current=True` ones. Old rows remain in
  the table for audit.

Run via `make score-replay` (Docker).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

# Allow `python -m scripts.score_replay` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_sessionmaker, init_engine  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402
from app.models import EntryScore, EntryScoreCriterion  # noqa: E402
from app.pipeline.scoring.entry import score_entry  # noqa: E402
from app.providers.llm import build_provider  # noqa: E402
from app.services.app_settings import get_effective_settings  # noqa: E402
from app.services.golden_set import golden_set  # noqa: E402

logger = get_logger("scripts.score_replay")

REPORTS_DIR = Path("reports")


def _today_tag() -> str:
    return date.today().isoformat()


def _output_path(tag: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"replay_{tag}.json"


async def _capture_old_scores(session, entry_id: int) -> dict[str, float]:
    """Snapshot the current is_current EntryScore's per-criterion awards
    BEFORE `score_entry` is called. Returns criterion -> awarded (or {} if
    the entry was never scored before)."""
    latest = (
        await session.execute(
            select(EntryScore)
            .where(EntryScore.entry_id == entry_id, EntryScore.is_current.is_(True))
            .order_by(EntryScore.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return {}
    rows = (
        await session.execute(
            select(EntryScoreCriterion).where(EntryScoreCriterion.score_id == latest.id)
        )
    ).scalars().all()
    return {c.criterion: float(c.awarded or 0) for c in rows}


async def _capture_new_scores(session, entry_id: int) -> dict[str, float]:
    """Read back the just-written per-criterion awards (the freshly
    is_current EntryScore)."""
    latest = (
        await session.execute(
            select(EntryScore)
            .where(EntryScore.entry_id == entry_id, EntryScore.is_current.is_(True))
            .order_by(EntryScore.id.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if latest is None:
        return {}
    rows = (
        await session.execute(
            select(EntryScoreCriterion).where(EntryScoreCriterion.score_id == latest.id)
        )
    ).scalars().all()
    return {c.criterion: float(c.awarded or 0) for c in rows}


async def replay(*, program_id: int | None, theme: str | None, tag: str) -> dict:
    settings = get_settings()
    init_engine(settings)
    configure_logging(settings.log_level, settings.log_json)
    maker = get_sessionmaker()

    rows: list[dict] = []
    summary = {
        "entries_replayed": 0,
        "entries_with_old_score": 0,
        "criterion_records": 0,
        "by_theme": {},
    }

    async with maker() as session:
        eff = await get_effective_settings(session)
        llm = build_provider(eff)
        gold = await golden_set(session, program_id=program_id, theme=theme)
        logger.info("score_replay_start", golden=len(gold), program_id=program_id, theme=theme)

        for g in gold:
            old_scores = await _capture_old_scores(session, g.entry.id)
            if old_scores:
                summary["entries_with_old_score"] += 1
            # capture human score + theme BEFORE replaying (replay mutates
            # the entry's total_score / percentage / confidence).
            human_total = float(g.entry.human_score or 0)
            theme_name = g.entry.theme
            # snapshot new EntryScore will be created by score_entry()
            result = await score_entry(
                session, g.entry, llm=llm, settings=eff, rubric_id=g.entry.program.rubric_id
            )
            new_scores = await _capture_new_scores(session, g.entry.id)

            for criterion, new_val in new_scores.items():
                old_val = old_scores.get(criterion)
                rows.append(
                    {
                        "entry_id": g.entry.id,
                        "external_ref": g.entry.external_ref,
                        "theme": theme_name,
                        "criterion": criterion,
                        "old_score": old_val,
                        "new_score": float(new_val),
                        "human_total": human_total,
                        "error": result.error,
                    }
                )
            summary["criterion_records"] += len(new_scores)
            summary["by_theme"].setdefault(theme_name or "(none)", 0)
            summary["by_theme"][theme_name or "(none)"] += 1
            summary["entries_replayed"] += 1
            # commit so the next iteration's "is_current" query is honest
            await session.commit()

    out = {
        "tag": tag,
        "generated_at": date.today().isoformat(),
        "program_id": program_id,
        "theme": theme,
        "llm_provider": settings.llm_provider,
        "model": settings.minimax_text_model,
        "records": rows,
        "summary": summary,
    }
    path = _output_path(tag)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(
        "score_replay_done",
        path=str(path),
        entries=summary["entries_replayed"],
        records=summary["criterion_records"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay rubric scoring on the golden set.")
    parser.add_argument("--program-id", type=int, default=None, help="filter to one program")
    parser.add_argument("--theme", type=str, default=None, help="filter to one theme slug")
    parser.add_argument("--tag", type=str, default=None, help="override the date tag (default: today)")
    args = parser.parse_args()
    tag = args.tag or _today_tag()
    asyncio.run(replay(program_id=args.program_id, theme=args.theme, tag=tag))


if __name__ == "__main__":
    main()
