"""Score-bench script (roadmap item 1 / "scoring iteration loop").

Reads `reports/replay_<date>.json` (produced by `scripts/score_replay.py`)
and computes simple agreement / error metrics:

  * Per-criterion MAE per theme — mean |new_score - old_score|.
    Honest limitation: this measures prompt-stability, not human agreement,
    because we only have one human_score per entry (totalled), not per
    criterion. When per-criterion human scores are persisted later, this
    metric can be re-pointed at them.
  * Entry-level MAE between the new LLM total and the human total.
  * Entry-level Cohen's κ between the new LLM total and the human total,
    both binned to a 4-bucket scale (0, 3, 6, 9) so each bucket covers a
    useful grade band. Missing human scores are skipped.

Cohen's κ is implemented locally (no sklearn dependency). It computes
po - pe / (1 - pe), the standard unweighted kappa on paired integer labels.

Run via `make bench-scoring` (Docker).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable

# Allow `python -m scripts.bench_scoring` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORTS_DIR = Path("reports")

# 4 buckets that roughly span a 0-10 per-criterion (or 0-40 total) rubric.
# Bucket boundaries are inclusive on the left.
_BIN_BOUNDARIES = (0, 3, 6, 9)
_BIN_LABELS = ("0-2", "3-5", "6-8", "9-10")


def _bin(value: float) -> int:
    """Bucket index (0..3) for a 0-10 score; clamps at 9."""
    for idx, upper in enumerate(_BIN_BOUNDARIES):
        if value < upper:
            return idx
    return len(_BIN_BOUNDARIES) - 1


def _bin_label(value: float) -> str:
    return _BIN_LABELS[_bin(value)]


def _mae(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    return sum(abs(a - b) for a, b in pairs) / n


def _cohen_kappa(a: list[int], b: list[int]) -> float | None:
    """Standard unweighted Cohen's κ on paired integer labels.

    Returns None if there are fewer than 2 pairs, fewer than 2 categories,
    or both label sets reduce to a single value (in which case κ is
    undefined by convention).
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    n = len(a)
    categories = sorted(set(a) | set(b))
    if len(categories) < 2:
        return None
    # observed agreement
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement
    pe = 0.0
    for cat in categories:
        pe += (a.count(cat) / n) * (b.count(cat) / n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _load_replay(tag: str) -> dict:
    path = REPORTS_DIR / f"replay_{tag}.json"
    if not path.exists():
        raise SystemExit(
            f"replay file not found: {path}\n"
            f"Run `make score-replay` (or scripts.score_replay with --tag {tag}) first."
        )
    return json.loads(path.read_text())


def _summarise(replay: dict) -> dict:
    records = replay.get("records") or []
    if not records:
        return {
            "tag": replay.get("tag"),
            "generated_at": date.today().isoformat(),
            "record_count": 0,
            "criteria": {},
            "themes": {},
            "entry_level": {"n": 0, "mae": None, "kappa": None},
            "notes": ["no records in replay file — nothing to bench"],
        }

    # per-criterion, per-theme: list of (old, new) pairs
    by_crit_theme: dict[tuple[str, str], list[tuple[float | None, float]]] = defaultdict(list)
    # per-criterion globally
    by_crit: dict[str, list[tuple[float | None, float]]] = defaultdict(list)
    # per-entry: list of (entry total new across all criteria, human_total, theme)
    per_entry: dict[int, dict] = {}

    for rec in records:
        crit = rec.get("criterion") or "(none)"
        theme = rec.get("theme") or "(none)"
        new_val = rec.get("new_score")
        old_val = rec.get("old_score")
        if new_val is None:
            continue
        by_crit_theme[(crit, theme)].append((old_val, float(new_val)))
        by_crit[crit].append((old_val, float(new_val)))
        eid = rec.get("entry_id")
        if eid is None:
            continue
        slot = per_entry.setdefault(
            eid,
            {
                "llm_total_new": 0.0,
                "llm_total_old": 0.0,
                "has_old": False,
                "human_total": rec.get("human_total"),
                "theme": theme,
            },
        )
        slot["llm_total_new"] += float(new_val)
        if old_val is not None:
            slot["llm_total_old"] += float(old_val)
            slot["has_old"] = True

    def _table(buckets: dict) -> dict:
        out = {}
        for key, pairs in buckets.items():
            old_vals = [o for o, _ in pairs if o is not None]
            new_vals = [n for _, n in pairs]
            out[key] = {
                "n": len(pairs),
                "mae_old_vs_new": round(_mae(old_vals, new_vals) or 0.0, 4),
            }
        return out

    criteria_table = _table(by_crit)
    themes_table = _table(by_crit_theme)

    # entry-level MAE / kappa (new LLM total vs human total, binned)
    entry_pairs = []
    entry_bins_llm: list[int] = []
    entry_bins_human: list[int] = []
    for eid, slot in per_entry.items():
        if slot.get("human_total") is None:
            continue
        llm_total = slot["llm_total_new"]
        hum_total = float(slot["human_total"])
        # bin on the same 0-9-ish scale; cap human_total to last bucket.
        entry_pairs.append((llm_total, hum_total))
        entry_bins_llm.append(_bin(min(llm_total, 9.0)))
        entry_bins_human.append(_bin(min(hum_total, 9.0)))

    mae_entry = _mae([a for a, _ in entry_pairs], [b for _, b in entry_pairs])
    kappa_entry = _cohen_kappa(entry_bins_llm, entry_bins_human)

    return {
        "tag": replay.get("tag"),
        "generated_at": date.today().isoformat(),
        "record_count": len(records),
        "entry_count": len(per_entry),
        "criteria": dict(sorted(criteria_table.items())),
        "themes": {f"{c}::{t}": v for (c, t), v in sorted(themes_table.items())},
        "entry_level": {
            "n": len(entry_pairs),
            "mae_new_vs_human": round(mae_entry, 4) if mae_entry is not None else None,
            "kappa_new_vs_human_binned": round(kappa_entry, 4) if kappa_entry is not None else None,
            "bin_definitions": list(_BIN_LABELS),
        },
        "notes": [
            "criteria-level MAE measures old-vs-new prompt stability, not human "
            "agreement (we only have entry-level human_score).",
            "kappa is computed on binned (0-2/3-5/6-8/9-10) totals; small samples "
            "will produce unstable estimates.",
        ],
    }


def _output_path(tag: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"scoring_bench_{tag}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bench rubric scoring on the replay JSON.")
    parser.add_argument("--tag", type=str, default=None, help="replay date tag (default: today)")
    parser.add_argument(
        "--from-replay", type=str, default=None, help="explicit replay filename under reports/"
    )
    args = parser.parse_args()
    tag = args.tag or date.today().isoformat()
    if args.from_replay:
        replay = json.loads(Path(args.from_replay).read_text())
        tag = replay.get("tag") or tag
    else:
        replay = _load_replay(tag)
    summary = _summarise(replay)
    out_path = _output_path(tag)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # short readable summary on stdout (Make target stays quiet unless verbose)
    print(json.dumps({"path": str(out_path), "entry_level": summary["entry_level"]}, indent=2))


if __name__ == "__main__":
    main()
