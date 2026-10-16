"""Replay the fusion gate on measured data, with and without Tesseract.

Issue #9 asks whether Tesseract can be retired. Before swapping the engine we
have to know what the fusion gate (`tess_len / vision_len > threshold`) is
actually *buying* us, because the gate is trained on Tesseract's failure
signature. If Tesseract goes, the gate must be re-validated or replaced.

The gate's two inputs are recovered from the benchmark runs, which contain the
two engine transcripts for every page:

  vision_len = len(path A transcript)   # the vision engine's reading
  tess_len   = len(path B transcript)   # the local OCR engine's reading

Ground truth for "did fusion actually help?" is the measured CER of path C
(fusion) versus path A (strict vision) on the same page with the same model.

This script is read-only analysis. It writes no pipeline state and calls no LLM.

Usage (Docker):
    docker compose exec web python -m scripts.replay_fusion_gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SRC = Path("reports/bench_models.json")
DEFAULT_OUT = Path("reports/fusion_gate_replay.json")

# Thresholds to compare. 1.5 is the shipped default
# (app/config.py: fusion_max_tess_to_vision_ratio).
THRESHOLDS = [1.0, 1.25, 1.5, 2.0, 3.0, float("inf")]


def collect_pages(payload: dict, model_key: str) -> list[dict]:
    """One record per page: both engine lengths plus the CER verdict."""
    by_page: dict[str, dict[str, Any]] = {}
    for run in payload["runs"]:
        if run.get("pricing_key") != model_key:
            continue
        page = run["page"]
        rec = by_page.setdefault(page, {"page": page})
        path = run["path"]
        if path == "vision":
            rec["vision_len"] = len(run.get("ocr_text") or "")
            rec["vision_cer"] = run.get("cer")
            rec["vision_ok"] = run.get("ok")
        elif path == "tesseract_llm":
            rec["tess_len"] = len(run.get("ocr_text") or "")
            rec["tess_cer"] = run.get("cer")
            rec["tess_ok"] = run.get("ok")
        elif path == "vision_ocr_fusion":
            rec["fusion_cer"] = run.get("cer")
            rec["fusion_ok"] = run.get("ok")

    rows = []
    for page, rec in by_page.items():
        if "vision_len" not in rec or "tess_len" not in rec:
            continue
        v, t = rec["vision_len"], rec["tess_len"]
        rec["ratio"] = round(t / v, 2) if v else float("inf")
        # did fusion beat strict vision on this page?
        if rec.get("vision_cer") is not None and rec.get("fusion_cer") is not None:
            rec["fusion_delta_cer"] = round(rec["fusion_cer"] - rec["vision_cer"], 1)
            rec["fusion_helped"] = rec["fusion_delta_cer"] < 0
        rows.append(rec)
    return sorted(rows, key=lambda r: r["page"])


def gate_fires(rec: dict, threshold: float) -> bool:
    """Mirror of app/services/entries.py fusion gate (ignoring empty-string
    handling, which is reported separately)."""
    return rec["vision_len"] > 0 and rec["tess_len"] > threshold * rec["vision_len"]


def evaluate(rows: list[dict], threshold: float) -> dict:
    fired = [r for r in rows if gate_fires(r, threshold)]
    helped = [r for r in rows if r.get("fusion_helped") is True]
    hurt = [r for r in rows if r.get("fusion_helped") is False]

    # The gate is *supposed* to catch pages where fusion folds in tesseract
    # noise. So a correct gate fires on pages where fusion hurt, and stays
    # silent where fusion helped.
    fired_and_hurt = [r for r in fired if r.get("fusion_helped") is False]
    fired_and_helped = [r for r in fired if r.get("fusion_helped") is True]
    silent_and_hurt = [r for r in rows if r not in fired and r.get("fusion_helped") is False]
    silent_and_helped = [r for r in rows if r not in fired and r.get("fusion_helped") is True]

    saved_cer = sum(abs(r["fusion_delta_cer"]) for r in fired_and_hurt)
    lost_cer = sum(abs(r["fusion_delta_cer"]) for r in fired_and_helped)

    tp, fp = len(fired_and_hurt), len(fired_and_helped)
    fn, tn = len(silent_and_hurt), len(silent_and_helped)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    return {
        "threshold": None if threshold == float("inf") else threshold,
        "label": "never gate" if threshold == float("inf") else threshold,
        "fired_pages": [r["page"] for r in fired],
        "fired_count": len(fired),
        "true_positive_fired_because_fusion_hurt": tp,
        "false_positive_fired_but_fusion_would_have_helped": fp,
        "false_negative_silent_but_fusion_hurt": fn,
        "true_negative_silent_and_fusion_helped": tn,
        "precision": round(precision, 2) if precision is not None else None,
        "recall": round(recall, 2) if recall is not None else None,
        "cer_avoided_by_gating": round(saved_cer, 1),
        "cer_forfeited_by_gating": round(lost_cer, 1),
        "net_cer_benefit": round(saved_cer - lost_cer, 1),
    }


def no_tesseract_scenario(rows: list[dict]) -> dict:
    """What happens to the gate's *input* if Tesseract is replaced.

    A replacement engine that is better than Tesseract produces text of a
    different length. The gate assumes 'much longer than vision' == 'noise'.
    That assumption only holds if the replacement is as noisy as Tesseract.
    This quantifies how many currently-gated pages would even remain
    detectable, by asking: on the pages the gate fires, is tesseract long
    because it is *wrong* (CER high) or merely verbose?
    """
    gated = [r for r in rows if gate_fires(r, 1.5)]
    detail = []
    for r in gated:
        detail.append({
            "page": r["page"],
            "ratio": r["ratio"],
            "tess_cer": r.get("tess_cer"),
            "vision_cer": r.get("vision_cer"),
            "tess_len": r["tess_len"],
            "vision_len": r["vision_len"],
            "tess_is_noise": (r.get("tess_cer") is not None and r["tess_cer"] > 40),
        })
    noisy = sum(1 for d in detail if d["tess_is_noise"])
    return {
        "gated_pages": len(gated),
        "gated_pages_that_are_actually_noise": noisy,
        "detail": detail,
        "conclusion": (
            "length-ratio still identifies noise"
            if noisy == len(gated) and noisy > 0
            else "length ratio is an unreliable noise signal for this engine set"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default="minimax-m3", help="pricing key to analyse")
    args = ap.parse_args()

    payload = json.loads(Path(args.src).read_text(encoding="utf-8"))
    rows = collect_pages(payload, args.model)
    if not rows:
        print("no comparable pages found", file=sys.stderr)
        return 1

    print(f"model={args.model}  pages with both engine transcripts: {len(rows)}")
    print()
    print(f"{'page':<10} {'vis_len':>8} {'tess_len':>9} {'ratio':>7} {'tessCER':>8} {'visCER':>7} {'fusCER':>7} {'Δ':>7} {'fusion':>9}")
    print("-" * 92)
    for r in rows:
        d = r.get("fusion_delta_cer")
        verdict = "helped" if r.get("fusion_helped") else ("hurt" if r.get("fusion_helped") is False else "?")
        print(
            f"{r['page']:<10} {r['vision_len']:>8} {r['tess_len']:>9} {r['ratio']:>7.2f} "
            f"{(f'{r['tess_cer']:.1f}' if r.get('tess_cer') is not None else '-'):>8} "
            f"{(f'{r['vision_cer']:.1f}' if r.get('vision_cer') is not None else '-'):>7} "
            f"{(f'{r['fusion_cer']:.1f}' if r.get('fusion_cer') is not None else '-'):>7} "
            f"{(f'{d:+.1f}' if d is not None else '-'):>7} {verdict:>9}"
        )

    results = [evaluate(rows, t) for t in THRESHOLDS]
    print()
    print("=" * 92)
    print("GATE REPLAY — does firing the gate actually help?")
    print("=" * 92)
    print(
        f"{'threshold':<11} {'fired':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
        f"{'prec':>6} {'recall':>7} {'netCER':>8}"
    )
    for r in results:
        print(
            f"{str(r['label']):<11} {r['fired_count']:>6} "
            f"{r['true_positive_fired_because_fusion_hurt']:>4} "
            f"{r['false_positive_fired_but_fusion_would_have_helped']:>4} "
            f"{r['false_negative_silent_but_fusion_hurt']:>4} "
            f"{r['true_negative_silent_and_fusion_helped']:>4} "
            f"{(r['precision'] if r['precision'] is not None else '-'):>6} "
            f"{(r['recall'] if r['recall'] is not None else '-'):>7} "
            f"{r['net_cer_benefit']:>+8.1f}"
        )

    scenario = no_tesseract_scenario(rows)
    print()
    print("=" * 92)
    print("IF TESSERACT IS REPLACED — is 'too long' still a noise signal?")
    print("=" * 92)
    for d in scenario["detail"]:
        flag = "NOISE" if d["tess_is_noise"] else "verbose-but-not-noise"
        print(
            f"  {d['page']:<10} ratio {d['ratio']:>5.2f}  tesseract CER "
            f"{(f'{d['tess_cer']:.1f}' if d['tess_cer'] is not None else '-'):>6}  -> {flag}"
        )
    print(f"\n  conclusion: {scenario['conclusion']}")

    out = {
        "source": str(args.src),
        "model": args.model,
        "pages": rows,
        "threshold_results": results,
        "no_tesseract_scenario": scenario,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
