"""Benchmark OCR paths across LLM models and emit a cost/accuracy/time report.

Drives the *real* pipeline through POST /sandbox/compare (one entry per run, the
same path the sandbox UI uses), so there is no parallel benchmark code path to
drift out of sync.

For each (model, page) it runs all three paths, then computes per-cell CER, token
usage, cached-token share, wall time and USD cost. Output goes to stdout plus a
JSON dump for the report generator.

Usage (Docker):
    docker compose exec web python -m scripts.bench_models --out reports/bench_models.json
    docker compose exec web python -m scripts.bench_models --pages page-01 page-02 --models MiniMax-M3

Defaults to every bundled sample page with a golden transcript and the two
MiniMax models (M3 and M3.1-Flash-Preview).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.cost import (  # noqa: E402
    PRICING,
    REFERENCE_PRICING,
    CostBreakdown,
    PathStats,
    RunSample,
    normalise_model,
    project_records,
    project_storage,
)

API = os.environ.get("SANDBOX_API", "http://127.0.0.1:8000")
PAGES_DIR = Path("data/samples/pages")

# Keep the prefill identical to the screenshot/text-report harnesses so scoring
# inputs are stable and only the OCR + model vary.
PREFILL = {
    "language": "auto",
    "theme": "ai_education",
    "situation": (
        "Class 7 students struggle with math word problems because they cannot "
        "understand the language of the question."
    ),
    "proposed_solution": (
        "An AI tutor that translates the word problem into simpler Hindi, "
        "explains each step, and gives practice problems of similar type so "
        "students can build confidence and speed."
    ),
    "ai_description": (
        "AI is a computer program that can learn patterns from examples and "
        "use those patterns to help solve new problems."
    ),
}

PATH_ORDER = ["vision", "tesseract_llm", "vision_ocr_fusion"]
PATH_LABEL = {
    "vision": "A (vision-LLM)",
    "tesseract_llm": "B (tesseract+LLM)",
    "vision_ocr_fusion": "C (fusion)",
}


def post_compare(image: Path, *, text_model: str, vision_model: str, timeout: int = 600) -> dict:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, val in PREFILL.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode())
    for key in ("run_vision", "run_tesseract", "run_fusion"):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\ntrue\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"text_model\"\r\n\r\n{text_model}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"vision_model\"\r\n\r\n{vision_model}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"{image.name}\"\r\nContent-Type: image/png\r\n\r\n".encode()
    )
    parts.append(image.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{API}/sandbox/compare",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def golden_for(page: Path) -> str:
    gt = Path("data/samples/golden") / f"{page.stem}.txt"
    return gt.read_text(encoding="utf-8").strip() if gt.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/bench_models.json")
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["MiniMax-M3", "MiniMax-M3.1-Flash-Preview"])
    ap.add_argument("--records", type=int, default=25_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    pages = sorted(PAGES_DIR.glob("page-*.png"))
    if args.pages:
        pages = [p for p in pages if p.stem in set(args.pages)]
    if not pages:
        print("no pages to benchmark", file=sys.stderr)
        return 1

    runs: list[dict] = []
    for model in args.models:
        for page in pages:
            golden = golden_for(page)
            print(f"[bench] {model} · {page.name} ...", file=sys.stderr, flush=True)
            t0 = time.time()
            try:
                resp = post_compare(
                    page,
                    text_model=model,
                    vision_model=model,
                    timeout=args.timeout,
                )
            except urllib.error.HTTPError as e:
                print(f"  HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}", file=sys.stderr)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            print(f"  done in {wall:.1f}s", file=sys.stderr, flush=True)
            for r in resp.get("results", []):
                if r.get("path") not in PATH_ORDER:
                    continue
                cost = CostBreakdown(
                    input_tokens=r.get("input_tokens") or 0,
                    cached_tokens=r.get("cached_tokens") or 0,
                    output_tokens=r.get("output_tokens") or 0,
                )
                runs.append(
                    {
                        "model": model,
                        "pricing_key": normalise_model(r.get("model") or model),
                        "page": page.stem,
                        "path": r["path"],
                        "ok": bool(r.get("ok")),
                        "error": r.get("error"),
                        "cer": r.get("cer"),
                        "score": (r.get("score") or {}).get("total_awarded"),
                        "score_max": (r.get("score") or {}).get("total_max"),
                        "input_tokens": cost.input_tokens,
                        "cached_tokens": cost.cached_tokens,
                        "output_tokens": cost.output_tokens,
                        "usd": round(cost.usd(r.get("model") or model), 6),
                        "elapsed_ms": r.get("elapsed_ms") or 0,
                        "ocr_chars": len(r.get("ocr_text") or ""),
                        "ocr_text": r.get("ocr_text") or "",
                        "has_golden": bool(golden),
                    }
                )

    # --- aggregate ----------------------------------------------------------
    stats: dict[str, PathStats] = {}
    for run in runs:
        key = f"{run['pricing_key']}|{run['path']}"
        st = stats.setdefault(key, PathStats(path=run["path"], model=run["pricing_key"]))
        st.add(
            RunSample(
                path=run["path"],
                model=run["pricing_key"],
                input_tokens=run["input_tokens"],
                cached_tokens=run["cached_tokens"],
                output_tokens=run["output_tokens"],
                elapsed_ms=run["elapsed_ms"],
                cer=run["cer"],
                score=run["score"],
                ok=run["ok"],
            )
        )

    # CER winner per page/model: best (lowest) CER among ok runs.
    winners: dict[tuple[str, str], str] = {}
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        by_cell.setdefault((run["pricing_key"], run["page"]), []).append(run)
    for (pk, page), cell_runs in by_cell.items():
        cands = [r for r in cell_runs if r["ok"] and r["cer"] is not None]
        if cands:
            winners[(pk, page)] = min(cands, key=lambda r: r["cer"])["path"]

    summary = []
    for key, st in stats.items():
        pk, path = key.split("|", 1)
        model = next(r["model"] for r in runs if r["pricing_key"] == pk and r["path"] == path)
        ti, tc, to = st.mean_tokens()
        wins = sum(1 for k, v in winners.items() if k[0] == pk and v == path)
        entry = {
            "pricing_key": pk,
            "model": model,
            "path": path,
            "label": PATH_LABEL.get(path, path),
            "samples": len(st.samples),
            "cer_mean": round(st.mean_cer(), 1) if st.mean_cer() is not None else None,
            "cer_values": st.cer_values(),
            "mean_input_tokens": ti,
            "mean_cached_tokens": tc,
            "mean_output_tokens": to,
            "mean_latency_ms": st.mean_ms(),
            "p95_latency_ms": st.percentile_ms(95),
            "mean_cost_usd": round(st.mean_cost(), 6),
            "cer_wins": wins,
            "runs_with_ocr": sum(
                1 for r in runs
                if r["pricing_key"] == pk and r["path"] == path and r["ocr_chars"] > 0
            ),
            "failed_runs": sum(1 for s in st.samples if not s.ok),
            "price_verified": PRICING.get(pk, {}).get("verified", True),
            "price_note": PRICING.get(pk, {}).get("note", ""),
        }
        entry["projection_25k"] = project_records(
            st, records=args.records, workers=args.workers
        )
        summary.append(entry)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "pricing": {**PRICING, "reference": REFERENCE_PRICING},
        "storage_25k": project_storage(args.records),
        "workers_assumed": args.workers,
        "records_target": args.records,
        "models": args.models,
        "pages": [p.stem for p in pages],
        "summary": summary,
        "runs": runs,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out} ({len(runs)} runs)", file=sys.stderr)

    # --- stdout table -------------------------------------------------------
    print()
    print("=" * 108)
    print("BENCH SUMMARY  (models x paths)")
    print("=" * 108)
    print(f"{'model':<26} {'path':<18} {'n':>3} {'meanCER':>8} {'p95 ms':>7} {'in':>6} {'cached':>7} {'out':>5} {'$/rec':>9} {'wins':>5} {'fail':>5}")
    for s in sorted(summary, key=lambda x: (x["pricing_key"], x["path"])):
        mcer = f"{s['cer_mean']:.1f}" if s["cer_mean"] is not None else "—"
        print(
            f"{s['model'][:25]:<26} {s['label']:<18} {s['samples']:>3} {mcer:>8} {s['p95_latency_ms']:>7} "
            f"{s['mean_input_tokens']:>6} {s['mean_cached_tokens']:>7} {s['mean_output_tokens']:>5} "
            f"{s['mean_cost_usd']:>9.5f} {s['cer_wins']:>5} {s['failed_runs']:>5}"
        )
    print()
    storage = payload["storage_25k"]
    print(f"storage for {storage['records']:,} pages: {storage['total_gb']} GB ({storage['mb_per_page']} MB/page, originals only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())