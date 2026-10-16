"""Benchmark a cloud OCR API on the golden set — Stage 3 for issue #9.

This script exists to answer one question with a measurement instead of a
literature claim: **does the Gemini vision path beat Tesseract on handwritten
Devanagari, and what does it cost?**

It drives the *real* pipeline seam — `build_provider("gemini")` feeding
`build_reader("gemini_vision")` — so the number it produces is the number the
app would produce. No reimplemented prompt, no reimplemented preprocessing.

Kept separate from `scripts/bench_ocr_engines.py` on purpose: that one is
network-free and stays in the default test surface, this one spends money and
must never be run by CI.

CER comes from the same function the local benchmark and the sandbox use, so
the three numbers are directly comparable.

Cost: the free tier of `gemini-2.5-flash-lite` is free in/out and needs no
card, so a full 11-page run is $0. On the paid tier it projects ~$0.0011/page
(~$0.33 for a 300-entry program). Free-tier data may be used to improve the
provider's products - goldens only, never real student pages.

    docker compose exec web python -m scripts.bench_cloud_ocr
    docker compose exec web python -m scripts.bench_cloud_ocr --model gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_ocr_engines import char_error_rate  # identical CER method

# Prefer the live bind mount over the image's baked copy, so edits to
# data/samples take effect without rebuilding. See docker-compose.yml (`./data:/data`).
_SAMPLES = Path("/data/samples") if Path("/data/samples/pages").is_dir() else Path("data/samples")
PAGES_DIR = _SAMPLES / "pages"
GOLDEN_DIR = _SAMPLES / "golden"
OUT = Path("reports/cloud_ocr_bench.json")

# The ship gate, written down before the measurement (issue #9). PP-OCRv3 was
# rejected at 91.2% mean CER against this same bar.
TESSERACT_MEAN_CER = 84.1
TESSERACT_BEST_CER = 70.8
GATE_MEAN_CER = TESSERACT_MEAN_CER
GATE_BEST_CER = TESSERACT_BEST_CER


def _pages() -> list[Path]:
    return sorted(PAGES_DIR.glob("page-*.png"))


def load_manifest() -> dict:
    """Image stem -> golden filename. Filenames in data/samples/golden were
    proven misaligned on 2026-10-14, so nothing may resolve a golden by name."""
    path = GOLDEN_DIR / "manifest.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Goldens must be resolved through the manifest, "
            "not by filename - see the file for why."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def golden_for(image: Path, manifest: dict) -> str | None:
    """Return the golden transcript text for an image, or None if the image
    has no verified transcript."""
    filename = manifest.get("map", {}).get(image.stem)
    if not filename:
        return None
    return (GOLDEN_DIR / filename).read_text(encoding="utf-8")


def _doc_for(image: Path):
    from app.schemas import CanonicalDocument, CanonicalPage, SourceType

    page = CanonicalPage(index=0, image_path=str(image))
    return CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename=image.name,
        sha256=image.name,
        pages=[page],
    )


async def _run_alignment(settings, mode: str, model: str, manifest: dict) -> dict:
    """Score every transcription against EVERY golden file.

    This is how the misalignment was found and how it is kept honest: a
    competent transcriber makes the correct pairing obvious (a few percent),
    while a misnamed pairing is obvious too (tens of percent). It costs 11 API
    calls, so it is a deliberate check, not part of a normal benchmark.
    """
    from app.pipeline.readers import build_reader
    from app.pipeline.readers.base import ReaderContext
    from app.providers.llm import build_provider

    goldens = {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(GOLDEN_DIR.glob("page-*.txt"))
    }
    provider = build_provider(settings)
    reader = build_reader(mode, llm=provider, settings=settings)
    hyps: dict[str, str] = {}
    try:
        for image in _pages():
            ctx = ReaderContext(_doc_for(image), page_image_path=image)
            result = await reader.read(ctx.document.pages[0], ctx)
            hyps[image.stem] = result.text
    finally:
        await provider.aclose()

    names = sorted(goldens)
    report: dict[str, str] = {}
    print("rows = image, cols = golden file, value = CER%")
    print("       " + " ".join(f"{n[-2:]:>6}" for n in names))
    for img in sorted(hyps):
        cers = {g: round(100 * char_error_rate(goldens[g], hyps[img]), 1) for g in names}
        best = min(cers, key=cers.get)
        report[img] = best
        print(f"{img[-2:]:>6}  " + " ".join(f"{cers[g]:6.1f}" for g in names)
              + f"   best={best} ({cers[best]:.1f}%)")

    expected = manifest.get("map", {})
    without_golden = set(manifest.get("images_without_golden", {}))
    print()
    print("verdict:")
    for img, best in report.items():
        if img not in expected:
            state = (
                "ok - no transcript on record (expected)"
                if img in without_golden
                else "UNEXPECTED - not in manifest"
            )
        else:
            want = expected[img].replace(".txt", "")
            state = "ok" if best == want else f"MISMATCH (manifest says {want})"
        print(f"  {img}: best={best}  {state}")

    mispaired = [
        img
        for img, best in report.items()
        if img in expected and expected[img].replace(".txt", "") != best
    ]
    return {
        "model": model,
        "scored_pages": len(report),
        "best_match": report,
        "mispaired_images": mispaired,
        "images_without_golden": sorted(
            img for img in report if img not in expected
        ),
        "aligned": not mispaired,
    }


async def run(
    model: str | None,
    mode: str,
    max_side: int,
    only: list[str] | None,
    align: bool = False,
) -> dict:
    from app.config import get_settings
    from app.pipeline.readers import build_reader
    from app.providers.llm import build_provider
    from app.pipeline.readers.base import ReaderContext
    from app.services.cost import CostBreakdown

    settings = get_settings().model_copy(
        update={
            "llm_provider": "gemini",
            "vision_max_image_side": max_side,
            "llm_concurrency": 1,
        }
    )
    # Default to the model pinned in config, so a retirement only has to be
    # fixed in one place.
    model = model or settings.gemini_model
    if not settings.gemini_key:
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Add it to .env (free tier, no card needed)."
        )

    manifest = load_manifest()
    if align:
        return await _run_alignment(settings, mode, model, manifest)

    calls: list[dict] = []

    async def usage_hook(row: dict) -> None:
        calls.append(row)

    provider = build_provider(settings, usage_hook=usage_hook)
    reader = build_reader(mode, llm=provider, settings=settings)

    rows = []
    skipped: list[str] = []
    try:
        for image in _pages():
            if only and image.stem not in only:
                continue
            golden = golden_for(image, manifest)
            if golden is None:
                skipped.append(image.stem)
                print(
                    f"  {image.stem:<9} SKIP     no verified golden transcript",
                    flush=True,
                )
                continue
            ctx = ReaderContext(_doc_for(image), page_image_path=image)

            calls.clear()
            started = time.perf_counter()
            try:
                result = await reader.read(ctx.document.pages[0], ctx)
                error = None
            except Exception as exc:  # noqa: BLE001 - a failed page is a result
                result, error = None, f"{type(exc).__name__}: {exc}"
            wall_ms = (time.perf_counter() - started) * 1000

            tokens_in = sum(c["prompt_tokens"] or 0 for c in calls)
            tokens_out = sum(c["completion_tokens"] or 0 for c in calls)
            cached = sum(c["cached_tokens"] or 0 for c in calls)
            cost = CostBreakdown(
                input_tokens=tokens_in, cached_tokens=cached, output_tokens=tokens_out
            ).usd(model)

            if result is None:
                rows.append(
                    {
                        "page": image.stem,
                        "cer_pct": None,
                        "chars_ref": len(golden.strip()),
                        "chars_hyp": 0,
                        "status": "ERROR",
                        "confidence": 0.0,
                        "warnings": ["provider_error"],
                        "error": error,
                        "wall_ms": int(wall_ms),
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "cost_usd": round(cost, 6),
                        "text_head": "",
                    }
                )
                print(
                    f"  {image.stem:<9} ERROR    {error[:60]}", flush=True
                )
                continue

            cer = char_error_rate(golden, result.text)
            rows.append(
                {
                    "page": image.stem,
                    "cer_pct": None if cer is None else round(100 * cer, 1),
                    "chars_ref": len(golden.strip()),
                    "chars_hyp": len(result.text.strip()),
                    "status": str(result.status),
                    "confidence": round(result.confidence, 2),
                    "warnings": list(result.warnings),
                    "wall_ms": int(wall_ms),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": round(cost, 6),
                    "text_head": result.text[:160],
                }
            )
            head = f"{100 * cer:5.1f}%" if cer is not None else "  n/a"
            print(
                f"  {image.stem:<9} CER {head}  {len(result.text.strip()):>5} chars  "
                f"{int(wall_ms):>6} ms  {cost:.5f}$  {result.status}",
                flush=True,
            )
    finally:
        await provider.aclose()

    scored = [r for r in rows if r["cer_pct"] is not None]
    if not scored:
        raise SystemExit("no pages scored")

    cers = [r["cer_pct"] for r in scored]
    mean = sum(cers) / len(cers)
    ordered = sorted(cers)
    median = ordered[len(ordered) // 2]
    best, worst = ordered[0], ordered[-1]
    mean_ms = sum(r["wall_ms"] for r in rows) / len(rows)
    total_cost = sum(r["cost_usd"] for r in rows)
    mean_cost = total_cost / len(rows)

    return {
        "engine": mode,
        "provider": "gemini",
        "model": model,
        "vision_max_image_side": max_side,
        "n": len(rows),
        "n_scored": len(scored),
        "mean_cer_pct": round(mean, 1),
        "median_cer_pct": round(median, 1),
        "best_cer_pct": best,
        "worst_cer_pct": worst,
        "mean_wall_ms": int(mean_ms),
        "total_cost_usd": round(total_cost, 6),
        "mean_cost_usd": round(mean_cost, 6),
        "cost_per_300_usd": round(mean_cost * 300, 3),
        "skipped_images": skipped,
        "gate": {
            "beat_tesseract_mean": round(TESSERACT_MEAN_CER, 1),
            "beat_tesseract_best": round(TESSERACT_BEST_CER, 1),
            "passed_mean": mean < GATE_MEAN_CER,
            "passed_best": best < GATE_BEST_CER,
        },
        "pages": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to GEMINI_MODEL from settings")
    ap.add_argument("--mode", default="gemini_vision")
    ap.add_argument("--max-side", type=int, default=2200)
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--align",
        action="store_true",
        help="score every transcription against every golden to verify the "
        "golden set is aligned with the page images (11 API calls)",
    )
    args = ap.parse_args()

    print(f"cloud OCR benchmark — {args.model or 'default'} via {args.mode} @ {args.max_side}px")
    report = asyncio.run(run(args.model, args.mode, args.max_side, args.pages, args.align))

    if args.align:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n{'ALIGNED' if report['aligned'] else 'MISALIGNED'}: "
              f"{len(report['mispaired_images'])} mispaired image(s); "
              f"{len(report['images_without_golden'])} image(s) with no transcript")
        print(f"wrote {args.out}")
        return

    print()
    print("=" * 72)
    print(f"{report['n']} pages | mean CER {report['mean_cer_pct']}% "
          f"| median {report['median_cer_pct']}% | best {report['best_cer_pct']}% "
          f"| worst {report['worst_cer_pct']}% | mean {report['mean_wall_ms']} ms")
    gate = report["gate"]
    print(
        f"gate: beat tesseract mean {gate['beat_tesseract_mean']}% -> "
        f"{'PASS' if gate['passed_mean'] else 'FAIL'} | "
        f"best {gate['beat_tesseract_best']}% -> "
        f"{'PASS' if gate['passed_best'] else 'FAIL'}"
    )
    if report.get("skipped_images"):
        print(f"skipped (no golden): {', '.join(report['skipped_images'])}")
    print(
        f"cost: {report['total_cost_usd']}$ for {report['n']} pages "
        f"({report['mean_cost_usd']}$/page, {report['cost_per_300_usd']}$ per 300-entry program)"
    )
    print("=" * 72)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()