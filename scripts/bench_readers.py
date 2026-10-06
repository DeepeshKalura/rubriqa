"""Reader benchmark: run each reader mode over the labeled synthetic dataset
and measure text accuracy against ground truth (character error rate).

This is the evidence needed to choose a reader mode; nothing here is claimed
without numbers.

Usage:
    python -m scripts.bench_readers --data /data/demo [--limit 8]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.config import get_settings
from app.logging import configure_logging
from app.pipeline.ingestion import ingest
from app.pipeline.preprocessing import render_pdf_page, load_image, save_page_image
from app.pipeline.readers import build_reader
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.providers.llm import build_provider
from app.schemas import SourceType
from app.services.storage import Storage


def cer(truth: str, hypothesis: str) -> float:
    """Character error rate (Levenshtein / len(truth)); normalized whitespace."""
    truth = " ".join((truth or "").split())
    hypothesis = " ".join((hypothesis or "").split())
    if not truth:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, char_truth in enumerate(truth, 1):
        current = [i]
        for j, char_hyp in enumerate(hypothesis, 1):
            cost = 0 if char_truth == char_hyp else 1
            current.append(
                min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(truth)


def answer_recall(answers: dict[str, str], hypothesis: str) -> float:
    """Fraction of ground-truth answers whose normalized text appears in the
    transcript. This is the metric that matters for the downstream task
    (CER alone punishes inserted question text)."""
    import re

    def norm(value: str) -> str:
        value = re.sub(r"[^\w\s\u0900-\u097F]", " ", value.lower())
        return re.sub(r"\s+", " ", value).strip()

    haystack = norm(hypothesis)
    if not answers:
        return 0.0
    hits = sum(1 for answer in answers.values() if norm(answer) and norm(answer) in haystack)
    return hits / len(answers)


async def main(data: Path, limit: int) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    llm = build_provider(settings)
    truth_all = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))
    students = list(truth_all["students"].items())[:limit]

    # label a few representative files per student
    files = []
    for external_id, info in students:
        base = data / "Class 8" / info["batch"] / external_id
        for filename in ("answer_sheet.jpg", "answer_sheet_scanned.pdf", "answer_sheet_mixed.pdf"):
            path = base / filename
            if path.exists():
                files.append((external_id, path, info))

    modes = ["pdf_text", "tesseract", "minimax_vision", "hybrid"]
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    storage = Storage()
    storage.ensure_dirs()

    for external_id, path, info in files:
        canonical = ingest(path)
        for mode in modes:
            reader = build_reader(mode, llm=llm, settings=settings)
            text_parts = []
            confidence = []
            error = None
            started = time.perf_counter()
            for page in canonical.pages:
                image_path = None
                if canonical.source_type in (SourceType.PDF, SourceType.IMAGE):
                    if canonical.source_type is SourceType.PDF:
                        image = render_pdf_page(Path(path), page.index)
                    else:
                        image = load_image(Path(path))
                    target = Path("/tmp") / f"bench_{external_id}_{page.index}.png"
                    save_page_image(image, target)
                    image_path = target
                context = ReaderContext(
                    document=canonical,
                    original_path=path,
                    page_image_path=image_path,
                )
                try:
                    reading = await reader.read(page, context)
                    text_parts.append(reading.text)
                    confidence.append(reading.confidence)
                except ReaderError as exc:
                    error = str(exc)[:200]
                    break
            elapsed = time.perf_counter() - started
            hypothesis = "\n".join(text_parts)
            expected = "\n".join(
                f"{q}. {info['answers'][q]}" for q in sorted(info["answers"])
            )
            results[mode].append(
                {
                    "student": external_id,
                    "file": path.name,
                    "cer": round(cer(expected, hypothesis), 3),
                    "answer_recall": round(answer_recall(info["answers"], hypothesis), 3),
                    "confidence": round(sum(confidence) / len(confidence), 3) if confidence else 0.0,
                    "seconds": round(elapsed, 2),
                    "chars": len(hypothesis),
                    "error": error,
                }
            )
            print(
                f"{mode:14s} {external_id} {path.name:26s} "
                f"CER={results[mode][-1]['cer']:.3f} "
                f"recall={results[mode][-1]['answer_recall']:.2f} "
                f"conf={results[mode][-1]['confidence']:.2f} "
                f"{elapsed:.1f}s" + (f" ERROR={error}" if error else "")
            )

    print()
    print(
        f"{'mode':16s} {'mean CER':>9s} {'recall':>7s} {'mean conf':>10s} "
        f"{'mean s':>7s} {'runs':>5s}"
    )
    summary = {}
    for mode, rows in results.items():
        valid = [row for row in rows if row["error"] is None]
        if not valid:
            print(f"{mode:16s} {'n/a':>9s} {'n/a':>7s} {'n/a':>10s} {'n/a':>7s} {len(rows):>5d}")
            summary[mode] = {"error": "all runs failed", "runs": len(rows)}
            continue
        mean_cer = sum(row["cer"] for row in valid) / len(valid)
        mean_recall = sum(row["answer_recall"] for row in valid) / len(valid)
        mean_conf = sum(row["confidence"] for row in valid) / len(valid)
        mean_seconds = sum(row["seconds"] for row in valid) / len(valid)
        print(
            f"{mode:16s} {mean_cer:9.3f} {mean_recall:7.3f} {mean_conf:10.3f} "
            f"{mean_seconds:7.2f} {len(valid):5d}"
        )
        summary[mode] = {
            "mean_cer": round(mean_cer, 3),
            "mean_answer_recall": round(mean_recall, 3),
            "mean_confidence": round(mean_conf, 3),
            "mean_seconds": round(mean_seconds, 2),
            "runs": len(valid),
            "failed_runs": len(rows) - len(valid),
        }

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": str(data),
        "files": len(files),
        "llm_provider": settings.llm_provider,
        "summary": summary,
        "detail": results,
        "note": (
            "Measured on synthetic pages against generator ground truth. "
            "CER counts inserted question text, so read it together with "
            "answer_recall (fraction of expected answers found in the transcript). "
            "This is a relative comparison between readers on this dataset only, "
            "not an accuracy claim for real handwritten exams."
        ),
    }
    path_out = reports / f"reader_bench_{stamp}.json"
    path_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nReport: {path_out}")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/data/demo"))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.data, args.limit))


if __name__ == "__main__":
    run()
