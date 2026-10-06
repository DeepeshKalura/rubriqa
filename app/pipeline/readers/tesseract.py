"""Tesseract OCR reader (eng+hin), run as a subprocess so the event loop is
never blocked and memory stays low. Confidence comes from tesseract TSV word
scores; very low scores usually mean handwriting -> escalation signal.
"""
from __future__ import annotations

import asyncio
import uuid
import csv
import io
from pathlib import Path

from app.config import get_settings
from app.logging import get_logger
from app.pipeline.preprocessing import load_image, preprocess_for_ocr
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.schemas import CanonicalPage, ReadingResult, ReadingStatus

logger = get_logger("exam_eval.reader.tesseract")


class TesseractReader:
    name = "tesseract"
    provider = "tesseract"
    model = None

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.langs = self.settings.tesseract_langs
        self.min_word_confidence = self.settings.tesseract_min_word_confidence

    async def read(self, page: CanonicalPage, context: ReaderContext) -> ReadingResult:
        image_path = context.page_image_path
        if image_path is None or not Path(image_path).exists():
            raise ReaderError(
                "Tesseract reader needs a rendered page image but none was provided",
                code="missing_page_image",
            )
        return await asyncio.to_thread(self._run, Path(image_path), page)

    def _run(self, image_path: Path, page: CanonicalPage) -> ReadingResult:
        import time

        started = time.perf_counter()
        try:
            image = load_image(image_path)
            processed = preprocess_for_ocr(image, max_side=self.settings.vision_max_image_side)
            png_bytes = io.BytesIO()
            processed.save(png_bytes, format="PNG")
            payload = png_bytes.getvalue()
        except Exception as exc:  # noqa: BLE001
            raise ReaderError(f"Image preprocessing failed: {exc}", code="preprocess_failed") from exc

        return self._run_tesseract(payload, page, started)

    def _run_tesseract(self, png_bytes: bytes, page: CanonicalPage, started: float) -> ReadingResult:
        import time

        png_path = Path(f"/tmp/tess-{uuid.uuid4().hex}.png")
        png_path.write_bytes(png_bytes)
        try:
            proc = __import__("subprocess").run(
                [
                    "tesseract",
                    str(png_path),
                    "stdout",
                    "-l",
                    self.langs,
                    "--psm",
                    "6",
                    "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise ReaderError(
                    f"tesseract failed: {proc.stderr.strip()[:300]}", code="tesseract_failed"
                )
            words = list(csv.DictReader(io.StringIO(proc.stdout), delimiter="\t"))
        finally:
            png_path.unlink(missing_ok=True)

        confidences: list[float] = []
        lines: dict[tuple[int, int, int], list[str]] = {}
        for word in words:
            text = (word.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(word.get("conf") or -1)
            except ValueError:
                conf = -1.0
            confidences.append(conf)
            key = (
                int(word.get("page_num") or 0),
                int(word.get("block_num") or 0),
                int(word.get("par_num") or 0),
                int(word.get("line_num") or 0),
            )
            lines.setdefault(key, []).append(text)

        text = "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
        low_ratio = (
            sum(1 for c in confidences if c < self.min_word_confidence) / len(confidences)
            if confidences
            else 1.0
        )
        # Map tesseract's -1..100 scale onto 0..1, then penalize low-confidence words.
        confidence = max(0.0, min(1.0, (mean_conf / 100.0))) * (1.0 - 0.5 * low_ratio)
        latency_ms = int((time.perf_counter() - started) * 1000)

        warnings: list[str] = []
        status = ReadingStatus.OK
        low_thresh = self.settings.low_confidence_threshold
        if not text.strip():
            status = ReadingStatus.EMPTY
            warnings.append("empty_ocr")
        elif confidence < low_thresh:
            status = ReadingStatus.LOW_CONFIDENCE
            warnings.append("low_confidence_ocr")
        if low_ratio > 0.5:
            warnings.append("many_low_confidence_words")

        return ReadingResult(
            text=text,
            confidence=round(confidence, 3),
            status=status,
            engine=self.name,
            provider=self.provider,
            model=None,
            mode=self.name,
            language=self.langs,
            warnings=warnings,
            latency_ms=latency_ms,
            raw={"mean_word_confidence": round(mean_conf, 1), "word_count": len(confidences)},
        )
