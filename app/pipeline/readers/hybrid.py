"""Hybrid reader: an ordered chain of readers with confidence thresholds.

All three spec modes are expressible:
  Mode A  chain = pdf_text,tesseract            (accept first success)
  Mode B  chain = pdf_text,tesseract,minimax_vision  + repair_enabled
  Mode C  chain = pdf_text,minimax_vision

The mode names are feature flags; nothing here hardcodes a provider.
"""
from __future__ import annotations

from app.config import get_settings
from app.logging import get_logger
from app.pipeline.readers.base import DocumentReader, ReaderContext, ReaderError
from app.schemas import CanonicalPage, ReadingResult, ReadingStatus

logger = get_logger("exam_eval.reader.hybrid")


class HybridReader:
    name = "hybrid"

    def __init__(
        self,
        readers: dict[str, DocumentReader],
        settings=None,
        *,
        chain: list[str] | None = None,
    ):
        self.settings = settings or get_settings()
        self.readers = readers
        if chain is None:
            chain = [part.strip() for part in self.settings.hybrid_reader_chain.split(",")]
        self.chain = [name for name in chain if name in readers]
        if not self.chain:
            raise ReaderError(
                f"Hybrid chain {chain!r} matches no registered reader",
                code="empty_chain",
            )
        self.provider = "hybrid"
        self.model = None

    async def read(self, page: CanonicalPage, context: ReaderContext) -> ReadingResult:
        best: ReadingResult | None = None
        attempts: list[dict] = []
        for name in self.chain:
            reader = self.readers[name]
            try:
                result = await reader.read(page, context)
            except ReaderError as exc:
                attempts.append({"reader": name, "error": str(exc)[:200], "code": exc.code})
                if exc.retryable:
                    # Infrastructure failure (network): stop the chain; the page
                    # is marked deferred and retried later, not silently skipped.
                    raise
                continue
            attempts.append(
                {
                    "reader": name,
                    "confidence": result.confidence,
                    "status": result.status.value,
                }
            )
            if best is None or result.confidence > best.confidence:
                best = result
            if result.confidence >= self.settings.hybrid_accept_confidence and result.text.strip():
                return _with_attempts(result, attempts)
            if not result.text.strip():
                continue

        if best is None or not best.text.strip():
            return ReadingResult(
                text="",
                confidence=0.0,
                status=ReadingStatus.EMPTY,
                engine=self.name,
                provider=self.provider,
                mode=self.name,
                warnings=["chain_exhausted_without_reading"],
                raw={"attempts": attempts},
            )

        if best.confidence < self.settings.hybrid_escalate_below and best.status == ReadingStatus.OK:
            best.status = ReadingStatus.LOW_CONFIDENCE
            if "below_escalation_threshold" not in best.warnings:
                best.warnings.append("below_escalation_threshold")
        return _with_attempts(best, attempts)


def _with_attempts(result: ReadingResult, attempts: list[dict]) -> ReadingResult:
    result.raw = {**(result.raw or {}), "chain_attempts": attempts}
    return result
