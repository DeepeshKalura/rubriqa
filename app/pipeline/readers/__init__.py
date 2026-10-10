"""Reader factory: builds the reader selected by READER_MODE.

The hybrid chain escalates whole pages to the next reader when confidence is
low. There is no separate repair pass (the chain itself handles low-confidence
pages).
"""
from __future__ import annotations

from app.config import Settings, get_settings
from app.logging import get_logger
from app.pipeline.readers.base import DocumentReader, ReaderError
from app.pipeline.readers.hybrid import HybridReader
from app.pipeline.readers.minimax_vision import MiniMaxVisionReader
from app.pipeline.readers.pdf_text import PdfTextReader
from app.pipeline.readers.tesseract import TesseractReader

logger = get_logger("entry_eval.readers")


def build_reader(
    mode: str | None = None,
    *,
    llm=None,
    settings: Settings | None = None,
) -> DocumentReader:
    settings = settings or get_settings()
    mode = (mode or settings.reader_mode).lower()

    primitives: dict[str, DocumentReader] = {
        "pdf_text": PdfTextReader(),
        "tesseract": TesseractReader(settings),
    }
    if mode in ("hybrid", "minimax_vision"):
        if llm is None:
            raise ReaderError(
                f"reader_mode={mode} needs an LLM provider but none was supplied",
                code="missing_llm",
            )
        primitives["minimax_vision"] = MiniMaxVisionReader(llm, settings)

    if mode in primitives:
        return primitives[mode]
    if mode == "hybrid":
        return HybridReader(primitives, settings)
    raise ReaderError(f"Unknown READER_MODE '{mode}'", code="bad_reader_mode")
