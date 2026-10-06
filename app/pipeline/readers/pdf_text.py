"""PDF text-layer reader: the cheapest, most accurate path for digital PDFs."""
from __future__ import annotations

from app.pipeline.readers.base import ReaderContext
from app.schemas import CanonicalPage, ReadingResult, ReadingStatus, TextBlock


class PdfTextReader:
    name = "pdf_text"
    provider = "pymupdf"
    model = None

    async def read(self, page: CanonicalPage, context: ReaderContext) -> ReadingResult:
        if not page.has_text_layer or not (page.text or "").strip():
            return ReadingResult(
                text="",
                confidence=0.0,
                status=ReadingStatus.EMPTY,
                engine=self.name,
                provider=self.provider,
                mode=self.name,
                warnings=["no_text_layer"],
            )
        text = page.text or ""
        return ReadingResult(
            text=text,
            confidence=1.0,
            status=ReadingStatus.OK,
            blocks=[TextBlock(text=text)],
            engine=self.name,
            provider=self.provider,
            mode=self.name,
        )
