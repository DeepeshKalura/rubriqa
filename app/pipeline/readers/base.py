"""Document reader abstraction.

A reader turns one CanonicalPage (plus its source document context) into a
ReadingResult. Readers must never mutate the source document and must be
safe to call concurrently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.schemas import CanonicalDocument, CanonicalPage, ReadingResult


class ReaderContext:
    """Everything a reader may need to do its job."""

    def __init__(
        self,
        document: CanonicalDocument,
        *,
        original_path: Path | None = None,
        page_image_path: Path | None = None,
    ):
        self.document = document
        self.original_path = original_path
        self.page_image_path = page_image_path


class DocumentReader(Protocol):
    name: str
    provider: str
    model: str | None

    async def read(self, page: CanonicalPage, context: ReaderContext) -> ReadingResult: ...


class ReaderError(Exception):
    def __init__(self, message: str, code: str = "reader_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
