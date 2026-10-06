"""Ingestion adapters: source file -> CanonicalDocument.

Each adapter is a small pure function so they can be tested without a DB and
replaced independently.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from app.schemas import CanonicalDocument, CanonicalPage, SourceType, TextBlock
from app.services.storage import Storage

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
MIN_TEXT_CHARS = 20


class IngestionError(Exception):
    def __init__(self, message: str, code: str = "ingestion_error"):
        super().__init__(message)
        self.code = code


def detect_source_type(path: Path) -> SourceType:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return SourceType.PDF
    if ext in IMAGE_EXTENSIONS:
        return SourceType.IMAGE
    if ext in TEXT_EXTENSIONS:
        return SourceType.TEXT
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return SourceType.IMAGE
    if guessed == "application/pdf":
        return SourceType.PDF
    if guessed and guessed.startswith("text/"):
        return SourceType.TEXT
    raise IngestionError(f"Unsupported file type: {path.name}", code="unsupported_type")


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def ingest_text(path: Path) -> CanonicalDocument:
    raw = path.read_bytes()
    text = _decode_text(raw)
    if path.suffix.lower() in (".html", ".htm"):
        text = _strip_html(text)
    if not text.strip():
        raise IngestionError("Text document is empty", code="empty_document")
    page = CanonicalPage(
        index=0,
        has_text_layer=True,
        text=text,
        blocks=[TextBlock(text=text)],
    )
    return CanonicalDocument(
        source_type=SourceType.TEXT,
        original_filename=path.name,
        sha256=Storage.sha256_file(path),
        size_bytes=len(raw),
        mime=mimetypes.guess_type(path.name)[0] or "text/plain",
        pages=[page],
        metadata={"line_count": text.count("\n") + 1},
    )


def _strip_html(text: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def ingest_pdf(path: Path) -> CanonicalDocument:
    """Classify every page: usable text layer vs scanned/image page.

    Text pages are NOT rendered or OCR'd here; images are rendered lazily by
    the reader stage only when needed.
    """
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Could not open PDF: {exc}", code="corrupt_pdf") from exc
    if doc.page_count == 0:
        raise IngestionError("PDF has no pages", code="empty_document")

    pages: list[CanonicalPage] = []
    with doc:
        for index, page in enumerate(doc):
            text = page.get_text("text") or ""
            cleaned = text.strip()
            has_text = len(cleaned) >= MIN_TEXT_CHARS and _alnum_ratio(cleaned) > 0.4
            rect = page.rect
            pages.append(
                CanonicalPage(
                    index=index,
                    width=int(rect.width),
                    height=int(rect.height),
                    has_text_layer=has_text,
                    text=cleaned if has_text else None,
                    metadata={"page_rotation": page.rotation},
                )
            )
    return CanonicalDocument(
        source_type=SourceType.PDF,
        original_filename=path.name,
        sha256=Storage.sha256_file(path),
        size_bytes=path.stat().st_size,
        mime="application/pdf",
        pages=pages,
        metadata={
            "page_count": len(pages),
            "text_pages": sum(1 for p in pages if p.has_text_layer),
            "image_pages": sum(1 for p in pages if not p.has_text_layer),
        },
    )


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(1 for ch in text if ch.isalnum())
    return alnum / len(text)


def ingest_image(path: Path) -> CanonicalDocument:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestionError(f"Could not open image: {exc}", code="corrupt_image") from exc
    if width < 32 or height < 32:
        raise IngestionError("Image is too small to read", code="image_too_small")
    page = CanonicalPage(index=0, width=width, height=height, has_text_layer=False)
    return CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename=path.name,
        sha256=Storage.sha256_file(path),
        size_bytes=path.stat().st_size,
        mime=mimetypes.guess_type(path.name)[0] or "image/png",
        pages=[page],
        metadata={},
    )


def ingest(path: Path) -> CanonicalDocument:
    """Dispatch by detected type."""
    if not path.exists():
        raise IngestionError(f"File not found: {path}", code="not_found")
    if path.stat().st_size == 0:
        raise IngestionError("File is empty (0 bytes)", code="empty_document")
    source_type = detect_source_type(path)
    if source_type is SourceType.PDF:
        return ingest_pdf(path)
    if source_type is SourceType.IMAGE:
        return ingest_image(path)
    return ingest_text(path)
