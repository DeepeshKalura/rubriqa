"""Ingestion tests: valid, invalid, duplicate, mixed PDF, empty."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.ingestion import IngestionError, ingest, ingest_pdf
from app.schemas import SourceType


def test_valid_text_document(tmp_path):
    path = tmp_path / "answers.txt"
    path.write_text("Q1. What is 7 x 8?\nAns: 56\nQ2. न्यूटन का नियम", encoding="utf-8")
    document = ingest(path)
    assert document.source_type is SourceType.TEXT
    assert document.pages[0].has_text_layer
    assert "56" in document.full_text
    assert document.sha256


def test_valid_image_document(image_factory):
    path = image_factory("sheet.png", ["Q1. 7 x 8 = ?", "Ans: 56"])
    document = ingest(path)
    assert document.source_type is SourceType.IMAGE
    assert document.pages[0].has_text_layer is False


def test_invalid_extension(tmp_path):
    path = tmp_path / "notes.xyz"
    path.write_bytes(b"data")
    with pytest.raises(IngestionError) as exc:
        ingest(path)
    assert exc.value.code == "unsupported_type"


def test_empty_document(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(IngestionError) as exc:
        ingest(path)
    assert exc.value.code == "empty_document"


def test_missing_document(tmp_path):
    with pytest.raises(IngestionError) as exc:
        ingest(tmp_path / "nope.txt")
    assert exc.value.code == "not_found"


def test_corrupt_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(IngestionError) as exc:
        ingest(path)
    assert exc.value.code == "corrupt_pdf"


def test_pdf_text_layer_detected(pdf_factory):
    path = pdf_factory("digital.pdf", [{"text": "Q1. What is 7 x 8?\nAns: 56"}])
    document = ingest_pdf(path)
    assert document.pages[0].has_text_layer is True
    assert "7 x 8" in document.full_text


def test_pdf_scan_detected_as_image(pdf_factory, image_factory):
    scanned = image_factory("scan.png", ["Q1. Answer"])
    jpeg_bytes = scanned.read_bytes()
    path = pdf_factory("scanned.pdf", [{"image": jpeg_bytes}])
    document = ingest_pdf(path)
    assert document.pages[0].has_text_layer is False
    assert document.metadata["image_pages"] == 1


def test_mixed_pdf_page_classification(pdf_factory, image_factory):
    scan = image_factory("scan2.png", ["Handwritten answer"])
    path = pdf_factory(
        "mixed.pdf",
        [{"text": "Q1. Printed question"}, {"image": scan.read_bytes()}],
    )
    document = ingest_pdf(path)
    assert document.pages[0].has_text_layer is True
    assert document.pages[1].has_text_layer is False
    assert document.metadata["text_pages"] == 1
    assert document.metadata["image_pages"] == 1


def test_duplicate_files_have_same_hash(tmp_path):
    content = "Q1. same content\nAns: 56"
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    assert ingest(first).sha256 == ingest(second).sha256


def test_html_is_stripped(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><body><h1>Q1</h1><p>Ans: 56</p></body></html>", encoding="utf-8")
    document = ingest(path)
    assert "<h1>" not in document.full_text
    assert "Ans: 56" in document.full_text
