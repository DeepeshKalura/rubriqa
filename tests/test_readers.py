"""Reader tests: success, failure, low confidence, repair flow, hybrid chain."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.preprocessing import render_pdf_page, save_page_image
from app.pipeline.readers import build_reader, build_repairer
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.pipeline.readers.hybrid import HybridReader
from app.pipeline.readers.minimax_vision import MiniMaxVisionReader
from app.pipeline.readers.pdf_text import PdfTextReader
from app.pipeline.readers.tesseract import TesseractReader
from app.providers.llm.fake import FakeProvider, FailingProvider
from app.schemas import CanonicalDocument, CanonicalPage, ReadingStatus, SourceType


def make_context(tmp_path, page: CanonicalPage, image: bool = True) -> ReaderContext:
    image_path = None
    if image:
        from PIL import Image

        pil = Image.new("RGB", (600, 800), "white")
        image_path = tmp_path / "page.png"
        save_page_image(pil, image_path)
    document = CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename="test.png",
        sha256="a" * 64,
    )
    return ReaderContext(document=document, page_image_path=image_path)


async def test_pdf_text_reader_accepts_text_layer(tmp_path):
    page = CanonicalPage(index=0, has_text_layer=True, text="Q1. 56 marks")
    result = await PdfTextReader().read(page, make_context(tmp_path, page, image=False))
    assert result.confidence == 1.0
    assert "56" in result.text
    assert result.status is ReadingStatus.OK


async def test_pdf_text_reader_reports_empty_without_layer(tmp_path):
    page = CanonicalPage(index=0, has_text_layer=False)
    result = await PdfTextReader().read(page, make_context(tmp_path, page, image=False))
    assert result.status is ReadingStatus.EMPTY
    assert result.confidence == 0


async def test_tesseract_reads_printed_page(tmp_path):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1400, 400), "white")
    draw = ImageDraw.Draw(image)
    from PIL import ImageFont

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 44)
    draw.text((60, 80), "Q1. What is 7 x 8?", font=font, fill="black")
    draw.text((60, 180), "Ans: 56", font=font, fill="black")
    path = tmp_path / "printed.png"
    image.save(path)
    page = CanonicalPage(index=0, has_text_layer=False)
    context = ReaderContext(
        document=CanonicalDocument(
            source_type=SourceType.IMAGE, original_filename="printed.png", sha256="b" * 64
        ),
        page_image_path=path,
    )
    result = await TesseractReader().read(page, context)
    assert "56" in result.text
    assert 0 <= result.confidence <= 1


async def test_tesseract_requires_image(tmp_path):
    page = CanonicalPage(index=0)
    context = ReaderContext(
        document=CanonicalDocument(
            source_type=SourceType.TEXT, original_filename="x.txt", sha256="c" * 64
        ),
        page_image_path=None,
    )
    with pytest.raises(ReaderError) as exc:
        await TesseractReader().read(page, context)
    assert exc.value.code == "missing_page_image"


async def test_vision_reader_success(tmp_path):
    llm = FakeProvider()
    page = CanonicalPage(index=0)
    result = await MiniMaxVisionReader(llm).read(page, make_context(tmp_path, page))
    assert result.text
    assert result.confidence > 0.5
    assert llm.calls[0]["stage"] == "vision_read"
    assert llm.calls[0]["images"] == 1


async def test_vision_reader_failure_propagates(tmp_path):
    llm = FailingProvider()
    page = CanonicalPage(index=0)
    with pytest.raises(ReaderError) as exc:
        await MiniMaxVisionReader(llm).read(page, make_context(tmp_path, page))
    assert exc.value.retryable is True


async def test_vision_repair_keeps_original_in_raw(tmp_path):
    llm = FakeProvider()
    page = CanonicalPage(index=0)
    context = make_context(tmp_path, page)
    original = await MiniMaxVisionReader(llm).read(page, context)
    repaired = await MiniMaxVisionReader(llm).repair(original, context, reason="low confidence")
    assert repaired.text
    assert repaired.raw["original_text"] == original.text
    assert repaired.mode == "repair"


class StubReader:
    def __init__(self, name: str, text: str, confidence: float, status=ReadingStatus.OK):
        self.name = name
        self.provider = name
        self.model = None
        self._text = text
        self._confidence = confidence
        self._status = status

    async def read(self, page, context):
        from app.schemas import ReadingResult

        return ReadingResult(
            text=self._text,
            confidence=self._confidence,
            status=self._status,
            engine=self.name,
            provider=self.name,
            mode=self.name,
        )


async def test_hybrid_accepts_first_confident_reader(tmp_path):
    readers = {"a": StubReader("a", "good text", 0.95), "b": StubReader("b", "bad", 0.1)}
    hybrid = HybridReader(readers, chain=["a", "b"])
    page = CanonicalPage(index=0)
    result = await hybrid.read(page, make_context(tmp_path, page, image=False))
    assert result.text == "good text"
    assert result.raw["chain_attempts"][0]["reader"] == "a"


async def test_hybrid_escalates_when_low_confidence(tmp_path):
    readers = {"a": StubReader("a", "weak", 0.2), "b": StubReader("b", "better", 0.9)}
    hybrid = HybridReader(readers, chain=["a", "b"])
    page = CanonicalPage(index=0)
    result = await hybrid.read(page, make_context(tmp_path, page, image=False))
    assert result.text == "better"
    assert len(result.raw["chain_attempts"]) == 2


async def test_hybrid_returns_best_when_nothing_confident(tmp_path):
    readers = {"a": StubReader("a", "weak", 0.3), "b": StubReader("b", "", 0.0, ReadingStatus.EMPTY)}
    hybrid = HybridReader(readers, chain=["a", "b"])
    page = CanonicalPage(index=0)
    result = await hybrid.read(page, make_context(tmp_path, page, image=False))
    assert result.text == "weak"
    assert result.status is ReadingStatus.LOW_CONFIDENCE


def test_build_reader_feature_flags():
    llm = FakeProvider()
    assert build_reader("pdf_text", llm=llm).name == "pdf_text"
    assert build_reader("tesseract", llm=llm).name == "tesseract"
    assert build_reader("minimax_vision", llm=llm).name == "minimax_vision"
    assert build_reader("hybrid", llm=llm).name == "hybrid"


def test_build_reader_unknown_mode():
    with pytest.raises(ReaderError):
        build_reader("magic")


def test_repair_disabled_by_flag(test_settings):
    llm = FakeProvider()
    settings = test_settings.model_copy(update={"repair_enabled": False})
    assert build_repairer(llm, settings) is None
    settings = test_settings.model_copy(update={"repair_enabled": True})
    assert build_repairer(llm, settings) is not None
