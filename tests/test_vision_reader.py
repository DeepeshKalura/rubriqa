"""The vision reader runs for every page that escalates out of Tesseract.

It had no test coverage, which let a typo survive: `read()` referenced
`settings.low_legibility_threshold`, a setting that does not exist, so every
single page raised AttributeError after a successful (and billable) LLM call.
These tests drive the real reader against the fake provider.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.pipeline.readers import build_reader
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.pipeline.readers.minimax_vision import MiniMaxVisionReader
from app.providers.llm import FakeProvider
from app.schemas import (
    CanonicalDocument,
    CanonicalPage,
    LLMMessage,
    LLMResult,
    ReadingStatus,
    SourceType,
)

# A real sample page, so the reader runs its actual image preprocessing
# (load_image -> to_jpeg_bytes) instead of a stub that would hide a decode bug.
SAMPLE_PAGE = Path("data/samples/pages/page-01.png")


def _sample_page() -> Path:
    if not SAMPLE_PAGE.exists():
        pytest.skip("sample page not available")
    return SAMPLE_PAGE


def _page(index: int = 0) -> CanonicalPage:
    return CanonicalPage(index=index, image_path="data/samples/pages/page-01.png")


def _doc() -> CanonicalDocument:
    return CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename="page-01.png",
        sha256="x",
        pages=[_page()],
    )


def _ctx(tmp_path) -> ReaderContext:
    return ReaderContext(_doc(), page_image_path=_sample_page())


def _settings(tmp_path, **overrides):
    base = {
        "llm_provider": "fake",
        "storage_root": tmp_path / "data",
        "vision_max_image_side": 2200,
        "low_confidence_threshold": 0.55,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


def test_read_returns_ok_and_uses_the_real_threshold_setting(tmp_path):
    """The regression: this raised AttributeError before the fix."""
    settings = _settings(tmp_path)
    reader = MiniMaxVisionReader(FakeProvider(), settings)
    result = asyncio_run(reader.read(_page(), _ctx(tmp_path)))
    assert result.text
    assert result.status == ReadingStatus.OK
    assert 0.0 <= result.confidence <= 1.0


def test_empty_transcript_is_marked_empty_not_ok(tmp_path):
    class EmptyProvider(FakeProvider):
        async def complete(self, messages, **kwargs) -> LLMResult:
            return LLMResult(
                text='{"text": "", "legibility": 0.9}',
                provider="fake",
                model="fake",
            )

    settings = _settings(tmp_path)
    reader = MiniMaxVisionReader(EmptyProvider(), settings)
    result = asyncio_run(reader.read(_page(), _ctx(tmp_path)))
    assert result.status == ReadingStatus.EMPTY
    assert "empty_vision_reading" in result.warnings


def test_low_legibility_is_flagged_using_the_threshold_flag(tmp_path):
    class BlurryProvider(FakeProvider):
        async def complete(self, messages, **kwargs) -> LLMResult:
            return LLMResult(
                text='{"text": "some text", "legibility": 0.2}',
                provider="fake",
                model="fake",
            )

    strict = MiniMaxVisionReader(
        BlurryProvider(), _settings(tmp_path, low_confidence_threshold=0.55)
    )
    lenient = MiniMaxVisionReader(
        BlurryProvider(), _settings(tmp_path, low_confidence_threshold=0.05)
    )
    strict_result = asyncio_run(strict.read(_page(), _ctx(tmp_path)))
    lenient_result = asyncio_run(lenient.read(_page(), _ctx(tmp_path)))
    assert strict_result.status == ReadingStatus.LOW_CONFIDENCE
    assert "low_legibility" in strict_result.warnings
    # The flag must actually move the decision, not just be read.
    assert lenient_result.status == ReadingStatus.OK


def test_missing_page_image_is_an_actionable_reader_error(tmp_path):
    settings = _settings(tmp_path)
    reader = MiniMaxVisionReader(FakeProvider(), settings)
    ctx = ReaderContext(_doc(), page_image_path=None)
    with pytest.raises(ReaderError) as exc:
        asyncio_run(reader.read(_page(), ctx))
    assert exc.value.code == "missing_page_image"


def test_vision_mode_requires_a_provider(tmp_path):
    with pytest.raises(ReaderError) as exc:
        build_reader("gemini_vision", llm=None, settings=_settings(tmp_path))
    assert exc.value.code == "missing_llm"


def test_gemini_vision_mode_reuses_the_provider_agnostic_reader(tmp_path):
    """gemini_vision must be the same implementation as minimax_vision: one
    reader, one prompt, two providers."""
    reader = build_reader(
        "gemini_vision", llm=FakeProvider(), settings=_settings(tmp_path)
    )
    assert isinstance(reader, MiniMaxVisionReader)


def test_building_a_gemini_reader_needs_no_minimax_key(tmp_path):
    """Provider boundaries are sacred: selecting the Gemini reader must not
    require a MiniMax API key to be present."""
    settings = _settings(tmp_path, minimax_api_key="", llm_api_key="")
    reader = build_reader("gemini_vision", llm=FakeProvider(), settings=settings)
    assert reader is not None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)