"""Every reader and every provider must actually run.

The vision reader raised `AttributeError` on every page it ever saw - after a
successful, billable LLM call - and nothing noticed, because nothing executed
it. These tests are deliberately shallow: they assert that each component can be
built from settings and can read one real page without raising.

They are not accuracy tests. They answer a different question: *is this thing
alive?* A component that is dead is the failure mode where every other test
still passes.

The mode/provider tests also pin the feature flags to the factories: adding a
value to a `Literal` without teaching the factory about it fails here instead of
at request time.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import get_args

import pytest

from app.config import Settings, get_settings
from app.pipeline.readers import build_reader
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.providers.llm import LLMProvider, build_provider
from app.providers.llm.fake import FakeProvider
from app.providers.llm.gemini import GeminiProvider
from app.providers.llm.minimax import MiniMaxProvider
from app.schemas import (
    CanonicalDocument,
    CanonicalPage,
    ReadingStatus,
    SourceType,
)

LEGAL_STATUSES = set(ReadingStatus)
SAMPLE_PAGE = Path("data/samples/pages/page-01.png")


def _sample_page() -> Path:
    if not SAMPLE_PAGE.exists():
        pytest.skip("sample page not available")
    return SAMPLE_PAGE


def _declared(field: str) -> set[str]:
    return set(get_args(Settings.model_fields[field].annotation))


def _settings(tmp_path, **overrides):
    base = {
        "llm_provider": "fake",
        "storage_root": tmp_path / "data",
        "vision_max_image_side": 2200,
        "low_confidence_threshold": 0.55,
        "tesseract_langs": "eng+hin",
        "hybrid_reader_chain": "pdf_text,tesseract,minimax_vision",
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


def _doc(page: CanonicalPage) -> CanonicalDocument:
    return CanonicalDocument(
        source_type=SourceType.IMAGE,
        original_filename="page-01.png",
        sha256="page-01",
        pages=[page],
    )


def _read(mode: str, tmp_path, *, text_layer: bool = False):
    page = CanonicalPage(
        index=0,
        image_path=str(_sample_page()),
        has_text_layer=text_layer,
        text="A printed question, digitised." if text_layer else None,
    )
    reader = build_reader(mode, llm=FakeProvider(), settings=_settings(tmp_path))
    ctx = ReaderContext(_doc(page), page_image_path=_sample_page())
    return asyncio.run(reader.read(page, ctx))


def test_reader_mode_literal_and_factory_agree():
    """A new READER_MODE value must be teachable to the factory, or the feature
    flag can be set to something that only fails at request time."""
    declared = _declared("reader_mode")
    assert declared == {"pdf_text", "tesseract", "minimax_vision", "gemini_vision", "hybrid"}
    unbuildable = []
    for mode in sorted(declared):
        try:
            build_reader(mode, llm=FakeProvider(), settings=_settings(Path("/tmp")))
        except ReaderError:
            unbuildable.append(mode)
    assert not unbuildable, f"declared but not buildable: {unbuildable}"


def test_llm_provider_literal_and_factory_agree():
    """Same guard for LLM_PROVIDER: the declared values and the implemented
    branches must not drift apart."""
    assert _declared("llm_provider") == {"minimax", "gemini", "fake"}


def test_unknown_reader_mode_is_rejected_clearly(tmp_path):
    with pytest.raises(ReaderError) as exc:
        build_reader("not_a_reader", llm=FakeProvider(), settings=_settings(tmp_path))
    assert exc.value.code == "bad_reader_mode"


@pytest.mark.parametrize(
    "mode", ["pdf_text", "tesseract", "minimax_vision", "gemini_vision", "hybrid"]
)
def test_every_reader_runs_a_real_page(mode, tmp_path):
    """Alive, not accurate: no exception, and a legal status.

    `pdf_text` on an image page legitimately returns EMPTY (no text layer), so
    asserting "returns text" here would be a test that forces the implementation
    to lie. The bug we are guarding against is a crash or a nonsense status.
    """
    result = _read(mode, tmp_path)
    assert result.status in LEGAL_STATUSES, f"{mode} returned {result.status}"
    assert result.engine or result.mode, f"{mode} did not identify itself"
    assert isinstance(result.text, str)


def test_pdf_text_reads_a_text_layer(tmp_path):
    """A cheap real assertion where one is meaningful: the text-layer path."""
    result = _read("pdf_text", tmp_path, text_layer=True)
    assert result.status == ReadingStatus.OK
    assert "digitised" in result.text


def test_pdf_text_reports_empty_rather_than_failing(tmp_path):
    result = _read("pdf_text", tmp_path)
    assert result.status == ReadingStatus.EMPTY
    assert "no_text_layer" in result.warnings


def test_tesseract_produces_text_for_a_handwritten_page(tmp_path):
    """Tesseract is measurably bad, but "empty" would mean the binary or the
    language packs are gone - a different failure worth catching here."""
    result = _read("tesseract", tmp_path)
    assert result.status in LEGAL_STATUSES
    assert result.text.strip(), "tesseract returned nothing at all"


def test_gemini_vision_needs_no_minimax_key(tmp_path):
    """Provider boundaries are sacred: the Gemini reader must not require a
    MiniMax credential to be configured."""
    settings = _settings(tmp_path, minimax_api_key="", llm_api_key="")
    reader = build_reader("gemini_vision", llm=FakeProvider(), settings=settings)
    assert reader is not None


# --- providers --------------------------------------------------------------


def test_fake_provider_is_the_offline_default(tmp_path):
    provider = build_provider(_settings(tmp_path))
    assert isinstance(provider, FakeProvider)
    assert provider.name == "fake"


def test_minimax_provider_builds_and_reports_its_name(tmp_path):
    provider = MiniMaxProvider(
        _settings(tmp_path, llm_provider="minimax", minimax_api_key="test-key")
    )
    assert provider.name == "minimax"
    assert provider.model


def test_gemini_provider_builds_and_reports_its_name(tmp_path):
    provider = GeminiProvider(
        _settings(tmp_path, llm_provider="gemini", google_api_key="test-key")
    )
    assert provider.name == "gemini"
    assert provider.model.startswith("gemini-")


@pytest.mark.parametrize(
    "missing,provider",
    [("minimax_api_key", MiniMaxProvider), ("google_api_key", GeminiProvider)],
)
def test_provider_refuses_to_construct_without_its_key(missing, provider, tmp_path):
    from app.providers.llm.base import LLMError

    settings = _settings(
        tmp_path,
        llm_provider="gemini" if provider is GeminiProvider else "minimax",
        minimax_api_key="",
        llm_api_key="",
        google_api_key="",
    )
    with pytest.raises(LLMError) as exc:
        provider(settings)
    assert exc.value.code == "missing_api_key"


@pytest.mark.parametrize(
    "provider",
    [
        FakeProvider(),
        MiniMaxProvider(get_settings().model_copy(update={"minimax_api_key": "k"})),
        GeminiProvider(get_settings().model_copy(update={"google_api_key": "k"})),
    ],
    ids=["fake", "minimax", "gemini"],
)
def test_every_concrete_provider_satisfies_the_protocol(provider):
    """The seam is only a seam if conformance is checkable. This test could not
    be written before `LLMProvider` was marked `@runtime_checkable`."""
    assert isinstance(provider, LLMProvider)
    assert provider.name
    assert hasattr(provider, "complete")


def test_pinned_gemini_model_is_not_a_rolling_alias():
    """Google retired gemini-2.5-flash-lite for new keys on 2026-10-14. A
    floating id like `-latest` would break without anyone noticing."""
    from app.config import Settings as S

    model = S.model_fields["gemini_model"].default
    assert model == "gemini-3.1-flash-lite"
    assert not model.endswith(("-latest", "-preview", "-exp"))