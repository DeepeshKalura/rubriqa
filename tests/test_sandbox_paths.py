"""Regression tests for the sandbox processing paths.

Guarded root cause: MiniMax-M3 answers ``json_mode`` prompts with
markdown-fenced JSON (``\\`\\`\\`json ... `````). The vision OCR step used to
strict-parse that response, fail, and silently discard a perfectly good
transcription — which then scored an empty submission as a confident 0/50.
"""
from __future__ import annotations

import json
import secrets

from sqlalchemy import select

from app.models import (
    Entry,
    EntryOCR,
    EntryScore,
    EntryState,
    OcrMethod,
    OcrStatus,
    Program,
    ProgramState,
)
from app.providers.llm.fake import FakeProvider
from app.schemas import ReadingResult
from app.services.entries import (
    process_entry_via_fusion,
    process_entry_via_tesseract_then_llm,
    process_entry_via_vision,
)


def _scoring_payload() -> str:
    return json.dumps(
        {
            "criteria": [
                {
                    "name": name,
                    "awarded": 6,
                    "max_score": 10,
                    "rationale": "r",
                    "confidence": 0.8,
                }
                for name in (
                    "relevance",
                    "feasibility",
                    "originality",
                    "ai_understanding",
                    "communication",
                )
            ],
            "strengths": ["s"],
            "gaps": ["g"],
            "observations": "o",
        }
    )


async def _make_entry(session, image_path) -> Entry:
    program = Program(
        name=f"Sandbox path test {secrets.token_hex(3)}", state=ProgramState.ACTIVE
    )
    session.add(program)
    await session.commit()
    entry = Entry(
        program_id=program.id,
        external_ref=secrets.token_hex(4),
        state=EntryState.PENDING,
        form_kind="sandbox",
        full_name="S",
        theme="ai_education",
        drive_file_id="drive-123",
        local_file_path=str(image_path),
        status="pending",
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def test_vision_path_tolerates_fenced_json(session, test_settings, image_factory) -> None:
    """MiniMax-M3 fences its json_mode OCR response; the transcription must
    survive (regression: strict json.loads discarded it → empty OCR → 0/50)."""
    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    transcription = "Name: Aashul Sharma\nTopic: AI in Health"
    llm.queue(
        "entry_vision_ocr",
        "```json\n" + json.dumps({"text": transcription, "language": "eng"}) + "\n```",
    )
    llm.queue("entry_rubric_llm", _scoring_payload())

    result = await process_entry_via_vision(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is True
    assert result["ocr_text"] == transcription
    assert result["score"]["total_awarded"] == 30.0

    ocr_run = (
        await session.execute(select(EntryOCR).where(EntryOCR.entry_id == entry.id))
    ).scalar_one()
    assert ocr_run.status == OcrStatus.OK
    assert ocr_run.original_text == transcription


async def test_vision_path_reports_nothing_to_score(session, test_settings, image_factory) -> None:
    """Empty OCR + blank form fields must surface as an error, not a
    confident 0/50, and the scorer LLM must never be called."""
    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": "", "language": "eng"}))

    result = await process_entry_via_vision(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is False
    assert "nothing to score" in result["error"]
    assert "entry_rubric_llm" not in [c["stage"] for c in llm.calls]

    await session.refresh(entry)
    assert entry.state == EntryState.FAILED
    assert entry.status == "nothing_to_score"
    score_run = (
        await session.execute(select(EntryScore).where(EntryScore.entry_id == entry.id))
    ).scalar_one()
    assert score_run.error == "nothing_to_score"
    assert score_run.is_current is False


async def test_tesseract_path_reports_nothing_to_score(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """Path B on unreadable handwriting (empty tesseract output) must also
    surface as an error instead of a confident zero."""

    from app.pipeline.readers.tesseract import TesseractReader

    async def _empty_read(self, page, ctx):
        return ReadingResult(text="", confidence=0.0)

    monkeypatch.setattr(TesseractReader, "read", _empty_read)

    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()

    result = await process_entry_via_tesseract_then_llm(
        session, entry, settings=test_settings, llm=llm
    )

    assert result["ok"] is False
    assert "nothing to score" in result["error"]
    assert "entry_rubric_llm" not in [c["stage"] for c in llm.calls]

    await session.refresh(entry)
    assert entry.status == "nothing_to_score"


async def test_fusion_path_merges_ocr_outputs(session, test_settings, image_factory, monkeypatch) -> None:
    """Path C merges the two engine transcripts via a vision LLM call and
    scores the fused text; the fused run is the selected OCR row."""
    from app.pipeline.readers.tesseract import TesseractReader

    async def _tess_read(self, page, ctx):
        return ReadingResult(text="tesseract raw words", confidence=0.5)

    monkeypatch.setattr(TesseractReader, "read", _tess_read)

    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    vision_text = "Vision transcription line."
    llm.queue("entry_vision_ocr", json.dumps({"text": vision_text, "language": "eng"}))
    fused_text = "Fused human-like transcription."
    llm.queue("entry_fusion_ocr", json.dumps({"text": fused_text, "language": "eng"}))
    llm.queue("entry_rubric_llm", _scoring_payload())

    result = await process_entry_via_fusion(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is True
    assert result["method"] == "vision_ocr_fusion"
    assert result["ocr_text"] == fused_text
    assert result["vision_text"] == vision_text
    assert result["tesseract_text"] == "tesseract raw words"
    assert result["score"]["total_awarded"] == 30.0

    rows = (
        await session.execute(select(EntryOCR).where(EntryOCR.entry_id == entry.id))
    ).scalars().all()
    assert len(rows) == 3
    selected = [r for r in rows if r.is_selected]
    assert len(selected) == 1
    assert selected[0].method == OcrMethod.HYBRID
    assert selected[0].original_text == fused_text

    # the fusion call re-sent the page image to arbitrate disagreements
    fusion_calls = [c for c in llm.calls if c["stage"] == "entry_fusion_ocr"]
    assert len(fusion_calls) == 1 and fusion_calls[0]["images"] == 1


async def test_fusion_path_skips_fusion_when_both_engines_empty(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """Both engines empty -> no fusion LLM call, no scorer call, honest error."""
    from app.pipeline.readers.tesseract import TesseractReader

    async def _empty_read(self, page, ctx):
        return ReadingResult(text="", confidence=0.0)

    monkeypatch.setattr(TesseractReader, "read", _empty_read)

    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": "", "language": "eng"}))

    result = await process_entry_via_fusion(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is False
    assert "nothing to score" in result["error"]
    stages = [c["stage"] for c in llm.calls]
    assert "entry_fusion_ocr" not in stages
    assert "entry_rubric_llm" not in stages


def test_char_error_rate() -> None:
    from app.routers.sandbox import _char_error_rate

    assert _char_error_rate("hello world", "hello world") == 0.0
    assert _char_error_rate("hello", "hallo") == 0.2
    assert _char_error_rate("abc", "") == 1.0
    assert _char_error_rate("", "abc") is None
    # case and whitespace are normalized
    assert _char_error_rate("Hello  World\n", "hello world") == 0.0


def test_render_for_tesseract_writes_png_and_cleans_up(test_settings, image_factory) -> None:
    """The sandbox tesseract paths must hand the reader a *rendered* page
    image (real pipeline renders via preprocessing; sandbox does it inline)
    and clean up the temp file after the read."""
    from app.services.entries import _render_for_tesseract

    img = image_factory("page.png", ["hello"])
    rendered = _render_for_tesseract(img, test_settings)
    assert rendered is not None and rendered.exists()
    assert rendered.suffix == ".png"
    # it's a real PNG, not the empty file
    assert rendered.stat().st_size > 100
    # the helper returns a path the caller is responsible for cleaning up
    rendered.unlink(missing_ok=True)
    assert not rendered.exists()


async def test_fusion_gate_skips_when_tesseract_much_longer_than_vision(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """When tesseract's text is suspiciously long compared to vision's,
    tesseract is hallucinating noise (e.g. English script on a Devanagari
    page). The fusion LLM call is skipped, the hybrid OCR = vision text,
    the result has fusion_skipped=True, and no LLM call to entry_fusion_ocr
    is made."""
    from app.pipeline.readers.tesseract import TesseractReader

    vision_text = "निशा कक्षा 8वीं विद्यालय बालिका"  # 28 chars
    # tesseract output is 3x longer — would be flagged as noise
    tess_text = "nisha kaksha 8vi vidyalaya balika vidya mandir extra words padded" * 2
    assert len(tess_text) > 1.5 * len(vision_text)  # gate will fire

    async def _long_tess(self, page, ctx):
        return ReadingResult(text=tess_text, confidence=0.5)

    monkeypatch.setattr(TesseractReader, "read", _long_tess)

    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": vision_text, "language": "hin"}))
    llm.queue("entry_rubric_llm", _scoring_payload())

    result = await process_entry_via_fusion(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is True
    assert result["fusion_skipped"] is True
    assert result["ocr_text"] == vision_text  # vision text used as hybrid
    assert result["vision_text"] == vision_text
    assert result["tesseract_text"] == tess_text
    # the only fusion LLM call would be the vision-OCR call; the actual
    # fusion (entry_fusion_ocr) was skipped
    stages = [c["stage"] for c in llm.calls]
    assert "entry_fusion_ocr" not in stages
    assert "entry_rubric_llm" in stages  # scoring still happened

    # the persisted HYBRID OCR row should be the vision text and selected
    rows = (
        await session.execute(select(EntryOCR).where(EntryOCR.entry_id == entry.id))
    ).scalars().all()
    hybrid = next(r for r in rows if r.method == OcrMethod.HYBRID)
    assert hybrid.original_text == vision_text
    assert hybrid.is_selected is True


async def test_fusion_gate_does_not_skip_when_tesseract_shorter_than_vision(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """When tesseract's text is within the ratio threshold, fusion runs as
    before (the existing happy path)."""
    from app.pipeline.readers.tesseract import TesseractReader

    vision_text = "A vision transcript with reasonable length and some structure."
    tess_text = "short tess output"  # well below 1.5x

    async def _short_tess(self, page, ctx):
        return ReadingResult(text=tess_text, confidence=0.4)

    monkeypatch.setattr(TesseractReader, "read", _short_tess)

    img = image_factory("page.png", ["hello student"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": vision_text, "language": "eng"}))
    fused_text = "Fused transcript from both engines."
    llm.queue("entry_fusion_ocr", json.dumps({"text": fused_text, "language": "eng"}))
    llm.queue("entry_rubric_llm", _scoring_payload())

    result = await process_entry_via_fusion(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is True
    assert result["fusion_skipped"] is False
    assert result["ocr_text"] == fused_text  # real fusion, not the gate fallback
    stages = [c["stage"] for c in llm.calls]
    assert "entry_fusion_ocr" in stages


async def test_fusion_gate_disabled_with_high_threshold(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """The flag is configurable: with the threshold set to 999, the gate
    never fires and fusion always runs."""
    from app.pipeline.readers.tesseract import TesseractReader

    # any tess text — gate is disabled
    tess_text = "x" * 10000
    vision_text = "y" * 100

    async def _huge_tess(self, page, ctx):
        return ReadingResult(text=tess_text, confidence=0.5)

    monkeypatch.setattr(TesseractReader, "read", _huge_tess)

    img = image_factory("page.png", ["hello"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": vision_text, "language": "eng"}))
    llm.queue("entry_fusion_ocr", json.dumps({"text": "fused", "language": "eng"}))
    llm.queue("entry_rubric_llm", _scoring_payload())

    # override the setting to disable the gate
    from app.config import Settings

    class _NoGate(Settings):
        fusion_max_tess_to_vision_ratio: float = 999.0  # type: ignore[misc]

    settings = _NoGate()
    result = await process_entry_via_fusion(session, entry, settings=settings, llm=llm)

    assert result["ok"] is True
    assert result["fusion_skipped"] is False
    assert result["ocr_text"] == "fused"
    stages = [c["stage"] for c in llm.calls]
    assert "entry_fusion_ocr" in stages


async def test_fusion_gate_forces_fusion_when_vision_empty(
    session, test_settings, image_factory, monkeypatch
) -> None:
    """When vision OCR returns empty text but tesseract returned something,
    fusion MUST run (tesseract is the only input). The gate only fires when
    vision is non-empty AND tesseract is suspiciously long."""
    from app.pipeline.readers.tesseract import TesseractReader

    tess_text = "x" * 5000  # very long tesseract output

    async def _long_tess(self, page, ctx):
        return ReadingResult(text=tess_text, confidence=0.5)

    monkeypatch.setattr(TesseractReader, "read", _long_tess)

    img = image_factory("page.png", ["hello"])
    entry = await _make_entry(session, img)
    llm = FakeProvider()
    llm.queue("entry_vision_ocr", json.dumps({"text": "", "language": "eng"}))
    llm.queue("entry_fusion_ocr", json.dumps({"text": "fused-from-tess-only", "language": "eng"}))
    llm.queue("entry_rubric_llm", _scoring_payload())

    result = await process_entry_via_fusion(session, entry, settings=test_settings, llm=llm)

    assert result["ok"] is True
    assert result["fusion_skipped"] is False
    stages = [c["stage"] for c in llm.calls]
    assert "entry_fusion_ocr" in stages  # forced to run because vision is empty
    assert result["ocr_text"] == "fused-from-tess-only"
