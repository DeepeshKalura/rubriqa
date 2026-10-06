"""Live tests against the real MiniMax API. Excluded by default.

Run with:  make test-live   (costs a few tokens)
"""
from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.providers.llm import build_provider
from app.providers.llm.minimax import parse_json_lenient, strip_reasoning
from app.schemas import LLMMessage

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    settings = get_settings()
    if not settings.api_key:
        pytest.skip("no MiniMax API key configured")
    return settings.model_copy(update={"llm_provider": "minimax"})


async def test_text_completion(settings):
    provider = build_provider(settings)
    result = await provider.complete([LLMMessage(role="user", content="Reply with OK")])
    assert result.text
    assert result.provider == "minimax"
    assert "<think" not in result.text.lower()
    await provider.aclose()


async def test_json_mode(settings):
    provider = build_provider(settings)
    result = await provider.complete(
        [LLMMessage(role="user", content='Return JSON: {"ok": true}')],
        json_mode=True,
        stage="live_test",
    )
    data = parse_json_lenient(result.text)
    assert data.get("ok") is True
    await provider.aclose()


async def test_vision_reads_synthetic_page(settings, tmp_path):
    provider = build_provider(settings)
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 60), "Q1. What is 7 x 8?", fill="black")
    draw.text((40, 140), "Ans: 56", fill="black")
    path = tmp_path / "page.jpg"
    image.save(path)
    from app.pipeline.preprocessing import to_jpeg_bytes

    jpeg = to_jpeg_bytes(Image.open(path))
    result = await provider.complete(
        [
            LLMMessage(
                role="user",
                content="Transcribe the text in this exam page. Return JSON "
                '{"text": "...", "legibility": 0-1, "language": "hi|en|mixed", "notes": ""}',
            )
        ],
        images=[jpeg],
        json_mode=True,
        stage="live_vision",
    )
    data = parse_json_lenient(result.text)
    assert "56" in data["text"] or "7 x 8" in data["text"]
    await provider.aclose()


def test_strip_reasoning_handles_variants():
    assert strip_reasoning("thinking>hmm</think>Hello") == "Hello"
    assert strip_reasoning("<thinking>a</think>B") == "B"
    assert strip_reasoning("Thinking...\nHello") == "Hello"
    assert strip_reasoning("Plain") == "Plain"
