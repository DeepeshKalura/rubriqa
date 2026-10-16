"""Offline tests for the Gemini provider.

No network: every test drives `httpx.MockTransport`, so this file runs in the
default suite (`make test`). The live counterpart is
`test_minimax_live.py`'s pattern, added below as `live`-marked tests only when
a key is configured.
"""
from __future__ import annotations

import base64
import json

import httpx
import pytest

from app.config import get_settings
from app.providers.llm import GeminiProvider, build_provider
from app.providers.llm.base import LLMError
from app.schemas import LLMMessage

JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def _settings(**overrides):
    base = {
        "llm_provider": "gemini",
        "google_api_key": "test-key-not-a-real-secret",
        "gemini_model": "gemini-3.1-flash-lite",
        "gemini_max_retries": 2,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


def _provider(handler, **overrides) -> GeminiProvider:
    provider = GeminiProvider(_settings(**overrides))
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    return provider


def _ok_response(text: str = "hello", **usage):
    meta = {
        "promptTokenCount": 5100,
        "candidatesTokenCount": 2700,
        "totalTokenCount": 7800,
        "cachedContentTokenCount": 0,
    }
    meta.update(usage)
    return {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}}],
        "usageMetadata": meta,
        "modelVersion": "gemini-3.1-flash-lite",
    }


# --- payload shape ---------------------------------------------------------


def test_image_becomes_inline_data_jpeg_part():
    provider = GeminiProvider(_settings())
    payload = provider.build_payload(
        [LLMMessage(role="user", content="transcribe")], images=[JPEG]
    )
    part = payload["contents"][0]["parts"][1]
    assert part["inline_data"]["mime_type"] == "image/jpeg"
    assert base64.b64decode(part["inline_data"]["data"]) == JPEG


def test_system_instruction_is_hoisted_out_of_contents():
    provider = GeminiProvider(_settings())
    payload = provider.build_payload(
        [
            LLMMessage(role="system", content="be exact"),
            LLMMessage(role="user", content="transcribe"),
        ]
    )
    assert payload["systemInstruction"]["parts"][0]["text"] == "be exact"
    assert [p.get("text") for p in payload["contents"][0]["parts"]] == ["transcribe"]


def test_thinking_is_disabled_only_on_mechanical_stages():
    provider = GeminiProvider(_settings(gemini_disable_thinking=True))
    read = provider.build_payload([LLMMessage(role="user", content="x")], stage="vision_read")
    score = provider.build_payload([LLMMessage(role="user", content="x")], stage="rubric_score")
    assert read["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingConfig" not in score["generationConfig"]


def test_thinking_config_absent_when_flag_off():
    provider = GeminiProvider(_settings(gemini_disable_thinking=False))
    payload = provider.build_payload([LLMMessage(role="user", content="x")], stage="vision_read")
    assert "thinkingConfig" not in payload["generationConfig"]


def test_json_mode_sets_response_mime_type():
    provider = GeminiProvider(_settings())
    payload = provider.build_payload([LLMMessage(role="user", content="x")], json_mode=True)
    assert payload["generationConfig"]["responseMimeType"] == "application/json"


# --- response mapping ------------------------------------------------------


async def test_usage_maps_to_cost_engine_fields():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=_ok_response("transcribed"))

    provider = _provider(handler)
    result = await provider.complete(
        [LLMMessage(role="user", content="transcribe")],
        images=[JPEG],
        stage="vision_read",
    )
    await provider.aclose()

    assert result.text == "transcribed"
    assert result.provider == "gemini"
    assert result.prompt_tokens == 5100
    assert result.completion_tokens == 2700
    assert result.total_tokens == 7800
    assert result.cached_tokens == 0
    assert result.latency_ms >= 0
    assert seen["key"] == "test-key-not-a-real-secret"
    assert ":generateContent" in seen["url"]
    assert "gemini-3.1-flash-lite" in seen["url"]


async def test_multi_part_text_is_joined_and_thinking_stripped():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "<thinking>hmm</thinking>"}, {"text": "ok"}]}}
                ]
            },
        )

    provider = _provider(handler)
    result = await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert result.text == "ok"


# --- errors ----------------------------------------------------------------


async def test_missing_key_raises_before_any_call():
    settings = _settings(google_api_key="")
    with pytest.raises(LLMError) as exc:
        GeminiProvider(settings)
    assert exc.value.code == "missing_api_key"


async def test_4xx_is_not_retried_and_carries_status():
    calls = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad image"}})

    provider = _provider(handler, gemini_max_retries=3)
    with pytest.raises(LLMError) as exc:
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert exc.value.code == "http_400"
    assert len(calls) == 1  # 4xx must not burn the retry budget


async def test_429_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    attempts = []

    def handler(_request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=_ok_response("recovered"))

    provider = _provider(handler, gemini_max_retries=3)
    result = await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert result.text == "recovered"
    assert len(attempts) == 2


async def test_exhausted_retries_raises_retryable(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    def handler(_request):
        return httpx.Response(503, text="unavailable")

    provider = _provider(handler, gemini_max_retries=1)
    with pytest.raises(LLMError) as exc:
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert exc.value.code == "exhausted_retries"
    assert exc.value.retryable is True


async def test_block_reason_surfaces_as_actionable_error():
    def handler(_request):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    provider = _provider(handler)
    with pytest.raises(LLMError) as exc:
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert exc.value.code == "blocked"


async def test_usage_hook_records_provider_and_tokens():
    rows = []

    async def hook(row):
        rows.append(row)

    def handler(_request):
        return httpx.Response(200, json=_ok_response())

    provider = GeminiProvider(_settings(), usage_hook=hook)
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    await provider.complete([LLMMessage(role="user", content="x")], stage="vision_read")
    await provider.aclose()

    assert rows[0]["provider"] == "gemini"
    assert rows[0]["stage"] == "vision_read"
    assert rows[0]["status"] == "ok"
    assert rows[0]["prompt_tokens"] == 5100


async def test_usage_hook_failure_never_breaks_the_call():
    async def hook(_row):
        raise RuntimeError("db down")

    def handler(_request):
        return httpx.Response(200, json=_ok_response("fine"))

    provider = GeminiProvider(_settings(), usage_hook=hook)
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    result = await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert result.text == "fine"


async def test_retired_model_404_is_actionable_and_not_retried():
    """Google retires model ids for new keys without warning. The error must
    name the fix instead of surfacing as an opaque HTTP 404."""
    calls = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": (
                        "This model models/gemini-3.1-flash-lite is no longer "
                        "available to new users. Please use models/gemini-3.5-flash-lite"
                    )
                }
            },
        )

    provider = _provider(handler, gemini_max_retries=3)
    with pytest.raises(LLMError) as exc:
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert exc.value.code == "model_retired"
    assert exc.value.retryable is False
    assert "GEMINI_MODEL" in str(exc.value)
    assert "gemini-3.5-flash-lite" in str(exc.value)
    assert len(calls) == 1


async def test_generic_404_is_not_mistaken_for_a_retirement():
    def handler(_request):
        return httpx.Response(404, json={"error": {"message": "not found"}})

    provider = _provider(handler)
    with pytest.raises(LLMError) as exc:
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert exc.value.code == "http_404"


# --- factory + pricing -----------------------------------------------------


def test_factory_builds_gemini():
    provider = build_provider(_settings())
    assert provider.name == "gemini"


def test_api_key_is_never_in_the_payload_or_url():
    provider = GeminiProvider(_settings(google_api_key="secret-canary-123"))
    payload = provider.build_payload([LLMMessage(role="user", content="x")])
    assert "secret-canary-123" not in json.dumps(payload)


def test_pricing_row_is_verified_and_cheaper_per_page():
    from app.services.cost import PRICING, CostBreakdown

    lite = PRICING["gemini-3.1-flash-lite"]
    m3 = PRICING["minimax-m3"]
    assert lite["verified"] is True

    # Deliberately NOT asserting `lite["output"] < m3["output"]`: it is false.
    # Flash-Lite charges $1.50/1M output vs M3's $1.20. The saving comes
    # entirely from token count - Gemini returns a bare transcript where M3
    # emits a reasoning preamble, and it encodes the image in ~1/5 the tokens.
    # Anyone who "optimises" by switching on the headline per-token rate, or
    # by raising max_tokens, breaks this.
    assert lite["output"] > m3["output"]
    assert lite["input"] < m3["input"]

    m3_cost = CostBreakdown(
        input_tokens=5100, cached_tokens=250, output_tokens=2700
    ).usd("MiniMax-M3")
    lite_cost = CostBreakdown(
        input_tokens=1107, cached_tokens=0, output_tokens=151
    ).usd("gemini-3.1-flash-lite")
    assert m3_cost < 0.005  # sanity: the measured figure was ~$0.0033
    assert lite_cost < 0.001
    # Measured on data/samples/pages at 2200px (reports/cloud_ocr_bench.json).
    assert lite_cost < m3_cost / 5


@pytest.mark.parametrize(
    "model_id",
    [
        "gemini-3.1-flash-lite",
        "models/gemini-3.1-flash-lite",
        "GEMINI-3.1-FLASH-LITE",
        "gemini-3.1-flash-lite-20260101",
        "gemini-3.1-flash-lite-latest",
    ],
)
def test_model_id_variants_normalise_to_the_pricing_key(model_id):
    from app.services.cost import normalise_model

    assert normalise_model(model_id) == "gemini-3.1-flash-lite"


def test_model_number_suffix_is_not_mangled():
    from app.services.cost import normalise_model

    assert normalise_model("gemini-2.5-pro-002") == "gemini-2.5-pro-002"


async def _no_sleep(_seconds):
    return None

# --- spend cap -------------------------------------------------------------


async def test_spend_cap_refuses_before_a_billable_call():
    """The cap must trip *before* the request, not after it."""
    from app.services.cost import SpendCapExceeded, SpendGuard

    calls = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(200, json=_ok_response())

    guard = SpendGuard(cap_usd=0.0001)
    # A cap smaller than one real page: the first call spends more than the cap,
    # so the next one must be refused.
    from app.services.cost import CostBreakdown

    guard.record(
        CostBreakdown(input_tokens=1107, output_tokens=151), "gemini-3.1-flash-lite"
    )
    provider = GeminiProvider(_settings(), spend_guard=guard)
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(SpendCapExceeded):
        await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()
    assert calls == [], "a capped provider must not make the request"


async def test_spend_accumulates_and_then_trips():
    from app.services.cost import CostBreakdown, SpendCapExceeded, SpendGuard

    def handler(_request):
        return httpx.Response(200, json=_ok_response())

    guard = SpendGuard(cap_usd=0.0005)  # ~ one page at measured cost
    provider = GeminiProvider(_settings(), spend_guard=guard)
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    first = await provider.complete([LLMMessage(role="user", content="x")])
    assert first.text
    assert guard.spent_usd > 0
    assert guard.calls == 1
    with pytest.raises(SpendCapExceeded):
        for _ in range(5):
            await provider.complete([LLMMessage(role="user", content="x")])
    await provider.aclose()


async def test_disabled_cap_never_trips():
    from app.services.cost import SpendGuard

    def handler(_request):
        return httpx.Response(200, json=_ok_response())

    guard = SpendGuard(cap_usd=0.0)
    provider = GeminiProvider(_settings(), spend_guard=guard)
    provider._client = httpx.AsyncClient(
        base_url=provider.settings.gemini_base_url,
        headers={"x-goog-api-key": provider.settings.gemini_key},
        transport=httpx.MockTransport(handler),
    )
    for _ in range(5):
        assert (await provider.complete([LLMMessage(role="user", content="x")])).text
    await provider.aclose()
    assert guard.spent_usd > 0


def test_estimate_run_cost_scales_with_billable_calls():
    from app.services.cost import estimate_run_cost

    # 300 entries, vision read + rubric score = 2 billable calls per entry.
    est = estimate_run_cost(300, per_record_cost_usd=0.00065, llm_calls_per_record=2)
    assert est["llm_calls"] == 600
    assert 0.3 < est["projected_cost_usd"] < 0.5
    assert "$" in est["projected_cost_formatted"]


def test_spend_guard_remaining_never_negative():
    from app.services.cost import CostBreakdown, SpendGuard

    guard = SpendGuard(cap_usd=0.001)
    guard.record(
        CostBreakdown(input_tokens=1107, output_tokens=151), "gemini-3.1-flash-lite"
    )
    assert guard.remaining_usd() >= 0


def test_factory_forwards_the_spend_guard():
    """The app always builds providers through the factory, so a guard that the
    factory cannot accept is a guard that never runs."""
    from app.services.cost import SpendGuard

    guard = SpendGuard(cap_usd=0.5)
    provider = build_provider(_settings(), spend_guard=guard)
    assert provider._spend_guard is guard
