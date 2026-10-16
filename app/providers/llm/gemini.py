"""Google Gemini provider (generativelanguage `generateContent`).

Same shape as the MiniMax provider so the two are interchangeable behind
`LLMProvider`. Used for handwriting transcription (issue #9 Stage 3), where the
MiniMax vision path measured 8-20% CER against Tesseract's ~84%.

- API key comes from settings only (`GOOGLE_API_KEY`); it is never logged.
- Free tier covers our volumes: `gemini-2.5-flash-lite` is free in/out with no
  card required, verified 2026-10-14 on https://ai.google.dev/gemini-api/docs/pricing.
  The paid tier is $0.10/$0.40 per 1M input/output tokens. Do not upgrade the
  key to a billing-enabled one without re-checking that table - the free tier
  is explicitly marked "used to improve our products: Yes", the paid tier "No".
- Vision input: JPEG bytes as `inline_data` parts.
- Thinking tokens are billed as output. `gemini_disable_thinking` pins
  `thinkingBudget` to 0 for transcription stages.
"""
from __future__ import annotations

import asyncio
import base64
import random
import time
from typing import Any, Callable

import httpx

from app.config import Settings, get_settings
from app.logging import get_logger, brief
from app.providers.llm.base import LLMError, strip_reasoning
from app.schemas import LLMMessage, LLMResult
from app.services.cost import CostBreakdown, SpendCapExceeded, SpendGuard

logger = get_logger("entry_eval.llm.gemini")

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Stages where deliberation only costs money and latency. Mirrors the
# MiniMax provider's stage list.
MECHANICAL_STAGES = ("vision_read", "vision_repair", "extract_answers")


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        semaphore: asyncio.Semaphore | None = None,
        usage_hook: Callable[[dict[str, Any]], Any] | None = None,
        spend_guard: SpendGuard | None = None,
    ):
        self.settings = settings or get_settings()
        if not self.settings.gemini_key:
            raise LLMError(
                "Google API key is missing. Set GOOGLE_API_KEY in .env",
                code="missing_api_key",
            )
        self.model = self.settings.gemini_model
        self._semaphore = semaphore or asyncio.Semaphore(self.settings.llm_concurrency)
        self._usage_hook = usage_hook
        # Optional cost guard. Injected rather than constructed here so the run
        # owns the accumulator across every reader and scoring call.
        self._spend_guard = spend_guard or self._default_guard()
        self._client: httpx.AsyncClient | None = None

    def _default_guard(self) -> SpendGuard:
        return SpendGuard(getattr(self.settings, "llm_spend_cap_usd", 0.0))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.gemini_base_url,
                timeout=httpx.Timeout(self.settings.gemini_timeout_seconds, connect=15.0),
                headers={"x-goog-api-key": self.settings.gemini_key},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _system_text(self, messages: list[LLMMessage]) -> str | None:
        systems = [m.content for m in messages if m.role == "system"]
        return "\n\n".join(systems) if systems else None

    def _contents(
        self, messages: list[LLMMessage], images: list[bytes] | None
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [
            {"text": m.content} for m in messages if m.role == "user"
        ]
        for image in images or []:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image).decode(),
                    }
                }
            )
        return [{"role": "user", "parts": parts}]

    def build_payload(
        self,
        messages: list[LLMMessage],
        *,
        images: list[bytes] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        stage: str = "llm",
    ) -> dict[str, Any]:
        """Build the request body. Public so tests can assert on it without a
        transport."""
        generation: dict[str, Any] = {
            "maxOutputTokens": max_tokens or self.settings.gemini_max_output_tokens,
            "temperature": temperature,
        }
        if self.settings.gemini_disable_thinking and stage in MECHANICAL_STAGES:
            generation["thinkingConfig"] = {"thinkingBudget": 0}
        if json_mode:
            generation["responseMimeType"] = "application/json"

        system = self._system_text(messages)
        return {
            "contents": self._contents(messages, images),
            "generationConfig": generation,
            **({"systemInstruction": {"parts": [{"text": system}]}} if system else {}),
        }

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        return strip_reasoning(
            "".join(p.get("text", "") for p in content.get("parts") or [])
        )

    @staticmethod
    def _usage(data: dict[str, Any]) -> dict[str, int | None]:
        meta = data.get("usageMetadata") or {}
        # Gemini bills cached content under `cachedContentTokenCount`; there is
        # no separate discounted rate on the free tier, but the cost engine
        # splits cached from fresh input so keep them distinguishable.
        return {
            "prompt_tokens": meta.get("promptTokenCount"),
            "completion_tokens": meta.get("candidatesTokenCount"),
            "total_tokens": meta.get("totalTokenCount"),
            "cached_tokens": meta.get("cachedContentTokenCount"),
        }

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        images: list[bytes] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        stage: str = "llm",
        context: dict[str, Any] | None = None,
    ) -> LLMResult:
        payload = self.build_payload(
            messages,
            images=images,
            json_mode=json_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            stage=stage,
        )
        url = f"/models/{self.model}:generateContent"

        started = time.perf_counter()
        attempt = 0
        last_error = ""
        while attempt <= self.settings.gemini_max_retries:
            attempt += 1
            try:
                # Refuse before the billable call, not after it.
                self._spend_guard.check(stage)
                async with self._semaphore:
                    client = await self._get_client()
                    response = await client.post(url, json=payload)
                if response.status_code in RETRYABLE_STATUS:
                    raise _Retryable(f"HTTP {response.status_code}: {response.text[:200]}")
                if response.status_code == 404 and "no longer available" in response.text:
                    # Google retires model ids for new keys without warning. Its
                    # error names the replacement, so pass that through: this
                    # is a config fix, not a transient fault, and retrying or
                    # billing will not help.
                    raise LLMError(
                        f"{self.model} is retired for this key. "
                        f"{response.text[:300]} Set GEMINI_MODEL to the model it names.",
                        code="model_retired",
                        retryable=False,
                    )
                if response.status_code >= 400:
                    raise LLMError(
                        f"Gemini API error HTTP {response.status_code}: {response.text[:400]}",
                        code=f"http_{response.status_code}",
                        retryable=False,
                    )
                data = response.json()
                if data.get("promptFeedback", {}).get("blockReason"):
                    raise LLMError(
                        f"Gemini blocked the request: "
                        f"{data['promptFeedback'].get('blockReason')}",
                        code="blocked",
                        retryable=False,
                    )
                result = LLMResult(
                    text=self._extract_text(data),
                    provider=self.name,
                    model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    raw=data,
                    **self._usage(data),
                )
                await self._record(stage, context, result, error=None)
                self._track_spend(result)
                return result
            except _Retryable as exc:
                last_error = str(exc)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except SpendCapExceeded:
                # Never retried: the cap exists to stop spending, so retrying
                # would spend more time for the same guaranteed refusal.
                raise
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt > self.settings.gemini_max_retries:
                break
            delay = 1.0 * (2 ** (attempt - 1))
            delay += random.uniform(0, delay / 2)
            logger.warning(
                "llm_retry",
                provider=self.name,
                stage=stage,
                attempt=attempt,
                delay_s=round(delay, 2),
                error=brief(last_error),
            )
            await asyncio.sleep(delay)

        await self._record(stage, context, None, error=last_error)
        raise LLMError(
            f"Gemini call failed after {attempt} attempts: {brief(last_error)}",
            code="exhausted_retries",
            retryable=True,
        )

    def _track_spend(self, result: LLMResult) -> None:
        self._spend_guard.record(
            CostBreakdown(
                input_tokens=result.prompt_tokens or 0,
                cached_tokens=result.cached_tokens or 0,
                output_tokens=result.completion_tokens or 0,
            ),
            result.model,
        )

    async def _record(
        self,
        stage: str,
        context: dict[str, Any] | None,
        result: LLMResult | None,
        error: str | None,
    ) -> None:
        if self._usage_hook is None:
            return
        try:
            await self._usage_hook(
                {
                    "provider": self.name,
                    "model": result.model if result else self.model,
                    "stage": stage,
                    "status": "ok" if error is None else "error",
                    "error": (error or "")[:500] or None,
                    "prompt_tokens": result.prompt_tokens if result else None,
                    "completion_tokens": result.completion_tokens if result else None,
                    "total_tokens": result.total_tokens if result else None,
                    "cached_tokens": result.cached_tokens if result else None,
                    "latency_ms": result.latency_ms if result else None,
                    **(context or {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - usage logging must never break the pipeline
            logger.warning("llm_usage_hook_failed", error=str(exc)[:200])


class _Retryable(Exception):
    pass