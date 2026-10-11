"""MiniMax provider (OpenAI-compatible /v1/chat/completions).

- API key comes from settings only; it is never logged.
- Retries with exponential backoff + jitter on timeouts/429/5xx.
- Usage is written to the `llm_calls` table when a session factory is given.
- Vision input: JPEG bytes as data URLs (max 10 MB per image on the API side;
  preprocessing keeps pages well below that).
"""
from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from typing import Any, Callable

import httpx

from app.config import Settings, get_settings
from app.logging import get_logger, redact
from app.providers.llm.base import LLMError, strip_reasoning
from app.schemas import LLMMessage, LLMResult

logger = get_logger("entry_eval.llm.minimax")

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class MiniMaxProvider:
    name = "minimax"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        semaphore: asyncio.Semaphore | None = None,
        usage_hook: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.settings = settings or get_settings()
        if not self.settings.api_key:
            raise LLMError(
                "MiniMax API key is missing. Set MINIMAX_API_KEY or LLM_API_KEY in .env",
                code="missing_api_key",
            )
        self.model = self.settings.minimax_text_model
        self.vision_model = self.settings.minimax_vision_model
        self._semaphore = semaphore or asyncio.Semaphore(self.settings.llm_concurrency)
        self._usage_hook = usage_hook
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.minimax_base_url,
                timeout=httpx.Timeout(self.settings.minimax_timeout_seconds, connect=15.0),
                headers={"Authorization": f"Bearer {self.settings.api_key}"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_content(
        self, messages: list[LLMMessage], images: list[bytes] | None
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": m.content} for m in messages if m.role == "user"
        ]
        for image in images or []:
            b64 = base64.b64encode(image).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        return content

    def _system_text(self, messages: list[LLMMessage]) -> str | None:
        systems = [m.content for m in messages if m.role == "system"]
        return "\n\n".join(systems) if systems else None

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
        payload: dict[str, Any] = {
            "model": self.vision_model if images else self.model,
            "max_tokens": max_tokens or self.settings.minimax_max_output_tokens,
            "temperature": temperature,
            "messages": [],
        }
        system = self._system_text(messages)
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append(
            {"role": "user", "content": self._build_content(messages, images)}
        )
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Transcription/extraction stages are pure mechanical tasks where the
        # reasoning preamble only eats the token budget. Rubric scoring keeps
        # thinking enabled because deliberation improves grading quality.
        if self.settings.minimax_disable_thinking and stage in (
            "vision_read",
            "vision_repair",
            "extract_answers",
        ):
            # verified against the live API: removes the thinking preamble
            payload["thinking"] = {"type": "disabled"}

        started = time.perf_counter()
        attempt = 0
        last_error: str = ""
        while attempt <= self.settings.minimax_max_retries:
            attempt += 1
            try:
                async with self._semaphore:
                    client = await self._get_client()
                    response = await client.post("/chat/completions", json=payload)
                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    raise _Retryable(last_error)
                if response.status_code >= 400:
                    body = response.text[:400]
                    raise LLMError(
                        f"MiniMax API error HTTP {response.status_code}: {body}",
                        code=f"http_{response.status_code}",
                        retryable=False,
                    )
                data = response.json()
                if data.get("base_resp", {}).get("status_code"):
                    raise LLMError(
                        f"MiniMax API error: {data['base_resp'].get('status_msg')}",
                        code="api_error",
                    )
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message", {})
                content = strip_reasoning(message.get("content") or "")
                usage = data.get("usage") or {}
                result = LLMResult(
                    text=content,
                    provider=self.name,
                    model=data.get("model") or payload["model"],
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    raw=None,
                )
                await self._record(stage, context, result, error=None)
                return result
            except _Retryable as exc:
                last_error = str(exc)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt > self.settings.minimax_max_retries:
                break
            delay = 1.0 * (2 ** (attempt - 1))
            delay += random.uniform(0, delay / 2)
            logger.warning(
                "llm_retry",
                stage=stage,
                attempt=attempt,
                delay_s=round(delay, 2),
                error=redact(last_error),
            )
            await asyncio.sleep(delay)

        await self._record(stage, context, None, error=last_error)
        raise LLMError(
            f"MiniMax call failed after {attempt} attempts: {redact(last_error)}",
            code="exhausted_retries",
            retryable=True,
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
                    "model": result.model if result else (self.vision_model),
                    "stage": stage,
                    "status": "ok" if error is None else "error",
                    "error": (error or "")[:500] or None,
                    "prompt_tokens": result.prompt_tokens if result else None,
                    "completion_tokens": result.completion_tokens if result else None,
                    "total_tokens": result.total_tokens if result else None,
                    "latency_ms": result.latency_ms if result else None,
                    **(context or {}),
                }
            )
        except Exception as exc:  # noqa: BLE001 - usage logging must never break the pipeline
            logger.warning("llm_usage_hook_failed", error=str(exc)[:200])


class _Retryable(Exception):
    pass


def _balance_close(text: str) -> str | None:
    """If `text` ends with unbalanced open braces/brackets (ignoring those
    inside strings — tracked with escape state), return `text` with the
    matching closers appended. Returns None if no `{` was ever seen.

    This recovers from truncated model output where the JSON never closed.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    seen_open = False
    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
            seen_open = True
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack and stack[-1] == ch:
                stack.pop()
    if not seen_open:
        return None
    suffix = '"' if in_string else ""
    return text + suffix + "".join(reversed(stack))


def parse_json_lenient(text: str) -> dict[str, Any]:
    """Parse model JSON output, tolerating code fences, surrounding prose,
    and *truncated* output (model ran out of tokens before closing braces)."""
    text = strip_reasoning(text)
    if not text:
        raise LLMError("Empty response from model", code="empty_response")

    def _attempt(candidate: str) -> dict[str, Any] | None:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None

    # 1) the raw response (after reasoning strip)
    parsed = _attempt(text)
    if parsed is not None:
        return parsed

    # 2) strip code fences, then parse
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
    parsed = _attempt(cleaned)
    if parsed is not None:
        return parsed

    # 3) find the first balanced { ... } block in the cleaned text
    depth = 0
    in_string = False
    escape = False
    start_idx = -1
    for i, ch in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if start_idx == -1:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx != -1:
                parsed = _attempt(cleaned[start_idx : i + 1])
                if parsed is not None:
                    return parsed
                start_idx = -1  # keep trying subsequent top-level objects

    # 4) truncated: synthesize the missing closers and re-parse
    repaired = _balance_close(cleaned)
    if repaired is not None:
        parsed = _attempt(repaired)
        if parsed is not None:
            return parsed

    raise LLMError(f"Model did not return valid JSON: {text[:200]}", code="bad_json")
