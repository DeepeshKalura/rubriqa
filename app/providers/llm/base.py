"""Provider-agnostic LLM interface."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.schemas import LLMMessage, LLMResult


class LLMError(Exception):
    """Raised after retries are exhausted. Carries an actionable message."""

    def __init__(self, message: str, *, code: str = "llm_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic seam. `@runtime_checkable` so conformance is
    checkable at runtime - otherwise a provider missing `complete()` only fails
    when a page is being read."""

    name: str
    model: str | None

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
    ) -> LLMResult: ...


def strip_reasoning(text: str) -> str:
    """MiniMax M-series models can emit `thinking>...` style blocks inside
    content even in JSON mode; strip them before parsing."""
    import re

    if not text:
        return text
    text = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", text)
    text = re.sub(r"(?is)Thinking\.\.\.\s*", "", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
