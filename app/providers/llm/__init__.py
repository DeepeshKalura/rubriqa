"""LLM provider factory. Provider selection is a feature flag."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.config import Settings, get_settings
from app.providers.llm.base import LLMError, LLMProvider
from app.providers.llm.fake import FakeProvider
from app.providers.llm.gemini import GeminiProvider
from app.providers.llm.minimax import MiniMaxProvider, parse_json_lenient
from app.services.cost import SpendGuard


def build_provider(
    settings: Settings | None = None,
    *,
    semaphore: asyncio.Semaphore | None = None,
    usage_hook: Callable[[dict[str, Any]], Any] | None = None,
    spend_guard: SpendGuard | None = None,
) -> LLMProvider:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "fake":
        return FakeProvider()
    if provider == "minimax":
        return MiniMaxProvider(settings, semaphore=semaphore, usage_hook=usage_hook)
    if provider == "gemini":
        return GeminiProvider(
            settings,
            semaphore=semaphore,
            usage_hook=usage_hook,
            spend_guard=spend_guard,
        )
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'", code="bad_provider")


__all__ = [
    "LLMError",
    "LLMProvider",
    "FakeProvider",
    "GeminiProvider",
    "MiniMaxProvider",
    "build_provider",
    "parse_json_lenient",
]
