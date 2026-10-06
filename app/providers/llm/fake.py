"""Deterministic fake LLM provider for tests and offline demos.

It implements the same interface as MiniMaxProvider and returns canned but
structure-correct JSON depending on the stage. It never touches the network.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.providers.llm.base import LLMError
from app.schemas import LLMMessage, LLMResult


class FakeProvider:
    name = "fake"

    def __init__(self, model: str = "fake-1", *, latency_ms: int = 1):
        self.model = model
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []
        # optional scripted responses keyed by stage, consumed in order
        self.scripted: dict[str, list[str]] = {}
        self.fail_stages: set[str] = set()

    def queue(self, stage: str, text: str) -> None:
        self.scripted.setdefault(stage, []).append(text)

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
        await asyncio.sleep(0)
        self.calls.append(
            {
                "stage": stage,
                "images": len(images or []),
                "json_mode": json_mode,
                "messages": [m.content for m in messages],
                "context": context or {},
            }
        )
        if stage in self.fail_stages:
            raise LLMError(f"fake provider failure for stage {stage}", code="fake_failure")

        if self.scripted.get(stage):
            text = self.scripted[stage].pop(0)
        elif stage == "vision_read":
            text = json.dumps(
                {
                    "text": "Fake vision transcription of the page.",
                    "legibility": 0.95,
                    "language": "en",
                    "notes": "",
                }
            )
        elif stage == "vision_repair":
            text = json.dumps(
                {
                    "text": "Fake repaired transcription.",
                    "legibility": 0.9,
                    "notes": "repaired",
                }
            )
        elif stage == "extract_answers":
            text = json.dumps(
                {
                    "answers": [
                        {"question_no": "Q1", "answer_text": "Fake answer Q1", "confidence": 0.9},
                        {"question_no": "Q2", "answer_text": "Fake answer Q2", "confidence": 0.9},
                    ]
                }
            )
        elif stage == "student_match":
            text = json.dumps({"student_external_id": None, "confidence": 0.0})
        elif stage == "score_rubric":
            text = json.dumps(
                {
                    "criteria": [
                        {
                            "criterion": "correctness",
                            "awarded": 2.0,
                            "rationale": "Fake rubric correctness.",
                            "confidence": 0.8,
                        }
                    ],
                    "feedback": "Fake feedback.",
                }
            )
        else:
            text = json.dumps({"ok": True})

        return LLMResult(
            text=text,
            provider=self.name,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            latency_ms=self.latency_ms,
        )


class FailingProvider(FakeProvider):
    """Fails every call; used to test job failure and repair paths."""

    def __init__(self, model: str = "fake-failing"):
        super().__init__(model)

    async def complete(self, *args, **kwargs):  # type: ignore[override]
        raise LLMError("simulated provider outage", code="simulated_outage", retryable=True)


def answer_from_prompt(messages: list[LLMMessage]) -> str | None:
    """Helper for tests: pull the student answer marker out of a scoring prompt."""
    pattern = re.compile(r"STUDENT_ANSWER:\s*(.*)", re.S)
    for message in messages:
        match = pattern.search(message.content)
        if match:
            return match.group(1).strip()
    return None
