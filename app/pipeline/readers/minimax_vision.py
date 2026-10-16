"""MiniMax vision reader: direct page reading and OCR repair.

Prompts are deliberately constrained: the model must transcribe only what is
visible and must not invent content. Every repair keeps the original OCR text.
"""
from __future__ import annotations

from app.config import get_settings
from app.logging import get_logger
from app.pipeline.preprocessing import load_image, to_jpeg_bytes
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.providers.llm import LLMError
from app.providers.llm.minimax import parse_json_lenient
from app.schemas import CanonicalPage, LLMMessage, ReadingResult, ReadingStatus

logger = get_logger("entry_eval.reader.minimax")

READ_SYSTEM_PROMPT = """You transcribe student exam pages (handwritten or printed) in Hindi, English, or mixed.
Rules:
- Transcribe ONLY text that is actually visible in the image.
- Do not answer questions, do not complete sentences, do not guess.
- Mark genuinely unreadable words as [illegible]. Never replace them with plausible alternatives.
- Preserve question numbers, line breaks and the original language mix.
Return JSON only: {"text": "...", "legibility": 0.0-1.0, "language": "hi|en|mixed", "notes": "..."}"""

REPAIR_SYSTEM_PROMPT = """You inspect one exam page image and an OCR transcript of it.
Your task: correct only clear OCR mistakes using text that is actually visible in the image.
Rules:
- Never invent or complete an answer that is not visible.
- If a word is unreadable, write [illegible].
- Output the full corrected transcript, preserving line breaks.
Return JSON only: {"text": "...", "legibility": 0.0-1.0, "notes": "brief reason for changes"}"""


class MiniMaxVisionReader:
    name = "minimax_vision"
    provider = "minimax"

    def __init__(self, llm, settings=None):
        self.llm = llm
        self.settings = settings or get_settings()
        self.model = self.settings.minimax_vision_model

    def _image_bytes(self, context: ReaderContext) -> bytes:
        if context.page_image_path is None:
            raise ReaderError("Vision reader needs a page image", code="missing_page_image")
        image = load_image(context.page_image_path)
        return to_jpeg_bytes(image, max_side=self.settings.vision_max_image_side)

    async def read(self, page: CanonicalPage, context: ReaderContext) -> ReadingResult:
        image_bytes = self._image_bytes(context)
        try:
            result = await self.llm.complete(
                [
                    LLMMessage(role="system", content=READ_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content="Transcribe this exam page. Respond with the JSON object only.",
                    ),
                ],
                images=[image_bytes],
                json_mode=True,
                stage="vision_read",
            )
        except LLMError as exc:
            raise ReaderError(str(exc), code=exc.code, retryable=exc.retryable) from exc

        data, parse_failed = _extract_payload(result.text)
        text = str(data.get("text") or "").strip()
        legibility = _clamp(float(data.get("legibility") or 0.0))
        warnings = []
        status = ReadingStatus.OK
        legibility_thresh = self.settings.low_confidence_threshold
        if parse_failed:
            # Model answered but not with valid JSON (usually truncation of long
            # transcripts). The raw transcript is still useful; keep it and mark
            # it low confidence instead of failing the page.
            warnings.append("unstructured_vision_output")
            status = ReadingStatus.LOW_CONFIDENCE
            legibility = min(legibility or 0.5, 0.5)
        if not text:
            status = ReadingStatus.EMPTY
            warnings.append("empty_vision_reading")
        elif legibility < legibility_thresh and "low_legibility" not in warnings:
            status = ReadingStatus.LOW_CONFIDENCE
            warnings.append("low_legibility")
        return ReadingResult(
            text=text,
            confidence=legibility,
            status=status,
            engine=self.name,
            provider=result.provider,
            model=result.model,
            mode=self.name,
            language=data.get("language"),
            warnings=warnings,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            raw={"notes": data.get("notes")},
        )

    async def repair(
        self, original: ReadingResult, context: ReaderContext, *, reason: str
    ) -> ReadingResult:
        image_bytes = self._image_bytes(context)
        user_prompt = (
            "OCR transcript to verify against the image:\n"
            f"-----\n{original.text}\n-----\n"
            "Return the corrected JSON now."
        )
        try:
            result = await self.llm.complete(
                [
                    LLMMessage(role="system", content=REPAIR_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_prompt),
                ],
                images=[image_bytes],
                json_mode=True,
                stage="vision_repair",
            )
        except LLMError as exc:
            raise ReaderError(str(exc), code=exc.code, retryable=exc.retryable) from exc

        data, parse_failed = _extract_payload(result.text)
        if parse_failed and data.get("text"):
            return ReadingResult(
                text=str(data["text"]).strip(),
                confidence=0.5,
                status=ReadingStatus.LOW_CONFIDENCE,
                engine=f"{self.name}_repair",
                provider=result.provider,
                model=result.model,
                mode="repair",
                warnings=["unstructured_vision_output", "repaired"],
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                raw={"reason": reason, "original_text": original.text},
            )
        text = str(data.get("text") or "").strip()
        legibility = _clamp(float(data.get("legibility") or 0.0))
        return ReadingResult(
            text=text,
            confidence=legibility,
            status=ReadingStatus.OK if text else ReadingStatus.EMPTY,
            engine=f"{self.name}_repair",
            provider=result.provider,
            model=result.model,
            mode="repair",
            warnings=["repaired"] if text else ["repair_empty"],
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            raw={"reason": reason, "notes": data.get("notes"), "original_text": original.text},
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_payload(text: str) -> tuple[dict, bool]:
    """Parse the model JSON; on failure recover the `text` field with a regex.

    Returns (payload, parse_failed). A recovered payload still carries the
    transcript so the page is not lost, but callers must treat it as
    low-confidence.
    """
    try:
        return parse_json_lenient(text), False
    except LLMError:
        import json
        import re

        match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if match:
            raw = match.group(1)
            try:
                recovered = json.loads(f'"{raw}"')
            except json.JSONDecodeError:
                recovered = raw.replace('\\n', '\n').replace('\\"', '"')
            return {"text": recovered, "legibility": 0.5}, True
        return {"text": text.strip()}, True
