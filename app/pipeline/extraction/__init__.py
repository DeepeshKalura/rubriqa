"""Answer extraction: student matching + question/answer segmentation.

Two paths:
  * LLM extraction (default): the exam's questions are given to the model and
    it returns answer spans per question with confidence; robust to messy
    handwriting transcripts.
  * Heuristic extraction (fallback / offline): regex on question markers like
    "Q1", "1.", "प्रश्न 1", "उत्तर" lines.
Both return the same ExtractedAnswer shape so scoring never knows the
difference.
"""
from __future__ import annotations

import json
import re

from app.logging import get_logger
from app.providers.llm.minimax import parse_json_lenient
from app.schemas import ExtractedAnswer, LLMMessage

logger = get_logger("exam_eval.extraction")

EXTRACT_SYSTEM_PROMPT = """You segment a student's exam transcript into answers.
You are given the exam question list. Return one answer object per question in the list.
Rules:
- Copy the student's answer text EXACTLY as it appears in the transcript. Do not improve,
  translate or complete it.
- If the student did not answer a question, use an empty string and confidence 0.
- Confidence is your certainty that this text is really the student's answer to that question.
Return JSON only:
{"answers": [{"question_no": "Q1", "answer_text": "...", "confidence": 0.0-1.0}]}"""

QUESTION_MARKER_PATTERNS = [
    re.compile(r"^\s*(?:Q(?:uestion)?\s*[.:-]?\s*|प्रश्न\s*[.:-]?\s*|प्र\.?\s*)(\d+)\b[.:)\-–]?", re.I),
    re.compile(r"^\s*(\d{1,2})\s*[.):\-–]\s+", re.I),
]


def build_extraction_prompt(questions: list[dict], transcript: str) -> str:
    question_lines = "\n".join(
        f"- {q['q_no']}: {q['text']}" for q in questions
    )
    return (
        "EXAM QUESTIONS:\n"
        f"{question_lines}\n\n"
        "STUDENT TRANSCRIPT (pages in order):\n"
        "-----\n"
        f"{transcript[:12000]}\n"
        "-----\n"
        'Return {"answers": [...]} now.'
    )


class LLMExtractor:
    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    async def extract(
        self, transcript: str, questions: list[dict], *, context: dict | None = None
    ) -> list[ExtractedAnswer]:
        if not transcript.strip() or not questions:
            return []
        try:
            result = await self.llm.complete(
                [
                    LLMMessage(role="system", content=EXTRACT_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=build_extraction_prompt(questions, transcript)),
                ],
                json_mode=True,
                stage="extract_answers",
                context=context,
            )
            data = parse_json_lenient(result.text)
        except Exception as exc:  # noqa: BLE001 - fall back to heuristic below
            logger.warning("llm_extraction_failed", error=str(exc)[:300])
            return HeuristicExtractor().extract(transcript, questions)

        by_no = {q["q_no"]: q for q in questions}
        answers: list[ExtractedAnswer] = []
        for item in data.get("answers", []):
            q_no = str(item.get("question_no") or "").strip()
            question = by_no.get(q_no)
            if question is None:
                # tolerate "1" vs "Q1"
                normalized = q_no.lstrip("Qq").strip(" .:-")
                question = next(
                    (q for q in questions if q["q_no"].lstrip("Qq").strip() == normalized),
                    None,
                )
            if question is None:
                continue
            answers.append(
                ExtractedAnswer(
                    question_id=question["id"],
                    question_no=question["q_no"],
                    answer_text=str(item.get("answer_text") or "").strip(),
                    confidence=_safe_float(item.get("confidence")),
                    method=self.name,
                )
            )
        return answers


class HeuristicExtractor:
    """Offline fallback: split the transcript on question markers."""

    name = "regex"

    def extract(self, transcript: str, questions: list[dict]) -> list[ExtractedAnswer]:
        segments = self._split(transcript)
        answers: list[ExtractedAnswer] = []
        for question in questions:
            key = question["q_no"].lstrip("Qq").strip()
            text = segments.get(key, "").strip()
            answers.append(
                ExtractedAnswer(
                    question_id=question["id"],
                    question_no=question["q_no"],
                    answer_text=text,
                    confidence=0.4 if text else 0.0,
                    method=self.name,
                )
            )
        return answers

    def _split(self, transcript: str) -> dict[str, str]:
        lines = transcript.splitlines()
        segments: dict[str, list[str]] = {}
        current: str | None = None
        for line in lines:
            matched: str | None = None
            remainder = ""
            for pattern in QUESTION_MARKER_PATTERNS:
                match = pattern.match(line)
                if match:
                    matched = match.group(1)
                    remainder = line[match.end():].strip()
                    break
            if matched:
                current = matched
                # first occurrence wins; later pages are merged in
                segments.setdefault(current, [])
                if remainder:
                    segments[current].append(remainder)
            elif current:
                segments[current].append(line)
        return {key: "\n".join(value) for key, value in segments.items()}


def _safe_float(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def answers_to_json(answers: list[ExtractedAnswer]) -> str:
    return json.dumps([a.model_dump() for a in answers], ensure_ascii=False)
