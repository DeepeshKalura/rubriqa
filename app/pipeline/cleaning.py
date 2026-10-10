"""Cleaning pipeline for OCR'd entries.

Takes an OCR string and produces a normalised version while the original is kept
in `entry_ocr.original_text`. Two modes:

  * "rules" — deterministic normalisation (strip control characters, normalise
    whitespace, remove obvious OCR garble like `|`, page headers/footers, repeated
    broken sentences). Always runs. Cost-free.

  * "llm" — applies the rules pass first, then asks the LLM to rewrite the
    cleaned text into a normalised form that matches the brief (situation 50-100
    words, proposed solution 100-150 words). Tokens are recorded so the final
    export contains the same Input/Output/Total Token columns the client sheet
    expects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings
from app.logging import get_logger
from app.schemas import LLMMessage

logger = get_logger("entry_eval.cleaning")


@dataclass
class CleanResult:
    cleaned_text: str
    notes: list[str]
    rule_changes: int


# ---------------------------------------------------------------------------
# rule pass — deterministic, always applied
# ---------------------------------------------------------------------------
_CONTROL_CHARS = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F]")
_MULTI_BLANKS = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_LONE_BARS = re.compile(r"^\s*\|+\s*$", re.MULTILINE)
_PAGE_MARKER = re.compile(r"--- page \d+\.?\s*(?:of\s+\S+)?\s*---", re.IGNORECASE)
_BROKEN_LINE = re.compile(r"\b(\w{1,2})(?:\s+\1\b){3,}", re.IGNORECASE)
_NON_PRINTABLE_RATIO = re.compile(r"[^A-Za-z\u0900-\u097F\s.,;:?!()'\"]")


def _rule_clean(text: str) -> CleanResult:
    notes: list[str] = []
    original = text or ""

    before = original
    s = _CONTROL_CHARS.sub(" ", original)
    if s != before:
        notes.append("stripped_control_chars")
    before = s
    s = _LONE_BARS.sub("", s)
    if s != before:
        notes.append("removed_lone_bars")
    before = s
    s = _PAGE_MARKER.sub(" ", s)
    if s != before:
        notes.append("stripped_page_markers")
    before = s
    s = _TRAILING_WS.sub("\n", s)
    if s != before:
        notes.append("trimmed_trailing_whitespace")
    before = s
    s = re.sub(r"[ \t]{2,}", " ", s)
    if s != before:
        notes.append("collapsed_runs_of_spaces")
    before = s
    s = _MULTI_BLANKS.sub("\n\n", s)
    if s != before:
        notes.append("collapsed_multiple_blank_lines")
    s = s.strip()
    # detect obviously broken lines (a 1-2 letter token repeating) — probably
    # OCR artefacts of the same word misread multiple times
    before = s
    s = _BROKEN_LINE.sub(r"\1", s)
    if s != before:
        notes.append("collapsed_repeated_short_tokens")

    changes = 0 if s == (original or "") else 1
    return CleanResult(cleaned_text=s, notes=notes, rule_changes=changes)


# ---------------------------------------------------------------------------
# LLM pass — optional, prompt-driven rewriting
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an editor for handwritten student project idea submissions
in Hindi, English, or Marathi. The student described a real-life situation and a
proposed AI solution. Your job is to lightly clean up OCR artefacts (broken
words, repeated tokens) while preserving the student's own words, language, and
tone.

Rules:
- Do NOT change meaning. Do NOT add facts.
- Do NOT translate.
- Fix spacing, join hyphenated line breaks, normalise punctuation.
- Keep the language the student used (Hindi, English, Marathi, mixed).
- Return JSON only: {"text": "<cleaned>", "notes": "<very short note>"}."""


async def clean_with_llm(
    text: str,
    *,
    settings: Settings,
    llm,
    language_hint: str | None = None,
    section: str = "situation",
) -> tuple[str, str, int, int, int]:
    """Returns (cleaned_text, notes, prompt_tokens, completion_tokens, total_tokens)."""
    user = (
        f"Section: {section}\n"
        f"Language: {language_hint or 'unknown'}\n"
        "Clean the text below. Return JSON only.\n\n"
        f"---BEGIN---\n{text}\n---END---"
    )
    result = await llm.complete(
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ],
        json_mode=True,
        max_tokens=getattr(settings, "minimax_max_output_tokens", 4096),
        stage="entry_clean",
        temperature=0.0,
    )
    import json as _json_mod
    data_dict = _json_mod.loads(result.text)
    cleaned = (data_dict.get("text") or text).strip()
    return cleaned, (data_dict.get("notes") or ""), result.prompt_tokens or 0, result.completion_tokens or 0, result.total_tokens or (result.prompt_tokens or 0) + (result.completion_tokens or 0)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def rule_clean(text: str) -> CleanResult:
    return _rule_clean(text)


async def clean_entry_text(
    text: str,
    *,
    settings: Settings,
    llm,
    language_hint: str | None,
    section: str = "situation",
    mode: str = "rules",
) -> CleanResult:
    base = _rule_clean(text)
    if mode == "rules" or not text.strip():
        return base
    try:
        cleaned, notes, in_t, out_t, total_t = await clean_with_llm(
            base.cleaned_text,
            settings=settings,
            llm=llm,
            language_hint=language_hint,
            section=section,
        )
        # merge notes
        merged = base.notes + [f"llm:{notes}"] if notes else base.notes
        return CleanResult(cleaned_text=cleaned, notes=merged, rule_changes=base.rule_changes + 1)
    except Exception as exc:  # noqa: BLE001 - cleaning failures fall back to rules-only
        logger.warning("cleaning_llm_failed", error=str(exc)[:200], section=section)
        return base