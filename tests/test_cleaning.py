"""Cleaning pipeline tests.

Rules pass is deterministic and exercised here. The LLM pass is exercised
in the entry-scoring test (where the fake provider queues a canned response).
"""
from __future__ import annotations

from app.pipeline.cleaning import rule_clean


def test_rule_clean_strips_control_chars() -> None:
    raw = "hello\x00world\x07"
    out = rule_clean(raw)
    assert "\x00" not in out.cleaned_text
    assert "hello" in out.cleaned_text and "world" in out.cleaned_text


def test_rule_clean_collapses_lone_bars() -> None:
    raw = "text\n|\nmore text\n|"
    out = rule_clean(raw)
    assert "|" not in out.cleaned_text.splitlines() or out.cleaned_text.count("|") < raw.count("|")


def test_rule_clean_collapsed_blank_lines() -> None:
    raw = "a\n\n\n\n\nb"
    out = rule_clean(raw)
    assert "\n\n\n" not in out.cleaned_text


def test_rule_clean_strip_page_markers() -> None:
    raw = "--- page 1 of foo ---\nBody text"
    out = rule_clean(raw)
    assert "page 1 of foo" not in out.cleaned_text
    assert "Body text" in out.cleaned_text


def test_rule_clean_collapsed_repeated_short_tokens() -> None:
    raw = "ma ma ma ma the student"
    out = rule_clean(raw)
    # the rule collapses 3+ repeated short tokens
    assert "ma ma ma ma" not in out.cleaned_text


def test_rule_clean_empty_input_is_safe() -> None:
    out = rule_clean("")
    assert out.cleaned_text == ""
    assert out.notes == [] or isinstance(out.notes, list)


def test_rule_clean_only_whitespace_is_empty() -> None:
    out = rule_clean("\n\n   \n")
    assert out.cleaned_text.strip() == ""


def test_rule_clean_preserves_unicode() -> None:
    raw = "कक्षा ७\nछात्र"
    out = rule_clean(raw)
    assert "कक्षा" in out.cleaned_text
    assert "छात्र" in out.cleaned_text