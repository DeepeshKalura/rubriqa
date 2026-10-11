"""Unit tests for app.providers.llm.minimax.parse_json_lenient.

Guards: the parser must recover from (a) code fences, (b) surrounding prose,
(c) balanced sub-objects when the top-level dict is malformed, and most
importantly (d) truncated JSON where the model ran out of tokens before
closing braces — the bug that broke Path C on Marathi pages.
"""
from __future__ import annotations

import json

import pytest

from app.providers.llm.base import LLMError
from app.providers.llm.minimax import parse_json_lenient


def test_parses_plain_json() -> None:
    assert parse_json_lenient('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_strips_fenced_json() -> None:
    payload = '{"criteria": [{"name": "relevance", "awarded": 5}], "strengths": ["ok"]}'
    text = "```json\n" + payload + "\n```"
    parsed = parse_json_lenient(text)
    assert parsed["criteria"][0]["name"] == "relevance"
    assert parsed["strengths"] == ["ok"]


def test_strips_prose_around_json() -> None:
    text = 'Sure, here is the result: {"a": 1} -- done'
    assert parse_json_lenient(text) == {"a": 1}


def test_recovers_balanced_subobject_when_outer_truncated() -> None:
    """If the model starts a top-level dict then opens a `criteria` array
    containing a complete sub-object but never closes the outer braces, the
    balancer must synthesize the closers and return the whole dict — not
    raise, not silently return a sub-object."""
    text = '{"criteria": [{"name": "relevance", "awarded": 7, "max_score": 10}]'
    parsed = parse_json_lenient(text)
    assert parsed["criteria"][0]["name"] == "relevance"
    assert parsed["criteria"][0]["awarded"] == 7


def test_recovers_truncated_top_level_dict() -> None:
    """The Marathi page-02 case: model ran out of output tokens mid-string,
    never closed the top-level dict. The balancer must synthesize the
    missing closers and return a valid dict."""
    text = (
        '```json\n'
        '{\n'
        '  "criteria": [\n'
        '    {"name": "relevance", "awarded": 5, "max_score": 10, '
        '"rationale": "प्रस्ताव स्मार्ट शेती विषयाशी संबंधित आहे"},\n'
        '    {"name": "feasibility", "awarded": 4, "max_score": 10, '
        '"rationale": "अंमलबजावणी शक्य आहे"},\n'
        '    {"name": "originality", "awarded": 6, "max_score": 10, '
        '"rationale": "मौलिक आहे"},\n'
        '    {"name": "ai_understanding", "awarded": 7, "max_score": 10, '
        '"rationale": "AI वापर स्पष्ट आहे"},\n'
        '    {"name": "communication", "awarded": 3, "max_score": 10, '
        '"rationale": "लेखनात काही चुका आहेत"}\n'
        '  ],\n'
        '  "strengths": ["on-theme", "uses Devanagari well"],\n'
        '  "gaps": ["no quantifier"],\n'
        '  "observations": "विद्यार्थ्याने स्मार्ट शेतीवर चांगला प्रस्ताव'
    )
    parsed = parse_json_lenient(text)
    assert len(parsed["criteria"]) == 5
    assert parsed["criteria"][0]["name"] == "relevance"
    assert parsed["strengths"] == ["on-theme", "uses Devanagari well"]


def test_recovers_truncated_mid_string() -> None:
    """Worst case: model truncated right inside a string value, no closing `"`."""
    text = '{"name": "relevance", "note": "partial obser'
    parsed = parse_json_lenient(text)
    assert parsed["name"] == "relevance"
    assert "note" in parsed  # the synthesized closer keeps it


def test_does_not_misread_braces_inside_strings() -> None:
    """A `}` character inside a JSON string value must not be treated as the
    end of the object. (This was the failure mode of the old rfind approach.)"""
    text = '{"note": "score is 0} and the rubric is half-done", "ok": true}'
    parsed = parse_json_lenient(text)
    assert parsed["note"] == "score is 0} and the rubric is half-done"
    assert parsed["ok"] is True


def test_raises_on_empty() -> None:
    with pytest.raises(LLMError):
        parse_json_lenient("")


def test_raises_on_unrecoverable() -> None:
    with pytest.raises(LLMError):
        parse_json_lenient("no JSON anywhere at all")
