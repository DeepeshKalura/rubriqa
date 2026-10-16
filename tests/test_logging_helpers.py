"""Logging helpers.

`redact()` masks secrets. `brief()` truncates diagnostics. They are not
interchangeable, and using the wrong one cost real debugging time: a
`httpx.ReadTimeout` was logged as `Read...`, so a transport failure looked like
an unexplained refusal.
"""
from __future__ import annotations

import pytest

from app.logging import brief, redact


def test_redact_hides_a_secret_body():
    # A realistic shape and length, but a fabricated value. GitHub push
    # protection blocks well-formed Google credentials even in a fixture,
    # and `redact()` is length-based, so a real token buys no extra
    # coverage here.
    masked = redact("AQ.Ab8RN6Kq-DO-NOT-LOG-ME-not-a-real-token")
    assert "Ab8RN6Kq" not in masked
    assert masked.startswith("AQ.A") and masked.endswith("en")
    assert "DO-NOT-LOG-ME" not in masked


def test_redact_on_short_values_is_fully_masked():
    assert redact("abc") == "***"
    assert redact("") == ""


def test_redact_never_returns_the_middle_of_the_value():
    secret = "sk-proj-verysecretvalue-1234567890"
    assert "verysecret" not in redact(secret)
    assert "1234567890" not in redact(secret)


def test_brief_keeps_the_start_of_the_message_readable():
    """The regression: this must NOT collapse to `Read...`."""
    message = (
        "httpx.ReadTimeout: timed out while reading the response after 120.0s "
        "for url https://generativelanguage.googleapis.com/v1beta/models/"
    )
    out = brief(message)
    assert out.startswith("httpx.ReadTimeout:")
    assert "timed out" in out
    assert len(out) > 30


def test_brief_respects_the_limit_and_marks_truncation():
    out = brief("x " * 500, limit=50)
    assert len(out) <= 53
    assert out.endswith("...")


def test_brief_collapses_newlines_so_logs_stay_one_line():
    out = brief("line one\nline two\n\nline three")
    assert "\n" not in out
    assert "line one line two line three" == out


def test_brief_handles_empty():
    assert brief(None) == ""
    assert brief("") == ""

# --- automatic redaction: the processor is the real guard -------------------
#
# `redact()` is only called by hand. `_redact_processor` runs on every log
# line, so it is the thing that actually keeps keys out of logs.

from app.logging import _redact_processor, is_secret_key


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "minimax_api_key",
        "google_api_key",
        "llm_api_key",
        "x-goog-api-key",
        "apiKey",
        "API_KEY",
        "Authorization",
        "access_token",
        "auth_token",
        "refresh_token",
        "session_secret",
        "admin_password_hash",
        "credential",
        "bearer",
    ],
)
def test_secret_bearing_names_are_recognised(key):
    """Regression: the old guard was exact-match, so most of these - including
    every API-key name this repo actually uses - passed straight through."""
    assert is_secret_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "page",
        "stage",
        "attempt",
        "provider",
        "model",
        "latency_ms",
        "duration_ms",
        "job_id",
        "status",
        # Token COUNTS are cost data, not secrets. Masking them would silently
        # corrupt the cost report while looking like hardening.
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens",
        "input_tokens",
        "cached_tokens",
        "error",
    ],
)
def test_ordinary_fields_are_not_masked(key):
    assert not is_secret_key(key)


def test_processor_masks_the_repo_own_key_names():
    event = {
        "stage": "vision_read",
        "minimax_api_key": "AQ.Ab8RN6Kq-DO-NOT-LOG-ME-fake",
        "google_api_key": "AIzaSy-DO-NOT-LOG-ME",
        "session_secret": "s3cr3t-value-here",
    }
    out = _redact_processor(None, None, dict(event))
    assert out["minimax_api_key"] == "***"
    assert out["google_api_key"] == "***"
    assert out["session_secret"] == "***"
    assert out["stage"] == "vision_read"


def test_processor_walks_nested_dictionaries():
    """A provider passes `context={...}`, so a key one level down is still a
    key in a log line."""
    out = _redact_processor(
        None,
        None,
        {
            "stage": "llm",
            "context": {"api_key": "AKIA-SHOULD-NOT-APPEAR", "entry_id": 42},
        },
    )
    assert out["context"]["api_key"] == "***"
    assert out["context"]["entry_id"] == 42


def test_processor_walks_lists_of_dicts():
    out = _redact_processor(
        None,
        None,
        {"items": [{"google_api_key": "leak-me"}, {"page": "page-01"}]},
    )
    assert out["items"][0]["google_api_key"] == "***"
    assert out["items"][1]["page"] == "page-01"


def test_processor_preserves_token_counts_for_the_cost_report():
    out = _redact_processor(
        None,
        None,
        {"prompt_tokens": 5100, "completion_tokens": 2700, "total_tokens": 7800},
    )
    assert out["prompt_tokens"] == 5100
    assert out["completion_tokens"] == 2700
    assert out["total_tokens"] == 7800


def test_key_matching_normalises_dashes_and_case():
    for variant in ("X-Goog-Api-Key", "x_goog_api_key", "X_GOOG_API_KEY"):
        assert is_secret_key(variant)
