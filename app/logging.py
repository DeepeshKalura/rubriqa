"""Structured logging.

Every pipeline stage logs job_id/document_id/student_id/stage/status/duration/
provider/model/error. Credentials are never logged, and the automatic guard is
`_redact_processor`, which masks any event key that looks like a secret.

What is and is not protected, precisely:

- **Protected.** Any log field whose name carries a secret marker
  (`*api_key*`, `*secret*`, `*password*`, `authorization`, `*_token` but not a
  bare `token`). Matching is substring-based, because an exact-match set does
  not match this repo's own variable names: `minimax_api_key`,
  `google_api_key` and `session_secret` all passed straight through the old
  exact-match guard. Nested dict/list values are walked too.
- **NOT protected.** Secrets embedded in free-text strings, notably an
  exception message. There is no scrubber, on purpose: mangling free text to
  hunt for a key pattern produces unreadable diagnostics, which is how
  `redact()` was being misused on errors and cost an hour of debugging. The
  contract is instead that a provider must never put a key in a message, and
  `tests/test_providers_gemini.py::test_api_key_is_never_in_the_payload_or_url`
  pins it for Gemini. See docs/architecture.md for the residual risk.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

# Substrings that mark a field name as secret-bearing. Normalised keys are
# lowercased with "-" folded to "_" before matching.
#
# Deliberately NOT a bare "token": the usage hooks log `prompt_tokens`,
# `completion_tokens` and `total_tokens`, and the cost report is built from
# them. Masking those would silently corrupt cost accounting while looking
# like a security improvement. Match the specific token-bearing names instead.
_SECRET_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "secret",
    "password",
    "passwd",
    "credential",
    "access_token",
    "auth_token",
    "refresh_token",
    "bearer",
    "session_key",
)


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def is_secret_key(key: str) -> bool:
    """True if a log field name should be masked."""
    normalised = _normalise_key(key)
    return any(marker in normalised for marker in _SECRET_MARKERS)


def redact(value: str | None) -> str:
    """Mask a *secret* value. Use this for keys and credentials only.

    Do not use it for error messages: it truncates to `first4...last2`, which
    destroys the part of a traceback you actually need ("httpx.ReadTimeout:
    ..." became "Read..." and cost an hour of debugging). Use `brief()` for
    diagnostics.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


def brief(value: str | None, limit: int = 300) -> str:
    """Truncate a diagnostic for logging without mangling it.

    This does NOT scrub secrets - see the module docstring. It is deliberately
    a pure truncation so error text stays readable.
    """
    if not value:
        return ""
    text = " ".join(value.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _mask(value: Any) -> Any:
    """Recursively mask secret-bearing keys inside a log payload."""
    if isinstance(value, dict):
        return {k: ("***" if is_secret_key(str(k)) else _mask(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask(item) for item in value)
    return value


def _redact_processor(_logger: Any, _method: str, event_dict: dict) -> dict:
    for key in list(event_dict):
        if is_secret_key(str(key)):
            event_dict[key] = "***"
    # Fields are frequently nested (a provider passes a `context=` dict), and a
    # key buried one level down is still a key in a log line.
    for key, value in list(event_dict.items()):
        if isinstance(value, (dict, list, tuple)):
            event_dict[key] = _mask(value)
    return event_dict


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _redact_processor,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=False))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "entry_eval"):
    return structlog.get_logger(name)


@contextmanager
def timed(logger, stage: str, **fields: Any) -> Iterator[dict]:
    """Log a stage with duration_ms on exit, success or failure."""
    started = time.perf_counter()
    payload: dict[str, Any] = {"stage": stage, **fields}
    try:
        yield payload
    except Exception as exc:  # noqa: BLE001 - log then re-raise
        payload.update(
            status="error",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            error=str(exc)[:500],
        )
        logger.error("stage_failed", **payload)
        raise
    else:
        payload.update(
            status="ok", duration_ms=round((time.perf_counter() - started) * 1000, 1)
        )
        logger.info("stage_ok", **payload)
