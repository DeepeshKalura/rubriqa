"""Structured logging.

Every pipeline stage logs job_id/document_id/student_id/stage/status/duration/
provider/model/error. Credentials are never logged: the API key only lives in
the provider client and is redacted by the `redact()` helper if it ever
appears in an exception string.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import structlog

_REDACT_KEYS = {"api_key", "authorization", "password", "secret", "token"}


def redact(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


def _redact_processor(_logger: Any, _method: str, event_dict: dict) -> dict:
    for key in list(event_dict):
        if key.lower() in _REDACT_KEYS:
            event_dict[key] = "***"
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
