"""Small text hygiene helpers. Postgres TEXT rejects NUL bytes; OCR/vision
output occasionally contains them. Also normalizes whitespace for storage.
"""
from __future__ import annotations


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("\x00", "")


def sanitize_optional(text: str | None) -> str | None:
    if text is None:
        return None
    return sanitize_text(text)
