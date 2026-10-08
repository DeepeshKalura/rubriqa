"""Runtime settings stored in the DB (feature flags overridable from the UI).

Priority: DB setting > .env value > code default.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import Setting

OVERRIDABLE_KEYS = {
    "reader_mode": "reader_mode",
    "llm_provider": "llm_provider",
    "scoring_strategy": "scoring_strategy",
    "scoring_enable_feedback": "scoring_enable_feedback",
    "repair_enabled": "repair_enabled",
    "hybrid_reader_chain": "hybrid_reader_chain",
    "google_forms_enabled": "google_forms_enabled",
    "google_forms_provider": "google_forms_provider",
    "hybrid_accept_confidence": "hybrid_accept_confidence",
    "hybrid_escalate_below": "hybrid_escalate_below",
    # entry-evaluation mode flags
    "eval_mode": "eval_mode",
    "eval_batch_size": "eval_batch_size",
    "cleaning_mode": "cleaning_mode",
    "google_drive_enabled": "google_drive_enabled",
    "google_drive_provider": "google_drive_provider",
    "sandbox_default_provider": "sandbox_default_provider",
}


async def get_overrides(session: AsyncSession) -> dict[str, str]:
    rows = (await session.execute(select(Setting))).scalars().all()
    return {row.key: row.value for row in rows if row.key in OVERRIDABLE_KEYS}


async def get_effective_settings(session: AsyncSession) -> Settings:
    """Return a copy of the env settings with DB overrides applied."""
    overrides = await get_overrides(session)
    if not overrides:
        return get_settings()
    base = get_settings()
    values = {OVERRIDABLE_KEYS[key]: _coerce(base, field, value) for key, value in overrides.items()}
    merged = base.model_copy(update=values)
    return merged


def _coerce(base: Settings, field: str, raw: str):
    current = getattr(base, field)
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            return current
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            return current
    return raw


async def set_setting(session: AsyncSession, key: str, value: str) -> None:
    if key not in OVERRIDABLE_KEYS:
        raise ValueError(f"Setting '{key}' is not overridable")
    row = await session.get(Setting, key)
    if row is None:
        session.add(Setting(key=key, value=value))
    else:
        row.value = value
    await session.flush()
