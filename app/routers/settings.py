"""Settings: runtime feature flags (DB overrides of `.env`)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import Setting
from app.routers.common import render
from app.services.app_settings import OVERRIDABLE_KEYS, get_overrides, set_setting

router = APIRouter(tags=["settings"])


@router.get("/settings")
async def settings_page(request: Request, session: AsyncSession = Depends(get_session)):
    env = get_settings()
    overrides = await get_overrides(session)
    return render(
        request,
        "settings.html",
        {
            "active": "settings",
            "env": env,
            "overrides": overrides,
            "keys": sorted(OVERRIDABLE_KEYS),
            "api_key_set": bool(env.api_key),
            "api_key_hint": _mask(env.api_key),
        },
    )


@router.post("/settings")
async def save_settings(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    form = await request.form()
    for key in OVERRIDABLE_KEYS:
        if key not in form:
            continue
        value = str(form.get(key) or "")
        if key in ("hybrid_accept_confidence", "hybrid_escalate_below"):
            value = value or "0.6"
        await set_setting(session, key, value)
    await session.commit()
    return RedirectResponse("/settings?saved=1", status_code=303)


@router.post("/settings/{key}/clear")
async def clear_setting(key: str, session: AsyncSession = Depends(get_session)):
    row = await session.get(Setting, key)
    if row is not None:
        await session.delete(row)
        await session.commit()
    return RedirectResponse("/settings?cleared=1", status_code=303)


def _mask(key: str) -> str:
    if not key:
        return "not set"
    return f"{key[:5]}…{key[-2:]} (len {len(key)})"
