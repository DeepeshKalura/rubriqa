"""Settings: runtime feature flags (DB overrides of .env)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.models import LlmCall
from app.routers.common import render
from app.services.app_settings import OVERRIDABLE_KEYS, get_overrides, set_setting

router = APIRouter(tags=["settings"])


@router.get("/settings")
async def settings_page(request: Request, session: AsyncSession = Depends(get_session)):
    env = get_settings()
    overrides = await get_overrides(session)
    llm_usage = (
        await session.execute(
            select(LlmCall).order_by(LlmCall.id.desc()).limit(20)
        )
    ).scalars().all()
    llm_totals = await _llm_totals(session)
    return render(
        request,
        "settings.html",
        {
            "active": "settings",
            "env": env,
            "overrides": overrides,
            "keys": sorted(OVERRIDABLE_KEYS),
            "llm_usage": llm_usage,
            "llm_totals": llm_totals,
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
    from app.models import Setting

    row = await session.get(Setting, key)
    if row is not None:
        await session.delete(row)
        await session.commit()
    return RedirectResponse("/settings?cleared=1", status_code=303)


async def _llm_totals(session: AsyncSession):
    from sqlalchemy import func

    result = (
        await session.execute(
            select(
                func.count(LlmCall.id),
                func.coalesce(func.sum(LlmCall.total_tokens), 0),
                func.coalesce(func.avg(LlmCall.latency_ms), 0),
            )
        )
    ).one()
    return {"calls": result[0], "tokens": result[1], "avg_latency_ms": round(result[2] or 0, 1)}


def _mask(key: str) -> str:
    if not key:
        return "not set"
    return f"{key[:5]}…{key[-2:]} (len {len(key)})"
