"""Dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.routers.common import render
from app.services.stats import dashboard_stats, recent_entries

router = APIRouter(tags=["dashboard"])


@router.get("/")
async def dashboard(request: Request, session: AsyncSession = Depends(get_session)):
    data = await dashboard_stats(session)
    recent = await recent_entries(session, limit=8)
    return render(
        request,
        "dashboard.html",
        {"active": "dashboard", "stats": data, "recent_entries": recent},
    )


@router.get("/api/stats")
async def api_stats(session: AsyncSession = Depends(get_session)):
    return await dashboard_stats(session)
