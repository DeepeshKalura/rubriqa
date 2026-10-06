"""Dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.routers.common import render
from app.services import stats

router = APIRouter(tags=["dashboard"])


@router.get("/")
async def dashboard(request: Request, session: AsyncSession = Depends(get_session)):
    data = await stats.dashboard_stats(session)
    jobs = await stats.recent_jobs(session, limit=8)
    return render(
        request,
        "dashboard.html",
        {"active": "dashboard", "stats": data, "jobs": jobs},
    )


@router.get("/api/stats")
async def api_stats(session: AsyncSession = Depends(get_session)):
    return await stats.dashboard_stats(session)
