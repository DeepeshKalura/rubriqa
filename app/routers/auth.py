"""Auth routes: `/auth/login`, `/auth/logout`, `/auth/me`.

Always registered. When `auth_enabled=False`, login/logout silently no-op
(no flag, no session middleware, no users table) and `/auth/me` always
returns null.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    SESSION_USER_KEY,
    current_user,
    seed_admin_if_configured,
    verify_password,
)
from app.config import get_settings
from app.db import get_session
from app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login(
    request: Request,
    session: AsyncSession = Depends(get_session),
    username: str = Form(...),
    password: str = Form(...),
):
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(
            status_code=403,
            detail="AUTH_ENABLED is false; set AUTH_ENABLED=true in .env",
        )
    # Make sure the env-driven admin exists (idempotent).
    await seed_admin_if_configured(session)
    user = (
        (
            await session.execute(select(User).where(User.username == username))
        )
        .scalars()
        .one_or_none()
    )
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid username or password")
    request.session[SESSION_USER_KEY] = user.id
    await session.commit()
    return {"ok": True, "user": {"id": user.id, "username": user.username}}


@router.post("/logout")
async def logout(request: Request, session: AsyncSession = Depends(get_session)):
    request.session.clear()
    await session.commit()
    return {"ok": True}


@router.get("/me")
async def me(user: User | None = Depends(current_user)):
    if user is None:
        return {"user": None}
    return {"user": {"id": user.id, "username": user.username, "is_active": user.is_active}}