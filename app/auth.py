"""Auth helpers. Gated by the `auth_enabled` flag (see roadmap item 2).

When the flag is off, the rest of the app behaves exactly as before. When it
is on, `SessionMiddleware` is registered, every existing router is wrapped
with `Depends(require_user)`, and the `/auth/login` + `/auth/logout` +
`/auth/me` routes light up.

Storage: signed cookies via Starlette `SessionMiddleware`. No server-side
session table. bcrypt for password hashing (no passlib).
"""
from __future__ import annotations

import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.logging import get_logger
from app.models import User

logger = get_logger("entry_eval.auth")

SESSION_USER_KEY = "user_id"
_DEV_SECRET = "dev-secret"


def session_secret_for(settings: Settings) -> str:
    """Pick a signing secret.

    Order: `session_secret` env → `llm_api_key` env → `dev-secret`.
    The fallback is intentionally weak so a misconfigured prod boots but is
    noisy in the log; a real prod must set `SESSION_SECRET`.
    """
    explicit = (settings.session_secret or "").strip()
    if explicit:
        return explicit
    api_key = (settings.api_key or "").strip()
    if api_key:
        return api_key
    return _DEV_SECRET


def warn_if_dev_secret(settings: Settings) -> None:
    """Log a one-shot warning when the dev secret is in use under real auth."""
    if not settings.auth_enabled:
        return
    if (settings.session_secret or "").strip():
        return
    if (settings.api_key or "").strip():
        return
    logger.warning(
        "auth_using_dev_secret",
        detail="set SESSION_SECRET in .env when AUTH_ENABLED=true in production",
    )


def hash_password(plain: str) -> str:
    """bcrypt hash, stored as a UTF-8 string (the framework's default prefix).

    bcrypt silently truncates inputs above 72 bytes; we surface that as a
    ValueError so the caller knows to shorten the password rather than
    shipping a stored hash that can never be verified against the original.
    """
    if not plain:
        raise ValueError("password must not be empty")
    encoded = plain.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("password exceeds bcrypt's 72-byte input limit")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


async def _user_from_session(
    request: Request, session: AsyncSession
) -> User | None:
    # SessionMiddleware is only installed when auth_enabled=True. When the
    # flag is off, request.session is not available and we always return None.
    if "session" not in request.scope:
        return None
    user_id = request.session.get(SESSION_USER_KEY)
    if user_id is None:
        return None
    user = await session.get(User, int(user_id))
    if user is None or not user.is_active:
        return None
    return user


async def current_user(
    request: Request, session: AsyncSession = Depends(get_session)
) -> User | None:
    """Convenience helper for routes/templates.

    Returns the active user from the session, or None. Never raises — the
    hard check is `require_user`.
    """
    return await _user_from_session(request, session)


async def require_user(
    request: Request, session: AsyncSession = Depends(get_session)
) -> User:
    """Router-level dependency. 403s when no active user is in the session.

    Only attached to routers when `auth_enabled=True` (see
    `app/routers/__init__.py`). When the flag is off, this dependency is
    never wired in, so every existing route is unaffected.
    """
    user = await _user_from_session(request, session)
    if user is None:
        raise HTTPException(status_code=403, detail="authentication required")
    return user


async def seed_admin_if_configured(session: AsyncSession) -> User | None:
    """Bootstrap the admin row from env if `auth_enabled=True`.

    Idempotent: if a user with `auth_admin_user` already exists, the
    password hash is overwritten so the env is the source of truth.
    Returns the admin row, or None when auth is off / no hash is set.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None
    if not settings.auth_admin_password_hash:
        return None
    from sqlalchemy import select

    admin = (
        (
            await session.execute(
                select(User).where(User.username == settings.auth_admin_user)
            )
        )
        .scalars()
        .one_or_none()
    )
    if admin is None:
        admin = User(
            username=settings.auth_admin_user,
            password_hash=settings.auth_admin_password_hash,
            is_active=True,
        )
        session.add(admin)
        await session.flush()
    else:
        admin.password_hash = settings.auth_admin_password_hash
        admin.is_active = True
        await session.flush()
    return admin