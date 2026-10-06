"""Lightweight signed OAuth state token.

We can't afford a session middleware just for Google OAuth, so we sign a
short-lived CSRF token with a server secret derived from APP_SECRET_KEY.
The token is set as a cookie on the /start redirect and verified in
/google/oauth/callback.
"""
from __future__ import annotations

import secrets
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from app.config import get_settings


def _signer() -> TimestampSigner:
    settings = get_settings()
    secret = (settings.app_secret_key or settings.api_key or "dev-fallback").encode()
    return TimestampSigner(secret, salt=b"google_oauth_state")


def make_state() -> str:
    return _signer().sign(secrets.token_urlsafe(16)).decode()


def verify_state(token: str, *, max_age_seconds: int = 600) -> Optional[str]:
    if not token:
        return None
    try:
        return _signer().unsign(token, max_age=max_age_seconds).decode()
    except (BadSignature, SignatureExpired):
        return None
