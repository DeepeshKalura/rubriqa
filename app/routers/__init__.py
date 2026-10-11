"""Router registry. Only the entry-evaluation routes are wired.

The exam-mode routers (and the Google OAuth + Forms connector that only
served them) were removed in ADR-0001.

When `auth_enabled=True`, every router below is wrapped with
`Depends(require_user)` so unauthenticated requests get a 403 before the
endpoint body runs. Login/logout/me are exempt so the user can actually sign
in. When the flag is off (default), the dependency is not applied and
every route behaves exactly as before.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI

from app.auth import require_user
from app.config import get_settings


def register_routers(app: FastAPI) -> None:
    from app.routers import (
        auth,
        dashboard,
        entries,
        files,
        sandbox,
        settings as settings_router,
    )

    settings = get_settings()
    # Auth is opt-in. When on, every router below is wrapped with the
    # require_user dependency. The auth router itself is NOT wrapped so
    # /auth/login and /auth/me stay reachable without a session.
    if settings.auth_enabled:
        protected_dep = [Depends(require_user)]
        auth_dep = []
    else:
        protected_dep = []
        auth_dep = []

    app.include_router(auth.router, dependencies=auth_dep)

    for module in (
        dashboard,        # /dashboard
        entries,          # /programs, /entries, /programs/{id}/export.csv
        sandbox,          # /sandbox  (3-way OCR comparison tool)
        settings_router,  # /settings (runtime feature flags)
        files,            # /files/*   (image serving for the sandbox)
    ):
        app.include_router(module.router, dependencies=protected_dep)