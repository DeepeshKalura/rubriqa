"""Router registry. Only the entry-evaluation routes are wired.

The exam-mode routers (and the Google OAuth + Forms connector that only
served them) were removed in ADR-0001.
"""
from __future__ import annotations

from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    from app.routers import (
        dashboard,
        entries,
        files,
        sandbox,
        settings,
    )

    for module in (
        dashboard,    # /dashboard
        entries,      # /programs, /entries, /programs/{id}/export.csv
        sandbox,      # /sandbox  (2-way OCR comparison tool)
        settings,     # /settings (runtime feature flags)
        files,        # /files/*   (image serving for the sandbox)
    ):
        app.include_router(module.router)
