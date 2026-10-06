"""Router registry. Each phase adds its routers here."""
from __future__ import annotations

from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    from app.routers import (
        classes,
        dashboard,
        exams,
        files,
        google,
        imports,
        jobs,
        results,
        review,
        settings,
        students,
        submissions,
    )

    for module in (
        dashboard,
        classes,
        students,
        exams,
        submissions,
        imports,
        jobs,
        review,
        results,
        settings,
        google,
        files,
    ):
        app.include_router(module.router)
