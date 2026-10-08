"""Router registry. Each phase adds its routers here."""
from __future__ import annotations

from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    from app.routers import (
        classes,
        dashboard,
        entries,
        exams,
        files,
        google,
        imports,
        jobs,
        results,
        review,
        sandbox,
        settings,
        students,
        submissions,
    )

    # Phase 1: entry-evaluation mode is the primary flow now.
    # Exam-mode routers remain so the demo data + existing tests still work.
    for module in (
        dashboard,
        entries,      # /programs, /entries, /programs/{id}/export.csv
        sandbox,      # /sandbox  (2-way OCR comparison tool)
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