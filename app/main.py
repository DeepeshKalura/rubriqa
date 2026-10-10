"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.auth import session_secret_for, warn_if_dev_secret
from app.config import get_settings
from app.db import dispose_engine, get_engine
from app.logging import configure_logging, get_logger
from app.services.migrations import run_migrations

logger = get_logger("entry_eval.web")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if settings.run_migrations:
        await asyncio.to_thread(run_migrations, settings)
    if settings.auth_enabled:
        warn_if_dev_secret(settings)
    logger.info("app_start", version=settings.app_version, reader_mode=settings.reader_mode)
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Entry Eval", version=settings.app_version, lifespan=lifespan)
    # SessionMiddleware is only registered when auth is enabled. When off
    # (the default) the rest of the app behaves exactly as before — no
    # cookie issuance, no signed payload on every request, no dependency.
    if settings.auth_enabled:
        app.add_middleware(
            SessionMiddleware,
            secret_key=session_secret_for(settings),
            same_site="lax",
            https_only=False,
        )
    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Routers are wired in app/routers/__init__.py as phases land.
    from app.routers import register_routers

    register_routers(app)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": settings.app_version}

    @app.get("/readyz")
    async def readyz():
        try:
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"status": "unavailable", "error": str(exc)}, status_code=503)
        return {"status": "ready"}

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.error("unhandled_error", path=str(request.url.path), error=str(exc)[:500])
        return JSONResponse({"detail": "internal error"}, status_code=500)

    return app


app = create_app()
