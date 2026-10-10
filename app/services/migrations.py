"""Programmatic Alembic migration runner (used at app startup)."""
from __future__ import annotations

import pathlib

from alembic import command
from alembic.config import Config

from app.config import Settings
from app.logging import get_logger

logger = get_logger("entry_eval.migrations")


def run_migrations(settings: Settings) -> None:
    root = pathlib.Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    logger.info("migrations_start", url=settings.database_url.split("@")[-1])
    command.upgrade(cfg, "head")
    logger.info("migrations_done")
