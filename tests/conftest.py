"""Shared test fixtures.

Tests run inside Docker against the scratch database `exam_eval_test`
(created by scripts/ensure_test_db.py). No network calls: the fake LLM
provider is used everywhere unless a test is marked `live`.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("READER_MODE", "hybrid")
os.environ.setdefault("REPAIR_ENABLED", "false")
os.environ.setdefault("STORAGE_ROOT", "/tmp/exam-eval-tests")

from app.config import Settings, get_settings  # noqa: E402
from app.db import reset_engine  # noqa: E402
from app.models import Base  # noqa: E402
from app.providers.llm.fake import FakeProvider  # noqa: E402

def _test_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://exameval:exameval@localhost:5435/exam_eval"
    )
    if base.rstrip("/").endswith("_test"):
        return base
    return re.sub(r"/([^/]+)$", r"/\1_test", base)


TEST_DATABASE_URL = _test_url()


async def _ensure_database() -> None:
    """Create the scratch database if it does not exist yet."""
    import asyncpg

    admin = re.sub(r"/([^/]+)$", "/postgres", TEST_DATABASE_URL).replace(
        "postgresql+asyncpg://", "postgresql://"
    )
    name = re.search(r"/([^/]+)$", TEST_DATABASE_URL).group(1)
    connection = await asyncpg.connect(admin)
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", name)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()


@pytest_asyncio.fixture
async def engine():
    """Function-scoped engine, NullPool: each test owns its connections and
    its event loop, so pytest-asyncio loop policy cannot leak between tests."""
    from sqlalchemy.pool import NullPool

    await _ensure_database()
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        yield session


@pytest_asyncio.fixture
async def clean_db(engine):
    """Truncate all tables between tests (fast, keeps schema)."""
    async with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())
    yield


@pytest.fixture
def fake_llm():
    return FakeProvider()


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    settings = get_settings().model_copy(
        update={
            "storage_root": tmp_path / "data",
            "llm_provider": "fake",
            "reader_mode": "hybrid",
            "repair_enabled": False,
            "scoring_strategy": "hybrid",
        }
    )
    return settings


@pytest_asyncio.fixture
async def client(engine, clean_db):
    """HTTP client against the real app with the test DB wired in."""
    from app.db import init_engine
    from app.main import create_app

    reset_engine()
    init_engine(get_settings().model_copy(update={"database_url": TEST_DATABASE_URL}))
    app = create_app()
    # app startup would run migrations + close the shared engine; skip lifespan
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
    reset_engine()


@pytest.fixture
def pdf_factory(tmp_path):
    """Build small PDFs: text-layer, scanned, mixed."""
    import pymupdf

    def make(name: str, pages: list[dict]) -> Path:
        path = tmp_path / name
        doc = pymupdf.open()
        for spec in pages:
            page = doc.new_page()
            if spec.get("text"):
                page.insert_textbox(
                    pymupdf.Rect(50, 50, 545, 790), spec["text"], fontsize=12
                )
            if spec.get("image"):
                page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=spec["image"])
        doc.save(path)
        doc.close()
        return path

    return make


@pytest.fixture
def image_factory(tmp_path):
    from PIL import Image, ImageDraw

    def make(name: str, lines: list[str], size=(800, 1000)) -> Path:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        y = 40
        for line in lines:
            draw.text((40, y), line, fill="black")
            y += 36
        path = tmp_path / name
        image.save(path)
        return path

    return make
