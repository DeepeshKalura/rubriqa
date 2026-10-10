"""Create the scratch test database if missing (compose-agnostic)."""
from __future__ import annotations

import asyncio
import os
import re

import asyncpg


def _admin_url() -> str:
    url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://entryeval:entryeval@db:5432/entry_eval")
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    return re.sub(r"/[^/]+$", "/postgres", url)


def _test_db() -> str:
    url = os.environ.get("DATABASE_URL", "")
    match = re.search(r"/([^/]+)$", url)
    name = match.group(1) if match else "entry_eval"
    return name if name.endswith("_test") else f"{name}_test"


async def main() -> None:
    admin = _admin_url()
    db_name = _test_db()
    connection = await asyncpg.connect(admin)
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname=$1", db_name)
        if not exists:
            await connection.execute(f'CREATE DATABASE "{db_name}"')
            print(f"created {db_name}")
        else:
            print(f"{db_name} already exists")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
