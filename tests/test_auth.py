"""Auth flag tests (roadmap item 2).

`auth_enabled=False` (the default): every existing route behaves as before.
`auth_enabled=True`: every existing route 403s until `/auth/login` succeeds;
`/auth/login` + `/auth/logout` + `/auth/me` round-trip cleanly.

The bcrypt admin hash is generated at test-time — nothing hardcoded.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.auth import hash_password


@pytest.mark.asyncio
async def test_auth_off_dashboard_returns_200(client_off: AsyncClient) -> None:
    r = await client_off.get("/")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_auth_off_login_me_returns_disabled(client_off: AsyncClient) -> None:
    r = await client_off.get("/auth/me")
    assert r.status_code == 200
    assert r.json() == {"user": None}
    r2 = await client_off.post(
        "/auth/login",
        data={"username": "admin", "password": "anything"},
        follow_redirects=False,
    )
    assert r2.status_code == 403


@pytest.mark.asyncio
async def test_auth_on_dashboard_returns_403(client_on: AsyncClient) -> None:
    r = await client_on.get("/")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_auth_on_programs_returns_403(client_on: AsyncClient) -> None:
    r = await client_on.get("/programs")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_auth_on_login_me_then_logout(client_on: AsyncClient, admin_password: str) -> None:
    # /auth/me before login → null
    r0 = await client_on.get("/auth/me")
    assert r0.status_code == 200
    assert r0.json() == {"user": None}

    # wrong password → 401
    bad = await client_on.post(
        "/auth/login",
        data={"username": "admin", "password": "wrong"},
        follow_redirects=False,
    )
    assert bad.status_code == 401

    # correct password → 200 + session cookie
    ok = await client_on.post(
        "/auth/login",
        data={"username": "admin", "password": admin_password},
        follow_redirects=False,
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["ok"] is True
    assert body["user"]["username"] == "admin"
    assert "session" in {c.name.lower() for c in ok.cookies.jar}

    # /dashboard now reachable
    r1 = await client_on.get("/")
    assert r1.status_code == 200

    # /auth/me → the user
    me = await client_on.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["username"] == "admin"

    # logout clears the cookie
    out = await client_on.post("/auth/logout", follow_redirects=False)
    assert out.status_code == 200

    # /dashboard 403 again
    r2 = await client_on.get("/")
    assert r2.status_code == 403

    # /auth/me → null
    me2 = await client_on.get("/auth/me")
    assert me2.status_code == 200
    assert me2.json() == {"user": None}