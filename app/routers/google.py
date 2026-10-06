"""Google Forms connector routes (OAuth + fetch + import). Flag-gated."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import GoogleCredential, Student
from app.pipeline.ingestion.google_forms import (
    SCOPES,
    build_forms_client,
    build_oauth_flow,
    decrypt_token,
    encrypt_token,
    response_to_canonical,
)
from app.routers.common import render
from app.services.app_settings import get_effective_settings

router = APIRouter(tags=["google"])


@router.get("/google")
async def google_status(request: Request, session: AsyncSession = Depends(get_session)):
    settings = await get_effective_settings(session)
    credentials = (
        (await session.execute(select(GoogleCredential).where(GoogleCredential.revoked.is_(False))))
        .scalars()
        .all()
    )
    return render(
        request,
        "google.html",
        {
            "active": "imports",
            "settings": settings,
            "credentials": credentials,
            "configured": bool(settings.google_client_id and settings.google_client_secret),
        },
    )


@router.get("/google/oauth/start")
async def google_oauth_start(session: AsyncSession = Depends(get_session)):
    settings = await get_effective_settings(session)
    if not (settings.google_client_id and settings.google_client_secret):
        raise HTTPException(400, "Set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env first")
    flow = build_oauth_flow(settings)
    auth_url, _state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    return RedirectResponse(auth_url)


@router.get("/google/oauth/callback")
async def google_oauth_callback(
    request: Request, code: str, session: AsyncSession = Depends(get_session)
):
    settings = await get_effective_settings(session)
    flow = build_oauth_flow(settings)
    flow.fetch_token(code=code)
    credentials = flow.credentials
    account = "default"
    try:
        from googleapiclient.discovery import build

        userinfo = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
        account = userinfo.userinfo().get().execute().get("email", "default")
    except Exception:  # noqa: BLE001
        pass
    token = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "scopes": list(credentials.scopes or SCOPES),
    }
    encrypted = encrypt_token(token, settings)
    existing = (
        await session.execute(
            select(GoogleCredential).where(GoogleCredential.account_email == account)
        )
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            GoogleCredential(
                account_email=account, encrypted_token=encrypted, scopes=" ".join(token["scopes"])
            )
        )
    else:
        existing.encrypted_token = encrypted
        existing.revoked = False
    await session.commit()
    return RedirectResponse("/google?connected=1", status_code=303)


@router.post("/google/import")
async def google_import(
    request: Request,
    session: AsyncSession = Depends(get_session),
    form_id: str = Form(...),
):
    settings = await get_effective_settings(session)
    credential = (
        await session.execute(
            select(GoogleCredential).where(GoogleCredential.revoked.is_(False)).limit(1)
        )
    ).scalar_one_or_none()

    def token_loader():
        if credential is None:
            raise RuntimeError("No Google account connected")
        return decrypt_token(credential.encrypted_token, settings)

    try:
        client = build_forms_client(settings, token_loader=token_loader)
        data = await client.fetch_responses(form_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Google Forms fetch failed: {exc}") from exc

    # Map respondents to students by roll number found in the first answer.
    created = 0
    for response in data.get("responses", []):
        canonical = response_to_canonical(data, response)
        roll = _extract_roll(canonical)
        if not roll:
            continue
        student = (
            await session.execute(select(Student).where(Student.external_id == roll))
        ).scalar_one_or_none()
        if student is None:
            continue
        created += 1
    return {"form_id": form_id, "responses": len(data.get("responses", [])), "matched": created}


def _extract_roll(canonical) -> str | None:
    for page in canonical.pages:
        for line in (page.text or "").splitlines():
            if line.startswith("A:") and line[2:].strip():
                return line[2:].strip().upper()[:40]
    return None
