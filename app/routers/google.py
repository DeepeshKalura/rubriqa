"""Google Forms connector routes (OAuth + fetch + import). Flag-gated."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    Exam,
    GoogleCredential,
    JobType,
    Student,
    Submission,
    SubmissionState,
)
from app.services.importer import register_documents, scan_folder
from app.services.storage import Storage
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
from app.services.oauth_state import make_state, verify_state

router = APIRouter(tags=["google"])


@router.get("/google")
async def google_status(request: Request, session: AsyncSession = Depends(get_session)):
    settings = await get_effective_settings(session)
    credentials = (
        (await session.execute(select(GoogleCredential).where(GoogleCredential.revoked.is_(False))))
        .scalars()
        .all()
    )
    exams = (await session.execute(select(Exam).order_by(Exam.name))).scalars().all()
    return render(
        request,
        "google.html",
        {
            "active": "imports",
            "settings": settings,
            "credentials": credentials,
            "configured": bool(settings.google_client_id and settings.google_client_secret),
            "exams": exams,
        },
    )


@router.get("/google/oauth/start")
async def google_oauth_start(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    settings = await get_effective_settings(session)
    if not (settings.google_client_id and settings.google_client_secret):
        raise HTTPException(400, "Set GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET in .env first")
    state_token = make_state()
    flow = build_oauth_flow(settings)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state_token,
    )
    response = RedirectResponse(auth_url)
    # SameSite=Lax keeps the cookie on the redirect-back, HttpOnly hides it.
    response.set_cookie(
        "google_oauth_state",
        state_token,
        max_age=600,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/google/oauth/callback")
async def google_oauth_callback(
    request: Request,
    code: str,
    state: str = "",
    session: AsyncSession = Depends(get_session),
):
    settings = await get_effective_settings(session)
    cookie_state = request.cookies.get("google_oauth_state", "")
    if not state or not verify_state(state) or state != cookie_state:
        raise HTTPException(400, "OAuth state mismatch; restart the connection.")
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
    response = RedirectResponse("/google?connected=1", status_code=303)
    response.delete_cookie("google_oauth_state")
    return response


@router.post("/google/import")
async def google_import(
    request: Request,
    session: AsyncSession = Depends(get_session),
    form_id: str = Form(...),
    exam_id: int = Form(...),
):
    settings = await get_effective_settings(session)
    credential = (
        await session.execute(
            select(GoogleCredential).where(GoogleCredential.revoked.is_(False)).limit(1)
        )
    ).scalar_one_or_none()
    if credential is None:
        raise HTTPException(400, "No Google account connected. Visit /google first.")
    exam = await session.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "Exam not found")

    def token_loader():
        return decrypt_token(credential.encrypted_token, settings)

    try:
        client = build_forms_client(settings, token_loader=token_loader)
        data = await client.fetch_responses(form_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Google Forms fetch failed: {exc}") from exc

    # Materialise each response as a single text page on disk, then reuse the
    # standard import pipeline (it dedupes by sha256 per submission and creates
    # submissions on demand). The text stays one canonical page per response.
    storage = Storage()
    storage.ensure_dirs()
    staging = storage.uploads / f"google_form_{form_id}"
    staging.mkdir(parents=True, exist_ok=True)
    for response in data.get("responses", []):
        canonical = response_to_canonical(data, response)
        (staging / f"{canonical.sha256[:16]}.txt").write_text(
            canonical.full_text, encoding="utf-8"
        )

    plan = scan_folder(staging)
    batch = await register_documents(
        session, exam_id=exam_id, plan=plan, storage=storage
    )
    await session.commit()

    pending = (
        await session.execute(
            select(Submission).where(
                Submission.exam_id == exam_id,
                Submission.state == SubmissionState.PENDING,
            )
        )
    ).scalars().all()
    queued = 0
    for submission in pending:
        await jobs_service.enqueue(
            session, JobType.PROCESS_SUBMISSION, submission_id=submission.id
        )
        queued += 1
    await session.commit()

    return {
        "form_id": form_id,
        "exam_id": exam_id,
        "responses": len(data.get("responses", [])),
        "new_documents": batch.new_documents,
        "already_imported": batch.already_imported,
        "queued": queued,
    }


def _extract_roll(canonical) -> str | None:
    for page in canonical.pages:
        for line in (page.text or "").splitlines():
            if line.startswith("A:") and line[2:].strip():
                return line[2:].strip().upper()[:40]
    return None
