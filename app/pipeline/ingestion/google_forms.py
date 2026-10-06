"""Google Forms connector behind feature flags.

GOOGLE_FORMS_PROVIDER=google  -> real Forms API with OAuth installed-app flow
GOOGLE_FORMS_PROVIDER=mock    -> fixture-driven client for tests/demo

Tokens are stored encrypted (Fernet, key from APP_SECRET_KEY) in
google_credentials; they are never logged or exposed to the browser.
Responses become one CanonicalDocument per respondent with one text page
rendered as stable Q/A lines, so the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.logging import get_logger
from app.schemas import CanonicalDocument, CanonicalPage, SourceType, TextBlock
from app.services.storage import Storage

logger = get_logger("exam_eval.google_forms")

SCOPES = [
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/forms.body.readonly",
]
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "google_form_responses.json"


class GoogleFormsClient(Protocol):
    name: str

    async def fetch_responses(self, form_id: str) -> dict[str, Any]: ...


class MockGoogleFormsClient:
    """Fixture client: deterministic responses, no network, no credentials."""

    name = "mock"

    def __init__(self, fixture_path: Path | None = None):
        self.fixture_path = fixture_path or FIXTURE_PATH

    async def fetch_responses(self, form_id: str) -> dict[str, Any]:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        items: list[dict] = []
        for response in data.get("responses", []):
            items.append(
                {
                    "responseId": response["responseId"],
                    "createTime": response.get("createTime"),
                    "respondentEmail": response.get("respondentEmail"),
                    "answers": {
                        q["questionId"]: {"textAnswers": {"answers": [{"value": a}]}}
                        for q, a in zip(data["questions"], response["answers"])
                    },
                }
            )
        return {"formId": form_id, "questions": data["questions"], "responses": items}


class HttpGoogleFormsClient:
    """Real client. Tokens are refreshed from the encrypted store on each use."""

    name = "google"

    def __init__(self, settings: Settings | None = None, token_loader=None):
        self.settings = settings or get_settings()
        self._token_loader = token_loader

    async def fetch_responses(self, form_id: str) -> dict[str, Any]:
        if not self.settings.google_client_id or not self.settings.google_client_secret:
            raise RuntimeError(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and "
                "GOOGLE_CLIENT_SECRET in .env (see docs/setup.md)."
            )
        import asyncio

        return await asyncio.to_thread(self._fetch_sync, form_id)

    def _fetch_sync(self, form_id: str) -> dict[str, Any]:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = self._load_credentials()
        service = build("forms", "v1", credentials=credentials, cache_discovery=False)
        form = service.forms().get(formId=form_id).execute()
        questions = [
            {
                "questionId": item["questionItem"]["question"]["questionId"],
                "title": item.get("title"),
            }
            for item in form.get("items", [])
            if "questionItem" in item
        ]
        responses: list[dict] = []
        page_token = None
        while True:
            result = (
                service.forms()
                .responses()
                .list(formId=form_id, pageSize=500, pageToken=page_token)
                .execute()
            )
            responses.extend(result.get("responses", []))
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        return {"formId": form_id, "questions": questions, "responses": responses}

    def _load_credentials(self):
        from google.oauth2.credentials import Credentials

        if self._token_loader is None:
            raise RuntimeError("No stored Google credentials found. Connect Google first.")
        token = self._token_loader()
        return Credentials(
            token=token.get("token"),
            refresh_token=token.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.settings.google_client_id,
            client_secret=self.settings.google_client_secret,
            scopes=token.get("scopes") or SCOPES,
        )


def build_forms_client(settings: Settings | None = None, token_loader=None) -> GoogleFormsClient:
    settings = settings or get_settings()
    if not settings.google_forms_enabled:
        raise RuntimeError("Google Forms connector is disabled (GOOGLE_FORMS_ENABLED=false)")
    if settings.google_forms_provider == "mock":
        return MockGoogleFormsClient()
    return HttpGoogleFormsClient(settings, token_loader=token_loader)


def build_oauth_flow(settings: Settings | None = None):
    """Installed-app OAuth flow. Only used for the /google routes."""
    from google_auth_oauthlib.flow import Flow

    settings = settings or get_settings()
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.google_redirect_uri],
        }
    }
    return Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=settings.google_redirect_uri
    )


def encrypt_token(token: dict, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not settings.app_secret_key:
        raise RuntimeError(
            "APP_SECRET_KEY is not set; refusing to store Google tokens unencrypted. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; "
            'print(Fernet.generate_key().decode())"'
        )
    from cryptography.fernet import Fernet

    return Fernet(settings.app_secret_key.encode()).encrypt(
        json.dumps(token).encode()
    ).decode()


def decrypt_token(encrypted: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    from cryptography.fernet import Fernet

    return json.loads(Fernet(settings.app_secret_key.encode()).decrypt(encrypted.encode()))


def response_to_canonical(
    form_data: dict, response: dict, *, sha_source: str | None = None
) -> CanonicalDocument:
    """One form response -> one CanonicalDocument (single text page)."""
    question_titles = {q["questionId"]: q.get("title") or q["questionId"] for q in form_data["questions"]}
    lines = []
    for question_id, answer in sorted(response.get("answers", {}).items()):
        values = [
            a.get("value", "")
            for a in answer.get("textAnswers", {}).get("answers", [])
        ]
        lines.append(f"Q: {question_titles.get(question_id, question_id)}")
        lines.append(f"A: {'; '.join(v for v in values if v)}")
    text = "\n".join(lines)
    import hashlib

    digest = sha_source or json.dumps(response, sort_keys=True)
    sha = hashlib.sha256(digest.encode()).hexdigest()
    response_id = response.get("responseId", "unknown")
    page = CanonicalPage(
        index=0,
        has_text_layer=True,
        text=text,
        blocks=[TextBlock(text=text, kind="form_field")],
    )
    return CanonicalDocument(
        source_type=SourceType.GOOGLE_FORM,
        original_filename=f"google_form_{response_id}.txt",
        sha256=sha,
        size_bytes=len(text.encode()),
        mime="text/plain",
        pages=[page],
        metadata={
            "response_id": response_id,
            "responded_at": response.get("createTime"),
            "respondent_email": response.get("respondentEmail"),
        },
    )
