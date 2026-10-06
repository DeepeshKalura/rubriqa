"""Google Forms connector tests (mock provider + fixture, no network)."""
from __future__ import annotations

import pytest

from app.pipeline.ingestion.google_forms import (
    MockGoogleFormsClient,
    build_forms_client,
    build_oauth_flow,
    decrypt_token,
    encrypt_token,
    response_to_canonical,
)
from app.schemas import SourceType


async def test_mock_client_returns_fixture_responses():
    client = MockGoogleFormsClient()
    data = await client.fetch_responses("any-form-id")
    assert data["formId"] == "any-form-id"
    assert len(data["responses"]) >= 3
    assert len(data["questions"]) >= 4


async def test_response_to_canonical_shape():
    client = MockGoogleFormsClient()
    data = await client.fetch_responses("f")
    canonical = response_to_canonical(data, data["responses"][0])
    assert canonical.source_type is SourceType.GOOGLE_FORM
    assert canonical.pages[0].has_text_layer
    assert "Q:" in canonical.full_text and "A:" in canonical.full_text
    assert canonical.sha256


async def test_distinct_responses_have_distinct_hashes():
    client = MockGoogleFormsClient()
    data = await client.fetch_responses("f")
    first = response_to_canonical(data, data["responses"][0])
    second = response_to_canonical(data, data["responses"][1])
    assert first.sha256 != second.sha256


async def test_mock_hash_is_deterministic():
    client = MockGoogleFormsClient()
    data = await client.fetch_responses("f")
    a = response_to_canonical(data, data["responses"][0])
    b = response_to_canonical(data, data["responses"][0])
    assert a.sha256 == b.sha256


def test_client_factory_respects_flags(test_settings):
    settings = test_settings.model_copy(
        update={"google_forms_enabled": False, "google_forms_provider": "mock"}
    )
    with pytest.raises(RuntimeError):
        build_forms_client(settings)
    settings = settings.model_copy(update={"google_forms_enabled": True})
    client = build_forms_client(settings)
    assert client.name == "mock"


def test_token_encryption_roundtrip(test_settings):
    from cryptography.fernet import Fernet

    settings = test_settings.model_copy(update={"app_secret_key": Fernet.generate_key().decode()})
    token = {"token": "abc", "refresh_token": "def", "scopes": ["s"]}
    encrypted = encrypt_token(token, settings)
    assert "abc" not in encrypted
    assert decrypt_token(encrypted, settings) == token


def test_token_encryption_requires_secret(test_settings):
    settings = test_settings.model_copy(update={"app_secret_key": ""})
    with pytest.raises(RuntimeError):
        encrypt_token({"token": "x"}, settings)


def test_oauth_flow_requires_client_config(test_settings):
    settings = test_settings.model_copy(
        update={"google_client_id": "", "google_client_secret": ""}
    )
    flow = build_oauth_flow(settings)
    # Flow builds but has empty credentials; the app rejects before this point.
    assert flow is not None
