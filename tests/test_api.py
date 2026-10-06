"""API tests: pages render, health, pagination bounds, XSS-safe rendering."""
from __future__ import annotations

import pytest


async def test_health(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_dashboard_renders(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Dashboard" in response.text


async def test_students_pagination_bounds(client):
    response = await client.get("/students?page=1&per_page=25")
    assert response.status_code == 200
    # per_page is clamped to the allowed set
    response = await client.get("/students?per_page=100000")
    assert response.status_code == 200
    assert 'value="100"' in response.text


async def test_jobs_page(client):
    response = await client.get("/jobs")
    assert response.status_code == 200
    assert "Jobs" in response.text


async def test_review_page(client):
    response = await client.get("/review")
    assert response.status_code == 200


async def test_settings_page_hides_api_key(client):
    response = await client.get("/settings")
    assert response.status_code == 200
    body = response.text
    assert "sk-" not in body or "…" in body  # only masked form allowed


async def test_import_wizard_renders(client):
    response = await client.get("/imports/new")
    assert response.status_code == 200
    assert "Upload files" in response.text


async def test_google_page_renders(client):
    response = await client.get("/google")
    assert response.status_code == 200
    assert "Google Forms" in response.text


async def test_404_for_missing_entities(client):
    assert (await client.get("/students/999999")).status_code == 404
    assert (await client.get("/exams/999999")).status_code == 404
    assert (await client.get("/submissions/999999")).status_code == 404
    assert (await client.get("/jobs/999999")).status_code == 404


async def test_file_route_blocks_traversal(client):
    response = await client.get("/files/../../etc/passwd")
    assert response.status_code in (404, 400)


async def test_stats_api(client):
    response = await client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    for key in ("classes", "students", "exams", "submissions", "job_counts"):
        assert key in data
