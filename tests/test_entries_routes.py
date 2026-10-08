"""HTTP route tests for entry-evaluation mode: /programs, /entries, /sandbox.

The HTTP test client is wired to the real app with the scratch `exam_eval_test`
database. Sandbox compare runs with the fake provider because the API key is
not available in CI; the path-A/path-B branching is exercised explicitly.
"""
from __future__ import annotations

import csv
import io
import secrets

import pytest
from httpx import AsyncClient

HIMACHAL_HEADERS = [
    "Timestamp", "Email Address", "Write Full Name in English", "Gender",
    "School Name In English", "Which class do you study in?",
    "State", "Districts of Himachal Pradesh", "Please select your school.",
    "Please select the language", "अपने प्रोजेक्ट विचार के लिए केवल एक थीम का चयन करें",
    "AI का सबसे अच्छा वर्णन क्या है?",
    "Resources [Wi-fi]", "Resources [Broadband Network]",
    "Resources [Computer]", "Resources [Tablet]",
    "Resources [Laptop]", "Resources [Mobile]",
    "Situation", "Proposed AI Solution",
    "Declaration of Originality", "Upload an image", "Consent",
]


def _himachal_csv(rows: list[str]) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(HIMACHAL_HEADERS)
    for row in rows:
        w.writerow(row)
    return out.getvalue()


def _row(i: int, name: str | None = None) -> list[str]:
    return [
        str(i),
        f"{i}@example.com",
        name or f"Student {i}",
        "Male",
        "School A",
        "Class 7",
        "Himachal Pradesh",
        "Hamirpur",
        "School X",
        "Hindi (हिंदी)",
        "AI for Education",
        "AI is software that learns.",
        "हाँ", "नहीं", "हाँ", "नहीं", "हाँ", "हाँ",
        f"situation {i}",
        f"solution {i}",
        "I confirm that the idea shared is my own.",
        f"https://drive.google.com/file/d/abc{i}/view",
        "Yes",
    ]


# ---------------------------------------------------------------------------
# /programs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_programs_page_renders(client: AsyncClient) -> None:
    r = await client.get("/programs")
    assert r.status_code == 200
    assert "Programs" in r.text


@pytest.mark.asyncio
async def test_create_program_then_redirect(client: AsyncClient) -> None:
    name = f"Test program {secrets.token_hex(3)}"
    r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "Test rubric", "batch_size": "250"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "/programs/" in r.headers["location"]


@pytest.mark.asyncio
async def test_program_detail_renders(client: AsyncClient) -> None:
    name = f"Test program {secrets.token_hex(3)}"
    create_r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "x", "batch_size": "250"},
        follow_redirects=False,
    )
    location = create_r.headers["location"]
    pid = int(location.rstrip("/").split("/")[-1])
    r = await client.get(location)
    assert r.status_code == 200
    assert name in r.text


@pytest.mark.asyncio
async def test_create_program_dedup_rejected(client: AsyncClient) -> None:
    name = f"Dup {secrets.token_hex(3)}"
    r1 = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    assert r1.status_code == 303
    r2 = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    assert r2.status_code == 409


# ---------------------------------------------------------------------------
# /programs/{id}/import-form
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_import_form_creates_entries(client: AsyncClient) -> None:
    name = f"Import test {secrets.token_hex(3)}"
    create_r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    pid = int(create_r.headers["location"].rstrip("/").split("/")[-1])
    csv_text = _himachal_csv([_row(0, "Test Student")])
    r = await client.post(
        f"/programs/{pid}/import-form",
        files={"file": ("responses.csv", csv_text.encode("utf-8"), "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r2 = await client.get(f"/entries?program_id={pid}")
    assert "Test Student" in r2.text


@pytest.mark.asyncio
async def test_import_form_ashramshala(client: AsyncClient) -> None:
    name = f"Ashr {secrets.token_hex(3)}"
    create_r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    pid = int(create_r.headers["location"].rstrip("/").split("/")[-1])
    headers = [
        "Timestamp", "Select your name", "Select your Division",
        "Select your Project Office", "Select your Ashramshala",
        "Please select class",
        "Q1 [Wi-fi]", "Q1 [Broadband Network]", "Q1 [Computer]",
        "Q1 [Smartboard]", "Q1 [TV]", "Q1 [Projector]",
        "Upload an image", "Count of student submissions",
        "PO Name", "School Name",
    ]
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(headers)
    w.writerow([
        "1", "कृति", "Nashik", "Yaval", "Puran", "Class-7",
        "Yes", "No", "Yes", "Yes", "Yes", "No",
        "https://drive.google.com/file/d/xyz/view", "5",
        "Yaval", "Puran",
    ])
    csv_text = out.getvalue()
    r = await client.post(
        f"/programs/{pid}/import-form",
        files={"file": ("ashr.csv", csv_text.encode("utf-8"), "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r2 = await client.get(f"/entries?program_id={pid}")
    assert "कृति" in r2.text


# ---------------------------------------------------------------------------
# /programs/{id}/export.csv
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_csv_has_client_columns(client: AsyncClient) -> None:
    name = f"Export test {secrets.token_hex(3)}"
    create_r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    pid = int(create_r.headers["location"].rstrip("/").split("/")[-1])
    r = await client.get(f"/programs/{pid}/export.csv")
    assert r.status_code == 200
    rows = list(csv.reader(io.StringIO(r.text)))
    headers = rows[0]
    for required in [
        "Timestamp", "External Reference",
        "OCR (Original)", "OCR (Reconstruction)",
        "Input Tokens", "Output Tokens", "Total Tokens",
        "Confidence", "Status",
        "Score", "Max Score", "Percentage",
        "Review Status", "Human Score",
    ]:
        assert required in headers, f"missing column: {required}"


# ---------------------------------------------------------------------------
# /entries/{id}/override
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_entry_override_writes_human_score(client: AsyncClient) -> None:
    name = f"Override test {secrets.token_hex(3)}"
    create_r = await client.post(
        "/programs/new",
        data={"name": name, "rubric_name": "", "batch_size": "250"},
        follow_redirects=False,
    )
    pid = int(create_r.headers["location"].rstrip("/").split("/")[-1])
    csv_text = _himachal_csv([_row(0, "Override Target")])
    await client.post(
        f"/programs/{pid}/import-form",
        files={"file": ("r.csv", csv_text.encode("utf-8"), "text/csv")},
        follow_redirects=False,
    )
    from sqlalchemy import select
    from app.db import get_sessionmaker
    from app.models import Entry
    sm = get_sessionmaker()
    async with sm() as s:
        rows = (
            await s.execute(select(Entry).where(Entry.program_id == pid).order_by(Entry.id))
        ).scalars().all()
        assert rows
        entry_id = rows[0].id
    r = await client.post(
        f"/entries/{entry_id}/override",
        data={"human_score": "42.5", "human_notes": "revised"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    async with sm() as s:
        e = await s.get(Entry, entry_id)
        assert e.human_score == 42.5
        assert e.review_status == "corrected"


# ---------------------------------------------------------------------------
# /sandbox
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sandbox_page_renders(client: AsyncClient) -> None:
    r = await client.get("/sandbox")
    assert r.status_code == 200
    assert "Sandbox" in r.text


@pytest.mark.asyncio
async def test_sandbox_samples_list(client: AsyncClient) -> None:
    r = await client.get("/sandbox/samples")
    assert r.status_code == 200
    data = r.json()
    assert "samples" in data


@pytest.mark.asyncio
async def test_sandbox_reset(client: AsyncClient) -> None:
    r = await client.post("/sandbox/reset")
    assert r.status_code == 200
    data = r.json()
    assert "deleted_entries" in data


# ---------------------------------------------------------------------------
# Latent google router bug fix
# ---------------------------------------------------------------------------
def test_google_router_does_not_reference_undefined_jobs_service() -> None:
    """Regression: app/routers/google.py used to call `jobs_service.enqueue`
    without importing it. The import must be present."""
    import pathlib
    src = pathlib.Path("app/routers/google.py").read_text()
    assert "from app.services import jobs as jobs_service" in src
    assert "jobs_service.enqueue" in src