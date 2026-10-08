"""Entry-evaluation routes.

  * `/entries` — list of entries per program with state badges
  * `/programs` — list of programs
  * `/programs/new` — quick-create a program (with rubric + batch size)
  * `/programs/{id}` — overview, batches, sample of entries, link to sandbox
  * `/programs/{id}/import-form` — upload the form CSV (Himachal or Ashramshala)
  * `/programs/{id}/batches/create` — split into batches of 200-300
  * `/programs/{id}/batches/{bid}` — batch view + sample queue + resume
  * `/programs/{id}/export.csv` — final dataset export with the client columns
  * `/entries/{id}` — single-entry detail (demographics, OCR, score, human override)

The sandbox lives in `app/routers/sandbox.py` because it has a different feel
(interactive 2-way comparison page).
"""
from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import (
    BatchState,
    Entry,
    EntryScore,
    EntryState,
    EvaluationBatch,
    Program,
    ProgramState,
    ReviewSample,
    ReviewStatus,
    Rubric,
    RubricCriterion,
)
from app.pipeline.ingestion.forms import parse_csv
from app.routers.common import render
from app.services.batching import (
    batches_for_program,
    complete_batch,
    create_batches_for_program,
    pause_batch_for_review,
    resume_batch,
)
from app.services.entries import (
    register_parsed_responses,
    write_entries_csv,
)

router = APIRouter(tags=["entries"])


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------
@router.get("/programs")
async def list_programs(request: Request, session: AsyncSession = Depends(get_session)):
    programs = (
        (await session.execute(select(Program).order_by(Program.id.desc())))
        .scalars()
        .all()
    )
    return render(request, "entries/programs.html", {"active": "entries", "programs": programs})


@router.post("/programs/new")
async def create_program(
    request: Request,
    session: AsyncSession = Depends(get_session),
    name: str = Form(...),
    description: str = Form(""),
    batch_size: int = Form(250),
    rubric_name: str = Form(""),
):
    if not name.strip():
        raise HTTPException(400, "Program name required")

    existing = (
        await session.execute(select(Program).where(Program.name == name.strip()))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(409, "Program name already exists")

    rubric_id: int | None = None
    if rubric_name.strip():
        rubric = (
            await session.execute(select(Rubric).where(Rubric.name == rubric_name.strip()))
        ).scalar_one_or_none()
        if rubric is None:
            rubric = Rubric(name=rubric_name.strip(), description=description)
            session.add(rubric)
            await session.flush()
        rubric_id = rubric.id
        # ensure default criteria exist
        existing_criteria = (
            await session.execute(
                select(RubricCriterion).where(RubricCriterion.rubric_id == rubric_id)
            )
        ).scalars().all()
        if not existing_criteria:
            for nm, desc in [
                ("relevance", "How relevant the proposal is to the chosen theme."),
                ("feasibility", "How realistic it is to build in the student's context."),
                ("originality", "How original / non-trivially-copied the idea is."),
                ("ai_understanding", "How clearly the student explains how AI is used."),
                ("communication", "Clarity, structure, and respect for the word-count guidance."),
            ]:
                session.add(
                    RubricCriterion(
                        rubric_id=rubric_id, name=nm, weight=1.0, description=desc,
                        knockout=False, knockout_below_fraction=0.0,
                    )
                )

    batch_size = max(200, min(300, int(batch_size)))
    program = Program(
        name=name.strip(),
        description=description or None,
        state=ProgramState.ACTIVE,
        rubric_id=rubric_id,
        batch_size=batch_size,
    )
    session.add(program)
    await session.commit()
    return RedirectResponse(f"/programs/{program.id}", status_code=303)


@router.get("/programs/{program_id}")
async def program_detail(
    request: Request, program_id: int, session: AsyncSession = Depends(get_session)
):
    program = await session.get(Program, program_id)
    if program is None:
        raise HTTPException(404, "Program not found")
    batches = await batch_for_program(session, program_id)
    n_entries = (
        await session.execute(
            select(Entry).where(Entry.program_id == program_id).order_by(Entry.id.desc()).limit(10)
        )
    ).scalars().all()
    counts = await _entry_state_counts(session, program_id)
    return render(
        request,
        "entries/program_detail.html",
        {
            "active": "entries",
            "program": program,
            "batches": batches,
            "recent_entries": n_entries,
            "counts": counts,
        },
    )


@router.post("/programs/{program_id}/batches/create")
async def create_batches(
    program_id: int,
    session: AsyncSession = Depends(get_session),
    batch_size: int | None = Form(None),
):
    from app.config import get_settings
    program = await session.get(Program, program_id)
    if program is None:
        raise HTTPException(404, "Program not found")
    settings = get_settings()
    bs = int(batch_size) if batch_size else program.batch_size or settings.eval_batch_size
    await create_batches_for_program(session, program, settings=settings, batch_size=bs)
    return RedirectResponse(f"/programs/{program_id}", status_code=303)


@router.post("/programs/{program_id}/batches/{batch_id}/pause")
async def pause_batch(
    program_id: int,
    batch_id: int,
    session: AsyncSession = Depends(get_session),
):
    from app.config import get_settings

    batch = await session.get(EvaluationBatch, batch_id)
    if batch is None or batch.program_id != program_id:
        raise HTTPException(404, "Batch not found")
    settings = get_settings()
    await pause_batch_for_review(session, batch, settings)
    return RedirectResponse(f"/programs/{program_id}/batches/{batch_id}", status_code=303)


@router.post("/programs/{program_id}/batches/{batch_id}/resume")
async def resume_batch_route(
    program_id: int,
    batch_id: int,
    session: AsyncSession = Depends(get_session),
):
    batch = await session.get(EvaluationBatch, batch_id)
    if batch is None or batch.program_id != program_id:
        raise HTTPException(404, "Batch not found")
    await resume_batch(session, batch)
    return RedirectResponse(f"/programs/{program_id}/batches/{batch_id}", status_code=303)


@router.post("/programs/{program_id}/batches/{batch_id}/complete")
async def complete_batch_route(
    program_id: int,
    batch_id: int,
    session: AsyncSession = Depends(get_session),
):
    batch = await session.get(EvaluationBatch, batch_id)
    if batch is None or batch.program_id != program_id:
        raise HTTPException(404, "Batch not found")
    await complete_batch(session, batch)
    return RedirectResponse(f"/programs/{program_id}", status_code=303)


@router.get("/programs/{program_id}/batches/{batch_id}")
async def batch_detail(
    request: Request,
    program_id: int,
    batch_id: int,
    session: AsyncSession = Depends(get_session),
):
    batch = await session.get(EvaluationBatch, batch_id)
    if batch is None or batch.program_id != program_id:
        raise HTTPException(404, "Batch not found")
    samples = (
        await session.execute(
            select(ReviewSample)
            .where(ReviewSample.batch_id == batch_id)
            .order_by(ReviewSample.id)
        )
    ).scalars().all()
    entries = (
        await session.execute(
            select(Entry).where(Entry.batch_id == batch_id).order_by(Entry.id)
        )
    ).scalars().all()
    return render(
        request,
        "entries/batch_detail.html",
        {
            "active": "entries",
            "program_id": program_id,
            "batch": batch,
            "samples": samples,
            "entries": entries,
        },
    )


@router.post("/programs/{program_id}/batches/{batch_id}/review/{sample_id}")
async def resolve_sample(
    program_id: int,
    batch_id: int,
    sample_id: int,
    session: AsyncSession = Depends(get_session),
    status: str = Form("approved"),
    final_score: str = Form(""),
    notes: str = Form(""),
):
    sample = await session.get(ReviewSample, sample_id)
    if sample is None or sample.batch_id != batch_id:
        raise HTTPException(404, "Sample not found")
    sample.status = ReviewStatus(status)
    sample.notes = notes or None
    try:
        sample.final_score = float(final_score) if final_score else None
    except ValueError:
        sample.final_score = None
    if sample.status is not ReviewStatus.PENDING:
        sample.resolved_at = datetime.now(timezone.utc)
        # If human corrected, mirror the corrected score onto the entry
        if sample.status is ReviewStatus.CORRECTED and sample.final_score is not None:
            entry = await session.get(Entry, sample.entry_id)
            if entry is not None:
                entry.human_score = sample.final_score
                entry.review_status = "corrected"
                entry.human_score_notes = notes or None
                entry.review_notes = notes or None
    await session.commit()
    return RedirectResponse(f"/programs/{program_id}/batches/{batch_id}", status_code=303)


# ---------------------------------------------------------------------------
# import (form CSV)
# ---------------------------------------------------------------------------
@router.post("/programs/{program_id}/import-form")
async def import_form_csv(
    program_id: int,
    session: AsyncSession = Depends(get_session),
    file: UploadFile = File(...),
):
    program = await session.get(Program, program_id)
    if program is None:
        raise HTTPException(404, "Program not found")
    text = (await file.read()).decode("utf-8-sig", errors="replace")
    parsed = parse_csv(text)
    from app.config import get_settings

    created, _ = await register_parsed_responses(
        session,
        program_id,
        parsed,
        settings=get_settings(),
    )
    return RedirectResponse(f"/programs/{program_id}", status_code=303)


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------
@router.get("/entries")
async def list_entries(
    request: Request,
    session: AsyncSession = Depends(get_session),
    program_id: int | None = None,
    state: str | None = None,
    q: str | None = None,
):
    stmt = select(Entry).order_by(Entry.id.desc())
    if program_id:
        stmt = stmt.where(Entry.program_id == program_id)
    if state:
        stmt = stmt.where(Entry.state == state)
    entries = (await session.execute(stmt.limit(200))).scalars().all()
    return render(
        request,
        "entries/list.html",
        {"active": "entries", "entries": entries, "program_id": program_id, "state": state, "q": q},
    )


@router.get("/entries/{entry_id}")
async def entry_detail(
    request: Request, entry_id: int, session: AsyncSession = Depends(get_session)
):
    entry = await session.get(Entry, entry_id)
    if entry is None:
        raise HTTPException(404, "Entry not found")
    return render(
        request,
        "entries/detail.html",
        {"active": "entries", "entry": entry},
    )


@router.post("/entries/{entry_id}/override")
async def override_score(
    entry_id: int,
    session: AsyncSession = Depends(get_session),
    human_score: str = Form(""),
    human_notes: str = Form(""),
):
    entry = await session.get(Entry, entry_id)
    if entry is None:
        raise HTTPException(404, "Entry not found")
    try:
        entry.human_score = float(human_score) if human_score else None
    except ValueError:
        entry.human_score = None
    entry.human_score_notes = human_notes or None
    entry.review_notes = human_notes or None
    entry.review_status = "corrected" if entry.human_score is not None else "approved"
    await session.commit()
    return RedirectResponse(f"/entries/{entry_id}", status_code=303)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
@router.get("/programs/{program_id}/export.csv")
async def export_program_csv(program_id: int, session: AsyncSession = Depends(get_session)):
    program = await session.get(Program, program_id)
    if program is None:
        raise HTTPException(404, "Program not found")
    entries = (
        (await session.execute(select(Entry).where(Entry.program_id == program_id).order_by(Entry.id)))
        .scalars()
        .all()
    )
    body = write_entries_csv(entries)
    filename = f"program_{program_id}_final_dataset.csv"
    return StreamingResponse(
        iter([body]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def _entry_state_counts(session: AsyncSession, program_id: int) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Entry.state, Entry.id).where(Entry.program_id == program_id)
        )
    ).all()
    out: dict[str, int] = {}
    for state, _id in rows:
        out[state.value] = out.get(state.value, 0) + 1
    return out


async def batch_for_program(session: AsyncSession, program_id: int) -> list[EvaluationBatch]:
    return await batches_for_program(session, program_id)