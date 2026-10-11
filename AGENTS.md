# AGENTS.md — How this project is built (working agreement)

This file describes the working style and command surface for anyone (human or
agent) contributing to this repository. It is also the operating contract for
AI coding agents: follow the flow, run the commands through Docker, keep the
evidence, append to TASTE.md.

## Non-negotiables

1. **Docker-first. Every Python command runs inside Docker.** No `pip install`,
   `pip3 install`, `python3 -m`, or any host-level Python execution against
   this repo. Use `make <target>` or `docker compose run --rm web <cmd>`.
   If a new Python dependency is needed, add it to `requirements.txt` and
   rebuild the image. **Never pass `--break-system-packages`** to pip — it
   pollutes the host. If the agent believes a quick local check is needed,
   it must use a `uv` venv (e.g. `uv venv .venv && uv pip install -r
   requirements.txt`), not the system interpreter. If neither Docker nor a
   venv is available, ask before touching pip.
2. **No hardcoded secrets.** Keys come from `.env` (git-ignored). Never print,
   log, or commit them. `app/logging.py` redacts known secret keys.
3. **Feature flags over hardcoding.** Reader mode, LLM provider, the entry
   flow's batch size / cleaning mode / drive provider are runtime flags
   (`.env` and/or the `settings` table, editable from the Settings page).
4. **Provider boundaries are sacred.** Ingestion adapters, document readers,
   and the LLM provider are all behind small interfaces. Replacing one must
   not touch the others.
5. **Evidence over claims.** A feature is done when a test passes *and* the UI
   shows it (screenshot in `docs/screenshots/`).
6. **Honest limitations.** Never claim accuracy or scale that was not measured.
7. **No-legacy principle (ADR-0001).** The dev phase can break and recreate
   data. We do not keep "fallback" code paths for the old exam pipeline. If
   a feature is not on the entry-evaluation critical path, it does not ship.

## The flow followed for every change

```
1. Recon        inspect env (docker ps, ports, keys, model catalog) before coding
2. Plan         write the plan, list decisions + rejected alternatives
3. Feature flag pick the switch that makes this replaceable
4. Schema       DB first: models -> migration -> constraints -> idempotency
5. Implement    pipeline module, then wire the job state machine
6. Test         unit -> integration -> API, all inside Docker (make test)
7. Verify UI    run the app, drive the browser, screenshot (make screenshots)
8. Measure      benchmarks for anything performance-related (make bench-*)
9. Document     README / docs/, update AGENTS.md and TASTE.md
10. Log taste   append a scored entry to TASTE.md for the milestone
```

## Command surface

| Command | Purpose |
|---|---|
| `make up` / `make down` | start/stop web + postgres |
| `make reset` | stop + delete the Postgres volume (full reset; dev phase) |
| `make logs` / `make ps` | observe |
| `make test` | full test suite against scratch DB `entry_eval_test` |
| `make seed` / `make demo` | seed a demo program + sample entries |
| `make screenshots` | Playwright UI verification + screenshots |
| `make migrate` | apply Alembic migrations |
| `make shell` | python REPL in the app image |

Web: http://127.0.0.1:8090 · Postgres: 127.0.0.1:5435

## Repository map

```
app/
  config.py            env-driven settings (single source of truth)
  db.py                async SQLAlchemy engine/session
  logging.py           structlog + secret redaction + stage timing
  models/              entry mode (program, entry, ocr, cleaning, scores, batches, samples, drive, rubric, settings)
  schemas/             pydantic contract objects (CanonicalDocument, ReadingResult, ...)
  pipeline/
    ingestion/         text, pdf, image, forms (CSV), drive (local mirror) adapters
    preprocessing/     render/deskew/denoise/upscale
    readers/           pdf_text, tesseract, minimax_vision, hybrid
    cleaning/          deterministic rules + optional LLM rewrite
    scoring/entry.py   rubric LLM scoring (the only scoring path)
  providers/llm/       minimax + fake providers behind one interface
  services/            storage, app_settings, batching, entries, stats
  routers/             HTTP endpoints (Jinja2 + HTMX, small JSON API)
scripts/               seed_demo, screenshot, screenshot_sandbox, ensure_test_db
alembic/               migrations
tests/                 unit/integration/API
docs/                  architecture, setup, limitations, adr, screenshots
```

## Pipeline shape (the only flow)

```
Program ─┬─ Entry ── EntryOCR (original + reconstructed) ── EntryScore (per criterion)
         │                                            └── EntryScoreCriterion
         ├─ EvaluationBatch (200–300 entries) ── ReviewSample
         └─ DriveFile
```

The pipeline runs synchronously inside a request handler. For the configured
batch size (200–300 per program) this is fast enough; if it ever stops being
fast enough, the first move is `asyncio.gather` inside the request, the
second move is FastAPI `BackgroundTasks`, and only then a separate worker.
There is intentionally no `jobs` table, no `worker/`, no `job_events`.

## Testing rules

- No network in the default suite: `LLM_PROVIDER=fake`, `READER_MODE` exercised
  with the hybrid reader + a local fake provider.
- Constraint tests must prove uniqueness/idempotency (duplicate entry import,
  double active batch, double review sample).
- Every list endpoint test asserts pagination bounds (no unbounded queries).
- Sandbox tests cover Path A (vision-LLM), Path B (tesseract + LLM), and Path C (vision+OCR fusion with image-arbitrated merge), plus the CER metric against golden transcripts in `data/samples/golden/`. Path C has a **fusion gate** (`FUSION_MAX_TESS_TO_VISION_RATIO`, default 1.5): when tesseract's text is much longer than vision's, tesseract is treated as noise and the fusion LLM call is skipped. Decision record: `docs/sandbox/fusion_gate.md`. Held-out test set: `data/samples/holdout.json` (page-03, page-11).

## Documentation rules

- `README.md` — setup + tour.
- `docs/architecture.md` — decisions and rejected alternatives, data model,
  failure-mode analysis (the answer to "what if X happens").
- `docs/adr/0001-no-legacy-dev-phase.md` — the no-legacy principle.
- `docs/setup.md` — install, config, Google console steps.
- `docs/screenshots/` — real screenshots, named `<page>_<state>.png`.

## Taste log

`TASTE.md` records each milestone with a score. It exists so the project keeps
reflecting the owner's stated preferences: Docker-first, feature flags,
evidence, honest numbers, trilingual (Hindi+English+Marathi) reality, no
over-engineering, no legacy.
