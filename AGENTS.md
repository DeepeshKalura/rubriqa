# AGENTS.md — How this project is built (working agreement)

This file describes the working style and command surface for anyone (human or
agent) contributing to this repository. It is also the operating contract for
AI coding agents: follow the flow, run the commands through Docker, keep the
evidence, append to TASTE.md.

## Non-negotiables

1. **Docker-first.** Every app command runs through `docker compose` / the
   `Makefile`. There is no "install Python locally and run it" path in the docs.
2. **No hardcoded secrets.** Keys come from `.env` (git-ignored). Never print,
   log, or commit them. `app/logging.py` redacts known secret keys.
3. **Feature flags over hardcoding.** Reader mode, LLM provider, scoring
   strategy and the Google Forms connector are runtime flags
   (`.env` and/or the `settings` table, editable from the Settings page).
4. **Provider boundaries are sacred.** Ingestion adapters, document readers,
   scoring strategies, and LLM providers are all behind small interfaces.
   Replacing one must not touch the others.
5. **Evidence over claims.** A feature is done when a test passes *and* the UI
   shows it (screenshot in `docs/screenshots/`).
6. **Honest limitations.** Never claim accuracy or scale that was not measured.

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
| `make up` / `make down` | start/stop web + worker + postgres |
| `make logs` / `make ps` | observe |
| `make test` | full test suite against scratch DB `exam_eval_test` |
| `make test-live` | tests hitting the real MiniMax API (costs tokens) |
| `make seed` | create org/class/batches/students/exam/rubric |
| `make demo` | generate synthetic dataset + seed + import |
| `make e2e` | run full pipeline over demo data, verify in DB |
| `make scale-test` | 20k synthetic submissions + API query benchmark |
| `make bench-readers` | reader-mode accuracy comparison on labeled data |
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
  models/              SQLAlchemy schema (hierarchy -> pages -> scores)
  schemas/             pydantic contract objects (CanonicalDocument, ReadingResult, ...)
  pipeline/
    ingestion/         text, pdf, image, google_forms adapters
    preprocessing/     render/deskew/denoise/upscale
    readers/           pdf_text, tesseract, minimax_vision, hybrid
    validation/        structural checks (extensible to content)
    extraction/        student matching + question/answer extraction
    scoring/           exact, numeric, rubric_llm, hybrid strategies
  providers/llm/       minimax + fake providers behind one interface
  services/            storage, idempotency, jobs, sweeper, stats, export
  routers/             HTTP endpoints (Jinja2 + HTMX, small JSON API)
worker/main.py         job claim -> pipeline stages -> heartbeat
scripts/               synthetic data, seeds, e2e, benchmarks, screenshots
alembic/               migrations
tests/                 unit/integration/API/E2E
docs/                  architecture, setup, screenshots
```

## Job state contract

```
QUEUED -> RUNNING -> READING -> VALIDATING -> EXTRACTING -> SCORING -> COMPLETED
                                        \-> NEEDS_REVIEW (blocking validation issue)
any -> FAILED (error_code + message) -> retry (attempts preserved, resume from checkpoint)
```

Rules:
- The browser never runs the pipeline; jobs are claimed by the worker with
  `FOR UPDATE SKIP LOCKED`.
- Every stage commits its own transaction and appends a `job_events` row.
- Retries resume from the last completed stage (`jobs.checkpoint`), and page
  reading is idempotent per `(page, reading_version, provider)`.
- Stale heartbeats are requeued by the sweeper; after `JOB_MAX_ATTEMPTS` the
  job is `FAILED` and shows an actionable error in the UI.

## Testing rules

- No network in the default suite: `LLM_PROVIDER=fake`, `READER_MODE` exercised
  with a local fake reader. Live-MiniMax tests are marked `@pytest.mark.live`.
- Constraint tests must prove uniqueness/idempotency (duplicate import, double
  answer, double active job).
- Crash/retry paths are tested by simulating a stale heartbeat, not by killing
  containers.
- Every list endpoint test asserts pagination bounds (no unbounded queries).

## Documentation rules

- `README.md` — setup + tour.
- `docs/architecture.md` — decisions and rejected alternatives, data model,
  failure-mode analysis (the answer to "what if X happens").
- `docs/setup.md` — install, config, Google console steps.
- `docs/screenshots/` — real screenshots, named `<page>_<state>.png`.
- `reports/` — generated benchmark outputs (reader accuracy, scale test).

## Taste log

`TASTE.md` records each milestone with a score. It exists so the project keeps
reflecting the owner's stated preferences: Docker-first, feature flags,
evidence, honest numbers, bilingual (Hindi+English) reality, no over-engineering.
