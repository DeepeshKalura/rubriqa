# Architecture

## Decisions and rejected alternatives

### Web framework: FastAPI + Jinja2 + HTMX, no JS build step

The users are a handful of staff on one laptop. Server-rendered pages with
HTMX polling cover every interaction (live job progress, pagination, repair
decisions) without a Node toolchain, a bundler, or client-side state. The CSS
is hand-written and HTMX is vendored locally, so the app works fully offline.

*Rejected:* Next.js/React SPA (build complexity, no requirement), Django
(admin is nice but the ORM+async story is worse for this pipeline), Gradio/
Streamlit (not a multi-page CRUD app with a job queue).

### Queue: Postgres `FOR UPDATE SKIP LOCKED`, not Redis/Celery

The jobs table is the queue. Claiming is atomic, crashes leave the row behind
for the sweeper, retries resume from a JSON checkpoint, and there is exactly
one new stateful component (Postgres) to back up. Worker concurrency is a
config value; scaling means running a second worker container.

*Rejected:* Celery (broker + result backend + serializer configuration for a
local tool), Redis (another daemon to lose data on), in-process asyncio queue
(dies with the web process, no crash recovery).

### Database: PostgreSQL

Relational integrity is the product here: exam+student uniqueness, document
hash dedupe, one answer per (submission, question), one active job per
submission — all enforced by constraints, not application code. JSONB covers
the flexible bits (checkpoints, evidence, metadata).

*Rejected:* SQLite (single-writer would serialize the worker and web app),
MongoDB (no uniqueness guarantees we need).

### OCR: Tesseract + MiniMax M3 vision behind one interface

Tesseract is CPU-cheap and adequate for printed Hindi/English; MiniMax M3
vision is strong on both handwriting and printed pages and needs no local
GPU. Both sit behind `DocumentReader`. The hybrid chain escalates only
low-confidence pages, so API spend tracks actual difficulty rather than page
count. On a text-layer PDF, no OCR or vision call happens at all.

*Rejected:* PaddleOCR / EasyOCR / Surya (torch adds 2–4 GB to the image and
CPU inference on an i5 is slow), OCR-only (fails handwriting), vision-only
(latency and cost on every page, needs internet for everything).

Measured on synthetic pages (see `reports/reader_bench_*.json`): answer recall
0.72 tesseract, 0.92 vision, 0.83 hybrid at roughly 2 s / 13 s / 4 s per page.

### LLM provider: MiniMax via OpenAI-compatible endpoint

`/v1/chat/completions` with `response_format=json_object` and
`thinking: {type: disabled}` for mechanical stages (verified against the live
API — otherwise the reasoning preamble eats the token budget). Rubric scoring
keeps thinking enabled. Retries with exponential backoff + jitter on
429/5xx/timeouts; usage is logged to `llm_calls` (tokens, latency, status, no
prompts, never the key). `FakeProvider` implements the same interface so the
whole suite runs without network.

### Storage: content-addressed files on disk

Originals are stored as `data/originals/<sha[:2]>/<sha>.<ext>`; derived page
images live under `data/derived/<document_id>/`. Postgres stores paths and
hashes. Writes are atomic (tmp + rename), so a crash never leaves a half-file
referenced by a row. Pages are rendered lazily — text PDFs never produce
images.

### Google Forms: connector, not a document type

An adapter converts each response into a `CanonicalDocument` (one text page),
after which the pipeline cannot tell it apart from an uploaded text file.
Tokens are Fernet-encrypted with `APP_SECRET_KEY`; the mock provider serves
fixtures for tests and demos.

## Data model (abridged)

```
organizations ─┬─ classes ─┬─ batches ── enrollments ── students
               │           └─ exams ── questions ── rubrics ── rubric_criteria
               └─ students
submissions (unique: exam_id+student_id)
  ├─ documents (unique: submission_id+sha256)
  │    └─ pages (unique: document_id+index)
  │         └─ reading_runs (immutable, is_selected flags the winner)
  │              └─ repairs (original_text + repaired_text, decision recorded)
  ├─ answers (unique: submission_id+question_id) ── scores
  ├─ scoring_runs (versioned, is_current)
  └─ validation_issues (severity, resolved)
jobs + job_events + import_batches + llm_calls + settings + google_credentials
```

## Job state contract

```
QUEUED ─► RUNNING ─► READING ─► VALIDATING ─► EXTRACTING ─► SCORING ─► COMPLETED
                                       └─ blocking issue ─► NEEDS_REVIEW
any ─► FAILED(error_code, message) ─► retry (attempts preserved, checkpoint kept)
```

- Claim: `SELECT ... FOR UPDATE SKIP LOCKED` + `attempts += 1` + heartbeat.
- Every stage commits separately and appends a `job_events` row.
- Reading is idempotent per `(page, reading_version, provider)`; a retry skips
  pages already `read` unless `force_pages` is set.
- The sweeper (in the worker, every 60 s) requeues jobs whose heartbeat is
  older than `JOB_STUCK_MINUTES`; after `JOB_MAX_ATTEMPTS` they become FAILED
  with an actionable message and a Retry button.

## Failure-mode analysis (answers to "what if X happens")

| Scenario | Behaviour | Mechanism |
|---|---|---|
| Same folder imported twice | 0 new documents, counted as "already imported" | sha256 dedupe + unique(submission, sha256); dry-run diff shows it before commit |
| Worker crashes mid-job | Job requeued by sweeper; completed stages skipped | heartbeat + checkpoint + per-page reading idempotency |
| MiniMax call fails | Retries with backoff; page marked `deferred`, job still completes with a needs-review note; sweeper retries later | provider retry loop + `PageState.DEFERRED` |
| Laptop loses internet | Text/PDF paths keep working; vision pages defer instead of failing the submission | per-page error isolation |
| Postgres restarts | Connections re-established (`pool_pre_ping`); stuck jobs picked up by the sweeper | no in-memory-only state |
| OCR returns garbage | Low confidence triggers repair; repair output is only auto-accepted above a legibility threshold; original text is always kept | `Repair` table + `repair_auto_accept_min_legibility` |
| LLM hallucinates a repair | Repair prompt forbids inventing content, `[illegible]` is the escape hatch; reviewer sees original vs repaired side by side and can reject | diff UI + accept/reject routes |
| A job is stuck | Sweeper fails it after max attempts with a specific error; UI shows what to fix | `sweep_stale_jobs` |
| Retry after a real fix | Same job requeued; attempts preserved; checkpoint means no double work; scores are versioned (old run kept, `is_current=false`) | `retry_job` + scoring runs |
| Duplicate score writes | Impossible: unique(scoring_run, answer, criterion) | DB constraint |
| Crash during file copy | No partial file is ever referenced | atomic tmp+rename |

## Concurrency and resource defaults (16 GB laptop)

| Knob | Default | Why |
|---|---|---|
| `WORKER_CONCURRENCY` | 2 | two jobs in flight without memory spikes |
| `OCR_THREADS` | 2 | tesseract subprocesses + render threads |
| `LLM_CONCURRENCY` | 3 | polite API parallelism, bounded retries |
| web memory limit | 1 GB | compose `deploy.resources` |
| worker memory limit | 2 GB | compose `deploy.resources` |

Rendering is capped at 220 DPI and 2200 px long side — enough for OCR, small
enough to keep vision payloads under the 10 MB API limit.

## What could become a bottleneck

1. **Vision API latency** (10–20 s/page for dense handwriting). Parallelism
   and escalation thresholds control it; a local OCR provider can be added
   behind the same interface later.
2. **Page rendering on very large scans.** Bounded DPI and lazy rendering keep
   it linear; the worker processes pages one at a time per job.
3. **Dashboard counts under millions of rows.** Fine at 20k (37 ms measured);
   beyond that, materialized counters would be the next step.
4. **Single worker container.** Compose scales horizontally
   (`docker compose up -d --scale worker=2`) — the queue is already safe for it.

## Deliberately not built

- User accounts/permissions (single local operator; UI binds to 127.0.0.1)
- Redis/Kafka/Celery (no demonstrated need)
- Vector search / embeddings (no matching requirement yet)
- A background "watch folder" daemon (import wizard is explicit; easier to audit)
