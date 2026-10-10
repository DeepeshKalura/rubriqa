# Architecture

## Decisions and rejected alternatives

### Web framework: FastAPI + Jinja2 + HTMX, no JS build step

The users are a handful of staff on one laptop. Server-rendered pages with
HTMX cover every interaction (live batch progress, pagination, human score
override) without a Node toolchain, a bundler, or client-side state. The CSS
is hand-written and HTMX is vendored locally, so the app works fully offline.

*Rejected:* Next.js/React SPA (build complexity, no requirement), Django
(admin is nice but the ORM+async story is worse for this pipeline), Gradio/
Streamlit (not a multi-page CRUD app).

### Queue: in-request, no separate worker

For 200-300 entries per program the synchronous in-request flow is fast
enough: register the form rows, OCR them with the configured reader, clean
the text, score it with the LLM, write the result, return. If this ever
stops being fast enough, the first move is `asyncio.gather` inside the
request, the second is FastAPI `BackgroundTasks`, and only then a separate
worker. The `FOR UPDATE SKIP LOCKED` queue pattern was deliberately
abandoned with the legacy pipeline in ADR-0001.

*Rejected:* Postgres-backed job queue with worker + sweeper + heartbeat
(over-engineering for this scale; was the legacy exam mode), Celery (broker
+ result backend + serializer configuration for a local tool), Redis
(another daemon to lose data on).

### Database: PostgreSQL

Relational integrity is the product here: program+entry uniqueness on
`(program_id, external_ref)`, document hash dedupe, one batch per program
section, no double review sampling — all enforced by constraints, not
application code. JSONB covers the flexible bits (token usage, settings
overrides, drive file metadata).

*Rejected:* SQLite (single-writer would serialize the future worker + web
process), MongoDB (no uniqueness guarantees we need).

### OCR: Tesseract + MiniMax M3 vision behind one interface

Tesseract is CPU-cheap and adequate for printed Hindi/English; MiniMax M3
vision is strong on both handwriting and printed pages and needs no local
GPU. Both sit behind `DocumentReader`. The hybrid chain escalates only
low-confidence pages, so API spend tracks actual difficulty rather than page
count. On a text-layer PDF, no OCR or vision call happens at all.

*Rejected:* PaddleOCR / EasyOCR / Surya (torch adds 2-4 GB to the image and
CPU inference on an i5 is slow), OCR-only (fails handwriting), vision-only
(latency and cost on every page, needs internet for everything).

Historical measurement on synthetic pages (see `reports/reader_bench_*.json`,
not re-run): answer recall 0.72 tesseract, 0.92 vision, 0.83 hybrid at
roughly 2 s / 13 s / 4 s per page.

### LLM provider: MiniMax via OpenAI-compatible endpoint

`/v1/chat/completions` with `response_format=json_object` and
`thinking: {type: disabled}` for mechanical stages (verified against the live
API — otherwise the reasoning preamble eats the token budget). Rubric scoring
keeps thinking enabled. Retries with exponential backoff + jitter on
429/5xx/timeouts. `FakeProvider` implements the same interface so the whole
suite runs without network.

### Storage: content-addressed files on disk

Originals are stored as `data/originals/<sha[:2]>/<sha>.<ext>`; derived page
images live under `data/derived/`. Postgres stores paths and hashes. Writes
are atomic (tmp + rename), so a crash never leaves a half-file referenced
by a row. Pages are rendered lazily — text PDFs never produce images.

### Drive ingestion: local mirror only (dev phase)

A folder of files acts as a Drive substitute. The `DriveClient` protocol is
preserved so a real provider can be re-added behind the same interface when
the dev phase ends. The OAuth + Google Forms connector was deleted in
ADR-0001 because it only served the legacy exam pipeline.

### Google Forms: CSV upload from the form, not a live connector

The two form shapes (Himachal typed-name + Ashramshala selected-name) are
auto-detected from the CSV header and parsed into the same `Entry` model.
The parser keeps the Hindi/Marathi theme slug mapping in 30 lines of
domain code that no library gives us.

## Data model

```
Program ─┬─ Entry ── EntryOCR (original + reconstructed) ── EntryScore (per criterion)
         │                                          └── EntryScoreCriterion
         │           └─ CleaningRun (rules or rules+LLM)
         ├─ EvaluationBatch (200–300 entries) ── ReviewSample
         └─ DriveFile

Rubric ─ RubricCriterion
Setting   (key/value feature-flag overrides, editable from /settings)
```

## Failure-mode analysis

- **LLM is unavailable.** Each reader + scoring call is wrapped in
  exponential-backoff retries. After `MINIMAX_MAX_RETRIES`, the entry
  state is set to `failed` and surfaced on the dashboard. The user can
  retry the entry from the entry detail page.
- **Image is unreadable.** Tesseract returns `EMPTY`; the hybrid chain
  escalates to the vision reader. The chain reports `chain_attempts` on
  the `ReadingResult` so the user can see which readers were tried.
- **Tesseract is misinstalled.** The container ships the `tesseract-ocr-eng`
  and `tesseract-ocr-hin` packages. The reader raises `ReaderError(code=
  "tesseract_missing")` on a clear startup check.
- **Re-importing a form CSV.** The `entries` unique constraint on
  `(program_id, external_ref)` makes re-import idempotent. The new entries
  have new ids; existing entries are not touched.
- **The Postgres volume is lost.** `make reset` deletes it. Recreate with
  `make up && make seed`. No fallback code paths; the dev phase is allowed
  to break.
- **The hybrid chain matches no reader.** `build_reader` raises
  `ReaderError(code="empty_chain")` with the bad chain in the message.
- **Cleaning rewrites the student's meaning.** `entry_ocr.original_text`
  keeps the raw OCR text forever. The `reconstructed_text` is the LLM
  rewrite. Per-criterion rationales in `entry_score_criteria` let a human
  reviewer see what the LLM thought the student meant.

## What we explicitly did NOT build

- **No background workers, no jobs table, no `JobEvent` log, no FOR UPDATE
  SKIP LOCKED claim, no heartbeat/sweeper, no retry/backoff checkpoints.**
  In-request synchronous processing is fast enough at 200-300 entries per
  batch. (ADR-0001.)
- **No `Repairs` table, no separate `repair_enabled` flag.** The hybrid
  chain already escalates the whole page to the next reader.
- **No per-call `LlmCall` log table.** The API dashboard already tracks
  token usage; each `EntryScore` row carries its own token counts.
- **No 4-strategy scoring factory (`exact` / `numeric` / `rubric_llm` /
  `hybrid`).** The entry pipeline calls one function (`score_entry`) that
  uses one strategy (LLM rubric).
- **No validation stage, no extraction stage.** The rubric prompt is
  specific enough that no separate structural pass is needed.
- **No `Organization → Class → Batch → Student` hierarchy.** `Program`
  carries `school_name`, `class_name`, `full_name`, `external_ref` as
  free text.
- **No real Google Drive / Google Forms connector.** The dev phase uses the
  local mirror + CSV upload. (ADR-0001.)
