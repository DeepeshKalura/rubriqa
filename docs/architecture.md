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

### OCR: Gemini vision first, Tesseract behind one interface

Measured on 9 human-verified handwritten pages (Hindi + Marathi + English),
same CER method for both engines:

| engine | mean CER | best page | $/page |
|---|---:|---:|---:|
| Tesseract (psm 11) | 82.4% | 70.8% | 0 |
| **Gemini 3.1 Flash-Lite** | **4.6%** | **0.3%** | $0.00065 |
| PP-OCRv3 (ONNX, rejected) | 90.7% | 78.8% | 0 |

Gemini won **9 of 9**. Tesseract stays in the hybrid chain as a CPU-only
fallback (useful when the API is rate-limiting or unreachable), not because it
is accurate. `READER_MODE=gemini_vision` with `LLM_PROVIDER=gemini` is the
shipped default; scoring still runs on MiniMax M3, which is strong at
text-only grading and cheap there.

Language disqualified two cloud OCR products before price ever mattered: AWS
Textract's handwriting is English-only, and Azure Document Intelligence's
handwriting model covers 9 languages with no Devanagari. "Just use Textract" is
not an option for Devanagari handwriting.

*Rejected:* PaddleOCR / EasyOCR / Surya (torch adds 2-4 GB to the image and
CPU inference on an i5 is slow), OCR-only (fails handwriting), vision-only with
no local fallback (an API outage then means zero throughput).

Full evidence, including the corrected golden-set alignment and the rejected
PP-OCR measurement, is in `docs/ocr-engine-decision.md`.

### LLM providers: MiniMax for scoring, Gemini for OCR

Both sit behind one `LLMProvider` protocol (`app/providers/llm/base.py`), which
is `@runtime_checkable` so conformance is testable.

- **MiniMax** via its OpenAI-compatible endpoint, with
  `thinking: {type: disabled}` for mechanical stages (verified against the live
  API — otherwise the reasoning preamble eats the token budget). Rubric scoring
  keeps thinking enabled.
- **Gemini** via `generateContent` with `thinkingBudget: 0` for transcription.
  Its free tier costs $0 and needs no billing card, but is marked *"used to
  improve our products: Yes"* — goldens only unless a paid key is used. The
  model id is **pinned**: Google retires ids without warning
  (`gemini-2.5-flash-lite` was closed to new keys on 2026-10-14), and a 404 now
  raises `model_retired` carrying Google's own suggested replacement.

Retries use exponential backoff + jitter on 429/5xx/timeouts, with
`SpendCapExceeded` re-raised rather than retried — a cap that retries spends
more time for the same guaranteed refusal. `LLM_SPEND_CAP_USD` caps one run and
is checked *before* each billable call. `FakeProvider` implements the same
interface so the whole suite runs without network.

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

## The failure mode where every test still passes

> A component is completely broken, the suite is green, and nobody notices.

This is not hypothetical — it shipped. `app/pipeline/readers/minimax_vision.py`
read `settings.low_legibility_threshold`, a field that has never existed. Every
page that went through the vision reader raised `AttributeError` **after** a
successful, billable LLM call. It was only found while wiring an unrelated
provider.

Four mechanisms allowed it, and all four had to go:

| Mechanism | Why it let the bug through | Guard now |
|---|---|---|
| `make fmt` is `compileall` | byte-compiles, so it only catches `SyntaxError`. An `AttributeError` is not a syntax error. | kept as-is; it is honestly labelled in `make check` |
| no linter or typechecker | a nonexistent `settings.<name>` is statically detectable and nothing checked | `tests/test_settings_references.py` walks the AST of `app/` and `scripts/` and fails on any `settings.<name>` that is not a real field/property. It also validates `model_copy(update={...})` keys, which pydantic silently drops. |
| no test for the reader | nothing ever executed it | `tests/test_reader_smoke.py` builds every `READER_MODE` and every provider from settings and runs one real page each. It asserts *aliveness* (no raise, legal status), not accuracy — `pdf_text` correctly returns `EMPTY` on an image with no text layer, and a test that demanded text would force the code to lie. |
| no CI | the suite only ran when a human ran it | **not fixed** — see below |

Three more, found while fixing it:

- **The secret-redaction guard did not match this repo's own key names.**
  `_REDACT_KEYS` was exact-match, so `minimax_api_key`, `google_api_key`,
  `llm_api_key`, `access_token` and `session_secret` all passed straight
  through. It is now substring-based (`app/logging.py`), recurses into nested
  dicts and lists, and deliberately does **not** match a bare `token` — the
  usage hooks log `prompt_tokens`/`completion_tokens`/`total_tokens` and the
  cost report is built from them.
- **`redact()` was being applied to error messages**, collapsing
  `httpx.ReadTimeout: timed out...` into `Read...` and hiding a real transport
  failure for an hour. `brief()` now truncates diagnostics and `redact()` is
  documented as secrets-only.
- **`LLMProvider` was not `@runtime_checkable`**, so provider conformance could
  not be verified even in a test.

### What is still not protected

- **No CI.** `make check` must be run by hand. Every guard above is only as
  strong as the last time somebody executed it. This is the single biggest
  remaining hole in the setup and it is a deliberate, recorded choice.
- **Secrets inside free-text exception strings are not scrubbed.** The
  automatic guard masks secret-bearing *keys*; it does not pattern-match values
  like `error="... key AQ.Ab8R ..."`. This is intentional — mangling free text
  to hunt for a key pattern produces unreadable diagnostics, which is exactly
  the mistake `redact()` was making. The contract is instead that a provider
  must never put a key in a message, pinned for Gemini by
  `tests/test_providers_gemini.py::test_api_key_is_never_in_the_payload_or_url`.
- **Aliveness is not correctness.** The smoke tests prove each reader *runs*.
  They do not prove Tesseract reads Devanagari well (it does not — see
  `docs/ocr-engine-decision.md`), and the golden set is 9 verified pages, not
  67 (issue #6).

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
