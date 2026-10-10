# Entry Eval — local multilingual handwritten entry evaluation

A local-first system that takes student project entries (Hindi, English,
Marathi, mixed) — typed in Google Forms or uploaded as a handwritten image to
Google Drive — OCRs them, cleans them, and scores them against a configurable
rubric in controlled batches (200-300 per batch). All data stays local.

Built to run on an ordinary office laptop (Intel i5 / 16 GB RAM, no GPU) with
one command:

```bash
cp .env.example .env        # put your MiniMax key in MINIMAX_API_KEY or LLM_API_KEY
make up                     # docker compose up --build -d
```

Open <http://127.0.0.1:8090>. Postgres is exposed on `127.0.0.1:5435`.

## Two paths, one sandbox (`/sandbox`)

The sandbox is the central testing tool: drop a handwritten page image (or pick
a bundled sample from `data/samples/pages/`) and run both OCR paths side by
side against your configured MiniMax endpoint:

- **Path A** — vision-capable LLM does OCR + scoring in one model call.
- **Path B** — Tesseract OCR locally, then LLM scores the cleaned text.

Both paths show OCR text, per-criterion rubric results with bars, strengths /
gaps / observations, token usage, and elapsed time. A side-by-side diff table
at the bottom highlights where the two paths agree and disagree.

## What works today

- **Entry-evaluation mode** — Program → Entry → OCR runs → cleaning
  pass → rubric scoring → batch orchestration (200-300 configurable) → review
  samples → final dataset CSV.
- **Both form shapes** — Himachal typed-name form and Marathi Ashramshala
  selected-name form are auto-detected and parsed into the same `Entry` model.
- **Local Drive mirror** — drop handwritten images into `./data/drive_mirror/`
  and the entry importer treats them as Drive files.
- **Cleaning** — deterministic rules pass + optional LLM rewrite; original
  text is always kept in `entry_ocr.original_text` for audit.
- **Batches + review** — `EvaluationBatch` rows in the configurable 200-300
  range; pause/resume creates `ReviewSample` rows for random + AI-flagged +
  low-confidence entries. Human overrides write to `Entry.human_score`.
- **Real MiniMax keys wired** — `.env` already configured with
  `LLM_PROVIDER=minimax`, vision model `MiniMax-M3`. Tests still use the
  offline `fake` provider.

## Command surface

All commands run through Docker.

```bash
make up            # start db + web
make down          # stop (keeps volumes and ./data)
make reset         # stop + delete the Postgres volume (full reset; dev phase)

make test          # test suite against scratch DB (fake provider, no network)
make seed          # seed a demo program + sample entries

make screenshots   # Playwright UI checks + docs/screenshots/*.png

make logs / make ps / make shell / make migrate
```

## Feature flags (`.env` or Settings page)

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | `minimax`, `fake` | `minimax` | Swap the LLM backend |
| `READER_MODE` | `pdf_text`, `tesseract`, `minimax_vision`, `hybrid` | `hybrid` | Document reader |
| `HYBRID_READER_CHAIN` | e.g. `pdf_text,tesseract,minimax_vision` | — | Reader composition |
| `HYBRID_ACCEPT_CONFIDENCE` | 0.0-1.0 | 0.75 | Stop the chain above this |
| `HYBRID_ESCALATE_BELOW` | 0.0-1.0 | 0.60 | Flag low_confidence below this |
| `EVAL_BATCH_SIZE` | 200-300 | `250` | Entries per batch |
| `CLEANING_MODE` | `rules`, `llm` | `rules` | OCR cleaning strategy |
| `GOOGLE_DRIVE_ENABLED` | true/false | false | Use the local Drive mirror |
| `SANDBOX_DEFAULT_PROVIDER` | `vision`, `ocr_llm` | `vision` | Initial sandbox provider |

Flags can also be changed at runtime on the Settings page (stored in the
`settings` table, overriding `.env` for the next request).

## Architecture in one picture

```
form responses (CSV) ──┐
                       │
Google Drive images ────┼─► ingestion (forms / drive) ─► Entry
                       │                                   │
                       │                              OCR (vision | tesseract)
                       │                                   │
                       │                              cleaning (rules + LLM)
                       │                                   │
                       │                          rubric scoring (LLM)
                       │                                   │
                       │                          batch (200-300) + review
                       │                                   │
                       │                          final dataset CSV
                       │
                       └─► /sandbox (2-way comparison tool)
```

Provider boundaries are small protocols: `DocumentReader`, `LLMProvider`,
`DriveClient`, `FormParser`. Replacing one never touches another.

## Documentation

- `docs/architecture.md` — decisions, rejected alternatives, failure-mode analysis
- `docs/adr/0001-no-legacy-dev-phase.md` — the no-legacy principle
- `docs/setup.md` — install, configuration, Google console steps
- `docs/limitations.md` — the honest list of what is not proven yet
- `reports/` — historical reader benchmark (synthetic data, not re-run)
- `docs/screenshots/` — verified UI screenshots
- `TASTE.md` — project preference log with milestone scores
- `AGENTS.md` — working agreement and command surface

## Measured numbers (this machine, real MiniMax)

| Measurement | Result |
|---|---|
| Test suite (default, no network) | 54 passed, 4 deselected |
| Live sandbox (handwritten Marathi sample) | vision 24/50 (48%, 7670 tok, 46 s) — tesseract+llm 27/50 (54%, 2288 tok, 18 s) |
| Reader comparison (15 synthetic pages, historical) | pdf_text 0.23, tesseract 0.72, vision 0.92, hybrid 0.83 |
| Memory (idle, 1 web + 1 db) | web ~80 MB, db ~180 MB |

**Honest limitations:** the live sandbox numbers above are one sample each
(handwritten Marathi). No broad batch accuracy claim yet — the 200-300 batch
gates are designed for that, not the sandbox. Full list: `docs/limitations.md`.

## ADR-0001: no-legacy principle

The dev phase can break and recreate data. The legacy Q&A exam pipeline,
the worker, the job state machine, the 4 scoring strategies, the
`Repairs`/`LlmCall` tables, the validation stage, and every router, template,
test and script that only served that pipeline were deleted in
`docs/adr/0001-no-legacy-dev-phase.md`. The repository is half the size
and the migration drops 17 tables on upgrade. We do not keep "fallback" code
paths for the old exam pipeline. `make reset` deletes the Postgres volume.
