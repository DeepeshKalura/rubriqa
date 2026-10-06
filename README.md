# Exam Eval — local multilingual handwritten exam evaluation

A local-first system that takes student exam work (text files, PDFs, phone
photos, Google Forms), reads it (printed **and** handwritten, **Hindi +
English**), validates the extraction, scores it against configurable rubrics,
and shows results in a local web dashboard.

Built to run on an ordinary office laptop (Intel i5 / 16 GB RAM, no GPU) with
two commands:

```bash
cp .env.example .env        # put your MiniMax key in MINIMAX_API_KEY or LLM_API_KEY
make up                     # docker compose up --build -d
```

Open <http://127.0.0.1:8090>. Postgres is exposed on `127.0.0.1:5435`.

## What works today

- **Ingestion** — text/markdown/CSV/HTML, PDFs (per-page text-vs-scan
  classification), images (JPG/PNG/TIFF/WebP), and a Google Forms connector
  (real OAuth flow + mock provider behind flags).
- **Reading** — pluggable readers: PDF text layer, Tesseract (`eng+hin`),
  MiniMax M3 vision, and a hybrid chain with confidence escalation. A repair
  path re-reads low-confidence pages with vision and keeps the original text.
- **Validation** — structural checks (empty pages, failed pages, duplicate
  questions, missing answers) with severity levels; blocking errors stop
  scoring and route the submission to the review queue.
- **Extraction** — LLM segmentation of the transcript into per-question
  answers, with a regex fallback that works offline.
- **Scoring** — exact, numeric, rubric-LLM and hybrid strategies; weighted
  rubric criteria with code-enforced knockout rules; per-criterion rationale;
  optional student feedback behind a flag. Every scoring run is versioned.
- **Jobs** — Postgres-backed queue with a separate worker container
  (`FOR UPDATE SKIP LOCKED`), heartbeats, stale-job sweeper, retry with
  checkpoint resume and per-page idempotency.
- **UI** — server-rendered Jinja2 + HTMX: dashboard, classes → batches →
  students, exams and rubrics, import wizard with dry-run duplicate diff,
  live job timeline, submission viewer (page image vs OCR vs repair),
  review queue, results with CSV export, settings with runtime feature flags.

## Command surface

All commands run through Docker.

```bash
make up            # start db + web + worker
make down          # stop (keeps volumes and ./data)

make test          # 96 tests: unit + integration + API + E2E (fake provider, no network)
make test-live     # 4 live MiniMax tests (costs a few tokens)

make demo          # generate the synthetic dataset and seed it
make e2e           # full pipeline over demo data, verified in the DB
make scale-test    # 20k synthetic submissions + API latency benchmark
make bench-readers # reader-mode accuracy comparison -> reports/
make screenshots   # Playwright UI checks + docs/screenshots/*.png

make logs / make ps / make shell / make migrate
```

## Feature flags (`.env` or Settings page)

| Flag | Values | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | `minimax`, `fake` | `minimax` | Swap the LLM backend |
| `READER_MODE` | `pdf_text`, `tesseract`, `minimax_vision`, `hybrid` | `hybrid` | Document reader |
| `HYBRID_READER_CHAIN` | e.g. `pdf_text,tesseract,minimax_vision` | — | Reader composition |
| `REPAIR_ENABLED` | `true`/`false` | `true` | Vision repair of low-confidence OCR |
| `SCORING_STRATEGY` | `hybrid`, `rubric_llm`, `exact`, `numeric` | `hybrid` | Scoring engine |
| `SCORING_ENABLE_FEEDBACK` | `true`/`false` | `false` | Generate student feedback text |
| `GOOGLE_FORMS_ENABLED` / `GOOGLE_FORMS_PROVIDER` | `google`/`mock` | off / mock | Forms connector |

Flags can also be changed at runtime on the Settings page (stored in the
`settings` table, overriding `.env` for the next job).

## Architecture in one picture

```
upload/scan ──► ingestion adapters ──► canonical document (pages)
                                          │
                        ┌─────────────────┼──────────────────┐
                        ▼                 ▼                  ▼
                  pdf_text reader   tesseract reader   minimax vision
                        └────────► hybrid chain ◄──────────┘
                                     │
                              validation (structural)
                                     │
                             LLM answer extraction
                                     │
                        exact / numeric / rubric LLM / hybrid
                                     │
                       Postgres: submissions→scores (versioned)
                                     │
                          Jinja2 + HTMX dashboard
```

Provider boundaries are small protocols: `DocumentReader`, `LLMProvider`,
`ScoringStrategy`, `IngestionAdapter`. Replacing one never touches another.

## Documentation

- `docs/architecture.md` — decisions, rejected alternatives, failure-mode analysis
- `docs/setup.md` — install, configuration, Google console steps
- `docs/limitations.md` — the honest list of what is not proven yet
- `reports/` — E2E results, reader benchmark, scale benchmark (generated)
- `docs/screenshots/` — verified UI screenshots
- `TASTE.md` — project preference log with milestone scores
- `AGENTS.md` — working agreement and command surface

## Measured numbers (this machine, synthetic data)

| Measurement | Result |
|---|---|
| Test suite | 96 passed (no network) + 4 live MiniMax tests |
| Demo E2E | 26 documents / 31 pages / 20 answers / 47 scores in ~30 s, 0 failed |
| Reader comparison (15 synthetic pages) | pdf_text recall 0.23 (text PDFs only), tesseract 0.72, minimax_vision 0.92, hybrid 0.83 |
| Scale (20,005 submissions, 200k answers) | API p50 10–126 ms, payloads 12–16 KB |
| Memory (idle, 3 containers) | web 83 MB, worker 83 MB, db 182 MB |

**Honest limitations:** no real client handwriting data has been benchmarked
yet; the numbers above come from synthetic pages. The 20k scale test measures
the data layer and UI pagination only — it does not run OCR/LLM on 20k exams.
Full list: `docs/limitations.md`.
