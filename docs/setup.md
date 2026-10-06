# Setup

## 1. Requirements

- Docker + Docker Compose (any recent version)
- ~2 GB free disk for images, plus space for `./data`
- A MiniMax API key (only needed for real reading/scoring; `LLM_PROVIDER=fake`
  runs everything without one)

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
# either name works; your existing files use LLM_API_KEY
MINIMAX_API_KEY=sk-...
# or
LLM_API_KEY=sk-...

# feature flags (defaults shown)
LLM_PROVIDER=minimax
READER_MODE=hybrid
HYBRID_READER_CHAIN=pdf_text,tesseract,minimax_vision
SCORING_STRATEGY=hybrid
REPAIR_ENABLED=true
GOOGLE_FORMS_ENABLED=false
```

`.env` is git-ignored. The key is never logged, never sent to the browser, and
the Settings page only shows a masked form (`sk-cp…ab (len 125)`).

For Google Forms you also need `APP_SECRET_KEY`:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## 3. Start

```bash
make up
```

- Web UI: <http://127.0.0.1:8090>
- Postgres: `127.0.0.1:5435` (user/pass/db: `exameval`/`exameval`/`exam_eval`)

Migrations run automatically at web startup (`RUN_MIGRATIONS=true`). To run
them manually: `make migrate`.

## 4. First data

```bash
make seed     # org/class/batches/students/exam/questions/rubric
make demo     # + synthetic dataset (Hindi+English, text/PDF/image/mixed) and import
```

Watch processing:

```bash
make logs     # or open /jobs in the UI
```

`make e2e` runs the whole pipeline and prints a verified summary plus a JSON
evidence file in `reports/`.

## 5. Importing your own data

Filename/path convention: each file must contain the student roll number,
matching `Student.external_id`:

```
Class 8/Batch A/STU001/answers_page1.jpg
STU002_sheet.pdf
```

Use **Import → Upload files** (multiple files or a ZIP) or **Scan a folder**
inside `./data`. The confirmation screen shows the dedupe diff before anything
is written:

```
28 files discovered · 27 already imported · 0 new · 1 unmatched
```

## 6. Google Forms (optional)

1. In Google Cloud Console create a project → enable **Google Forms API**.
2. OAuth consent screen → External → add yourself as a test user.
3. Credentials → Create OAuth client ID → **Web application**.
   Authorized redirect URI: `http://127.0.0.1:8090/google/oauth/callback`.
4. Put the client ID/secret into `.env`, set `GOOGLE_FORMS_ENABLED=true`
   (`GOOGLE_FORMS_PROVIDER=google`), restart.
5. In the app open **Google Forms → Connect Google account**, approve, then
   paste a Form ID and import responses.

Without credentials, set `GOOGLE_FORMS_PROVIDER=mock` — the connector then
serves the recorded fixture in
`app/pipeline/ingestion/fixtures/google_form_responses.json`, which is what
the tests use.

## 7. Tests

```bash
make test        # full suite against scratch DB exam_eval_test
make test-live   # 4 tests against the real MiniMax API (costs tokens)
```

## 8. Benchmarks and evidence

```bash
make bench-readers   # reader accuracy on the labeled synthetic dataset
make scale-test      # 20k submissions + API latency
make screenshots     # Playwright UI checks + docs/screenshots/*.png
```

Outputs land in `reports/`.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| Web container exits, log says `Path doesn't exist: .../alembic` | Rebuild: `make build && make up` |
| `missing_api_key` in job errors | Set `MINIMAX_API_KEY`/`LLM_API_KEY` in `.env`, `docker compose restart web worker` |
| Pages become `deferred` | MiniMax unreachable; jobs retry automatically, or `make logs` to confirm |
| Job stuck in `running` | Sweeper requeues it within `JOB_STUCK_MINUTES`; or open the job and press Retry |
| Duplicate import check | Re-run import — the diff should show everything as already imported |
| Port conflict on 8090/5435 | Change the host side of the port mappings in `docker-compose.yml` |
