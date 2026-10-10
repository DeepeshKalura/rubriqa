# ADR-0001 — No-legacy principle, dev phase

- **Status:** accepted
- **Date:** 2026-10-08
- **Deciders:** owner + explore audit
- **Related:** `docs/architecture.md`, `AGENTS.md` (the working agreement)

## Context

The repository is in active development. It contains two parallel pipelines:

- A Q&A exam pipeline (Organization → Class → Batch → Student → Submission → Page → Reading → Repair → Score), with a job state machine, a separate worker process, 4 scoring strategies, a validation stage, an extraction stage, and 18 tables.
- An entry-evaluation pipeline (Program → Entry → EntryOCR → CleaningRun → EntryScore → EvaluationBatch → ReviewSample), with synchronous processing, one scoring path (`score_entry`), and 15 tables.

The entry pipeline is the product the user actually runs today. The exam pipeline is not on the critical path. It is kept "for backwards compatibility" (per its own module docstring) and powers 18 tables, 26 routes, a 30 KB `SubmissionPipeline`, a separate `worker/` process, a synthetic-data generator, a 20k scale test, and a reader-benchmark harness.

The audit in chat on 2026-10-08 confirmed the owner's instinct: this is two systems in one repo, and almost all of the "industrial" weight comes from the older one.

## Decision

We adopt a **no-legacy principle** for the rest of development.

1. **There is no "old mode" to keep working.** The exam pipeline, the worker, the 4 scoring strategies, the validation stage, the `Repairs` table, the `LlmCall` table, and every router, template, test and script that only serves the exam pipeline are deleted in this ADR. We do not preserve "fallback" code paths.

2. **Data is disposable until the user says otherwise.** This is the dev phase. Migrations will drop tables. The local Postgres volume may be deleted (`make down -v`) at any time. The synthetic dataset is not a record. The user re-runs `make up && make seed` to recreate it.

3. **One product, one name, one set of interfaces.** Every reference to "exam_eval" (logger channels, the FastAPI title, the Makefile header, the docker image name, the `app/main.py` `title=`) is renamed to "entry_eval" in this ADR. The container is `entry-eval`. The DB defaults to `entry_eval`. The two form-parsers stay (Himachal + Ashramshala — that is real domain code).

4. **Smallest set of interfaces wins.** We keep the patterns that earned their keep (`DocumentReader`, `LLMProvider`, `FormParser`, `DriveClient`, `IngestionAdapter`, the `Setting` table). We delete the patterns that did not earn their keep (the 4-strategy scoring factory, the validation stage, the extraction stage, the repair pass — the hybrid reader already escalates whole pages).

5. **Async is added back only when measured.** The job state machine, the worker, the sweeper, the heartbeat, the FOR UPDATE SKIP LOCKED claim, the `JobEvent` log, the retry — all deleted. If the entry pipeline ever needs async, we re-introduce it as `asyncio.gather` in a request handler first, then `BackgroundTasks` second, and only then a worker process. The git log keeps the old code if we ever need to look at it.

## Consequences

### What gets deleted in this ADR (concretely)

- 17 tables: `organizations`, `classes`, `batches`, `enrollments`, `students`, `submissions`, `documents`, `pages`, `reading_runs`, `repairs`, `validation_issues`, `answers`, `scoring_runs`, `scores`, `import_batches`, `jobs`, `job_events`, `llm_calls`. (`exams` and `questions` go too — only `rubrics` and `rubric_criteria` from that file survive.)
- Files: `app/models/submission.py`, `app/models/org.py`, `app/models/job.py` (split — keep `Setting` only), `app/services/pipeline.py`, `app/services/importer.py`, `app/services/rescore.py`, `app/services/stats.py`, `app/services/jobs.py`, `app/services/textutil.py`, `app/pipeline/validation/`, `app/pipeline/scoring/__init__.py`, `app/pipeline/scoring/factory.py`, `app/pipeline/extraction/`, `app/routers/{submissions,exams,classes,students,imports,review,results,jobs}.py`, the `worker/` directory.
- 18 templates + 4 partials.
- The `make test-live`, `make scale-test`, `make bench-readers`, `make e2e` targets.
- 5 scripts: `gen_synthetic.py`, `seed_large.py`, `bench_api.py`, `requeue.py`, `run_e2e.py`. (`seed_demo.py` rewritten to seed an entry program, not an exam class.)
- 8 test files: `test_api.py`, `test_db_and_jobs.py`, `test_e2e_pipeline.py`, `test_import.py`, `test_scoring.py`, `test_validation.py`, `test_readers.py` (trimmed to current reader semantics only), `test_ingestion.py` (kept slim, only canonical page + canonical document).
- The `worker` service in `docker-compose.yml`.
- The `Repairs` table and the `repair_enabled` flag and `build_repairer` (hybrid chain already does the right thing).
- The `LlmCall` table, the `usage_hook` and the per-call token log (the API dashboard already does this; the entry row already has `input_tokens`/`output_tokens`/`total_tokens`).
- The `Job`, `JobEvent`, `JobState`, `JobType`, `ACTIVE_STATES`, `ImportBatch` models entirely.
- The `Organization`, `Klass`, `Batch`, `Enrollment`, `Student` hierarchy. The Program table already carries `school_name`, `class_name`, `student_external_id`, `full_name` as free text — that's enough.
- The `Submission`/`Document`/`Page`/`ReadingRun`/`Answer`/`Score`/`ScoringRun` chain.
- The `ValidationIssue` model and the validation pipeline stage.
- The exam column on `Exam` (`scoring_strategy`, `enable_feedback` — the `eval_mode`/`scoring_strategy` flags in `app/config.py` cover it).

### What stays (because it earned it)

- The entry-mode models (`Program`, `Entry`, `EntryOCR`, `CleaningRun`, `EntryScore`, `EntryScoreCriterion`, `EvaluationBatch`, `ReviewSample`, `DriveFile`, `GoogleCredential`).
- `Rubric` and `RubricCriterion` (used by entry mode for per-criterion scoring).
- `Setting` and the Settings page + `app_settings.py` (lets a non-technical user flip flags at runtime — real value, 90 lines).
- The form parsers (`app/pipeline/ingestion/forms.py` — 30 lines of Hindi/Marathi theme slugs no library gives you).
- The two OCR paths (`tesseract`, `minimax_vision`) and the hybrid chain reader.
- The 2-way sandbox at `/sandbox` (`app/routers/sandbox.py` + `app/templates/sandbox.html`).
- The cleaning pass (`app/pipeline/cleaning.py`) that keeps both `original_text` and `reconstructed_text`.
- The batch orchestration (`app/services/batching.py`) with `eval_batch_size` 200-300, sample-based review.
- The LLM provider interface (`app/providers/llm/base.py` + `minimax.py` + `fake.py`).
- The Google Drive local mirror + real-Drive provider (`app/pipeline/ingestion/google_drive.py`).
- `app/schemas/` (canonical pydantic contracts).
- `structlog` with secret redaction (`app/logging.py`).
- The Docker-first / one-command contract.

### What changes for the user

- `make up` starts one process, not three.
- The container image is `entry-eval:local`. Postgres DB defaults to `entry_eval`.
- The FastAPI title is "Entry Eval". All log channels are `entry_eval.*`.
- After migration, the local Postgres volume is reset (the 17 legacy tables and the `llm_calls` and `repairs` tables are dropped). The user re-runs `make seed` to recreate demo data.
- The Makefile has 10 targets instead of 15.

### What is explicitly NOT in this ADR

- No new feature. This ADR only deletes. New features (program templates, rubric library, multi-language cleaning evaluation, drive multi-account, etc.) get their own ADR.
- No "let's add an event bus". No "let's add Redis". No "let's add Celery". The dev phase is for measuring, not for queueing.
- No "let's keep the old exam pipeline behind a feature flag". Feature flags are for replacing the present, not for keeping the past.

## Verification

After applying this ADR:

1. `make up` succeeds. `make test` passes against `entry_eval_test`. (Snapshot removed: the dev phase doesn't track memory.)
2. The FastAPI `/docs` page lists only the entry-mode routes: dashboard, programs, entries, batches, sandbox, settings, google, files.
3. `psql -h 127.0.0.1 -p 5435 -U entry_eval -d entry_eval -c "\dt"` shows 14 tables (entry-mode + rubrics + settings + google_credentials + alembic_version).
4. `make bench-readers` no longer exists. The reports in `reports/` are kept as historical artefacts and are not regenerated.
5. The repository size is roughly half what it was on 2026-10-08.

## Reversal

If, at some future point, we need to grade Q&A exams from photos again, we will not re-introduce the deleted code from git. We will write a new ADR describing the new requirement, the new pipeline shape, and the new constraints (e.g. a real exam mode for a real customer). The old code is in the git history; it is not a maintenance burden by being absent.

## Cross-references

- `docs/architecture.md` — updated to describe the entry pipeline as the only pipeline.
- `AGENTS.md` — updated to drop the `make test-live`, `make scale-test`, `make bench-readers`, `make e2e` targets from the command surface.
- `TASTE.md` — appended a milestone entry for "exam pipeline removed, dev phase no-legacy principle adopted".
- The reader-benchmark report in `reports/reader_bench_*.json` is kept as a historical measurement, with a note that it was run against synthetic data and is not being re-run.
