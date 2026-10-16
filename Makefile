# Entry Eval - local multilingual handwritten entry evaluation.
# Every command runs through Docker.

COMPOSE = docker compose

.PHONY: up down logs ps build migrate shell test check seed demo screenshots fmt reset score-replay bench-scoring

up:            ## start db + web
	$(COMPOSE) up --build -d
	@echo "web:  http://127.0.0.1:8090"
	@echo "db:   127.0.0.1:5435"

down:          ## stop everything (keeps volumes and ./data)
	$(COMPOSE) down

reset:         ## stop + delete the Postgres volume (full reset; dev phase)
	$(COMPOSE) down -v
	rm -rf data/originals data/derived data/exports data/uploads data/drive_mirror

logs:          ## tail logs
	$(COMPOSE) logs -f --tail=100

ps:            ## show container status
	$(COMPOSE) ps

build:         ## rebuild images
	$(COMPOSE) build

migrate:       ## apply DB migrations (web runs them at startup too)
	$(COMPOSE) run --rm web alembic upgrade head

shell:         ## python shell inside the app image
	$(COMPOSE) run --rm web python

test:          ## unit + integration + api tests against a scratch database
	$(COMPOSE) run --rm \
		-e DATABASE_URL=postgresql+asyncpg://entryeval:entryeval@db:5432/entry_eval_test \
		-e STORAGE_ROOT=/tmp/entry-eval-tests \
		-e GOOGLE_DRIVE_LOCAL_ROOT=/tmp/entry-eval-tests/drive_mirror \
		web bash -c "python -m scripts.ensure_test_db && pytest -q"

seed:          ## seed a demo program + a few entries
	$(COMPOSE) run --rm web python -m scripts.seed_demo

demo:          ## seed + create sample program (alias for seed)
	$(COMPOSE) run --rm web python -m scripts.seed_demo

screenshots:   ## Playwright UI verification + screenshots (host browsers)
	python3 scripts/screenshot.py
	python3 scripts/screenshot_sandbox.py

fmt:
	$(COMPOSE) run --rm web python -m compileall -q app scripts

check:         ## prove the tree is sound: compileall + guards + full suite
	@echo "== 1/3 byte-compile (syntax only; cannot catch AttributeError) =="
	$(COMPOSE) run --rm web python -m compileall -q app scripts
	@echo "== 2/3 static guards =="
	$(COMPOSE) run --rm \
		-e STORAGE_ROOT=/tmp/entry-eval-tests \
		web pytest -q \
			tests/test_settings_references.py \
			tests/test_golden_alignment.py \
			tests/test_logging_helpers.py
	@echo "== 3/3 full suite =="
	$(MAKE) test
	@echo ""
	@echo "check: OK"
	@echo ""
	@echo "Not covered here: there is no CI. This must be run by hand or by"
	@echo "something else, or these guards protect nothing. See Phase 5 notes in"
	@echo "docs/architecture.md."

score-replay:  ## re-run rubric scoring on entries that have a human_score
	$(COMPOSE) run --rm web python -m scripts.score_replay

bench-scoring: ## read reports/replay_<date>.json -> reports/scoring_bench_<date>.json
	$(COMPOSE) run --rm web python -m scripts.bench_scoring
