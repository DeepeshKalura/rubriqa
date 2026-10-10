# Entry Eval - local multilingual handwritten entry evaluation.
# Every command runs through Docker.

COMPOSE = docker compose

.PHONY: up down logs ps build migrate shell test seed demo screenshots fmt reset

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
