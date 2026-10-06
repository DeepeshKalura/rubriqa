# Exam Eval - local multilingual handwritten exam evaluation.
# Every command runs through Docker.

COMPOSE = docker compose

.PHONY: up down logs ps build migrate shell test test-live seed demo scale-test bench-readers screenshots e2e fmt

up:            ## start db + web + worker
	$(COMPOSE) up --build -d
	@echo "web:  http://127.0.0.1:8090"
	@echo "db:   127.0.0.1:5435"

down:          ## stop everything (keeps volumes and ./data)
	$(COMPOSE) down

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
	$(COMPOSE) run --rm -e DATABASE_URL=postgresql+asyncpg://exameval:exameval@db:5432/exam_eval_test web bash -c "python -m scripts.ensure_test_db && pytest -x -q"

test-live:     ## tests marked as live (real MiniMax calls, costs tokens)
	$(COMPOSE) run --rm web pytest -q -m live

seed:          ## create org/class/batches/exam AND import ./data/demo
	$(COMPOSE) run --rm web python -m scripts.seed_demo --docs /data/demo

demo:          ## generate synthetic dataset AND seed it
	$(COMPOSE) run --rm web python -m scripts.gen_synthetic --out /data/demo
	$(COMPOSE) run --rm web python -m scripts.seed_demo --docs /data/demo

e2e:           ## run the full pipeline against the demo dataset and verify in DB
	$(COMPOSE) run --rm web python -m scripts.run_e2e

scale-test:    ## seed ~20k submissions and benchmark API queries
	$(COMPOSE) run --rm web python -m scripts.seed_large --submissions 20000
	$(COMPOSE) run --rm web python -m scripts.bench_api

bench-readers: ## compare reader modes against the labeled synthetic dataset
	$(COMPOSE) run --rm web python -m scripts.bench_readers --data /data/demo

screenshots:   ## Playwright verification + screenshots (host browsers)
	python3 scripts/screenshot.py

fmt:
	$(COMPOSE) run --rm web python -m compileall -q app worker scripts
