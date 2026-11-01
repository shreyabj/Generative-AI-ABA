.PHONY: help setup web dev serve ingest build sample bench evaluate redteam ablate test lint docker clean all-checks
.DEFAULT_GOAL := help

PY      ?= python
LANGS   ?= hi,ta,bn
ROWS    ?= 220
N       ?= 300

help:  ## show this help
	@echo "SETU - sub-200ms multilingual voice RAG"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:  make setup && make build && make web && make serve"

setup:  ## install python dependencies
	$(PY) -m pip install -e ".[ingest,dev]"

web:  ## build the frontend into web/dist
	cd web && npm install --no-audit --no-fund && npm run build

dev:  ## run the vite dev server (proxies /api to :8000)
	cd web && npm run dev

ingest:  ## download MS MARCO-XI shards, build corpus + index  (LANGS=hi,ta ROWS=220)
	cd server && $(PY) -m setu.cli ingest --langs $(LANGS) --rows $(ROWS)

build:  ## index the bundled sample corpus (no download, a few seconds)
	cd server && $(PY) -m setu.cli build --sample

sample:  ## regenerate data/sample/corpus.jsonl from the ingested corpus
	cd server && $(PY) -m setu.cli sample --n 420

serve:  ## run the API + UI on :8000
	cd server && $(PY) -m setu.cli serve

bench:  ## latency benchmark -> docs/latency.json  (N=300)
	$(PY) scripts/bench.py --n $(N) --progress

evaluate:  ## retrieval + answer quality vs gold labels -> docs/quality.json
	$(PY) scripts/evaluate.py --n 200 --progress

redteam:  ## guardrail red-team suite -> docs/redteam.json
	$(PY) scripts/redteam.py --verbose

ablate:  ## chunking ablation -> docs/ablation.json (slow, rebuilds the index per config)
	$(PY) scripts/ablate.py --docs 1500 --n 120

test:  ## run the unit tests
	$(PY) -m pytest -q

lint:  ## ruff + tsc
	$(PY) -m ruff check server scripts tests
	cd web && npm run typecheck

all-checks: test lint bench redteam  ## everything CI runs

docker:  ## build and run the container
	docker compose up --build

clean:  ## remove the built index and generated reports
	rm -rf data/index data/corpus web/dist docs/*.json
