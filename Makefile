SHELL := /bin/bash
# Use the project venv automatically once `make setup` has created it.
export PATH := $(abspath .venv/bin):$(PATH)
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

TRAIN_ARGS ?=
EVAL_ARGS ?=
SIM_ARGS ?=
PORT ?= 8000
API_REPLICAS ?= 2

.PHONY: help setup dirs dvc-init data download-model train train-smoke eval eval-smoke promote test lint serve up down \
	    drift simulate retrain check-retrain all clean k8s-apply

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## create .venv, install dependencies, init git + DVC, create folders
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -q
	@$(MAKE) --no-print-directory dirs dvc-init

dirs:
	@mkdir -p data reports models/hf_cache models/production mlruns

dvc-init: ## init DVC with the local remote ./.dvc-storage
	@$(PY) scripts/dvc_snapshot.py --init-only

data: dirs ## generate the synthetic dataset (train/val JSONL) and version it with DVC
	$(PY) scripts/make_dataset.py
	@$(PY) scripts/dvc_snapshot.py -m "data: regenerate synthetic dataset" || true

download-model: dirs ## cache the base model from the Hugging Face Hub (free)
	$(PY) scripts/download_model.py

train: ## full LoRA fine-tuning (CPU: tens of minutes)
	$(PY) -m src.train $(TRAIN_ARGS)

train-smoke: ## 20-step LoRA run (a few minutes on CPU)
	$(PY) -m src.train --smoke $(TRAIN_ARGS)

eval: ## evaluate the registry `candidate` (writes reports/eval.json)
	$(PY) -m src.evaluate --model candidate $(EVAL_ARGS)

eval-smoke: ## evaluate on a handful of samples
	$(PY) -m src.evaluate --model candidate --smoke $(EVAL_ARGS)

promote: ## quality gate: candidate -> production (non-zero exit if it fails), then DVC-version the model
	$(PY) scripts/promote_model.py
	@$(PY) scripts/dvc_snapshot.py -m "model: promote production adapter" || true

test: ## unit tests
	$(PY) -m pytest

lint: ## ruff
	$(PY) -m ruff check .

serve: dirs ## run the API locally (default :8000)
	$(PY) -m uvicorn src.serve.app:create_app --factory --host 0.0.0.0 --port $(PORT) --no-access-log

up: dirs ## docker compose: api replicas, nginx :8080, mlflow :5000, prometheus :9090, grafana :3000
	docker compose up -d --build --scale api=$(API_REPLICAS)

down: ## stop the compose stack
	docker compose down

drift: ## compute data drift -> reports/drift.html + reports/drift.json
	$(PY) -m src.monitor.drift

simulate: ## send normal then shifted traffic (SIM_ARGS="--url http://localhost:8080 --shifted 10")
	$(PY) scripts/simulate_traffic.py $(SIM_ARGS)

retrain: ## feedback -> merge (new DVC version) -> train -> eval -> gate -> promote
	$(PY) scripts/export_feedback.py
	$(PY) scripts/merge_feedback.py
	@$(PY) scripts/dvc_snapshot.py -m "data: merge user feedback" || true
	$(PY) -m src.train $(TRAIN_ARGS)
	$(PY) -m src.evaluate --model candidate $(EVAL_ARGS)
	$(PY) scripts/promote_model.py

check-retrain: ## drift/feedback-triggered retraining (add ARGS=--dry-run to only decide)
	$(PY) scripts/check_and_retrain.py $(ARGS)

all: data download-model train-smoke eval-smoke promote test ## whole pipeline on the smoke profile (after `make setup`)

k8s-apply: ## apply k8s manifests (kind/minikube)
	kubectl apply -k k8s/

clean: ## remove generated state (keeps code)
	rm -rf mlruns mlflow.db models/adapters models/merged models/production/* data/monitoring.db reports/*.json reports/*.html reports/*.jsonl
