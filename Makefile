.PHONY: help install dev test lint format build up down migrate dashboard eval load

PYTHON := python3.12
UV     := uv
APP    := src/rag_agent

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package + dev dependencies
	$(UV) pip install -e ".[dev,eval,guardrails,dashboard]"
	pre-commit install

dev: ## Run FastAPI in hot-reload mode
	$(UV) run uvicorn rag_agent.api.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run tests with coverage
	$(UV) run pytest

test-unit: ## Run only unit tests
	$(UV) run pytest tests/unit -v

test-integration: ## Run only integration tests
	$(UV) run pytest tests/integration -v

lint: ## Lint (ruff) + type check (mypy)
	$(UV) run ruff check $(APP) tests
	$(UV) run mypy $(APP)

format: ## Auto-format with ruff
	$(UV) run ruff format $(APP) tests
	$(UV) run ruff check --fix $(APP) tests

build: ## Build Docker image
	docker build -t rag-agent:latest .

up: ## Start all services via docker compose
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Tail app logs
	docker compose logs -f app

migrate: ## Run Alembic migrations
	$(UV) run alembic upgrade head

migrate-new: ## Create new Alembic migration (usage: make migrate-new MSG="add users table")
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

dashboard: ## Launch Streamlit admin dashboard
	$(UV) run streamlit run src/rag_agent/dashboard/app.py --server.port 8501

eval: ## Run RAG evaluation against qa_dataset.json
	$(UV) run python scripts/eval_rag.py

eval-ocr: ## Run OCR accuracy evaluation against annotated dataset
	$(UV) run python scripts/eval_ocr.py --output reports/ocr_eval_latest.json

load: ## Run Locust load test (headless, 10 users, 30s)
	$(UV) run locust -f tests/load/locustfile_rag.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000

worker: ## Start Celery worker
	$(UV) run celery -A rag_agent.core.celery_app worker --loglevel=info

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -rf .coverage htmlcov dist .mypy_cache .ruff_cache
