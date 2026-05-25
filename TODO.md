# TODO — rag-agent

## Phase 0 — Fondations
- [x] Structure projet (`src/` layout, package Python)
- [x] `pyproject.toml` (build, deps, ruff, mypy, pytest)
- [x] `Makefile`
- [x] `.pre-commit-config.yaml`
- [x] `CLAUDE.md`
- [x] `Dockerfile` multi-stage
- [x] `docker-compose.yml` (app + ChromaDB + Postgres + Redis + MinIO + Langfuse + Grafana)
- [x] `.env.example`
- [ ] `alembic` init + migration initiale
- [x] `src/rag_agent/core/config.py` (pydantic-settings)
- [x] `src/rag_agent/core/logging.py` (structlog JSON)
- [x] `src/rag_agent/core/exceptions.py`
- [x] `src/rag_agent/api/main.py` (FastAPI app + handlers)
- [x] `src/rag_agent/cli.py` (typer)
- [x] `src/rag_agent/__init__.py` (version)

## Phase 0b — Transverses
- [ ] OpenTelemetry middleware
- [ ] Prometheus metrics (`/metrics`)
- [ ] Dashboard Grafana (JSON dans `infra/grafana/dashboards/`)
- [ ] Langfuse tracing wrapper
- [ ] `src/rag_agent/services/semantic_cache.py`
- [ ] Endpoint SSE `/api/v1/chat/stream`
- [ ] Auth API key (table `api_keys` + dépendance FastAPI)
- [ ] Rate limiting (`slowapi`)
- [ ] Tests de charge Locust (`tests/load/`)
- [ ] Dataset eval + script Ragas (`tests/eval/`, `scripts/eval_rag.py`)
- [ ] Job CI `eval` (GitHub Actions)

## Phase 0c — Différenciants
- [ ] LangGraph graph (`src/rag_agent/services/graph.py`)
- [ ] Guardrails PII (Presidio) + hallucination scoring
- [ ] `src/rag_agent/services/model_router.py` (A/B test, cheapest, fastest)
- [ ] Event-driven ingestion MinIO webhook + Celery
- [ ] Streamlit dashboard (`src/rag_agent/dashboard/`)
- [ ] Fine-tuning pipeline (`scripts/finetune/`)

## Phase 1 — RAG documentaire
- [x] Ingestion PDF/DOCX/HTML + OCR fallback (`services/document_loader.py`)
- [x] Chunking récursif avec overlap (`services/chunker.py`)
- [x] Embeddings via OpenRouter (`services/embedder.py`)
- [x] ChromaDB vector store wrapper (`services/vector_store.py`)
- [x] Retrieval hybride dense + BM25 + RRF + cross-encoder (`services/retriever.py`)
- [x] Cache sémantique (`services/semantic_cache.py`)
- [x] Pipeline RAG complet + streaming (`services/rag_pipeline.py`)
- [x] Celery ingestion async (`services/ingestion_tasks.py`)
- [x] LLM client OpenRouter (`services/llm_client.py`)
- [x] Endpoint `POST /api/v1/chat`
- [x] Endpoint `GET /api/v1/chat/stream` (SSE)
- [x] Endpoint `POST /api/v1/ingest/file`
- [x] Endpoint `POST /api/v1/ingest/text`
- [x] Endpoint `GET /api/v1/jobs/{id}`
- [x] Schémas Pydantic (`models/schemas.py`)
- [ ] Endpoint `/api/v1/evaluate` (Ragas)
- [x] Tests unitaires chunker, document_loader, BM25
- [x] Tests intégration chat + ingest endpoints

## Phase 2 — Agent multi-étapes (LangGraph) ✅
- [x] Tools : web_search (DuckDuckGo), fetch_url, rag_search, sql_query, generate_report
- [x] ReAct agent LangGraph avec boucle conditionnelle (max 8 steps)
- [x] Memory Redis par session avec compression LLM automatique
- [x] `POST /api/v1/agent/run` — exécution synchrone
- [x] `GET /api/v1/agent/run/stream` — SSE step-by-step
- [x] `GET /api/v1/agent/run/sessions/{id}` — historique session
- [x] `DELETE /api/v1/agent/run/sessions/{id}` — suppression session
- [x] Workflow n8n veille concurrentielle (infra/n8n/workflow_competitive_intel.json)
- [x] n8n dans docker-compose
- [x] Tests : 18 nouveaux tests (tools, memory, endpoint)

## Phase 3 — OCR pipeline avancé
- [ ] Preprocessing image (deskew, denoise)
- [ ] Extraction structurée JSON via OpenRouter vision
- [ ] Score de confiance par champ
- [ ] Tests avec dataset annoté

## CI/CD
- [ ] `.github/workflows/ci.yml` (lint → test → build → security)
- [ ] `.github/workflows/cd.yml` (tag → push GHCR → release)
- [ ] Badges README

## Documentation
- [ ] README avec architecture diagram (Mermaid)
- [ ] `docs/finetune.md`
- [ ] Postman/Bruno collection pour l'API
