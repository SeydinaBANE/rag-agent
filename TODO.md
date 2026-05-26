# TODO — rag-agent

## Phase 0 — Fondations
- [x] Structure projet (`src/` layout, package Python)
- [x] `pyproject.toml` (build, deps, ruff, mypy, pytest)
- [x] `Makefile`
- [x] `.pre-commit-config.yaml`
- [x] `CLAUDE.md`
- [x] `Dockerfile` multi-stage
- [x] `docker-compose.yml` (app + worker + frontend + ChromaDB + Postgres + Redis + MinIO + Langfuse + Grafana + Jaeger + Flower + n8n)
- [x] `.env.example`
- [x] `alembic` init + migration initiale
- [x] `src/rag_agent/core/config.py` (pydantic-settings)
- [x] `src/rag_agent/core/logging.py` (structlog JSON)
- [x] `src/rag_agent/core/exceptions.py`
- [x] `src/rag_agent/api/main.py` (FastAPI app + handlers)
- [x] `src/rag_agent/cli.py` (typer)
- [x] `src/rag_agent/__init__.py` (version)

## Phase 0b — Transverses
- [x] OpenTelemetry middleware
- [x] Prometheus metrics (`/metrics`)
- [x] Dashboard Grafana (JSON dans `infra/grafana/dashboards/`)
- [x] Langfuse tracing wrapper
- [x] `src/rag_agent/services/semantic_cache.py`
- [x] Endpoint SSE `/api/v1/chat/stream`
- [x] Auth API key (table `api_keys` + dépendance FastAPI)
- [x] Rate limiting (`slowapi`)
- [x] Tests de charge Locust (`tests/load/`)
- [x] Dataset eval + script Ragas (`tests/eval/`, `scripts/eval_rag.py`)
- [x] Job CI `eval` (GitHub Actions)

## Phase 0c — Différenciants
- [x] LangGraph graph (`src/rag_agent/services/graph.py`)
- [x] Guardrails PII (Presidio) + hallucination scoring
- [x] `src/rag_agent/services/model_router.py` (A/B test, cheapest, fastest)
- [x] Event-driven ingestion MinIO webhook + Celery
- [x] Streamlit dashboard (`src/rag_agent/dashboard/`)
- [x] Fine-tuning pipeline (`scripts/finetune/`)

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
- [x] Endpoint `/api/v1/evaluate` (Ragas)
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
- [x] Preprocessing image (deskew, denoise)
- [x] Extraction structurée JSON via OpenRouter vision
- [x] Score de confiance par champ
- [x] Tests avec dataset annoté

## CI/CD
- [x] `.github/workflows/ci.yml` (lint → test → build → security)
- [x] `.github/workflows/cd.yml` (tag → push GHCR → release)
- [x] `.github/workflows/eval.yml` (Ragas evaluation hebdomadaire)
- [x] Badges README

## Documentation
- [x] README avec architecture diagram (Mermaid)
- [x] `docs/api.md` — référence complète endpoints + observabilité
- [x] `docs/finetune.md`
- [x] `frontend/README.md`
- [x] Postman/Bruno collection pour l'API (`docs/bruno/`)

## Frontend — Interface chat Next.js ✅
- [x] `frontend/` scaffolding (Next.js 14, App Router, TypeScript, Tailwind)
- [x] `lib/api.ts` — `streamChat()` SSE client via fetch+ReadableStream (supporte X-API-Key)
- [x] `components/ChatWindow.tsx` — liste messages + auto-scroll
- [x] `components/MessageBubble.tsx` — Markdown (react-markdown + remark-gfm), curseur streaming
- [x] `components/InputBar.tsx` — textarea auto-resize, Enter=envoi
- [x] `components/SettingsModal.tsx` — clé API localStorage + reset session
- [x] `components/MetaBadges.tsx` — badges cache / confiance / tokens
- [x] `components/SourceList.tsx` — sources collapsibles avec score
- [x] `next.config.mjs` — rewrite `/api/*` → `API_URL` (localhost en dev, `app:8000` en Docker)
- [x] `frontend/Dockerfile` — multi-stage Node 20 Alpine, standalone output
- [x] Makefile: `frontend-install`, `frontend-dev`, `frontend-build`
- [x] docker-compose: service `frontend` sur :3003

## Hardening config ✅
- [x] `config.py` — validation Pydantic sur tous les seuils (rate_limit, hallucination, cache similarity)
- [x] `config.py` — 12 nouveaux paramètres exposés (max_agent_steps, agent_temperature, web_search_results…)
- [x] `config.py` — validator production : erreur si le sel par défaut est utilisé en prod # pragma: allowlist secret
- [x] `main.py` — rate limiting par clé API (pas par IP)
- [x] `agent_run.py` — sessions isolées par scope SHA-256(api_key)[:16]
- [x] `chat.py` + `agent_run.py` — SSE error events structurés
- [x] `graph.py`, `multi_agent.py`, `agent_tools.py`, `retriever.py` — plus de constantes hardcodées

## Pending
- [ ] `cli.py` `ingest` command — stub only (`# TODO: call ingestion service`)
- [ ] `cli.py` `eval` command — stub only (`# TODO: call eval script`)
- [ ] Remonter la couverture de test à 80% (seuil actuel en flux)
