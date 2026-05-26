# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview
Production-ready RAG + AI Agent platform. FastAPI microservice packaged as a Python package (`rag-agent`) and Docker image. Uses `uv` as the package manager and Python 3.12.

## First-time setup
```bash
cp .env.example .env          # add OPENROUTER_API_KEY at minimum
make install                  # install deps + pre-commit hooks
make up                       # start Docker services
make migrate                  # create DB schema (run once)
uv run rag-agent create-key mykey  # bootstrap first API key (outputs raw key)
make dev                      # FastAPI on :8000 → /docs for Swagger
```

## Key commands
```bash
make install        # install all optional dep groups + pre-commit hooks
make dev            # FastAPI hot-reload on :8000
make test           # pytest (all tests, min 80% coverage)
make test-unit      # unit tests only (no Docker needed)
make test-integration # integration tests (requires Docker services)
make lint           # ruff check + mypy strict
make format         # ruff format + ruff --fix
make up             # docker compose: app + ChromaDB + Postgres + Redis + MinIO
make down           # stop all services
make logs           # tail app container logs
make migrate        # alembic upgrade head
make migrate-new MSG="describe change"  # autogenerate new Alembic migration
make worker         # start Celery worker (required for async ingest)
make dashboard      # Streamlit admin UI on :8501
make eval           # RAG quality eval via Ragas (requires qa_dataset.json)
make eval-ocr       # OCR accuracy eval → reports/ocr_eval_latest.json
make load           # Locust headless load test: 10 users, 30s against :8000
make build          # build Docker image
make clean          # remove __pycache__, .mypy_cache, .ruff_cache, htmlcov
```

Run a single test file: `uv run pytest tests/unit/test_guardrails.py -v`
Run a single test: `uv run pytest tests/unit/test_guardrails.py::test_toxicity_detected -v`

## CLI (`rag-agent` command)
```bash
rag-agent create-key <name>   # bootstrap API key (writes to DB directly, no HTTP auth needed)
rag-agent serve               # production server (no reload)
rag-agent ingest <path>       # STUB — not yet implemented
rag-agent eval                # STUB — not yet implemented
```

## Local service URLs (after `make up`)
| Service | URL | Credentials |
|---|---|---|
| FastAPI / Swagger | http://localhost:8000/docs | X-API-Key header |
| ChromaDB | http://localhost:8001 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Langfuse | http://localhost:3000 | — |
| Grafana | http://localhost:3001 | admin / admin |
| n8n | http://localhost:5678 | admin / admin |
| Prometheus | http://localhost:9090 | — |

## Required env vars (copy `.env.example` → `.env`)
- `OPENROUTER_API_KEY` — OpenRouter LLM access
- `DATABASE_URL` — async postgres: `postgresql+asyncpg://user:pass@localhost/ragdb`
- `REDIS_URL` — `redis://localhost:6379/0`
- `CHROMA_HOST` / `CHROMA_PORT` — ChromaDB (default port 8001)
- `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`
- `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_HOST` — LLM tracing (optional; host defaults to `http://localhost:3000`; pin SDK to v2 — server image is `langfuse/langfuse:2`)
- `API_SECRET_SALT` — seed for API key hashing (default `changeme` in dev)
- `APP_ENV` — `development` (default) or `production`; production disables `/docs` and CORS wildcard
- `LOG_LEVEL` — `INFO` (default); set to `DEBUG` for verbose output
- `RATE_LIMIT_PER_MINUTE` — per-key request cap (default `60`)
- `OTEL_EXPORTER_ENDPOINT` — OTLP endpoint for OpenTelemetry traces (e.g. `http://jaeger:4317`); empty disables export
- `MINIO_BUCKET` — object storage bucket name (default `rag-documents`)

All settings live in `src/rag_agent/core/config.py` (pydantic-settings `Settings` class). Never add settings elsewhere.

## Docker networking overrides
`.env` uses `localhost` for all service URLs (correct for `make dev`). `docker-compose.yml` overrides these for the `app` and `worker` services so they resolve to Docker service names:

| Variable | `.env` default | Docker override |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` |
| `DATABASE_URL` | `postgresql+asyncpg://…@localhost/…` | `…@postgres:5432/ragdb` |
| `CHROMA_HOST` | `localhost` | `chromadb` |
| `CHROMA_PORT` | `8001` (host port) | `8000` (container port) |
| `MINIO_ENDPOINT` | `localhost:9000` | `minio:9000` |
| `LANGFUSE_HOST` | `http://localhost:3000` | `http://langfuse:3000` |

When adding a new service dependency, add the Docker service name override under `environment:` in both `app` and `worker` sections of `docker-compose.yml`.

## Architecture

```
src/rag_agent/
├── api/
│   ├── main.py          # FastAPI app, Prometheus middleware, exception handlers, router registration
│   └── v1/
│       ├── deps.py      # require_api_key dependency (X-API-Key header)
│       ├── chat.py      # /chat — sync + SSE streaming RAG
│       ├── agent.py     # /agent — LangGraph graph-based agent
│       ├── agent_run.py # /agent/run — ReAct multi-step agent with SSE streaming
│       ├── ingest.py    # /ingest — file upload → Celery task dispatch
│       ├── jobs.py      # /jobs/{id} — Celery task status polling
│       ├── ocr.py       # /ocr — image → structured extraction
│       └── webhooks.py  # MinIO event webhooks → auto-ingest
├── core/
│   ├── config.py        # pydantic-settings (singleton: `settings`)
│   ├── exceptions.py    # domain exceptions (RagAgentError hierarchy)
│   ├── logging.py       # structlog setup (JSON in prod, colored in dev)
│   └── celery_app.py    # Celery app with Redis broker
├── services/
│   ├── rag_pipeline.py  # simple RAG: cache check → retrieve → prompt → generate → cache
│   ├── retriever.py     # hybrid retrieval: dense (ChromaDB) + BM25 + RRF fusion + cross-encoder rerank
│   ├── embedder.py      # async text embedding via OpenRouter-compatible endpoint
│   ├── vector_store.py  # ChromaDB upsert/query wrapper
│   ├── llm_client.py    # async OpenAI-compatible client (complete + stream)
│   ├── semantic_cache.py # Redis cache: embedding similarity threshold (default 0.92)
│   ├── chunker.py       # recursive text chunker (chunk_size=512, overlap=64)
│   ├── document_loader.py # bytes → text (PDF, DOCX, plain text)
│   ├── ingestion_tasks.py # Celery task: load → chunk → embed → upsert
│   ├── graph.py         # LangGraph agent: retrieve → grade → (web_search) → generate → hallucination_check
│   ├── multi_agent.py   # ReAct multi-step agent: plan → tool_loop → synthesize (SSE streaming)
│   ├── agent_memory.py  # per-session message history in Redis with compression
│   ├── agent_tools.py   # Tool registry: web_search (DuckDuckGo), rag_search, calculator, code_runner
│   ├── model_router.py  # model selection: default/quality/ab_test/cheapest/fastest modes + cost tracking
│   ├── guardrails.py    # PII detection (Presidio), toxicity filter, hallucination scoring
│   ├── langfuse_client.py # LLM tracing: trace_generation() context manager; no-op when keys absent
│   └── ocr/
│       ├── pipeline.py  # orchestrator: preprocess → tesseract → vision LLM → validate
│       ├── preprocessor.py # image deskew, denoise, contrast enhancement; doc type detection
│       ├── extractor.py # Tesseract raw text + vision LLM structured extraction
│       └── schemas.py   # ExtractionResult, DocumentType, field schemas
├── models/
│   └── schemas.py       # Pydantic request/response schemas
├── dashboard/
│   └── app.py           # Streamlit admin: ingest, search, metrics, eval results
└── cli.py               # Typer CLI entrypoint (`rag-agent` command)
```

## Request flows

**Chat/RAG** (`POST /api/v1/chat`):
`guardrails.guard_input` → `semantic_cache` hit? → `retriever.retrieve` (dense + BM25 + RRF + cross-encoder) → `llm_client.complete` → cache result → return sources

**Agent** (`POST /api/v1/agent`):
Same retrieval path but routed through the LangGraph state machine in `graph.py`. Grades chunk relevance; falls back to web search if score is low; retries generation up to 2× on hallucination.

**ReAct Agent** (`POST /api/v1/agent/run`):
`multi_agent.py` streams SSE `AgentStep` events: thought → tool_call → observation → repeat → answer. Tools available: `web_search` (DuckDuckGo), `fetch_url`, `rag_search`, `sql_query`, `generate_report`.

**Ingest** (`POST /api/v1/ingest`):
Upload accepted → base64-encoded → `ingest_document` Celery task dispatched → poll `/api/v1/jobs/{task_id}`. Task: `document_loader.load_bytes` → `chunker.chunk_text` → `embedder.embed_texts` → `vector_store.upsert_chunks`.

**OCR** (`POST /api/v1/ocr`):
`preprocessor.preprocess` (deskew/denoise/enhance) → `extractor.run_tesseract` → auto-detect doc type → `extractor.extract_with_vision` (vision LLM) → confidence scoring → `ExtractionResult`.

## API endpoints

All endpoints require `X-API-Key` header.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/chat` | POST | RAG question answering |
| `/api/v1/chat/stream` | GET | Streaming SSE tokens |
| `/api/v1/agent` | POST | LangGraph agent (grade → web fallback → hallucination check) |
| `/api/v1/agent/run` | POST | ReAct multi-step agent (sync) |
| `/api/v1/agent/run/stream` | GET | ReAct agent with SSE step-by-step |
| `/api/v1/agent/run/sessions/{id}` | GET/DELETE | Session history / clear session |
| `/api/v1/ingest/file` | POST | Upload PDF/DOCX/TXT async (max 50 MB) |
| `/api/v1/ingest/text` | POST | Ingest raw text |
| `/api/v1/jobs/{id}` | GET | Celery task status |
| `/api/v1/ocr/extract` | POST | Image → structured JSON extraction |
| `/api/v1/ocr/extract/url` | POST | OCR from URL |
| `/api/v1/ocr/schemas` | GET | List supported document types |
| `/api/v1/keys` | POST/GET | Create / list API keys |
| `/api/v1/keys/{id}` | DELETE | Revoke key |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Testing notes
- `tests/conftest.py` globally overrides `require_api_key` to bypass the DB — any request with a non-empty `X-API-Key` header passes. Do not re-apply auth overrides in individual test files.
- Integration tests in `tests/integration/` hit real FastAPI routes but mock downstream services (LLM, ChromaDB, etc.). They do not require Docker.
- Manual API testing: import `docs/bruno/` into the Bruno client (or use `/docs` Swagger in dev mode).
- `tests/load/locustfile_rag.py` runs Locust load tests; `tests/eval/` contains Ragas evaluation datasets.
- Coverage excludes `cli.py` and `dashboard/app.py` — gaps there are expected, not regressions.

## Infrastructure notes
- **MinIO bucket**: `rag-documents` is auto-created by the `minio-init` service on every `make up`. No manual step needed.
- **Postgres init**: `infra/postgres/init.sql` creates the `n8n` and `langfuse` databases on first boot (only runs when the volume is fresh).
- **Langfuse**: pinned to `langfuse/langfuse:2` (v3 requires ClickHouse). Python SDK pinned to `>=2.0,<3.0` to match.
- **Rebuilding images**: after changing `pyproject.toml` or `Dockerfile`, run `docker compose build app worker` before `make up`.

## Coding conventions
- All code fully typed — mypy strict mode
- Logging: `structlog.get_logger()` only; never `print()`
- Config: always read from `settings` singleton; never hardcode values
- Errors: raise from `core/exceptions.py`; `RagAgentError` subclasses map to HTTP 400/502/500 in `api/main.py`
- Async: `async def` for all IO-bound functions; Celery tasks use `asyncio.run()` internally
- Prometheus metrics: declared at module level as `Counter`/`Histogram`, exported at `GET /metrics`
- Auth: all `/api/v1/*` routes require `X-API-Key` header via `deps.require_api_key`
- Optional deps: `guardrails` extras needed for Presidio; `finetune` for Unsloth; install all with `make install`
- mypy `ignore_errors = true` applies to `guardrails`, `multi_agent`, `vector_store`, and `dashboard/app.py` — strict typing is not enforced there

## OpenRouter usage
```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)
```
Model constants live in `settings`: `default_model` (`google/gemini-flash-1.5`), `quality_model` (`anthropic/claude-3.5-sonnet`). Use `model_router.select_model(mode=...)` for anything beyond the default. Never hardcode model strings outside `config.py`.

## Never do
- Use `print()` instead of structlog
- Skip type annotations
- Write sync DB calls inside async functions
- Hardcode model names, API URLs, or thresholds outside `config.py`
- Add new settings outside the `Settings` class
