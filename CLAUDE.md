# rag-agent — Claude Code Instructions

## Project overview
Production-ready RAG + AI Agent platform. FastAPI microservice packaged as a Python package and Docker image.

## Key commands
```bash
make install    # install deps + pre-commit hooks
make dev        # FastAPI hot-reload on :8000
make test       # pytest with coverage (min 80%)
make lint       # ruff + mypy
make format     # auto-fix lint issues
make up         # docker compose (app + ChromaDB + Postgres + Redis + MinIO)
make migrate    # alembic upgrade head
make eval       # RAG evaluation via Ragas
make dashboard  # Streamlit on :8501
```

## Required env vars (copy .env.example → .env)
- `OPENROUTER_API_KEY` — OpenRouter API key (compatible openai client)
- `DATABASE_URL` — async postgres: `postgresql+asyncpg://user:pass@localhost/ragdb`
- `REDIS_URL` — `redis://localhost:6379/0`
- `CHROMA_HOST` / `CHROMA_PORT` — ChromaDB connection
- `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`
- `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` — LLM tracing (optional)

## Architecture
```
src/rag_agent/
├── api/v1/          # FastAPI routers (chat, ingest, jobs, keys, webhooks)
├── core/            # config (pydantic-settings), logging (structlog), exceptions, celery
├── services/        # rag_pipeline, llm_client, semantic_cache, graph (LangGraph),
│                    # guardrails, model_router, ocr, embedder
├── models/          # SQLAlchemy ORM + Pydantic schemas
└── dashboard/       # Streamlit app
```

## Coding conventions
- All code fully typed — mypy strict
- Logging: always use `structlog.get_logger()`, never `print()`
- Config: all settings via `src/rag_agent/core/config.py` (pydantic-settings), never hardcoded
- Errors: raise custom exceptions from `core/exceptions.py`, caught by FastAPI handlers in `api/main.py`
- Async: use `async def` for all IO-bound functions
- Tests: unit tests mock external services; integration tests use real Docker services
- Commits: follow Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`)

## OpenRouter usage
Use the `openai` client pointing to OpenRouter base URL:
```python
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)
```
Preferred models: `google/gemini-flash-1.5` (fast/cheap), `anthropic/claude-3.5-sonnet` (quality), `mistralai/mistral-large` (balanced)

## Never do
- Commit secrets or `.env` files
- Use `print()` instead of structlog
- Skip type annotations
- Write sync DB calls inside async functions
- Hardcode model names or API URLs outside `config.py`
