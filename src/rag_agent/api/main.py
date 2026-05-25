import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from rag_agent.core.config import settings
from rag_agent.core.exceptions import GuardrailError, IngestError, LLMError, RagAgentError
from rag_agent.core.logging import setup_logging

setup_logging()
log = structlog.get_logger()

# ── Prometheus metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)

app = FastAPI(
    title="rag-agent",
    description="Production-ready RAG + AI Agent platform",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next: object) -> Response:
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)

    start = time.perf_counter()
    response: Response = await call_next(request)  # type: ignore[operator]
    duration = time.perf_counter() - start

    endpoint = request.url.path
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)

    response.headers["X-Request-ID"] = request_id
    log.info("request", method=request.method, status=response.status_code, duration_s=round(duration, 3))
    return response


# ── Exception handlers ───────────────────────────────────────────────────────
@app.exception_handler(RagAgentError)
async def domain_error_handler(request: Request, exc: RagAgentError) -> JSONResponse:
    log.warning("domain_error", code=exc.code, detail=exc.message)
    status_code = 400 if isinstance(exc, (IngestError, GuardrailError)) else 500
    if isinstance(exc, LLMError):
        status_code = 502
    return JSONResponse(status_code=status_code, content={"error": exc.code, "detail": exc.message})


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


from rag_agent.api.v1 import agent, chat, ingest, jobs, webhooks

app.include_router(chat.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(webhooks.router, prefix="/api/v1")
