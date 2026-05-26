import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="rag-agent",
    description="Production-ready RAG + AI Agent platform",
    version="0.1.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)

# ── Rate limiting ────────────────────────────────────────────────────────────
_limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"],
)
app.state.limiter = _limiter
app.add_middleware(SlowAPIMiddleware)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    log.warning("rate_limit_exceeded", client=request.client.host if request.client else "unknown")
    return JSONResponse(
        status_code=429,
        content={"error": "RATE_LIMIT_EXCEEDED", "detail": str(exc.detail)},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]


# ── OpenTelemetry ────────────────────────────────────────────────────────────
def _setup_otel() -> None:
    try:
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor,  # type: ignore[import]
        )

        if settings.otel_exporter_endpoint:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,  # type: ignore[import]
                )

                provider = TracerProvider()
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint))
                )
                trace.set_tracer_provider(provider)
                log.info("otel_otlp_configured", endpoint=settings.otel_exporter_endpoint)
            except ImportError:
                log.warning(
                    "otel_otlp_unavailable",
                    hint="pip install opentelemetry-exporter-otlp-proto-http",
                )

        FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/metrics")
        log.info("otel_instrumented")
    except Exception as exc:
        log.warning("otel_setup_failed", error=str(exc))


_setup_otel()

# ── Middleware ───────────────────────────────────────────────────────────────
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

    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)

    response.headers["X-Request-ID"] = request_id
    log.info(
        "request", method=request.method, status=response.status_code, duration_s=round(duration, 3)
    )
    return response


# ── Exception handlers ───────────────────────────────────────────────────────
@app.exception_handler(RagAgentError)
async def domain_error_handler(request: Request, exc: RagAgentError) -> JSONResponse:
    log.warning("domain_error", code=exc.code, detail=exc.message)
    status_code = 400 if isinstance(exc, IngestError | GuardrailError) else 500
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


from rag_agent.api.v1 import (  # noqa: E402
    agent,
    agent_run,
    chat,
    evaluate,
    ingest,
    jobs,
    keys,
    ocr,
    webhooks,
)

app.include_router(chat.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(agent_run.router, prefix="/api/v1")
app.include_router(evaluate.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(keys.router, prefix="/api/v1")
app.include_router(ocr.router, prefix="/api/v1")
app.include_router(webhooks.router, prefix="/api/v1")
