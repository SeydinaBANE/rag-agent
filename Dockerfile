# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir "uv>=0.5"

COPY pyproject.toml README.md ./
COPY src/ src/

# Install package + runtime deps into /app/.venv
RUN uv venv && \
    uv pip install --no-cache ".[guardrails]"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# System deps for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtualenv from builder (same path so shebangs resolve correctly)
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY src/ src/
COPY alembic.ini ./
COPY alembic/ alembic/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "rag_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
