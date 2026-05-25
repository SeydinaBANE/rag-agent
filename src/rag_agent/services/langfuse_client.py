"""Langfuse LLM tracing — no-op if not configured or package absent."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from typing import Any

import structlog

from rag_agent.core.config import settings

log = structlog.get_logger()

_client: Any = None
_initialized = False


def _get_client() -> Any:
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import]
        _client = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
        )
        log.info("langfuse_initialized", host=settings.langfuse_host)
    except ImportError:
        log.warning("langfuse_unavailable", reason="langfuse package not installed")
    return _client


@contextlib.contextmanager
def trace_generation(
    name: str,
    model: str,
    messages: list[dict[str, str]],
    trace_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that records one LLM call to Langfuse.

        with trace_generation("rag-answer", model, messages) as meta:
            text, usage = await llm_client.complete(messages, model=model)
            meta["output"] = text
            meta["usage"] = usage
    """
    meta: dict[str, Any] = {}
    start = time.perf_counter()
    yield meta

    client = _get_client()
    if client is None:
        return

    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        trace = client.trace(id=trace_id, name="rag-agent")
        trace.generation(
            name=name,
            model=model,
            input=messages,
            output=meta.get("output", ""),
            usage={
                "input": meta.get("usage", {}).get("prompt_tokens", 0),
                "output": meta.get("usage", {}).get("completion_tokens", 0),
            },
            metadata={"latency_ms": latency_ms},
        )
        client.flush()
    except Exception as exc:
        log.warning("langfuse_trace_failed", error=str(exc))
