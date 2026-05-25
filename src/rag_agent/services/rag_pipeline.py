"""Core RAG pipeline: retrieve → build prompt → generate answer."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog
from prometheus_client import Counter, Histogram

from rag_agent.core.config import settings
from rag_agent.core.exceptions import LLMError
from rag_agent.services import llm_client
from rag_agent.services.retriever import retrieve
from rag_agent.services.semantic_cache import get_cached, set_cached

log = structlog.get_logger()

QUERY_COUNT = Counter("rag_queries_total", "Total RAG queries", ["cached"])
RETRIEVAL_SCORE = Histogram("rag_retrieval_score", "Top chunk retrieval score")
LLM_TOKENS = Counter("llm_tokens_total", "LLM token usage", ["type", "model"])

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information, say so clearly. Do not make up information.
Be concise and precise."""


async def answer(
    query: str,
    model: str | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    """Full RAG pipeline. Returns answer + sources + usage."""

    # 1. Semantic cache check
    cached = await get_cached(query)
    if cached:
        QUERY_COUNT.labels(cached="true").inc()
        return {"answer": cached, "sources": [], "cached": True, "usage": {}}

    QUERY_COUNT.labels(cached="false").inc()

    # 2. Retrieve relevant chunks
    chunks = await retrieve(query)
    if chunks:
        RETRIEVAL_SCORE.observe(float(chunks[0].get("score", 0)))

    # 3. Build context
    context_parts = [f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)]
    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # 4. Build messages
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    # 5. Generate
    answer_text, usage = await llm_client.complete(messages, model=model)

    # Track token usage
    mdl = model or settings.default_model
    LLM_TOKENS.labels(type="prompt", model=mdl).inc(usage.get("prompt_tokens", 0))
    LLM_TOKENS.labels(type="completion", model=mdl).inc(usage.get("completion_tokens", 0))

    # 6. Cache result
    await set_cached(query, answer_text)

    sources = [
        {
            "text": str(c.get("text", ""))[:200],
            "source": str(c.get("metadata", {}).get("source", "")),  # type: ignore[union-attr]
            "score": round(float(c.get("score", 0)), 3),
        }
        for c in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "cached": False,
        "usage": usage,
    }


async def answer_stream(
    query: str,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream tokens. Sources are sent as final SSE event."""
    chunks = await retrieve(query)
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)) or "No relevant context."

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    full_answer = ""
    async for token in llm_client.stream(messages, model=model):
        full_answer += token
        yield token

    # Cache after full response received
    await set_cached(query, full_answer)
