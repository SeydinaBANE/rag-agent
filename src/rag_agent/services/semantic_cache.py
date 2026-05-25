"""Semantic cache: similar questions return cached answers without hitting the LLM."""

from __future__ import annotations

import hashlib
import json
import time

import structlog
from prometheus_client import Counter

from rag_agent.core.config import settings
from rag_agent.services.embedder import embed_query
from rag_agent.services.vector_store import get_collection

log = structlog.get_logger()

CACHE_HIT = Counter("semantic_cache_hits_total", "Semantic cache hits")
CACHE_MISS = Counter("semantic_cache_misses_total", "Semantic cache misses")

COLLECTION_CACHE = "semantic_cache"


async def get_cached(query: str) -> str | None:
    if not settings.semantic_cache_enabled:
        return None

    query_emb = await embed_query(query)
    collection = await get_collection(COLLECTION_CACHE)

    try:
        results = await collection.query(
            query_embeddings=[query_emb],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return None

    distances = results.get("distances", [[]])[0]
    if not distances:
        CACHE_MISS.inc()
        return None

    similarity = 1.0 - float(distances[0])
    if similarity >= settings.semantic_cache_similarity_threshold:
        metas = results.get("metadatas", [[]])[0]
        meta = metas[0] if metas else {}
        expires_at = float(meta.get("expires_at", 0))
        if expires_at > time.time():
            docs = results.get("documents", [[]])[0]
            cached_data = docs[0] if docs else None
            if cached_data:
                data = json.loads(cached_data)
                log.info("cache_hit", similarity=round(similarity, 3))
                CACHE_HIT.inc()
                return str(data.get("answer", ""))

    CACHE_MISS.inc()
    return None


async def set_cached(query: str, answer: str) -> None:
    if not settings.semantic_cache_enabled:
        return

    query_emb = await embed_query(query)
    collection = await get_collection(COLLECTION_CACHE)

    cache_id = hashlib.sha256(query.encode()).hexdigest()
    expires_at = time.time() + settings.semantic_cache_ttl_seconds

    payload = json.dumps({"query": query, "answer": answer})
    await collection.upsert(
        ids=[cache_id],
        documents=[payload],
        embeddings=[query_emb],
        metadatas=[{"expires_at": str(expires_at)}],
    )
    log.debug("cache_set", query_preview=query[:60])
