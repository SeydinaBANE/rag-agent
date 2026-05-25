"""Hybrid retrieval: dense (ChromaDB) + BM25 sparse, with cross-encoder reranker."""

from __future__ import annotations

import math
from collections import Counter

import structlog

from rag_agent.core.config import settings
from rag_agent.services.embedder import embed_query
from rag_agent.services.vector_store import query_similar

log = structlog.get_logger()


async def retrieve(query: str, top_k: int | None = None) -> list[dict[str, object]]:
    """Hybrid retrieval: dense + BM25 fusion, then optional cross-encoder rerank."""
    k = top_k or settings.top_k
    query_emb = await embed_query(query)

    # Dense retrieval
    dense_results = await query_similar(query_emb, top_k=k * 2)

    # BM25 rerank on top of dense results (no extra call needed)
    texts = [str(r["text"]) for r in dense_results]
    bm25_scores = _bm25_scores(query, texts)

    # Reciprocal Rank Fusion
    fused = _rrf_fuse(dense_results, bm25_scores)

    # Cross-encoder reranker (optional — requires sentence-transformers)
    try:
        fused = await _cross_encoder_rerank(query, fused, top_k=k)
    except ImportError:
        fused = fused[:k]

    log.debug("retrieved", n=len(fused), query_preview=query[:60])
    return fused


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Compute BM25 relevance scores."""
    query_terms = query.lower().split()
    avg_len = sum(len(d.split()) for d in documents) / max(len(documents), 1)
    scores: list[float] = []

    for doc in documents:
        doc_terms = Counter(doc.lower().split())
        doc_len = sum(doc_terms.values())
        score = 0.0
        for term in query_terms:
            tf = doc_terms.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((len(documents) + 1) / (1 + sum(1 for d in documents if term in d.lower())))
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += idf * tf_norm
        scores.append(score)

    return scores


def _rrf_fuse(
    dense_results: list[dict[str, object]],
    bm25_scores: list[float],
    rrf_k: int = 60,
) -> list[dict[str, object]]:
    """Reciprocal Rank Fusion of dense and BM25 rankings."""
    n = len(dense_results)
    # Dense ranking (already sorted by score desc)
    dense_rank = {i: 1 / (rrf_k + i + 1) for i in range(n)}

    # BM25 ranking
    bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {orig: 1 / (rrf_k + rank + 1) for rank, orig in enumerate(bm25_order)}

    fused_scores = {i: dense_rank[i] + bm25_rank[i] for i in range(n)}

    sorted_indices = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)
    return [{**dense_results[i], "rrf_score": fused_scores[i]} for i in sorted_indices]


async def _cross_encoder_rerank(
    query: str,
    results: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    """Rerank with a local cross-encoder model. Raises ImportError if not installed."""
    from sentence_transformers import CrossEncoder  # type: ignore[import]

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, str(r["text"])) for r in results]
    ce_scores = model.predict(pairs).tolist()

    reranked = sorted(
        zip(results, ce_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{**r, "ce_score": float(s)} for r, s in reranked[:top_k]]
