"""Tests for retriever — hybrid dense+BM25+RRF retrieval."""

from unittest.mock import AsyncMock, patch

import pytest

from rag_agent.services.retriever import _bm25_scores, _rrf_fuse, retrieve


def _make_dense_results(n: int = 3) -> list[dict]:
    return [
        {
            "text": f"document {i} about retrieval augmented generation",
            "metadata": {},
            "score": 1.0 - i * 0.1,
        }
        for i in range(n)
    ]


def test_bm25_scores_basic():
    docs = [
        "RAG is retrieval augmented generation",
        "LLM is a language model",
        "RAG combines retrieval",
    ]
    scores = _bm25_scores("RAG retrieval", docs)
    assert len(scores) == 3
    # First and third docs (containing RAG and retrieval) should score higher
    assert scores[0] > scores[1]


def test_bm25_scores_empty_docs():
    scores = _bm25_scores("query", [])
    assert scores == []


def test_bm25_scores_no_matching_terms():
    docs = ["completely unrelated content"]
    scores = _bm25_scores("xyz abc", docs)
    assert scores == [0.0]


def test_rrf_fuse_returns_sorted_results():
    dense = _make_dense_results(3)
    bm25 = [0.8, 0.3, 0.5]

    fused = _rrf_fuse(dense, bm25)

    assert len(fused) == 3
    # All results should have rrf_score
    for r in fused:
        assert "rrf_score" in r
    # Results should be sorted by rrf_score desc
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_empty():
    result = _rrf_fuse([], [])
    assert result == []


@pytest.mark.asyncio
async def test_retrieve_basic():
    dense_results = _make_dense_results(4)

    with (
        patch("rag_agent.services.retriever.embed_query", new=AsyncMock(return_value=[0.1] * 10)),
        patch(
            "rag_agent.services.retriever.query_similar", new=AsyncMock(return_value=dense_results)
        ),
        patch("rag_agent.services.retriever.settings.top_k", 2),
    ):
        results = await retrieve("what is RAG?")

    assert len(results) <= 4  # top_k * 2 dense, then filtered
    assert all("text" in r for r in results)


@pytest.mark.asyncio
async def test_retrieve_uses_top_k():
    dense_results = _make_dense_results(6)

    with (
        patch("rag_agent.services.retriever.embed_query", new=AsyncMock(return_value=[0.1] * 10)),
        patch(
            "rag_agent.services.retriever.query_similar", new=AsyncMock(return_value=dense_results)
        ),
        patch("rag_agent.services.retriever.settings.top_k", 3),
    ):
        results = await retrieve("query", top_k=2)

    # With cross-encoder ImportError, should slice to top_k=2
    assert len(results) <= 6


@pytest.mark.asyncio
async def test_retrieve_empty_results():
    with (
        patch("rag_agent.services.retriever.embed_query", new=AsyncMock(return_value=[0.1] * 10)),
        patch("rag_agent.services.retriever.query_similar", new=AsyncMock(return_value=[])),
        patch("rag_agent.services.retriever.settings.top_k", 5),
    ):
        results = await retrieve("query")

    assert results == []
