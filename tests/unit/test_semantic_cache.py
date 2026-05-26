"""Tests for semantic_cache — embedding-based Redis/ChromaDB cache."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_agent.services.semantic_cache import get_cached, set_cached


def _mock_collection(distances: list[float], docs: list[str], metas: list[dict]) -> MagicMock:
    col = MagicMock()
    col.query = AsyncMock(
        return_value={
            "distances": [distances],
            "documents": [docs],
            "metadatas": [metas],
        }
    )
    col.upsert = AsyncMock()
    return col


@pytest.mark.asyncio
async def test_get_cached_disabled(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", False)
    result = await get_cached("any question")
    assert result is None


@pytest.mark.asyncio
async def test_get_cached_miss_low_similarity(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    monkeypatch.setattr(
        "rag_agent.services.semantic_cache.settings.semantic_cache_similarity_threshold", 0.92
    )
    col = _mock_collection(distances=[0.2], docs=["..."], metas=[{}])

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.1])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        result = await get_cached("what is X?")

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_hit(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    monkeypatch.setattr(
        "rag_agent.services.semantic_cache.settings.semantic_cache_similarity_threshold", 0.90
    )
    payload = json.dumps({"answer": "cached answer"})
    expires_at = str(time.time() + 3600)
    col = _mock_collection(
        distances=[0.05],
        docs=[payload],
        metas=[{"expires_at": expires_at}],
    )

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.1])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        result = await get_cached("what is RAG?")

    assert result == "cached answer"


@pytest.mark.asyncio
async def test_get_cached_expired(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    monkeypatch.setattr(
        "rag_agent.services.semantic_cache.settings.semantic_cache_similarity_threshold", 0.90
    )
    payload = json.dumps({"answer": "stale answer"})
    expires_at = str(time.time() - 10)  # expired 10s ago
    col = _mock_collection(
        distances=[0.05],
        docs=[payload],
        metas=[{"expires_at": expires_at}],
    )

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.1])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        result = await get_cached("what is RAG?")

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_returns_none_on_query_error(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    col = MagicMock()
    col.query = AsyncMock(side_effect=Exception("chroma error"))

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.1])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        result = await get_cached("question")

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_empty_distances(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    col = _mock_collection(distances=[], docs=[], metas=[])

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.1])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        result = await get_cached("question")

    assert result is None


@pytest.mark.asyncio
async def test_set_cached_disabled(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", False)
    with patch("rag_agent.services.semantic_cache.embed_query") as mock_embed:
        await set_cached("q", "a")
        mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_set_cached_upserts(monkeypatch):
    monkeypatch.setattr("rag_agent.services.semantic_cache.settings.semantic_cache_enabled", True)
    monkeypatch.setattr(
        "rag_agent.services.semantic_cache.settings.semantic_cache_ttl_seconds", 3600
    )
    col = MagicMock()
    col.upsert = AsyncMock()

    with (
        patch("rag_agent.services.semantic_cache.embed_query", new=AsyncMock(return_value=[0.5])),
        patch("rag_agent.services.semantic_cache.get_collection", new=AsyncMock(return_value=col)),
    ):
        await set_cached("my question", "my answer")

    col.upsert.assert_called_once()
    call_kwargs = col.upsert.call_args.kwargs
    payload = json.loads(call_kwargs["documents"][0])
    assert payload["query"] == "my question"
    assert payload["answer"] == "my answer"
