"""Tests for vector_store — ChromaDB wrapper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import rag_agent.services.vector_store as vs_module
from rag_agent.services.vector_store import (
    delete_by_source,
    get_collection,
    query_similar,
    upsert_chunks,
)


@pytest.fixture(autouse=True)
def reset_chroma_client():
    original = vs_module._client
    yield
    vs_module._client = original


def _make_mock_collection() -> MagicMock:
    col = MagicMock()
    col.upsert = AsyncMock()
    col.query = AsyncMock()
    col.delete = AsyncMock()
    return col


def _make_mock_chroma(collection: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection = AsyncMock(return_value=collection)
    return client


@pytest.mark.asyncio
async def test_get_collection_creates_collection():
    col = _make_mock_collection()
    mock_chroma = _make_mock_chroma(col)
    vs_module._client = mock_chroma

    result = await get_collection("my_collection")

    mock_chroma.get_or_create_collection.assert_called_once_with(
        name="my_collection", metadata={"hnsw:space": "cosine"}
    )
    assert result is col


@pytest.mark.asyncio
async def test_upsert_chunks_delegates_to_collection():
    col = _make_mock_collection()
    vs_module._client = _make_mock_chroma(col)

    await upsert_chunks(
        chunks=["text1", "text2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        ids=["id1", "id2"],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
    )

    col.upsert.assert_called_once_with(
        ids=["id1", "id2"],
        documents=["text1", "text2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}],
    )


@pytest.mark.asyncio
async def test_query_similar_returns_ranked_results():
    col = _make_mock_collection()
    col.query = AsyncMock(
        return_value={
            "documents": [["chunk A", "chunk B"]],
            "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
            "distances": [[0.1, 0.3]],
        }
    )
    vs_module._client = _make_mock_chroma(col)

    results = await query_similar([0.1, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "chunk A"
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[1]["score"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_query_similar_empty_results():
    col = _make_mock_collection()
    col.query = AsyncMock(return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]})
    vs_module._client = _make_mock_chroma(col)

    results = await query_similar([0.1] * 10)
    assert results == []


@pytest.mark.asyncio
async def test_delete_by_source():
    col = _make_mock_collection()
    vs_module._client = _make_mock_chroma(col)

    await delete_by_source("report.pdf")

    col.delete.assert_called_once_with(where={"source": "report.pdf"})


@pytest.mark.asyncio
async def test_get_chroma_singleton():
    vs_module._client = None
    mock_client = MagicMock()
    mock_client.get_or_create_collection = AsyncMock(return_value=_make_mock_collection())

    with patch(
        "rag_agent.services.vector_store.chromadb.AsyncHttpClient", new_callable=AsyncMock
    ) as mock_cls:
        mock_cls.return_value = mock_client
        await get_collection("test")
        await get_collection("test")

    # Should only instantiate client once (singleton)
    mock_cls.assert_called_once()
