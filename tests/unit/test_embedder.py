"""Tests for embedder — batch embedding via OpenRouter-compatible endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import rag_agent.services.llm_client as llm_module
from rag_agent.services.embedder import EMBED_DIM, embed_query, embed_texts


def _make_embed_response(n: int) -> MagicMock:
    items = []
    for i in range(n):
        item = MagicMock()
        item.embedding = [float(i)] * EMBED_DIM
    items = [MagicMock(embedding=[float(j)] * EMBED_DIM) for j in range(n)]
    resp = MagicMock()
    resp.data = items
    return resp


@pytest.fixture(autouse=True)
def reset_client():
    original = llm_module._client
    yield
    llm_module._client = original


@pytest.mark.asyncio
async def test_embed_texts_empty():
    result = await embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_texts_single_batch():
    texts = ["hello", "world"]
    resp = _make_embed_response(2)

    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(return_value=resp)
    llm_module._client = mock_openai

    result = await embed_texts(texts)

    assert len(result) == 2
    assert len(result[0]) == EMBED_DIM
    mock_openai.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_texts_batches_over_100():
    """Should make 2 API calls for 150 texts."""
    texts = [f"text_{i}" for i in range(150)]

    call_count = 0

    async def fake_create(model: str, input: list[str]) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return _make_embed_response(len(input))

    mock_openai = MagicMock()
    mock_openai.embeddings.create = fake_create
    llm_module._client = mock_openai

    result = await embed_texts(texts)

    assert len(result) == 150
    assert call_count == 2


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector():
    resp = _make_embed_response(1)
    mock_openai = MagicMock()
    mock_openai.embeddings.create = AsyncMock(return_value=resp)
    llm_module._client = mock_openai

    result = await embed_query("what is RAG?")

    assert isinstance(result, list)
    assert len(result) == EMBED_DIM
