"""Integration tests for chat endpoints. Mocks LLM + vector store calls."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app

HEADERS = {"X-API-Key": "test-key-for-integration"}


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@patch("rag_agent.services.rag_pipeline.get_cached", new_callable=AsyncMock, return_value=None)
@patch("rag_agent.services.rag_pipeline.set_cached", new_callable=AsyncMock)
@patch("rag_agent.services.rag_pipeline.retrieve", new_callable=AsyncMock)
@patch("rag_agent.services.rag_pipeline.llm_client.complete", new_callable=AsyncMock)
async def test_chat_returns_answer(
    mock_complete: AsyncMock,
    mock_retrieve: AsyncMock,
    mock_set_cache: AsyncMock,
    mock_get_cache: AsyncMock,
    client: AsyncClient,
) -> None:
    mock_retrieve.return_value = [
        {
            "text": "RAG combines retrieval and generation.",
            "metadata": {"source": "test.pdf"},
            "score": 0.92,
        }
    ]
    mock_complete.return_value = (
        "RAG stands for Retrieval-Augmented Generation.",
        {"prompt_tokens": 100, "completion_tokens": 20},
    )

    response = await client.post(
        "/api/v1/chat",
        json={"query": "What is RAG?"},
        headers=HEADERS,
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["cached"] is False
    assert len(data["sources"]) == 1


@patch(
    "rag_agent.services.rag_pipeline.get_cached",
    new_callable=AsyncMock,
    return_value="Cached answer.",
)
async def test_chat_returns_cached(mock_get_cache: AsyncMock, client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"query": "What is RAG?"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["answer"] == "Cached answer."


async def test_chat_missing_api_key(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={"query": "test"})
    assert response.status_code == 401


async def test_chat_empty_query(client: AsyncClient) -> None:
    response = await client.post("/api/v1/chat", json={"query": ""}, headers=HEADERS)
    assert response.status_code == 422
