"""Integration tests for the LangGraph agent endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@patch("rag_agent.api.v1.agent.run_agent", new_callable=AsyncMock)
@patch("rag_agent.api.v1.agent.guard_input", return_value="what is RAG?")
@patch("rag_agent.api.v1.agent.guard_output", return_value={"answer": "RAG is great.", "confidence": 0.95})
async def test_agent_returns_answer(
    mock_guard_out: object,
    mock_guard_in: object,
    mock_run: AsyncMock,
    client: AsyncClient,
) -> None:
    mock_run.return_value = {
        "answer": "RAG is great.",
        "sources": [],
        "iterations": 1,
        "web_searched": False,
        "hallucination_score": 0.95,
    }
    response = await client.post(
        "/api/v1/agent",
        json={"query": "what is RAG?"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert data["iterations"] == 1


@patch("rag_agent.api.v1.agent.guard_input", side_effect=__import__("rag_agent.core.exceptions", fromlist=["GuardrailError"]).GuardrailError("toxic content"))
async def test_agent_guardrail_blocks(mock_guard: object, client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent",
        json={"query": "how to hack"},
        headers=HEADERS,
    )
    assert response.status_code == 400
    assert response.json()["error"] == "GUARDRAIL_BLOCKED"


async def test_agent_missing_key(client: AsyncClient) -> None:
    response = await client.post("/api/v1/agent", json={"query": "test"})
    assert response.status_code == 401
