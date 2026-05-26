"""Integration tests for the multi-step agent endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app

HEADERS = {"X-API-Key": "test-key"}

MOCK_RESULT = {
    "session_id": "test-session-123",
    "objective": "Research OpenAI",
    "answer": "OpenAI is an AI research company founded in 2015.",
    "total_steps": 3,
    "steps": [
        {
            "step": 1,
            "type": "thought",
            "content": "I'll search for OpenAI news.",
            "tool": None,
            "done": False,
        },
        {
            "step": 1,
            "type": "tool_call",
            "content": "OpenAI latest news",
            "tool": "web_search",
            "done": False,
        },
        {
            "step": 1,
            "type": "observation",
            "content": "OpenAI released GPT-5...",
            "tool": "web_search",
            "done": False,
        },
        {
            "step": 2,
            "type": "answer",
            "content": "OpenAI is an AI research company.",
            "tool": None,
            "done": True,
        },
    ],
}


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@patch(
    "rag_agent.api.v1.agent_run.run_multi_agent", new_callable=AsyncMock, return_value=MOCK_RESULT
)
async def test_run_agent_success(mock_run: AsyncMock, client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/run",
        json={"objective": "Research OpenAI"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "OpenAI is an AI research company founded in 2015."
    assert data["total_steps"] == 3
    assert "session_id" in data
    assert len(data["steps"]) == 4


@patch(
    "rag_agent.api.v1.agent_run.run_multi_agent", new_callable=AsyncMock, return_value=MOCK_RESULT
)
async def test_run_agent_with_session_id(mock_run: AsyncMock, client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/run",
        json={"objective": "Research OpenAI", "session_id": "my-session"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    mock_run.assert_called_once_with(objective="Research OpenAI", session_id="my-session")


async def test_run_agent_empty_objective(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/run",
        json={"objective": ""},
        headers=HEADERS,
    )
    assert response.status_code == 422


async def test_run_agent_missing_key(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/run",
        json={"objective": "Research something"},
    )
    assert response.status_code == 401


@patch(
    "rag_agent.api.v1.agent_run.load_messages",
    return_value=[
        {"role": "user", "content": "Research OpenAI", "ts": "1234567890"},
        {"role": "assistant", "content": "OpenAI is...", "ts": "1234567891"},
    ],
)
async def test_get_session_history(mock_load: object, client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/agent/run/sessions/test-session-123",
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message_count"] == 2
    assert data["session_id"] == "test-session-123"


@patch("rag_agent.api.v1.agent_run.delete_session")
async def test_clear_session(mock_delete: object, client: AsyncClient) -> None:
    response = await client.delete(
        "/api/v1/agent/run/sessions/test-session-123",
        headers=HEADERS,
    )
    assert response.status_code == 204
