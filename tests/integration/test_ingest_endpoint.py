"""Integration tests for ingestion endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from rag_agent.api.main import app

HEADERS = {"X-API-Key": "test-key-for-integration"}


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@patch("rag_agent.api.v1.ingest.ingest_document")
async def test_ingest_text(mock_task: MagicMock, client: AsyncClient) -> None:
    mock_task.delay.return_value = MagicMock(id="task-123")

    response = await client.post(
        "/api/v1/ingest/text",
        json={"text": "This is a test document about RAG.", "source": "test-doc"},
        headers=HEADERS,
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["job_id"] == "task-123"


async def test_ingest_unsupported_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.exe", b"binary data", "application/octet-stream")},
        headers=HEADERS,
    )
    assert response.status_code == 415


async def test_ingest_missing_key(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/ingest/text",
        json={"text": "test", "source": "src"},
    )
    assert response.status_code == 401
