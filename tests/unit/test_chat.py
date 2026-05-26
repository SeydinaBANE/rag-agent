"""Tests for chat endpoints — sync and streaming."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def test_chat_success():
    fake_result = {
        "answer": "RAG is Retrieval-Augmented Generation.",
        "sources": [{"text": "context", "source": "doc.pdf", "score": 0.92}],
        "cached": False,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch("rag_agent.api.v1.chat.answer", new=AsyncMock(return_value=fake_result)):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "What is RAG?"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "RAG is Retrieval-Augmented Generation."
    assert len(data["sources"]) == 1
    assert data["cached"] is False


def test_chat_cached_response():
    fake_result = {
        "answer": "cached answer",
        "sources": [],
        "cached": True,
        "usage": {},
    }

    with patch("rag_agent.api.v1.chat.answer", new=AsyncMock(return_value=fake_result)):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "repeated question"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json()["cached"] is True


def test_chat_requires_auth():
    resp = client.post("/api/v1/chat", json={"query": "test"})
    assert resp.status_code == 401


def test_chat_stream():
    async def fake_stream(*args, **kwargs):
        for token in ["Hello", " world"]:
            yield token

    with patch("rag_agent.api.v1.chat.answer_stream", side_effect=fake_stream):
        resp = client.get(
            "/api/v1/chat/stream?query=what+is+rag",
            headers=HEADERS,
        )

    assert resp.status_code == 200
    content = resp.text
    assert "Hello" in content
    assert "world" in content
    assert '"done": true' in content or '"done":true' in content


def test_chat_stream_requires_auth():
    resp = client.get("/api/v1/chat/stream?query=test")
    assert resp.status_code == 401
