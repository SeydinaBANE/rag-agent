"""Tests for agent endpoint — LangGraph agent with guardrails."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def test_agent_chat_success():
    fake_result = {
        "answer": "RAG answer via graph.",
        "sources": [{"text": "relevant chunk", "source": "doc.pdf", "score": 0.88}],
        "iterations": 2,
        "web_searched": False,
        "hallucination_score": 0.85,
    }

    with (
        patch("rag_agent.api.v1.agent.guard_input", return_value="what is RAG?"),
        patch("rag_agent.api.v1.agent.run_agent", new=AsyncMock(return_value=fake_result)),
        patch(
            "rag_agent.api.v1.agent.guard_output",
            return_value={"answer": "RAG answer via graph.", "confidence": 0.85},
        ),
    ):
        resp = client.post(
            "/api/v1/agent",
            json={"query": "what is RAG?"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "RAG answer via graph."
    assert data["confidence"] == pytest.approx(0.85)
    assert data["web_searched"] is False
    assert data["iterations"] == 2


def test_agent_chat_with_web_search():
    fake_result = {
        "answer": "answer from web search",
        "sources": [],
        "iterations": 3,
        "web_searched": True,
        "hallucination_score": 0.7,
    }

    with (
        patch("rag_agent.api.v1.agent.guard_input", return_value="latest news?"),
        patch("rag_agent.api.v1.agent.run_agent", new=AsyncMock(return_value=fake_result)),
        patch(
            "rag_agent.api.v1.agent.guard_output",
            return_value={"answer": "answer from web search", "confidence": 0.7},
        ),
    ):
        resp = client.post(
            "/api/v1/agent",
            json={"query": "latest news?"},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert resp.json()["web_searched"] is True


def test_agent_requires_auth():
    resp = client.post("/api/v1/agent", json={"query": "test"})
    assert resp.status_code == 401
