"""Tests for evaluate endpoint — RAG quality evaluation."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def _write_dataset(samples: list[dict], path: Path) -> None:
    path.write_text(json.dumps(samples))


def test_evaluate_dataset_not_found():
    resp = client.post(
        "/api/v1/evaluate?dataset=nonexistent/file.json",
        headers=HEADERS,
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_evaluate_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all")
    resp = client.post(f"/api/v1/evaluate?dataset={bad_file}", headers=HEADERS)
    assert resp.status_code == 400
    assert "Invalid dataset" in resp.json()["detail"]


def test_evaluate_empty_dataset(tmp_path):
    empty_file = tmp_path / "empty.json"
    _write_dataset([], empty_file)
    resp = client.post(f"/api/v1/evaluate?dataset={empty_file}", headers=HEADERS)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_evaluate_all_pipeline_calls_fail(tmp_path):
    dataset = tmp_path / "qa.json"
    _write_dataset([{"question": "q1", "ground_truth": "a1"}], dataset)

    with patch(
        "rag_agent.api.v1.evaluate.rag_pipeline.answer",
        new=AsyncMock(side_effect=RuntimeError("pipeline error")),
    ):
        resp = client.post(f"/api/v1/evaluate?dataset={dataset}", headers=HEADERS)

    assert resp.status_code == 502


def test_evaluate_success(tmp_path):
    dataset = tmp_path / "qa.json"
    samples = [
        {
            "question": "What is RAG?",
            "ground_truth": "RAG is retrieval-augmented generation.",
            "context": "...",
        },
        {"question": "What is LLM?", "ground_truth": "A large language model.", "context": "..."},
    ]
    _write_dataset(samples, dataset)

    fake_answer = {
        "answer": "RAG combines retrieval and generation.",
        "sources": [{"text": "context chunk"}],
    }
    fake_scores = {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_recall": 0.80}

    with (
        patch(
            "rag_agent.api.v1.evaluate.rag_pipeline.answer", new=AsyncMock(return_value=fake_answer)
        ),
        patch("rag_agent.api.v1.evaluate._run_ragas", return_value=fake_scores),
    ):
        resp = client.post(f"/api/v1/evaluate?dataset={dataset}&max_samples=2", headers=HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["faithfulness"] == pytest.approx(0.9)
    assert data["passed"] is True
    assert data["n_samples"] == 2


def test_evaluate_partial_failures(tmp_path):
    """If some samples fail, succeeded ones are still evaluated."""
    dataset = tmp_path / "qa.json"
    samples = [
        {"question": "q1", "ground_truth": "a1"},
        {"question": "q2", "ground_truth": "a2"},
    ]
    _write_dataset(samples, dataset)

    fake_answer = {"answer": "ans", "sources": []}
    call_count = 0

    async def sometimes_fail(question):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first one fails")
        return fake_answer

    fake_scores = {"faithfulness": 0.75, "answer_relevancy": 0.70, "context_recall": 0.65}

    with (
        patch("rag_agent.api.v1.evaluate.rag_pipeline.answer", side_effect=sometimes_fail),
        patch("rag_agent.api.v1.evaluate._run_ragas", return_value=fake_scores),
    ):
        resp = client.post(f"/api/v1/evaluate?dataset={dataset}&max_samples=2", headers=HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["n_samples"] == 1  # only q2 succeeded


def test_evaluate_requires_auth():
    resp = client.post("/api/v1/evaluate")
    assert resp.status_code == 401
