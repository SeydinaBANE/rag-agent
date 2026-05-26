"""Tests for jobs endpoint — Celery task status polling."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def _mock_result(state: str, result=None, info=None) -> MagicMock:
    r = MagicMock()
    r.state = state
    r.result = result
    r.info = info
    return r


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        ("PENDING", "PENDING"),
        ("STARTED", "STARTED"),
    ],
)
def test_job_pending_started(state: str, expected_status: str):
    with patch("rag_agent.api.v1.jobs.AsyncResult", return_value=_mock_result(state)):
        resp = client.get("/api/v1/jobs/abc-123", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == expected_status
    assert resp.json()["job_id"] == "abc-123"


def test_job_success():
    result_data = {"doc_id": "xyz", "chunks": 5, "status": "done"}
    with patch(
        "rag_agent.api.v1.jobs.AsyncResult",
        return_value=_mock_result("SUCCESS", result=result_data),
    ):
        resp = client.get("/api/v1/jobs/job-xyz", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["result"] == result_data


def test_job_failure():
    with patch(
        "rag_agent.api.v1.jobs.AsyncResult",
        return_value=_mock_result("FAILURE", info=Exception("boom")),
    ):
        resp = client.get("/api/v1/jobs/fail-id", headers=HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FAILURE"
    assert "boom" in data["error"]


def test_job_unknown_state():
    with patch("rag_agent.api.v1.jobs.AsyncResult", return_value=_mock_result("RETRY")):
        resp = client.get("/api/v1/jobs/retry-id", headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "RETRY"


def test_job_requires_auth():
    resp = client.get("/api/v1/jobs/abc-123")
    assert resp.status_code == 401
