"""Tests for ingest endpoints — file upload and text ingestion."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def _mock_task(task_id: str = "task-abc") -> MagicMock:
    task = MagicMock()
    task.id = task_id
    return task


def test_ingest_file_unsupported_type():
    resp = client.post(
        "/api/v1/ingest/file",
        files={"file": ("image.jpg", b"fake image data", "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 415
    assert "Unsupported" in resp.json()["detail"]


def test_ingest_file_success():
    mock_task = _mock_task("task-123")

    with patch("rag_agent.api.v1.ingest.ingest_document") as mock_ingest:
        mock_ingest.delay = MagicMock(return_value=mock_task)
        resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("doc.txt", b"Hello world document content", "text/plain")},
            headers=HEADERS,
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "task-123"
    assert data["status"] == "queued"
    assert data["filename"] == "doc.txt"


def test_ingest_file_pdf_success():
    mock_task = _mock_task("task-pdf-456")

    with patch("rag_agent.api.v1.ingest.ingest_document") as mock_ingest:
        mock_ingest.delay = MagicMock(return_value=mock_task)
        resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("report.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            headers=HEADERS,
        )

    assert resp.status_code == 202
    assert resp.json()["filename"] == "report.pdf"


def test_ingest_text_success():
    mock_task = _mock_task("task-text-789")

    with patch("rag_agent.api.v1.ingest.ingest_document") as mock_ingest:
        mock_ingest.delay = MagicMock(return_value=mock_task)
        resp = client.post(
            "/api/v1/ingest/text",
            json={"text": "This is a document to ingest.", "source": "manual-input"},
            headers=HEADERS,
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "task-text-789"
    assert data["filename"] == "manual-input"
    assert data["status"] == "queued"


def test_ingest_file_requires_auth():
    resp = client.post(
        "/api/v1/ingest/file",
        files={"file": ("doc.txt", b"content", "text/plain")},
    )
    assert resp.status_code == 401


def test_ingest_text_requires_auth():
    resp = client.post("/api/v1/ingest/text", json={"text": "hello", "source": "src"})
    assert resp.status_code == 401
