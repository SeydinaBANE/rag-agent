"""Tests for webhooks — MinIO event → Celery ingestion."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_agent.api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def _minio_payload(
    event: str = "s3:ObjectCreated:Put", key: str = "doc.pdf", size: int = 1024
) -> dict:
    return {
        "Records": [
            {
                "eventName": event,
                "s3": {
                    "bucket": {"name": "rag-documents"},
                    "object": {"key": key, "size": size},
                },
            }
        ]
    }


def test_webhook_invalid_json():
    resp = client.post(
        "/api/v1/webhooks/minio",
        content=b"not-json",
        headers={**HEADERS, "content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["detail"]


def test_webhook_ignores_non_create_event():
    payload = _minio_payload(event="s3:ObjectRemoved:Delete")
    resp = client.post("/api/v1/webhooks/minio", json=payload, headers=HEADERS)
    assert resp.status_code == 202
    assert resp.json()["queued"] == 0


def test_webhook_ignores_zero_size_object():
    payload = _minio_payload(size=0)
    resp = client.post("/api/v1/webhooks/minio", json=payload, headers=HEADERS)
    assert resp.status_code == 202
    assert resp.json()["queued"] == 0


def test_webhook_queues_ingestion_on_put_event():
    mock_task = MagicMock()
    mock_task.id = "task-123"

    with (
        patch("rag_agent.api.v1.webhooks._download_from_minio", return_value="base64content"),
        patch("rag_agent.api.v1.webhooks.ingest_document") as mock_ingest,
    ):
        mock_ingest.delay = MagicMock(return_value=mock_task)
        resp = client.post("/api/v1/webhooks/minio", json=_minio_payload(), headers=HEADERS)

    assert resp.status_code == 202
    data = resp.json()
    assert data["queued"] == 1
    assert "task-123" in data["task_ids"]


def test_webhook_continues_on_download_error():
    """If MinIO download fails, the record is skipped but the endpoint returns 202."""
    with patch(
        "rag_agent.api.v1.webhooks._download_from_minio",
        side_effect=RuntimeError("minio unreachable"),
    ):
        resp = client.post("/api/v1/webhooks/minio", json=_minio_payload(), headers=HEADERS)

    assert resp.status_code == 202
    assert resp.json()["queued"] == 0


def test_webhook_empty_records():
    resp = client.post("/api/v1/webhooks/minio", json={"Records": []}, headers=HEADERS)
    assert resp.status_code == 202
    assert resp.json()["queued"] == 0


def test_webhook_no_key_in_object():
    payload = {
        "Records": [
            {
                "eventName": "s3:ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "rag-documents"},
                    "object": {"key": "", "size": 100},
                },
            }
        ]
    }
    resp = client.post("/api/v1/webhooks/minio", json=payload, headers=HEADERS)
    assert resp.status_code == 202
    assert resp.json()["queued"] == 0
