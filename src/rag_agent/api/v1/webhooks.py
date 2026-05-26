"""MinIO event webhook → triggers async Celery ingestion."""

from __future__ import annotations

import base64

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from rag_agent.core.config import settings
from rag_agent.services.ingestion_tasks import ingest_document

log = structlog.get_logger()
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class MinioRecord(BaseModel):
    eventName: str
    s3: dict[str, object]


class MinioEvent(BaseModel):
    EventName: str | None = None
    Records: list[MinioRecord] | None = None


@router.post("/minio", status_code=status.HTTP_202_ACCEPTED)
async def minio_webhook(request: Request) -> dict[str, object]:
    """
    Receive MinIO bucket notification events.
    Triggers ingestion for s3:ObjectCreated:* events.
    Configure in MinIO: mc event add myminio/rag-documents arn:... --event put
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from None

    records = body.get("Records", [])
    queued: list[str] = []

    for record in records:
        event_name = record.get("eventName", "")
        if not event_name.startswith("s3:ObjectCreated"):
            continue

        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name", "")
        obj = s3_info.get("object", {})
        key = obj.get("key", "")
        size = obj.get("size", 0)

        if not key or size == 0:
            continue

        log.info("minio_webhook_received", bucket=bucket, key=key, size=size)

        # Download from MinIO and queue ingestion
        try:
            content_b64 = _download_from_minio(bucket, key)
        except Exception as exc:
            log.error("minio_download_failed", key=key, error=str(exc))
            continue

        task = ingest_document.delay(content_b64, key)
        queued.append(task.id)
        log.info("minio_ingestion_queued", key=key, task_id=task.id)

    return {"queued": len(queued), "task_ids": queued}


def _download_from_minio(bucket: str, key: str) -> str:
    """Download object from MinIO and return as base64 string."""
    from minio import Minio  # type: ignore[import]

    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    response = client.get_object(bucket, key)
    try:
        content = response.read()
    finally:
        response.close()
        response.release_conn()

    return base64.b64encode(content).decode()
