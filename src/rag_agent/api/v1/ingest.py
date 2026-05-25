"""Document ingestion endpoints: file upload and raw text."""

import base64
import uuid

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from rag_agent.api.v1.deps import require_api_key
from rag_agent.core.exceptions import IngestError
from rag_agent.models.schemas import IngestResponse, IngestTextRequest
from rag_agent.services.ingestion_tasks import ingest_document

log = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/html",
}
MAX_UPLOAD_MB = 50


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    file: UploadFile = File(...),
    _: str = Depends(require_api_key),
) -> IngestResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_MB}MB limit",
        )

    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    content_b64 = base64.b64encode(content).decode()

    task = ingest_document.delay(content_b64, filename, doc_id)
    log.info("ingest_queued", doc_id=doc_id, filename=filename, task_id=task.id)

    return IngestResponse(job_id=task.id, filename=filename, status="queued")


@router.post("/text", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_text(
    request: IngestTextRequest,
    _: str = Depends(require_api_key),
) -> IngestResponse:
    doc_id = str(uuid.uuid4())
    content_b64 = base64.b64encode(request.text.encode()).decode()

    task = ingest_document.delay(content_b64, f"{request.source}.txt", doc_id)
    log.info("ingest_text_queued", doc_id=doc_id, source=request.source)

    return IngestResponse(job_id=task.id, filename=request.source, status="queued")
