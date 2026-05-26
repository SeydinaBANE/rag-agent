"""Job status tracking for async ingestion tasks."""

from celery.result import AsyncResult
from fastapi import APIRouter, Depends

from rag_agent.api.v1.deps import require_api_key
from rag_agent.core.celery_app import celery_app
from rag_agent.models.schemas import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatus)
async def get_job(
    job_id: str,
    _: str = Depends(require_api_key),
) -> JobStatus:
    result = AsyncResult(job_id, app=celery_app)

    if result.state == "PENDING":
        return JobStatus(job_id=job_id, status="PENDING")
    elif result.state == "STARTED":
        return JobStatus(job_id=job_id, status="STARTED")
    elif result.state == "SUCCESS":
        return JobStatus(job_id=job_id, status="SUCCESS", result=result.result)
    elif result.state == "FAILURE":
        return JobStatus(job_id=job_id, status="FAILURE", error=str(result.info))
    else:
        return JobStatus(job_id=job_id, status=result.state)
