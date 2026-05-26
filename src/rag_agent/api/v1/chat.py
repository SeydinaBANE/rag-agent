"""Chat endpoints: standard Q&A and SSE streaming."""

import json
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag_agent.api.v1.deps import require_api_key
from rag_agent.models.schemas import ChatRequest, ChatResponse, SourceChunk
from rag_agent.services.rag_pipeline import answer, answer_stream

log = structlog.get_logger()
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: str = Depends(require_api_key),
) -> ChatResponse:
    result = await answer(
        query=request.query,
        model=request.model,
        session_id=request.session_id,
    )
    return ChatResponse(
        answer=str(result["answer"]),
        sources=[SourceChunk(**s) for s in result["sources"]],  # type: ignore[union-attr,arg-type]
        cached=bool(result["cached"]),
        usage=result["usage"],  # type: ignore[arg-type]
    )


@router.get("/stream")
async def chat_stream(
    query: str,
    model: str | None = None,
    _: str = Depends(require_api_key),
) -> StreamingResponse:
    async def _event_generator() -> AsyncGenerator[str, None]:
        async for token in answer_stream(query=query, model=model):
            payload = json.dumps({"token": token, "done": False})
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
