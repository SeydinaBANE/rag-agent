"""Multi-step agent endpoints: run sync + stream SSE + session history."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_agent.api.v1.deps import require_api_key
from rag_agent.services.agent_memory import delete_session, load_messages
from rag_agent.services.multi_agent import run_multi_agent, stream_multi_agent

log = structlog.get_logger()
router = APIRouter(prefix="/agent/run", tags=["multi-agent"])


class AgentRunRequest(BaseModel):
    objective: str = Field(..., min_length=1, max_length=2000, description="Goal for the agent")
    session_id: str | None = Field(None, description="Optional session ID for memory continuity")


class AgentRunResponse(BaseModel):
    session_id: str
    objective: str
    answer: str
    total_steps: int
    steps: list[dict[str, object]]


# ── Sync run ──────────────────────────────────────────────────────────────────


@router.post("", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    _: str = Depends(require_api_key),
) -> AgentRunResponse:
    """
    Run the multi-step ReAct agent to completion.
    Returns all steps + final answer once done.
    """
    result = await run_multi_agent(
        objective=request.objective,
        session_id=request.session_id,
    )
    return AgentRunResponse(
        session_id=result["session_id"],
        objective=result["objective"],
        answer=result["answer"],
        total_steps=result["total_steps"],
        steps=result["steps"],
    )


# ── Streaming SSE ─────────────────────────────────────────────────────────────


@router.get("/stream")
async def stream_agent(
    objective: str = Query(..., min_length=1),
    session_id: str | None = Query(None),
    _: str = Depends(require_api_key),
) -> StreamingResponse:
    """
    Stream agent steps as Server-Sent Events.

    Event format:
        data: {"step": 1, "type": "thought", "content": "...", "tool": null, "done": false}
        data: {"step": 1, "type": "tool_call", "content": "Apple Inc news", "tool": "web_search", "done": false}
        data: {"step": 1, "type": "observation", "content": "...", "tool": "web_search", "done": false}
        data: {"step": 2, "type": "answer", "content": "...", "tool": null, "done": true}
    """

    async def _generate() -> AsyncGenerator[str, None]:
        async for step in stream_multi_agent(objective=objective, session_id=session_id):
            payload = json.dumps(dict(step))
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session management ────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    _: str = Depends(require_api_key),
) -> dict[str, object]:
    """Get the message history of a session."""
    messages = load_messages(session_id)
    return {"session_id": session_id, "message_count": len(messages), "messages": messages}


@router.delete("/sessions/{session_id}", status_code=204)
async def clear_session(
    session_id: str,
    _: str = Depends(require_api_key),
) -> None:
    """Delete a session's memory from Redis."""
    delete_session(session_id)
    log.info("session_cleared", session_id=session_id)
