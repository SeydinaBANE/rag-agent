"""LangGraph agent endpoint with guardrails."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from rag_agent.api.v1.deps import require_api_key
from rag_agent.models.schemas import ChatRequest
from rag_agent.services.graph import run_agent
from rag_agent.services.guardrails import guard_input, guard_output

log = structlog.get_logger()
router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("")
async def agent_chat(
    request: ChatRequest,
    _: str = Depends(require_api_key),
) -> dict[str, object]:
    """
    Full LangGraph RAG agent:
    retrieve → grade → (web_search) → generate → check_hallucination → (retry)
    """
    safe_query = guard_input(request.query)
    result = await run_agent(safe_query)

    context_chunks = [str(s.get("text", "")) for s in result.get("sources", [])]
    guarded = guard_output(str(result["answer"]), context_chunks)

    return {
        "answer": guarded["answer"],
        "confidence": guarded["confidence"],
        "sources": result.get("sources", []),
        "iterations": result.get("iterations", 1),
        "web_searched": result.get("web_searched", False),
        "hallucination_score": result.get("hallucination_score"),
        "cached": False,
        "usage": {},
    }
