"""LangGraph agent: retrieve → grade → (web_search) → generate → check_hallucination → retry."""

from __future__ import annotations

from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from rag_agent.core.config import settings
from rag_agent.services import llm_client
from rag_agent.services.retriever import retrieve

log = structlog.get_logger()


class AgentState(TypedDict):
    query: str
    context_chunks: list[dict[str, Any]]
    answer: str
    hallucination_score: float
    iteration: int
    sources: list[dict[str, Any]]
    web_searched: bool


# ── Nodes ───────────────────────────────────────────────────────────────────


async def node_retrieve(state: AgentState) -> AgentState:
    chunks = await retrieve(state["query"])
    log.debug("graph_retrieve", n=len(chunks))
    return {**state, "context_chunks": chunks}


async def node_grade_relevance(state: AgentState) -> AgentState:
    """Grade whether retrieved chunks are relevant enough. Low score → web search."""
    chunks = state["context_chunks"]
    if not chunks:
        return {**state, "context_chunks": []}

    avg_score = sum(float(c.get("score", 0)) for c in chunks) / len(chunks)
    log.debug("graph_grade", avg_score=round(avg_score, 3))
    # Mark low-quality context so the router triggers web search
    return {**state, "hallucination_score": avg_score}


async def node_web_search(state: AgentState) -> AgentState:
    """Fallback: enrich context with a web search result via LLM knowledge."""
    log.info("graph_web_search", query=state["query"])
    # Real impl: call a search tool (Tavily, SerpAPI, etc.)
    # Here we ask the LLM itself to provide general knowledge as a fallback chunk
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                f"Provide a brief factual summary about: {state['query']}. "
                "Be concise and accurate. Max 200 words."
            ),
        }
    ]
    web_text, _ = await llm_client.complete(messages, model=settings.default_model, max_tokens=256)
    web_chunk = {"text": web_text, "metadata": {"source": "web_search"}, "score": 0.6}
    return {**state, "context_chunks": state["context_chunks"] + [web_chunk], "web_searched": True}


async def node_generate(state: AgentState) -> AgentState:
    chunks = state["context_chunks"]
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using ONLY the provided context. "
                "If insufficient, say so. Be concise."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"},
    ]
    answer, _ = await llm_client.complete(messages, model=settings.quality_model)
    log.debug("graph_generate", answer_len=len(answer))
    return {**state, "answer": answer, "sources": chunks}


async def node_check_hallucination(state: AgentState) -> AgentState:
    """Lightweight hallucination check: ask LLM to score its own answer."""
    if not state.get("answer") or not state["context_chunks"]:
        return {**state, "hallucination_score": 1.0}

    context = "\n".join(
        c["text"] for c in state["context_chunks"][: settings.hallucination_check_chunks]
    )
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                f"Context: {context}\n\nAnswer: {state['answer']}\n\n"
                "Is the answer fully supported by the context? "
                "Reply with only a number between 0.0 (not supported) and 1.0 (fully supported)."
            ),
        }
    ]
    score_text, _ = await llm_client.complete(
        messages, model=settings.default_model, max_tokens=10, temperature=0.0
    )
    try:
        score = float(score_text.strip())
    except ValueError:
        score = 0.5
    score = max(0.0, min(1.0, score))
    log.info("graph_hallucination_check", score=score, iteration=state["iteration"])
    return {**state, "hallucination_score": score}


# ── Routing conditions ───────────────────────────────────────────────────────


def route_after_grade(state: AgentState) -> str:
    """If avg retrieval score below web_search_fallback_threshold and no web search yet → web_search."""
    if state.get(
        "hallucination_score", 1.0
    ) < settings.web_search_fallback_threshold and not state.get("web_searched"):
        return "web_search"
    return "generate"


def route_after_hallucination(state: AgentState) -> str:
    threshold = settings.guardrails_hallucination_threshold
    if (
        state["hallucination_score"] < threshold
        and state["iteration"] < settings.max_retrieval_retries
    ):
        log.warning("graph_retry", score=state["hallucination_score"], iteration=state["iteration"])
        return "retrieve"  # retry the full loop
    return "end"


# ── Build graph ──────────────────────────────────────────────────────────────


def build_rag_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_relevance", node_grade_relevance)
    graph.add_node("web_search", node_web_search)
    graph.add_node("generate", node_generate)
    graph.add_node("check_hallucination", node_check_hallucination)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "check_hallucination")
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "retrieve": "retrieve",
            "end": END,
        },
    )

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


async def run_agent(query: str) -> dict[str, Any]:
    """Run the full LangGraph RAG agent and return result."""
    graph = get_graph()
    initial_state: AgentState = {
        "query": query,
        "context_chunks": [],
        "answer": "",
        "hallucination_score": 1.0,
        "iteration": 0,
        "sources": [],
        "web_searched": False,
    }
    result = await graph.ainvoke(initial_state)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "hallucination_score": result["hallucination_score"],
        "iterations": result["iteration"],
        "web_searched": result["web_searched"],
    }
