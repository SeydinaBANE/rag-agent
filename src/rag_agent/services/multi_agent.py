"""
Multi-step ReAct agent using LangGraph.

Graph: plan → execute_loop (tool calls) → synthesize → end
Streams each step as AgentStep events for SSE.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from rag_agent.services import llm_client
from rag_agent.services.agent_memory import append_message, compress_if_needed
from rag_agent.services.agent_tools import TOOL_MAP, get_tools_description

log = structlog.get_logger()

MAX_STEPS = 8


class AgentStep(TypedDict):
    step: int
    type: str          # "thought" | "tool_call" | "observation" | "answer"
    content: str
    tool: str | None
    done: bool


class MultiAgentState(TypedDict):
    objective: str
    session_id: str
    messages: list[dict[str, str]]
    steps: list[AgentStep]
    step_count: int
    final_answer: str
    done: bool


# ── Prompts ───────────────────────────────────────────────────────────────────

REACT_SYSTEM = """You are a methodical research agent. To answer the user's objective, you reason step-by-step using the ReAct pattern.

Available tools:
{tools}

Format EACH response as valid JSON:
{{
  "thought": "your reasoning about what to do next",
  "action": "tool_name OR 'answer'",
  "action_input": "the input for the tool, OR the final answer if action is 'answer'"
}}

Rules:
- Use tools to gather information before answering.
- When you have enough information, set action to "answer".
- Be concise in thoughts. Be thorough in the final answer.
- If a tool returns an error, try a different approach.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def node_react_step(state: MultiAgentState) -> MultiAgentState:
    """Single ReAct iteration: think → act → observe."""
    steps = list(state["steps"])
    step_n = state["step_count"] + 1

    # Build conversation context from previous steps
    history = []
    for s in steps[-6:]:  # keep last 6 steps to avoid context overflow
        if s["type"] == "thought":
            history.append({"role": "assistant", "content": f"Thought: {s['content']}"})
        elif s["type"] == "observation":
            history.append({"role": "user", "content": f"Observation: {s['content']}"})

    messages: list[dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM.format(tools=get_tools_description())},
        {"role": "user", "content": f"Objective: {state['objective']}"},
        *history,
    ]

    raw, _ = await llm_client.complete(messages, model=None, temperature=0.1, max_tokens=600)

    # Parse JSON response
    try:
        # Extract JSON block even if LLM adds markdown
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group() if match else raw)
    except Exception:
        parsed = {"thought": raw, "action": "answer", "action_input": raw}

    thought = parsed.get("thought", "")
    action = parsed.get("action", "answer")
    action_input = parsed.get("action_input", "")

    steps.append(AgentStep(step=step_n, type="thought", content=thought, tool=None, done=False))
    log.debug("agent_thought", step=step_n, action=action)

    # Final answer
    if action == "answer" or step_n >= MAX_STEPS:
        steps.append(AgentStep(step=step_n, type="answer", content=action_input, tool=None, done=True))
        return {**state, "steps": steps, "step_count": step_n, "final_answer": action_input, "done": True}

    # Tool call
    steps.append(AgentStep(step=step_n, type="tool_call", content=action_input, tool=action, done=False))

    tool = TOOL_MAP.get(action)
    if tool:
        try:
            observation = await tool.run(action_input)
        except Exception as exc:
            observation = f"Tool error: {exc}"
    else:
        observation = f"Unknown tool '{action}'. Available: {', '.join(TOOL_MAP)}"

    observation = observation[:1500]  # cap observation length
    steps.append(AgentStep(step=step_n, type="observation", content=observation, tool=action, done=False))
    log.info("agent_step", step=step_n, tool=action, obs_len=len(observation))

    return {**state, "steps": steps, "step_count": step_n, "done": False}


def route_continue_or_end(state: MultiAgentState) -> str:
    if state["done"] or state["step_count"] >= MAX_STEPS:
        return "end"
    return "react_step"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_multi_agent_graph() -> Any:
    graph = StateGraph(MultiAgentState)
    graph.add_node("react_step", node_react_step)
    graph.set_entry_point("react_step")
    graph.add_conditional_edges("react_step", route_continue_or_end, {
        "react_step": "react_step",
        "end": END,
    })
    return graph.compile()


_graph: Any | None = None


def get_multi_agent_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_multi_agent_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

async def run_multi_agent(
    objective: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the agent to completion. Returns all steps + final answer."""
    sid = session_id or str(uuid.uuid4())
    await compress_if_needed(sid)

    initial: MultiAgentState = {
        "objective": objective,
        "session_id": sid,
        "messages": [],
        "steps": [],
        "step_count": 0,
        "final_answer": "",
        "done": False,
    }

    graph = get_multi_agent_graph()
    result = await graph.ainvoke(initial)

    # Persist to memory
    append_message(sid, "user", objective)
    append_message(sid, "assistant", result["final_answer"])

    return {
        "session_id": sid,
        "objective": objective,
        "answer": result["final_answer"],
        "steps": result["steps"],
        "total_steps": result["step_count"],
    }


async def stream_multi_agent(
    objective: str,
    session_id: str | None = None,
) -> AsyncGenerator[AgentStep, None]:
    """Stream each ReAct step as it happens."""
    sid = session_id or str(uuid.uuid4())

    state: MultiAgentState = {
        "objective": objective,
        "session_id": sid,
        "messages": [],
        "steps": [],
        "step_count": 0,
        "final_answer": "",
        "done": False,
    }

    seen_steps: set[int] = set()

    graph = get_multi_agent_graph()
    async for event in graph.astream(state):
        node_output = event.get("react_step", {})
        if not node_output:
            continue
        new_steps: list[AgentStep] = node_output.get("steps", [])
        for step in new_steps:
            key = (step["step"], step["type"])
            if key not in seen_steps:
                seen_steps.add(key)
                yield step
        if node_output.get("done"):
            break

    append_message(sid, "user", objective)
    if state.get("final_answer"):
        append_message(sid, "assistant", state["final_answer"])
