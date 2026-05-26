"""Session memory: persist conversation history in Redis, compress when too long."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from rag_agent.core.config import settings

log = structlog.get_logger()

SESSION_TTL = 3600 * 24  # 24h
MAX_MESSAGES_BEFORE_COMPRESS = 20
COMPRESS_TO = 8  # keep last N messages after compression


def _redis() -> Any:
    import redis as redis_lib  # type: ignore[import]

    return redis_lib.from_url(settings.redis_url, decode_responses=True)  # type: ignore[no-untyped-call]


def _key(session_id: str) -> str:
    return f"agent:session:{session_id}:messages"


def _meta_key(session_id: str) -> str:
    return f"agent:session:{session_id}:meta"


def load_messages(session_id: str) -> list[dict[str, str]]:
    """Load message history for a session. Returns [] if not found."""
    try:
        r = _redis()
        raw = r.get(_key(session_id))
        if not raw:
            return []
        return json.loads(raw)  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("memory_load_error", session_id=session_id, error=str(exc))
        return []


def save_messages(session_id: str, messages: list[dict[str, str]]) -> None:
    try:
        r = _redis()
        r.setex(_key(session_id), SESSION_TTL, json.dumps(messages))
    except Exception as exc:
        log.warning("memory_save_error", session_id=session_id, error=str(exc))


def append_message(session_id: str, role: str, content: str) -> list[dict[str, str]]:
    messages = load_messages(session_id)
    messages.append({"role": role, "content": content, "ts": str(time.time())})
    save_messages(session_id, messages)
    return messages


async def compress_if_needed(session_id: str) -> list[dict[str, str]]:
    """If history is too long, summarize old messages and keep recent ones."""
    messages = load_messages(session_id)
    if len(messages) <= MAX_MESSAGES_BEFORE_COMPRESS:
        return messages

    old_msgs = messages[:-COMPRESS_TO]
    recent_msgs = messages[-COMPRESS_TO:]

    old_text = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in old_msgs)

    from rag_agent.services import llm_client

    summary_prompt: list[dict[str, str]] = [
        {
            "role": "user",
            "content": (
                f"Summarize this conversation history in 3-5 sentences, "
                f"preserving key facts, decisions, and context:\n\n{old_text}"
            ),
        }
    ]
    summary, _ = await llm_client.complete(summary_prompt, max_tokens=300, temperature=0.1)

    compressed: list[dict[str, str]] = [
        {"role": "system", "content": f"[Conversation summary]: {summary}", "ts": str(time.time())},
        *recent_msgs,
    ]
    save_messages(session_id, compressed)
    log.info(
        "memory_compressed",
        session_id=session_id,
        old_count=len(old_msgs),
        summary_len=len(summary),
    )
    return compressed


def delete_session(session_id: str) -> None:
    try:
        r = _redis()
        r.delete(_key(session_id), _meta_key(session_id))
    except Exception as exc:
        log.warning("memory_delete_error", session_id=session_id, error=str(exc))


def get_session_meta(session_id: str) -> dict[str, Any]:
    try:
        r = _redis()
        raw = r.get(_meta_key(session_id))
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def set_session_meta(session_id: str, meta: dict[str, Any]) -> None:
    try:
        r = _redis()
        r.setex(_meta_key(session_id), SESSION_TTL, json.dumps(meta))
    except Exception as exc:
        log.warning("memory_meta_error", session_id=session_id, error=str(exc))
