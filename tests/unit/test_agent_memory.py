"""Unit tests for agent memory — Redis mocked."""

from unittest.mock import MagicMock, patch

import pytest

from rag_agent.services.agent_memory import (
    append_message,
    delete_session,
    load_messages,
    save_messages,
)


@pytest.fixture
def mock_redis() -> MagicMock:
    store: dict[str, str] = {}

    class FakeRedis:
        def get(self, key: str) -> str | None:
            return store.get(key)

        def setex(self, key: str, ttl: int, value: str) -> None:
            store[key] = value

        def delete(self, *keys: str) -> None:
            for k in keys:
                store.pop(k, None)

    return FakeRedis()  # type: ignore[return-value]


def test_load_messages_empty(mock_redis: MagicMock) -> None:
    with patch("rag_agent.services.agent_memory._redis", return_value=mock_redis):
        msgs = load_messages("session-new")
        assert msgs == []


def test_save_and_load_messages(mock_redis: MagicMock) -> None:
    with patch("rag_agent.services.agent_memory._redis", return_value=mock_redis):
        messages = [{"role": "user", "content": "hello"}]
        save_messages("sess-1", messages)
        loaded = load_messages("sess-1")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "hello"


def test_append_message(mock_redis: MagicMock) -> None:
    with patch("rag_agent.services.agent_memory._redis", return_value=mock_redis):
        append_message("sess-2", "user", "first message")
        append_message("sess-2", "assistant", "first reply")
        msgs = load_messages("sess-2")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


def test_delete_session(mock_redis: MagicMock) -> None:
    with patch("rag_agent.services.agent_memory._redis", return_value=mock_redis):
        save_messages("sess-del", [{"role": "user", "content": "test"}])
        delete_session("sess-del")
        assert load_messages("sess-del") == []


def test_load_messages_redis_error() -> None:
    """Should return [] gracefully when Redis is unavailable."""
    with patch("rag_agent.services.agent_memory._redis", side_effect=ConnectionError("redis down")):
        msgs = load_messages("any-session")
        assert msgs == []
