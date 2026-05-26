"""Tests for llm_client — OpenRouter async wrapper."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import rag_agent.services.llm_client as llm_module
from rag_agent.core.exceptions import LLMError
from rag_agent.services.llm_client import complete, get_client, stream


@pytest.fixture(autouse=True)
def reset_client():
    original = llm_module._client
    yield
    llm_module._client = original


def _make_completion(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> Any:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_get_client_singleton():
    llm_module._client = None
    with patch("rag_agent.services.llm_client.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        c1 = get_client()
        c2 = get_client()
    assert c1 is c2
    mock_cls.assert_called_once()


def _noop_trace(name, model, messages, trace_id=None):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield {}

    return _ctx()


@pytest.mark.asyncio
async def test_complete_returns_answer_and_usage():
    fake_resp = _make_completion("The answer", 20, 8)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=fake_resp)
    llm_module._client = mock_openai

    with patch("rag_agent.services.langfuse_client.trace_generation", side_effect=_noop_trace):
        answer, usage = await complete([{"role": "user", "content": "hi"}], model="gpt-4")

    assert answer == "The answer"
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_complete_raises_llm_error_on_exception():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))
    llm_module._client = mock_openai

    with patch("rag_agent.services.langfuse_client.trace_generation", side_effect=_noop_trace):
        with pytest.raises(LLMError, match="API down"):
            await complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_stream_yields_tokens():
    async def _fake_chunks() -> AsyncIterator[Any]:
        for token in ["Hello", " world"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=_fake_chunks())
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_stream_ctx)
    llm_module._client = mock_openai

    tokens = []
    async for token in stream([{"role": "user", "content": "hi"}], model="gpt-4"):
        tokens.append(token)

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_stream_raises_llm_error_on_exception():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
    llm_module._client = mock_openai

    with pytest.raises(LLMError, match="timeout"):
        await stream([{"role": "user", "content": "hi"}]).__anext__()


@pytest.mark.asyncio
async def test_complete_handles_no_usage():
    """When response.usage is None, tokens default to 0."""
    msg = MagicMock()
    msg.content = "ans"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(return_value=resp)
    llm_module._client = mock_openai

    with patch("rag_agent.services.langfuse_client.trace_generation", side_effect=_noop_trace):
        answer, usage = await complete([{"role": "user", "content": "hi"}])

    assert answer == "ans"
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
