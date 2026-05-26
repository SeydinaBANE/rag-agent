"""Tests for rag_pipeline — core RAG answer + streaming."""

from unittest.mock import AsyncMock, patch

import pytest

from rag_agent.services import rag_pipeline


def _fake_chunks() -> list[dict]:
    return [
        {
            "text": "RAG stands for Retrieval-Augmented Generation.",
            "metadata": {"source": "doc.pdf"},
            "score": 0.92,
        },
        {
            "text": "It combines retrieval with generation.",
            "metadata": {"source": "doc.pdf"},
            "score": 0.85,
        },
    ]


@pytest.mark.asyncio
async def test_answer_cache_hit():
    with (
        patch(
            "rag_agent.services.rag_pipeline.get_cached",
            new=AsyncMock(return_value="cached answer"),
        ),
        patch("rag_agent.services.rag_pipeline.retrieve") as mock_retrieve,
    ):
        result = await rag_pipeline.answer("what is RAG?")

    assert result["answer"] == "cached answer"
    assert result["cached"] is True
    assert result["sources"] == []
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_answer_cache_miss_full_pipeline():
    chunks = _fake_chunks()
    with (
        patch("rag_agent.services.rag_pipeline.get_cached", new=AsyncMock(return_value=None)),
        patch("rag_agent.services.rag_pipeline.retrieve", new=AsyncMock(return_value=chunks)),
        patch(
            "rag_agent.services.rag_pipeline.llm_client.complete",
            new=AsyncMock(
                return_value=("The answer.", {"prompt_tokens": 20, "completion_tokens": 10})
            ),
        ),
        patch("rag_agent.services.rag_pipeline.set_cached", new=AsyncMock()),
    ):
        result = await rag_pipeline.answer("what is RAG?")

    assert result["answer"] == "The answer."
    assert result["cached"] is False
    assert len(result["sources"]) == 2
    assert result["usage"]["prompt_tokens"] == 20


@pytest.mark.asyncio
async def test_answer_no_chunks():
    with (
        patch("rag_agent.services.rag_pipeline.get_cached", new=AsyncMock(return_value=None)),
        patch("rag_agent.services.rag_pipeline.retrieve", new=AsyncMock(return_value=[])),
        patch(
            "rag_agent.services.rag_pipeline.llm_client.complete",
            new=AsyncMock(return_value=("I don't know.", {})),
        ),
        patch("rag_agent.services.rag_pipeline.set_cached", new=AsyncMock()),
    ):
        result = await rag_pipeline.answer("unknown question")

    assert result["answer"] == "I don't know."
    assert result["sources"] == []


@pytest.mark.asyncio
async def test_answer_stream_yields_tokens():
    chunks = _fake_chunks()

    async def fake_stream(*args, **kwargs):
        for token in ["Hello", " there"]:
            yield token

    with (
        patch("rag_agent.services.rag_pipeline.retrieve", new=AsyncMock(return_value=chunks)),
        patch("rag_agent.services.rag_pipeline.llm_client.stream", side_effect=fake_stream),
        patch("rag_agent.services.rag_pipeline.set_cached", new=AsyncMock()),
    ):
        tokens = []
        async for token in rag_pipeline.answer_stream("what is RAG?"):
            tokens.append(token)

    assert tokens == ["Hello", " there"]


@pytest.mark.asyncio
async def test_answer_stream_no_chunks():
    async def fake_stream(*args, **kwargs):
        yield "fallback answer"

    with (
        patch("rag_agent.services.rag_pipeline.retrieve", new=AsyncMock(return_value=[])),
        patch("rag_agent.services.rag_pipeline.llm_client.stream", side_effect=fake_stream),
        patch("rag_agent.services.rag_pipeline.set_cached", new=AsyncMock()),
    ):
        tokens = []
        async for token in rag_pipeline.answer_stream("unknown"):
            tokens.append(token)

    assert "fallback answer" in tokens


@pytest.mark.asyncio
async def test_answer_tracks_retrieval_score():
    chunks = _fake_chunks()
    with (
        patch("rag_agent.services.rag_pipeline.get_cached", new=AsyncMock(return_value=None)),
        patch("rag_agent.services.rag_pipeline.retrieve", new=AsyncMock(return_value=chunks)),
        patch(
            "rag_agent.services.rag_pipeline.llm_client.complete",
            new=AsyncMock(return_value=("ans", {})),
        ),
        patch("rag_agent.services.rag_pipeline.set_cached", new=AsyncMock()),
        patch("rag_agent.services.rag_pipeline.RETRIEVAL_SCORE") as mock_hist,
    ):
        await rag_pipeline.answer("q")

    mock_hist.observe.assert_called_once_with(0.92)
