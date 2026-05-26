"""Tests for ingestion_tasks — Celery document ingestion task."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_agent.services.ingestion_tasks import _embed, _upsert, ingest_document


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _fake_chunk(text: str = "chunk", index: int = 0, source: str = "test.txt") -> MagicMock:
    c = MagicMock()
    c.text = text
    c.index = index
    c.metadata = {"source": source}
    return c


def test_ingest_document_success():
    content_b64 = _b64("This is a test document with some content for chunking.")

    with (
        patch("rag_agent.services.ingestion_tasks.load_bytes", return_value="test text content"),
        patch("rag_agent.services.ingestion_tasks.chunk_text", return_value=[_fake_chunk()]),
        patch("rag_agent.services.ingestion_tasks.asyncio.run", side_effect=[[[0.1, 0.2]], None]),
    ):
        result = ingest_document(content_b64, "test.txt")

    assert result["status"] == "done"
    assert result["chunks"] == 1


def test_ingest_document_empty_text():
    content_b64 = _b64("   ")

    with patch("rag_agent.services.ingestion_tasks.load_bytes", return_value="   "):
        result = ingest_document(content_b64, "empty.txt")

    assert result["status"] == "empty"
    assert result["chunks"] == 0


def test_ingest_document_retry_on_error():
    content_b64 = _b64("some content")

    with patch(
        "rag_agent.services.ingestion_tasks.load_bytes", side_effect=RuntimeError("load error")
    ):
        with pytest.raises(RuntimeError):
            ingest_document(content_b64, "bad.txt")


def test_ingest_document_uses_provided_doc_id():
    content_b64 = _b64("content")

    with (
        patch("rag_agent.services.ingestion_tasks.load_bytes", return_value="content"),
        patch("rag_agent.services.ingestion_tasks.chunk_text", return_value=[_fake_chunk()]),
        patch("rag_agent.services.ingestion_tasks.asyncio.run", side_effect=[[[0.1]], None]),
    ):
        result = ingest_document(content_b64, "x.txt", doc_id="my-custom-id")

    assert result["doc_id"] == "my-custom-id"


@pytest.mark.asyncio
async def test_embed_helper():
    with patch("rag_agent.services.embedder.embed_texts", new=AsyncMock(return_value=[[0.1, 0.2]])):
        result = await _embed(["text"])
    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_upsert_helper():
    with patch("rag_agent.services.ingestion_tasks.upsert_chunks", new=AsyncMock()) as mock:
        await _upsert(["t"], [[0.1]], ["id1"], [{"source": "a"}])
    mock.assert_called_once()
