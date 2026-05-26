"""Celery tasks for async document ingestion."""

from __future__ import annotations

import asyncio
import uuid

import structlog

from rag_agent.core.celery_app import celery_app
from rag_agent.services.chunker import chunk_text
from rag_agent.services.document_loader import load_bytes
from rag_agent.services.vector_store import upsert_chunks

log = structlog.get_logger()


@celery_app.task(bind=True, name="ingest_document", max_retries=3)
def ingest_document(
    self: object,
    content_b64: str,
    filename: str,
    doc_id: str | None = None,
) -> dict[str, object]:
    """Ingest a single document (base64-encoded bytes)."""
    import base64

    doc_id = doc_id or str(uuid.uuid4())
    log.info("ingestion_start", doc_id=doc_id, filename=filename)

    try:
        content = base64.b64decode(content_b64)
        text = load_bytes(content, filename)

        if not text.strip():
            return {"doc_id": doc_id, "status": "empty", "chunks": 0}

        chunks = chunk_text(text, source=filename)
        texts = [c.text for c in chunks]
        ids = [f"{doc_id}_{c.index}" for c in chunks]
        metadatas = [{**c.metadata, "doc_id": doc_id} for c in chunks]

        # Run async embedding in sync Celery task
        embeddings = asyncio.run(_embed(texts))
        asyncio.run(_upsert(texts, embeddings, ids, metadatas))

        log.info("ingestion_done", doc_id=doc_id, n_chunks=len(chunks))
        return {"doc_id": doc_id, "status": "done", "chunks": len(chunks)}

    except Exception as exc:
        log.error("ingestion_error", doc_id=doc_id, error=str(exc))
        raise self.retry(exc=exc, countdown=10) from exc  # type: ignore[attr-defined]


async def _embed(texts: list[str]) -> list[list[float]]:
    from rag_agent.services.embedder import embed_texts

    return await embed_texts(texts)


async def _upsert(
    texts: list[str],
    embeddings: list[list[float]],
    ids: list[str],
    metadatas: list[dict[str, str]],
) -> None:
    await upsert_chunks(texts, embeddings, ids, metadatas)
