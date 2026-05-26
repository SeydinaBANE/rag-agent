"""ChromaDB vector store wrapper."""

from __future__ import annotations

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from rag_agent.core.config import settings

log = structlog.get_logger()

COLLECTION_DOCS = "documents"
COLLECTION_CACHE = "semantic_cache"

_client: chromadb.AsyncHttpClient | None = None


async def get_chroma() -> chromadb.AsyncHttpClient:
    global _client
    if _client is None:
        _client = await chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


async def get_collection(name: str) -> chromadb.AsyncCollection:
    client = await get_chroma()
    return await client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


async def upsert_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    ids: list[str],
    metadatas: list[dict[str, str]],
) -> None:
    collection = await get_collection(COLLECTION_DOCS)
    await collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    log.info("upserted_chunks", n=len(chunks))


async def query_similar(
    query_embedding: list[float],
    top_k: int | None = None,
    where: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    k = top_k or settings.top_k
    collection = await get_collection(COLLECTION_DOCS)
    results = await collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "metadata": meta,
            "score": 1.0 - float(dist),  # cosine similarity
        }
        for doc, meta, dist in zip(docs, metas, distances, strict=False)
    ]


async def delete_by_source(source: str) -> None:
    collection = await get_collection(COLLECTION_DOCS)
    await collection.delete(where={"source": source})
    log.info("deleted_chunks_by_source", source=source)
