"""Text embedding via OpenRouter-compatible endpoint (text-embedding-3-small)."""

import structlog

from rag_agent.services.llm_client import get_client

log = structlog.get_logger()

EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIM = 1536


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts. Batches up to 100 items."""
    if not texts:
        return []

    client = get_client()
    all_embeddings: list[list[float]] = []

    # OpenRouter embeds up to 100 texts per call
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        log.debug("embedded_batch", n=len(batch), offset=i)

    return all_embeddings


async def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    results = await embed_texts([text])
    return results[0]
