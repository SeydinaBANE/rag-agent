"""Recursive character text splitter with semantic boundary awareness."""

from __future__ import annotations

from rag_agent.core.config import settings


class Chunk:
    __slots__ = ("index", "metadata", "text")

    def __init__(self, text: str, index: int, metadata: dict[str, str] | None = None) -> None:
        self.text = text
        self.index = index
        self.metadata = metadata or {}


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    source: str = "",
) -> list[Chunk]:
    """Split text into overlapping chunks respecting paragraph and sentence boundaries."""
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    # Split on paragraph breaks first, then sentences, then words
    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    raw_chunks = _recursive_split(text.strip(), separators, size)

    # Apply overlap by including tail of previous chunk
    chunks: list[Chunk] = []
    for i, chunk_text_ in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            tail = prev[-overlap:].lstrip()
            chunk_text_ = tail + " " + chunk_text_ if tail else chunk_text_
        chunks.append(Chunk(text=chunk_text_.strip(), index=i, metadata={"source": source}))

    return [c for c in chunks if c.text]


def _recursive_split(text: str, separators: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]

    sep = ""
    for s in separators:
        if s and s in text:
            sep = s
            break

    if not sep:
        # Hard split by character count
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself may be too large — recurse with next separator
            remaining_seps = separators[separators.index(sep) + 1 :]
            if len(part) > size and remaining_seps:
                chunks.extend(_recursive_split(part, remaining_seps, size))
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks
