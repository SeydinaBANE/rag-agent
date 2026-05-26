"""Pydantic request/response schemas for the API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# ── Chat ────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User question")
    model: str | None = Field(None, description="Override LLM model (OpenRouter model ID)")
    session_id: str | None = Field(None, description="Optional session ID for memory")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    cached: bool
    usage: dict[str, int]


# ── Ingest ──────────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    job_id: str
    filename: str
    status: str  # queued | done | error


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to ingest")
    source: str = Field(..., min_length=1, description="Source identifier")


# ── Jobs ────────────────────────────────────────────────────────────────────


class JobStatus(BaseModel):
    job_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE
    result: dict[str, object] | None = None
    error: str | None = None


# ── API Keys ────────────────────────────────────────────────────────────────


class KeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)


class KeyInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime
    last_used_at: datetime | None = None


class KeyCreated(KeyInfo):
    key: str  # returned only on creation, never again


# ── Eval ────────────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    n_samples: int
    passed: bool
