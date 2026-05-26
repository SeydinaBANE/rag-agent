# API Reference

Base URL: `http://localhost:8000`

## Authentication

All `/api/v1/*` routes require an `X-API-Key` header.

**Bootstrap (first key):**
```bash
# With Docker running and DB migrated:
uv run rag-agent create-key myapp
# → prints the plaintext key once, store it securely
```

**Subsequent keys** (via API once you have one):
```bash
curl -X POST http://localhost:8000/api/v1/keys \
  -H "X-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"name": "ci-runner"}'
```

**Dev shortcut:** set `X-API-Key: dev-key-<API_SECRET_SALT>` (only works when `APP_ENV=development`).

---

## Error format

```json
{ "error": "ERROR_CODE", "detail": "human-readable message" }
```

| HTTP | Code | Cause |
|---|---|---|
| 401 | — | Missing or invalid `X-API-Key` |
| 400 | `INGEST_ERROR` | Unsupported file type or parse failure |
| 400 | `GUARDRAIL_BLOCKED` | PII or toxic content detected |
| 502 | `LLM_ERROR` | OpenRouter call failed |
| 500 | `INTERNAL_ERROR` | Unexpected server error |

---

## Chat

### `POST /api/v1/chat`

Standard RAG Q&A. Checks semantic cache first; on miss runs full retrieval pipeline.

**Request**
```json
{
  "query": "What is retrieval-augmented generation?",
  "model": "google/gemini-flash-1.5-8b",
  "session_id": "user-abc",
  "top_k": 5
}
```
`model` and `session_id` are optional. `top_k` defaults to `settings.top_k` (5).

**Response**
```json
{
  "answer": "RAG combines retrieval of relevant documents with LLM generation…",
  "sources": [
    { "text": "…chunk excerpt…", "source": "rag_intro.pdf", "score": 0.87 }
  ],
  "cached": false,
  "usage": { "prompt_tokens": 512, "completion_tokens": 128 }
}
```

---

### `GET /api/v1/chat/stream?query=…&session_id=…&model=…`

Same pipeline as `/chat` but streams tokens as SSE events. `query` is required (1–4096 chars). `session_id` is optional (reuses conversation memory).

**Events**
```
data: {"token": "RAG", "done": false}
data: {"token": " combines", "done": false}
data: {"token": "", "done": true, "sources": [...], "cached": false, "usage": {...}}
```

**SSE error event:**
```
data: {"error": "LLM_ERROR: upstream timeout", "done": true}
```

---

## Agent

### `POST /api/v1/agent`

LangGraph graph-based agent. Runs: **retrieve → grade relevance → (web_search if low score) → generate → hallucination check → (retry ×2)**.

**Request** — same schema as `/chat`.

**Response**
```json
{
  "answer": "…",
  "confidence": 0.91,
  "sources": [...],
  "iterations": 1,
  "web_searched": false,
  "hallucination_score": 0.88
}
```

---

### `POST /api/v1/agent/run`

ReAct multi-step agent. Breaks the objective into thought → tool → observation loops (max 8 steps), then synthesizes a final answer.

**Request**
```json
{
  "objective": "Find the latest GPU benchmarks for Mistral-7B and summarize them.",
  "session_id": "session-xyz"
}
```

**Response**
```json
{
  "session_id": "session-xyz",
  "objective": "…",
  "answer": "According to recent benchmarks…",
  "total_steps": 4,
  "steps": [
    { "step": 1, "type": "thought", "content": "I need to search for…", "tool": null, "done": false },
    { "step": 2, "type": "tool_call", "content": "GPU benchmarks Mistral-7B", "tool": "web_search", "done": false },
    { "step": 3, "type": "observation", "content": "…search results…", "tool": null, "done": false },
    { "step": 4, "type": "answer", "content": "According to…", "tool": null, "done": true }
  ]
}
```

**Available tools:** `web_search`, `fetch_url`, `rag_search`, `sql_query`, `generate_report`, `code_runner`

---

### `GET /api/v1/agent/run/stream?objective=…&session_id=…`

Same as `/agent/run` but streams each `AgentStep` as an SSE event.

```
data: {"step": 1, "type": "thought", "content": "…", "tool": null, "done": false}
data: {"step": 2, "type": "tool_call", "content": "…", "tool": "web_search", "done": false}
data: {"step": 4, "type": "answer", "content": "…", "tool": null, "done": true}
```

**SSE error event** (emitted before the stream closes on any exception):
```
data: {"error": "human-readable message", "done": true}
```

---

### `GET /api/v1/agent/run/sessions/{session_id}`

Return the message history for a session (stored in Redis).

### `DELETE /api/v1/agent/run/sessions/{session_id}`

Delete session history from Redis.

---

## Ingestion

Documents are chunked (512 tokens, 64 overlap), embedded, and stored in ChromaDB. Processing is async via Celery — poll `/jobs/{id}` for status.

### `POST /api/v1/ingest/file`

Upload a file. Supported: `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/plain`, `text/html`. Max 50 MB.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -H "X-API-Key: <key>" \
  -F "file=@report.pdf"
```

**Response** `202 Accepted`
```json
{ "job_id": "abc-123", "filename": "report.pdf", "status": "queued" }
```

---

### `POST /api/v1/ingest/text`

```json
{ "text": "LangGraph is a library for building stateful multi-actor applications…", "source": "langraph-docs" }
```

**Response** `202 Accepted` — same as file ingest.

---

### `GET /api/v1/jobs/{job_id}`

Poll ingestion task status.

```json
{
  "job_id": "abc-123",
  "status": "SUCCESS",
  "result": { "doc_id": "…", "chunks": 14 },
  "error": null
}
```

`status` values: `PENDING` → `STARTED` → `SUCCESS` | `FAILURE`

---

## OCR

### `GET /api/v1/ocr/schemas`

List document types and their extraction schemas.

```json
{
  "schemas": [
    { "type": "invoice", "description": "Facture — numéro, date, vendeur, client, montants, articles" },
    { "type": "receipt", "description": "Reçu / ticket de caisse — commerçant, total, articles" },
    { "type": "contract", "description": "Contrat — parties, dates, type, clauses clés" },
    { "type": "form",    "description": "Formulaire générique — paires champ/valeur" },
    { "type": "unknown", "description": "Type auto-détecté" }
  ]
}
```

---

### `POST /api/v1/ocr/extract`

Extract structured data from an image or PDF. Supported: `image/png`, `image/jpeg`, `image/tiff`, `image/webp`, `application/pdf`. Max 20 MB.

**Pipeline:** deskew/denoise → Tesseract raw text → auto-detect document type → vision LLM structured extraction → confidence scoring.

```bash
curl -X POST http://localhost:8000/api/v1/ocr/extract \
  -H "X-API-Key: <key>" \
  -F "file=@invoice.png" \
  -F "doc_type=invoice" \
  -F "lang=fra+eng"
```

`doc_type` is optional (auto-detected if omitted). `lang` defaults to `fra+eng`.

**Response**
```json
{
  "doc_type": "invoice",
  "raw_text": "FACTURE N° 2024-001…",
  "fields": {
    "invoice_number": { "value": "2024-001", "confidence": 0.97 },
    "date":           { "value": "2024-03-15", "confidence": 0.95 },
    "total_amount":   { "value": "1250.00", "confidence": 0.91 }
  },
  "overall_confidence": 0.94,
  "warnings": []
}
```

---

## API Keys

### `POST /api/v1/keys`

Create a new API key. Returns the plaintext key **once** — store it securely.

**Request** `{ "name": "production-app" }`

**Response** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "production-app",
  "key": "rXq8…43_A",
  "created_at": "2026-05-25T20:00:00Z",
  "last_used_at": null
}
```

---

### `GET /api/v1/keys`

List active keys. The plaintext key is never returned after creation.

```json
[
  { "id": "550e…", "name": "production-app", "created_at": "…", "last_used_at": "…" }
]
```

---

### `DELETE /api/v1/keys/{key_id}`

Revoke a key. `204 No Content`.

---

## Rate limiting

All `/api/v1/*` endpoints are rate-limited per API key (default 60 req/min, configurable via `RATE_LIMIT_PER_MINUTE`). Exceeding the limit returns `429 Too Many Requests`. The limiter key is the `X-API-Key` header value; falls back to client IP when the header is absent.

---

## Observability

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "version": "0.1.0"}` |
| `GET /metrics` | Prometheus metrics (text/plain) |

Key metrics: `http_requests_total`, `http_request_duration_seconds`, `rag_queries_total`, `llm_tokens_total`, `model_router_cost_usd_total`, `guardrail_blocked_total`, `ocr_documents_total`.

| Tool | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3001 | Dashboards (auto-provisioned, admin/admin) |
| Langfuse | http://localhost:3000 | LLM prompt/completion traces |
| Jaeger | http://localhost:16686 | Distributed traces (OTLP via port 4317) |
| Flower | http://localhost:5555 | Celery task monitor |
