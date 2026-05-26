# Architecture technique

## Vue d'ensemble

rag-agent est une plateforme RAG + Agent IA constituée de :

- **FastAPI** — API principale, SSE streaming, auth, rate limiting
- **Celery + Redis** — ingestion asynchrone de documents
- **ChromaDB** — stockage vectoriel (embeddings)
- **PostgreSQL** — métadonnées, clés API
- **MinIO** — fichiers bruts (PDF, DOCX, images)
- **Next.js 14** — interface chat utilisateur
- **LangGraph** — orchestration des agents (RAG graph + ReAct)
- **OpenRouter** — accès unifié aux LLMs

---

## Topologie Docker

```
Internet
    │
    ▼
┌─────────────────────────────────────────────┐
│  Docker network (bridge)                    │
│                                             │
│  frontend:3000 ──/api/*──► app:8000         │
│                                             │
│  app:8000                                   │
│    ├── postgres:5432  (api_keys, docs)      │
│    ├── redis:6379     (cache, sessions)     │
│    ├── chromadb:8000  (vectors)             │
│    ├── minio:9000     (raw files)           │
│    └── jaeger:4317    (OTLP traces)         │
│                                             │
│  worker (Celery)                            │
│    ├── redis:6379     (broker/backend)      │
│    ├── chromadb:8000  (upsert)              │
│    └── minio:9000     (upload)              │
│                                             │
│  flower:5555  → redis:6379                  │
│  langfuse:3000 → postgres:5432              │
│  grafana:3001  → prometheus:9090            │
│  n8n:5678      → postgres:5432              │
└─────────────────────────────────────────────┘
```

Ports exposés à l'hôte : 8000, 3003, 3000, 3001, 5432, 6379, 8001, 9000, 9001, 9090, 16686, 5555, 5678.

---

## Flux de requête

### 1. Chat RAG (`POST /api/v1/chat`)

```
Client
  │ X-API-Key
  ▼
deps.require_api_key        ← vérifie hash SHA-256 en base
  │
  ▼
guardrails.guard_input      ← toxicity keyword check
  │                           PII detection (Presidio)
  │                           → anonymise ou lève GuardrailError (400)
  ▼
semantic_cache.get_cached   ← embedde la query, cherche dans Redis
  │ HIT → retourne réponse cachée                    (cache hit)
  │ MISS ↓
  ▼
retriever.retrieve
  ├── embedder.embed_query   → OpenRouter /embeddings
  ├── vector_store.query_similar (ChromaDB, top_k×2)
  ├── _bm25_scores           → BM25 sur les textes récupérés
  ├── _rrf_fuse              → Reciprocal Rank Fusion (dense + BM25)
  └── _cross_encoder_rerank  → cross-encoder/ms-marco-MiniLM-L-6-v2
  │
  ▼
llm_client.complete         → OpenRouter /chat/completions
  │
  ▼
semantic_cache.set_cached   → stocke embedding + réponse dans Redis (TTL 3600s)
  │
  ▼
guardrails.guard_output     ← NLI hallucination score (DeBERTa)
  │                           log warning si score < threshold
  ▼
ChatResponse { answer, sources, cached, confidence, usage }
```

**Streaming** (`GET /api/v1/chat/stream`) : même pipeline jusqu'à `llm_client.stream()` qui yield les tokens via SSE. Le cache est mis à jour après réception du token final.

---

### 2. Agent LangGraph (`POST /api/v1/agent`)

Utilise un `StateGraph` LangGraph avec les nœuds suivants :

```
retrieve ──► grade_relevance
                │
                ├── avg_score < web_search_fallback_threshold (0.5)
                │   └──► web_search ──► generate
                │
                └── score OK ──► generate
                                    │
                                    ▼
                              check_hallucination
                                    │
                                    ├── score < guardrails_hallucination_threshold (0.75)
                                    │   AND iteration < max_retrieval_retries (2)
                                    │   └──► retrieve  (retry)
                                    │
                                    └── score OK ──► END
```

**Nœuds :**
- `retrieve` — appelle `retriever.retrieve()` (dense + BM25 + RRF + cross-encoder)
- `grade_relevance` — calcule le score moyen des chunks récupérés
- `web_search` — fallback LLM avec prompt "provide a brief factual summary" (max 256 tokens)
- `generate` — utilise `quality_model` (claude-3.5-sonnet) avec contexte
- `check_hallucination` — demande au LLM de scorer son propre output (0.0–1.0)

Le graph est compilé une seule fois au démarrage (`_graph` singleton) et réutilisé.

---

### 3. Agent ReAct multi-étapes (`POST /api/v1/agent/run`)

Pattern **Reason + Act** avec boucle outil. Max `max_agent_steps` (8) itérations.

```
objective (user)
    │
    ▼
node_react_step  ← LLM génère JSON : {thought, action, action_input}
    │
    ├── action == "answer" ──► node_synthesize ──► SSE done event
    │
    └── action == "tool_name"
            │
            ▼
        TOOL_MAP[tool_name](action_input)
            ├── web_search     → DuckDuckGo (max web_search_results=5)
            ├── fetch_url      → httpx GET (max fetch_url_max_chars=3000 chars)
            ├── rag_search     → retriever.retrieve()
            ├── sql_query      → sqlite3 en mémoire (max sql_max_rows=50)
            ├── generate_report → LLM formatting
            └── code_runner    → exec() isolé
            │
            ▼
        node_react_step  (observation → next thought)
```

**Mémoire de session** : l'historique des messages est persisté dans Redis avec TTL 24h. Le scope Redis est `sha256(api_key)[:16]` — chaque clé API a son espace de nommage isolé.

**Compression automatique** : si `len(messages) > 20`, les anciens messages sont résumés par le LLM (3-5 phrases) et remplacés par un message système `[Conversation summary]`.

---

### 4. Ingestion asynchrone (`POST /api/v1/ingest/file`)

```
Client (multipart/form-data)
    │
    ▼
ingest.py ──► base64-encode ──► ingest_document.delay()  (Celery task)
    │
    └── retourne { job_id, status: "queued" }

# Dans le worker Celery :
    ▼
document_loader.load_bytes
    ├── PDF   → PyMuPDF (fitz)
    ├── DOCX  → python-docx
    └── TXT/HTML → décoder + nettoyer
    │
    ▼
chunker.chunk_text
    └── RecursiveCharacterTextSplitter(chunk_size=512, overlap=64)
    │
    ▼
embedder.embed_texts          → OpenRouter /embeddings (batch)
    │
    ▼
vector_store.upsert_chunks    → ChromaDB collection "documents"
    │
    ▼
MinIO upload (raw file)       → bucket "rag-documents"

# Polling :
GET /api/v1/jobs/{job_id} → { status: PENDING|STARTED|SUCCESS|FAILURE }
```

**Webhook MinIO** : les objets déposés directement dans le bucket déclenchent `POST /api/v1/webhooks/minio` qui dispatche automatiquement la même tâche Celery.

---

### 5. Pipeline OCR (`POST /api/v1/ocr/extract`)

```
image/pdf (multipart)
    │
    ▼
preprocessor.preprocess
    ├── deskew (scipy rotation correction)
    ├── denoise (OpenCV Gaussian blur)
    └── contrast enhancement
    │
    ▼
extractor.run_tesseract       → texte brut OCR
    │
    ▼
auto-detect doc type          → heuristiques sur mots-clés (invoice/receipt/contract/form)
    │
    ▼
extractor.extract_with_vision → vision LLM (OpenRouter, max 2048 tokens)
    │                           prompt structuré par type de document
    ▼
confidence scoring            → par champ, global
    │
    ▼
ExtractionResult { doc_type, raw_text, fields: {value, confidence}, overall_confidence }
```

---

## Couche de stockage

### PostgreSQL

Tables :
- `api_keys` — `id, name, key_hash (SHA-256 + salt), created_at, last_used_at, is_active`
- `documents` — métadonnées des fichiers ingérés
- `alembic_version` — suivi des migrations

Les bases `n8n` et `langfuse` sont créées par `infra/postgres/init.sql` au premier démarrage du conteneur.

### Redis

| Préfixe de clé | Contenu | TTL |
|---|---|---|
| `semantic_cache:{hash}` | réponse RAG cachée | 3600s (configurable) |
| `agent:session:{scope}:{sid}:messages` | historique de messages JSON | 86400s (24h) |
| `agent:session:{scope}:{sid}:meta` | métadonnées de session | 86400s |
| Celery broker | tâches en queue | — |
| Celery backend | résultats de tâches | — |

Le `scope` est `sha256(api_key)[:16]` — garantit l'isolation des sessions entre clés API.

### ChromaDB

Collection unique `documents`. Chaque chunk stocke :
- `id` — UUID
- `embedding` — vecteur float (dimension selon le modèle)
- `document` — texte du chunk
- `metadata` — `{ source, chunk_index, doc_id }`

### MinIO

Bucket `rag-documents` (auto-créé par `minio-init`). Un objet par fichier ingéré, chemin `{doc_id}/{filename}`.

---

## Sécurité

### Authentification

Toutes les routes `/api/v1/*` passent par `deps.require_api_key` :

```python
# deps.py
async def require_api_key(x_api_key: str = Header(...)) -> str:
    key_hash = sha256(f"{x_api_key}{settings.api_secret_salt}".encode()).hexdigest()
    # Vérifie key_hash en base → 401 si absent ou inactif
```

La clé en clair n'est jamais stockée — seulement le SHA-256 salé.

### Rate limiting

`slowapi` avec `key_func = _rate_limit_key` :
- Si `X-API-Key` présent → limite par clé (résistant aux proxies)
- Sinon → limite par IP (pour `/health`, `/metrics`)

Défaut : 60 req/min, configurable via `RATE_LIMIT_PER_MINUTE`. Dépasse → `429` avec header `Retry-After: 60`.

### Guardrails

1. **Toxicity** — liste de mots-clés (extensible avec un modèle de classification)
2. **PII** — Microsoft Presidio (9 types d'entités : PERSON, EMAIL, PHONE, CREDIT_CARD, IBAN, IP, LOCATION, NRP, MEDICAL_LICENSE) — anonymise vers `<ENTITY_TYPE>`
3. **Hallucination** — cross-encoder NLI (DeBERTa v3 small) — score [0,1], log warning si < `guardrails_hallucination_threshold`

### CORS

`APP_ENV=development` → `allow_origins=["*"]`
`APP_ENV=production` → `allow_origins=[]` (à configurer explicitement)

---

## Observabilité

### Prometheus (`/metrics`)

Métriques instrumentées au niveau middleware :

| Métrique | Type | Labels |
|---|---|---|
| `http_requests_total` | Counter | method, endpoint, status_code |
| `http_request_duration_seconds` | Histogram | method, endpoint |
| `rag_queries_total` | Counter | cached (true/false) |
| `rag_retrieval_score` | Histogram | — |
| `llm_tokens_total` | Counter | type (prompt/completion), model |
| `model_router_cost_usd_total` | Counter | model |
| `guardrail_blocked_total` | Counter | reason |
| `ocr_documents_total` | Counter | doc_type |

### OpenTelemetry (Jaeger)

`FastAPIInstrumentor` crée automatiquement des spans pour chaque requête. Si `OTEL_EXPORTER_ENDPOINT` est défini, les traces sont exportées en OTLP gRPC vers Jaeger (`jaeger:4317`).

### Langfuse

`langfuse_client.trace_generation()` est un context manager qui enregistre :
- prompt / completion / model / tokens / latency
- No-op si `LANGFUSE_SECRET_KEY` est vide

### Logging

`structlog` en JSON structuré en production, coloré en développement. Chaque requête reçoit un `request_id` UUID qui est propagé dans tous les logs du même scope via `contextvars`.

---

## Frontend (Next.js 14)

### Architecture

```
Browser
  │ SSE stream
  ▼
frontend:3003 (conteneur Docker) ou :3000 (dev local)
  │ /api/* rewrite
  ▼
app:8000 (FastAPI)
```

### Streaming SSE

Le frontend utilise `fetch` + `ReadableStream` (pas `EventSource`) pour pouvoir envoyer le header `X-API-Key` :

```typescript
// lib/api.ts
const res = await fetch(`/api/v1/chat/stream?query=${q}&session_id=${sid}`, {
  headers: { "X-API-Key": apiKey },
  signal,
});
// Lecture ligne par ligne du body stream
for await (const line of readLines(res.body)) {
  if (line.startsWith("data: ")) {
    const { token, done, ...meta } = JSON.parse(line.slice(6));
    if (token) callbacks.onToken(token);
    if (done) callbacks.onMeta(meta);
  }
}
```

### Persistance

- `localStorage["rag_agent_api_key"]` — clé API (jamais envoyée en clair dans l'URL)
- `localStorage["rag_agent_session_id"]` — UUID de session (mémoire conversationnelle côté serveur)

---

## Modèle de routage LLM

`model_router.select_model(mode)` expose 5 modes :

| Mode | Modèle sélectionné | Usage |
|---|---|---|
| `default` | `settings.default_model` (Gemini Flash 1.5) | Requêtes standard |
| `quality` | `settings.quality_model` (Claude 3.5 Sonnet) | Génération agent |
| `ab_test` | Tirage pondéré entre N modèles | A/B testing |
| `cheapest` | Modèle au coût/token le plus bas du catalogue | Batch, eval |
| `fastest` | Modèle à la latence la plus faible | Streaming temps réel |

Le routeur trace le coût USD par appel dans la métrique Prometheus `model_router_cost_usd_total`.
