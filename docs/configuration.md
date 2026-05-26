# Référence de configuration

Toutes les variables sont lues par `src/rag_agent/core/config.py` via `pydantic-settings`. Elles peuvent être définies dans `.env` (copier `.env.example`) ou comme variables d'environnement shell. Les valeurs définies dans `docker-compose.yml` sous `environment:` écrasent `.env` pour les conteneurs `app` et `worker`.

---

## LLM / OpenRouter

| Variable | Type | Défaut | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | str | **requis** | Clé OpenRouter — seule variable obligatoire |
| `OPENROUTER_BASE_URL` | str | `https://openrouter.ai/api/v1` | Base URL OpenAI-compatible |
| `DEFAULT_MODEL` | str | `google/gemini-flash-1.5` | Modèle par défaut (chat, tools, web_search) |
| `QUALITY_MODEL` | str | `anthropic/claude-3.5-sonnet` | Modèle haute qualité (génération agent) |

Les identifiants de modèles suivent la convention OpenRouter (`provider/model-name`). Ne pas hardcoder ces valeurs en dehors de `config.py`.

---

## Base de données

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `DATABASE_URL` | str | **requis** | `postgresql+asyncpg://raguser:ragpass@postgres:5432/ragdb` | <!-- pragma: allowlist secret -->

Utiliser le driver `asyncpg` (async) — jamais `psycopg2` dans les routes FastAPI.

---

## Redis

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `REDIS_URL` | str | `redis://localhost:6379/0` | `redis://redis:6379/0` |

Utilisé par : semantic cache, sessions agent, broker et backend Celery.

---

## ChromaDB

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `CHROMA_HOST` | str | `localhost` | `chromadb` |
| `CHROMA_PORT` | int | `8001` | `8000` (port interne conteneur) |

La propriété `settings.chroma_url` combine host + port : `http://{host}:{port}`.

---

## MinIO

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `MINIO_ENDPOINT` | str | `localhost:9000` | `minio:9000` |
| `MINIO_ACCESS_KEY` | str | `minioadmin` | — |
| `MINIO_SECRET_KEY` | str | `minioadmin` | — |
| `MINIO_BUCKET` | str | `rag-documents` | — |
| `MINIO_SECURE` | bool | `false` | — |

Le bucket est auto-créé par le service `minio-init` à chaque `make up`.

---

## Langfuse (tracing LLM)

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `LANGFUSE_SECRET_KEY` | str | `""` | — |
| `LANGFUSE_PUBLIC_KEY` | str | `""` | — |
| `LANGFUSE_HOST` | str | `http://localhost:3000` | `http://langfuse:3000` |

Si `LANGFUSE_SECRET_KEY` est vide, `langfuse_client.trace_generation()` est un no-op — aucune erreur levée. Pinné à SDK v2 (`>=2.0,<3.0`), image serveur `langfuse/langfuse:2`.

---

## Authentification

| Variable | Type | Défaut | Notes |
|---|---|---|---|
| `API_SECRET_SALT` | str | `changeme` | Sel SHA-256 pour le hachage des clés API |

**Production** : si `APP_ENV=production` et que le sel vaut `changeme`, le démarrage échoue (`model_validator`). Générer une valeur aléatoire : `openssl rand -hex 32`.

---

## Rate limiting

| Variable | Type | Défaut | Contraintes |
|---|---|---|---|
| `RATE_LIMIT_PER_MINUTE` | int | `60` | 1 ≤ n ≤ 10 000 |

Limite par clé API (pas par IP) sur toutes les routes `/api/v1/*`. Dépassement → `429 Too Many Requests` avec `Retry-After: 60`.

---

## Guardrails

| Variable | Type | Défaut | Contraintes | Description |
|---|---|---|---|---|
| `GUARDRAILS_PII_ENABLED` | bool | `true` | — | Active la détection PII (Presidio) |
| `GUARDRAILS_TOXICITY_ENABLED` | bool | `true` | — | Active le filtre toxicité |
| `GUARDRAILS_HALLUCINATION_THRESHOLD` | float | `0.75` | 0.0–1.0 | Seuil NLI post-génération pour log warning + retry agent |

**À distinguer** de `WEB_SEARCH_FALLBACK_THRESHOLD` qui gouverne le grade des chunks *avant* génération.

---

## Cache sémantique

| Variable | Type | Défaut | Contraintes | Description |
|---|---|---|---|---|
| `SEMANTIC_CACHE_ENABLED` | bool | `true` | — | Active/désactive le cache |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | float | `0.92` | 0.0–1.0 | Similarité cosinus minimum pour un cache hit |
| `SEMANTIC_CACHE_TTL_SECONDS` | int | `3600` | ≥ 1 | TTL Redis des entrées cachées |

Valeur de 0.92 = correspondance quasi-exacte. Baisser à ~0.85 pour un cache plus agressif au prix de réponses parfois légèrement hors-sujet.

---

## RAG / Retrieval

| Variable | Type | Défaut | Description |
|---|---|---|---|
| `CHUNK_SIZE` | int | `512` | Taille max d'un chunk (tokens approximatifs) |
| `CHUNK_OVERLAP` | int | `64` | Overlap entre chunks consécutifs |
| `TOP_K` | int | `5` | Nombre de chunks renvoyés après reranking |
| `CROSS_ENCODER_MODEL` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modèle sentence-transformers pour le reranking |

Le cross-encoder est chargé localement (sentence-transformers). Si la lib n'est pas installée, le pipeline se rabat sur RRF seul (`fused[:k]`).

---

## Agent ReAct

| Variable | Type | Défaut | Contraintes | Description |
|---|---|---|---|---|
| `MAX_AGENT_STEPS` | int | `8` | 1–50 | Nombre max d'itérations ReAct avant arrêt forcé |
| `AGENT_CONTEXT_STEPS` | int | `6` | 1–20 | Nombre de steps récents injectés dans le contexte LLM |
| `AGENT_TEMPERATURE` | float | `0.1` | 0.0–2.0 | Température de génération des steps ReAct |
| `AGENT_MAX_TOKENS` | int | `600` | 64–8192 | Max tokens par step ReAct |
| `MAX_OBSERVATION_LENGTH` | int | `1500` | ≥ 100 | Troncature des résultats d'outils (chars) |
| `MAX_RETRIEVAL_RETRIES` | int | `2` | 0–10 | Nombre de retries du graph LangGraph si hallucination trop haute |
| `HALLUCINATION_CHECK_CHUNKS` | int | `3` | 1–20 | Nombre de chunks fournis au vérificateur d'hallucination |
| `WEB_SEARCH_FALLBACK_THRESHOLD` | float | `0.5` | 0.0–1.0 | Score moyen minimum des chunks avant déclenchement du web search |

---

## Outils agent

| Variable | Type | Défaut | Contraintes | Description |
|---|---|---|---|---|
| `WEB_SEARCH_RESULTS` | int | `5` | 1–20 | Nombre de résultats DuckDuckGo renvoyés |
| `FETCH_URL_MAX_CHARS` | int | `3000` | ≥ 100 | Troncature du texte de la page fetchée |
| `SQL_MAX_ROWS` | int | `50` | 1–1000 | Limite de lignes retournées par `sql_query` |

---

## OpenTelemetry

| Variable | Type | Défaut | Docker override |
|---|---|---|---|
| `OTEL_EXPORTER_ENDPOINT` | str | `""` | `http://jaeger:4317` |

Vide = pas d'export de traces. Défini = export OTLP gRPC (Jaeger, Tempo, etc.). Requiert `opentelemetry-exporter-otlp-proto-http` installé.

---

## Application

| Variable | Type | Défaut | Valeurs |
|---|---|---|---|
| `APP_ENV` | str | `development` | `development` \| `production` |
| `LOG_LEVEL` | str | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

**Effets de `APP_ENV=production` :**
- `/docs` et `/redoc` désactivés
- `allow_origins=[]` (CORS fermé)
- Validation que le sel par défaut a été changé <!-- pragma: allowlist secret -->

---

## Overrides Docker Compose

Le fichier `docker-compose.yml` injecte ces variables dans les conteneurs `app` et `worker` pour remplacer les valeurs `localhost` de `.env` par les noms de service internes :

| Variable | `.env` (dev local) | Docker |
|---|---|---|
| `DATABASE_URL` | `…@localhost/ragdb` | `…@postgres:5432/ragdb` |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://redis:6379/0` |
| `CHROMA_HOST` | `localhost` | `chromadb` |
| `CHROMA_PORT` | `8001` | `8000` |
| `MINIO_ENDPOINT` | `localhost:9000` | `minio:9000` |
| `LANGFUSE_HOST` | `http://localhost:3000` | `http://langfuse:3000` |
| `OTEL_EXPORTER_ENDPOINT` | `""` | `http://jaeger:4317` |

Lors de l'ajout d'un nouveau service, ajouter son override dans les sections `environment:` de `app` **et** `worker`.
