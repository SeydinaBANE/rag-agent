# rag-agent — Chat UI

Next.js 14 chat interface for the rag-agent platform. Streams responses token-by-token from the FastAPI SSE endpoints and displays sources, confidence scores, and cache status after each answer.

## Stack

- **Next.js 14** (App Router, TypeScript, Tailwind CSS)
- **react-markdown + remark-gfm** — Markdown rendering in assistant messages
- **fetch + ReadableStream** — SSE streaming with custom `X-API-Key` header (not `EventSource`)
- **localStorage** — session ID and API key persistence

## Local development

```bash
# From repo root — requires FastAPI running on :8000
make frontend-install   # npm ci
make frontend-dev       # Next.js dev server → http://localhost:3000
```

The dev server proxies all `/api/*` requests to `http://localhost:8000` via `next.config.mjs` rewrites, so CORS is not an issue.

## Docker

The frontend is included in the full stack:

```bash
make up   # starts frontend container on http://localhost:3003
```

The container uses a multi-stage build (Node 20 Alpine, standalone output) and receives `API_URL=http://app:8000` at runtime so the rewrite points to the FastAPI service name inside the Docker network.

## First use

1. Open the app — the SettingsModal opens automatically when no API key is stored
2. Enter your API key (create one with `uv run rag-agent create-key myapp`)
3. Send a message — tokens appear in real time
4. After the response: sources expand, badges show cache hit / confidence / token count
5. Use the ⚙️ button to change key or reset the session (clears server-side memory)

## Architecture

```
app/
├── page.tsx          → redirect to /chat
└── chat/
    └── page.tsx      → state management (messages, streaming, session, settings)

components/
├── ChatWindow.tsx    → message list + auto-scroll + empty state
├── MessageBubble.tsx → Markdown rendering, streaming cursor, MetaBadges + SourceList
├── InputBar.tsx      → auto-resize textarea, Enter=send / Shift+Enter=newline
├── SettingsModal.tsx → API key input (localStorage) + session reset
├── MetaBadges.tsx    → ⚡ Cache · Confiance X% · N tokens
└── SourceList.tsx    → collapsible source chunks with relevance score

lib/
└── api.ts            → streamChat() + chatSync() fallback
```

## Environment

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI base URL used in Next.js rewrites |

Set in `docker-compose.yml` for the container; leave unset for local dev (falls back to localhost).
