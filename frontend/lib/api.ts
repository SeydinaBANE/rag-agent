export interface SourceChunk {
  text: string;
  source: string;
  score: number;
}

export interface ChatMeta {
  sources: SourceChunk[];
  cached: boolean;
  confidence: number | null;
  usage: { prompt_tokens: number; completion_tokens: number } | null;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onMeta: (meta: ChatMeta) => void;
  onError: (msg: string) => void;
  onDone: () => void;
}

/**
 * Stream tokens from GET /api/v1/chat/stream via fetch + ReadableStream.
 * Uses fetch (not EventSource) so we can send the X-API-Key header.
 * Falls back to POST /api/v1/chat on SSE failure.
 */
export async function streamChat(
  query: string,
  sessionId: string | null,
  apiKey: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const params = new URLSearchParams({ query });
  if (sessionId) params.set("session_id", sessionId);

  try {
    const res = await fetch(`/api/v1/chat/stream?${params}`, {
      headers: { "X-API-Key": apiKey },
      signal,
    });

    if (!res.ok) {
      const body = await res.text();
      callbacks.onError(`HTTP ${res.status}: ${body}`);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError("No response body");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === "[DONE]") continue;

        try {
          const payload = JSON.parse(raw) as {
            token?: string;
            done?: boolean;
            error?: string;
          };
          if (payload.error) {
            callbacks.onError(payload.error);
            return;
          }
          if (payload.token !== undefined && !payload.done) {
            callbacks.onToken(payload.token);
          }
        } catch {
          // ignore malformed SSE frames
        }
      }
    }

    callbacks.onDone();
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    // Fallback to sync POST
    await chatSync(query, sessionId, apiKey, callbacks, signal);
  }
}

async function chatSync(
  query: string,
  sessionId: string | null,
  apiKey: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  try {
    const res = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "X-API-Key": apiKey, "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: sessionId }),
      signal,
    });
    if (!res.ok) {
      callbacks.onError(`HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    callbacks.onToken(data.answer ?? "");
    callbacks.onMeta({
      sources: data.sources ?? [],
      cached: data.cached ?? false,
      confidence: data.confidence ?? null,
      usage: data.usage ?? null,
    });
    callbacks.onDone();
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      callbacks.onError(String(err));
    }
  }
}
