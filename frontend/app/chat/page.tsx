"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import SettingsModal from "@/components/SettingsModal";
import { streamChat, type ChatMeta } from "@/lib/api";
import type { Message } from "@/components/MessageBubble";

const STORAGE_API_KEY = "rag_agent_api_key"; // pragma: allowlist secret
const STORAGE_SESSION = "rag_agent_session_id";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Hydrate from localStorage
  useEffect(() => {
    const storedKey = localStorage.getItem(STORAGE_API_KEY) ?? "";
    const storedSession = localStorage.getItem(STORAGE_SESSION);
    setApiKey(storedKey);
    setSessionId(storedSession);
    if (!storedKey) setShowSettings(true);
  }, []);

  const saveApiKey = (key: string) => {
    setApiKey(key);
    localStorage.setItem(STORAGE_API_KEY, key);
  };

  const resetSession = () => {
    const newId = uuidv4();
    setSessionId(newId);
    localStorage.setItem(STORAGE_SESSION, newId);
    setMessages([]);
  };

  const handleSend = useCallback(
    async (query: string) => {
      if (!apiKey) {
        setShowSettings(true);
        return;
      }

      setError(null);
      const userMsg: Message = { id: uuidv4(), role: "user", content: query };
      const assistantId = uuidv4();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setLoading(true);

      // Ensure session exists
      let sid = sessionId;
      if (!sid) {
        sid = uuidv4();
        setSessionId(sid);
        localStorage.setItem(STORAGE_SESSION, sid);
      }

      const controller = new AbortController();
      abortRef.current = controller;

      let accumulated = "";
      let finalMeta: ChatMeta | null = null;

      await streamChat(
        query,
        sid,
        apiKey,
        {
          onToken(token) {
            accumulated += token;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: accumulated } : m
              )
            );
          },
          onMeta(meta) {
            finalMeta = meta;
          },
          onError(msg) {
            setError(msg);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: `⚠️ ${msg}`, streaming: false }
                  : m
              )
            );
          },
          onDone() {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      streaming: false,
                      meta: finalMeta ?? undefined,
                    }
                  : m
              )
            );
          },
        },
        controller.signal
      );

      setLoading(false);
    },
    [apiKey, sessionId]
  );

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-xs font-bold">
            R
          </div>
          <span className="font-semibold text-gray-800 text-sm">rag-agent</span>
        </div>

        <div className="flex items-center gap-3">
          {sessionId && (
            <span className="hidden sm:block text-xs text-gray-400 font-mono truncate max-w-[140px]">
              {sessionId.slice(0, 8)}…
            </span>
          )}
          <button
            onClick={() => setShowSettings(true)}
            title="Paramètres"
            className="w-8 h-8 rounded-lg flex items-center justify-center text-gray-500
                       hover:bg-gray-100 transition-colors"
          >
            ⚙️
          </button>
        </div>
      </header>

      {/* Error toast */}
      {error && (
        <div className="mx-4 mt-3 px-4 py-2 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="ml-3 text-red-400 hover:text-red-600">
            ×
          </button>
        </div>
      )}

      {/* Messages */}
      <ChatWindow messages={messages} />

      {/* Input */}
      <InputBar onSend={handleSend} disabled={loading} />

      {/* Settings modal */}
      {showSettings && (
        <SettingsModal
          apiKey={apiKey}
          onSave={saveApiKey}
          onClose={() => setShowSettings(false)}
          onResetSession={resetSession}
        />
      )}
    </div>
  );
}
