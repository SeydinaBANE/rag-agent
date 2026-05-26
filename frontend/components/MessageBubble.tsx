import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMeta } from "@/lib/api";
import MetaBadges from "./MetaBadges";
import SourceList from "./SourceList";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  streaming?: boolean;
}

export default function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-indigo-600 text-white rounded-br-sm"
            : "bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        ) : (
          <>
            <div className="prose prose-sm max-w-none prose-p:my-1 prose-pre:bg-gray-100">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
            {message.streaming && (
              <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5 align-middle" />
            )}
            {!message.streaming && message.meta && (
              <>
                <MetaBadges meta={message.meta} />
                <SourceList sources={message.meta.sources} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
