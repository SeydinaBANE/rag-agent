import type { ChatMeta } from "@/lib/api";

export default function MetaBadges({ meta }: { meta: ChatMeta }) {
  return (
    <div className="flex flex-wrap gap-2 mt-2 text-xs">
      {meta.cached && (
        <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium">
          ⚡ Cache
        </span>
      )}
      {meta.confidence !== null && (
        <span className="px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
          Confiance {Math.round(meta.confidence * 100)}%
        </span>
      )}
      {meta.usage && (
        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
          {(meta.usage.prompt_tokens ?? 0) + (meta.usage.completion_tokens ?? 0)} tokens
        </span>
      )}
    </div>
  );
}
