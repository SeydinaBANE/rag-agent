"use client";

import { useState } from "react";
import type { SourceChunk } from "@/lib/api";

export default function SourceList({ sources }: { sources: SourceChunk[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;

  return (
    <div className="mt-3 text-xs">
      <button
        onClick={() => setOpen((o) => !o)}
        className="text-gray-500 hover:text-gray-800 flex items-center gap-1 transition-colors"
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>
          {sources.length} source{sources.length > 1 ? "s" : ""}
        </span>
      </button>

      {open && (
        <ul className="mt-2 space-y-2">
          {sources.map((s, i) => (
            <li
              key={i}
              className="border border-gray-200 rounded-lg p-2 bg-gray-50"
            >
              <div className="flex justify-between items-center mb-1">
                <span className="font-medium text-gray-700 truncate max-w-[70%]">
                  {s.source}
                </span>
                <span className="text-gray-400 shrink-0 ml-2">
                  {Math.round(s.score * 100)}%
                </span>
              </div>
              <p className="text-gray-600 line-clamp-2">{s.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
