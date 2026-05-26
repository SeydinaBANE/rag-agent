"use client";

import { useState } from "react";

interface Props {
  apiKey: string;
  onSave: (apiKey: string) => void;
  onClose: () => void;
  onResetSession: () => void;
}

export default function SettingsModal({
  apiKey,
  onSave,
  onClose,
  onResetSession,
}: Props) {
  const [draft, setDraft] = useState(apiKey);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    onSave(draft.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6 space-y-5">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-800">Paramètres</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700">
            Clé API{" "}
            <span className="text-gray-400 font-normal">(X-API-Key)</span>
          </label>
          <input
            type="password"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Votre clé API rag-agent"
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm
                       focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <p className="text-xs text-gray-400">
            Générez une clé avec{" "}
            <code className="bg-gray-100 px-1 rounded">
              uv run rag-agent create-key mon-app
            </code>
          </p>
        </div>

        <button
          onClick={handleSave}
          className="w-full py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium
                     hover:bg-indigo-700 transition-colors"
        >
          {saved ? "✓ Sauvegardé" : "Sauvegarder"}
        </button>

        <div className="border-t border-gray-100 pt-4">
          <p className="text-xs text-gray-500 mb-2">Session actuelle</p>
          <button
            onClick={() => {
              onResetSession();
              onClose();
            }}
            className="text-sm text-red-500 hover:text-red-700 transition-colors"
          >
            Réinitialiser la session (effacer la mémoire)
          </button>
        </div>
      </div>
    </div>
  );
}
