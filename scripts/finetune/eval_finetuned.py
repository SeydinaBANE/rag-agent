#!/usr/bin/env python3
"""
Compare fine-tuned model vs base model on the Ragas eval dataset.

Usage:
    python scripts/finetune/eval_finetuned.py \
        --base google/gemini-flash-1.5 \
        --finetuned models/rag-agent-ft/gguf/model-q4_k_m.gguf \
        --dataset tests/eval/qa_dataset.json \
        --output reports/ft_comparison_latest.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def query_openrouter(question: str, model: str, api_key: str) -> str:
    import httpx

    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 256,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def query_ollama(question: str, model: str = "rag-agent") -> str:
    """Query a local Ollama model (for GGUF)."""
    import httpx

    r = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": question, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["response"]


def _rouge1(answer: str, ground_truth: str) -> float:
    """Simple ROUGE-1 recall: fraction of GT words present in answer."""
    a_words = set(answer.lower().split())
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0
    return len(a_words & gt_words) / len(gt_words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="google/gemini-flash-1.5", help="Base model (OpenRouter)")
    parser.add_argument("--finetuned", default="rag-agent", help="Fine-tuned model name in Ollama")
    parser.add_argument("--dataset", default="tests/eval/qa_dataset.json")
    parser.add_argument("--output", default="reports/ft_comparison_latest.json")
    parser.add_argument("--openrouter-key", default="", help="OpenRouter API key")
    args = parser.parse_args()

    import os

    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

    samples = json.loads(Path(args.dataset).read_text())
    results = []

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"[{i + 1}/{len(samples)}] {question[:60]}…")

        base_answer = query_openrouter(question, args.base, api_key) if api_key else "N/A"

        try:
            ft_answer = query_ollama(question, args.finetuned)
        except Exception as e:
            ft_answer = f"ERROR: {e}"

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "base_answer": base_answer,
                "finetuned_answer": ft_answer,
                "base_rouge1": round(_rouge1(base_answer, ground_truth), 3),
                "finetuned_rouge1": round(_rouge1(ft_answer, ground_truth), 3),
            }
        )

    avg_base = sum(r["base_rouge1"] for r in results) / len(results)
    avg_ft = sum(r["finetuned_rouge1"] for r in results) / len(results)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(results),
        "base_model": args.base,
        "finetuned_model": args.finetuned,
        "avg_rouge1_base": round(avg_base, 3),
        "avg_rouge1_finetuned": round(avg_ft, 3),
        "delta": round(avg_ft - avg_base, 3),
        "samples": results,
    }

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 50}")
    print(f"Base ROUGE-1:       {avg_base:.3f}")
    print(f"Fine-tuned ROUGE-1: {avg_ft:.3f}")
    print(f"Delta:              {avg_ft - avg_base:+.3f}")
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
