#!/usr/bin/env python3
"""RAG evaluation script using Ragas. Run via `make eval` or `rag-agent eval`."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


async def _run_pipeline(samples: list[dict[str, str]]) -> list[dict[str, object]]:
    from rag_agent.services.rag_pipeline import answer as rag_answer

    results: list[dict[str, object]] = []
    for i, sample in enumerate(samples, 1):
        q = sample["question"]
        print(f"  [{i}/{len(samples)}] {q[:70]}", end="", flush=True)
        result = await rag_answer(q)
        print(" ✓")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with Ragas")
    parser.add_argument("--dataset", default="tests/eval/qa_dataset.json")
    parser.add_argument("--output", default="reports/eval_latest.json")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.80)
    args = parser.parse_args()

    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            answer_relevancy,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("Install eval extras: uv pip install -e '.[eval]'")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    samples: list[dict[str, str]] = json.loads(dataset_path.read_text())
    print(f"Loaded {len(samples)} samples. Running RAG pipeline...\n")

    pipeline_results = asyncio.run(_run_pipeline(samples))

    data = {
        "question": [s["question"] for s in samples],
        "answer": [str(r["answer"]) for r in pipeline_results],
        "contexts": [
            [src["text"] for src in r.get("sources", [])] or [""]  # type: ignore[index]
            for r in pipeline_results
        ],
        "ground_truth": [s["ground_truth"] for s in samples],
    }

    print("\nEvaluating with Ragas...")
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    output: dict[str, object] = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(samples),
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))

    if float(output["faithfulness"]) < args.faithfulness_threshold:
        print(f"\nFAIL: faithfulness {output['faithfulness']:.3f} < {args.faithfulness_threshold}")
        sys.exit(1)
    print("\nPASS: all quality gates met")


if __name__ == "__main__":
    main()
