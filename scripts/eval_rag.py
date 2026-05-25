#!/usr/bin/env python3
"""RAG evaluation script using Ragas. Run via `make eval`."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with Ragas")
    parser.add_argument("--dataset", default="tests/eval/qa_dataset.json")
    parser.add_argument("--output", default="reports/eval_latest.json")
    parser.add_argument("--faithfulness-threshold", type=float, default=0.80)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
    except ImportError:
        print("Install eval extras: uv pip install -e '.[eval]'")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    samples = json.loads(dataset_path.read_text())
    print(f"Loaded {len(samples)} samples from {dataset_path}")

    # TODO: run samples through live RAG pipeline to collect contexts + answers
    # For now, placeholder structure
    data = {
        "question": [s["question"] for s in samples],
        "answer": ["[placeholder — run pipeline first]"] * len(samples),
        "contexts": [["[placeholder context]"]] * len(samples),
        "ground_truth": [s["ground_truth"] for s in samples],
    }

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_samples": len(samples),
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))

    if output["faithfulness"] < args.faithfulness_threshold:
        print(f"\nFAIL: faithfulness {output['faithfulness']:.3f} < {args.faithfulness_threshold}")
        sys.exit(1)
    print("\nPASS: all quality gates met")


if __name__ == "__main__":
    main()
