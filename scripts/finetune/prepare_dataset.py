#!/usr/bin/env python3
"""
Convert RAG eval dataset + ChromaDB chunks into Alpaca/ShareGPT format for fine-tuning.

Usage:
    python scripts/finetune/prepare_dataset.py \
        --input tests/eval/qa_dataset.json \
        --output data/finetune_dataset.json \
        --format alpaca
"""

import argparse
import json
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert assistant. Answer questions precisely based on the provided context."
)


def to_alpaca(sample: dict[str, str]) -> dict[str, str]:
    return {
        "instruction": sample["question"],
        "input": f"Context: {sample.get('context', '')}",
        "output": sample["ground_truth"],
    }


def to_sharegpt(sample: dict[str, str]) -> dict[str, object]:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {
                "from": "human",
                "value": f"Context: {sample.get('context', '')}\n\nQuestion: {sample['question']}",
            },
            {"from": "gpt", "value": sample["ground_truth"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tests/eval/qa_dataset.json")
    parser.add_argument("--output", default="data/finetune_dataset.json")
    parser.add_argument("--format", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument("--min-length", type=int, default=20, help="Min answer length")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    samples = json.loads(input_path.read_text())
    # Filter quality
    samples = [s for s in samples if len(s.get("ground_truth", "")) >= args.min_length]

    converter = to_alpaca if args.format == "alpaca" else to_sharegpt
    dataset = [converter(s) for s in samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

    print(f"Saved {len(dataset)} samples ({args.format}) → {output_path}")


if __name__ == "__main__":
    main()
