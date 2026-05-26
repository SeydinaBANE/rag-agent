#!/usr/bin/env python3
"""
Fine-tune a model with LoRA/QLoRA via Unsloth (4x faster than HuggingFace).

Usage:
    pip install -e ".[finetune]"
    python scripts/finetune/train.py \
        --model unsloth/mistral-7b-v0.3-bnb-4bit \
        --dataset data/finetune_dataset.json \
        --output models/rag-agent-ft \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune with Unsloth LoRA")
    parser.add_argument("--model", default="unsloth/mistral-7b-v0.3-bnb-4bit")
    parser.add_argument("--dataset", default="data/finetune_dataset.json")
    parser.add_argument("--output", default="models/rag-agent-ft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--export-gguf", action="store_true", help="Export to GGUF for Ollama")
    args = parser.parse_args()

    try:
        from datasets import Dataset  # type: ignore[import]
        from trl import SFTTrainer, TrainingArguments  # type: ignore[import]
        from unsloth import FastLanguageModel  # type: ignore[import]
    except ImportError:
        print("Install finetune extras: pip install -e '.[finetune]'")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}. Run prepare_dataset.py first.")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )

    raw = json.loads(dataset_path.read_text())
    # Format to Alpaca prompt
    alpaca_prompt = (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    )

    def format_sample(sample: dict[str, str]) -> dict[str, str]:
        return {
            "text": alpaca_prompt.format(
                instruction=sample.get("instruction", ""),
                input=sample.get("input", ""),
                output=sample.get("output", ""),
            )
        }

    dataset = Dataset.from_list([format_sample(s) for s in raw])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            output_dir=args.output,
            save_strategy="epoch",
            report_to="none",
        ),
    )

    print(f"Training for {args.epochs} epochs…")
    trainer.train()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Model saved → {output_path}")

    if args.export_gguf:
        print("Exporting to GGUF (q4_k_m)…")
        model.save_pretrained_gguf(
            str(output_path / "gguf"), tokenizer, quantization_method="q4_k_m"
        )
        print("GGUF saved. Load with: ollama create rag-agent -f Modelfile")


if __name__ == "__main__":
    main()
