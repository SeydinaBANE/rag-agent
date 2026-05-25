# Fine-tuning Guide

Fine-tune a small open-source model on your RAG Q&A data using LoRA/QLoRA via [Unsloth](https://github.com/unslothai/unsloth) (4× faster than vanilla HuggingFace, runs on a single consumer GPU).

## Prerequisites

- GPU with ≥ 16 GB VRAM (A10G, RTX 3090, RTX 4090) — or use a Colab A100
- Python 3.12, CUDA 12.x
- A `qa_dataset.json` with at least 200 question/answer pairs

Install the finetune extras:
```bash
uv pip install -e ".[finetune]"
```

## Workflow

```
qa_dataset.json
    │
    ▼
prepare_dataset.py   →   data/finetune_dataset.json   (Alpaca format)
    │
    ▼
train.py             →   models/rag-agent-ft/          (LoRA adapters)
    │
    ▼
eval_finetuned.py    →   reports/ft_comparison_latest.json
    │
    ▼
Export GGUF          →   Ollama / OpenRouter custom model
```

---

## 1. Prepare the dataset

The eval dataset (`tests/eval/qa_dataset.json`) must follow this schema:
```json
[
  {
    "question": "What is the capital of Senegal?",
    "ground_truth": "Dakar",
    "context": "Senegal is a country in West Africa. Its capital is Dakar…"
  }
]
```

Convert to Alpaca format (default) or ShareGPT:
```bash
python scripts/finetune/prepare_dataset.py \
  --input tests/eval/qa_dataset.json \
  --output data/finetune_dataset.json \
  --format alpaca \
  --min-length 20        # filter out very short answers
```

The script filters samples where `ground_truth` is too short and formats them as:
```json
{
  "instruction": "What is the capital of Senegal?",
  "input": "Context: Senegal is a country in West Africa…",
  "output": "Dakar"
}
```

For ShareGPT format (`--format sharegpt`), each sample becomes a multi-turn conversation with a system prompt.

---

## 2. Train

```bash
python scripts/finetune/train.py \
  --model unsloth/mistral-7b-v0.3-bnb-4bit \
  --dataset data/finetune_dataset.json \
  --output models/rag-agent-ft \
  --epochs 3
```

### Key parameters

| Flag | Default | Notes |
|---|---|---|
| `--model` | `unsloth/mistral-7b-v0.3-bnb-4bit` | Any Unsloth-compatible model |
| `--epochs` | `3` | Start at 3; increase to 5 if loss still dropping |
| `--batch-size` | `2` | Increase to 4 if VRAM allows |
| `--lr` | `2e-4` | Standard LoRA learning rate |
| `--lora-r` | `16` | LoRA rank — 8 for speed, 32 for quality |
| `--max-seq-len` | `2048` | Must cover your longest context + answer |
| `--export-gguf` | false | Add to also export GGUF for Ollama |

### Recommended models (Unsloth)

| Model | VRAM | Speed |
|---|---|---|
| `unsloth/mistral-7b-v0.3-bnb-4bit` | 16 GB | Fast |
| `unsloth/llama-3.1-8b-bnb-4bit` | 16 GB | Fast |
| `unsloth/llama-3.3-70b-bnb-4bit` | 48 GB | Slow, high quality |

Adapters are saved in `models/rag-agent-ft/` after each epoch. Training logs go to the console (no W&B by default — set `report_to="wandb"` in `TrainingArguments` to enable).

---

## 3. Evaluate

Compare the fine-tuned model against the base model on the eval dataset:

```bash
python scripts/finetune/eval_finetuned.py \
  --base google/gemini-flash-1.5 \
  --finetuned models/rag-agent-ft \
  --dataset tests/eval/qa_dataset.json \
  --output reports/ft_comparison_latest.json
```

The script runs each question through both models and computes BLEU/ROUGE scores. Output:
```json
{
  "base_model": "google/gemini-flash-1.5",
  "finetuned_model": "models/rag-agent-ft",
  "n_samples": 150,
  "base_bleu": 0.42,
  "finetuned_bleu": 0.61,
  "base_rouge1": 0.55,
  "finetuned_rouge1": 0.72
}
```

---

## 4. Deploy

### Option A — Ollama (local)

```bash
# Export GGUF during training:
python scripts/finetune/train.py --export-gguf ...

# Create Modelfile
cat > Modelfile <<EOF
FROM ./models/rag-agent-ft/gguf/model-q4_k_m.gguf
SYSTEM "You are an expert assistant. Answer using only the provided context."
EOF

ollama create rag-agent -f Modelfile
ollama run rag-agent
```

### Option B — OpenRouter custom model

Upload your LoRA adapters or merged model to Hugging Face, then add the `HF_MODEL_ID` to your OpenRouter account as a custom model. Update `config.py`:

```python
default_model: str = "your-hf-username/rag-agent-ft"
```

### Option C — Use via model router

The model router can A/B test your fine-tuned model against the base:

```python
from rag_agent.services.model_router import ABTestConfig, call_with_routing

ab = ABTestConfig(
    models=["google/gemini-flash-1.5", "your-hf-username/rag-agent-ft"],
    weights=[50.0, 50.0],
)
answer, usage, model_used = await call_with_routing(messages, mode="ab_test", ab_config=ab)
```

---

## Tips

- **Dataset size:** 200 samples is the minimum; 1 000+ gives significantly better results
- **Overfitting:** if validation loss rises while training loss drops, reduce epochs or add more data
- **Context length:** set `--max-seq-len` to at least `chunk_size * top_k` (default: 512 × 5 = 2560, so use 4096)
- **QLoRA vs LoRA:** the `bnb-4bit` models use QLoRA (quantized base + full-precision adapters) — keeps VRAM low with minimal quality loss
