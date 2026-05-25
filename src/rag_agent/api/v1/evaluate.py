"""RAG quality evaluation endpoint — runs Ragas metrics on a QA dataset."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from rag_agent.api.v1.deps import require_api_key
from rag_agent.models.schemas import EvalResult
from rag_agent.services import rag_pipeline

log = structlog.get_logger()
router = APIRouter(prefix="/evaluate", tags=["eval"])

DEFAULT_DATASET = "tests/eval/qa_dataset.json"
FAITHFULNESS_THRESHOLD = 0.80


@router.post("", response_model=EvalResult)
async def evaluate(
    dataset: str = Query(DEFAULT_DATASET, description="Path to QA dataset JSON"),
    max_samples: int = Query(20, ge=1, le=100, description="Max samples to evaluate"),
    _: str = Depends(require_api_key),
) -> EvalResult:
    """
    Run Ragas evaluation (faithfulness, answer_relevancy, context_recall) on a QA dataset.

    Dataset format:
        [{"question": "...", "ground_truth": "...", "context": "..."}]

    Requires `.[eval]` extras: `uv pip install -e '.[eval]'`
    """
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset}",
        )

    try:
        samples: list[dict[str, str]] = json.loads(dataset_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset JSON: {exc}") from exc

    samples = samples[:max_samples]
    if not samples:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    log.info("eval_start", n_samples=len(samples), dataset=dataset)

    # Run RAG pipeline for each question concurrently
    pipeline_results = await asyncio.gather(
        *[rag_pipeline.answer(s["question"]) for s in samples],
        return_exceptions=True,
    )

    questions, answers, contexts, ground_truths = [], [], [], []
    for sample, result in zip(samples, pipeline_results):
        if isinstance(result, Exception):
            log.warning("eval_sample_failed", question=sample["question"][:60], error=str(result))
            continue
        questions.append(sample["question"])
        answers.append(str(result["answer"]))
        contexts.append([c["text"] for c in result.get("sources", [])] or [sample.get("context", "")])
        ground_truths.append(sample.get("ground_truth", ""))

    if not questions:
        raise HTTPException(status_code=502, detail="All RAG pipeline calls failed")

    # Ragas evaluation runs synchronously — offload to thread pool
    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    scores = await asyncio.get_event_loop().run_in_executor(None, _run_ragas, ragas_data)

    result = EvalResult(
        faithfulness=scores["faithfulness"],
        answer_relevancy=scores["answer_relevancy"],
        context_recall=scores["context_recall"],
        n_samples=len(questions),
        passed=scores["faithfulness"] >= FAITHFULNESS_THRESHOLD,
    )
    log.info("eval_done", **scores, n_samples=len(questions), passed=result.passed)
    return result


def _run_ragas(data: dict[str, Any]) -> dict[str, float]:
    """Blocking Ragas call — must be run in a thread pool."""
    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import answer_relevancy, context_recall, faithfulness  # type: ignore[import]
    except ImportError as exc:
        raise HTTPException(  # type: ignore[misc]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ragas not installed. Run: uv pip install -e '.[eval]'",
        ) from exc

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }
