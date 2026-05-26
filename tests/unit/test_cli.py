"""Tests for CLI commands: ingest and eval."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from rag_agent.cli import app

runner = CliRunner()


# ── ingest ───────────────────────────────────────────────────────────────────


def test_ingest_missing_path() -> None:
    result = runner.invoke(app, ["ingest", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_ingest_empty_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = runner.invoke(app, ["ingest", tmp])
    assert result.exit_code == 0
    assert "No supported files found" in result.output


def test_ingest_single_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Hello world.")
        tmp_path = f.name

    mock_chunks = [MagicMock(text="Hello world.", index=0, metadata={"source": tmp_path})]

    with (
        patch(
            "rag_agent.services.document_loader.load_file",
            return_value="Hello world.",
        ),
        patch("rag_agent.services.chunker.chunk_text", return_value=mock_chunks),
        patch(
            "rag_agent.services.embedder.embed_texts",
            new_callable=AsyncMock,
            return_value=[[0.1, 0.2]],
        ),
        patch("rag_agent.services.vector_store.upsert_chunks", new_callable=AsyncMock),
    ):
        result = runner.invoke(app, ["ingest", tmp_path])

    assert result.exit_code == 0
    assert "1 chunks" in result.output
    assert "Done" in result.output


def test_ingest_empty_file_skipped() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("")
        tmp_path = f.name

    with patch("rag_agent.services.document_loader.load_file", return_value=""):
        result = runner.invoke(app, ["ingest", tmp_path])

    assert result.exit_code == 0
    assert "empty file" in result.output


def test_ingest_directory_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "sub"
        sub.mkdir()
        (sub / "doc.txt").write_text("content")

        mock_chunks = [MagicMock(text="content", index=0, metadata={"source": "doc.txt"})]
        with (
            patch("rag_agent.services.document_loader.load_file", return_value="content"),
            patch("rag_agent.services.chunker.chunk_text", return_value=mock_chunks),
            patch(
                "rag_agent.services.embedder.embed_texts",
                new_callable=AsyncMock,
                return_value=[[0.1]],
            ),
            patch("rag_agent.services.vector_store.upsert_chunks", new_callable=AsyncMock),
        ):
            result = runner.invoke(app, ["ingest", tmp, "--recursive"])

    assert result.exit_code == 0
    assert "Done" in result.output


def test_ingest_unsupported_extension_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "file.xyz").write_text("data")
        result = runner.invoke(app, ["ingest", tmp])
    assert result.exit_code == 0
    assert "No supported files found" in result.output


# ── eval ─────────────────────────────────────────────────────────────────────


def test_eval_missing_dataset() -> None:
    result = runner.invoke(app, ["eval", "--dataset", "/nonexistent/qa.json"])
    assert result.exit_code == 1
    assert "Dataset not found" in result.output


def test_eval_missing_extras() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump([{"question": "q", "ground_truth": "a"}], f)
        tmp_path = f.name

    with patch.dict("sys.modules", {"datasets": None, "ragas": None, "ragas.metrics": None}):
        result = runner.invoke(app, ["eval", "--dataset", tmp_path])

    assert result.exit_code == 1
    assert "Install" in result.output


def _make_ragas_modules(scores: dict[str, float]) -> dict[str, object]:
    """Return sys.modules patches that make ragas importable and return given scores."""
    mock_evaluate = MagicMock(return_value=scores)
    mock_dataset_cls = MagicMock()
    mock_dataset_cls.from_dict.return_value = MagicMock()

    mock_datasets = MagicMock()
    mock_datasets.Dataset = mock_dataset_cls

    mock_ragas = MagicMock()
    mock_ragas.evaluate = mock_evaluate

    mock_metrics = MagicMock()

    return {
        "datasets": mock_datasets,
        "ragas": mock_ragas,
        "ragas.metrics": mock_metrics,
        "ragas.evaluation": MagicMock(),
        "ragas.llms": MagicMock(),
    }


def test_eval_runs_pipeline_and_passes() -> None:
    samples = [{"question": "What is RAG?", "ground_truth": "Retrieval-Augmented Generation."}]
    mock_rag_result = {
        "answer": "RAG is Retrieval-Augmented Generation.",
        "sources": [{"text": "RAG stands for Retrieval-Augmented Generation.", "score": 0.9}],
        "cached": False,
        "usage": {},
    }
    scores = {"faithfulness": 0.95, "answer_relevancy": 0.88, "context_recall": 0.90}

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / "qa.json"
        dataset_path.write_text(json.dumps(samples))
        output_path = Path(tmp) / "reports" / "out.json"

        with (
            patch.dict("sys.modules", _make_ragas_modules(scores)),
            patch(
                "rag_agent.services.rag_pipeline.answer",
                new_callable=AsyncMock,
                return_value=mock_rag_result,
            ),
        ):
            result = runner.invoke(
                app,
                ["eval", "--dataset", str(dataset_path), "--output", str(output_path)],
            )

    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_eval_fails_below_threshold() -> None:
    samples = [{"question": "What is RAG?", "ground_truth": "RAG answer."}]
    mock_rag_result = {"answer": "I don't know.", "sources": [], "cached": False, "usage": {}}
    scores = {"faithfulness": 0.30, "answer_relevancy": 0.40, "context_recall": 0.20}

    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / "qa.json"
        dataset_path.write_text(json.dumps(samples))
        output_path = Path(tmp) / "out.json"

        with (
            patch.dict("sys.modules", _make_ragas_modules(scores)),
            patch(
                "rag_agent.services.rag_pipeline.answer",
                new_callable=AsyncMock,
                return_value=mock_rag_result,
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "eval",
                    "--dataset",
                    str(dataset_path),
                    "--output",
                    str(output_path),
                    "--threshold",
                    "0.80",
                ],
            )

    assert result.exit_code == 1
    assert "FAIL" in result.output
