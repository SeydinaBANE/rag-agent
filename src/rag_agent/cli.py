import asyncio
import uuid
from pathlib import Path

import typer
import uvicorn

app = typer.Typer(name="rag-agent", help="RAG + AI Agent platform CLI")

_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".text", ".html", ".htm"}


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Hot reload (dev only)"),
) -> None:
    """Start the FastAPI server."""
    uvicorn.run("rag_agent.api.main:app", host=host, port=port, reload=reload)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to document or directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
) -> None:
    """Ingest documents into the RAG pipeline (ChromaDB + embeddings)."""
    from rag_agent.services.chunker import chunk_text
    from rag_agent.services.document_loader import load_file
    from rag_agent.services.embedder import embed_texts
    from rag_agent.services.vector_store import upsert_chunks

    target = Path(path)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        glob = "**/*" if recursive else "*"
        files = sorted(
            f for f in target.glob(glob) if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )
    else:
        typer.echo(f"Error: '{path}' does not exist.", err=True)
        raise typer.Exit(1)

    if not files:
        typer.echo("No supported files found.")
        raise typer.Exit(0)

    typer.echo(f"Found {len(files)} file(s). Starting ingestion...\n")

    async def _process_all() -> list[int]:
        counts: list[int] = []
        for i, file in enumerate(files, 1):
            typer.echo(f"  [{i}/{len(files)}] {file.name} ... ", nl=False)
            text = load_file(file)
            if not text.strip():
                typer.echo("0 chunks (empty file)")
                counts.append(0)
                continue
            chunks = chunk_text(text, source=str(file))
            texts = [c.text for c in chunks]
            doc_id = str(uuid.uuid4())
            ids = [f"{doc_id}_{c.index}" for c in chunks]
            metadatas = [{**c.metadata, "doc_id": doc_id} for c in chunks]
            embeddings = await embed_texts(texts)
            await upsert_chunks(texts, embeddings, ids, metadatas)
            typer.echo(f"{len(chunks)} chunks")
            counts.append(len(chunks))
        return counts

    counts = asyncio.run(_process_all())
    typer.echo(f"\nDone. {len(files)} file(s) → {sum(counts)} chunks ingested.")


@app.command()
def eval(
    dataset: str = typer.Option("tests/eval/qa_dataset.json", help="Path to QA dataset"),
    output: str = typer.Option("reports/eval_latest.json", help="Output report path"),
    threshold: float = typer.Option(0.80, help="Minimum faithfulness score (0-1) to pass"),
) -> None:
    """Run RAG quality evaluation with Ragas. Requires live services (ChromaDB, Redis, OpenRouter)."""
    import json
    import sys
    from datetime import UTC, datetime

    dataset_path = Path(dataset)
    if not dataset_path.exists():
        typer.echo(f"Dataset not found: {dataset_path}", err=True)
        raise typer.Exit(1)

    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            answer_relevancy,
            context_recall,
            faithfulness,
        )
    except ImportError:
        typer.echo("Install eval extras: uv pip install -e '.[eval]'", err=True)
        raise typer.Exit(1) from None

    samples: list[dict[str, str]] = json.loads(dataset_path.read_text())
    typer.echo(f"Loaded {len(samples)} samples. Running RAG pipeline...\n")

    async def _run_all() -> list[dict[str, object]]:
        from rag_agent.services.rag_pipeline import answer as rag_answer

        results: list[dict[str, object]] = []
        for i, sample in enumerate(samples, 1):
            q = sample["question"]
            typer.echo(f"  [{i}/{len(samples)}] {q[:70]}", nl=False)
            result = await rag_answer(q)
            typer.echo(" ✓")
            results.append(result)
        return results

    pipeline_results = asyncio.run(_run_all())

    questions = [s["question"] for s in samples]
    answers = [str(r["answer"]) for r in pipeline_results]
    contexts = [
        [str(src["text"]) for src in (r.get("sources") or [])]  # type: ignore[union-attr,index,attr-defined]
        or [""]
        for r in pipeline_results
    ]
    ground_truths = [s["ground_truth"] for s in samples]

    typer.echo("\nEvaluating with Ragas...")
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    scores = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy, context_recall])

    report: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_samples": len(samples),
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_recall": float(scores["context_recall"]),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    typer.echo(f"\nResults → {out_path}\n")
    typer.echo(json.dumps(report, indent=2))

    faithfulness_score = float(report["faithfulness"])  # type: ignore[arg-type]
    if faithfulness_score < threshold:
        typer.echo(f"\nFAIL: faithfulness {faithfulness_score:.3f} < {threshold}", err=True)
        raise typer.Exit(1)
    typer.echo(f"\nPASS: faithfulness {faithfulness_score:.3f} >= {threshold}")
    sys.exit(0)


@app.command("create-key")
def create_key(
    name: str = typer.Argument(..., help="Human-readable name for this API key"),
) -> None:
    """Bootstrap: create the first API key directly via the database (no HTTP auth needed)."""
    import secrets

    from rag_agent.api.v1.deps import hash_key
    from rag_agent.models.database import ApiKey, async_session

    async def _insert() -> str:
        import uuid

        raw = secrets.token_urlsafe(32)
        async with async_session() as db:
            db.add(ApiKey(id=uuid.uuid4(), key_hash=hash_key(raw), name=name))
            await db.commit()
        return raw

    raw_key = asyncio.run(_insert())
    typer.echo(f"\nAPI key '{name}' created:")
    typer.echo(f"  {raw_key}")
    typer.echo("\nStore this securely — it will not be shown again.\n")


if __name__ == "__main__":
    app()
