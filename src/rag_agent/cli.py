import typer
import uvicorn

app = typer.Typer(name="rag-agent", help="RAG + AI Agent platform CLI")


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
    recursive: bool = typer.Option(False, "--recursive", "-r"),
) -> None:
    """Ingest documents into the RAG pipeline."""
    typer.echo(f"Ingesting: {path} (recursive={recursive})")
    # TODO: call ingestion service


@app.command()
def eval(
    dataset: str = typer.Option("tests/eval/qa_dataset.json", help="Path to QA dataset"),
    output: str = typer.Option("reports/eval_latest.json", help="Output report path"),
) -> None:
    """Run RAG evaluation with Ragas."""
    typer.echo(f"Running evaluation on {dataset} → {output}")
    # TODO: call eval script


if __name__ == "__main__":
    app()
