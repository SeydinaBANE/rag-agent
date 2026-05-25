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


@app.command("create-key")
def create_key(
    name: str = typer.Argument(..., help="Human-readable name for this API key"),
) -> None:
    """Bootstrap: create the first API key directly via the database (no HTTP auth needed)."""
    import asyncio
    import secrets
    import uuid

    from rag_agent.api.v1.deps import hash_key
    from rag_agent.models.database import ApiKey, async_session

    async def _insert() -> str:
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
