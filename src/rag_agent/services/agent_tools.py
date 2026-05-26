"""Tools disponibles pour l'agent multi-étapes."""

from __future__ import annotations

import json
import re

import httpx
import structlog

from rag_agent.core.config import settings
from rag_agent.services import llm_client

log = structlog.get_logger()


class Tool:
    """Simple tool descriptor — name, description, callable."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    async def run(self, input: str) -> str:
        raise NotImplementedError


# ── Web Search (DuckDuckGo via API) ──────────────────────────────────────────


class WebSearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="web_search",
            description=(
                "Search the web for recent information. "
                "Input: a search query string. "
                "Returns a list of results with title, URL, and snippet."
            ),
        )

    async def run(self, input: str) -> str:
        try:
            from duckduckgo_search import AsyncDDGS  # type: ignore[import,attr-defined]

            async with AsyncDDGS() as ddgs:
                results = await ddgs.atext(input, max_results=5)
            formatted = [f"[{r['title']}]({r['href']})\n{r['body']}" for r in results]
            return "\n\n---\n\n".join(formatted) if formatted else "No results found."
        except ImportError:
            return await self._fallback_search(input)
        except Exception as exc:
            log.warning("web_search_error", error=str(exc))
            return f"Web search failed: {exc}"

    async def _fallback_search(self, query: str) -> str:
        """Fallback: ask LLM for general knowledge when DuckDuckGo unavailable."""
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    f"Provide recent factual information about: {query}\n"
                    "Be concise. Note that your knowledge has a cutoff date."
                ),
            }
        ]
        answer, _ = await llm_client.complete(messages, max_tokens=400)
        return f"[LLM knowledge fallback]\n{answer}"


# ── URL Fetcher ───────────────────────────────────────────────────────────────


class FetchUrlTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="fetch_url",
            description=(
                "Fetch and extract the text content of a URL. "
                "Input: a valid HTTP/HTTPS URL. "
                "Returns the page text (stripped of HTML tags)."
            ),
        )

    async def run(self, input: str) -> str:
        url = input.strip()
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://"
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "rag-agent/0.1 research-bot"})
                r.raise_for_status()
                text = re.sub(r"<[^>]+>", " ", r.text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:3000]  # cap to avoid context explosion
        except Exception as exc:
            return f"Fetch failed: {exc}"


# ── RAG Search (internal vector store) ───────────────────────────────────────


class RAGSearchTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="rag_search",
            description=(
                "Search the internal document knowledge base using semantic similarity. "
                "Input: a natural language question or keyword query. "
                "Returns the most relevant document excerpts."
            ),
        )

    async def run(self, input: str) -> str:
        try:
            from rag_agent.services.retriever import retrieve

            chunks = await retrieve(input)
            if not chunks:
                return "No relevant documents found in the knowledge base."
            parts = [
                f"[Source: {c.get('metadata', {}).get('source', 'unknown')}] "
                f"(score: {float(c.get('score', 0)):.2f})\n{c['text']}"
                for c in chunks[:4]
            ]
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"RAG search failed: {exc}"


# ── SQL Query ─────────────────────────────────────────────────────────────────


class SQLQueryTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="sql_query",
            description=(
                "Execute a read-only SQL query against the internal database. "
                "Input: a valid SELECT SQL statement. "
                "Returns the query results as JSON."
            ),
        )

    async def run(self, input: str) -> str:
        query = input.strip()
        # Security: only allow SELECT
        if not re.match(r"^\s*SELECT\s", query, re.IGNORECASE):
            return "Error: only SELECT statements are allowed."
        # Strip dangerous keywords
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "EXEC"]
        for kw in forbidden:
            if re.search(rf"\b{kw}\b", query, re.IGNORECASE):
                return f"Error: forbidden keyword '{kw}' detected."
        try:
            import asyncpg

            conn = await asyncpg.connect(settings.database_url.replace("+asyncpg", ""))
            try:
                rows = await conn.fetch(query)
                result = [dict(row) for row in rows[:50]]
                return json.dumps(result, default=str)
            finally:
                await conn.close()
        except Exception as exc:
            return f"SQL error: {exc}"


# ── Report Generator ──────────────────────────────────────────────────────────


class GenerateReportTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="generate_report",
            description=(
                "Generate a structured Markdown report from collected information. "
                "Input: JSON string with keys: title, company, sections (list of {heading, content}). "
                "Returns the formatted report text."
            ),
        )

    async def run(self, input: str) -> str:
        try:
            data = json.loads(input)
        except json.JSONDecodeError:
            # Treat raw text as content to format
            data = {
                "title": "Research Report",
                "company": "",
                "sections": [{"heading": "Summary", "content": input}],
            }

        lines = [
            f"# {data.get('title', 'Report')}",
            "",
            f"**Company**: {data.get('company', 'N/A')}",
            f"**Date**: {__import__('datetime').date.today().isoformat()}",
            "",
        ]
        for section in data.get("sections", []):
            lines.append(f"## {section.get('heading', 'Section')}")
            lines.append("")
            lines.append(section.get("content", ""))
            lines.append("")

        report = "\n".join(lines)
        log.info("report_generated", chars=len(report))
        return report


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS: list[Tool] = [
    WebSearchTool(),
    FetchUrlTool(),
    RAGSearchTool(),
    SQLQueryTool(),
    GenerateReportTool(),
]

TOOL_MAP: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def get_tools_description() -> str:
    return "\n".join(f"- **{t.name}**: {t.description}" for t in ALL_TOOLS)
