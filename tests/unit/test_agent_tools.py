"""Unit tests for agent tools — no external calls."""

from rag_agent.services.agent_tools import (
    TOOL_MAP,
    FetchUrlTool,
    GenerateReportTool,
    SQLQueryTool,
    get_tools_description,
)


def test_tool_map_has_all_tools() -> None:
    assert "web_search" in TOOL_MAP
    assert "fetch_url" in TOOL_MAP
    assert "rag_search" in TOOL_MAP
    assert "sql_query" in TOOL_MAP
    assert "generate_report" in TOOL_MAP


def test_tools_description_non_empty() -> None:
    desc = get_tools_description()
    assert "web_search" in desc
    assert "fetch_url" in desc
    assert len(desc) > 100


async def test_sql_tool_rejects_non_select() -> None:
    tool = SQLQueryTool()
    result = await tool.run("DROP TABLE users")
    assert "only SELECT" in result


async def test_sql_tool_rejects_forbidden_keywords() -> None:
    tool = SQLQueryTool()
    result = await tool.run("SELECT * FROM users; DELETE FROM users")
    assert "forbidden keyword" in result.lower() or "only SELECT" in result


async def test_fetch_url_rejects_non_http() -> None:
    tool = FetchUrlTool()
    result = await tool.run("file:///etc/passwd")
    assert "Error" in result


async def test_generate_report_basic() -> None:
    tool = GenerateReportTool()
    import json

    payload = json.dumps(
        {
            "title": "Test Report",
            "company": "ACME Corp",
            "sections": [{"heading": "Summary", "content": "ACME is a leading company."}],
        }
    )
    result = await tool.run(payload)
    assert "# Test Report" in result
    assert "ACME Corp" in result
    assert "## Summary" in result


async def test_generate_report_fallback_raw_text() -> None:
    tool = GenerateReportTool()
    result = await tool.run("This is a plain text report without JSON structure.")
    assert "Summary" in result or "Report" in result
