import pytest

from rag_agent.services.document_loader import load_bytes


def test_plain_text() -> None:
    content = b"Hello, this is plain text."
    result = load_bytes(content, "doc.txt")
    assert "Hello" in result


def test_html_strips_tags() -> None:
    content = b"<html><body><p>Hello <b>world</b></p><script>alert(1)</script></body></html>"
    result = load_bytes(content, "page.html")
    assert "Hello" in result
    assert "alert" not in result
    assert "<" not in result


def test_unknown_extension_returns_text() -> None:
    content = "plain content".encode()
    result = load_bytes(content, "file.xyz")
    assert "plain content" in result
