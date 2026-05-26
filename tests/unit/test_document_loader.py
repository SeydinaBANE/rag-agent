from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_agent.services.document_loader import load_bytes, load_file


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


def test_html_style_tags_stripped() -> None:
    content = b"<html><head><style>.x{color:red}</style></head><body>content</body></html>"
    result = load_bytes(content, "page.htm")
    assert "content" in result
    assert "color" not in result


def test_unknown_extension_returns_text() -> None:
    content = b"plain content"
    result = load_bytes(content, "file.xyz")
    assert "plain content" in result


def test_load_bytes_pdf_fallback_on_error() -> None:
    with patch("rag_agent.services.document_loader._load_pdf_bytes", return_value="extracted pdf"):
        result = load_bytes(b"%PDF-1.4 fake", "doc.pdf")
    assert result == "extracted pdf"


def test_load_bytes_docx() -> None:
    with patch("rag_agent.services.document_loader._load_docx_bytes", return_value="docx text"):
        result = load_bytes(b"fake docx", "doc.docx")
    assert result == "docx text"


def test_load_bytes_doc_extension() -> None:
    with patch("rag_agent.services.document_loader._load_docx_bytes", return_value="doc text"):
        result = load_bytes(b"fake doc", "doc.doc")
    assert result == "doc text"


def test_load_docx_bytes_success() -> None:
    mock_doc = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Hello World"
    mock_para2 = MagicMock()
    mock_para2.text = ""  # empty — should be skipped
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with patch("rag_agent.services.document_loader.docx", create=True) as mock_docx_module:
        mock_docx_module.Document.return_value = mock_doc
        from rag_agent.services import document_loader

        with patch.object(document_loader, "_load_docx_bytes") as patched:
            patched.return_value = "Hello World"
            result = load_bytes(b"fake docx", "test.docx")

    assert result == "Hello World"


def test_load_docx_bytes_error_returns_empty() -> None:
    from rag_agent.services.document_loader import _load_docx_bytes

    with patch.dict("sys.modules", {"docx": None}):
        result = _load_docx_bytes(b"bad bytes")
    assert result == ""


def test_load_pdf_bytes_short_text_triggers_ocr() -> None:
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "short"
    mock_reader.pages = [mock_page]

    with (
        patch("rag_agent.services.document_loader.pypdf", create=True) as mock_pypdf,
        patch("rag_agent.services.document_loader._ocr_pdf", return_value="ocr result"),
    ):
        mock_pypdf.PdfReader.return_value = mock_reader
        from rag_agent.services.document_loader import _load_pdf_bytes

        result = _load_pdf_bytes(b"fake pdf bytes")

    assert result == "ocr result"


def test_load_pdf_bytes_error_returns_ocr() -> None:
    from rag_agent.services.document_loader import _load_pdf_bytes

    with (
        patch("rag_agent.services.document_loader.pypdf", create=True) as mock_pypdf,
        patch("rag_agent.services.document_loader._ocr_pdf", return_value="fallback"),
    ):
        mock_pypdf.PdfReader.side_effect = Exception("corrupt pdf")
        result = _load_pdf_bytes(b"corrupt pdf")

    assert result == "fallback"


def test_load_file_txt(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello from file")
    result = load_file(f)
    assert "hello from file" in result


def test_load_file_html(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_bytes(b"<html><body><p>page content</p></body></html>")
    result = load_file(f)
    assert "page content" in result


def test_load_file_pdf(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    with patch("rag_agent.services.document_loader._load_pdf_bytes", return_value="pdf text"):
        result = load_file(f)
    assert result == "pdf text"


def test_load_file_docx(tmp_path: Path) -> None:
    f = tmp_path / "doc.docx"
    f.write_bytes(b"fake docx")
    with patch("rag_agent.services.document_loader._load_docx_bytes", return_value="docx text"):
        result = load_file(f)
    assert result == "docx text"
