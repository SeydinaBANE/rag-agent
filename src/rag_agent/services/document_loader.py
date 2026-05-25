"""Load documents from PDF, DOCX, HTML, or plain text. OCR fallback for scanned PDFs."""

import io
from pathlib import Path

import structlog

log = structlog.get_logger()


def load_file(path: str | Path) -> str:
    """Extract plain text from a file. Returns empty string on failure."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _load_docx(path)
    elif suffix in (".html", ".htm"):
        return _load_html(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def load_bytes(content: bytes, filename: str) -> str:
    """Extract plain text from raw bytes (e.g. from upload or MinIO)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_bytes(content)
    elif suffix in (".docx", ".doc"):
        return _load_docx_bytes(content)
    elif suffix in (".html", ".htm"):
        return _load_html_bytes(content)
    return content.decode("utf-8", errors="replace")


# ── PDF ─────────────────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> str:
    return _load_pdf_bytes(path.read_bytes())


def _load_pdf_bytes(content: bytes) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

        if len(text) < 100:
            log.info("pdf_ocr_fallback", reason="text too short, trying OCR")
            text = _ocr_pdf(content)

        return text
    except Exception as exc:
        log.warning("pdf_load_error", error=str(exc))
        return _ocr_pdf(content)


def _ocr_pdf(content: bytes) -> str:
    """Rasterize PDF pages and OCR them with Tesseract."""
    try:
        import pytesseract
        from PIL import Image

        try:
            import fitz  # PyMuPDF — fast rasterizer

            doc = fitz.open(stream=content, filetype="pdf")
            pages_text = []
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages_text.append(pytesseract.image_to_string(img, lang="fra+eng"))
            return "\n".join(pages_text)
        except ImportError:
            log.debug("pymupdf_not_available_falling_back")
            return ""
    except Exception as exc:
        log.warning("ocr_error", error=str(exc))
        return ""


# ── DOCX ─────────────────────────────────────────────────────────────────────

def _load_docx(path: Path) -> str:
    return _load_docx_bytes(path.read_bytes())


def _load_docx_bytes(content: bytes) -> str:
    try:
        import docx

        doc = docx.Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as exc:
        log.warning("docx_load_error", error=str(exc))
        return ""


# ── HTML ─────────────────────────────────────────────────────────────────────

def _load_html(path: Path) -> str:
    return _load_html_bytes(path.read_bytes())


def _load_html_bytes(content: bytes) -> str:
    try:
        from html.parser import HTMLParser

        class _Extractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag: str, attrs: object) -> None:
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag: str) -> None:
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data: str) -> None:
                if not self._skip and data.strip():
                    self._parts.append(data.strip())

        parser = _Extractor()
        parser.feed(content.decode("utf-8", errors="replace"))
        return " ".join(parser._parts)
    except Exception as exc:
        log.warning("html_load_error", error=str(exc))
        return content.decode("utf-8", errors="replace")
