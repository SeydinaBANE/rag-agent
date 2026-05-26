"""Unit tests for OCR image preprocessor."""

import io

from PIL import Image

from rag_agent.services.ocr.preprocessor import (
    PreprocessingResult,
    detect_document_type_from_text,
    preprocess,
)


def _make_image(width: int = 800, height: int = 1000, color: int = 255) -> bytes:
    img = Image.new("RGB", (width, height), color=(color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_returns_result() -> None:
    img_bytes = _make_image()
    result = preprocess(img_bytes)
    assert isinstance(result, PreprocessingResult)
    assert result.image is not None
    assert result.final_size[0] > 0


def test_preprocess_upscales_small_image() -> None:
    # 200px wide — should be upscaled
    img_bytes = _make_image(width=200, height=300)
    result = preprocess(img_bytes)
    assert result.final_size[0] > 200
    assert len(result.warnings) > 0
    assert "upscaled" in result.warnings[0].lower()


def test_preprocess_no_upscale_large_image() -> None:
    # 1500px wide — should not be upscaled
    img_bytes = _make_image(width=1500, height=2000)
    result = preprocess(img_bytes)
    assert len([w for w in result.warnings if "upscaled" in w.lower()]) == 0


def test_preprocess_to_bytes() -> None:
    img_bytes = _make_image()
    result = preprocess(img_bytes)
    output = result.to_bytes("PNG")
    assert isinstance(output, bytes)
    assert len(output) > 0
    # Verify it's a valid PNG
    img = Image.open(io.BytesIO(output))
    assert img.format == "PNG"


def test_detect_document_type_invoice() -> None:
    text = "INVOICE\nAmount due: $1,200\nPayment terms: Net 30\nInvoice number: INV-001\nbill to: client"
    assert detect_document_type_from_text(text) == "invoice"


def test_detect_document_type_receipt() -> None:
    text = "RECEIPT\nThank you for your purchase\nTotal paid: $45.00\nCashier: Marie"
    assert detect_document_type_from_text(text) == "receipt"


def test_detect_document_type_contract() -> None:
    text = "SERVICE AGREEMENT\nThis contract is made between the parties.\nTerms and conditions apply.\nWhereas the parties agree..."
    assert detect_document_type_from_text(text) == "contract"


def test_detect_document_type_unknown() -> None:
    text = "Random text with no clear document type keywords here."
    assert detect_document_type_from_text(text) == "unknown"


def test_preprocess_skew_angle_stored() -> None:
    img_bytes = _make_image()
    result = preprocess(img_bytes, deskew=True)
    # Angle should be in [-10, 10]
    assert -10.0 <= result.deskew_angle <= 10.0


def test_preprocess_no_deskew() -> None:
    img_bytes = _make_image()
    result = preprocess(img_bytes, deskew=False)
    assert result.deskew_angle == 0.0
