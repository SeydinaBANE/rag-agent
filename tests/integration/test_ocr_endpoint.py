"""Integration tests for OCR endpoints."""

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from rag_agent.api.main import app
from rag_agent.services.ocr.schemas import DocumentType, ExtractionResult, FieldValue, InvoiceSchema

HEADERS = {"X-API-Key": "test-key"}

MOCK_RESULT = ExtractionResult(
    doc_type=DocumentType.INVOICE,
    overall_confidence=0.88,
    raw_text="Invoice INV-001\nTotal: EUR 1500.00",
    structured=InvoiceSchema(
        invoice_number=FieldValue(value="INV-001", confidence=0.95),
        total_amount=FieldValue(value=1500.0, confidence=0.92),
        currency=FieldValue(value="EUR", confidence=0.99),
    ),
    warnings=[],
)


def _make_png_bytes() -> bytes:
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@patch("rag_agent.api.v1.ocr.run_ocr_pipeline", new_callable=AsyncMock, return_value=MOCK_RESULT)
async def test_extract_invoice(mock_pipeline: AsyncMock, client: AsyncClient) -> None:
    png = _make_png_bytes()
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("invoice.png", png, "image/png")},
        data={"doc_type": "invoice"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["doc_type"] == "invoice"
    assert data["overall_confidence"] == 0.88
    assert data["structured"]["invoice_number"]["value"] == "INV-001"


@patch("rag_agent.api.v1.ocr.run_ocr_pipeline", new_callable=AsyncMock, return_value=MOCK_RESULT)
async def test_extract_auto_detect_type(mock_pipeline: AsyncMock, client: AsyncClient) -> None:
    png = _make_png_bytes()
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("doc.png", png, "image/png")},
        headers=HEADERS,
    )
    assert response.status_code == 200
    # doc_type not forced — pipeline auto-detects
    called_kwargs = mock_pipeline.call_args.kwargs
    assert called_kwargs.get("doc_type") is None


async def test_extract_unsupported_mime(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("file.exe", b"binary", "application/octet-stream")},
        headers=HEADERS,
    )
    assert response.status_code == 415


async def test_extract_invalid_doc_type(client: AsyncClient) -> None:
    png = _make_png_bytes()
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("doc.png", png, "image/png")},
        data={"doc_type": "not_a_valid_type"},
        headers=HEADERS,
    )
    assert response.status_code == 422


async def test_extract_missing_api_key(client: AsyncClient) -> None:
    png = _make_png_bytes()
    response = await client.post(
        "/api/v1/ocr/extract",
        files={"file": ("doc.png", png, "image/png")},
    )
    assert response.status_code == 401


async def test_list_schemas(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ocr/schemas", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    schema_types = [s["type"] for s in data["schemas"]]
    assert "invoice" in schema_types
    assert "receipt" in schema_types
    assert "contract" in schema_types
