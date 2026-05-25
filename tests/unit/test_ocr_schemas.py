"""Unit tests for OCR schemas and extractor helpers."""

import pytest

from rag_agent.services.ocr.extractor import (
    _build_field,
    _parse_json_response,
    build_schema,
    compute_overall_confidence,
)
from rag_agent.services.ocr.schemas import (
    DocumentType,
    ExtractionResult,
    FieldValue,
    InvoiceSchema,
)


def test_field_value_from_dict() -> None:
    fv = _build_field({"value": "INV-001", "confidence": 0.92})
    assert fv is not None
    assert fv.value == "INV-001"
    assert fv.confidence == 0.92
    assert fv.source == "llm"


def test_field_value_from_raw_string() -> None:
    fv = _build_field("plain string")
    assert fv is not None
    assert fv.value == "plain string"
    assert fv.confidence == 0.5


def test_field_value_from_none() -> None:
    assert _build_field(None) is None


def test_parse_json_clean() -> None:
    raw = '{"invoice_number": {"value": "INV-001", "confidence": 0.9}}'
    result = _parse_json_response(raw)
    assert result["invoice_number"]["value"] == "INV-001"


def test_parse_json_with_markdown() -> None:
    raw = '```json\n{"key": "value"}\n```'
    result = _parse_json_response(raw)
    assert result["key"] == "value"


def test_parse_json_embedded_in_text() -> None:
    raw = 'Here is the result: {"total": {"value": 42.5, "confidence": 0.9}} End.'
    result = _parse_json_response(raw)
    assert result["total"]["value"] == 42.5


def test_parse_json_invalid_returns_empty() -> None:
    result = _parse_json_response("This is not JSON at all")
    assert result == {}


def test_build_invoice_schema() -> None:
    data = {
        "invoice_number": {"value": "INV-001", "confidence": 0.95},
        "vendor_name": {"value": "ACME Corp", "confidence": 0.90},
        "total_amount": {"value": 1500.0, "confidence": 0.92},
        "currency": {"value": "EUR", "confidence": 0.99},
        "items": [{"description": "Service", "qty": 1, "unit_price": 1500.0}],
    }
    schema = build_schema(DocumentType.INVOICE, data)
    assert isinstance(schema, InvoiceSchema)
    assert schema.invoice_number is not None
    assert schema.invoice_number.value == "INV-001"
    assert schema.total_amount is not None
    assert schema.total_amount.value == 1500.0
    assert len(schema.items) == 1


def test_compute_overall_confidence() -> None:
    schema = InvoiceSchema(
        invoice_number=FieldValue(value="INV-001", confidence=0.9),
        vendor_name=FieldValue(value="ACME", confidence=0.8),
        total_amount=FieldValue(value=100.0, confidence=0.95),
    )
    conf = compute_overall_confidence(schema)
    assert 0.8 < conf < 0.95
    assert isinstance(conf, float)


def test_extraction_result_field_confidences() -> None:
    schema = InvoiceSchema(
        invoice_number=FieldValue(value="INV-001", confidence=0.9),
        total_amount=FieldValue(value=100.0, confidence=0.85),
    )
    result = ExtractionResult(
        doc_type=DocumentType.INVOICE,
        overall_confidence=0.875,
        structured=schema,
    )
    confs = result.field_confidences()
    assert "invoice_number" in confs
    assert confs["invoice_number"] == 0.9
    assert "total_amount" in confs
    assert confs["total_amount"] == 0.85


def test_extraction_result_no_structured() -> None:
    result = ExtractionResult(
        doc_type=DocumentType.UNKNOWN,
        overall_confidence=0.0,
        structured=None,
    )
    assert result.field_confidences() == {}
