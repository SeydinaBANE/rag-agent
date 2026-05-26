"""Pydantic schemas for structured document extraction with per-field confidence."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FORM = "form"
    UNKNOWN = "unknown"


class FieldValue(BaseModel):
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "llm"  # "llm" | "tesseract" | "merged"


class InvoiceSchema(BaseModel):
    invoice_number: FieldValue | None = None
    date: FieldValue | None = None
    due_date: FieldValue | None = None
    vendor_name: FieldValue | None = None
    vendor_address: FieldValue | None = None
    client_name: FieldValue | None = None
    client_address: FieldValue | None = None
    subtotal: FieldValue | None = None
    tax_rate: FieldValue | None = None
    tax_amount: FieldValue | None = None
    total_amount: FieldValue | None = None
    currency: FieldValue | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class ReceiptSchema(BaseModel):
    merchant: FieldValue | None = None
    date: FieldValue | None = None
    total: FieldValue | None = None
    tax: FieldValue | None = None
    payment_method: FieldValue | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)


class ContractSchema(BaseModel):
    parties: list[FieldValue] = Field(default_factory=list)
    effective_date: FieldValue | None = None
    expiry_date: FieldValue | None = None
    contract_type: FieldValue | None = None
    jurisdiction: FieldValue | None = None
    key_clauses: list[str] = Field(default_factory=list)


class FormSchema(BaseModel):
    fields: dict[str, FieldValue] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    doc_type: DocumentType
    overall_confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""
    structured: InvoiceSchema | ReceiptSchema | ContractSchema | FormSchema | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def field_confidences(self) -> dict[str, float]:
        if self.structured is None:
            return {}
        result: dict[str, float] = {}
        for name, _val in self.structured.__class__.model_fields.items():
            field_val = getattr(self.structured, name, None)
            if isinstance(field_val, FieldValue):
                result[name] = field_val.confidence
        return result


# Prompt templates per document type
EXTRACTION_PROMPTS: dict[DocumentType, str] = {
    DocumentType.INVOICE: """Extract all invoice fields from this document.
Return a JSON object with these fields (include a "confidence" 0.0-1.0 for each):
{
  "invoice_number": {"value": "...", "confidence": 0.9},
  "date": {"value": "YYYY-MM-DD", "confidence": 0.95},
  "due_date": {"value": "YYYY-MM-DD or null", "confidence": 0.8},
  "vendor_name": {"value": "...", "confidence": 0.95},
  "vendor_address": {"value": "...", "confidence": 0.85},
  "client_name": {"value": "...", "confidence": 0.9},
  "client_address": {"value": "...", "confidence": 0.8},
  "subtotal": {"value": 123.45, "confidence": 0.9},
  "tax_rate": {"value": 0.20, "confidence": 0.85},
  "tax_amount": {"value": 24.69, "confidence": 0.9},
  "total_amount": {"value": 148.14, "confidence": 0.95},
  "currency": {"value": "EUR", "confidence": 0.9},
  "items": [{"description": "...", "qty": 1, "unit_price": 123.45, "total": 123.45}]
}
Set confidence lower when a field is unclear or inferred.""",
    DocumentType.RECEIPT: """Extract all receipt fields from this document.
Return a JSON object with "confidence" 0.0-1.0 per field:
{
  "merchant": {"value": "...", "confidence": 0.9},
  "date": {"value": "YYYY-MM-DD", "confidence": 0.9},
  "total": {"value": 42.50, "confidence": 0.95},
  "tax": {"value": 7.08, "confidence": 0.85},
  "payment_method": {"value": "card|cash|...", "confidence": 0.8},
  "items": [{"name": "...", "qty": 1, "price": 42.50}]
}""",
    DocumentType.CONTRACT: """Extract key contract information from this document.
Return a JSON object with "confidence" 0.0-1.0 per field:
{
  "parties": [{"value": "Party Name", "confidence": 0.9}],
  "effective_date": {"value": "YYYY-MM-DD", "confidence": 0.85},
  "expiry_date": {"value": "YYYY-MM-DD or null", "confidence": 0.8},
  "contract_type": {"value": "NDA|Service Agreement|...", "confidence": 0.85},
  "jurisdiction": {"value": "...", "confidence": 0.8},
  "key_clauses": ["Payment terms: ...", "Termination: ..."]
}""",
    DocumentType.FORM: """Extract all form fields and their values from this document.
Return a JSON object:
{
  "fields": {
    "field_name": {"value": "...", "confidence": 0.9},
    ...
  }
}""",
}
