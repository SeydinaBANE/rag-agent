"""
Structured extraction via:
  1. Tesseract OCR → raw text
  2. LLM Vision (OpenRouter) → structured JSON + confidence
  3. Merge & validate via Pydantic
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import structlog

from rag_agent.services import llm_client
from rag_agent.services.ocr.schemas import (
    EXTRACTION_PROMPTS,
    ContractSchema,
    DocumentType,
    ExtractionResult,
    FieldValue,
    FormSchema,
    InvoiceSchema,
    ReceiptSchema,
)

log = structlog.get_logger()

VISION_MODEL = "google/gemini-flash-1.5"  # supports vision via OpenRouter


# ── Tesseract OCR ─────────────────────────────────────────────────────────────

def run_tesseract(image_bytes: bytes, lang: str = "fra+eng") -> str:
    """Extract raw text via Tesseract. Returns empty string on failure."""
    try:
        import pytesseract
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as exc:
        log.warning("tesseract_error", error=str(exc))
        return ""


# ── LLM Vision extraction ─────────────────────────────────────────────────────

async def extract_with_vision(
    image_bytes: bytes,
    doc_type: DocumentType,
    hint_text: str = "",
) -> dict[str, Any]:
    """
    Send image to a vision LLM via OpenRouter. Returns parsed dict.
    Falls back to text-only extraction if image encoding fails.
    """
    prompt = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS[DocumentType.FORM])

    # Encode image as base64
    b64 = base64.b64encode(image_bytes).decode()
    mime = _detect_mime(image_bytes)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"{prompt}\n\n"
                        + (f"Additional text context:\n{hint_text[:1000]}\n\n" if hint_text else "")
                        + "Return ONLY the JSON object, no markdown, no explanation."
                    ),
                },
            ],
        }
    ]

    try:
        raw, _ = await llm_client.complete(
            messages,  # type: ignore[arg-type]
            model=VISION_MODEL,
            temperature=0.0,
            max_tokens=1024,
        )
        return _parse_json_response(raw)
    except Exception as exc:
        log.warning("vision_extraction_error", error=str(exc))
        # Fallback: text-only extraction using OCR text
        return await _extract_text_only(hint_text, doc_type)


async def _extract_text_only(text: str, doc_type: DocumentType) -> dict[str, Any]:
    """Text-only fallback when vision is unavailable."""
    prompt = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS[DocumentType.FORM])
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": f"Document text:\n{text}\n\n{prompt}\n\nReturn ONLY valid JSON.",
        }
    ]
    raw, _ = await llm_client.complete(messages, temperature=0.0, max_tokens=1024)
    return _parse_json_response(raw)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract JSON from LLM response, stripping markdown fences."""
    # Strip ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    else:
        # Find first { ... }
        match = re.search(r"\{[\s\S]+\}", raw)
        if match:
            raw = match.group()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("json_parse_failed", raw_preview=raw[:200])
        return {}


def _detect_mime(data: bytes) -> str:
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:2] in (b"\xff\xd8", b"\xff\xe0", b"\xff\xe1"):
        return "image/jpeg"
    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "image/png"


# ── Schema mapping ────────────────────────────────────────────────────────────

def _build_field(raw: Any) -> FieldValue | None:
    if raw is None:
        return None
    if isinstance(raw, dict) and "value" in raw:
        return FieldValue(
            value=raw["value"],
            confidence=float(raw.get("confidence", 0.5)),
            source="llm",
        )
    return FieldValue(value=raw, confidence=0.5, source="llm")


def build_schema(doc_type: DocumentType, data: dict[str, Any]) -> Any:
    """Map raw LLM dict to the typed Pydantic schema."""
    if doc_type == DocumentType.INVOICE:
        return InvoiceSchema(
            invoice_number=_build_field(data.get("invoice_number")),
            date=_build_field(data.get("date")),
            due_date=_build_field(data.get("due_date")),
            vendor_name=_build_field(data.get("vendor_name")),
            vendor_address=_build_field(data.get("vendor_address")),
            client_name=_build_field(data.get("client_name")),
            client_address=_build_field(data.get("client_address")),
            subtotal=_build_field(data.get("subtotal")),
            tax_rate=_build_field(data.get("tax_rate")),
            tax_amount=_build_field(data.get("tax_amount")),
            total_amount=_build_field(data.get("total_amount")),
            currency=_build_field(data.get("currency")),
            items=data.get("items", []),
        )
    elif doc_type == DocumentType.RECEIPT:
        return ReceiptSchema(
            merchant=_build_field(data.get("merchant")),
            date=_build_field(data.get("date")),
            total=_build_field(data.get("total")),
            tax=_build_field(data.get("tax")),
            payment_method=_build_field(data.get("payment_method")),
            items=data.get("items", []),
        )
    elif doc_type == DocumentType.CONTRACT:
        parties_raw = data.get("parties", [])
        parties = [_build_field(p) for p in parties_raw if _build_field(p) is not None]
        return ContractSchema(
            parties=parties,  # type: ignore[arg-type]
            effective_date=_build_field(data.get("effective_date")),
            expiry_date=_build_field(data.get("expiry_date")),
            contract_type=_build_field(data.get("contract_type")),
            jurisdiction=_build_field(data.get("jurisdiction")),
            key_clauses=data.get("key_clauses", []),
        )
    else:
        fields_raw = data.get("fields", data)
        fields = {}
        for k, v in fields_raw.items():
            f = _build_field(v)
            if f:
                fields[k] = f
        return FormSchema(fields=fields)


def compute_overall_confidence(structured: Any) -> float:
    """Weighted average of all field confidences."""
    confidences: list[float] = []
    for name in structured.__class__.model_fields:
        val = getattr(structured, name, None)
        if isinstance(val, FieldValue):
            confidences.append(val.confidence)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, FieldValue):
                    confidences.append(item.confidence)
    if not confidences:
        return 0.5
    return round(sum(confidences) / len(confidences), 3)
