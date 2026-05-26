"""OCR endpoints: single image extraction and batch processing."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from rag_agent.api.v1.deps import require_api_key
from rag_agent.services.ocr.pipeline import run_ocr_pipeline
from rag_agent.services.ocr.schemas import DocumentType, ExtractionResult

log = structlog.get_logger()
router = APIRouter(prefix="/ocr", tags=["ocr"])

ALLOWED_MIME = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/webp",
    "application/pdf",
}
MAX_SIZE_MB = 20

SCHEMA_DESCRIPTIONS = {
    "invoice": "Facture — numéro, date, vendeur, client, montants, articles",
    "receipt": "Reçu / ticket de caisse — commerçant, total, articles",
    "contract": "Contrat — parties, dates, type, clauses clés",
    "form": "Formulaire générique — paires champ/valeur",
    "unknown": "Type auto-détecté",
}


@router.get("/schemas")
async def list_schemas() -> dict[str, object]:
    """List available document schemas."""
    return {
        "schemas": [
            {"type": t.value, "description": SCHEMA_DESCRIPTIONS.get(t.value, "")}
            for t in DocumentType
        ]
    }


@router.post("/extract", response_model=ExtractionResult)
async def extract_document(
    file: UploadFile = File(..., description="Image or PDF to extract"),
    doc_type: str | None = Form(
        None, description="Force document type (invoice|receipt|contract|form)"
    ),
    lang: str = Form("fra+eng", description="Tesseract language(s)"),
    use_vision: bool = Form(True, description="Use LLM vision for structured extraction"),
    _: str = Depends(require_api_key),
) -> ExtractionResult:
    """
    Full OCR pipeline on a single document:
    preprocess → Tesseract → LLM vision → structured JSON + confidence scores.
    """
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type: {file.content_type}. Allowed: {', '.join(ALLOWED_MIME)}",
        )

    content = await file.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_SIZE_MB}MB limit",
        )

    if doc_type and doc_type not in [t.value for t in DocumentType]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown doc_type '{doc_type}'. Use: {[t.value for t in DocumentType]}",
        )

    log.info("ocr_extract_request", filename=file.filename, doc_type=doc_type, size=len(content))

    result = await run_ocr_pipeline(
        content,
        doc_type=doc_type,
        lang=lang,
        use_vision=use_vision,
    )
    return result


@router.post("/extract/url")
async def extract_from_url(
    url: str,
    doc_type: str | None = None,
    _: str = Depends(require_api_key),
) -> ExtractionResult:
    """Download an image from URL and run OCR pipeline."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            content = r.content
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}") from exc

    result = await run_ocr_pipeline(content, doc_type=doc_type)
    return result
