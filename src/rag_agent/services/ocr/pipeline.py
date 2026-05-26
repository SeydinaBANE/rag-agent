"""Full OCR pipeline orchestrator: preprocess → tesseract → vision → validate."""

from __future__ import annotations

import time
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

from rag_agent.services.ocr.extractor import (
    build_schema,
    compute_overall_confidence,
    extract_with_vision,
    run_tesseract,
)
from rag_agent.services.ocr.preprocessor import detect_document_type_from_text, preprocess
from rag_agent.services.ocr.schemas import DocumentType, ExtractionResult

log = structlog.get_logger()

OCR_PROCESSED = Counter("ocr_documents_total", "Documents processed", ["doc_type"])
OCR_CONFIDENCE = Histogram("ocr_overall_confidence", "Overall extraction confidence")
OCR_LATENCY = Histogram("ocr_pipeline_latency_seconds", "Pipeline latency")

CONFIDENCE_THRESHOLD = 0.60  # warn if below this


async def run_ocr_pipeline(
    image_bytes: bytes,
    doc_type: str | None = None,
    lang: str = "fra+eng",
    use_vision: bool = True,
    use_tesseract: bool = True,
) -> ExtractionResult:
    """
    Full pipeline:
    1. Preprocess image (deskew, denoise, enhance)
    2. Run Tesseract for raw text
    3. Detect document type (auto or provided)
    4. Run vision LLM for structured extraction
    5. Validate + compute confidence
    """
    start = time.perf_counter()
    warnings: list[str] = []

    # 1. Preprocess
    try:
        prep = preprocess(image_bytes, denoise=True, deskew=True, enhance_contrast=True)
        clean_bytes = prep.to_bytes()
        warnings.extend(prep.warnings)
        if abs(prep.deskew_angle) > 5.0:
            warnings.append(f"Large skew detected and corrected: {prep.deskew_angle:.1f}°")
    except Exception as exc:
        log.warning("preprocessing_failed", error=str(exc))
        clean_bytes = image_bytes
        warnings.append(f"Preprocessing failed: {exc}")

    # 2. Tesseract OCR
    raw_text = ""
    if use_tesseract:
        raw_text = run_tesseract(clean_bytes, lang=lang)
        if not raw_text:
            warnings.append("Tesseract returned no text — image may be low quality or non-text")

    # 3. Detect document type
    if doc_type:
        detected_type = (
            DocumentType(doc_type)
            if doc_type in DocumentType.__members__.values()
            else DocumentType.UNKNOWN
        )
    else:
        type_hint = detect_document_type_from_text(raw_text)
        try:
            detected_type = DocumentType(type_hint)
        except ValueError:
            detected_type = DocumentType.UNKNOWN

    log.info("ocr_doc_type_detected", doc_type=detected_type, tesseract_chars=len(raw_text))

    # 4. LLM vision extraction
    llm_data: dict[str, Any] = {}
    if use_vision:
        try:
            llm_data = await extract_with_vision(clean_bytes, detected_type, hint_text=raw_text)
        except Exception as exc:
            log.warning("vision_failed", error=str(exc))
            warnings.append(f"Vision extraction failed: {exc}")

    # 5. Build typed schema + compute confidence
    structured = None
    overall_confidence = 0.0

    if llm_data:
        try:
            structured = build_schema(detected_type, llm_data)
            overall_confidence = compute_overall_confidence(structured)
        except Exception as exc:
            log.warning("schema_build_failed", error=str(exc))
            warnings.append(f"Schema validation partial: {exc}")

    if overall_confidence < CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Low confidence extraction ({overall_confidence:.2f} < {CONFIDENCE_THRESHOLD}). "
            "Manual review recommended."
        )

    latency = time.perf_counter() - start
    OCR_PROCESSED.labels(doc_type=detected_type.value).inc()
    OCR_CONFIDENCE.observe(overall_confidence)
    OCR_LATENCY.observe(latency)

    log.info(
        "ocr_pipeline_done",
        doc_type=detected_type,
        confidence=overall_confidence,
        latency_s=round(latency, 3),
        warnings=len(warnings),
    )

    return ExtractionResult(
        doc_type=detected_type,
        overall_confidence=overall_confidence,
        raw_text=raw_text,
        structured=structured,
        metadata={
            "latency_s": round(latency, 3),
            "deskew_angle": prep.deskew_angle if "prep" in dir() else 0.0,
            "original_size": list(prep.original_size) if "prep" in dir() else [],
        },
        warnings=warnings,
    )


async def run_ocr_batch(
    documents: list[tuple[bytes, str]],  # (image_bytes, filename)
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Process multiple documents. Returns list of results with filename."""
    results = []
    for image_bytes, filename in documents:
        try:
            result = await run_ocr_pipeline(image_bytes, doc_type=doc_type)
            results.append({"filename": filename, "result": result.model_dump(), "error": None})
        except Exception as exc:
            log.error("batch_ocr_error", filename=filename, error=str(exc))
            results.append({"filename": filename, "result": None, "error": str(exc)})
    return results
