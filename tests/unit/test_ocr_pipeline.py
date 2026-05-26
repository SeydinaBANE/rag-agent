"""Tests for ocr/pipeline — full OCR orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_agent.services.ocr.pipeline import run_ocr_batch, run_ocr_pipeline
from rag_agent.services.ocr.schemas import DocumentType, ExtractionResult


def _make_prep_result(angle: float = 0.0, warnings: list | None = None) -> MagicMock:
    prep = MagicMock()
    prep.to_bytes = MagicMock(return_value=b"processed_image")
    prep.warnings = warnings or []
    prep.deskew_angle = angle
    prep.original_size = (100, 200)
    return prep


def _make_extraction_result(doc_type: DocumentType = DocumentType.UNKNOWN) -> ExtractionResult:
    return ExtractionResult(
        doc_type=doc_type,
        overall_confidence=0.85,
        raw_text="extracted text",
        structured=None,
        metadata={},
        warnings=[],
    )


@pytest.mark.asyncio
async def test_run_ocr_pipeline_basic():
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="extracted text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"image_bytes")

    assert isinstance(result, ExtractionResult)
    assert result.raw_text == "extracted text"
    assert result.overall_confidence == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_run_ocr_pipeline_preprocessing_failure():
    """When preprocessing fails, pipeline uses original bytes and continues."""
    with (
        patch(
            "rag_agent.services.ocr.pipeline.preprocess",
            side_effect=RuntimeError("preprocess error"),
        ),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text from original"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"raw_image")

    assert result.raw_text == "text from original"
    assert any("Preprocessing failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_run_ocr_pipeline_no_tesseract():
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"image", use_tesseract=False, use_vision=True)

    assert result.raw_text == ""


@pytest.mark.asyncio
async def test_run_ocr_pipeline_no_vision():
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="tesseract text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
    ):
        result = await run_ocr_pipeline(b"image", use_vision=False)

    assert result.raw_text == "tesseract text"
    assert result.overall_confidence == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_run_ocr_pipeline_with_explicit_doc_type():
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text"),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"image", doc_type="invoice")

    assert result.doc_type == DocumentType.INVOICE


@pytest.mark.asyncio
async def test_run_ocr_pipeline_large_skew_warning():
    prep = _make_prep_result(angle=10.0)

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"image")

    assert any("skew" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_run_ocr_pipeline_low_confidence_warning():
    """Low confidence is reached when vision returns data but build_schema fails."""
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        result = await run_ocr_pipeline(b"image")

    assert any("Low confidence" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_run_ocr_pipeline_vision_failure_warning():
    prep = _make_prep_result()

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision",
            new=AsyncMock(side_effect=RuntimeError("vision API error")),
        ),
    ):
        result = await run_ocr_pipeline(b"image")

    assert any("Vision extraction failed" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_run_ocr_batch():
    prep = _make_prep_result()
    docs = [(b"img1", "file1.jpg"), (b"img2", "file2.png")]

    with (
        patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
        patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="text"),
        patch(
            "rag_agent.services.ocr.pipeline.detect_document_type_from_text", return_value="unknown"
        ),
        patch(
            "rag_agent.services.ocr.pipeline.extract_with_vision", new=AsyncMock(return_value={})
        ),
    ):
        results = await run_ocr_batch(docs)

    assert len(results) == 2
    assert results[0]["filename"] == "file1.jpg"
    assert results[0]["error"] is None


@pytest.mark.asyncio
async def test_run_ocr_batch_error_per_file():
    docs = [(b"img1", "good.jpg"), (b"img2", "bad.jpg")]
    call_count = 0

    async def sometimes_fail(image_bytes, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("bad image")
        prep = _make_prep_result()
        with (
            patch("rag_agent.services.ocr.pipeline.preprocess", return_value=prep),
            patch("rag_agent.services.ocr.pipeline.run_tesseract", return_value="t"),
            patch(
                "rag_agent.services.ocr.pipeline.detect_document_type_from_text",
                return_value="unknown",
            ),
            patch(
                "rag_agent.services.ocr.pipeline.extract_with_vision",
                new=AsyncMock(return_value={}),
            ),
        ):
            return await run_ocr_pipeline(image_bytes)

    with patch("rag_agent.services.ocr.pipeline.run_ocr_pipeline", side_effect=sometimes_fail):
        results = await run_ocr_batch(docs)

    assert results[0]["error"] is None
    assert "bad image" in results[1]["error"]
