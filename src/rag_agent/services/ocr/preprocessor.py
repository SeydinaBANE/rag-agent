"""Image preprocessing: deskew, denoise, enhance, binarize."""

from __future__ import annotations

import io

import structlog
from PIL import Image, ImageEnhance, ImageFilter

log = structlog.get_logger()


class PreprocessingResult:
    __slots__ = ("deskew_angle", "final_size", "image", "original_size", "warnings")

    def __init__(
        self,
        image: Image.Image,
        deskew_angle: float = 0.0,
        original_size: tuple[int, int] = (0, 0),
        final_size: tuple[int, int] = (0, 0),
        warnings: list[str] | None = None,
    ) -> None:
        self.image = image
        self.deskew_angle = deskew_angle
        self.original_size = original_size
        self.final_size = final_size
        self.warnings = warnings or []

    def to_bytes(self, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format=fmt)
        return buf.getvalue()


def preprocess(
    image_bytes: bytes,
    target_dpi: int = 300,
    denoise: bool = True,
    deskew: bool = True,
    enhance_contrast: bool = True,
    binarize: bool = False,
) -> PreprocessingResult:
    """
    Full preprocessing pipeline.
    Returns a PreprocessingResult with the cleaned image and metadata.
    """
    image = _load_image(image_bytes)
    original_size = image.size
    warnings: list[str] = []

    # 1. Convert to RGB (handles RGBA, palette, grayscale)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # 2. Upscale if resolution is too low (< target_dpi heuristic)
    image = _ensure_resolution(image, target_dpi, warnings)

    # 3. Deskew
    angle = 0.0
    if deskew:
        angle = _detect_skew(image)
        if abs(angle) > 0.3:
            image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            log.debug("deskew_applied", angle=round(angle, 2))

    # 4. Denoise
    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    # 5. Enhance contrast
    if enhance_contrast:
        image = ImageEnhance.Contrast(image).enhance(1.4)
        image = ImageEnhance.Sharpness(image).enhance(1.2)

    # 6. Binarize (for Tesseract — optional)
    if binarize:
        gray = image.convert("L")
        image = gray.point(lambda x: 0 if x < 140 else 255, "1").convert("L")

    log.info(
        "preprocessing_done",
        original_size=original_size,
        final_size=image.size,
        deskew_angle=round(angle, 2),
    )

    return PreprocessingResult(
        image=image,
        deskew_angle=angle,
        original_size=original_size,
        final_size=image.size,
        warnings=warnings,
    )


def _load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _ensure_resolution(image: Image.Image, target_dpi: int, warnings: list[str]) -> Image.Image:
    """Upscale if image is smaller than 1000px wide (heuristic for low-DPI scans)."""
    min_width = int(target_dpi * 3.5)  # ~8.5" at target DPI
    w, h = image.size
    if w < min_width:
        scale = min_width / w
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        warnings.append(f"Image upscaled {scale:.1f}x (original: {w}x{h})")
        log.debug("image_upscaled", scale=round(scale, 2), new_size=new_size)
    return image


def _detect_skew(image: Image.Image, n_angles: int = 180) -> float:
    """
    Detect document skew using horizontal projection profile.
    Tests angles in [-10°, +10°] and returns the angle with
    the highest variance (most aligned horizontal lines).
    """
    gray = image.convert("L")
    # Downscale for speed
    small = gray.resize((gray.width // 4, gray.height // 4), Image.Resampling.LANCZOS)

    best_angle = 0.0
    best_variance = -1.0

    for deg_tenth in range(-100, 101, 5):  # -10° to +10° in 0.5° steps
        angle = deg_tenth / 10.0
        rotated = small.rotate(angle, expand=False, fillcolor=255)

        # Horizontal projection: sum each row
        import numpy as np

        arr = np.array(rotated)
        row_sums = arr.sum(axis=1).tolist()

        mean = sum(row_sums) / len(row_sums)
        variance = sum((s - mean) ** 2 for s in row_sums) / len(row_sums)

        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    return best_angle


def detect_document_type_from_text(text: str) -> str:
    """Heuristic document type detection from raw OCR text."""
    text_lower = text.lower()
    invoice_keywords = {"invoice", "facture", "bill", "amount due", "payment", "total"}
    receipt_keywords = {"receipt", "reçu", "thank you for", "total paid", "cashier"}
    contract_keywords = {
        "agreement",
        "contrat",
        "contract",
        "parties",
        "whereas",
        "terms and conditions",
    }

    scores = {
        "invoice": sum(1 for k in invoice_keywords if k in text_lower),
        "receipt": sum(1 for k in receipt_keywords if k in text_lower),
        "contract": sum(1 for k in contract_keywords if k in text_lower),
    }
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 2 else "unknown"
