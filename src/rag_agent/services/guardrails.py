"""Guardrails: PII detection (Presidio), hallucination scoring, toxicity filter."""

from __future__ import annotations

import structlog
from prometheus_client import Counter

from rag_agent.core.config import settings
from rag_agent.core.exceptions import GuardrailError

log = structlog.get_logger()

GUARDRAIL_BLOCKED = Counter("guardrail_blocked_total", "Blocked requests", ["reason"])

# ── PII Detection ────────────────────────────────────────────────────────────

_analyzer: object | None = None
_anonymizer: object | None = None

_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
    "MEDICAL_LICENSE",
]


def _get_analyzer() -> object:
    global _analyzer
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]

        _analyzer = AnalyzerEngine()
    return _analyzer


def _get_anonymizer() -> object:
    global _anonymizer
    if _anonymizer is None:
        from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]

        _anonymizer = AnonymizerEngine()
    return _anonymizer


def detect_pii(text: str, language: str = "en") -> list[dict[str, object]]:
    """Return list of detected PII entities."""
    if not settings.guardrails_pii_enabled:
        return []
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]  # noqa: F401

        analyzer = _get_analyzer()
        results = analyzer.analyze(text=text, entities=_PII_ENTITIES, language=language)  # type: ignore[union-attr]
        return [
            {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
            for r in results
        ]
    except ImportError:
        log.warning(
            "presidio_not_installed", hint="pip install presidio-analyzer presidio-anonymizer"
        )
        return []


def anonymize_pii(text: str, language: str = "en") -> str:
    """Replace PII with placeholders like <PERSON>, <EMAIL_ADDRESS>."""
    if not settings.guardrails_pii_enabled:
        return text
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]  # noqa: F401
        from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]  # noqa: F401

        analyzer = _get_analyzer()
        anonymizer = _get_anonymizer()
        results = analyzer.analyze(text=text, entities=_PII_ENTITIES, language=language)  # type: ignore[union-attr]
        if not results:
            return text
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore[union-attr]
        return anonymized.text
    except ImportError:
        return text


# ── Toxicity Filter ──────────────────────────────────────────────────────────

_TOXIC_KEYWORDS = {
    "bomb",
    "hack",
    "exploit",
    "malware",
    "ransomware",
    "ddos",
    "kill",
    "murder",
    "suicide",
    "illegal drugs",
}


def check_toxicity(text: str) -> bool:
    """Return True if input is likely toxic. Simple keyword gate — extend with model."""
    if not settings.guardrails_toxicity_enabled:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _TOXIC_KEYWORDS)


# ── Hallucination Score ───────────────────────────────────────────────────────


def score_hallucination_nli(answer: str, context_chunks: list[str]) -> float:
    """
    NLI-based hallucination scoring using a local cross-encoder.
    Returns a float [0, 1] where 1 = fully grounded, 0 = likely hallucinated.
    Falls back to 0.5 if sentence-transformers is not installed.
    """
    if not context_chunks:
        return 0.0
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        context = " ".join(context_chunks[:3])
        # NLI: [contradiction, neutral, entailment]
        scores = model.predict([[context, answer]])[0]
        entailment_score = float(scores[2])
        return entailment_score
    except ImportError:
        return 0.5


# ── Main guard function ───────────────────────────────────────────────────────


def guard_input(text: str) -> str:
    """
    Run all input guardrails. Returns anonymized text.
    Raises GuardrailError if the input should be blocked.
    """
    if check_toxicity(text):
        GUARDRAIL_BLOCKED.labels(reason="toxicity").inc()
        log.warning("guardrail_toxicity_blocked", preview=text[:60])
        raise GuardrailError("Input contains disallowed content.")

    pii = detect_pii(text)
    if pii:
        log.info("guardrail_pii_detected", entities=[p["entity_type"] for p in pii])
        GUARDRAIL_BLOCKED.labels(reason="pii_anonymized").inc()
        return anonymize_pii(text)

    return text


def guard_output(answer: str, context_chunks: list[str]) -> dict[str, object]:
    """
    Run output guardrails. Returns dict with answer + confidence score.
    Logs a warning if hallucination score is below threshold.
    """
    score = score_hallucination_nli(answer, context_chunks)
    if score < settings.guardrails_hallucination_threshold:
        log.warning(
            "guardrail_hallucination_low",
            score=round(score, 3),
            threshold=settings.guardrails_hallucination_threshold,
        )
        GUARDRAIL_BLOCKED.labels(reason="hallucination_warning").inc()

    return {"answer": answer, "confidence": round(score, 3)}
