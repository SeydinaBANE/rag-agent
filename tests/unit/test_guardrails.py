from unittest.mock import MagicMock, patch

import pytest

from rag_agent.core.exceptions import GuardrailError
from rag_agent.services.guardrails import (
    anonymize_pii,
    check_toxicity,
    detect_pii,
    guard_input,
    guard_output,
    score_hallucination_nli,
)


def test_toxicity_detected() -> None:
    assert check_toxicity("how to make a bomb") is True


def test_toxicity_clean() -> None:
    assert check_toxicity("what is machine learning?") is False


def test_toxicity_case_insensitive() -> None:
    assert check_toxicity("How to HACK a system") is True


def test_guard_input_blocks_toxic() -> None:
    with pytest.raises(GuardrailError, match="disallowed"):
        guard_input("how to make a bomb at home")


def test_guard_input_passes_clean() -> None:
    with patch("rag_agent.services.guardrails.detect_pii", return_value=[]):
        result = guard_input("what is retrieval augmented generation?")
    assert "retrieval" in result


def test_pii_detection_no_presidio(monkeypatch: pytest.MonkeyPatch) -> None:
    # When presidio is not available, should return empty list gracefully
    import rag_agent.services.guardrails as g

    original = g._analyzer
    g._analyzer = None

    # Mock the import to fail
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "presidio_analyzer":
            raise ImportError("presidio not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    result = detect_pii("My name is John Doe")
    # Should return empty list, not raise
    assert isinstance(result, list)
    g._analyzer = original


def test_check_toxicity_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_toxicity_enabled", False)
    assert check_toxicity("how to make a bomb") is False


def test_detect_pii_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_pii_enabled", False)
    result = detect_pii("My email is test@example.com")
    assert result == []


def test_anonymize_pii_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_pii_enabled", False)
    text = "My name is John Doe"
    result = anonymize_pii(text)
    assert result == text


def test_detect_pii_with_mock_presidio(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import rag_agent.services.guardrails as g
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_pii_enabled", True)

    mock_result = MagicMock()
    mock_result.entity_type = "EMAIL_ADDRESS"
    mock_result.start = 0
    mock_result.end = 20
    mock_result.score = 0.95

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = [mock_result]

    mock_analyzer_engine = MagicMock()
    mock_module = MagicMock()
    mock_module.AnalyzerEngine = mock_analyzer_engine

    original = g._analyzer
    g._analyzer = mock_analyzer

    with patch.dict(sys.modules, {"presidio_analyzer": mock_module}):
        results = detect_pii("test@example.com")

    assert len(results) == 1
    assert results[0]["entity_type"] == "EMAIL_ADDRESS"
    g._analyzer = original


def test_anonymize_pii_with_mock_presidio(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import rag_agent.services.guardrails as g
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_pii_enabled", True)

    mock_result = MagicMock()
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = [mock_result]

    mock_anonymized = MagicMock()
    mock_anonymized.text = "<EMAIL_ADDRESS>"
    mock_anonymizer = MagicMock()
    mock_anonymizer.anonymize.return_value = mock_anonymized

    mock_ana_module = MagicMock()
    mock_anon_module = MagicMock()

    original_ana = g._analyzer
    original_anon = g._anonymizer
    g._analyzer = mock_analyzer
    g._anonymizer = mock_anonymizer

    with patch.dict(
        sys.modules, {"presidio_analyzer": mock_ana_module, "presidio_anonymizer": mock_anon_module}
    ):
        result = anonymize_pii("test@example.com")

    assert result == "<EMAIL_ADDRESS>"
    g._analyzer = original_ana
    g._anonymizer = original_anon


def test_anonymize_pii_no_results_returns_original(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    import rag_agent.services.guardrails as g
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "guardrails_pii_enabled", True)

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = []  # No PII detected

    mock_ana_module = MagicMock()

    original = g._analyzer
    g._analyzer = mock_analyzer

    text = "No PII here at all."
    with patch.dict(sys.modules, {"presidio_analyzer": mock_ana_module}):
        result = anonymize_pii(text)

    assert result == text
    g._analyzer = original


def test_score_hallucination_nli_no_context() -> None:
    score = score_hallucination_nli("answer", [])
    assert score == 0.0


def test_score_hallucination_nli_no_sentence_transformers() -> None:
    score = score_hallucination_nli("The answer is correct", ["The answer is correct."])
    # sentence-transformers not installed → fallback
    assert score == 0.5


def test_guard_output_high_confidence() -> None:
    with patch("rag_agent.services.guardrails.score_hallucination_nli", return_value=0.9):
        with patch("rag_agent.services.guardrails.settings") as mock_settings:
            mock_settings.guardrails_hallucination_threshold = 0.5
            result = guard_output("The answer", ["context chunk"])

    assert result["confidence"] == pytest.approx(0.9)
    assert result["answer"] == "The answer"


def test_guard_output_low_confidence_logs_warning() -> None:
    with patch("rag_agent.services.guardrails.score_hallucination_nli", return_value=0.2):
        with patch("rag_agent.services.guardrails.settings") as mock_settings:
            mock_settings.guardrails_hallucination_threshold = 0.5
            result = guard_output("hallucinated answer", ["context chunk"])

    assert result["confidence"] == pytest.approx(0.2)


def test_guard_input_returns_text_when_pii_detected() -> None:
    """When PII detected but no toxicity, anonymized text is returned."""
    with (
        patch("rag_agent.services.guardrails.check_toxicity", return_value=False),
        patch("rag_agent.services.guardrails.detect_pii", return_value=[{"entity_type": "EMAIL"}]),
        patch("rag_agent.services.guardrails.anonymize_pii", return_value="text with <EMAIL>"),
    ):
        result = guard_input("text with test@example.com")

    assert result == "text with <EMAIL>"
