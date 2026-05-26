import pytest

from rag_agent.core.exceptions import GuardrailError
from rag_agent.services.guardrails import check_toxicity, detect_pii, guard_input


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
