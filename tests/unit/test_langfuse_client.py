"""Tests for langfuse_client — no-op tracing when unconfigured."""

from unittest.mock import MagicMock, patch

import pytest

import rag_agent.services.langfuse_client as lf_module
from rag_agent.services.langfuse_client import trace_generation


@pytest.fixture(autouse=True)
def reset_langfuse_state():
    """Reset module-level singletons between tests."""
    original_client = lf_module._client
    original_init = lf_module._initialized
    yield
    lf_module._client = original_client
    lf_module._initialized = original_init


def test_trace_generation_noop_when_unconfigured():
    """When no langfuse keys are configured, context manager is a no-op."""
    lf_module._client = None
    lf_module._initialized = False

    with patch("rag_agent.services.langfuse_client.settings") as mock_settings:
        mock_settings.langfuse_secret_key = None
        mock_settings.langfuse_public_key = None
        mock_settings.langfuse_host = "https://cloud.langfuse.com"

        with trace_generation("test", "gpt-4", [{"role": "user", "content": "hi"}]) as meta:
            meta["output"] = "response"
            meta["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}

        # No exception raised — noop completed successfully


def test_trace_generation_yields_meta_dict():
    lf_module._initialized = True
    lf_module._client = None

    with trace_generation("gen", "model", []) as meta:
        assert isinstance(meta, dict)
        meta["output"] = "answer"

    assert meta["output"] == "answer"


def test_trace_generation_calls_langfuse_when_configured():
    lf_module._initialized = True
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_client.trace.return_value = mock_trace
    lf_module._client = mock_client

    with trace_generation("name", "model", [{"role": "user", "content": "q"}]) as meta:
        meta["output"] = "ans"
        meta["usage"] = {"prompt_tokens": 5, "completion_tokens": 3}

    mock_client.trace.assert_called_once()
    mock_trace.generation.assert_called_once()
    mock_client.flush.assert_called_once()


def test_trace_generation_swallows_langfuse_errors():
    lf_module._initialized = True
    mock_client = MagicMock()
    mock_client.trace.side_effect = RuntimeError("langfuse down")
    lf_module._client = mock_client

    # Should NOT raise
    with trace_generation("name", "model", []) as meta:
        meta["output"] = "ans"


def test_get_client_returns_none_without_import():
    lf_module._initialized = False
    lf_module._client = None

    with patch("rag_agent.services.langfuse_client.settings") as mock_settings:
        mock_settings.langfuse_secret_key = "sk-xxx"  # pragma: allowlist secret
        mock_settings.langfuse_public_key = "pk-xxx"
        mock_settings.langfuse_host = "https://cloud.langfuse.com"

        with patch.dict("sys.modules", {"langfuse": None}):
            result = lf_module._get_client()

    assert result is None
