from unittest.mock import AsyncMock, patch

import pytest

from rag_agent.services.model_router import (
    MODEL_PRICING,
    MODEL_SPEED_TIER,
    ABTestConfig,
    call_with_routing,
    select_model,
    track_usage,
)


def test_select_default_returns_config_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "default_model", "google/gemini-flash-1.5")
    assert select_model("default") == "google/gemini-flash-1.5"


def test_select_cheapest() -> None:
    model = select_model("cheapest")
    min_price = min(MODEL_PRICING.values())
    assert MODEL_PRICING[model] == min_price


def test_select_fastest() -> None:
    model = select_model("fastest")
    min_tier = min(MODEL_SPEED_TIER.values())
    assert MODEL_SPEED_TIER[model] == min_tier


def test_ab_test_picks_from_configured_models() -> None:
    config = ABTestConfig(
        models=["model-a", "model-b"],
        weights=[50.0, 50.0],
    )
    for _ in range(20):
        chosen = config.pick()
        assert chosen in ("model-a", "model-b")


def test_ab_test_weighted_distribution() -> None:
    # 100% weight on model-a → always picks model-a
    config = ABTestConfig(models=["model-a", "model-b"], weights=[100.0, 0.001])
    results = [config.pick() for _ in range(50)]
    assert results.count("model-a") > 40


def test_unknown_mode_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "default_model", "fallback-model")
    assert select_model("nonexistent_mode") == "fallback-model"


def test_select_quality_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_agent.core import config

    monkeypatch.setattr(config.settings, "quality_model", "anthropic/claude-3.5-sonnet")
    assert select_model("quality") == "anthropic/claude-3.5-sonnet"


def test_select_ab_test_uses_default_config() -> None:
    result = select_model("ab_test")
    assert result in ["google/gemini-flash-1.5", "anthropic/claude-3.5-sonnet"]


def test_select_ab_test_custom_config_always_picks_winner() -> None:
    config = ABTestConfig(models=["winner", "loser"], weights=[100.0, 0.0])
    result = select_model("ab_test", ab_config=config)
    assert result == "winner"


def test_track_usage_known_model() -> None:
    track_usage("google/gemini-flash-1.5", prompt_tokens=1000, completion_tokens=500, latency_s=0.3)


def test_track_usage_unknown_model_uses_default_price() -> None:
    track_usage("unknown/model-xyz", prompt_tokens=100, completion_tokens=50, latency_s=0.1)


@pytest.mark.asyncio
async def test_call_with_routing_returns_answer_model() -> None:
    def noop_trace(*a, **kw):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            yield {}

        return _ctx()

    with (
        patch("rag_agent.services.langfuse_client.trace_generation", side_effect=noop_trace),
        patch(
            "rag_agent.services.llm_client.complete",
            new=AsyncMock(return_value=("answer", {"prompt_tokens": 5, "completion_tokens": 3})),
        ),
    ):
        from rag_agent.core import config

        answer, usage, model = await call_with_routing(
            [{"role": "user", "content": "hello"}], mode="default"
        )

    assert answer == "answer"
    assert "prompt_tokens" in usage
    assert model == config.settings.default_model
