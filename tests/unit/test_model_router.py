import pytest

from rag_agent.services.model_router import (
    ABTestConfig,
    MODEL_PRICING,
    MODEL_SPEED_TIER,
    select_model,
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
