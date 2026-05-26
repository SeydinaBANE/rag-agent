"""Model router: A/B test, cheapest, fastest selection via OpenRouter."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import structlog
from prometheus_client import Counter, Histogram

from rag_agent.core.config import settings

log = structlog.get_logger()

MODEL_CALLS = Counter("model_router_calls_total", "Model calls by router", ["model", "mode"])
MODEL_LATENCY = Histogram("model_router_latency_seconds", "Model latency by model", ["model"])
MODEL_COST = Counter("model_router_cost_usd_total", "Estimated cost in USD", ["model"])

# Cost in USD per 1M tokens (prompt + completion averaged)
# Source: OpenRouter pricing — update as needed
MODEL_PRICING: dict[str, float] = {
    "google/gemini-flash-1.5": 0.075,
    "google/gemini-flash-2.0": 0.10,
    "mistralai/mistral-large": 4.0,
    "mistralai/mistral-small": 0.60,
    "anthropic/claude-3.5-sonnet": 3.0,
    "anthropic/claude-3-haiku": 0.25,
    "openai/gpt-4o-mini": 0.15,
    "openai/gpt-4o": 2.50,
    "meta-llama/llama-3.1-8b-instruct": 0.06,
    "meta-llama/llama-3.3-70b-instruct": 0.59,
}

# Rough latency tier (lower = faster), subjective ranking
MODEL_SPEED_TIER: dict[str, int] = {
    "google/gemini-flash-2.0": 1,
    "google/gemini-flash-1.5": 2,
    "openai/gpt-4o-mini": 2,
    "anthropic/claude-3-haiku": 2,
    "meta-llama/llama-3.1-8b-instruct": 3,
    "mistralai/mistral-small": 4,
    "meta-llama/llama-3.3-70b-instruct": 5,
    "openai/gpt-4o": 6,
    "anthropic/claude-3.5-sonnet": 6,
    "mistralai/mistral-large": 7,
}


@dataclass
class ABTestConfig:
    """Define A/B test weights. Weights are relative (don't need to sum to 100)."""

    models: list[str]
    weights: list[float]

    def pick(self) -> str:
        return random.choices(self.models, weights=self.weights, k=1)[0]


# Default A/B test: 70% fast/cheap vs 30% quality
DEFAULT_AB_CONFIG = ABTestConfig(
    models=["google/gemini-flash-1.5", "anthropic/claude-3.5-sonnet"],
    weights=[70.0, 30.0],
)


def select_model(
    mode: str = "default",
    ab_config: ABTestConfig | None = None,
    quality_threshold: float = 0.0,
) -> str:
    """
    Select a model based on mode:
    - 'default':  use settings.default_model
    - 'quality':  use settings.quality_model
    - 'ab_test':  weighted random from ab_config
    - 'cheapest': cheapest model in MODEL_PRICING
    - 'fastest':  fastest model by speed tier
    """
    if mode == "default":
        chosen = settings.default_model
    elif mode == "quality":
        chosen = settings.quality_model
    elif mode == "ab_test":
        config = ab_config or DEFAULT_AB_CONFIG
        chosen = config.pick()
    elif mode == "cheapest":
        chosen = min(MODEL_PRICING, key=lambda m: MODEL_PRICING[m])
    elif mode == "fastest":
        chosen = min(MODEL_SPEED_TIER, key=lambda m: MODEL_SPEED_TIER[m])
    else:
        chosen = settings.default_model

    MODEL_CALLS.labels(model=chosen, mode=mode).inc()
    log.debug("model_router_selected", model=chosen, mode=mode)
    return chosen


def track_usage(model: str, prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
    """Track cost and latency metrics after a model call."""
    total_tokens = prompt_tokens + completion_tokens
    price_per_million = MODEL_PRICING.get(model, 1.0)
    cost_usd = (total_tokens / 1_000_000) * price_per_million

    MODEL_COST.labels(model=model).inc(cost_usd)
    MODEL_LATENCY.labels(model=model).observe(latency_s)
    log.info(
        "model_usage",
        model=model,
        tokens=total_tokens,
        cost_usd=round(cost_usd, 6),
        latency_s=round(latency_s, 3),
    )


async def call_with_routing(
    messages: list[dict[str, str]],
    mode: str = "default",
    ab_config: ABTestConfig | None = None,
    **kwargs: object,
) -> tuple[str, dict[str, int], str]:
    """Route to the right model, call it, track metrics. Returns (answer, usage, model)."""
    from rag_agent.services import llm_client

    model = select_model(mode=mode, ab_config=ab_config)
    start = time.perf_counter()
    answer, usage = await llm_client.complete(messages, model=model, **kwargs)  # type: ignore[arg-type]
    latency = time.perf_counter() - start

    track_usage(
        model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        latency_s=latency,
    )
    return answer, usage, model
