"""OpenRouter LLM client — drop-in openai-compatible."""

from collections.abc import AsyncGenerator

import structlog
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from rag_agent.core.config import settings
from rag_agent.core.exceptions import LLMError

log = structlog.get_logger()

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/baneseydina/rag-agent",
                "X-Title": "rag-agent",
            },
        )
    return _client


async def complete(
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> tuple[str, dict[str, int]]:
    """Return (answer, usage_dict)."""
    model = model or settings.default_model
    try:
        response = await get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        content = response.choices[0].message.content or ""
        log.debug("llm_complete", model=model, tokens=usage)
        return content, usage
    except Exception as exc:
        raise LLMError(str(exc)) from exc


async def stream(
    messages: list[ChatCompletionMessageParam],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """Yield tokens one by one."""
    model = model or settings.default_model
    try:
        async with await get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ) as response:
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
    except Exception as exc:
        raise LLMError(str(exc)) from exc
