"""Unified LLM client — routes to the correct provider based on name."""

from __future__ import annotations

from typing import Any, AsyncIterator

from daaw.config import AppConfig
from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse, LLMStreamChunk
from daaw.llm.rate_limiter import RateLimiter, get_rate_limiter


class UnifiedLLMClient:
    """Dispatcher that lazily initialises only providers whose API keys are set."""

    def __init__(self, config: AppConfig, rate_limiter: RateLimiter | None = None):
        self._config = config
        self._providers: dict[str, LLMProvider] = {}
        self._rate_limiter = rate_limiter if rate_limiter is not None else get_rate_limiter()
        self._init_providers()

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    def _init_providers(self) -> None:
        if self._config.groq_api_key:
            from daaw.llm.providers.groq_provider import GroqProvider

            self._providers["groq"] = GroqProvider(self._config.groq_api_key)

        if self._config.gemini_api_key:
            from daaw.llm.providers.gemini_provider import GeminiProvider

            self._providers["gemini"] = GeminiProvider(self._config.gemini_api_key)

        if self._config.openai_api_key:
            from daaw.llm.providers.openai_provider import OpenAIProvider

            self._providers["openai"] = OpenAIProvider(self._config.openai_api_key)

        if self._config.anthropic_api_key:
            from daaw.llm.providers.anthropic_provider import AnthropicProvider

            self._providers["anthropic"] = AnthropicProvider(
                self._config.anthropic_api_key
            )

        # Generic OpenAI-compatible gateway (LiteLLM, Ollama, vLLM, etc.)
        if self._config.gateway_url:
            from daaw.llm.providers.gateway_provider import GatewayProvider

            self._providers["gateway"] = GatewayProvider(
                gateway_url=self._config.gateway_url,
                token=self._config.gateway_token,
                default_model=self._config.gateway_model,
            )

        # Claude Code API gateway — separate from `gateway` so both can
        # coexist (Ollama for execution, Claude for compilation).
        if self._config.claude_api_url:
            from daaw.llm.providers.gateway_provider import GatewayProvider

            self._providers["claude_api"] = GatewayProvider(
                gateway_url=self._config.claude_api_url,
                token=self._config.claude_api_token,
                default_model="claude-sonnet-4-5-20250929",
            )

    def available_providers(self) -> list[str]:
        return list(self._providers.keys())

    async def chat(
        self,
        provider: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if provider not in self._providers:
            available = ", ".join(self._providers) or "(none)"
            raise ValueError(
                f"Provider '{provider}' not available. "
                f"Configured providers: {available}"
            )
        # Cheap heuristic: assume worst-case token spend = prompt_chars/4 + max_tokens.
        # The exact usage is reconciled below once the response lands.
        estimated_tokens = max_tokens + sum(len(m.content) for m in messages) // 4
        await self._rate_limiter.acquire(provider, estimated_tokens=estimated_tokens)
        resp = await self._providers[provider].chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
            tools=tools,
        )
        self._rate_limiter.record_actual_usage(provider, resp.usage)
        return resp

    async def chat_stream(
        self,
        provider: str,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream chat completions, honouring the same rate limit as ``chat``."""
        if provider not in self._providers:
            available = ", ".join(self._providers) or "(none)"
            raise ValueError(
                f"Provider '{provider}' not available. "
                f"Configured providers: {available}"
            )
        estimated_tokens = max_tokens + sum(len(m.content) for m in messages) // 4
        await self._rate_limiter.acquire(provider, estimated_tokens=estimated_tokens)
        last_usage: dict[str, Any] = {}
        async for chunk in self._providers[provider].chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
            tools=tools,
        ):
            if chunk.usage:
                last_usage = chunk.usage
            yield chunk
        if last_usage:
            self._rate_limiter.record_actual_usage(provider, last_usage)
