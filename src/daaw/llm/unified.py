"""Unified LLM client — routes to the correct provider based on name."""

from __future__ import annotations

from typing import Any

from daaw.config import AppConfig
from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse


class UnifiedLLMClient:
    """Dispatcher that lazily initialises only providers whose API keys are set."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._providers: dict[str, LLMProvider] = {}
        self._init_providers()

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
        return await self._providers[provider].chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
            tools=tools,
        )
