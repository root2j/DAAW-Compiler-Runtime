"""Anthropic LLM provider (AsyncAnthropic)."""

from __future__ import annotations

from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    def name(self) -> str:
        return "anthropic"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL

        # Anthropic requires system separate from messages
        system_text = ""
        api_messages: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                api_messages.append({"role": m.role, "content": m.content})

        # Anthropic doesn't have native response_format — inject JSON instruction
        if response_format and response_format.get("type") == "json_object":
            system_text += (
                "\n\nIMPORTANT: You MUST respond with valid JSON only. "
                "No markdown, no explanation — just the JSON object."
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        resp = await self._client.messages.create(**kwargs)

        content = resp.content[0].text.strip() if resp.content else ""
        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        return LLMResponse(
            content=content,
            model=resp.model,
            usage=usage,
            raw=resp,
        )
