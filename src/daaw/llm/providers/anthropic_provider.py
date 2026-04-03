"""Anthropic LLM provider (AsyncAnthropic)."""

from __future__ import annotations

from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

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
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL

        # Anthropic requires system separate from messages
        system_text = ""
        api_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            elif m.role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content,
                    }],
                })
            elif m.tool_calls_raw is not None:
                api_messages.append({
                    "role": "assistant",
                    "content": m.tool_calls_raw,
                })
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

        # Convert OpenAI-style tools to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn["description"],
                    "input_schema": fn["parameters"],
                })
            kwargs["tools"] = anthropic_tools

        resp = await self._client.messages.create(**kwargs)

        # Extract text and tool_use blocks
        content_text = ""
        tool_calls = []
        raw_content_blocks = []
        for block in resp.content:
            raw_content_blocks.append(block)
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        usage = {}
        if resp.usage:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        return LLMResponse(
            content=content_text.strip(),
            model=resp.model,
            usage=usage,
            raw=resp,
            tool_calls=tool_calls,
        )
