"""Groq LLM provider (AsyncGroq, OpenAI-compatible format)."""

from __future__ import annotations

import asyncio
from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # 8b-instant has unreliable structured tool use


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        from groq import Groq

        self._api_key = api_key
        self._sync_client = Groq(api_key=api_key)

    def name(self) -> str:
        return "groq"

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
        import json as _json

        model = model or DEFAULT_MODEL
        oai_messages = _build_oai_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools

        resp = await asyncio.to_thread(
            self._sync_client.chat.completions.create, **kwargs
        )

        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=(choice.message.content or "").strip(),
            model=resp.model,
            usage=usage,
            raw=resp,
            tool_calls=tool_calls,
        )


def _build_oai_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Convert LLMMessages to OpenAI-format dicts, handling tool results."""
    result = []
    for m in messages:
        if m.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
        elif m.tool_calls_raw is not None:
            result.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": m.tool_calls_raw,
            })
        else:
            result.append({"role": m.role, "content": m.content})
    return result
