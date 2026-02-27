"""Groq LLM provider (AsyncGroq, OpenAI-compatible format)."""

from __future__ import annotations

import asyncio
from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse

DEFAULT_MODEL = "llama-3.1-8b-instant"


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
    ) -> LLMResponse:
        model = model or DEFAULT_MODEL
        oai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

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
        return LLMResponse(
            content=choice.message.content.strip(),
            model=resp.model,
            usage=usage,
            raw=resp,
        )
