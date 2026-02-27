"""Google Gemini LLM provider (google-genai SDK, sync wrapped in asyncio.to_thread)."""

from __future__ import annotations

import asyncio
from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)

    def name(self) -> str:
        return "gemini"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        from google.genai import types

        model = model or DEFAULT_MODEL

        # Separate system instruction from contents
        system_instruction = None
        contents: list[str] = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                contents.append(m.content)

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_format and response_format.get("type") == "json_object":
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        # Combine user/assistant messages into a single content string for simple use
        combined_content = "\n\n".join(contents)

        resp = await asyncio.to_thread(
            self._client.models.generate_content,
            model=model,
            contents=combined_content,
            config=config,
        )

        return LLMResponse(
            content=resp.text.strip(),
            model=model,
            usage={},
            raw=resp,
        )
