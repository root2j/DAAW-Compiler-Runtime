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
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        from google.genai import types

        model = model or DEFAULT_MODEL

        # Separate system instruction and build structured contents
        system_instruction = None
        contents: list[types.Content] = []
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                gemini_role = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=m.content)],
                ))

        # Gemini needs at least one content entry
        if not contents:
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text="")],
            ))

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_format and response_format.get("type") == "json_object":
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        resp = await asyncio.to_thread(
            self._client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        return LLMResponse(
            content=resp.text.strip(),
            model=model,
            usage={},
            raw=resp,
        )
