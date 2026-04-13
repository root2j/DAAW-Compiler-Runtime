"""Groq LLM provider (AsyncGroq, OpenAI-compatible format)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from daaw.llm.base import (
    LLMMessage, LLMProvider, LLMResponse, LLMStreamChunk, ToolCall,
)

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

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream via Groq's native ``stream=True`` SDK support.

        Tool calls are intentionally NOT forwarded — streaming is used
        for display-oriented paths (compile, final text answers), where
        tool-loop semantics add complexity without benefit. Fall back
        to ``chat()`` when you need tools.
        """
        model = model or DEFAULT_MODEL
        oai_messages = _build_oai_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # The Groq SDK stream object is a sync iterator; run the whole
        # iteration in a worker thread to an asyncio.Queue so we stay
        # non-blocking for the event loop.
        import queue as _q
        import threading as _t

        q: _q.Queue = _q.Queue()

        def _drain():
            try:
                stream = self._sync_client.chat.completions.create(**kwargs)
                for event in stream:
                    try:
                        delta = event.choices[0].delta
                        piece = getattr(delta, "content", None) or ""
                    except Exception:  # noqa: BLE001
                        piece = ""
                    if piece:
                        q.put(("chunk", piece))
                q.put(("done", None))
            except Exception as e:  # noqa: BLE001
                q.put(("error", e))

        _t.Thread(target=_drain, daemon=True, name="groq-stream").start()

        accumulated: list[str] = []
        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "chunk":
                accumulated.append(payload)
                yield LLMStreamChunk(
                    delta=payload, done=False,
                    full_content="".join(accumulated),
                )
            elif kind == "done":
                break
            elif kind == "error":
                raise payload

        yield LLMStreamChunk(
            delta="", done=True, full_content="".join(accumulated),
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
