"""LLM provider ABC and message/response data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None  # set when role == "tool"
    tool_calls_raw: Any = None  # preserved from assistant msg for round-trip


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LLMStreamChunk:
    """One increment from a streaming LLM call.

    ``delta`` is the new text since the previous chunk. ``done`` is True
    for the final chunk, which also carries the accumulated ``full_content``
    and any ``usage`` numbers reported by the provider.
    """

    delta: str
    done: bool = False
    full_content: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base for all LLM provider implementations."""

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

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
        """Yield incremental output chunks.

        Default implementation falls back to the non-streaming ``chat``
        and emits one final chunk — so providers that don't support
        real SSE streaming still satisfy this interface. Override in
        subclasses that do support ``stream=True``.
        """
        resp = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            model=model,
            tools=tools,
        )
        yield LLMStreamChunk(
            delta=resp.content,
            done=True,
            full_content=resp.content,
            usage=resp.usage,
            tool_calls=resp.tool_calls,
        )

    @abstractmethod
    def name(self) -> str: ...
