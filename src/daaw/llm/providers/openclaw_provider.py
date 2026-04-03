"""OpenClaw provider — routes through the local OpenClaw Gateway OpenAI-compatible endpoint.

This gives DAAW access to Claude (Sonnet/Opus) via the already-configured
OpenClaw instance, including its auth, rate limiting, and model failover.

Config via env vars:
    OPENCLAW_GATEWAY_URL   default: http://127.0.0.1:18789
    OPENCLAW_GATEWAY_TOKEN required: gateway bearer token
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

# Map our model aliases to OpenClaw agent targets
# "openclaw/default" → whatever model OpenClaw is configured with (Sonnet by default)
DEFAULT_MODEL = "openclaw/default"


class OpenClawProvider(LLMProvider):
    """OpenAI-compatible provider backed by the local OpenClaw Gateway."""

    def __init__(self, gateway_url: str = GATEWAY_URL, token: str = GATEWAY_TOKEN):
        self._base_url = gateway_url.rstrip("/")
        self._token = token

    def name(self) -> str:
        return "openclaw"

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        import json as _json

        import httpx

        target_model = model or DEFAULT_MODEL
        # Remap bare model aliases to openclaw targets
        if "/" not in target_model or not target_model.startswith("openclaw"):
            target_model = DEFAULT_MODEL

        oai_messages = _build_oai_messages(messages)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # response_format may not be honoured by all OpenClaw-proxied models;
        # inject a JSON instruction in the system message instead as a fallback.
        if response_format and response_format.get("type") == "json_object":
            # Try passing it natively; also reinforce via system prompt
            payload["response_format"] = response_format
            # Prepend JSON instruction to first system message if not already there
            for msg in oai_messages:
                if msg["role"] == "system":
                    if "JSON" not in (msg["content"] or ""):
                        msg["content"] = "Respond with ONLY valid JSON. No markdown, no explanation.\n" + (msg["content"] or "")
                    break
        elif response_format:
            payload["response_format"] = response_format
        # NOTE: Do NOT forward tools to the OpenClaw gateway.
        # OpenClaw's /v1/chat/completions runs a full agent turn in OpenClaw's
        # own tool system — it cannot execute DAAW's local tool registry.
        # DAAW's GenericLLMAgent handles tool dispatch locally; OpenClaw only
        # needs to return the raw LLM text response.
        # If tools were passed, inject a system hint so the LLM includes
        # structured output that DAAW can parse without needing tool_calls.
        if tools:
            tool_names = ", ".join(t["function"]["name"] for t in tools if "function" in t)
            # Append a hint to the last user message so the model knows what tools exist
            if oai_messages and oai_messages[-1]["role"] == "user":
                oai_messages[-1]["content"] = (
                    oai_messages[-1]["content"]
                    + f"\n\n[Available tools (call via structured output if needed): {tool_names}]"
                )

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = (msg.get("content") or "").strip()

        usage = data.get("usage", {})

        tool_calls: list[ToolCall] = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = _json.loads(fn.get("arguments", "{}"))
                except _json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                ))

        return LLMResponse(
            content=content,
            model=data.get("model", target_model),
            usage=usage,
            raw=data,
            tool_calls=tool_calls,
        )


def _build_oai_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
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
