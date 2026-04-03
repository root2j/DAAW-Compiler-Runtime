"""OpenClaw-backed tool implementations.

Routes tool calls through the local OpenClaw Gateway /tools/invoke endpoint.
This gives DAAW access to:
  - web_search: OpenClaw's DuckDuckGo plugin (real results, no scraping)
  - notify: Send a Telegram message via OpenClaw's configured channel

Config via env vars:
    OPENCLAW_GATEWAY_URL    default: http://127.0.0.1:18789
    OPENCLAW_GATEWAY_TOKEN  required
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from daaw.tools.registry import tool_registry

_GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
_GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")


async def _invoke(tool: str, args: dict[str, Any]) -> Any:
    """Call the OpenClaw Gateway /tools/invoke endpoint."""
    headers = {
        "Authorization": f"Bearer {_GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"tool": tool, "args": args}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_GATEWAY_URL}/tools/invoke",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"OpenClaw tool error: {data.get('error', data)}")
    return data.get("result")


@tool_registry.register(
    name="web_search",
    description="Search the web for information using OpenClaw's DuckDuckGo integration",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def openclaw_web_search(query: str) -> str:
    try:
        result = await _invoke("web_search", {"query": query})
        if isinstance(result, str):
            return result
        return str(result)
    except Exception as e:
        # Graceful fallback message so agents don't crash
        return f"[Search unavailable: {e}] Could not retrieve results for: {query}"


@tool_registry.register(
    name="notify",
    description="Send a notification message to the user via Telegram (OpenClaw)",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message text to send to the user",
            }
        },
        "required": ["message"],
    },
)
async def openclaw_notify(message: str) -> str:
    """Send a Telegram message via OpenClaw's message tool."""
    try:
        await _invoke("message", {
            "action": "send",
            "channel": "telegram",
            "message": message,
        })
        return "Notification sent via Telegram."
    except Exception as e:
        return f"[Notify failed: {e}]"
