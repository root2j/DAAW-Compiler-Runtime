"""Webhook-based notification tools.

Sends notifications via generic webhooks (Discord, Slack, or custom HTTP).

Config via env vars:
    NOTIFY_WEBHOOK_URL   required: webhook endpoint URL
    NOTIFY_WEBHOOK_TYPE  optional: "discord", "slack", or "generic" (default: "generic")
"""

from __future__ import annotations

import os
from typing import Any

from daaw.tools.registry import tool_registry

_WEBHOOK_URL = os.environ.get("NOTIFY_WEBHOOK_URL", "")
_WEBHOOK_TYPE = os.environ.get("NOTIFY_WEBHOOK_TYPE", "generic").lower()


def _build_payload(message: str) -> dict[str, Any]:
    """Build the webhook payload based on the configured type."""
    if _WEBHOOK_TYPE == "discord":
        return {"content": message}
    if _WEBHOOK_TYPE == "slack":
        return {"text": message}
    # Generic: send as JSON with a "message" key
    return {"message": message}


@tool_registry.register(
    name="notify",
    description="Send a notification message via webhook (Discord, Slack, or generic HTTP)",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message text to send",
            }
        },
        "required": ["message"],
    },
)
async def webhook_notify(message: str) -> str:
    """Send a notification via the configured webhook."""
    if not _WEBHOOK_URL:
        return "[Notify skipped: NOTIFY_WEBHOOK_URL not set]"

    import httpx

    payload = _build_payload(message)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        return f"Notification sent via {_WEBHOOK_TYPE} webhook."
    except Exception as e:
        return f"[Notify failed: {e}]"
