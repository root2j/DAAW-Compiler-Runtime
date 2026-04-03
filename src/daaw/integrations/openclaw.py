"""OpenClaw integration helpers for DAAW.

Provides:
  - notify_workflow_complete(): Send Telegram summary when a pipeline finishes
  - is_available(): Check if OpenClaw gateway is reachable
"""

from __future__ import annotations

import os

import httpx

_GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
_GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")


def is_available() -> bool:
    """Return True if the OpenClaw Gateway is reachable and token is set."""
    if not _GATEWAY_TOKEN:
        return False
    try:
        import httpx as _httpx
        resp = _httpx.get(
            f"{_GATEWAY_URL}/health",
            headers={"Authorization": f"Bearer {_GATEWAY_TOKEN}"},
            timeout=3.0,
        )
        return resp.status_code < 500
    except Exception:
        return False


async def notify_workflow_complete(
    workflow_name: str,
    total: int,
    passed: int,
    failed: int,
    elapsed: float,
) -> bool:
    """Send a Telegram notification summarising the completed DAAW pipeline.

    Returns True if notification was sent successfully.
    """
    if not _GATEWAY_TOKEN:
        return False

    status_emoji = "✅" if failed == 0 else ("⚠️" if passed > 0 else "❌")
    lines = [
        f"{status_emoji} *DAAW Pipeline Complete*",
        f"📋 *{workflow_name}*",
        f"",
        f"• Tasks: {passed}/{total} passed",
        f"• Failed: {failed}",
        f"• Total time: {elapsed:.1f}s",
    ]
    if failed == 0:
        lines.append(f"")
        lines.append(f"All tasks passed critic evaluation 🎉")

    text = "\n".join(lines)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_GATEWAY_URL}/tools/invoke",
                json={
                    "tool": "message",
                    "args": {
                        "action": "send",
                        "channel": "telegram",
                        "message": text,
                    },
                },
                headers={
                    "Authorization": f"Bearer {_GATEWAY_TOKEN}",
                    "Content-Type": "application/json",
                },
            )
            data = resp.json()
            return data.get("ok", False)
    except Exception:
        return False
