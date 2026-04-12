"""Webhook-based notification helpers for DAAW.

Provides:
  - notify_workflow_complete(): Send summary when a pipeline finishes
  - is_available(): Check if webhook notifications are configured

Config via env vars:
    NOTIFY_WEBHOOK_URL   required: webhook endpoint URL
    NOTIFY_WEBHOOK_TYPE  optional: "discord", "slack", or "generic" (default: "generic")
"""

from __future__ import annotations

import os

_WEBHOOK_URL = os.environ.get("NOTIFY_WEBHOOK_URL", "")
_WEBHOOK_TYPE = os.environ.get("NOTIFY_WEBHOOK_TYPE", "generic").lower()


def is_available() -> bool:
    """Return True if webhook notifications are configured."""
    if not _WEBHOOK_URL:
        return False
    try:
        import httpx as _httpx
        resp = _httpx.head(_WEBHOOK_URL, timeout=3.0)
        # Most webhooks return 2xx or 4xx (method not allowed) — anything but 5xx is fine
        return resp.status_code < 500
    except Exception:
        # URL is set but unreachable — still consider it "configured"
        # so the UI shows the status. Actual sends will fail gracefully.
        return bool(_WEBHOOK_URL)


def _build_payload(text: str) -> dict:
    """Build the webhook payload based on the configured type."""
    if _WEBHOOK_TYPE == "discord":
        return {"content": text}
    if _WEBHOOK_TYPE == "slack":
        return {"text": text}
    return {"message": text}


async def notify_workflow_complete(
    workflow_name: str,
    total: int,
    passed: int,
    failed: int,
    elapsed: float,
) -> bool:
    """Send a webhook notification summarising the completed DAAW pipeline.

    Returns True if notification was sent successfully.
    """
    if not _WEBHOOK_URL:
        return False

    status_emoji = "+" if failed == 0 else ("!" if passed > 0 else "X")
    lines = [
        f"[{status_emoji}] DAAW Pipeline Complete",
        f"Workflow: {workflow_name}",
        f"",
        f"Tasks: {passed}/{total} passed",
        f"Failed: {failed}",
        f"Total time: {elapsed:.1f}s",
    ]
    if failed == 0:
        lines.append("")
        lines.append("All tasks passed critic evaluation.")

    text = "\n".join(lines)
    payload = _build_payload(text)

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code < 400
    except Exception:
        return False
