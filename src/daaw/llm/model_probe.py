"""Quick gateway-model compatibility probe.

Sends a minimal JSON-mode chat to the gateway and classifies the response.
Surfaces in the UI as a sidebar badge so the user finds out a model can't
produce valid JSON BEFORE running a 40-second compile that silently drifts.

Only meaningful for the ``gateway`` provider — cloud providers are reliable
enough that probing is wasted tokens.

Classification buckets match ``scripts/probe_local_models.py`` so CLI and UI
agree:
    VALID-JSON    parseable JSON
    JSON-BROKEN   starts with '{' but malformed
    TOKEN-SALAD   reserved-token salad (<unused42>, <tool|>, <pad>)
    EMPTY         zero bytes
    PROSE         plain text, not JSON
    ERROR         HTTP/network failure (diagnostic in detail)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

PROBE_TIMEOUT_SECONDS = 30.0

# Cache: (gateway_url, model) -> (ProbeResult, timestamp)
_CACHE: dict[tuple[str, str], tuple["ProbeResult", float]] = {}
CACHE_TTL_SECONDS = 3600  # re-probe hourly


@dataclass
class ProbeResult:
    model: str
    gateway_url: str
    classification: str  # one of the buckets above
    elapsed_seconds: float
    preview: str  # first 120 chars of content / error
    is_usable: bool  # shorthand: VALID-JSON only

    @property
    def badge(self) -> str:
        return {
            "VALID-JSON": "OK",
            "JSON-BROKEN": "FLAKY",
            "TOKEN-SALAD": "DRIFT",
            "EMPTY": "EMPTY",
            "PROSE": "NO-JSON",
            "ERROR": "ERROR",
        }.get(self.classification, self.classification)


def _classify(content: str) -> str:
    if not content or not content.strip():
        return "EMPTY"
    if "<unused" in content or "<tool|" in content or "<pad>" in content:
        return "TOKEN-SALAD"
    stripped = content.strip().lstrip("`").removeprefix("json").lstrip()
    if stripped.startswith("{"):
        try:
            if "```" in stripped:
                stripped = stripped.split("```")[0]
            json.loads(stripped)
            return "VALID-JSON"
        except Exception:
            return "JSON-BROKEN"
    return "PROSE"


async def probe_model(
    gateway_url: str,
    model: str,
    *,
    force: bool = False,
) -> ProbeResult:
    """Return a cached probe result, or run a fresh probe if expired.

    The probe sends one `chat/completions` request asking for a trivial JSON
    object. The full prompt is ~30 tokens so it costs negligible VRAM /
    latency.

    Args:
        gateway_url: base URL of the OpenAI-compatible gateway
            (e.g. ``http://127.0.0.1:11434/v1``).
        model: model name to probe.
        force: bypass the cache and re-probe.
    """
    key = (gateway_url.rstrip("/"), model)
    if not force:
        cached = _CACHE.get(key)
        if cached and (time.monotonic() - cached[1]) < CACHE_TTL_SECONDS:
            return cached[0]

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Respond with ONLY valid JSON. No prose, no markdown."
                ),
            },
            {
                "role": "user",
                "content": 'Return exactly this: {"ok": true, "n": 1}',
            },
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }
    # Ollama heuristic — pass options through without needing to recheck
    if ":11434" in gateway_url or "ollama" in gateway_url.lower():
        payload["options"] = {"num_ctx": 2048}
        payload["keep_alive"] = "5m"

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{gateway_url.rstrip('/')}/chat/completions",
                json=payload,
            )
    except Exception as e:  # noqa: BLE001
        result = ProbeResult(
            model=model,
            gateway_url=gateway_url,
            classification="ERROR",
            elapsed_seconds=round(time.monotonic() - start, 2),
            preview=f"{type(e).__name__}: {e}"[:120],
            is_usable=False,
        )
        _CACHE[key] = (result, time.monotonic())
        return result

    elapsed = round(time.monotonic() - start, 2)
    if resp.status_code != 200:
        result = ProbeResult(
            model=model, gateway_url=gateway_url,
            classification="ERROR", elapsed_seconds=elapsed,
            preview=f"HTTP {resp.status_code}: {resp.text[:100]}",
            is_usable=False,
        )
    else:
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            content = ""
            _ = e
        klass = _classify(content)
        result = ProbeResult(
            model=model, gateway_url=gateway_url,
            classification=klass, elapsed_seconds=elapsed,
            preview=content[:120].replace("\n", " "),
            is_usable=(klass == "VALID-JSON"),
        )

    _CACHE[key] = (result, time.monotonic())
    return result


def get_cached_probe(
    gateway_url: str, model: str,
) -> ProbeResult | None:
    """Return the cached probe without triggering a new one."""
    entry = _CACHE.get((gateway_url.rstrip("/"), model))
    if entry is None:
        return None
    result, ts = entry
    if (time.monotonic() - ts) >= CACHE_TTL_SECONDS:
        return None
    return result


def clear_probe_cache() -> None:
    _CACHE.clear()


__all__ = [
    "ProbeResult",
    "clear_probe_cache",
    "get_cached_probe",
    "probe_model",
]
