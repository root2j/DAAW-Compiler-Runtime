"""A/B probe the locally-installed Ollama models against DAAW's actual
compile workload. Answers: which model reliably produces valid JSON for
our planner, and did my num_ctx changes regress anything?

Usage:
    python scripts/probe_local_models.py

Does NOT use the DAAW gateway provider directly — it talks to Ollama via
raw HTTP so we can hold all other variables constant and isolate model
behaviour from our retry/degeneracy logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from daaw.compiler.prompts import PLANNER_SYSTEM_PROMPT  # noqa: E402

GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:11434/v1").rstrip("/")

# Real-ish goal from the user's recent runs.
USER_GOAL = (
    "Plan a 5-day Tokyo trip balancing tech and tradition. Include "
    "Akihabara, Shibuya Crossing, Senso-ji Temple, takoyaki, ramen, "
    "a sushi omakase, a matcha ceremony, and an arcade evening."
)

# Build the same messages the Compiler would send.
SYSTEM = PLANNER_SYSTEM_PROMPT.format(available_tools="web_search, file_write")
USER = f"Create a workflow plan for this goal:\n\n{USER_GOAL}"


def _classify(content: str) -> str:
    if not content or not content.strip():
        return "EMPTY"
    if "<unused" in content or "<tool|" in content or "<pad>" in content:
        return "TOKEN-SALAD"
    stripped = content.strip().lstrip("` ").removeprefix("json").lstrip()
    if stripped.startswith("{"):
        # Try to parse
        try:
            # Strip trailing fences if any
            if "```" in stripped:
                stripped = stripped.split("```")[0]
            data = json.loads(stripped)
            if "tasks" in data and isinstance(data["tasks"], list):
                return f"VALID-JSON ({len(data['tasks'])} tasks)"
            return "VALID-JSON (no tasks)"
        except Exception as e:
            return f"JSON-BROKEN ({type(e).__name__})"
    return "PROSE"


async def _one_call(
    client: httpx.AsyncClient,
    model: str,
    num_ctx: int | None,
    keep_alive: str | None,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    if num_ctx is not None:
        payload["options"] = {"num_ctx": num_ctx}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    t0 = time.monotonic()
    try:
        resp = await client.post(f"{GATEWAY}/chat/completions", json=payload)
    except Exception as e:
        return {"status": "HTTP-ERROR", "detail": f"{type(e).__name__}: {e}",
                "elapsed": round(time.monotonic() - t0, 1)}
    elapsed = round(time.monotonic() - t0, 1)
    if resp.status_code != 200:
        body = resp.text[:200]
        return {"status": f"HTTP {resp.status_code}", "detail": body,
                "elapsed": elapsed}
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"status": "PARSE-ERROR", "detail": str(e)[:120],
                "elapsed": elapsed}
    return {
        "status": _classify(content),
        "preview": content[:120].replace("\n", " "),
        "elapsed": elapsed,
        "length": len(content),
    }


async def probe_model(model: str, configurations: list[dict], trials: int = 2):
    print(f"\n== {model} ==")
    print(f"{'CONFIG':<35} {'TRIAL':<6} {'STATUS':<28} {'TIME':<8} PREVIEW")
    print("-" * 120)
    async with httpx.AsyncClient(timeout=180.0) as client:
        # Warm the model once so the first real trial isn't dominated by load time.
        await _one_call(client, model, num_ctx=2048, keep_alive="5m")
        for cfg in configurations:
            name = cfg["name"]
            for trial in range(1, trials + 1):
                res = await _one_call(
                    client, model,
                    num_ctx=cfg.get("num_ctx"),
                    keep_alive=cfg.get("keep_alive"),
                )
                preview = res.get("preview") or res.get("detail", "")
                print(f"{name:<35} {trial:<6} {res['status']:<28} "
                      f"{res['elapsed']:<8} {preview[:60]}")


async def main() -> int:
    # Discover installed models.
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{GATEWAY.rstrip('/v1')}/api/tags")
        models = [m["name"] for m in r.json().get("models", [])]
    print(f"Installed models: {models}")

    configurations = [
        {"name": "ollama default (num_ctx=2048)", "num_ctx": 2048, "keep_alive": "5m"},
        {"name": "daaw pre-fix (no options)",       "num_ctx": None, "keep_alive": None},
        {"name": "daaw new default (num_ctx=4096)", "num_ctx": 4096, "keep_alive": "5m"},
        {"name": "num_ctx=8192",                    "num_ctx": 8192, "keep_alive": "5m"},
    ]

    for m in models:
        await probe_model(m, configurations, trials=2)

    print()
    print("Legend:")
    print("  VALID-JSON  = parseable JSON with a 'tasks' array (works with compiler)")
    print("  JSON-BROKEN = starts with '{' but not parseable")
    print("  TOKEN-SALAD = reserved tokens like <unused42>, <tool|>, <pad>")
    print("  EMPTY       = zero bytes returned")
    print("  PROSE       = plain text, not JSON at all")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
