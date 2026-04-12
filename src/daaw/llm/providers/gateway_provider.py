"""Generic OpenAI-compatible gateway provider.

Routes LLM calls through any OpenAI-compatible endpoint such as:
  - LM Studio (http://127.0.0.1:1234/v1)
  - LiteLLM proxy (https://litellm.ai)
  - Ollama (http://localhost:11434/v1)
  - vLLM (OpenAI-compatible server)
  - LocalAI, text-generation-webui, etc.

Config via env vars:
    GATEWAY_URL    default: http://127.0.0.1:11434/v1
    GATEWAY_TOKEN  optional: bearer token for auth
    GATEWAY_MODEL  optional: default model name

Compatibility:
    Local backends often don't support OpenAI's response_format or tools
    parameters.  This provider injects instructions into the system prompt
    instead.  Tool calls use the XML text format that GenericLLMAgent
    already parses via _parse_text_tool_calls.
"""

from __future__ import annotations

import json as _json
import os
import re
from typing import Any

from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:11434/v1")
DEFAULT_GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")
DEFAULT_GATEWAY_MODEL = os.environ.get("GATEWAY_MODEL", "default")

# Reasoning models burn tokens on chain-of-thought before the response.
_REASONING_TOKEN_MULTIPLIER = 4
_MIN_REASONING_TOKENS = 8192

# How many times to retry if the model produces garbage.
# Local models crash intermittently from VRAM fragmentation (~every 7
# requests) but recover within seconds.  We retry generously.
_MAX_RETRIES = 4


class GatewayProvider(LLMProvider):
    """OpenAI-compatible provider for any local or remote gateway."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_GATEWAY_URL,
        token: str = DEFAULT_GATEWAY_TOKEN,
        default_model: str = DEFAULT_GATEWAY_MODEL,
    ):
        self._base_url = gateway_url.rstrip("/")
        self._token = token
        self._default_model = default_model

    def name(self) -> str:
        return "gateway"

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
        import httpx

        target_model = model or self._default_model
        oai_messages = _build_oai_messages(messages)

        # Adjust max_tokens for local model constraints:
        # - Reasoning models need a multiplier for chain-of-thought
        # - Non-reasoning models must stay within context window
        #   (e.g. 4096 context - 300 prompt = 3700 max generation)
        _lower = target_model.lower()
        is_reasoning = any(k in _lower for k in ("qwen3", "deepseek-r1", "qwq"))
        if is_reasoning:
            effective_max = max(
                max_tokens * _REASONING_TOKEN_MULTIPLIER,
                _MIN_REASONING_TOKENS,
            )
        else:
            # Cap generation to fit within a single context slot.
            # LM Studio often runs parallel=4, splitting a 4096 context
            # into 1024 per slot.  Reserve ~300 tokens for the prompt.
            effective_max = min(max_tokens, 700)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": effective_max,
        }
        # Only add repetition penalty for reasoning models prone to ///... loops.
        # Non-reasoning models (gemma, llama) can crash with conflicting penalties.
        if is_reasoning:
            payload["frequency_penalty"] = 0.5
            payload["presence_penalty"] = 0.3

        # ── JSON mode ── prompt reinforcement instead of API param
        if response_format and response_format.get("type") in (
            "json_object", "json_schema",
        ):
            _inject_system(oai_messages, (
                "IMPORTANT: Respond with ONLY valid JSON. "
                "No markdown fences, no explanation, no text "
                "outside the JSON object.\n\n"
            ), guard="ONLY valid JSON")

        # ── Tool calling ──
        # Try native tools API first (Ollama supports it).
        # Fall back to text-injection for backends that don't.
        if tools:
            payload["tools"] = tools

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        # Retry loop: handles degenerate outputs (empty/garbage) and
        # transient model crashes (LM Studio auto-reloads after crash).
        last_response: LLMResponse | None = None
        async with httpx.AsyncClient(timeout=300.0) as client:
            for attempt in range(_MAX_RETRIES + 1):
                if attempt > 0:
                    payload["temperature"] = min(temperature + 0.3 * attempt, 1.5)

                try:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                except (httpx.ReadError, httpx.ConnectError) as e:
                    if attempt < _MAX_RETRIES:
                        await _wait_for_model(client, self._base_url, attempt)
                        continue
                    raise RuntimeError(f"Gateway connection failed: {e}")

                if resp.status_code != 200:
                    try:
                        body = resp.json()
                        detail = body.get("error", body)
                    except Exception:
                        detail = resp.text[:500]
                    detail_str = str(detail).lower()

                    # If backend rejected `tools` param, fall back to
                    # text-injection and retry immediately.
                    if "tools" in payload and attempt < _MAX_RETRIES:
                        if any(k in detail_str for k in
                               ("tools", "unrecognized", "invalid")):
                            payload.pop("tools", None)
                            _inject_system(oai_messages,
                                           _tools_to_prompt(tools),
                                           guard="<function=")
                            continue

                    is_transient = any(k in detail_str for k in
                                       ("crashed", "reloaded", "loading",
                                        "not loaded", "channel error"))
                    if is_transient and attempt < _MAX_RETRIES:
                        await _wait_for_model(client, self._base_url, attempt)
                        continue
                    raise RuntimeError(
                        f"Gateway returned {resp.status_code}: {detail}"
                    )

                last_response = _parse_response(resp.json(), target_model)

                # Check for degenerate output
                if _is_degenerate(last_response.content):
                    if attempt < _MAX_RETRIES:
                        continue  # retry
                return last_response

        # All retries exhausted — return whatever we got
        return last_response  # type: ignore[return-value]


async def _wait_for_model(
    client: Any, base_url: str, attempt: int,
) -> None:
    """Wait for a local model to finish reloading after a crash.

    Uses an actual inference probe (not just /models) because the server
    can report models as available before they're ready for generation.
    """
    import asyncio
    wait = 15 + attempt * 10  # 15s, 25s, 35s, 45s — generous
    await asyncio.sleep(wait)
    # Probe with a tiny inference request until it works
    for _ in range(8):
        try:
            probe = await client.post(
                f"{base_url}/chat/completions",
                json={"model": "gemma-4-e4b-it",
                      "messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 5},
                timeout=15,
            )
            if probe.status_code == 200:
                await asyncio.sleep(2)
                return
        except Exception:
            pass
        await asyncio.sleep(5)


# ─────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────

def _parse_response(data: dict, target_model: str) -> LLMResponse:
    choice = data["choices"][0]
    msg = choice["message"]
    content = (msg.get("content") or "").strip()

    # Reasoning models put chain-of-thought in reasoning_content.
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not content and reasoning and not _is_degenerate(reasoning):
        content = reasoning

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


def _is_degenerate(text: str) -> bool:
    """Detect garbage output: empty, repeated chars, or special token spam."""
    if not text or len(text.strip()) < 5:
        return True
    # Special token spam: <unused42><tool|> etc.
    if "<unused" in text or text.count("<") > len(text) // 10:
        return True
    # >80% single repeated character (///... loops)
    from collections import Counter
    counts = Counter(text)
    if counts:
        _, top_count = counts.most_common(1)[0]
        if top_count / len(text) > 0.8:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────

def _inject_system(
    oai_messages: list[dict[str, Any]],
    block: str,
    guard: str,
) -> None:
    """Prepend *block* to the system message unless *guard* already present."""
    for msg in oai_messages:
        if msg["role"] == "system":
            if guard not in (msg["content"] or ""):
                msg["content"] = block + (msg["content"] or "")
            return
    oai_messages.insert(0, {"role": "system", "content": block})


def _tools_to_prompt(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI tool schemas to a compact text block for the system prompt.

    Kept deliberately short to minimize token usage on local models.
    """
    parts = ['Tools (call with <function=NAME({"key":"val"})></function>):\n']
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        params = ", ".join(f'{k}:{v.get("type", "str")}' for k, v in props.items())
        parts.append(f"- {name}({params}): {desc}\n")
    parts.append("Use tools when needed, then give your final answer.\n\n")
    return "".join(parts)


def _build_oai_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI-compatible format.

    Preserves system, tool, and assistant-with-tool-calls roles for
    backends that support them (Ollama, vLLM, LiteLLM).
    """
    result: list[dict[str, Any]] = []

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
