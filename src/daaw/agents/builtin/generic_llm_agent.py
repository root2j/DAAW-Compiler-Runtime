"""GenericLLMAgent — catch-all agent for planner-created tasks."""

from __future__ import annotations

import json
import json as _json
import re
import uuid
from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.llm.base import LLMMessage
from daaw.schemas.results import AgentResult
from daaw.tools.registry import tool_registry

DEFAULT_PROVIDER = "groq"
MAX_TOOL_ROUNDS = 10

DEFAULT_SYSTEM_PROMPT = (
    "You are a task execution agent. Complete the given task thoroughly and concisely. "
    "Use the provided tools when helpful. "
    "Return a concrete, structured result — do NOT ask clarifying questions or "
    "offer further assistance. Just do the task.\n"
    "If a task asks for specific values (e.g. dates, numbers) and none are provided, "
    "make reasonable concrete assumptions and state them clearly in your output."
)

# Patterns for text-based tool calls emitted by various local models.
# Group 1: tool name.  Group 2: raw args (may be JSON or keyword-style).
_TOOL_CALL_PATTERNS = [
    # Llama / gateway-injected: <function=web_search({"query": "..."})></function>
    re.compile(r'<function=(\w+)\((\{.*?\})\)>?</function>', re.DOTALL),
    # Qwen/Gemma with JSON args: <|tool_call>call:web_search({"query":"..."})
    re.compile(r'<\|tool_call>(?:call:)?(\w+)\((\{.*?\})\)', re.DOTALL),
    # Gemma E4B keyword: <|tool_call>call:web_search(query: "...")<tool_call|>
    re.compile(r'<\|tool_call>(?:call:)?(\w+)\(([^)]+)\)(?:<tool_call\|>)?', re.DOTALL),
    # Gemma E2B compact: <call:web_search("query text")> or <call:web_search(query)>
    re.compile(r'<call:(\w+)\(([^)]+)\)>', re.DOTALL),
    # [tool_call: web_search({"query":"..."})]
    re.compile(r'\[tool_call:\s*(\w+)\((\{.*?\})\)\]', re.DOTALL),
    # Bare function-call: web_search({"query": "..."}) at start of line
    re.compile(r'^(\w+)\((\{"[^"]+":.*?\})\)\s*$', re.MULTILINE),
]


def _parse_args(raw: str) -> dict:
    """Parse tool call arguments from JSON or keyword-style format.

    Handles:
      - JSON: {"query": "goa"}
      - Keyword: query: "goa beaches", limit: 5
      - Mixed: query="goa", limit=5
    """
    raw = raw.strip()
    # Try JSON first
    if raw.startswith("{"):
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass

    # Bare quoted string: "recent world events" → {"query": "recent world events"}
    # Common from Gemma E2B: <call:web_search("query text")>
    stripped = raw.strip('"\'')
    if stripped and not any(c in raw for c in ":={"):
        return {"query": stripped}

    # Keyword-style: key: "value", key: value, key="value"
    result = {}
    for m in re.finditer(r'(\w+)\s*[:=]\s*(?:"([^"]*?)"|(\S+))', raw):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else m.group(3)
        if val.isdigit():
            result[key] = int(val)
        elif val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
        else:
            result[key] = val
    return result


def _parse_text_tool_calls(content: str) -> list[dict] | None:
    """Detect and parse text-based tool calls from various LLM formats.

    Supports Llama XML, Qwen native, Gemma keyword-style, and bare
    function-call formats.  Returns parsed tool call dicts, or None.
    """
    results = []
    seen = set()
    for pattern in _TOOL_CALL_PATTERNS:
        for name, args_raw in pattern.findall(content):
            key = (name, args_raw)
            if key in seen:
                continue
            seen.add(key)
            args = _parse_args(args_raw)
            results.append({
                "name": name,
                "arguments": args,
                "id": f"text_tc_{uuid.uuid4().hex[:8]}",
            })
    return results or None


@register_agent("generic_llm")
class GenericLLMAgent(BaseAgent):
    """Generic LLM-powered agent with tool-call loop support."""

    async def run(self, task_input: Any) -> AgentResult:
        provider = self.config.get("provider", DEFAULT_PROVIDER)
        model = self.config.get("model")
        system_prompt = self.config.get("system_prompt_override", "")
        tools_allowed: list[str] = self.config.get("tools_allowed", [])

        messages: list[LLMMessage] = []
        # Always inject a system prompt — use override if provided, else default
        messages.append(LLMMessage(
            role="system",
            content=system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT,
        ))

        # Serialize dicts as proper JSON (not Python repr) so the LLM can parse them
        if isinstance(task_input, str):
            content = task_input
        elif isinstance(task_input, dict):
            content = _json.dumps(task_input, indent=2, default=str)
        else:
            content = str(task_input)
        messages.append(LLMMessage(role="user", content=content))

        # Get tool schemas for tools this agent is allowed to use
        tool_schemas = tool_registry.list_tools(
            allowed=tools_allowed if tools_allowed else None
        )

        tool_results_log: list[dict[str, Any]] = []

        for _ in range(MAX_TOOL_ROUNDS):
            # Stream only when (a) a token observer is attached AND
            # (b) the user's spec explicitly said no tools. Looking at
            # ``tool_schemas`` directly is wrong because an empty
            # ``tools_allowed`` means "no restriction" in the registry —
            # we'd stream exactly zero tasks. ``tools_allowed`` is the
            # user intent and does what we want.
            use_streaming = (
                self.on_token is not None and not tools_allowed
            )
            if use_streaming:
                content_accum: list[str] = []
                final_usage: dict[str, Any] = {}
                async for chunk in self.llm_client.chat_stream(
                    provider,
                    messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=4096,
                ):
                    if chunk.delta:
                        content_accum.append(chunk.delta)
                        try:
                            self.on_token(
                                chunk.delta, "".join(content_accum),
                            )
                        except Exception:
                            # Broken UI callback must never crash the agent.
                            pass
                    if chunk.done:
                        final_usage = chunk.usage
                full_content = "".join(content_accum)
                from daaw.llm.base import LLMResponse as _LLMR
                resp = _LLMR(content=full_content, model=model or "",
                             usage=final_usage, raw=None, tool_calls=[])
            else:
                try:
                    resp = await self.llm_client.chat(
                        provider,
                        messages,
                        model=model,
                        temperature=0.7,
                        max_tokens=4096,
                        tools=tool_schemas if tool_schemas else None,
                    )
                except Exception as chat_err:
                    err_str = str(chat_err).lower()
                    # Groq/OpenAI return 400 when the model generates a
                    # malformed tool call ("tool_use_failed", "Failed to
                    # call a function"). Retry once without the tools
                    # parameter so the model falls back to plain text.
                    if tool_schemas and any(k in err_str for k in (
                        "tool_use_failed", "failed to call a function",
                        "not in request.tools",
                    )):
                        resp = await self.llm_client.chat(
                            provider,
                            messages,
                            model=model,
                            temperature=0.7,
                            max_tokens=4096,
                            tools=None,  # retry without tools
                        )
                    else:
                        raise

            # Check for Llama-style text tool calls in content even if
            # no structured tool_calls were returned (Groq 400 prevention).
            text_tcs = _parse_text_tool_calls(resp.content) if not resp.tool_calls else None

            # No tool calls at all — LLM is done
            if not resp.tool_calls and not text_tcs:
                return AgentResult(
                    output=resp.content,
                    status="success",
                    metadata={"tool_calls": tool_results_log, "usage": resp.usage},
                )

            # Build a unified list of tool calls to execute
            if resp.tool_calls:
                active_tool_calls = [
                    {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
                    for tc in resp.tool_calls
                ]
                # Append assistant msg with structured tool calls for round-trip
                messages.append(LLMMessage(
                    role="assistant",
                    content=resp.content,
                    tool_calls_raw=_extract_raw_tool_calls(resp),
                ))
            else:
                # Text tool calls: strip all tool-call patterns from content,
                # then append a clean assistant message.
                clean_content = resp.content
                for pat in _TOOL_CALL_PATTERNS:
                    clean_content = pat.sub("", clean_content)
                clean_content = clean_content.strip()
                active_tool_calls = text_tcs  # type: ignore[assignment]
                messages.append(LLMMessage(
                    role="assistant",
                    content=clean_content or "(tool call)",
                ))

            # Execute each tool call and append results.
            # Each item is either a ToolCall object or a plain dict from text parsing.
            for tc in active_tool_calls:  # type: ignore[assignment]
                if isinstance(tc, dict):
                    tool_name = tc["name"]
                    tc_args = tc["arguments"]
                    tc_id = tc["id"]
                else:
                    tool_name = tc.name
                    tc_args = tc.arguments
                    tc_id = tc.id

                # Case-insensitive tool name resolution: LLMs hallucinate
                # capitalization (Claude calls "WebSearch" instead of
                # "web_search", Groq calls "brave_search"). Try exact
                # match first, then lowercase, then known aliases.
                resolved_name = tool_name
                all_tools = {t.name for t in tool_registry._tools.values()}
                if tool_name not in all_tools:
                    # Try case-insensitive match.
                    lower_map = {t.lower(): t for t in all_tools}
                    if tool_name.lower() in lower_map:
                        resolved_name = lower_map[tool_name.lower()]

                if tools_allowed and resolved_name not in tools_allowed:
                    # Also check case-insensitive against allowed list.
                    allowed_lower = {t.lower() for t in tools_allowed}
                    if resolved_name.lower() not in allowed_lower:
                        result_str = f"Tool '{tool_name}' is not allowed for this agent."
                    else:
                        try:
                            result = await tool_registry.execute(resolved_name, **tc_args)
                            result_str = str(result)
                        except Exception as e:
                            result_str = f"Tool execution error: {e}"
                else:
                    try:
                        result = await tool_registry.execute(resolved_name, **tc_args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"Tool execution error: {e}"

                tool_results_log.append({
                    "tool": tool_name,
                    "args": tc_args,
                    "result": result_str[:2000],
                })
                messages.append(LLMMessage(
                    role="tool",
                    content=result_str,
                    tool_call_id=tc_id,
                ))

        # Exhausted tool rounds — salvage whatever the agent accumulated
        # rather than discarding 10+ rounds of useful tool results.
        # Concatenate the last assistant content + last tool result as the
        # output so the downstream task and critic can still use the data.
        salvaged_parts: list[str] = []
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                salvaged_parts.insert(0, msg.content.strip())
                break
        # Include the last 3 tool results (most recent, most relevant).
        recent_tool_results = [
            tr["result"] for tr in tool_results_log[-3:]
            if tr.get("result")
        ]
        if recent_tool_results:
            salvaged_parts.append(
                "Tool results (last 3):\n" + "\n---\n".join(recent_tool_results)
            )
        salvaged = "\n\n".join(salvaged_parts) if salvaged_parts else None

        if salvaged and len(salvaged) > 50:
            # Enough content to be useful — return as partial success.
            return AgentResult(
                output=salvaged,
                status="success",
                metadata={"tool_calls": tool_results_log,
                           "note": f"Partial: tool loop hit {MAX_TOOL_ROUNDS} rounds"},
            )

        return AgentResult(
            output=salvaged,
            status="failure",
            error_message=(
                f"Max tool call rounds ({MAX_TOOL_ROUNDS}) reached without a final response. "
                "The agent may be stuck in a tool loop or calling unregistered tools."
            ),
            metadata={"tool_calls": tool_results_log},
        )


def _extract_raw_tool_calls(resp: Any) -> Any:
    """Extract raw tool_calls from the provider response for round-trip serialization."""
    raw = resp.raw
    if raw is None:
        return None
    # OpenAI / Groq format
    if hasattr(raw, "choices") and raw.choices:
        msg = raw.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls
    # Anthropic format — return content blocks
    if hasattr(raw, "content"):
        blocks = []
        for block in raw.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
        if blocks:
            return blocks
    return None
