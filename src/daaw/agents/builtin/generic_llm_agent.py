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

# Regex to detect Llama-style text tool calls: <function=name({...})></function>
_TEXT_TOOL_CALL_RE = re.compile(
    r'<function=(?P<name>\w+)\((?P<args>\{.*?\})\)</function>',
    re.DOTALL,
)


def _parse_text_tool_calls(content: str) -> list[dict] | None:
    """Detect and parse Llama XML-style tool calls emitted as plain text.

    Returns a list of parsed tool call dicts, or None if none found.
    """
    matches = _TEXT_TOOL_CALL_RE.findall(content)
    if not matches:
        return None
    result = []
    for name, args_str in matches:
        try:
            args = _json.loads(args_str)
        except _json.JSONDecodeError:
            args = {}
        result.append({"name": name, "arguments": args, "id": f"text_tc_{uuid.uuid4().hex[:8]}"})

    return result or None


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
            resp = await self.llm_client.chat(
                provider,
                messages,
                model=model,
                temperature=0.7,
                max_tokens=4096,
                tools=tool_schemas if tool_schemas else None,
            )

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
                # Text tool calls: strip the XML from content to avoid Groq 400,
                # then append a clean assistant message.
                clean_content = _TEXT_TOOL_CALL_RE.sub("", resp.content).strip()
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

                if tools_allowed and tool_name not in tools_allowed:
                    result_str = f"Tool '{tool_name}' is not allowed for this agent."
                else:
                    try:
                        result = await tool_registry.execute(tool_name, **tc_args)
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

        # Exhausted tool rounds — treat as failure so the Critic can retry
        return AgentResult(
            output=None,
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
