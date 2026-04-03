"""Tests for GenericLLMAgent tool-call loop — unit tests with mocked LLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from daaw.agents.builtin.generic_llm_agent import GenericLLMAgent
from daaw.llm.base import LLMMessage, LLMResponse, ToolCall
from daaw.schemas.results import AgentResult
from daaw.store.artifact_store import ArtifactStore
from daaw.tools.registry import ToolRegistry


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tool_reg():
    """Fresh tool registry with test tools."""
    registry = ToolRegistry()

    @registry.register("add", "Add two numbers", {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    })
    async def add(a: int, b: int) -> str:
        return str(a + b)

    @registry.register("greet", "Greet someone", {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    })
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    return registry


@pytest.fixture
def mock_llm():
    """Mock UnifiedLLMClient."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(persist_dir=str(tmp_path))


def make_agent(llm, store, config=None, tool_reg=None):
    """Create a GenericLLMAgent with optional config and tool registry."""
    import daaw.agents.builtin.generic_llm_agent as mod
    if tool_reg:
        original_registry = mod.tool_registry
        mod.tool_registry = tool_reg

    agent = GenericLLMAgent(
        agent_id="test-agent",
        llm_client=llm,
        store=store,
        config=config or {},
    )

    if tool_reg:
        # Restore after creation — the agent captures the module-level ref at call time
        # So we need to keep it patched during run(). We'll restore in teardown.
        pass

    return agent


class TestNoToolCalls:
    def test_simple_response(self, mock_llm, store):
        """Agent returns LLM response when no tools are called."""
        mock_llm.chat.return_value = LLMResponse(
            content="The answer is 42.",
            model="test-model",
            tool_calls=[],
        )
        agent = make_agent(mock_llm, store)
        result = run(agent.run("What is the meaning of life?"))
        assert isinstance(result, AgentResult)
        assert result.output == "The answer is 42."
        assert result.status == "success"

    def test_system_prompt_override(self, mock_llm, store):
        """Agent uses system_prompt_override from config."""
        mock_llm.chat.return_value = LLMResponse(
            content="response", model="m", tool_calls=[],
        )
        agent = make_agent(mock_llm, store, config={
            "system_prompt_override": "You are a pirate.",
        })
        run(agent.run("Hello"))
        call_args = mock_llm.chat.call_args
        messages = call_args[0][1]  # positional arg: messages
        assert any(m.role == "system" and "pirate" in m.content for m in messages)


class TestToolCallLoop:
    def test_single_tool_call(self, mock_llm, store, tool_reg, monkeypatch):
        """Agent executes a tool and feeds result back to LLM."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", tool_reg)

        # First call: LLM requests tool
        tool_response = LLMResponse(
            content="",
            model="test",
            tool_calls=[ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3})],
            raw=None,
        )
        # Second call: LLM gives final answer
        final_response = LLMResponse(
            content="The sum is 5.",
            model="test",
            tool_calls=[],
        )
        mock_llm.chat.side_effect = [tool_response, final_response]

        agent = make_agent(mock_llm, store, config={"tools_allowed": ["add"]})
        result = run(agent.run("What is 2 + 3?"))

        assert result.output == "The sum is 5."
        assert result.status == "success"
        assert len(result.metadata["tool_calls"]) == 1
        assert result.metadata["tool_calls"][0]["tool"] == "add"
        assert "5" in result.metadata["tool_calls"][0]["result"]

    def test_multiple_tool_rounds(self, mock_llm, store, tool_reg, monkeypatch):
        """Agent handles multiple rounds of tool calls."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", tool_reg)

        responses = [
            # Round 1: call add
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]),
            # Round 2: call greet
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c2", name="greet", arguments={"name": "World"})]),
            # Round 3: final answer
            LLMResponse(content="Done! 3 and Hello, World!", model="t", tool_calls=[]),
        ]
        mock_llm.chat.side_effect = responses

        agent = make_agent(mock_llm, store, config={"tools_allowed": ["add", "greet"]})
        result = run(agent.run("Add 1+2 then greet"))

        assert result.status == "success"
        assert len(result.metadata["tool_calls"]) == 2
        assert mock_llm.chat.call_count == 3

    def test_disallowed_tool_rejected(self, mock_llm, store, tool_reg, monkeypatch):
        """Agent rejects tool calls not in tools_allowed."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", tool_reg)

        responses = [
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="greet", arguments={"name": "X"})]),
            LLMResponse(content="I can't greet, only add.", model="t", tool_calls=[]),
        ]
        mock_llm.chat.side_effect = responses

        agent = make_agent(mock_llm, store, config={"tools_allowed": ["add"]})
        result = run(agent.run("Greet someone"))

        assert result.status == "success"
        assert "not allowed" in result.metadata["tool_calls"][0]["result"]

    def test_tool_execution_error(self, mock_llm, store, tool_reg, monkeypatch):
        """Agent handles tool execution errors gracefully."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", tool_reg)

        responses = [
            # LLM calls a tool with wrong args
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="add", arguments={"a": "not_a_number", "b": 3})]),
            LLMResponse(content="Error occurred.", model="t", tool_calls=[]),
        ]
        mock_llm.chat.side_effect = responses

        agent = make_agent(mock_llm, store)
        result = run(agent.run("Add broken things"))

        assert result.status == "success"
        # The tool call log should contain the error
        assert len(result.metadata["tool_calls"]) == 1

    def test_max_rounds_exhausted(self, mock_llm, store, tool_reg, monkeypatch):
        """Agent stops after MAX_TOOL_ROUNDS."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", tool_reg)
        monkeypatch.setattr(mod, "MAX_TOOL_ROUNDS", 3)

        # LLM always requests tools, never gives final answer
        infinite_tool = LLMResponse(
            content="still thinking", model="t", raw=None,
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1})],
        )
        mock_llm.chat.side_effect = [infinite_tool] * 3

        agent = make_agent(mock_llm, store)
        result = run(agent.run("infinite loop"))

        # Exhaustion now returns status=failure with error_message (output is None)
        assert result.status == "failure"
        assert result.output is None
        assert "Max tool call rounds" in result.error_message
        assert mock_llm.chat.call_count == 3


class TestProviderRouting:
    def test_uses_provider_from_config(self, mock_llm, store):
        """Agent routes to the provider specified in config."""
        mock_llm.chat.return_value = LLMResponse(
            content="ok", model="m", tool_calls=[],
        )
        agent = make_agent(mock_llm, store, config={"provider": "anthropic"})
        run(agent.run("test"))
        call_args = mock_llm.chat.call_args
        assert call_args[0][0] == "anthropic"

    def test_defaults_to_groq(self, mock_llm, store):
        """Agent defaults to groq provider."""
        mock_llm.chat.return_value = LLMResponse(
            content="ok", model="m", tool_calls=[],
        )
        agent = make_agent(mock_llm, store)
        run(agent.run("test"))
        call_args = mock_llm.chat.call_args
        assert call_args[0][0] == "groq"
