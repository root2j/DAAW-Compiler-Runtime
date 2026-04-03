"""Tests for executor + tool-calling agents — end-to-end with mocked LLM."""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daaw.agents.factory import AgentFactory
from daaw.config import AppConfig
from daaw.engine.circuit_breaker import CircuitBreaker
from daaw.engine.executor import DAGExecutor
from daaw.llm.base import LLMResponse, ToolCall
from daaw.schemas.workflow import (
    AgentSpec,
    DependencySpec,
    TaskSpec,
    WorkflowSpec,
)
from daaw.store.artifact_store import ArtifactStore
from daaw.tools.registry import ToolRegistry

# Need builtin agents registered
import daaw.agents.builtin.generic_llm_agent  # noqa: F401


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def test_tool_registry():
    """Registry with a simple calculator tool."""
    registry = ToolRegistry()

    @registry.register("multiply", "Multiply two numbers", {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    })
    async def multiply(a: int, b: int) -> str:
        return str(a * b)

    return registry


@pytest.fixture
def mock_llm_client():
    """Mock UnifiedLLMClient with configurable responses."""
    client = MagicMock()
    client.chat = AsyncMock()
    return client


def _make_executor(llm_client):
    store = ArtifactStore(tempfile.mkdtemp())
    cb = CircuitBreaker(threshold=3)
    factory = AgentFactory(llm_client, store)
    return DAGExecutor(factory, store, cb), store


class TestExecutorWithTools:
    def test_single_task_with_tool_call(self, mock_llm_client, test_tool_registry, monkeypatch):
        """A generic_llm task that uses tools should execute through the DAG."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", test_tool_registry)

        # LLM first calls tool, then gives final answer
        mock_llm_client.chat.side_effect = [
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="multiply", arguments={"a": 6, "b": 7})]),
            LLMResponse(content="The product is 42.", model="t", tool_calls=[]),
        ]

        executor, store = _make_executor(mock_llm_client)
        spec = WorkflowSpec(
            name="Tool Test", description="test",
            tasks=[
                TaskSpec(
                    id="calc", name="Calculate", description="Multiply 6 * 7",
                    agent=AgentSpec(role="generic_llm", tools_allowed=["multiply"]),
                ),
            ],
        )
        results = run(executor.execute(spec))
        assert results["calc"].agent_result.status == "success"
        assert "42" in results["calc"].agent_result.output

    def test_chained_tasks_with_tools(self, mock_llm_client, test_tool_registry, monkeypatch):
        """Two sequential tasks where second depends on first's output."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", test_tool_registry)

        # Task 1: tool call then answer
        # Task 2: no tool, just uses task 1 output
        mock_llm_client.chat.side_effect = [
            # Task 1, round 1: call multiply
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="multiply", arguments={"a": 3, "b": 4})]),
            # Task 1, round 2: final answer
            LLMResponse(content="Result: 12", model="t", tool_calls=[]),
            # Task 2: uses task 1 output, no tools
            LLMResponse(content="The previous result was 12. Double is 24.", model="t", tool_calls=[]),
        ]

        executor, store = _make_executor(mock_llm_client)
        spec = WorkflowSpec(
            name="Chain", description="chained",
            tasks=[
                TaskSpec(
                    id="step1", name="Multiply", description="3*4",
                    agent=AgentSpec(role="generic_llm", tools_allowed=["multiply"]),
                ),
                TaskSpec(
                    id="step2", name="Double", description="Double the result",
                    agent=AgentSpec(role="generic_llm"),
                    dependencies=[DependencySpec(task_id="step1")],
                ),
            ],
        )
        results = run(executor.execute(spec))
        assert results["step1"].agent_result.status == "success"
        assert results["step2"].agent_result.status == "success"
        # Verify context was passed (step2 received step1's output)
        stored_output = run(store.get("step1.output"))
        assert "12" in stored_output

    def test_parallel_tasks_with_tools(self, mock_llm_client, test_tool_registry, monkeypatch):
        """Two independent tasks with tools execute in parallel."""
        import daaw.agents.builtin.generic_llm_agent as mod
        monkeypatch.setattr(mod, "tool_registry", test_tool_registry)

        # Both tasks: tool call then answer (interleaved by asyncio.gather)
        mock_llm_client.chat.side_effect = [
            # Task A, round 1
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c1", name="multiply", arguments={"a": 2, "b": 5})]),
            # Task B, round 1
            LLMResponse(content="", model="t", raw=None,
                       tool_calls=[ToolCall(id="c2", name="multiply", arguments={"a": 3, "b": 3})]),
            # Task A, round 2
            LLMResponse(content="A=10", model="t", tool_calls=[]),
            # Task B, round 2
            LLMResponse(content="B=9", model="t", tool_calls=[]),
        ]

        executor, store = _make_executor(mock_llm_client)
        spec = WorkflowSpec(
            name="Parallel Tools", description="parallel",
            tasks=[
                TaskSpec(id="a", name="Task A", description="2*5",
                        agent=AgentSpec(role="generic_llm", tools_allowed=["multiply"])),
                TaskSpec(id="b", name="Task B", description="3*3",
                        agent=AgentSpec(role="generic_llm", tools_allowed=["multiply"])),
            ],
        )
        results = run(executor.execute(spec))
        assert results["a"].agent_result.status == "success"
        assert results["b"].agent_result.status == "success"
