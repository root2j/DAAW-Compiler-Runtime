"""Tests for the DAG executor — uses a mock agent to avoid real LLM calls."""

import asyncio
import tempfile
import time

import pytest

from daaw.agents.base import BaseAgent
from daaw.agents.factory import AgentFactory
from daaw.agents.registry import AGENT_REGISTRY
from daaw.config import AppConfig
from daaw.engine.circuit_breaker import CircuitBreaker
from daaw.engine.executor import DAGExecutor
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.results import AgentResult
from daaw.schemas.workflow import (
    AgentSpec,
    DependencySpec,
    TaskSpec,
    WorkflowSpec,
)
from daaw.store.artifact_store import ArtifactStore


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Test agent that doesn't call any LLM ──

class _EchoAgent(BaseAgent):
    """Returns a fixed response — no LLM needed."""
    async def run(self, task_input):
        return AgentResult(output=f"echo: {task_input}", status="success")


class _FailAgent(BaseAgent):
    """Always fails."""
    async def run(self, task_input):
        return AgentResult(status="failure", error_message="intentional failure")


class _SlowAgent(BaseAgent):
    """Takes a configurable amount of time."""
    async def run(self, task_input):
        await asyncio.sleep(0.3)
        return AgentResult(output="slow_done", status="success")


# Temporarily register test agents
_ORIGINAL_REGISTRY = None


def setup_module():
    global _ORIGINAL_REGISTRY
    _ORIGINAL_REGISTRY = dict(AGENT_REGISTRY)
    AGENT_REGISTRY["_echo"] = _EchoAgent
    AGENT_REGISTRY["_fail"] = _FailAgent
    AGENT_REGISTRY["_slow"] = _SlowAgent


def teardown_module():
    for key in ["_echo", "_fail", "_slow"]:
        AGENT_REGISTRY.pop(key, None)


def _make_executor():
    config = AppConfig()
    llm = UnifiedLLMClient(config)
    store = ArtifactStore(tempfile.mkdtemp())
    cb = CircuitBreaker(threshold=3)
    factory = AgentFactory(llm, store)
    return DAGExecutor(factory, store, cb), store


class TestExecutorLinear:
    def test_single_task(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Single", description="one task",
            tasks=[
                TaskSpec(id="t1", name="Echo", description="d", agent=AgentSpec(role="_echo")),
            ],
        )
        results = run(executor.execute(spec))
        assert "t1" in results
        assert results["t1"].agent_result.status == "success"
        assert "echo:" in results["t1"].agent_result.output

    def test_two_sequential_tasks(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Linear", description="two tasks",
            tasks=[
                TaskSpec(id="t1", name="First", description="d", agent=AgentSpec(role="_echo")),
                TaskSpec(
                    id="t2", name="Second", description="d",
                    agent=AgentSpec(role="_echo"),
                    dependencies=[DependencySpec(task_id="t1")],
                ),
            ],
        )
        results = run(executor.execute(spec))
        assert results["t1"].agent_result.status == "success"
        assert results["t2"].agent_result.status == "success"

    def test_output_stored_in_artifact_store(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Store", description="d",
            tasks=[
                TaskSpec(id="t1", name="Echo", description="d", agent=AgentSpec(role="_echo")),
            ],
        )
        run(executor.execute(spec))
        output = run(store.get("t1.output"))
        assert "echo:" in output
        assert run(store.get("t1.status")) == "success"


class TestExecutorParallel:
    def test_three_parallel_tasks(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Parallel", description="3 independent",
            tasks=[
                TaskSpec(id="p1", name="A", description="d", agent=AgentSpec(role="_slow")),
                TaskSpec(id="p2", name="B", description="d", agent=AgentSpec(role="_slow")),
                TaskSpec(id="p3", name="C", description="d", agent=AgentSpec(role="_slow")),
            ],
        )
        start = time.monotonic()
        results = run(executor.execute(spec))
        elapsed = time.monotonic() - start

        # All should succeed
        assert all(r.agent_result.status == "success" for r in results.values())
        # If parallel, total time should be ~0.3s not ~0.9s
        assert elapsed < 0.8, f"Expected parallel execution, took {elapsed:.2f}s"


class TestExecutorFailures:
    def test_failure_blocks_downstream(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Fail", description="d",
            tasks=[
                TaskSpec(id="t1", name="Fail", description="d", agent=AgentSpec(role="_fail")),
                TaskSpec(
                    id="t2", name="Blocked", description="d",
                    agent=AgentSpec(role="_echo"),
                    dependencies=[DependencySpec(task_id="t1")],
                ),
            ],
        )
        results = run(executor.execute(spec))
        assert results["t1"].agent_result.status == "failure"
        # t2 should not have run
        assert "t2" not in results

    def test_failure_doesnt_block_independent(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Independent", description="d",
            tasks=[
                TaskSpec(id="t1", name="Fail", description="d", agent=AgentSpec(role="_fail")),
                TaskSpec(id="t2", name="Echo", description="d", agent=AgentSpec(role="_echo")),
            ],
        )
        results = run(executor.execute(spec))
        assert results["t1"].agent_result.status == "failure"
        assert results["t2"].agent_result.status == "success"


class TestExecutorEmptyDAG:
    def test_empty_workflow(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(name="Empty", description="d", tasks=[])
        results = run(executor.execute(spec))
        assert results == {}


class _VerySlowAgent(BaseAgent):
    """Sleeps for 3 seconds — used to test timeouts."""
    async def run(self, task_input):
        await asyncio.sleep(3)
        return AgentResult(output="should_not_reach", status="success")


class TestExecutorTimeout:
    def test_task_timeout(self):
        AGENT_REGISTRY["_very_slow"] = _VerySlowAgent
        try:
            executor, store = _make_executor()
            spec = WorkflowSpec(
                name="Timeout", description="d",
                tasks=[
                    TaskSpec(
                        id="t1", name="Slow", description="d",
                        agent=AgentSpec(role="_very_slow"),
                        timeout_seconds=1,  # shorter than the 3s sleep
                    ),
                ],
            )
            results = run(executor.execute(spec))
            assert results["t1"].agent_result.status == "failure"
            assert "timed out" in results["t1"].agent_result.error_message.lower()
        finally:
            AGENT_REGISTRY.pop("_very_slow", None)


class TestExecutorValidation:
    def test_invalid_dag_raises(self):
        executor, store = _make_executor()
        spec = WorkflowSpec(
            name="Cyclic", description="d",
            tasks=[
                TaskSpec(
                    id="a", name="A", description="d",
                    agent=AgentSpec(role="_echo"),
                    dependencies=[DependencySpec(task_id="b")],
                ),
                TaskSpec(
                    id="b", name="B", description="d",
                    agent=AgentSpec(role="_echo"),
                    dependencies=[DependencySpec(task_id="a")],
                ),
            ],
        )
        with pytest.raises(ValueError, match="Invalid workflow DAG"):
            run(executor.execute(spec))
