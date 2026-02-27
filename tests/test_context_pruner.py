"""Tests for the context pruner."""

import asyncio
import tempfile

import pytest

from daaw.engine.context_pruner import prune_context
from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec
from daaw.store.artifact_store import ArtifactStore


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestContextPruner:
    def test_no_deps_no_filter(self):
        """Task with no dependencies and no filter returns empty dict."""
        store = ArtifactStore(tempfile.mkdtemp())
        task = TaskSpec(
            id="t1", name="T", description="d", agent=AgentSpec(role="pm"),
        )
        result = run(prune_context(task, store))
        assert result == {}

    def test_gathers_dependency_outputs(self):
        """Without input_filter, pruner gathers {dep_id}.output for each dep."""
        store = ArtifactStore(tempfile.mkdtemp())
        run(store.put("task_001.output", "result from task 1"))
        run(store.put("task_002.output", "result from task 2"))

        task = TaskSpec(
            id="t3", name="T3", description="d",
            agent=AgentSpec(role="pm"),
            dependencies=[
                DependencySpec(task_id="task_001"),
                DependencySpec(task_id="task_002"),
            ],
        )
        result = run(prune_context(task, store))
        assert result == {
            "task_001.output": "result from task 1",
            "task_002.output": "result from task 2",
        }

    def test_input_filter_overrides_deps(self):
        """If input_filter is set, it takes priority over dependency-based gathering."""
        store = ArtifactStore(tempfile.mkdtemp())
        run(store.put("custom.key1", "val1"))
        run(store.put("custom.key2", "val2"))
        run(store.put("task_001.output", "should be ignored"))

        task = TaskSpec(
            id="t2", name="T2", description="d",
            agent=AgentSpec(role="pm"),
            dependencies=[DependencySpec(task_id="task_001")],
            input_filter=["custom.key1", "custom.key2"],
        )
        result = run(prune_context(task, store))
        assert result == {"custom.key1": "val1", "custom.key2": "val2"}

    def test_missing_dep_output_skipped(self):
        """If a dependency hasn't stored output yet, it's simply absent."""
        store = ArtifactStore(tempfile.mkdtemp())
        run(store.put("task_001.output", "exists"))

        task = TaskSpec(
            id="t2", name="T2", description="d",
            agent=AgentSpec(role="pm"),
            dependencies=[
                DependencySpec(task_id="task_001"),
                DependencySpec(task_id="task_missing"),
            ],
        )
        result = run(prune_context(task, store))
        assert result == {"task_001.output": "exists"}
