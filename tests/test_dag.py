"""Tests for the DAG engine — validation, readiness, mutation."""

import pytest

from daaw.engine.dag import DAG
from daaw.schemas.enums import TaskStatus
from daaw.schemas.workflow import (
    AgentSpec,
    DependencySpec,
    TaskSpec,
    WorkflowSpec,
)


def _make_spec(tasks):
    return WorkflowSpec(name="Test", description="test", tasks=tasks)


def _task(tid, deps=None):
    return TaskSpec(
        id=tid, name=tid, description=f"Task {tid}",
        agent=AgentSpec(role="generic_llm"),
        dependencies=[DependencySpec(task_id=d) for d in (deps or [])],
    )


# ── Validation ──


class TestDAGValidation:
    def test_valid_linear(self, sample_workflow_spec):
        dag = DAG(sample_workflow_spec)
        assert dag.validate() == []

    def test_valid_parallel(self, parallel_workflow_spec):
        dag = DAG(parallel_workflow_spec)
        assert dag.validate() == []

    def test_empty_dag(self):
        spec = _make_spec([])
        dag = DAG(spec)
        assert dag.validate() == []
        assert dag.is_complete()

    def test_cycle_two_nodes(self):
        spec = _make_spec([_task("a", ["b"]), _task("b", ["a"])])
        dag = DAG(spec)
        errors = dag.validate()
        assert any("Cycle" in e for e in errors)

    def test_cycle_three_nodes(self):
        spec = _make_spec([
            _task("a", ["c"]),
            _task("b", ["a"]),
            _task("c", ["b"]),
        ])
        dag = DAG(spec)
        errors = dag.validate()
        assert any("Cycle" in e for e in errors)

    def test_self_loop(self):
        spec = _make_spec([_task("a", ["a"])])
        dag = DAG(spec)
        errors = dag.validate()
        assert any("Cycle" in e for e in errors)

    def test_missing_dependency(self):
        spec = _make_spec([_task("a", ["nonexistent"])])
        dag = DAG(spec)
        errors = dag.validate()
        assert any("unknown task" in e for e in errors)

    def test_diamond_dag(self):
        """A -> B, A -> C, B -> D, C -> D (valid diamond)."""
        spec = _make_spec([
            _task("a"),
            _task("b", ["a"]),
            _task("c", ["a"]),
            _task("d", ["b", "c"]),
        ])
        dag = DAG(spec)
        assert dag.validate() == []


# ── Task Readiness ──


class TestDAGReadiness:
    def test_initial_ready_linear(self, sample_workflow_spec):
        dag = DAG(sample_workflow_spec)
        ready = dag.get_ready_tasks()
        assert ready == ["task_001"]

    def test_initial_ready_parallel(self, parallel_workflow_spec):
        dag = DAG(parallel_workflow_spec)
        ready = sorted(dag.get_ready_tasks())
        assert ready == ["p1", "p2", "p3"]

    def test_ready_after_completion(self):
        spec = _make_spec([_task("a"), _task("b", ["a"]), _task("c", ["a"])])
        dag = DAG(spec)

        assert dag.get_ready_tasks() == ["a"]

        dag.mark("a", TaskStatus.SUCCESS)
        ready = sorted(dag.get_ready_tasks())
        assert ready == ["b", "c"]

    def test_not_ready_if_dep_failed(self):
        spec = _make_spec([_task("a"), _task("b", ["a"])])
        dag = DAG(spec)
        dag.mark("a", TaskStatus.FAILURE)
        assert dag.get_ready_tasks() == []

    def test_not_ready_if_dep_running(self):
        spec = _make_spec([_task("a"), _task("b", ["a"])])
        dag = DAG(spec)
        dag.mark("a", TaskStatus.RUNNING)
        assert dag.get_ready_tasks() == []

    def test_diamond_readiness(self):
        spec = _make_spec([
            _task("a"),
            _task("b", ["a"]),
            _task("c", ["a"]),
            _task("d", ["b", "c"]),
        ])
        dag = DAG(spec)

        # Only a is ready
        assert dag.get_ready_tasks() == ["a"]

        # After a succeeds, b and c are ready
        dag.mark("a", TaskStatus.SUCCESS)
        assert sorted(dag.get_ready_tasks()) == ["b", "c"]

        # After b succeeds, d is NOT ready (c still pending)
        dag.mark("b", TaskStatus.SUCCESS)
        assert dag.get_ready_tasks() == ["c"]

        # After c succeeds, d is ready
        dag.mark("c", TaskStatus.SUCCESS)
        assert dag.get_ready_tasks() == ["d"]


# ── Status Tracking ──


class TestDAGStatus:
    def test_initial_all_pending(self, sample_workflow_spec):
        dag = DAG(sample_workflow_spec)
        statuses = dag.all_statuses()
        assert all(s == TaskStatus.PENDING for s in statuses.values())

    def test_mark_and_get(self):
        spec = _make_spec([_task("a")])
        dag = DAG(spec)
        dag.mark("a", TaskStatus.RUNNING)
        assert dag.get_status("a") == TaskStatus.RUNNING

    def test_mark_unknown_raises(self):
        spec = _make_spec([_task("a")])
        dag = DAG(spec)
        with pytest.raises(ValueError, match="Unknown task"):
            dag.mark("nonexistent", TaskStatus.SUCCESS)

    def test_is_complete(self):
        spec = _make_spec([_task("a"), _task("b")])
        dag = DAG(spec)
        assert not dag.is_complete()

        dag.mark("a", TaskStatus.SUCCESS)
        assert not dag.is_complete()

        dag.mark("b", TaskStatus.FAILURE)
        assert dag.is_complete()

    def test_has_failures(self):
        spec = _make_spec([_task("a"), _task("b")])
        dag = DAG(spec)
        assert not dag.has_failures()

        dag.mark("a", TaskStatus.FAILURE)
        assert dag.has_failures()


# ── Mutation (for Critic patches) ──


class TestDAGMutation:
    def test_add_task(self):
        spec = _make_spec([_task("a")])
        dag = DAG(spec)

        new_task = _task("b", ["a"])
        dag.add_task(new_task)

        dag.mark("a", TaskStatus.SUCCESS)
        assert "b" in dag.get_ready_tasks()

    def test_remove_task(self):
        spec = _make_spec([_task("a"), _task("b", ["a"])])
        dag = DAG(spec)
        dag.remove_task("a")
        assert dag.get_status("a") == TaskStatus.SKIPPED

        # b should now be ready since a is skipped and in-degree adjusted
        ready = dag.get_ready_tasks()
        assert "b" in ready

    def test_reset_task(self):
        spec = _make_spec([_task("a")])
        dag = DAG(spec)
        dag.mark("a", TaskStatus.FAILURE)
        assert dag.get_status("a") == TaskStatus.FAILURE

        dag.reset_task("a")
        assert dag.get_status("a") == TaskStatus.PENDING
        assert "a" in dag.get_ready_tasks()
