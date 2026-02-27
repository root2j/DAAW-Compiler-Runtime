"""Tests for Pydantic schemas — enums, workflow, results, events."""

import uuid

import pytest

from daaw.schemas.enums import AgentRole, PatchAction, TaskStatus
from daaw.schemas.events import InteractionEvent, PatchOperation, WorkflowPatch
from daaw.schemas.results import AgentResult, TaskResult
from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec, WorkflowSpec


# ── Enums ──


class TestEnums:
    def test_task_status_values(self):
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.FAILURE == "failure"
        assert TaskStatus.SKIPPED == "skipped"
        assert TaskStatus.NEEDS_HUMAN == "needs_human"
        assert TaskStatus.RETRYING == "retrying"

    def test_agent_role_values(self):
        assert AgentRole.PLANNER == "planner"
        assert AgentRole.PM == "pm"
        assert AgentRole.BREAKDOWN == "breakdown"
        assert AgentRole.CRITIC == "critic"
        assert AgentRole.USER_PROXY == "user_proxy"
        assert AgentRole.GENERIC_LLM == "generic_llm"

    def test_patch_action_values(self):
        assert PatchAction.RETRY == "retry"
        assert PatchAction.INSERT == "insert"
        assert PatchAction.REMOVE == "remove"
        assert PatchAction.UPDATE_INPUT == "update_input"

    def test_enum_string_comparison(self):
        """Enums inherit from str so they compare directly with strings."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus("success") == TaskStatus.SUCCESS


# ── AgentSpec ──


class TestAgentSpec:
    def test_minimal(self):
        spec = AgentSpec(role="pm")
        assert spec.role == "pm"
        assert spec.tools_allowed == []
        assert spec.model_config_override is None
        assert spec.system_prompt_override is None

    def test_full(self):
        spec = AgentSpec(
            role="generic_llm",
            model_config_override={"provider": "gemini"},
            tools_allowed=["web_search", "file_read"],
            system_prompt_override="You are a helper.",
        )
        assert spec.tools_allowed == ["web_search", "file_read"]
        assert spec.model_config_override["provider"] == "gemini"


# ── DependencySpec ──


class TestDependencySpec:
    def test_basic(self):
        dep = DependencySpec(task_id="task_001")
        assert dep.task_id == "task_001"
        assert dep.output_key is None

    def test_with_output_key(self):
        dep = DependencySpec(task_id="task_001", output_key="summary")
        assert dep.output_key == "summary"


# ── TaskSpec ──


class TestTaskSpec:
    def test_defaults(self):
        task = TaskSpec(
            id="t1", name="Test", description="A test task",
            agent=AgentSpec(role="pm"),
        )
        assert task.timeout_seconds == 300
        assert task.max_retries == 2
        assert task.dependencies == []
        assert task.input_filter == []
        assert task.success_criteria == ""

    def test_full_task(self):
        task = TaskSpec(
            id="t1", name="Test", description="desc",
            agent=AgentSpec(role="breakdown"),
            dependencies=[DependencySpec(task_id="t0")],
            input_filter=["t0.output"],
            success_criteria="Non-empty output",
            timeout_seconds=60,
            max_retries=5,
        )
        assert task.timeout_seconds == 60
        assert task.max_retries == 5
        assert len(task.dependencies) == 1


# ── WorkflowSpec ──


class TestWorkflowSpec:
    def test_auto_id(self):
        spec = WorkflowSpec(name="W", description="d", tasks=[])
        # Should be a valid UUID
        uuid.UUID(spec.id)

    def test_get_task(self):
        task = TaskSpec(id="t1", name="T", description="d", agent=AgentSpec(role="pm"))
        spec = WorkflowSpec(name="W", description="d", tasks=[task])
        assert spec.get_task("t1") is task
        assert spec.get_task("nonexistent") is None

    def test_task_ids(self):
        tasks = [
            TaskSpec(id=f"t{i}", name=f"T{i}", description="d", agent=AgentSpec(role="pm"))
            for i in range(3)
        ]
        spec = WorkflowSpec(name="W", description="d", tasks=tasks)
        assert spec.task_ids() == ["t0", "t1", "t2"]

    def test_dependency_graph(self):
        tasks = [
            TaskSpec(id="a", name="A", description="d", agent=AgentSpec(role="pm")),
            TaskSpec(
                id="b", name="B", description="d", agent=AgentSpec(role="pm"),
                dependencies=[DependencySpec(task_id="a")],
            ),
            TaskSpec(
                id="c", name="C", description="d", agent=AgentSpec(role="pm"),
                dependencies=[DependencySpec(task_id="a"), DependencySpec(task_id="b")],
            ),
        ]
        spec = WorkflowSpec(name="W", description="d", tasks=tasks)
        graph = spec.dependency_graph()
        assert graph == {"a": [], "b": ["a"], "c": ["a", "b"]}

    def test_json_roundtrip(self):
        task = TaskSpec(id="t1", name="T", description="d", agent=AgentSpec(role="pm"))
        spec = WorkflowSpec(id="fixed-id", name="W", description="d", tasks=[task])
        json_str = spec.model_dump_json()
        restored = WorkflowSpec.model_validate_json(json_str)
        assert restored.id == "fixed-id"
        assert restored.tasks[0].id == "t1"


# ── AgentResult ──


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult()
        assert r.output is None
        assert r.status == "success"
        assert r.metadata == {}
        assert r.error_message == ""

    def test_failure(self):
        r = AgentResult(status="failure", error_message="boom")
        assert r.status == "failure"
        assert r.error_message == "boom"

    def test_arbitrary_output(self):
        r = AgentResult(output={"key": [1, 2, 3]})
        assert r.output["key"] == [1, 2, 3]


# ── TaskResult ──


class TestTaskResult:
    def test_basic(self):
        ar = AgentResult(output="hello", status="success")
        tr = TaskResult(task_id="t1", agent_result=ar, attempt=2, elapsed_seconds=1.5)
        assert tr.task_id == "t1"
        assert tr.attempt == 2
        assert tr.elapsed_seconds == 1.5


# ── Events ──


class TestEvents:
    def test_interaction_event(self):
        e = InteractionEvent(role="user", content="hello")
        assert e.role == "user"
        assert e.timestamp is not None

    def test_patch_operation(self):
        op = PatchOperation(
            action=PatchAction.RETRY,
            target_task_id="t1",
            feedback="Try again with more detail",
        )
        assert op.action == PatchAction.RETRY

    def test_workflow_patch(self):
        patch = WorkflowPatch(
            operations=[
                PatchOperation(action=PatchAction.RETRY, target_task_id="t1"),
                PatchOperation(action=PatchAction.REMOVE, target_task_id="t2"),
            ],
            reasoning="t1 needs retry, t2 is redundant",
        )
        assert len(patch.operations) == 2
