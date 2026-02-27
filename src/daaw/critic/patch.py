"""Apply WorkflowPatch operations to a live DAG."""

from __future__ import annotations

from daaw.engine.dag import DAG
from daaw.engine.executor import DAGExecutor
from daaw.schemas.enums import PatchAction, TaskStatus
from daaw.schemas.events import WorkflowPatch
from daaw.schemas.workflow import TaskSpec
from daaw.store.artifact_store import ArtifactStore


async def apply_patch(
    patch: WorkflowPatch,
    dag: DAG,
    executor: DAGExecutor,
    store: ArtifactStore,
) -> list[str]:
    """Apply patch operations to the DAG. Returns log messages."""
    logs: list[str] = []

    for op in patch.operations:
        if op.action == PatchAction.RETRY:
            logs.append(f"RETRY {op.target_task_id}: {op.feedback}")
            await executor.retry_task(dag, op.target_task_id, op.feedback)

        elif op.action == PatchAction.INSERT:
            if op.new_task:
                new_task = TaskSpec.model_validate(op.new_task)
                dag.add_task(new_task)
                logs.append(f"INSERT {new_task.id}: {new_task.name}")
            else:
                logs.append(f"INSERT {op.target_task_id}: skipped (no task spec)")

        elif op.action == PatchAction.REMOVE:
            dag.remove_task(op.target_task_id)
            logs.append(f"REMOVE {op.target_task_id}: {op.feedback}")

        elif op.action == PatchAction.UPDATE_INPUT:
            if op.updated_input:
                for key, value in op.updated_input.items():
                    await store.put(key, value)
                dag.reset_task(op.target_task_id)
                logs.append(f"UPDATE_INPUT {op.target_task_id}: {list(op.updated_input.keys())}")

    return logs
