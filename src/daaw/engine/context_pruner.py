"""Context pruner — builds the input dict for a task from the artifact store."""

from __future__ import annotations

from typing import Any

from daaw.schemas.workflow import TaskSpec
from daaw.store.artifact_store import ArtifactStore


async def prune_context(task: TaskSpec, store: ArtifactStore) -> dict[str, Any]:
    """Build task input: use input_filter if set, else gather dependency outputs."""
    if task.input_filter:
        return await store.get_many(task.input_filter)

    # Default: gather {dep_id}.output for all dependencies
    keys = [f"{dep.task_id}.output" for dep in task.dependencies]
    if not keys:
        return {}
    return await store.get_many(keys)
