"""Context pruner — builds the input dict for a task from the artifact store.

Truncates large dependency outputs to keep context size manageable for
small/local LLMs.  The limit is generous enough for cloud APIs but
prevents OOM on 8K-context local models.
"""

from __future__ import annotations

from typing import Any

from daaw.schemas.workflow import TaskSpec
from daaw.store.artifact_store import ArtifactStore

# Max characters per dependency output.  Keeps downstream tasks
# from getting a 10K-char wall of text as input.
MAX_DEP_OUTPUT_CHARS = 2000


async def prune_context(task: TaskSpec, store: ArtifactStore) -> dict[str, Any]:
    """Build task input: use input_filter if set, else gather dependency outputs."""
    if task.input_filter:
        raw = await store.get_many(task.input_filter)
        return _truncate(raw)

    # Default: gather {dep_id}.output for all dependencies
    keys = [f"{dep.task_id}.output" for dep in task.dependencies]
    if not keys:
        return {}
    raw = await store.get_many(keys)
    return _truncate(raw)


def _truncate(data: dict[str, Any]) -> dict[str, Any]:
    """Truncate string values that exceed MAX_DEP_OUTPUT_CHARS."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > MAX_DEP_OUTPUT_CHARS:
            result[key] = value[:MAX_DEP_OUTPUT_CHARS] + "\n...[truncated]"
        else:
            result[key] = value
    return result
