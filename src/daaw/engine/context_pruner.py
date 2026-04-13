"""Context pruner — builds the input dict for a task from the artifact store.

Truncates large dependency outputs to keep context size manageable for
small/local LLMs.  The limit is generous enough for cloud APIs but
prevents OOM on 8K-context local models.

Also strips tokenizer reserved-token strings (``<unused42>``, ``<tool|>``,
``<bos>``, etc.) from dependency outputs.  When a local model like Gemma
drifts into its reserved vocab, its raw output contains those marker
strings.  If we feed the string back into the next task's prompt, the
next model sees the same tokens in its input and happily regenerates them
— one garbage task cascades into every downstream task.  Sanitizing here
breaks that chain.
"""

from __future__ import annotations

import re
from typing import Any

from daaw.schemas.workflow import TaskSpec
from daaw.store.artifact_store import ArtifactStore

# Max characters per dependency output.  Keeps downstream tasks
# from getting a 10K-char wall of text as input.
MAX_DEP_OUTPUT_CHARS = 2000

# Matches tokenizer reserved-token strings emitted by Gemma/Llama when the
# model drifts: <unused42>, <tool|>, <tool_call|>, <tool_response|>,
# <bos>, <eos>, <mask>, <unk>, <pad>, <start_of_turn>, <end_of_turn>.
_RESERVED_TOKEN_RE = re.compile(
    r"<(?:unused\d+|tool(?:_call|_response)?\|?|bos|eos|mask|unk|pad|"
    r"start_of_turn|end_of_turn|\|.*?\|)>",
    re.IGNORECASE,
)

# If after sanitizing less than this fraction of original length remains,
# the whole value was essentially reserved-token salad — replace with an
# explicit error marker so the downstream LLM knows not to echo garbage.
_MIN_REAL_CONTENT_RATIO = 0.2

# Sentinel replacing an upstream output that was detected as degenerate.
# Phrased so the next task's LLM treats it as a broken dependency rather
# than repeating it.
DEGENERATE_PLACEHOLDER = (
    "[upstream task produced no usable output; continue without its content]"
)


async def prune_context(task: TaskSpec, store: ArtifactStore) -> dict[str, Any]:
    """Build task input: use input_filter if set, else gather dependency outputs."""
    if task.input_filter:
        raw = await store.get_many(task.input_filter)
        return _sanitize_and_truncate(raw)

    # Default: gather {dep_id}.output for all dependencies
    keys = [f"{dep.task_id}.output" for dep in task.dependencies]
    if not keys:
        return {}
    raw = await store.get_many(keys)
    return _sanitize_and_truncate(raw)


def _sanitize_value(value: Any) -> Any:
    """Strip reserved-token strings from string values; leave other types alone."""
    if not isinstance(value, str) or not value:
        return value
    cleaned = _RESERVED_TOKEN_RE.sub("", value).strip()
    # If almost nothing real remains, the whole output was garbage.
    if not cleaned or (len(cleaned) / max(len(value), 1)) < _MIN_REAL_CONTENT_RATIO:
        return DEGENERATE_PLACEHOLDER
    return cleaned


def _sanitize_and_truncate(data: dict[str, Any]) -> dict[str, Any]:
    """Strip reserved tokens then truncate overly-long string values."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        sanitized = _sanitize_value(value)
        if isinstance(sanitized, str) and len(sanitized) > MAX_DEP_OUTPUT_CHARS:
            result[key] = sanitized[:MAX_DEP_OUTPUT_CHARS] + "\n...[truncated]"
        else:
            result[key] = sanitized
    return result


# Backwards-compat alias — older code/tests may import `_truncate`.
_truncate = _sanitize_and_truncate
