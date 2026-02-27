"""Pydantic schemas — the core contracts for DAAW."""

from daaw.schemas.enums import AgentRole, PatchAction, TaskStatus
from daaw.schemas.events import InteractionEvent, PatchOperation, WorkflowPatch
from daaw.schemas.results import AgentResult, TaskResult
from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec, WorkflowSpec

__all__ = [
    "TaskStatus", "AgentRole", "PatchAction",
    "AgentSpec", "DependencySpec", "TaskSpec", "WorkflowSpec",
    "AgentResult", "TaskResult",
    "InteractionEvent", "PatchOperation", "WorkflowPatch",
]
