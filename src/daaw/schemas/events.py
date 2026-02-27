"""Event models — interactions, patches, and workflow mutations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from daaw.schemas.enums import PatchAction


class InteractionEvent(BaseModel):
    """A single message/interaction in a conversation."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PatchOperation(BaseModel):
    """A single mutation the critic wants to apply to the live DAG."""

    action: PatchAction
    target_task_id: str
    feedback: str = ""
    new_task: dict[str, Any] | None = None  # for INSERT
    updated_input: dict[str, Any] | None = None  # for UPDATE_INPUT


class WorkflowPatch(BaseModel):
    """A set of patch operations with the critic's reasoning."""

    operations: list[PatchOperation]
    reasoning: str = ""
