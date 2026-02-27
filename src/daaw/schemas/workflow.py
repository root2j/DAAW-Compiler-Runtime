"""Core workflow contract — WorkflowSpec, TaskSpec, AgentSpec, DependencySpec."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """Specifies which agent to use for a task and how to configure it."""

    role: str  # maps to agent registry key
    model_config_override: dict[str, Any] | None = None
    tools_allowed: list[str] = Field(default_factory=list)
    system_prompt_override: str | None = None


class DependencySpec(BaseModel):
    """Declares a dependency on another task's output."""

    task_id: str
    output_key: str | None = None  # specific key from output dict; None = whole output


class TaskSpec(BaseModel):
    """A single unit of work inside a workflow."""

    id: str
    name: str
    description: str
    agent: AgentSpec
    dependencies: list[DependencySpec] = Field(default_factory=list)
    input_filter: list[str] = Field(default_factory=list)
    success_criteria: str = ""
    timeout_seconds: int = 300
    max_retries: int = 2


class WorkflowSpec(BaseModel):
    """The top-level compiled workflow — a DAG of TaskSpecs."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    tasks: list[TaskSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_task(self, task_id: str) -> TaskSpec | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def task_ids(self) -> list[str]:
        return [t.id for t in self.tasks]

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return {task_id: [ids it depends on]}."""
        return {t.id: [d.task_id for d in t.dependencies] for t in self.tasks}
