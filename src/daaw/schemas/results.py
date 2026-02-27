"""Result models returned by agents and the executor."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """What an agent returns after running."""

    output: Any = None
    status: str = "success"  # "success" | "failure" | "needs_human"
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""


class TaskResult(BaseModel):
    """Wrapper around AgentResult with execution metadata."""

    task_id: str
    agent_result: AgentResult
    attempt: int = 1
    elapsed_seconds: float = 0.0
