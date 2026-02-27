"""CriticAgent — registered for completeness; typically used via Critic class directly."""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.schemas.results import AgentResult


@register_agent("critic")
class CriticAgent(BaseAgent):
    """Placeholder agent for the critic role in the registry."""

    async def run(self, task_input: Any) -> AgentResult:
        return AgentResult(
            output="Critic evaluation should be run via the Critic class, not as a task agent.",
            status="success",
        )
