"""PlannerAgent — thin wrapper around the Compiler for use inside a workflow."""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.compiler.compiler import Compiler
from daaw.config import get_config
from daaw.schemas.results import AgentResult


@register_agent("planner")
class PlannerAgent(BaseAgent):
    """Compiles a user goal into a WorkflowSpec JSON string."""

    async def run(self, task_input: Any) -> AgentResult:
        provider = self.config.get("provider", "groq")
        model = self.config.get("model")
        config = get_config()

        compiler = Compiler(
            llm_client=self.llm_client,
            config=config,
            provider=provider,
            model=model,
        )

        goal = task_input if isinstance(task_input, str) else str(task_input)
        spec = await compiler.compile(goal)
        return AgentResult(
            output=spec.model_dump_json(indent=2),
            status="success",
            metadata={"workflow_id": spec.id},
        )
