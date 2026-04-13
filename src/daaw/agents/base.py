"""BaseAgent ABC — all agents inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from daaw.interaction import (
    InteractionHandler,
    InteractionRequest,
    InteractionUnavailableError,
    NullInteractionHandler,
)
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.results import AgentResult
from daaw.store.artifact_store import ArtifactStore


class BaseAgent(ABC):
    """Abstract agent with DI for LLM client, artifact store, and HITL handler."""

    def __init__(
        self,
        agent_id: str,
        llm_client: UnifiedLLMClient,
        store: ArtifactStore,
        config: dict[str, Any] | None = None,
        interaction: InteractionHandler | None = None,
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.store = store
        self.config = config or {}
        self.interaction: InteractionHandler = interaction or NullInteractionHandler()

    async def ask_user(
        self,
        prompt: str,
        *,
        hint: str | None = None,
        choices: list[str] | None = None,
        step_id: str | None = None,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Convenience helper — ask the user a question via the injected handler.

        Raises ``InteractionUnavailableError`` if no handler was configured.
        """
        req = InteractionRequest(
            agent_id=self.agent_id,
            task_id=task_id,
            prompt=prompt,
            hint=hint,
            choices=choices,
            step_id=step_id,
            context=context or {},
        )
        return await self.interaction.ask(req)

    @abstractmethod
    async def run(self, task_input: Any) -> AgentResult:
        """Execute the agent's work and return a result."""
        ...

    def get_tool_schemas(self) -> list[dict]:
        """Override to expose tools this agent can use."""
        return []
