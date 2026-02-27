"""BaseAgent ABC — all agents inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.results import AgentResult
from daaw.store.artifact_store import ArtifactStore


class BaseAgent(ABC):
    """Abstract agent with DI for LLM client and artifact store."""

    def __init__(
        self,
        agent_id: str,
        llm_client: UnifiedLLMClient,
        store: ArtifactStore,
        config: dict[str, Any] | None = None,
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.store = store
        self.config = config or {}

    @abstractmethod
    async def run(self, task_input: Any) -> AgentResult:
        """Execute the agent's work and return a result."""
        ...

    def get_tool_schemas(self) -> list[dict]:
        """Override to expose tools this agent can use."""
        return []
