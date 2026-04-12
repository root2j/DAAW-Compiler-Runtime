"""Agent factory — creates agent instances from AgentSpec with DI."""

from __future__ import annotations

from daaw.agents.base import BaseAgent
from daaw.agents.registry import get_agent_class
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.workflow import AgentSpec
from daaw.store.artifact_store import ArtifactStore


class AgentFactory:
    """Creates BaseAgent instances with injected dependencies."""

    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        store: ArtifactStore,
        default_provider: str | None = None,
        default_model: str | None = None,
    ):
        self._llm_client = llm_client
        self._store = store
        self._default_provider = default_provider
        self._default_model = default_model

    def create(self, agent_id: str, agent_spec: AgentSpec) -> BaseAgent:
        cls = get_agent_class(agent_spec.role)
        config = {
            "tools_allowed": agent_spec.tools_allowed,
            "system_prompt_override": agent_spec.system_prompt_override,
        }
        # Inject the factory-level default provider so agents don't fall back
        # to a hardcoded default that may not be configured.
        if self._default_provider:
            config["provider"] = self._default_provider
        if self._default_model:
            config["model"] = self._default_model
        if agent_spec.model_config_override:
            # Per-task overrides (including provider) take precedence
            config.update(agent_spec.model_config_override)
        return cls(
            agent_id=agent_id,
            llm_client=self._llm_client,
            store=self._store,
            config=config,
        )
