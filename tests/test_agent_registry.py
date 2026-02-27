"""Tests for agent registry and factory."""

import asyncio

import pytest

from daaw.agents.base import BaseAgent
from daaw.agents.registry import AGENT_REGISTRY, get_agent_class, register_agent
from daaw.schemas.results import AgentResult


class TestAgentRegistry:
    def test_builtin_agents_registered(self):
        """Importing builtin modules should register all 6 agents."""
        import daaw.agents.builtin.user_proxy  # noqa: F401
        import daaw.agents.builtin.pm_agent  # noqa: F401
        import daaw.agents.builtin.breakdown_agent  # noqa: F401
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        import daaw.agents.builtin.planner_agent  # noqa: F401
        import daaw.agents.builtin.critic_agent  # noqa: F401

        expected = {"user_proxy", "pm", "breakdown", "generic_llm", "planner", "critic"}
        assert expected.issubset(set(AGENT_REGISTRY.keys()))

    def test_get_agent_class(self):
        cls = get_agent_class("generic_llm")
        assert issubclass(cls, BaseAgent)

    def test_get_agent_class_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            get_agent_class("nonexistent_agent_type")

    def test_duplicate_registration_raises(self):
        """Registering the same name twice should raise."""
        with pytest.raises(ValueError, match="already registered"):
            @register_agent("generic_llm")
            class DuplicateAgent(BaseAgent):
                async def run(self, task_input):
                    return AgentResult()

    def test_all_agents_are_base_agent_subclass(self):
        for name, cls in AGENT_REGISTRY.items():
            assert issubclass(cls, BaseAgent), f"{name} is not a BaseAgent subclass"


class TestAgentFactory:
    def test_create_agent(self, app_config):
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.store.artifact_store import ArtifactStore
        from daaw.schemas.workflow import AgentSpec
        import tempfile

        llm = UnifiedLLMClient(app_config)
        store = ArtifactStore(tempfile.mkdtemp())
        factory = AgentFactory(llm, store)

        agent = factory.create("test_id", AgentSpec(role="generic_llm"))
        assert isinstance(agent, BaseAgent)
        assert agent.agent_id == "test_id"

    def test_create_with_config_override(self, app_config):
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.store.artifact_store import ArtifactStore
        from daaw.schemas.workflow import AgentSpec
        import tempfile

        llm = UnifiedLLMClient(app_config)
        store = ArtifactStore(tempfile.mkdtemp())
        factory = AgentFactory(llm, store)

        agent = factory.create(
            "test_id",
            AgentSpec(
                role="generic_llm",
                model_config_override={"provider": "gemini"},
                tools_allowed=["web_search"],
                system_prompt_override="Be helpful.",
            ),
        )
        assert agent.config["provider"] == "gemini"
        assert agent.config["tools_allowed"] == ["web_search"]
        assert agent.config["system_prompt_override"] == "Be helpful."

    def test_create_unknown_role_raises(self, app_config):
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.store.artifact_store import ArtifactStore
        from daaw.schemas.workflow import AgentSpec
        import tempfile

        llm = UnifiedLLMClient(app_config)
        store = ArtifactStore(tempfile.mkdtemp())
        factory = AgentFactory(llm, store)

        with pytest.raises(ValueError, match="not found"):
            factory.create("test_id", AgentSpec(role="nonexistent"))
