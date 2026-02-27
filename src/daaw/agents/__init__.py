"""Agent framework — base class, registry, factory."""

from daaw.agents.base import BaseAgent
from daaw.agents.registry import AGENT_REGISTRY, get_agent_class, register_agent

__all__ = ["BaseAgent", "AGENT_REGISTRY", "register_agent", "get_agent_class"]
