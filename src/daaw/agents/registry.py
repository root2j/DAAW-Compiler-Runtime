"""Agent registry — register and look up agent classes by name."""

from __future__ import annotations

from typing import Type

from daaw.agents.base import BaseAgent

AGENT_REGISTRY: dict[str, Type[BaseAgent]] = {}


def register_agent(name: str):
    """Class decorator to register an agent under a given name."""

    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        if name in AGENT_REGISTRY:
            raise ValueError(
                f"Agent '{name}' already registered ({AGENT_REGISTRY[name].__name__})"
            )
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_agent_class(name: str) -> Type[BaseAgent]:
    """Look up an agent class by registry name."""
    if name not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY) or "(none)"
        raise ValueError(
            f"Agent '{name}' not found. Available agents: {available}"
        )
    return AGENT_REGISTRY[name]
