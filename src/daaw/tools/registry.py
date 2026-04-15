"""Tool registry — register, discover, and execute tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Coroutine[Any, Any, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        # Invalidation flags reset by register(); recomputed lazily.
        self._names_cache: frozenset[str] | None = None
        self._lower_cache: dict[str, str] | None = None

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator to register an async function as a tool."""

        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}},
                handler=fn,
            )
            self._names_cache = None  # invalidate
            self._lower_cache = None
            return fn

        return decorator

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        """Cached set of registered tool names — cheap in hot paths."""
        if self._names_cache is None:
            self._names_cache = frozenset(self._tools.keys())
        return self._names_cache

    def resolve_name(self, name: str) -> str | None:
        """Find a tool by name, case-insensitively. Returns the canonical
        registered name or None. LLMs frequently hallucinate capitalization
        (Claude → ``WebSearch``, Llama → ``brave_search``), so the agent's
        tool dispatcher uses this to map them back to real tools.
        """
        if name in self._tools:
            return name
        if self._lower_cache is None:
            self._lower_cache = {n.lower(): n for n in self._tools}
        return self._lower_cache.get(name.lower())

    def list_tools(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-style tool schemas, optionally filtered."""
        result = []
        for tool_name, tool_def in self._tools.items():
            if allowed is not None and tool_name not in allowed:
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "parameters": tool_def.parameters,
                    },
                }
            )
        return result

    async def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not registered")
        return await tool.handler(**kwargs)


# Global singleton
tool_registry = ToolRegistry()
