"""Unit tests for the compiler's task-count retry behaviour.

Mocks the LLM so we can drive specific response sequences and assert on
whether the compiler retries, accepts, or hard-fails.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from daaw.compiler.compiler import Compiler
from daaw.config import AppConfig
from daaw.llm.base import LLMProvider, LLMResponse
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.workflow import WorkflowSpec

# Ensure agents are registered so _fixup_agent_roles doesn't downgrade.
import daaw.agents.builtin.generic_llm_agent  # noqa: F401
import daaw.tools.mock_tools  # noqa: F401


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _spec_json(n_tasks: int) -> str:
    """Build a valid flat-schema compiler response with N tasks."""
    tasks = []
    for i in range(n_tasks):
        deps = [{"task_id": f"task_{i:03d}"}] if i > 0 else []
        tasks.append({
            "id": f"task_{i + 1:03d}",
            "name": f"Step {i + 1}",
            "description": f"Do step {i + 1}",
            "role": "generic_llm",
            "tools_allowed": [],
            "dependencies": deps,
            "success_criteria": "done",
            "timeout_seconds": 120,
            "max_retries": 2,
        })
    return json.dumps({
        "name": "Test plan", "description": "desc", "tasks": tasks, "metadata": {},
    })


class ScriptedProvider(LLMProvider):
    """LLMProvider that returns pre-programmed responses, tracking calls."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []  # list of message lists

    def name(self):
        return "scripted"

    async def chat(self, messages, **kw):
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("ScriptedProvider exhausted")
        return LLMResponse(content=self._responses.pop(0), model="scripted",
                           usage={}, raw=None)


def _compiler_with(responses: list[str], *, max_retries: int = 3):
    cfg = AppConfig(
        groq_api_key="x",  # non-empty so UnifiedLLMClient doesn't skip
        max_planner_retries=max_retries,
    )
    llm = UnifiedLLMClient(cfg)
    provider = ScriptedProvider(responses)
    llm._providers["scripted"] = provider
    return Compiler(llm, cfg, provider="scripted"), provider


class TestTaskCountHandling:
    def test_multi_task_accepted_immediately(self):
        c, prov = _compiler_with([_spec_json(3)])
        spec = run(c.compile("plan a trip"))
        assert isinstance(spec, WorkflowSpec)
        assert len(spec.tasks) == 3
        assert len(prov.calls) == 1  # no retry needed

    def test_single_task_retried_then_multi_accepted(self):
        c, prov = _compiler_with([_spec_json(1), _spec_json(2)])
        spec = run(c.compile("plan a trip"))
        assert len(spec.tasks) == 2
        assert len(prov.calls) == 2  # retried once
        # Retry prompt should mention the previous problem.
        second_user_msg = next(
            m for m in prov.calls[1] if m.role == "user"
        ).content
        assert "Only 1 task" in second_user_msg

    def test_single_task_accepted_on_final_attempt(self):
        """Don't hard-fail if small model can't decompose. Accept 1-task."""
        c, prov = _compiler_with(
            [_spec_json(1), _spec_json(1), _spec_json(1)],
            max_retries=3,
        )
        spec = run(c.compile("plan a trip"))
        assert isinstance(spec, WorkflowSpec)
        assert len(spec.tasks) == 1  # accepted rather than raised
        assert len(prov.calls) == 3  # used all retries trying to decompose

    def test_zero_task_always_fails(self):
        c, prov = _compiler_with(
            [_spec_json(0), _spec_json(0), _spec_json(0)],
            max_retries=3,
        )
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            run(c.compile("plan a trip"))
        assert len(prov.calls) == 3

    def test_zero_then_multi_succeeds(self):
        c, prov = _compiler_with([_spec_json(0), _spec_json(2)])
        spec = run(c.compile("plan a trip"))
        assert len(spec.tasks) == 2
        assert len(prov.calls) == 2
