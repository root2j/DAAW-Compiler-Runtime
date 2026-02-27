"""Tests for the Compiler — requires Groq API key."""

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from conftest import skip_on_rate_limit
from daaw.compiler.compiler import Compiler
from daaw.config import AppConfig
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.workflow import WorkflowSpec

# Ensure builtin agents are registered (compiler reads AGENT_REGISTRY)
import daaw.agents.builtin.breakdown_agent  # noqa: F401
import daaw.agents.builtin.critic_agent  # noqa: F401
import daaw.agents.builtin.generic_llm_agent  # noqa: F401
import daaw.agents.builtin.planner_agent  # noqa: F401
import daaw.agents.builtin.pm_agent  # noqa: F401
import daaw.agents.builtin.user_proxy  # noqa: F401
import daaw.tools.mock_tools  # noqa: F401


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)


@pytest.fixture
def compiler():
    config = AppConfig(groq_api_key=os.environ.get("GROQ_API_KEY", ""))
    llm = UnifiedLLMClient(config)
    return Compiler(llm, config, provider="groq")


class TestCompiler:
    @skip_on_rate_limit
    def test_compile_returns_workflow_spec(self, compiler):
        """Compiler should produce a valid WorkflowSpec from a simple goal."""
        spec = run(compiler.compile("Send a daily summary email of new GitHub issues"))
        assert isinstance(spec, WorkflowSpec)
        assert spec.name  # non-empty
        assert len(spec.tasks) >= 1
        assert spec.id  # UUID assigned

    @skip_on_rate_limit
    def test_compile_tasks_have_ids(self, compiler):
        spec = run(compiler.compile("Classify incoming emails as spam or not spam"))
        for task in spec.tasks:
            assert task.id, f"Task missing ID: {task}"
            assert task.name, f"Task missing name: {task}"
            assert task.agent.role, f"Task missing agent role: {task}"

    @skip_on_rate_limit
    def test_compile_dag_is_valid(self, compiler):
        """The compiled workflow should be a valid DAG (no cycles, valid refs)."""
        from daaw.engine.dag import DAG

        spec = run(compiler.compile("Monitor a website for price changes and notify via Slack"))
        dag = DAG(spec)
        errors = dag.validate()
        assert errors == [], f"DAG validation failed: {errors}"

    @skip_on_rate_limit
    def test_compile_dependencies_reference_real_tasks(self, compiler):
        spec = run(compiler.compile("Fetch RSS feeds, summarize articles, post to Slack"))
        task_ids = set(spec.task_ids())
        for task in spec.tasks:
            for dep in task.dependencies:
                assert dep.task_id in task_ids, (
                    f"Task '{task.id}' depends on '{dep.task_id}' "
                    f"which is not in {task_ids}"
                )

    @skip_on_rate_limit
    def test_refine_preserves_id(self, compiler):
        """Refinement should keep the same workflow ID."""
        original = run(compiler.compile("Automate invoice processing"))
        refined = run(compiler.refine(original, "Add an approval step before payment"))
        assert refined.id == original.id
        assert isinstance(refined, WorkflowSpec)
        assert len(refined.tasks) >= 1
