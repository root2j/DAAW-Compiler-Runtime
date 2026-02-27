"""Tests for the Critic — requires Groq API key."""

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from conftest import skip_on_rate_limit
from daaw.config import AppConfig
from daaw.critic.critic import Critic
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.events import WorkflowPatch
from daaw.schemas.results import AgentResult, TaskResult
from daaw.schemas.workflow import AgentSpec, TaskSpec


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set",
)


@pytest.fixture
def critic():
    config = AppConfig(groq_api_key=os.environ.get("GROQ_API_KEY", ""))
    llm = UnifiedLLMClient(config)
    return Critic(llm, config, provider="groq")


def _make_task(success_criteria="Output is non-empty"):
    return TaskSpec(
        id="t1", name="Test Task", description="A test task",
        agent=AgentSpec(role="generic_llm"),
        success_criteria=success_criteria,
    )


class TestCritic:
    def test_skip_no_criteria(self, critic):
        """Tasks without success_criteria auto-pass — no API call needed."""
        task = TaskSpec(
            id="t1", name="No Criteria", description="d",
            agent=AgentSpec(role="generic_llm"),
            success_criteria="",
        )
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(output="anything", status="success"),
        )
        passed, patch = run(critic.evaluate(task, result))
        assert passed is True
        assert patch is None

    def test_auto_retry_on_failure(self, critic):
        """Failed tasks get an auto-retry patch — no API call needed."""
        task = _make_task()
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(status="failure", error_message="crashed"),
        )
        passed, patch = run(critic.evaluate(task, result))
        assert passed is False
        assert patch is not None
        assert patch.operations[0].action.value == "retry"

    @skip_on_rate_limit
    def test_pass_good_output(self, critic):
        """A clearly successful output should pass."""
        task = _make_task(success_criteria="Output contains a greeting")
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(
                output="Hello! Welcome to the system. Everything is set up.",
                status="success",
            ),
            elapsed_seconds=1.0,
        )
        passed, patch = run(critic.evaluate(task, result))
        assert passed is True

    @skip_on_rate_limit
    def test_fail_bad_output(self, critic):
        """Output that clearly doesn't meet criteria should fail."""
        task = _make_task(
            success_criteria="Output must contain a valid JSON object with 'status' key"
        )
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(
                output="This is just plain text with no JSON at all.",
                status="success",
            ),
            elapsed_seconds=0.5,
        )
        passed, patch = run(critic.evaluate(task, result))
        assert passed is False


class TestCriticPatch:
    @skip_on_rate_limit
    def test_patch_operations_valid(self, critic):
        """When critic fails a task, patch should have valid operations."""
        task = _make_task(success_criteria="Output must be a numbered list of at least 5 items")
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(output="Just one item.", status="success"),
            elapsed_seconds=0.5,
        )
        passed, patch = run(critic.evaluate(task, result))
        if not passed and patch is not None:
            assert len(patch.operations) > 0
            for op in patch.operations:
                assert op.target_task_id == "t1"
