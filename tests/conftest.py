"""Shared fixtures for the DAAW test suite."""

import asyncio
import functools
import os
import sys
import tempfile
import time

import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Delay between API calls (seconds). Groq free tier allows 30 req/min.
# Override with env var: API_TEST_DELAY=10
API_DELAY = int(os.environ.get("API_TEST_DELAY", "3"))

# Tracks when the last API call was made so we only wait the remaining gap.
_last_api_call: float = 0.0


def skip_on_rate_limit(fn):
    """Decorator: wait between API calls, catch 429 and skip gracefully."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _last_api_call

        # Wait out the remaining cooldown since the last API call
        elapsed = time.monotonic() - _last_api_call
        remaining = API_DELAY - elapsed
        if _last_api_call > 0 and remaining > 0:
            print(f"\n    [rate-limit] waiting {remaining:.0f}s before next API call...")
            time.sleep(remaining)

        _last_api_call = time.monotonic()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower():
                pytest.skip("Rate limit hit (429)")
            raise

    return wrapper


@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_store_dir(tmp_path):
    """Return a temp directory for artifact store tests."""
    return str(tmp_path / "test_store")


@pytest.fixture
def app_config():
    """AppConfig with keys read from env (Gemini expected)."""
    from daaw.config import AppConfig
    from dotenv import load_dotenv

    load_dotenv()
    return AppConfig(
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        artifact_store_dir=str(tempfile.mkdtemp()),
    )


@pytest.fixture
def sample_workflow_spec():
    """A simple 3-task linear workflow for testing."""
    from daaw.schemas.workflow import (
        AgentSpec,
        DependencySpec,
        TaskSpec,
        WorkflowSpec,
    )

    return WorkflowSpec(
        name="Test Workflow",
        description="A test workflow with 3 sequential tasks",
        tasks=[
            TaskSpec(
                id="task_001",
                name="Gather Input",
                description="Collect user input",
                agent=AgentSpec(role="generic_llm"),
                success_criteria="Output is non-empty",
            ),
            TaskSpec(
                id="task_002",
                name="Process",
                description="Process the input",
                agent=AgentSpec(role="generic_llm"),
                dependencies=[DependencySpec(task_id="task_001")],
                success_criteria="Output contains processed data",
            ),
            TaskSpec(
                id="task_003",
                name="Summarize",
                description="Summarize results",
                agent=AgentSpec(role="generic_llm"),
                dependencies=[DependencySpec(task_id="task_002")],
                success_criteria="Summary is present",
            ),
        ],
    )


@pytest.fixture
def parallel_workflow_spec():
    """3 independent tasks that can run in parallel."""
    from daaw.schemas.workflow import AgentSpec, TaskSpec, WorkflowSpec

    return WorkflowSpec(
        name="Parallel Workflow",
        description="3 independent tasks",
        tasks=[
            TaskSpec(
                id="p1",
                name="Task A",
                description="Independent A",
                agent=AgentSpec(role="generic_llm"),
            ),
            TaskSpec(
                id="p2",
                name="Task B",
                description="Independent B",
                agent=AgentSpec(role="generic_llm"),
            ),
            TaskSpec(
                id="p3",
                name="Task C",
                description="Independent C",
                agent=AgentSpec(role="generic_llm"),
            ),
        ],
    )
