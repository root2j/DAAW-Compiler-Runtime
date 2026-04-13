"""Tests for the HITL interaction layer."""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import pytest

from daaw.interaction import (
    AutoAnswerInteractionHandler,
    InteractionRequest,
    InteractionUnavailableError,
    NullInteractionHandler,
    QueueInteractionHandler,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestAutoAnswer:
    def test_returns_answers_in_order(self):
        h = AutoAnswerInteractionHandler(["a", "b", "c"])
        assert run(h.ask(InteractionRequest(agent_id="x", prompt="p1"))) == "a"
        assert run(h.ask(InteractionRequest(agent_id="x", prompt="p2"))) == "b"
        assert run(h.ask(InteractionRequest(agent_id="x", prompt="p3"))) == "c"

    def test_records_calls(self):
        h = AutoAnswerInteractionHandler(["x"])
        run(h.ask(InteractionRequest(agent_id="ag1", prompt="hi")))
        assert len(h.calls) == 1
        assert h.calls[0].agent_id == "ag1"

    def test_exhausted_raises(self):
        h = AutoAnswerInteractionHandler(["only"])
        run(h.ask(InteractionRequest(agent_id="x", prompt="p")))
        with pytest.raises(InteractionUnavailableError):
            run(h.ask(InteractionRequest(agent_id="x", prompt="p2")))

    def test_fallback(self):
        h = AutoAnswerInteractionHandler([], fallback="default")
        assert run(h.ask(InteractionRequest(agent_id="x", prompt="p"))) == "default"


class TestNull:
    def test_always_raises(self):
        h = NullInteractionHandler()
        with pytest.raises(InteractionUnavailableError):
            run(h.ask(InteractionRequest(agent_id="x", prompt="p")))


class TestQueueHandler:
    def test_roundtrip_via_threads(self):
        """Simulate the UI pattern: worker thread asks, main thread answers."""
        questions: queue.Queue = queue.Queue()
        answers: queue.Queue = queue.Queue()
        handler = QueueInteractionHandler(questions, answers, timeout=5.0)

        captured = {}

        def worker():
            ans = run(handler.ask(InteractionRequest(agent_id="ag", prompt="hello?")))
            captured["answer"] = ans

        t = threading.Thread(target=worker)
        t.start()

        req = questions.get(timeout=2.0)
        assert req.prompt == "hello?"
        answers.put("world")

        t.join(timeout=2.0)
        assert not t.is_alive()
        assert captured["answer"] == "world"

    def test_timeout(self):
        questions: queue.Queue = queue.Queue()
        answers: queue.Queue = queue.Queue()
        handler = QueueInteractionHandler(questions, answers, timeout=0.2)
        with pytest.raises(InteractionUnavailableError):
            run(handler.ask(InteractionRequest(agent_id="ag", prompt="p")))


class TestAgentIntegration:
    """End-to-end: user_proxy + factory + handler."""

    def test_user_proxy_prompt_mode_uses_handler(self, tmp_path):
        import daaw.agents.builtin.user_proxy  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.config import AppConfig
        from daaw.schemas.workflow import AgentSpec
        from daaw.store.artifact_store import ArtifactStore

        store_dir = str(tmp_path / "ip")
        handler = AutoAnswerInteractionHandler(["my-answer"])
        cfg = AppConfig(artifact_store_dir=store_dir)
        factory = AgentFactory(
            UnifiedLLMClient(cfg),
            ArtifactStore(store_dir),
            interaction_handler=handler,
        )
        agent = factory.create(
            "t1", AgentSpec(role="user_proxy", model_config_override={"mode": "prompt"})
        )
        result = run(agent.run("Please provide a city"))
        assert result.status == "success"
        assert result.output == "my-answer"
        assert len(handler.calls) == 1
        assert handler.calls[0].agent_id == "t1"

    def test_user_proxy_questionnaire_runs_seven_questions(self, tmp_path):
        import daaw.agents.builtin.user_proxy  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.config import AppConfig
        from daaw.schemas.workflow import AgentSpec
        from daaw.store.artifact_store import ArtifactStore

        store_dir = str(tmp_path / "q")
        answers = [f"ans{i}" for i in range(7)]
        handler = AutoAnswerInteractionHandler(answers)
        cfg = AppConfig(artifact_store_dir=store_dir)
        factory = AgentFactory(
            UnifiedLLMClient(cfg),
            ArtifactStore(store_dir),
            interaction_handler=handler,
        )
        agent = factory.create("t1", AgentSpec(role="user_proxy"))
        result = run(agent.run(None))
        assert result.status == "success"
        assert len(handler.calls) == 7
        assert result.metadata["answers"]["task"] == "ans0"
        assert result.metadata["answers"]["manual_process"] == "ans6"

    def test_agent_without_handler_raises_when_asking(self, tmp_path):
        import daaw.agents.builtin.user_proxy  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.config import AppConfig
        from daaw.schemas.workflow import AgentSpec
        from daaw.store.artifact_store import ArtifactStore

        store_dir = str(tmp_path / "noh")
        cfg = AppConfig(artifact_store_dir=store_dir)
        factory = AgentFactory(
            UnifiedLLMClient(cfg),
            ArtifactStore(store_dir),
            interaction_handler=None,
        )
        agent = factory.create(
            "t1", AgentSpec(role="user_proxy", model_config_override={"mode": "prompt"})
        )
        with pytest.raises(InteractionUnavailableError):
            run(agent.run("hi"))


class TestExecutorWithHITL:
    def test_user_proxy_task_runs_via_executor(self, tmp_path):
        """Full DAG: executor → factory → user_proxy agent → handler."""
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        import daaw.agents.builtin.user_proxy  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.config import AppConfig
        from daaw.engine.executor import DAGExecutor
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.schemas.workflow import AgentSpec, TaskSpec, WorkflowSpec
        from daaw.store.artifact_store import ArtifactStore

        handler = AutoAnswerInteractionHandler(["Tokyo"])
        cfg = AppConfig(artifact_store_dir=str(tmp_path / "store"))
        store = ArtifactStore(str(tmp_path / "store"))
        factory = AgentFactory(
            UnifiedLLMClient(cfg), store, interaction_handler=handler,
        )
        executor = DAGExecutor(factory, store, CircuitBreaker())

        spec = WorkflowSpec(
            name="hitl-test",
            description="test",
            tasks=[
                TaskSpec(
                    id="ask",
                    name="Ask user",
                    description="Which city do you want to visit?",
                    agent=AgentSpec(
                        role="user_proxy",
                        model_config_override={"mode": "prompt"},
                    ),
                    success_criteria="user replies",
                    timeout_seconds=30,
                ),
            ],
        )
        results = run(executor.execute(spec))
        assert results["ask"].agent_result.status == "success"
        assert results["ask"].agent_result.output == "Tokyo"
        assert len(handler.calls) == 1
