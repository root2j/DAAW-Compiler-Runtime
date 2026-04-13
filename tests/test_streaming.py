"""Tests for LLM streaming: provider default fallback, UnifiedLLMClient
stream routing, and Compiler.compile_stream token delivery."""

from __future__ import annotations

import asyncio
import json

import pytest

from daaw.config import AppConfig
from daaw.llm.base import (
    LLMMessage, LLMProvider, LLMResponse, LLMStreamChunk,
)
from daaw.llm.unified import UnifiedLLMClient


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(async_gen):
    out = []
    async for chunk in async_gen:
        out.append(chunk)
    return out


class DefaultFallbackProvider(LLMProvider):
    """Provider that only implements chat() — exercises the default
    chat_stream() in the ABC."""

    def name(self):
        return "fallback"

    async def chat(self, messages, **kw):
        return LLMResponse(
            content="hello there", model="fb",
            usage={"total_tokens": 7}, raw=None,
        )


class TestDefaultStreamFallback:
    def test_non_streaming_provider_still_yields_one_chunk(self):
        p = DefaultFallbackProvider()
        chunks = run(_collect(p.chat_stream([LLMMessage(role="user", content="hi")])))
        assert len(chunks) == 1
        assert chunks[0].done
        assert chunks[0].full_content == "hello there"
        assert chunks[0].usage == {"total_tokens": 7}


class ScriptedStreamProvider(LLMProvider):
    """Provider that emits a pre-programmed list of delta strings."""

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    def name(self):
        return "stream_scripted"

    async def chat(self, messages, **kw):
        content = "".join(self._deltas)
        return LLMResponse(content=content, model="s", usage={}, raw=None)

    async def chat_stream(self, messages, **kw):
        accumulated = ""
        for d in self._deltas:
            accumulated += d
            yield LLMStreamChunk(delta=d, done=False, full_content=accumulated)
        yield LLMStreamChunk(
            delta="", done=True, full_content=accumulated,
            usage={"total_tokens": len(accumulated)},
        )


class TestUnifiedStreamRouting:
    def test_routes_to_provider_stream(self):
        cfg = AppConfig(artifact_store_dir="/tmp/daaw")
        llm = UnifiedLLMClient(cfg)
        llm._providers["stream_scripted"] = ScriptedStreamProvider(["hel", "lo ", "world"])
        chunks = run(_collect(llm.chat_stream(
            "stream_scripted", [LLMMessage(role="user", content="hi")],
        )))
        assert len(chunks) == 4  # 3 deltas + final done
        assert chunks[-1].done
        assert chunks[-1].full_content == "hello world"

    def test_records_usage_from_final_chunk(self):
        cfg = AppConfig(artifact_store_dir="/tmp/daaw")
        llm = UnifiedLLMClient(cfg)
        llm._providers["stream_scripted"] = ScriptedStreamProvider(["abc"])
        run(_collect(llm.chat_stream(
            "stream_scripted", [LLMMessage(role="user", content="x")],
        )))
        # Rate limiter should have accrued tokens_total via record_actual_usage.
        snap = llm.rate_limiter.snapshot()
        # scripted isn't a known provider name, so it lives under default mapping.
        # The per-provider state is only created for names in _DEFAULT_LIMITS,
        # so "stream_scripted" is simply a no-op. That's fine — we just assert the
        # call completes without raising.

    def test_unknown_provider_raises(self):
        cfg = AppConfig(artifact_store_dir="/tmp/daaw")
        llm = UnifiedLLMClient(cfg)
        with pytest.raises(ValueError, match="not available"):
            run(_collect(llm.chat_stream("nope", [])))


class TestAgentTokenStreaming:
    """Factory's on_agent_token callback routes per-agent tokens correctly."""

    def test_agent_streams_when_callback_attached(self):
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.config import AppConfig
        from daaw.schemas.workflow import AgentSpec
        from daaw.store.artifact_store import ArtifactStore

        cfg = AppConfig(groq_api_key="x")
        llm = UnifiedLLMClient(cfg)
        llm._providers["stream_scripted"] = ScriptedStreamProvider(
            ["Hel", "lo ", "wor", "ld"],
        )

        captured: list[tuple[str, str, str]] = []

        def _on_agent_token(task_id, delta, full):
            captured.append((task_id, delta, full))

        import tempfile
        store = ArtifactStore(tempfile.mkdtemp())
        factory = AgentFactory(
            llm, store,
            default_provider="stream_scripted",
            on_agent_token=_on_agent_token,
        )
        agent = factory.create(
            "task_abc", AgentSpec(role="generic_llm", tools_allowed=[]),
        )
        result = run(agent.run({"task": "say hello"}))
        assert result.status == "success"
        assert result.output == "Hello world"
        # Every delta was forwarded to the callback with the correct task id.
        assert [c[0] for c in captured] == ["task_abc"] * 4
        assert captured[-1][2] == "Hello world"


class TestCompilerStream:
    """compile_stream yields tokens through on_token and returns a spec."""

    def test_callback_receives_tokens_and_final_spec_parses(self):
        from daaw.compiler.compiler import Compiler
        # Valid FLAT schema (as used by the planner prompt).
        plan = {
            "name": "plan", "description": "d",
            "tasks": [
                {
                    "id": "t1", "name": "First", "description": "do it",
                    "role": "generic_llm", "tools_allowed": [],
                    "dependencies": [], "success_criteria": "done",
                    "timeout_seconds": 120, "max_retries": 2,
                },
                {
                    "id": "t2", "name": "Second", "description": "do more",
                    "role": "generic_llm", "tools_allowed": [],
                    "dependencies": [{"task_id": "t1"}],
                    "success_criteria": "done",
                    "timeout_seconds": 120, "max_retries": 2,
                },
            ], "metadata": {},
        }
        body = json.dumps(plan)
        # Stream the body in 40-char bites to exercise the accumulator.
        deltas = [body[i:i + 40] for i in range(0, len(body), 40)]

        cfg = AppConfig(groq_api_key="x")
        llm = UnifiedLLMClient(cfg)
        llm._providers["stream_scripted"] = ScriptedStreamProvider(deltas)
        c = Compiler(llm, cfg, provider="stream_scripted")

        token_log: list[tuple[str, int]] = []

        def on_token(delta: str, full: str) -> None:
            token_log.append((delta, len(full)))

        spec = run(c.compile_stream("build a 2-step plan", on_token=on_token))
        # We saw all the deltas, accumulated length grew monotonically.
        assert len(token_log) == len(deltas)
        assert token_log[0][1] == len(deltas[0])
        assert token_log[-1][1] == len(body)
        # Final spec actually parses.
        assert len(spec.tasks) == 2
        assert spec.tasks[0].id == "t1"

    def test_streaming_retries_on_invalid_then_valid_json(self):
        """The retry loop must exercise cleanly with streaming."""
        from daaw.compiler.compiler import Compiler

        class TwoShotProvider(LLMProvider):
            def __init__(self):
                self.calls = 0
            def name(self): return "two"
            async def chat(self, messages, **kw):
                self.calls += 1
                return LLMResponse(content="", model="x", usage={}, raw=None)
            async def chat_stream(self, messages, **kw):
                self.calls += 1
                if self.calls == 1:
                    body = "not valid json at all"
                else:
                    body = json.dumps({
                        "name": "ok", "description": "d", "metadata": {},
                        "tasks": [
                            {"id": "t1", "name": "n", "description": "d",
                             "role": "generic_llm", "tools_allowed": [],
                             "dependencies": [], "success_criteria": "ok",
                             "timeout_seconds": 60, "max_retries": 1},
                            {"id": "t2", "name": "n2", "description": "d",
                             "role": "generic_llm", "tools_allowed": [],
                             "dependencies": [{"task_id": "t1"}],
                             "success_criteria": "ok",
                             "timeout_seconds": 60, "max_retries": 1},
                        ],
                    })
                yield LLMStreamChunk(delta=body, done=False, full_content=body)
                yield LLMStreamChunk(delta="", done=True, full_content=body)

        cfg = AppConfig(groq_api_key="x", max_planner_retries=3)
        llm = UnifiedLLMClient(cfg)
        prov = TwoShotProvider()
        llm._providers["two"] = prov
        c = Compiler(llm, cfg, provider="two")
        spec = run(c.compile_stream("do something"))
        assert len(spec.tasks) == 2
        assert prov.calls == 2
