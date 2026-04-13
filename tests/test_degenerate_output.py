"""Regression tests for the Gemma/Ollama reserved-token-salad failure mode.

Symptoms of that failure (from an actual user report):
    - Agent output is a string of <unused42>, <tool|>, <bos>, etc.
    - Gateway silently returns those as status=success after exhausting retries.
    - Downstream tasks receive the garbage as dependency_outputs and echo it.
    - Critic can't parse a verdict and used to fail-open → false PASS.

These tests pin each of the three fixes so the regression can't sneak back.
"""

from __future__ import annotations

import asyncio

import pytest

from daaw.engine.context_pruner import (
    DEGENERATE_PLACEHOLDER,
    _sanitize_value,
    _sanitize_and_truncate,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ─── Fix 2: dependency sanitizer ─────────────────────────────────────────

class TestSanitizeValue:
    def test_unused_token_salad_becomes_placeholder(self):
        garbage = (
            "<unused30><unused28><unused0><unused42><unused49><unused39>"
            "<unused17><unused15><unused24><unused45><unused5><bos>"
            "<unused37><unused24><unused19><unused46><unused8><tool|>"
            "<unused9><mask><unk><unused14>"
        )
        assert _sanitize_value(garbage) == DEGENERATE_PLACEHOLDER

    def test_tool_markers_stripped(self):
        mixed = "<tool_call|>Here is the answer.<tool_response|><bos>Final."
        out = _sanitize_value(mixed)
        assert "<tool" not in out
        assert "<bos>" not in out
        assert "Here is the answer." in out
        assert "Final." in out

    def test_mostly_real_text_kept(self):
        text = "Tokyo is the capital of Japan. <eos>"
        out = _sanitize_value(text)
        assert out.startswith("Tokyo")
        assert "<eos>" not in out

    def test_non_string_passthrough(self):
        assert _sanitize_value({"k": "v"}) == {"k": "v"}
        assert _sanitize_value([1, 2, 3]) == [1, 2, 3]
        assert _sanitize_value(None) is None

    def test_empty_string(self):
        assert _sanitize_value("") == ""


class TestSanitizeAndTruncate:
    def test_strips_then_truncates(self):
        big = "<unused1>" * 10 + "A" * 3000
        out = _sanitize_and_truncate({"k": big})
        assert "<unused" not in out["k"]
        # Truncation marker present because the real text (3000 A's)
        # still exceeds the 2000-char cap after stripping.
        assert "[truncated]" in out["k"]

    def test_all_garbage_becomes_placeholder(self):
        garbage = "<unused42><tool|><unused5><bos>" * 50
        out = _sanitize_and_truncate({"task_001.output": garbage})
        assert out["task_001.output"] == DEGENERATE_PLACEHOLDER


# ─── Fix 1: gateway raises on exhausted degenerate retries ───────────────

class TestGatewayDegenerateRaises:
    """We monkeypatch httpx.AsyncClient.post so no network is required."""

    def test_raises_after_all_retries_garbage(self, monkeypatch):
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "<unused1><unused2><tool|><bos><mask>"
                            }
                        }
                    ],
                    "model": "gemma4:e4b",
                    "usage": {},
                }

        class FakeClient:
            def __init__(self, *a, **kw):
                self.calls = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, *a, **kw):
                self.calls += 1
                return FakeResp()

        monkeypatch.setattr(gateway_provider, "httpx", None, raising=False)

        # Replace httpx.AsyncClient inside the module namespace at call time.
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        # Also skip the async wait to keep the test fast.
        async def _no_wait(*a, **kw):
            return None

        monkeypatch.setattr(gateway_provider, "_wait_for_model", _no_wait)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://fake:9999/v1", default_model="gemma4:e4b",
        )
        with pytest.raises(RuntimeError, match="degenerate output"):
            run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=100))


# ─── Fix 3: critic fail-closed on unparseable verdict ────────────────────

class TestCriticFailClosed:
    def test_unparseable_verdict_returns_fail(self):
        from daaw.config import AppConfig
        from daaw.critic.critic import Critic
        from daaw.llm.base import LLMResponse, LLMProvider
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.schemas.results import AgentResult, TaskResult
        from daaw.schemas.workflow import AgentSpec, TaskSpec

        class JunkProvider(LLMProvider):
            def name(self):
                return "junk"

            async def chat(self, messages, **kw):
                return LLMResponse(
                    content="this is not JSON at all",
                    model="junk", usage={}, raw=None,
                )

        cfg = AppConfig()
        llm = UnifiedLLMClient(cfg)
        llm._providers["junk"] = JunkProvider()

        critic = Critic(llm, cfg, provider="junk")
        task = TaskSpec(
            id="t1", name="t", description="do a thing",
            agent=AgentSpec(role="generic_llm"),
            success_criteria="thing done",
        )
        result = TaskResult(
            task_id="t1",
            agent_result=AgentResult(output="some output", status="success"),
            elapsed_seconds=1.0,
        )
        passed, patch, reasoning = run(critic.evaluate(task, result))
        assert passed is False, "critic must fail-closed on unparseable verdict"
        assert "fail-closed" in reasoning.lower()
        assert patch is None
