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

    def _make_fake_client(self, response_content: str):
        """Factory returning a FakeClient that always serves ``response_content``.

        Captures every payload sent to ``post`` so tests can assert what
        was on the wire.
        """
        captured_payloads: list[dict] = []

        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": response_content}}],
                    "model": "gemma4:e4b",
                    "usage": {},
                }

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, *a, **kw):
                captured_payloads.append(kw.get("json", {}))
                return FakeResp()

        return FakeClient, captured_payloads

    def test_raises_after_all_retries_token_salad(self, monkeypatch):
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        FakeClient, _ = self._make_fake_client(
            "<unused1><unused2><tool|><bos><mask>"
        )
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        async def _no_wait(*a, **kw):
            return None
        monkeypatch.setattr(gateway_provider, "_wait_for_model", _no_wait)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://fake:9999/v1", default_model="gemma4:e4b",
        )
        with pytest.raises(RuntimeError, match=r"degenerate \(token-salad\)"):
            run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=100))

    def test_raises_with_empty_output_hint(self, monkeypatch):
        """Empty completions get the context-overflow diagnostic, not token-salad."""
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        FakeClient, _ = self._make_fake_client("")
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        async def _no_wait(*a, **kw):
            return None
        monkeypatch.setattr(gateway_provider, "_wait_for_model", _no_wait)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://fake:9999/v1", default_model="gemma4:e4b",
        )
        with pytest.raises(RuntimeError) as ei:
            run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=100))
        msg = str(ei.value)
        assert "empty" in msg
        assert "GATEWAY_NUM_CTX" in msg  # diagnostic points at the fix


class TestOllamaNumCtx:
    """Verify that num_ctx + keep_alive are sent only to Ollama-ish gateways."""

    def _make_capturing_client(self):
        captured: list[dict] = []

        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "choices": [{"message": {"content": "hello world."}}],
                    "model": "m", "usage": {},
                }

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                captured.append(kw.get("json", {}))
                return FakeResp()

        return FakeClient, captured

    def test_ollama_url_gets_num_ctx_options(self, monkeypatch):
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        FakeClient, captured = self._make_capturing_client()
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://localhost:11434/v1", default_model="gemma4:e4b",
        )
        run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=50))
        assert captured, "request was never made"
        payload = captured[0]
        assert "options" in payload, "Ollama endpoint missing options block"
        assert payload["options"].get("num_ctx") >= 4096
        assert payload.get("keep_alive"), "Ollama endpoint missing keep_alive"

    def test_non_ollama_url_no_num_ctx(self, monkeypatch):
        """LM Studio (port 1234) shouldn't get Ollama-only fields."""
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        FakeClient, captured = self._make_capturing_client()
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://localhost:1234/v1",
            default_model="some-model",
        )
        run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=50))
        assert captured
        payload = captured[0]
        assert "options" not in payload
        assert "keep_alive" not in payload

    def test_num_ctx_override_from_env(self, monkeypatch):
        monkeypatch.setenv("GATEWAY_NUM_CTX", "16384")
        # Module-level constant was captured at import time; reload to pick up env.
        import importlib
        from daaw.llm.providers import gateway_provider as gp
        importlib.reload(gp)
        assert gp.DEFAULT_NUM_CTX == 16384


class TestCudaOomAutoRetry:
    """A CUDA OOM 500 should trigger automatic num_ctx halving."""

    def _client_factory(self, response_sequence: list):
        """Return a FakeClient that yields the given (status, body) tuples in order.

        Captures every payload sent for assertions.
        """
        captured: list[dict] = []
        idx = {"i": 0}

        class FakeResp:
            def __init__(self, status, body):
                self.status_code = status
                self._body = body
                self.text = str(body)[:500]

            def json(self):
                return self._body

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                captured.append(kw.get("json", {}))
                if idx["i"] >= len(response_sequence):
                    status, body = response_sequence[-1]
                else:
                    status, body = response_sequence[idx["i"]]
                idx["i"] += 1
                return FakeResp(status, body)

        return FakeClient, captured

    def test_cuda_oom_halves_num_ctx_then_succeeds(self, monkeypatch):
        # Pin DEFAULT_NUM_CTX to a known value so the assertion is env-independent.
        monkeypatch.setenv("GATEWAY_NUM_CTX", "8192")
        import importlib
        from daaw.llm.providers import gateway_provider
        importlib.reload(gateway_provider)
        from daaw.llm.base import LLMMessage

        oom = (500, {"error": {
            "message": "an error was encountered while running the model: CUDA error",
        }})
        ok = (200, {
            "choices": [{"message": {"content": "hello there friend."}}],
            "model": "gemma4:e4b", "usage": {},
        })
        FakeClient, _capture_func = self._client_factory([oom, ok])
        # Snapshot each payload deeply so later mutations don't overwrite
        # what we observed on earlier requests.
        import copy
        payloads_seen: list[dict] = []

        class SnapshotClient(FakeClient):
            async def post(self, *a, **kw):
                payloads_seen.append(copy.deepcopy(kw.get("json", {})))
                return await super().post(*a, **kw)

        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", SnapshotClient)

        async def _no_wait(*a, **kw):
            return None
        monkeypatch.setattr(gateway_provider, "_wait_for_model", _no_wait)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://localhost:11434/v1", default_model="gemma4:e4b",
        )
        resp = run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=50))
        assert resp.content == "hello there friend."
        assert len(payloads_seen) >= 2
        first_ctx = payloads_seen[0]["options"]["num_ctx"]
        second_ctx = payloads_seen[1]["options"]["num_ctx"]
        assert first_ctx == 8192, f"expected first attempt at 8192, got {first_ctx}"
        assert second_ctx == 4096, f"expected halved 4096, got {second_ctx}"

    def test_cuda_oom_raises_with_actionable_hint_when_floor_hit(self, monkeypatch):
        from daaw.llm.base import LLMMessage
        from daaw.llm.providers import gateway_provider

        # Always-OOM. After enough halvings we hit _MIN_NUM_CTX and raise.
        oom = (500, {"error": {"message": "CUDA error: out of memory"}})
        FakeClient, captured = self._client_factory([oom] * 10)
        import httpx as _real_httpx
        monkeypatch.setattr(_real_httpx, "AsyncClient", FakeClient)

        async def _no_wait(*a, **kw):
            return None
        monkeypatch.setattr(gateway_provider, "_wait_for_model", _no_wait)

        p = gateway_provider.GatewayProvider(
            gateway_url="http://localhost:11434/v1", default_model="gemma4:e4b",
        )
        with pytest.raises(RuntimeError, match="CUDA OOM after"):
            run(p.chat([LLMMessage(role="user", content="hi")], max_tokens=50))


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
