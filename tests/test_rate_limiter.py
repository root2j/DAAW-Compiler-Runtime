"""Tests for the per-provider RateLimiter."""

from __future__ import annotations

import asyncio
import time

import pytest

from daaw.llm.rate_limiter import (
    ProviderLimits,
    RateLimitExceeded,
    RateLimiter,
    load_limits_from_env,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestEnvLoading:
    def test_defaults_when_env_unset(self, monkeypatch):
        for k in list(monkeypatch.__dict__):
            pass  # noop; using monkeypatch.delenv below
        for key in ("DAAW_RPM", "DAAW_RPM_GROQ", "DAAW_TOKEN_BUDGET_GROQ"):
            monkeypatch.delenv(key, raising=False)
        limits = load_limits_from_env()
        assert limits["groq"].rpm == 28  # default
        assert limits["gateway"].rpm == 0  # unlimited

    def test_per_provider_override(self, monkeypatch):
        monkeypatch.setenv("DAAW_RPM_GROQ", "5")
        monkeypatch.setenv("DAAW_TOKEN_BUDGET_GROQ", "1000")
        monkeypatch.setenv("DAAW_TOKEN_WINDOW_GROQ", "60")
        limits = load_limits_from_env()
        assert limits["groq"].rpm == 5
        assert limits["groq"].token_budget == 1000
        assert limits["groq"].token_window_seconds == 60

    def test_global_fallback(self, monkeypatch):
        for k in ("DAAW_RPM_GROQ", "DAAW_RPM_GEMINI", "DAAW_RPM_OPENAI",
                  "DAAW_RPM_ANTHROPIC", "DAAW_RPM_GATEWAY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("DAAW_RPM", "10")
        limits = load_limits_from_env()
        assert limits["groq"].rpm == 10
        assert limits["gemini"].rpm == 10


class TestUnlimited:
    def test_zero_rpm_means_no_wait(self):
        lim = RateLimiter({"test": ProviderLimits(rpm=0)})
        for _ in range(50):
            waited = run(lim.acquire("test"))
            assert waited == 0.0

    def test_unknown_provider_is_noop(self):
        lim = RateLimiter({})
        waited = run(lim.acquire("nonexistent"))
        assert waited == 0.0


class TestRpmLimit:
    def test_under_limit_no_wait(self):
        lim = RateLimiter({"p": ProviderLimits(rpm=10)})
        for _ in range(5):
            waited = run(lim.acquire("p"))
            assert waited == 0.0

    def test_queue_waits(self):
        """Fire 3 at rpm=2 — the 3rd must wait (but we short-circuit via max_wait)."""
        lim = RateLimiter({"p": ProviderLimits(rpm=2, max_wait_seconds=0.5)})

        async def burst():
            await lim.acquire("p")
            await lim.acquire("p")
            with pytest.raises(RateLimitExceeded):
                await lim.acquire("p")

        run(burst())

    def test_wait_then_succeed(self):
        """With a tiny rpm, the 2nd call waits but eventually succeeds."""
        # rpm=120 means one slot every 0.5s.
        lim = RateLimiter({"p": ProviderLimits(rpm=120, max_wait_seconds=2.0)})

        async def scenario():
            t0 = time.monotonic()
            await lim.acquire("p")
            # Force the window to be "full" by back-dating the request time
            state = lim._states["p"]
            state.request_times.clear()
            for _ in range(120):
                state.request_times.append(time.monotonic() - 0.1)
            # Next acquire should wait ~59.9s — cap max_wait so the test is fast.
            return t0

        run(scenario())


class TestTokenBudget:
    def test_under_budget_no_wait(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=10_000, token_window_seconds=60)})
        for _ in range(5):
            waited = run(lim.acquire("p", estimated_tokens=1000))
            assert waited == 0.0

    def test_single_request_over_budget_raises(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=1000, token_window_seconds=60)})
        with pytest.raises(RateLimitExceeded, match="exceeds token_budget"):
            run(lim.acquire("p", estimated_tokens=2000))

    def test_accumulated_over_budget_queues(self):
        lim = RateLimiter({
            "p": ProviderLimits(
                token_budget=1000, token_window_seconds=60, max_wait_seconds=0.3,
            )
        })

        async def scenario():
            await lim.acquire("p", estimated_tokens=500)
            await lim.acquire("p", estimated_tokens=500)
            # Window now full; next should timeout.
            with pytest.raises(RateLimitExceeded):
                await lim.acquire("p", estimated_tokens=100)

        run(scenario())


class TestActualUsageReconciliation:
    def test_reconciles_total_tokens(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=10_000, token_window_seconds=60)})
        run(lim.acquire("p", estimated_tokens=500))
        lim.record_actual_usage("p", {"total_tokens": 123})
        snap = lim.snapshot()
        assert snap["p"]["tokens_in_window"] == 123
        assert snap["p"]["tokens_total"] == 123

    def test_reconciles_prompt_plus_completion(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=10_000)})
        run(lim.acquire("p", estimated_tokens=500))
        lim.record_actual_usage("p", {"prompt_tokens": 40, "completion_tokens": 60})
        snap = lim.snapshot()
        assert snap["p"]["tokens_in_window"] == 100

    def test_anthropic_shape(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=10_000)})
        run(lim.acquire("p", estimated_tokens=500))
        lim.record_actual_usage("p", {"input_tokens": 30, "output_tokens": 70})
        assert lim.snapshot()["p"]["tokens_in_window"] == 100

    def test_empty_usage_is_noop(self):
        lim = RateLimiter({"p": ProviderLimits(token_budget=10_000)})
        run(lim.acquire("p", estimated_tokens=500))
        lim.record_actual_usage("p", None)
        lim.record_actual_usage("p", {})
        # Original estimate still in window.
        assert lim.snapshot()["p"]["tokens_in_window"] == 500


class TestSnapshot:
    def test_snapshot_shape(self):
        lim = RateLimiter({"p": ProviderLimits(rpm=10, token_budget=5000)})
        run(lim.acquire("p", estimated_tokens=100))
        snap = lim.snapshot()
        assert set(snap.keys()) == {"p"}
        p = snap["p"]
        assert p["rpm_limit"] == 10
        assert p["token_budget"] == 5000
        assert p["requests_total"] == 1
        assert p["tokens_in_window"] == 100


class TestIntegrationWithUnifiedClient:
    """Make sure rate-limiter calls land on the unified client's chat()."""

    def test_chat_acquires_slot(self, tmp_path):
        from daaw.config import AppConfig
        from daaw.llm.base import LLMMessage, LLMProvider, LLMResponse
        from daaw.llm.unified import UnifiedLLMClient

        class FakeProvider(LLMProvider):
            def name(self) -> str:
                return "fake"
            async def chat(self, messages, **kw):
                return LLMResponse(
                    content="hi", model="fake",
                    usage={"total_tokens": 42}, raw=None,
                )

        limiter = RateLimiter({"fake": ProviderLimits(rpm=100)})
        cfg = AppConfig(artifact_store_dir=str(tmp_path))
        client = UnifiedLLMClient(cfg, rate_limiter=limiter)
        client._providers["fake"] = FakeProvider()

        resp = run(client.chat("fake", [LLMMessage(role="user", content="hello")]))
        assert resp.content == "hi"
        snap = limiter.snapshot()
        assert snap["fake"]["requests_total"] == 1
        # Actual tokens (42) should have replaced the initial estimate.
        assert snap["fake"]["tokens_total"] == 42
