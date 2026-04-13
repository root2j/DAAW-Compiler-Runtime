"""Per-provider API rate limiting.

DAAW fans out to multiple LLM providers, each with its own free-tier limits
(Groq free = 30 req/min, Gemini flash = 60 req/min, etc.). A misconfigured
workflow can easily burn through a daily budget or trip a 429 loop with
retries. This module enforces two caps per provider:

    - Requests-per-minute (sliding window)
    - Token budget across a configurable window (default 1 day)

Usage from ``UnifiedLLMClient.chat``:

    async with limiter.slot(provider, estimated_tokens=max_tokens):
        resp = await self._providers[provider].chat(...)
        limiter.record_actual_usage(provider, resp.usage)

The limiter is a process-global singleton keyed off ``AppConfig`` so every
agent, retry, critic pass, and compile share the same budget.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class RateLimitExceeded(RuntimeError):
    """Raised when a request can't be scheduled within a bounded wait."""


@dataclass
class ProviderLimits:
    """Configuration for a single provider."""

    rpm: int = 0  # requests-per-minute; 0 = unlimited
    token_budget: int = 0  # total tokens per token_window_seconds; 0 = unlimited
    token_window_seconds: int = 86_400  # default: 1 day
    # Max seconds a caller will wait for a slot before giving up. Set
    # generously so the critic/retry loop doesn't fail the task on a
    # transient queue; cap prevents indefinite hangs in the UI.
    max_wait_seconds: float = 120.0


# Sensible defaults per provider. Zero means "no limit" (the limiter is a
# no-op). These are conservative free-tier numbers — override via env.
_DEFAULT_LIMITS: dict[str, ProviderLimits] = {
    "groq": ProviderLimits(rpm=28, token_budget=0),  # free tier ~30 rpm
    "gemini": ProviderLimits(rpm=55, token_budget=0),  # flash ~60 rpm
    "openai": ProviderLimits(rpm=50, token_budget=0),
    "anthropic": ProviderLimits(rpm=45, token_budget=0),
    "gateway": ProviderLimits(rpm=0, token_budget=0),  # local = unlimited
}


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def load_limits_from_env() -> dict[str, ProviderLimits]:
    """Build a provider→ProviderLimits map from environment variables.

    Env var schema (all optional; unset = use default):
        DAAW_RPM_<PROVIDER>              e.g. DAAW_RPM_GROQ=30
        DAAW_TOKEN_BUDGET_<PROVIDER>     e.g. DAAW_TOKEN_BUDGET_GEMINI=1000000
        DAAW_TOKEN_WINDOW_<PROVIDER>     window in seconds, default 86400
        DAAW_MAX_WAIT_<PROVIDER>         max queue wait in seconds, default 120

    Also supports global fallbacks: DAAW_RPM, DAAW_TOKEN_BUDGET,
    DAAW_TOKEN_WINDOW, DAAW_MAX_WAIT (applied when the per-provider
    override is absent).
    """
    limits: dict[str, ProviderLimits] = {}
    global_rpm = _env_int("DAAW_RPM", -1)
    global_tok = _env_int("DAAW_TOKEN_BUDGET", -1)
    global_win = _env_int("DAAW_TOKEN_WINDOW", -1)
    global_wait = _env_int("DAAW_MAX_WAIT", -1)

    for provider, default in _DEFAULT_LIMITS.items():
        up = provider.upper()
        rpm = _env_int(f"DAAW_RPM_{up}", global_rpm if global_rpm >= 0 else default.rpm)
        tok = _env_int(
            f"DAAW_TOKEN_BUDGET_{up}",
            global_tok if global_tok >= 0 else default.token_budget,
        )
        win = _env_int(
            f"DAAW_TOKEN_WINDOW_{up}",
            global_win if global_win > 0 else default.token_window_seconds,
        )
        wait = _env_int(
            f"DAAW_MAX_WAIT_{up}",
            global_wait if global_wait > 0 else int(default.max_wait_seconds),
        )
        limits[provider] = ProviderLimits(
            rpm=max(0, rpm),
            token_budget=max(0, tok),
            token_window_seconds=max(1, win),
            max_wait_seconds=float(max(1, wait)),
        )
    return limits


@dataclass
class _ProviderState:
    limits: ProviderLimits
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Monotonic timestamps of recent request starts (for rpm sliding window).
    request_times: deque = field(default_factory=deque)
    # (monotonic_ts, tokens) for token-budget window.
    token_events: deque = field(default_factory=deque)
    # Counters for introspection / reporting.
    total_requests: int = 0
    total_tokens: int = 0
    total_wait_seconds: float = 0.0


class RateLimiter:
    """Async-safe per-provider rate limiter.

    Not safe across processes — every process gets its own counters. For a
    single-process dev workflow (CLI, Streamlit UI) that's sufficient; a
    multi-worker production setup would need a Redis-backed variant.
    """

    def __init__(self, limits: dict[str, ProviderLimits] | None = None):
        self._limits = limits if limits is not None else load_limits_from_env()
        self._states: dict[str, _ProviderState] = {
            p: _ProviderState(limits=lim) for p, lim in self._limits.items()
        }

    # ---- public API -----------------------------------------------------

    def limits_for(self, provider: str) -> ProviderLimits:
        return self._limits.get(provider, ProviderLimits())

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return current counters for every configured provider."""
        out: dict[str, dict[str, Any]] = {}
        now = time.monotonic()
        for p, s in self._states.items():
            win = s.limits.token_window_seconds
            tokens_in_window = sum(
                t for ts, t in s.token_events if now - ts <= win
            )
            out[p] = {
                "rpm_limit": s.limits.rpm,
                "token_budget": s.limits.token_budget,
                "token_window_seconds": win,
                "requests_total": s.total_requests,
                "tokens_total": s.total_tokens,
                "tokens_in_window": tokens_in_window,
                "queue_wait_total": round(s.total_wait_seconds, 2),
            }
        return out

    async def acquire(
        self, provider: str, estimated_tokens: int = 0
    ) -> float:
        """Block until the caller may proceed. Returns seconds waited.

        Raises :class:`RateLimitExceeded` if the wait would exceed
        ``limits.max_wait_seconds``. Token accounting is applied optimistically
        with ``estimated_tokens``; call :meth:`record_actual_usage` after the
        response lands to reconcile.
        """
        state = self._states.get(provider)
        if state is None:
            # Unknown provider = no limit. (Don't silently create state so
            # misspellings surface during snapshot.)
            return 0.0

        lim = state.limits
        waited_total = 0.0
        start = time.monotonic()

        while True:
            async with state.lock:
                now = time.monotonic()

                # ---- rpm (sliding 60s window) -------------------------
                wait_rpm = 0.0
                if lim.rpm > 0:
                    cutoff = now - 60.0
                    while state.request_times and state.request_times[0] < cutoff:
                        state.request_times.popleft()
                    if len(state.request_times) >= lim.rpm:
                        # Wait until the oldest slot rolls off.
                        wait_rpm = state.request_times[0] + 60.0 - now

                # ---- token budget (rolling window) --------------------
                wait_tok = 0.0
                if lim.token_budget > 0 and estimated_tokens > 0:
                    cutoff = now - lim.token_window_seconds
                    while state.token_events and state.token_events[0][0] < cutoff:
                        state.token_events.popleft()
                    in_window = sum(t for _, t in state.token_events)
                    if in_window + estimated_tokens > lim.token_budget:
                        # Earliest event that, once rolled off, frees enough.
                        needed = in_window + estimated_tokens - lim.token_budget
                        freed = 0
                        roll_off_at: float | None = None
                        for ts, tok in state.token_events:
                            freed += tok
                            if freed >= needed:
                                roll_off_at = ts + lim.token_window_seconds
                                break
                        if roll_off_at is not None:
                            wait_tok = roll_off_at - now
                        else:
                            # Request alone exceeds the whole budget.
                            raise RateLimitExceeded(
                                f"{provider}: estimated_tokens={estimated_tokens} "
                                f"exceeds token_budget={lim.token_budget}"
                            )

                wait = max(wait_rpm, wait_tok)
                if wait <= 0:
                    # Record and release.
                    state.request_times.append(now)
                    state.total_requests += 1
                    if estimated_tokens > 0:
                        state.token_events.append((now, estimated_tokens))
                    state.total_wait_seconds += waited_total
                    return waited_total

            # Out of lock: honour max_wait_seconds then sleep.
            elapsed = time.monotonic() - start
            remaining_budget = lim.max_wait_seconds - elapsed
            if remaining_budget <= 0:
                raise RateLimitExceeded(
                    f"{provider}: wait would exceed max_wait_seconds="
                    f"{lim.max_wait_seconds} (rpm={lim.rpm}, "
                    f"token_budget={lim.token_budget})"
                )
            sleep = min(wait + 0.01, remaining_budget)
            waited_total += sleep
            await asyncio.sleep(sleep)

    def record_actual_usage(self, provider: str, usage: dict[str, Any] | None) -> None:
        """Reconcile the estimated token count with what the provider reported.

        Accepts the ``usage`` dict returned on ``LLMResponse`` (shape varies by
        provider; we look for any of ``total_tokens`` / ``prompt_tokens`` +
        ``completion_tokens``). Does nothing if usage is missing.
        """
        state = self._states.get(provider)
        if state is None or not usage:
            return
        actual = 0
        for key in ("total_tokens", "totalTokenCount"):
            if key in usage and isinstance(usage[key], (int, float)):
                actual = int(usage[key])
                break
        if actual == 0:
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            actual = prompt + completion
        if actual <= 0:
            return
        # Replace the most recent estimated event with the actual count so
        # the window stays accurate. Guarded by the lock-free fast path —
        # a small race here is fine (off-by-one token doesn't matter).
        if state.token_events:
            ts, _est = state.token_events[-1]
            state.token_events[-1] = (ts, actual)
        else:
            state.token_events.append((time.monotonic(), actual))
        state.total_tokens += actual


# ---------- process-wide singleton ---------------------------------------

_singleton: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _singleton
    if _singleton is None:
        _singleton = RateLimiter()
    return _singleton


def reset_rate_limiter() -> None:
    """Drop the singleton so the next ``get_rate_limiter()`` re-reads env."""
    global _singleton
    _singleton = None


__all__ = [
    "ProviderLimits",
    "RateLimitExceeded",
    "RateLimiter",
    "get_rate_limiter",
    "load_limits_from_env",
    "reset_rate_limiter",
]
