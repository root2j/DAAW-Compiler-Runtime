"""Tiny live smoke test across every provider with configured keys.

Runs a 2-task workflow per available provider under strict per-provider
rate limits so we don't accidentally burn through a free tier.

Usage:
    python scripts/live_smoke.py                 # all providers with keys
    python scripts/live_smoke.py --only groq     # single provider
    python scripts/live_smoke.py --max-calls 3   # override cap

Prints a per-provider pass/fail table plus total API calls and token spend.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Make `daaw` importable when run from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Load .env before reading config.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Force a very low per-provider RPM so the rate limiter actually exercises.
# Callers can override via env before invoking this script.
os.environ.setdefault("DAAW_RPM_GROQ", "10")
os.environ.setdefault("DAAW_RPM_GEMINI", "10")
os.environ.setdefault("DAAW_RPM_OPENAI", "10")
os.environ.setdefault("DAAW_RPM_ANTHROPIC", "10")
os.environ.setdefault("DAAW_MAX_WAIT_GROQ", "60")
os.environ.setdefault("DAAW_MAX_WAIT_GEMINI", "60")

import daaw.agents.builtin.generic_llm_agent  # noqa: E402,F401
import daaw.tools.real_tools  # noqa: E402,F401
from daaw.agents.factory import AgentFactory  # noqa: E402
from daaw.config import get_config, reset_config  # noqa: E402
from daaw.engine.circuit_breaker import CircuitBreaker  # noqa: E402
from daaw.engine.executor import DAGExecutor  # noqa: E402
from daaw.llm.rate_limiter import get_rate_limiter, reset_rate_limiter  # noqa: E402
from daaw.llm.unified import UnifiedLLMClient  # noqa: E402
from daaw.schemas.workflow import (  # noqa: E402
    AgentSpec, DependencySpec, TaskSpec, WorkflowSpec,
)
from daaw.store.artifact_store import ArtifactStore  # noqa: E402


# Minimum-cost models per provider for the smoke run.
_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash-lite",
    "openai": "gpt-4.1-nano",
    "anthropic": "claude-haiku-4-5-20251001",
    "gateway": None,  # use whatever gateway default is
}


def _build_spec(provider: str) -> WorkflowSpec:
    """Dead-simple 2-task workflow: one-sentence answer + one-sentence rephrase."""
    return WorkflowSpec(
        name=f"smoke-{provider}",
        description=f"Live smoke test for provider {provider}",
        tasks=[
            TaskSpec(
                id="t1",
                name="One-line summary",
                description=(
                    "Answer in ONE short sentence: what is the capital of Japan? "
                    "Reply with just the sentence, nothing else."
                ),
                agent=AgentSpec(role="generic_llm"),
                success_criteria="Single sentence containing 'Tokyo'.",
                timeout_seconds=60,
                max_retries=1,
            ),
            TaskSpec(
                id="t2",
                name="Rephrase",
                description=(
                    "Rephrase the previous answer as a question in ONE sentence. "
                    "Reply with just the sentence."
                ),
                agent=AgentSpec(role="generic_llm"),
                dependencies=[DependencySpec(task_id="t1")],
                success_criteria="Single sentence ending with '?'.",
                timeout_seconds=60,
                max_retries=1,
            ),
        ],
    )


async def _run_one(provider: str, model: str | None, tmpdir: str) -> dict:
    reset_config()  # pick up any env changes
    reset_rate_limiter()
    cfg = get_config()
    llm = UnifiedLLMClient(cfg)
    if provider not in llm.available_providers():
        return {
            "provider": provider, "status": "skip",
            "reason": "provider not configured",
            "tasks": 0, "elapsed": 0.0,
        }

    store = ArtifactStore(os.path.join(tmpdir, provider))
    factory = AgentFactory(
        llm, store, default_provider=provider, default_model=model,
    )
    executor = DAGExecutor(factory, store, CircuitBreaker(),
                           max_concurrent=1 if provider == "gateway" else None)
    spec = _build_spec(provider)

    start = time.monotonic()
    try:
        results = await asyncio.wait_for(executor.execute(spec), timeout=180)
    except asyncio.TimeoutError:
        return {"provider": provider, "status": "timeout",
                "reason": "execute() exceeded 180s", "tasks": 0,
                "elapsed": round(time.monotonic() - start, 1)}
    except Exception as e:  # noqa: BLE001
        return {"provider": provider, "status": "error",
                "reason": f"{type(e).__name__}: {str(e)[:120]}",
                "tasks": 0, "elapsed": round(time.monotonic() - start, 1)}

    elapsed = time.monotonic() - start
    passed = sum(1 for r in results.values() if r.agent_result.status == "success")
    total = len(results)
    rl = llm.rate_limiter.snapshot().get(provider, {})
    outputs = {
        tid: (str(r.agent_result.output)[:80] if r.agent_result.output else "")
        for tid, r in results.items()
    }
    return {
        "provider": provider,
        "status": "pass" if passed == total else "fail",
        "reason": "" if passed == total else
                  f"{total - passed} task(s) failed",
        "tasks": f"{passed}/{total}",
        "elapsed": round(elapsed, 1),
        "api_calls": rl.get("requests_total", 0),
        "tokens": rl.get("tokens_total", 0),
        "outputs": outputs,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run just one provider")
    parser.add_argument("--tmpdir", default=None,
                        help="Artifact store dir (default: tempfile.mkdtemp)")
    args = parser.parse_args()

    import tempfile
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="daaw_smoke_")

    cfg = get_config()
    llm = UnifiedLLMClient(cfg)
    providers_with_keys = llm.available_providers()
    if args.only:
        providers_with_keys = [p for p in providers_with_keys if p == args.only]
    if not providers_with_keys:
        print("No providers configured. Set at least one *_API_KEY in .env.")
        return 1

    print(f"Providers to test: {providers_with_keys}")
    print(f"Tmpdir: {tmpdir}\n")
    print(f"{'PROVIDER':<12} {'STATUS':<8} {'TASKS':<6} {'CALLS':<6} "
          f"{'TOKENS':<8} {'TIME':<7} REASON / OUTPUT")
    print("-" * 110)

    results: list[dict] = []
    for p in providers_with_keys:
        model = _MODELS.get(p)
        res = await _run_one(p, model, tmpdir)
        results.append(res)
        detail = res.get("reason") or (
            list(res.get("outputs", {}).values())[-1]
            if res.get("outputs") else ""
        )
        print(f"{res['provider']:<12} {res['status']:<8} "
              f"{str(res.get('tasks', '-')):<6} "
              f"{str(res.get('api_calls', '-')):<6} "
              f"{str(res.get('tokens', '-')):<8} "
              f"{res.get('elapsed', 0):<7.1f} {detail[:60]}")

    failures = [r for r in results if r["status"] not in ("pass", "skip")]
    skipped = [r for r in results if r["status"] == "skip"]
    print()
    print(f"Summary: {len(results) - len(failures) - len(skipped)} passed, "
          f"{len(failures)} failed, {len(skipped)} skipped.")
    total_calls = sum(int(r.get("api_calls", 0) or 0) for r in results)
    total_tokens = sum(int(r.get("tokens", 0) or 0) for r in results)
    print(f"Total API calls: {total_calls}   Total tokens: {total_tokens}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
