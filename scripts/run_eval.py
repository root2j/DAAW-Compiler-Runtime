"""Batch-run the 25 evaluation prompts through the DAAW pipeline.

Reads ``docs/eval/test_prompts.md`` (or any file with the same P01-style
markers), runs each prompt through compile -> execute -> critic, and
writes a structured log tree. Designed for free-tier Gemini — rate
limits conservatively, saves progress after every prompt so a quota
kill mid-run doesn't lose work, and supports ``--skip-existing`` for
resume.

Output tree (default ``evals/runs/<timestamp>/``):

    summary.md                     human-readable summary with rubric table
    summary.json                   combined structured data for all prompts
    aggregate/
      all_outputs.json             every task output keyed by prompt+task
      metrics.csv                  per-prompt scalar metrics
    by_prompt/
      P01/
        meta.json                  prompt text, category, complexity
        spec.json                  compiled WorkflowSpec
        results.json               per-task status / output / elapsed
        verdicts.json              critic verdicts
        log.txt                    line-by-line execution log
        outputs/
          task_001.json            individual task output files
          task_002.json
          ...

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --only P01,P02,P03
    python scripts/run_eval.py --provider groq --model llama-3.1-8b-instant
    python scripts/run_eval.py --rpm 4 --sleep-between 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()


# ---------------------------------------------------------------------------
# Prompt parser
# ---------------------------------------------------------------------------
_PROMPT_HEADER_RE = re.compile(
    r"\*\*(P\d{2})\s*\|\s*(Simple|Medium|Complex)\s*\|\s*([^*]+?)\*\*",
    re.IGNORECASE,
)


@dataclass
class Prompt:
    id: str                # "P01"
    complexity: str        # "Simple" / "Medium" / "Complex"
    characteristics: str   # "Linear" / "Parallel" / "Conditional + Loop" / ...
    category: str          # "Email & Communication Automation"
    text: str              # the prompt itself (from the > quote)


def parse_prompts(md_path: Path) -> list[Prompt]:
    text = md_path.read_text(encoding="utf-8")
    prompts: list[Prompt] = []
    current_category = "Uncategorized"
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_cat = re.match(r"### Category \d+:\s*(.+?)(?:\s*\(|$)", line)
        if m_cat:
            current_category = m_cat.group(1).strip()
            i += 1
            continue
        m = _PROMPT_HEADER_RE.search(line)
        if m:
            pid = m.group(1).upper()
            complexity = m.group(2).strip().title()
            chars = m.group(3).strip()
            # The prompt body begins on the next non-blank line, prefixed with ">".
            body_lines: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                body_lines.append(lines[j].lstrip().lstrip(">").strip())
                j += 1
            if body_lines:
                prompts.append(Prompt(
                    id=pid, complexity=complexity,
                    characteristics=chars,
                    category=current_category,
                    text=" ".join(body_lines).strip(),
                ))
            i = j
            continue
        i += 1
    return prompts


# ---------------------------------------------------------------------------
# Per-prompt runner
# ---------------------------------------------------------------------------
@dataclass
class PromptRun:
    prompt: Prompt
    status: str = "pending"       # success / failure / skipped
    error: str = ""
    spec: dict | None = None
    results: dict[str, dict] = field(default_factory=dict)
    verdicts: list[dict] = field(default_factory=list)
    elapsed_compile: float = 0.0
    elapsed_execute: float = 0.0
    elapsed_critic: float = 0.0
    elapsed_total: float = 0.0
    api_calls_total: int = 0
    tokens_total: int = 0


async def run_one(
    prompt: Prompt,
    provider: str,
    model: str | None,
    out_dir: Path,
    log_lines: list[str],
) -> PromptRun:
    """Run compile -> execute -> critic for one prompt.

    Returns a ``PromptRun`` with structured data. Logging goes through the
    caller-provided ``log_lines`` so we can write it verbatim to log.txt.
    """
    run = PromptRun(prompt=prompt)

    # Lazy-import so a module-init failure doesn't abort the whole eval.
    from daaw.agents.factory import AgentFactory
    from daaw.compiler.compiler import Compiler
    from daaw.config import get_config, reset_config
    from daaw.critic.critic import Critic
    from daaw.engine.circuit_breaker import CircuitBreaker
    from daaw.engine.executor import DAGExecutor
    from daaw.llm.rate_limiter import reset_rate_limiter
    from daaw.llm.unified import UnifiedLLMClient
    from daaw.store.artifact_store import ArtifactStore

    import daaw.agents.builtin.breakdown_agent  # noqa: F401
    import daaw.agents.builtin.critic_agent  # noqa: F401
    import daaw.agents.builtin.generic_llm_agent  # noqa: F401
    import daaw.agents.builtin.planner_agent  # noqa: F401
    import daaw.agents.builtin.pm_agent  # noqa: F401
    import daaw.agents.builtin.user_proxy  # noqa: F401
    try:
        import daaw.tools.real_tools  # noqa: F401
    except ImportError:
        import daaw.tools.mock_tools  # noqa: F401

    reset_config()
    reset_rate_limiter()
    cfg = get_config()
    llm = UnifiedLLMClient(cfg)
    if provider not in llm.available_providers():
        run.status = "failure"
        run.error = (
            f"Provider '{provider}' not configured. "
            f"Available: {llm.available_providers()}"
        )
        log_lines.append(f"[ERROR] {run.error}")
        return run

    store = ArtifactStore(str(out_dir / ".artifact_store"))
    cb = CircuitBreaker(threshold=cfg.circuit_breaker_threshold)
    factory = AgentFactory(
        llm, store,
        default_provider=provider, default_model=model,
    )
    is_local = provider == "gateway"
    executor = DAGExecutor(factory, store, cb,
                           max_concurrent=1 if is_local else None)

    # ── Compile ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    compiler = Compiler(llm, cfg, provider=provider, model=model)
    log_lines.append(f"[compile] provider={provider} model={model}")
    try:
        spec = await compiler.compile(prompt.text)
    except Exception as e:  # noqa: BLE001
        run.elapsed_compile = time.monotonic() - t0
        run.status = "failure"
        run.error = f"compile: {type(e).__name__}: {e}"
        log_lines.append(f"[compile FAIL] {run.error}")
        return run
    run.elapsed_compile = time.monotonic() - t0
    run.spec = json.loads(spec.model_dump_json())
    log_lines.append(
        f"[compile OK] {len(spec.tasks)} tasks in {run.elapsed_compile:.1f}s",
    )
    for t in spec.tasks:
        deps = ", ".join(d.task_id for d in t.dependencies) or "-"
        tools = ", ".join(t.agent.tools_allowed) or "-"
        log_lines.append(
            f"          · {t.id}: {t.name}  [deps={deps}] [tools={tools}]",
        )

    # ── Execute ─────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        results = await executor.execute(spec)
    except Exception as e:  # noqa: BLE001
        run.elapsed_execute = time.monotonic() - t0
        run.status = "failure"
        run.error = f"execute: {type(e).__name__}: {e}"
        log_lines.append(f"[execute FAIL] {run.error}")
        return run
    run.elapsed_execute = time.monotonic() - t0
    for tid, r in results.items():
        meta = r.agent_result.metadata or {}
        run.results[tid] = {
            "status": r.agent_result.status,
            "output": r.agent_result.output,
            "error_message": getattr(r.agent_result, "error_message", "") or "",
            "elapsed_seconds": r.elapsed_seconds,
            "tool_calls": meta.get("tool_calls", []),
            "usage": meta.get("usage"),
        }
        run.api_calls_total += len(meta.get("tool_calls", []) or []) + 1
        log_lines.append(
            f"[task {r.agent_result.status}] {tid} "
            f"({r.elapsed_seconds:.1f}s, "
            f"{len(meta.get('tool_calls', []) or [])} tool calls)",
        )

    # ── Critic ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    critic = Critic(llm, cfg, provider=provider, model=model)
    for task in spec.tasks:
        if task.id not in results:
            continue
        try:
            passed, _patch, reason = await critic.evaluate(
                task, results[task.id],
            )
        except Exception as e:  # noqa: BLE001
            passed, reason = False, f"critic error: {e}"
        run.verdicts.append({
            "task_id": task.id,
            "task_name": task.name,
            "verdict": "PASS" if passed else "FAIL",
            "reasoning": reason or "",
        })
        log_lines.append(
            f"[critic {'PASS' if passed else 'FAIL'}] {task.id}  "
            f"— {reason[:80] if reason else ''}",
        )
    run.elapsed_critic = time.monotonic() - t0

    # ── Token usage ─────────────────────────────────────────────────────
    try:
        snap = llm.rate_limiter.snapshot()
        if provider in snap:
            run.tokens_total = snap[provider].get("tokens_total", 0)
            run.api_calls_total = snap[provider].get("requests_total",
                                                     run.api_calls_total)
    except Exception:
        pass

    run.elapsed_total = (
        run.elapsed_compile + run.elapsed_execute + run.elapsed_critic
    )
    run.status = "success"
    log_lines.append(
        f"[DONE] total {run.elapsed_total:.1f}s  "
        f"api_calls={run.api_calls_total}  tokens={run.tokens_total}",
    )
    return run


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "x"


def write_prompt_run(run_dir: Path, run: PromptRun) -> None:
    pdir = run_dir / "by_prompt" / run.prompt.id
    outputs_dir = pdir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    (pdir / "meta.json").write_text(json.dumps({
        "id": run.prompt.id,
        "category": run.prompt.category,
        "complexity": run.prompt.complexity,
        "characteristics": run.prompt.characteristics,
        "text": run.prompt.text,
    }, indent=2), encoding="utf-8")

    if run.spec is not None:
        (pdir / "spec.json").write_text(
            json.dumps(run.spec, indent=2, default=str), encoding="utf-8",
        )

    (pdir / "results.json").write_text(
        json.dumps(run.results, indent=2, default=str), encoding="utf-8",
    )
    (pdir / "verdicts.json").write_text(
        json.dumps(run.verdicts, indent=2, default=str), encoding="utf-8",
    )
    for tid, res in run.results.items():
        (outputs_dir / f"{_safe(tid)}.json").write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8",
        )


def write_aggregate(run_dir: Path, runs: list[PromptRun]) -> None:
    agg_dir = run_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    all_outputs: dict[str, Any] = {}
    for r in runs:
        all_outputs[r.prompt.id] = {
            "prompt": r.prompt.text,
            "category": r.prompt.category,
            "complexity": r.prompt.complexity,
            "characteristics": r.prompt.characteristics,
            "status": r.status,
            "task_count": len(r.results),
            "outputs": {
                tid: res.get("output") for tid, res in r.results.items()
            },
        }
    (agg_dir / "all_outputs.json").write_text(
        json.dumps(all_outputs, indent=2, default=str), encoding="utf-8",
    )

    # metrics.csv — scalars for later rubric scoring
    with (agg_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "category", "complexity", "characteristics", "status",
            "task_count", "tasks_success", "tasks_failed",
            "verdicts_pass", "verdicts_fail",
            "elapsed_compile_s", "elapsed_execute_s", "elapsed_critic_s",
            "elapsed_total_s", "api_calls", "tokens", "error",
        ])
        for r in runs:
            ts_ok = sum(1 for v in r.results.values() if v["status"] == "success")
            ts_fail = len(r.results) - ts_ok
            v_pass = sum(1 for v in r.verdicts if v["verdict"] == "PASS")
            v_fail = len(r.verdicts) - v_pass
            w.writerow([
                r.prompt.id, r.prompt.category, r.prompt.complexity,
                r.prompt.characteristics, r.status, len(r.results),
                ts_ok, ts_fail, v_pass, v_fail,
                f"{r.elapsed_compile:.1f}", f"{r.elapsed_execute:.1f}",
                f"{r.elapsed_critic:.1f}", f"{r.elapsed_total:.1f}",
                r.api_calls_total, r.tokens_total, r.error[:120],
            ])


def write_summary(run_dir: Path, runs: list[PromptRun],
                  provider: str, model: str | None) -> None:
    # summary.json — flat dump of PromptRun dataclasses
    data = []
    for r in runs:
        data.append({
            "id": r.prompt.id,
            "category": r.prompt.category,
            "complexity": r.prompt.complexity,
            "characteristics": r.prompt.characteristics,
            "prompt_text": r.prompt.text,
            "status": r.status,
            "error": r.error,
            "task_count": len(r.results),
            "verdicts": r.verdicts,
            "elapsed": {
                "compile": round(r.elapsed_compile, 2),
                "execute": round(r.elapsed_execute, 2),
                "critic": round(r.elapsed_critic, 2),
                "total": round(r.elapsed_total, 2),
            },
            "api_calls_total": r.api_calls_total,
            "tokens_total": r.tokens_total,
        })
    (run_dir / "summary.json").write_text(
        json.dumps({
            "provider": provider, "model": model,
            "generated_at": datetime.now().isoformat(),
            "prompts": data,
        }, indent=2, default=str), encoding="utf-8",
    )

    # summary.md — human-readable
    lines: list[str] = []
    lines.append(f"# DAAW eval — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"Provider: `{provider}` · Model: `{model or 'default'}`")
    lines.append("")
    succ = sum(1 for r in runs if r.status == "success")
    fail = sum(1 for r in runs if r.status == "failure")
    skip = sum(1 for r in runs if r.status == "skipped")
    total_tokens = sum(r.tokens_total for r in runs)
    total_calls = sum(r.api_calls_total for r in runs)
    total_time = sum(r.elapsed_total for r in runs)
    lines.append(
        f"Ran **{len(runs)}** prompts: {succ} success · {fail} failed · "
        f"{skip} skipped"
    )
    lines.append(
        f"Total wall time: {total_time:.0f}s · "
        f"API calls: {total_calls} · Tokens: {total_tokens}"
    )
    lines.append("")
    lines.append("## Per-prompt summary")
    lines.append("")
    lines.append("| ID | Complexity | Status | Tasks | Pass/Fail | "
                 "Compile s | Exec s | Total s |")
    lines.append("|----|-----------|--------|-------|-----------|"
                 "-----------|--------|---------|")
    for r in runs:
        v_pass = sum(1 for v in r.verdicts if v["verdict"] == "PASS")
        v_tot = len(r.verdicts)
        lines.append(
            f"| {r.prompt.id} | {r.prompt.complexity} | {r.status} | "
            f"{len(r.results)} | {v_pass}/{v_tot} | "
            f"{r.elapsed_compile:.1f} | {r.elapsed_execute:.1f} | "
            f"{r.elapsed_total:.1f} |"
        )
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    any_fail = False
    for r in runs:
        if r.status == "failure":
            any_fail = True
            lines.append(f"- **{r.prompt.id}** — `{r.error}`")
    if not any_fail:
        lines.append("None.")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=Path, default=None,
                    help="Path to test_prompts.md. Default: shipped copy in docs/eval/.")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--rpm", type=int, default=6,
                    help="Override DAAW_RPM_<PROVIDER> for the run (free-tier safe).")
    ap.add_argument("--sleep-between", type=float, default=3.0,
                    help="Seconds to sleep between prompts.")
    ap.add_argument("--only", default="",
                    help="Comma-separated list of prompt IDs to run "
                         "(e.g. 'P01,P05'). Default: all.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip prompts that already have results on disk.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: evals/runs/<timestamp>/.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse prompts and print them; don't call any API.")
    args = ap.parse_args()

    # Apply rate limit via env BEFORE DAAW modules import their config.
    os.environ[f"DAAW_RPM_{args.provider.upper()}"] = str(args.rpm)

    # Locate prompts file.
    if args.prompts is None:
        for candidate in [
            Path(_REPO) / "docs" / "eval" / "test_prompts.md",
            Path(_REPO) / "test_prompts.md",
            Path.home() / "Downloads" / "test_prompts.md",
        ]:
            if candidate.exists():
                args.prompts = candidate
                break
    if args.prompts is None or not args.prompts.exists():
        print("Could not find test_prompts.md. Pass --prompts.", file=sys.stderr)
        return 1
    prompts = parse_prompts(args.prompts)
    print(f"Parsed {len(prompts)} prompts from {args.prompts}")
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        prompts = [p for p in prompts if p.id in want]
        print(f"Filtered to {len(prompts)} prompts: "
              f"{[p.id for p in prompts]}")

    if args.dry_run:
        for p in prompts:
            print(f"  {p.id} [{p.complexity}] ({p.characteristics}) "
                  f"{p.category}: {p.text[:80]}...")
        return 0

    # Output dir.
    run_dir = args.out_dir
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(_REPO) / "evals" / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {run_dir}")

    runs: list[PromptRun] = []
    for i, prompt in enumerate(prompts, 1):
        pdir = run_dir / "by_prompt" / prompt.id
        if args.skip_existing and (pdir / "results.json").exists():
            print(f"[{i}/{len(prompts)}] {prompt.id} -- skipping (exists)")
            continue

        print(f"\n[{i}/{len(prompts)}] {prompt.id} "
              f"[{prompt.complexity}/{prompt.characteristics}]  "
              f"{prompt.text[:70]}...")

        log_lines: list[str] = [
            f"# {prompt.id} — {prompt.category}",
            f"# complexity: {prompt.complexity}",
            f"# characteristics: {prompt.characteristics}",
            f"# prompt: {prompt.text}",
            "",
        ]
        t0 = time.monotonic()
        try:
            run = await run_one(prompt, args.provider, args.model,
                                run_dir, log_lines)
        except Exception as e:  # noqa: BLE001
            run = PromptRun(prompt=prompt, status="failure",
                            error=f"outer: {type(e).__name__}: {e}")
            log_lines.append(
                f"[OUTER ERROR] {run.error}\n{traceback.format_exc()}",
            )
        elapsed = time.monotonic() - t0
        print(f"  -> {run.status}  "
              f"tasks={len(run.results)}  "
              f"elapsed={elapsed:.1f}s  "
              f"api_calls={run.api_calls_total}")

        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "log.txt").write_text("\n".join(log_lines), encoding="utf-8")
        write_prompt_run(run_dir, run)
        runs.append(run)

        # Save incremental aggregate + summary after every prompt so a
        # mid-run quota kill doesn't lose everything.
        write_aggregate(run_dir, runs)
        write_summary(run_dir, runs, args.provider, args.model)

        if i < len(prompts):
            await asyncio.sleep(args.sleep_between)

    print(f"\nDone. Results in {run_dir}")
    print(f"  summary.md          — human-readable report")
    print(f"  summary.json        — structured dump")
    print(f"  aggregate/          — combined outputs + metrics CSV")
    print(f"  by_prompt/PXX/      — per-prompt spec / results / verdicts / outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
