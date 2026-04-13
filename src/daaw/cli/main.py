"""CLI entry point — ties the full Compiler-Runtime pipeline together."""

from __future__ import annotations

import argparse
import asyncio
import sys

# ── Import builtin agents to trigger @register_agent decorators ──
import daaw.agents.builtin.breakdown_agent  # noqa: F401
import daaw.agents.builtin.critic_agent  # noqa: F401
import daaw.agents.builtin.generic_llm_agent  # noqa: F401
import daaw.agents.builtin.planner_agent  # noqa: F401
import daaw.agents.builtin.pm_agent  # noqa: F401
import daaw.agents.builtin.user_proxy  # noqa: F401

# ── Tools are registered dynamically in main() based on --mock-tools flag ──

from daaw.agents.factory import AgentFactory
from daaw.cli.display import (
    display_critic_verdict,
    display_execution_summary,
    display_patch_applied,
    display_workflow_plan,
)
from daaw.compiler.compiler import Compiler
from daaw.compiler.plan_reviewer import interactive_plan_review
from daaw.config import AppConfig, get_config
from daaw.critic.critic import Critic
from daaw.critic.patch import apply_patch
from daaw.engine.circuit_breaker import CircuitBreaker
from daaw.engine.dag import DAG
from daaw.engine.executor import DAGExecutor
from daaw.interaction import StdinInteractionHandler
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.workflow import AgentSpec, DependencySpec, TaskSpec, WorkflowSpec
from daaw.store.artifact_store import ArtifactStore


# ─────────────────────────────────────────────────────────────
# Full pipeline: goal → compile → review → execute → critique
# ─────────────────────────────────────────────────────────────


async def run_full_pipeline(
    goal: str, provider: str, model: str | None, config: AppConfig
) -> None:
    llm = UnifiedLLMClient(config)
    store = ArtifactStore(config.artifact_store_dir)
    cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
    interaction = StdinInteractionHandler()
    factory = AgentFactory(
        llm, store, default_provider=provider, interaction_handler=interaction
    )
    # Local LLMs (gateway) can only handle one request at a time
    max_conc = 1 if provider == "gateway" else None
    executor = DAGExecutor(factory, store, cb, max_concurrent=max_conc)

    print(f"\nAvailable LLM providers: {', '.join(llm.available_providers())}\n")

    # ── 1. Compile
    if not goal:
        goal = (await asyncio.to_thread(input, "Enter your workflow goal: ")).strip()
        if not goal:
            print("No goal provided. Exiting.")
            return

    print(f"\nCompiling workflow for: {goal}\n")
    compiler = Compiler(llm, config, provider=provider, model=model)
    spec = await compiler.compile(goal)

    # ── 2. Review
    display_workflow_plan(spec)
    spec = await interactive_plan_review(compiler, spec)
    if spec is None:
        print("Workflow aborted.")
        return

    # ── 3. Execute
    print("\nExecuting workflow...\n")
    results = await executor.execute(spec)

    # ── 4. Critique
    critic = Critic(llm, config, provider=provider, model=model)
    dag = DAG(spec)
    # Sync DAG statuses from results
    for tid, res in results.items():
        if res.agent_result.status == "success":
            from daaw.schemas.enums import TaskStatus
            dag.mark(tid, TaskStatus.SUCCESS)
        elif res.agent_result.status == "failure":
            dag.mark(tid, TaskStatus.FAILURE)

    for task in spec.tasks:
        if task.id not in results:
            continue
        result = results[task.id]
        passed, patch, _reasoning = await critic.evaluate(task, result)
        display_critic_verdict(task.id, passed)

        if not passed and patch:
            logs = await apply_patch(patch, dag, executor, store)
            display_patch_applied(logs)

    # ── 5. Summary
    # Re-gather results after any retries
    final_results = executor._results
    display_execution_summary(final_results)


# ─────────────────────────────────────────────────────────────
# Legacy pipeline: questionnaire → PM → breakdown
# ─────────────────────────────────────────────────────────────


async def run_legacy_pipeline(config: AppConfig) -> None:
    llm = UnifiedLLMClient(config)
    store = ArtifactStore(config.artifact_store_dir)
    interaction = StdinInteractionHandler()
    factory = AgentFactory(llm, store, interaction_handler=interaction)

    # Build a simple 3-task sequential workflow
    spec = WorkflowSpec(
        name="Legacy Pipeline",
        description="Original questionnaire -> PM -> breakdown flow",
        tasks=[
            TaskSpec(
                id="task_001",
                name="User Intake",
                description="Collect project requirements via questionnaire",
                agent=AgentSpec(role="user_proxy"),
                dependencies=[],
            ),
            TaskSpec(
                id="task_002",
                name="PM Refinement",
                description="Clarify and produce a detailed project draft",
                agent=AgentSpec(role="pm"),
                dependencies=[DependencySpec(task_id="task_001")],
            ),
            TaskSpec(
                id="task_003",
                name="Task Breakdown",
                description="Break the draft into detailed subtasks",
                agent=AgentSpec(role="breakdown"),
                dependencies=[DependencySpec(task_id="task_002")],
            ),
        ],
    )

    display_workflow_plan(spec)

    cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
    executor = DAGExecutor(factory, store, cb)
    results = await executor.execute(spec)
    display_execution_summary(results)


# ─────────────────────────────────────────────────────────────
# Streamlit UI launcher
# ─────────────────────────────────────────────────────────────


def launch_ui(port: int = 8501) -> None:
    """Launch the Streamlit dashboard UI."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    if not app_path.exists():
        print(f"UI app not found at {app_path}")
        sys.exit(1)

    print(f"\nLaunching DAAW Dashboard on port {port}...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", str(port), "--server.headless", "true"],
    )


def launch_demo_ui(port: int = 8502) -> None:
    """Launch the Under-the-Hood demonstration UI."""
    import subprocess
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "ui" / "demo_app.py"
    if not app_path.exists():
        print(f"Demo UI app not found at {app_path}")
        sys.exit(1)

    print(f"\nLaunching DAAW — Under the Hood on port {port}...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.port", str(port), "--server.headless", "true"],
    )


# ─────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daaw",
        description="DAAW Compiler-Runtime -- workflow automation",
    )
    sub = parser.add_subparsers(dest="command")

    # daaw run
    run_parser = sub.add_parser("run", help="Full pipeline: compile -> execute -> critique")
    run_parser.add_argument("--goal", type=str, default="", help="Workflow goal")
    run_parser.add_argument("--provider", type=str, default="groq", help="LLM provider")
    run_parser.add_argument("--model", type=str, default=None, help="LLM model override")
    run_parser.add_argument("--mock-tools", action="store_true", help="Use mock tools instead of real ones")

    # daaw legacy
    sub.add_parser("legacy", help="Original questionnaire -> PM -> breakdown pipeline")

    # daaw ui
    ui_parser = sub.add_parser("ui", help="Launch the Streamlit dashboard UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port to run on")

    # daaw demo
    demo_parser = sub.add_parser("demo", help="Launch the Under-the-Hood demonstration UI")
    demo_parser.add_argument("--port", type=int, default=8502, help="Port to run on")

    return parser


def _register_tools(use_mock: bool = False) -> None:
    """Register tools based on mode."""
    if use_mock:
        import daaw.tools.mock_tools  # noqa: F401
    else:
        import daaw.tools.real_tools  # noqa: F401


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = get_config()

    if args.command == "run":
        _register_tools(use_mock=getattr(args, "mock_tools", False))
        asyncio.run(run_full_pipeline(args.goal, args.provider, args.model, config))
    elif args.command == "legacy":
        _register_tools(use_mock=True)
        asyncio.run(run_legacy_pipeline(config))
    elif args.command == "ui":
        launch_ui(args.port)
    elif args.command == "demo":
        launch_demo_ui(args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
