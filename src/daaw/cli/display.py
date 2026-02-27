"""Plain text display helpers for the CLI."""

from __future__ import annotations

from daaw.schemas.results import TaskResult
from daaw.schemas.workflow import WorkflowSpec


def display_workflow_plan(spec: WorkflowSpec) -> None:
    """Print a table-like summary of the workflow plan."""
    print(f"\n{'='*60}")
    print(f"  Workflow Plan: {spec.name}")
    print(f"  ID: {spec.id}")
    print(f"{'='*60}")
    print(f"  {spec.description}\n")

    for i, task in enumerate(spec.tasks, 1):
        deps = ", ".join(d.task_id for d in task.dependencies) or "(start)"
        tools = ", ".join(task.agent.tools_allowed) or "(none)"
        print(f"  Task {i}: {task.id}")
        print(f"    Name:     {task.name}")
        print(f"    Agent:    {task.agent.role}")
        print(f"    Deps:     {deps}")
        print(f"    Tools:    {tools}")
        print(f"    Criteria: {task.success_criteria or '(none)'}")
        print(f"    Timeout:  {task.timeout_seconds}s | Retries: {task.max_retries}")
        print()

    print(f"  Total: {len(spec.tasks)} tasks")
    print(f"{'='*60}\n")


def display_task_start(task_id: str, task_name: str) -> None:
    print(f"  [START] {task_id}: {task_name}")


def display_task_result(task_id: str, task_name: str, result: TaskResult) -> None:
    icon = "OK" if result.agent_result.status == "success" else "FAIL"
    print(f"  [{icon}] {task_id}: {task_name} ({result.elapsed_seconds:.1f}s)")


def display_execution_summary(results: dict[str, TaskResult]) -> None:
    """Print a summary of all task results."""
    print(f"\n{'='*60}")
    print("  Execution Summary")
    print(f"{'='*60}")

    successes = sum(1 for r in results.values() if r.agent_result.status == "success")
    failures = sum(1 for r in results.values() if r.agent_result.status == "failure")
    total = len(results)

    print(f"  Total: {total} | Success: {successes} | Failure: {failures}\n")

    for task_id, result in results.items():
        icon = "OK" if result.agent_result.status == "success" else "FAIL"
        output_preview = str(result.agent_result.output)[:100]
        print(f"  [{icon}] {task_id} ({result.elapsed_seconds:.1f}s)")
        if result.agent_result.error_message:
            print(f"        Error: {result.agent_result.error_message}")
        else:
            print(f"        Output: {output_preview}...")
        print()

    print(f"{'='*60}\n")


def display_critic_verdict(
    task_id: str, passed: bool, reasoning: str = ""
) -> None:
    icon = "PASS" if passed else "FAIL"
    print(f"  [CRITIC {icon}] {task_id}")
    if reasoning:
        print(f"    Reasoning: {reasoning}")


def display_patch_applied(logs: list[str]) -> None:
    print("  [PATCH APPLIED]")
    for log in logs:
        print(f"    {log}")
