"""Interactive plan review — CLI loop for user to approve/refine a WorkflowSpec."""

from __future__ import annotations

import asyncio

from daaw.compiler.compiler import Compiler
from daaw.schemas.workflow import WorkflowSpec


def _display_plan(spec: WorkflowSpec) -> None:
    """Print a human-readable plan summary."""
    print(f"\n{'='*60}")
    print(f"  Workflow: {spec.name}")
    print(f"  ID: {spec.id}")
    print(f"{'='*60}")
    print(f"  {spec.description}\n")

    for i, task in enumerate(spec.tasks, 1):
        deps = ", ".join(d.task_id for d in task.dependencies) or "(none)"
        print(f"  [{i}] {task.id}: {task.name}")
        print(f"      Agent: {task.agent.role}")
        print(f"      Deps:  {deps}")
        print(f"      Criteria: {task.success_criteria or '(none)'}")
        print()

    print(f"  Total tasks: {len(spec.tasks)}")
    print(f"{'='*60}")


async def interactive_plan_review(
    compiler: Compiler, initial_spec: WorkflowSpec
) -> WorkflowSpec | None:
    """Loop: display plan → prompt user → approve / abort / refine."""
    current = initial_spec

    while True:
        _display_plan(current)
        print("\n  Type 'yes' to approve, 'abort' to cancel,")
        print("  or describe changes you'd like to make.")
        print()

        user_input = (await asyncio.to_thread(input, "  [You]: ")).strip()

        if user_input.lower() in ("yes", "y"):
            print("\n  Plan approved!\n")
            return current

        if user_input.lower() == "abort":
            print("\n  Plan aborted.\n")
            return None

        print("\n  Refining plan based on your feedback...\n")
        current = await compiler.refine(current, user_input)

    return current
