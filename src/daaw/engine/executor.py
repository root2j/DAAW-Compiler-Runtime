"""DAG Executor — async parallel execution of workflow tasks."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from daaw.agents.factory import AgentFactory
from daaw.engine.circuit_breaker import CircuitBreaker
from daaw.engine.context_pruner import prune_context
from daaw.engine.dag import DAG
from daaw.schemas.enums import TaskStatus
from daaw.schemas.results import AgentResult, TaskResult
from daaw.schemas.workflow import WorkflowSpec
from daaw.store.artifact_store import ArtifactStore


class DAGExecutor:
    """Executes a WorkflowSpec as a parallel DAG."""

    def __init__(
        self,
        factory: AgentFactory,
        store: ArtifactStore,
        circuit_breaker: CircuitBreaker | None = None,
        max_concurrent: int | None = None,
    ):
        self._factory = factory
        self._store = store
        self._cb = circuit_breaker or CircuitBreaker()
        self._results: dict[str, TaskResult] = {}
        self._max_concurrent = max_concurrent

    async def execute(self, spec: WorkflowSpec) -> dict[str, TaskResult]:
        """Run all tasks respecting dependencies. Returns results dict."""
        if not spec.tasks:
            return {}

        dag = DAG(spec)
        errors = dag.validate()
        if errors:
            raise ValueError(f"Invalid workflow DAG: {'; '.join(errors)}")

        self._results = {}

        while not dag.is_complete():
            ready = dag.get_ready_tasks()

            if not ready and dag.has_failures():
                # Failures blocking progress — break for Critic
                break

            if not ready and not dag.has_failures():
                raise RuntimeError("Deadlock: no ready tasks and no failures")

            if self._max_concurrent == 1:
                # Sequential execution — for local LLMs that can't handle
                # concurrent requests (LM Studio, Ollama single-GPU, etc.)
                for tid in ready:
                    await self._run_task(dag, spec.get_task(tid))
            elif self._max_concurrent and self._max_concurrent > 1:
                sem = asyncio.Semaphore(self._max_concurrent)

                async def _guarded(tid: str) -> None:
                    async with sem:
                        await self._run_task(dag, spec.get_task(tid))

                await asyncio.gather(*[_guarded(tid) for tid in ready])
            else:
                await asyncio.gather(
                    *[self._run_task(dag, spec.get_task(tid)) for tid in ready]
                )

        return self._results

    async def _run_task(self, dag: DAG, task: Any) -> None:
        """Execute a single task: create agent, run, store results."""
        task_id = task.id
        dag.mark(task_id, TaskStatus.RUNNING)
        print(f"  [START] {task_id}: {task.name}")

        # Enforce minimum timeouts.
        # Local models (max_concurrent=1) are much slower than cloud APIs
        # and need generous timeouts to avoid premature kills.
        effective_timeout = task.timeout_seconds
        is_local = self._max_concurrent == 1
        if task.agent.tools_allowed:
            effective_timeout = max(effective_timeout, 600 if is_local else 300)
        else:
            effective_timeout = max(effective_timeout, 300 if is_local else 60)

        start = time.monotonic()
        try:
            dep_context = await prune_context(task, self._store)
            agent = self._factory.create(task_id, task.agent)

            # Always include the task goal so the agent knows what to do.
            # Dependency outputs are nested under "dependency_outputs" when present.
            agent_input: dict = {
                "task": task.description,
                "success_criteria": task.success_criteria,
            }
            if dep_context:
                agent_input["dependency_outputs"] = dep_context

            result = await asyncio.wait_for(
                agent.run(agent_input), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                status="failure",
                error_message=f"Task timed out after {effective_timeout}s (spec: {task.timeout_seconds}s)",
            )
        except Exception as e:
            result = AgentResult(status="failure", error_message=str(e))

        elapsed = time.monotonic() - start

        # Store artifacts
        await self._store.put(f"{task_id}.output", result.output)
        await self._store.put(f"{task_id}.status", result.status)
        await self._store.put(f"{task_id}.metadata", result.metadata)

        # Update DAG status
        if result.status == "success":
            dag.mark(task_id, TaskStatus.SUCCESS)
            self._cb.record_success(task_id)
        elif result.status == "needs_human":
            dag.mark(task_id, TaskStatus.NEEDS_HUMAN)
        else:
            tripped = self._cb.record_failure(task_id)
            dag.mark(task_id, TaskStatus.FAILURE)
            if tripped:
                print(f"  [CIRCUIT BREAKER] {task_id} — too many failures")

        task_result = TaskResult(
            task_id=task_id,
            agent_result=result,
            attempt=1,
            elapsed_seconds=round(elapsed, 2),
        )
        self._results[task_id] = task_result

        status_icon = "OK" if result.status == "success" else "FAIL"
        print(f"  [{status_icon}] {task_id}: {task.name} ({elapsed:.1f}s)")

    async def retry_task(self, dag: DAG, task_id: str, feedback: str = "") -> None:
        """Store feedback, reset task, re-run."""
        if feedback:
            await self._store.put(f"{task_id}.feedback", feedback)
        dag.reset_task(task_id)
        self._cb.reset(task_id)
        task = dag.get_task(task_id)
        await self._run_task(dag, task)
