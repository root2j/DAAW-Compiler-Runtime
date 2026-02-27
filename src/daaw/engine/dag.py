"""DAG — adjacency list, topological validation (Kahn's), status tracking, mutation."""

from __future__ import annotations

from collections import deque

from daaw.schemas.enums import TaskStatus
from daaw.schemas.workflow import TaskSpec, WorkflowSpec

TERMINAL_STATES = {TaskStatus.SUCCESS, TaskStatus.FAILURE, TaskStatus.SKIPPED}


class DAG:
    """Directed acyclic graph built from a WorkflowSpec."""

    def __init__(self, spec: WorkflowSpec):
        self._spec = spec
        self._tasks: dict[str, TaskSpec] = {t.id: t for t in spec.tasks}
        self._statuses: dict[str, TaskStatus] = {
            t.id: TaskStatus.PENDING for t in spec.tasks
        }

        # adjacency: task_id -> list of task_ids it BLOCKS (downstream)
        self._adj: dict[str, list[str]] = {t.id: [] for t in spec.tasks}
        # in_degree: how many upstream deps each task has
        self._in_degree: dict[str, int] = {t.id: 0 for t in spec.tasks}

        for task in spec.tasks:
            for dep in task.dependencies:
                if dep.task_id in self._adj:
                    self._adj[dep.task_id].append(task.id)
                self._in_degree[task.id] = self._in_degree.get(task.id, 0) + 1

    @property
    def spec(self) -> WorkflowSpec:
        return self._spec

    def validate(self) -> list[str]:
        """Validate the DAG. Returns a list of error messages (empty = valid)."""
        errors: list[str] = []
        known_ids = set(self._tasks.keys())

        # Check dependency references
        for task in self._spec.tasks:
            for dep in task.dependencies:
                if dep.task_id not in known_ids:
                    errors.append(
                        f"Task '{task.id}' depends on unknown task '{dep.task_id}'"
                    )

        # Cycle detection via Kahn's algorithm
        in_deg = dict(self._in_degree)
        queue: deque[str] = deque()
        for tid, deg in in_deg.items():
            if deg <= 0:
                queue.append(tid)

        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in self._adj.get(node, []):
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if visited != len(self._tasks):
            errors.append("Cycle detected in task dependency graph")

        return errors

    def get_ready_tasks(self) -> list[str]:
        """Return task IDs that are PENDING and have all deps resolved (SUCCESS or SKIPPED)."""
        ready = []
        resolved = {TaskStatus.SUCCESS, TaskStatus.SKIPPED}
        for tid, status in self._statuses.items():
            if status != TaskStatus.PENDING:
                continue
            task = self._tasks[tid]
            all_deps_done = all(
                self._statuses.get(d.task_id) in resolved
                for d in task.dependencies
            )
            if all_deps_done:
                ready.append(tid)
        return ready

    def mark(self, task_id: str, status: TaskStatus) -> None:
        if task_id not in self._statuses:
            raise ValueError(f"Unknown task: {task_id}")
        self._statuses[task_id] = status

    def get_status(self, task_id: str) -> TaskStatus:
        return self._statuses[task_id]

    def all_statuses(self) -> dict[str, TaskStatus]:
        return dict(self._statuses)

    def is_complete(self) -> bool:
        return all(s in TERMINAL_STATES for s in self._statuses.values())

    def has_failures(self) -> bool:
        return any(s == TaskStatus.FAILURE for s in self._statuses.values())

    def get_task(self, task_id: str) -> TaskSpec:
        return self._tasks[task_id]

    # ── Mutation methods (for Critic patches) ──

    def add_task(self, task_spec: TaskSpec) -> None:
        """Insert a new task into the live DAG."""
        self._tasks[task_spec.id] = task_spec
        self._statuses[task_spec.id] = TaskStatus.PENDING
        self._adj[task_spec.id] = []
        self._in_degree[task_spec.id] = 0

        for dep in task_spec.dependencies:
            if dep.task_id in self._adj:
                self._adj[dep.task_id].append(task_spec.id)
            self._in_degree[task_spec.id] += 1

        # Update spec
        self._spec.tasks.append(task_spec)

    def remove_task(self, task_id: str) -> None:
        """Mark a task as SKIPPED and adjust downstream in-degrees."""
        self._statuses[task_id] = TaskStatus.SKIPPED
        for downstream in self._adj.get(task_id, []):
            self._in_degree[downstream] = max(0, self._in_degree[downstream] - 1)

    def reset_task(self, task_id: str) -> None:
        """Set a task back to PENDING for retry."""
        self._statuses[task_id] = TaskStatus.PENDING
