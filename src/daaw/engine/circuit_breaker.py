"""Circuit breaker — tracks consecutive failures per task."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    threshold: int = 3
    _counters: dict[str, int] = field(default_factory=dict)

    def record_failure(self, task_id: str) -> bool:
        """Record a failure. Returns True if the breaker has tripped."""
        self._counters[task_id] = self._counters.get(task_id, 0) + 1
        return self._counters[task_id] >= self.threshold

    def record_success(self, task_id: str) -> None:
        """Reset counter on success."""
        self._counters.pop(task_id, None)

    def is_tripped(self, task_id: str) -> bool:
        return self._counters.get(task_id, 0) >= self.threshold

    def reset(self, task_id: str) -> None:
        self._counters.pop(task_id, None)
