"""Tests for the circuit breaker."""

from daaw.engine.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_default_threshold(self):
        cb = CircuitBreaker()
        assert cb.threshold == 3

    def test_custom_threshold(self):
        cb = CircuitBreaker(threshold=5)
        assert cb.threshold == 5

    def test_not_tripped_initially(self):
        cb = CircuitBreaker(threshold=3)
        assert not cb.is_tripped("task_1")

    def test_trips_at_threshold(self):
        cb = CircuitBreaker(threshold=3)
        assert not cb.record_failure("task_1")  # 1
        assert not cb.record_failure("task_1")  # 2
        assert cb.record_failure("task_1")      # 3 — trips
        assert cb.is_tripped("task_1")

    def test_success_resets(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("task_1")
        cb.record_failure("task_1")
        cb.record_success("task_1")
        assert not cb.is_tripped("task_1")

        # Should need 3 more failures to trip
        assert not cb.record_failure("task_1")
        assert not cb.record_failure("task_1")
        assert cb.record_failure("task_1")

    def test_reset(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("task_1")
        cb.record_failure("task_1")
        assert cb.is_tripped("task_1")

        cb.reset("task_1")
        assert not cb.is_tripped("task_1")

    def test_independent_tasks(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure("a")
        cb.record_failure("a")
        assert cb.is_tripped("a")
        assert not cb.is_tripped("b")

    def test_reset_nonexistent_is_noop(self):
        cb = CircuitBreaker()
        cb.reset("never_seen")  # should not raise
        assert not cb.is_tripped("never_seen")
