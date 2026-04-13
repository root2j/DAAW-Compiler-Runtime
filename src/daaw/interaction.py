"""Human-in-the-loop (HITL) interaction layer.

Agents that need to prompt the user mid-execution call
``await handler.ask(InteractionRequest(...))`` and receive the answer as a
string. The executor, factory, and agents all share one handler instance
per pipeline invocation.

Implementations
---------------
- ``StdinInteractionHandler`` — CLI; uses ``input()`` on a worker thread.
- ``QueueInteractionHandler`` — UI; bridges an async executor thread with
  a Streamlit (or any non-async) frontend via two ``queue.Queue`` objects.
- ``AutoAnswerInteractionHandler`` — unit tests / demo mode; returns a
  pre-programmed answer for each prompt.
- ``NullInteractionHandler`` — raises if asked; useful in batch runs where
  any HITL request is a bug.

All handlers are async and thread-safe.
"""

from __future__ import annotations

import asyncio
import queue
import sys
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class InteractionRequest(BaseModel):
    """A single question posed to the user by an agent."""

    agent_id: str
    task_id: str | None = None
    prompt: str
    hint: str | None = None
    choices: list[str] | None = None
    multi: bool = False
    context: dict[str, Any] = Field(default_factory=dict)
    # Free-form identifier the frontend can use to correlate
    # multiple prompts from the same agent (e.g. questionnaire step).
    step_id: str | None = None


class InteractionUnavailableError(RuntimeError):
    """Raised when an agent asks for input but no handler is configured."""


@runtime_checkable
class InteractionHandler(Protocol):
    """Pluggable async handler for human-in-the-loop prompts."""

    async def ask(self, request: InteractionRequest) -> str: ...


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


class NullInteractionHandler:
    """Refuses every prompt — use for truly non-interactive pipelines."""

    async def ask(self, request: InteractionRequest) -> str:
        raise InteractionUnavailableError(
            f"Agent {request.agent_id!r} requested user input "
            f"({request.prompt!r}) but no InteractionHandler is configured."
        )


class StdinInteractionHandler:
    """CLI handler — reads a line from stdin on a worker thread.

    Falls back to ``default`` (or raises) if stdin isn't a TTY.
    """

    def __init__(self, default: str | None = None, stream: Any = None):
        self._default = default
        self._stream = stream or sys.stdout

    async def ask(self, request: InteractionRequest) -> str:
        if not sys.stdin.isatty() and self._default is None:
            raise InteractionUnavailableError(
                "stdin is not a TTY; cannot ask {!r}".format(request.prompt)
            )
        self._render(request)
        prompt = "  [You]: "
        try:
            answer = (await asyncio.to_thread(input, prompt)).strip()
        except EOFError:
            if self._default is not None:
                return self._default
            raise
        if not answer and self._default is not None:
            return self._default
        return answer

    def _render(self, request: InteractionRequest) -> None:
        s = self._stream
        print("\n" + "=" * 60, file=s)
        print(f"  HUMAN INPUT REQUESTED  (agent: {request.agent_id})", file=s)
        print("=" * 60, file=s)
        print(request.prompt, file=s)
        if request.hint:
            print(f"  Hint: {request.hint}", file=s)
        if request.choices:
            for i, c in enumerate(request.choices, 1):
                print(f"    {i}. {c}", file=s)
        s.flush() if hasattr(s, "flush") else None


class AutoAnswerInteractionHandler:
    """Returns pre-programmed answers in order. For tests / demos."""

    def __init__(self, answers: list[str] | str, fallback: str | None = None):
        self._answers: list[str] = (
            [answers] if isinstance(answers, str) else list(answers)
        )
        self._idx = 0
        self._fallback = fallback
        self.calls: list[InteractionRequest] = []

    async def ask(self, request: InteractionRequest) -> str:
        self.calls.append(request)
        if self._idx < len(self._answers):
            ans = self._answers[self._idx]
            self._idx += 1
            return ans
        if self._fallback is not None:
            return self._fallback
        raise InteractionUnavailableError(
            f"AutoAnswerInteractionHandler exhausted after {len(self._answers)} "
            f"answers; next prompt was {request.prompt!r}."
        )


class QueueInteractionHandler:
    """Bridges an async executor thread with a non-async frontend.

    The executor (running in a worker thread) puts an ``InteractionRequest``
    on ``questions`` and blocks on ``answers.get()``. The frontend (e.g.
    Streamlit) polls ``questions`` each rerun, renders a form, and puts the
    user's reply on ``answers`` to unblock the executor.

    Parameters
    ----------
    questions, answers: ``queue.Queue``
        Standard-library queues — thread-safe, cross-thread/async friendly.
    timeout:
        Seconds to wait for an answer before giving up. ``None`` = forever.
    """

    def __init__(
        self,
        questions: "queue.Queue[InteractionRequest]",
        answers: "queue.Queue[str]",
        timeout: float | None = 600.0,
    ):
        self._questions = questions
        self._answers = answers
        self._timeout = timeout

    async def ask(self, request: InteractionRequest) -> str:
        # Non-blocking enqueue (Queue has unbounded size by default).
        self._questions.put(request)
        try:
            # Block in a worker thread so the event loop keeps running
            # inside the executor's thread.
            answer = await asyncio.to_thread(self._answers.get, True, self._timeout)
        except queue.Empty as exc:  # noqa: PERF203
            raise InteractionUnavailableError(
                f"Timed out waiting for answer to {request.prompt!r}"
            ) from exc
        return str(answer)


__all__ = [
    "AutoAnswerInteractionHandler",
    "InteractionHandler",
    "InteractionRequest",
    "InteractionUnavailableError",
    "NullInteractionHandler",
    "QueueInteractionHandler",
    "StdinInteractionHandler",
]
