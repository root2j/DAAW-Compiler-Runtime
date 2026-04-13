"""UserProxyAgent — human-in-the-loop agent.

Works in two modes:

- ``questionnaire`` (default): walks the user through a 7-question project
  intake.
- ``prompt``: displays the task context and collects one free-form answer.

Both modes delegate actual IO to the injected :class:`InteractionHandler`
so the same agent works in CLI, Streamlit UI, and automated tests.
"""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.schemas.results import AgentResult

QUESTIONS = [
    {
        "id": "task",
        "label": "What's the task or activity you want to automate?",
        "hint": "e.g. 'when someone fills a form, send them a welcome email and add them to my CRM'",
    },
    {
        "id": "trigger",
        "label": "What triggers this workflow?",
        "hint": "e.g. a form submission, a scheduled time, an incoming email, a webhook, manually",
    },
    {
        "id": "apps",
        "label": "What apps or services are involved?",
        "hint": "e.g. Gmail, Slack, Google Sheets, Notion, Typeform — list all you can think of",
    },
    {
        "id": "outcome",
        "label": "What should happen at the end of the workflow? What does 'done' look like?",
        "hint": "e.g. a notification is sent, a row is added, a file is created",
    },
    {
        "id": "frequency",
        "label": "How often do you expect this workflow to run?",
        "hint": "e.g. once a day, every time an event happens, hundreds of times a day",
    },
    {
        "id": "conditions",
        "label": "Are there any conditions or special cases the workflow should handle?",
        "hint": "e.g. 'only send the email if the country is India', 'skip if the row already exists' — type 'none' if not applicable",
    },
    {
        "id": "manual_process",
        "label": "Have you tried automating this before, or are you currently doing this manually? If manually, walk me through the steps you take.",
        "hint": "Be as detailed as possible — this helps us understand edge cases you might not think to mention",
    },
]


@register_agent("user_proxy")
class UserProxyAgent(BaseAgent):
    """Human-in-the-loop agent: questionnaire mode or generic prompt mode."""

    async def run(self, task_input: Any) -> AgentResult:
        mode = self.config.get("mode", "questionnaire")
        if mode == "questionnaire":
            return await self._run_questionnaire()
        return await self._run_prompt(task_input)

    async def _run_questionnaire(self) -> AgentResult:
        """Walk the user through the 7-question intake via the interaction handler."""
        answers: dict[str, str] = {}
        for i, q in enumerate(QUESTIONS, start=1):
            answer = await self.ask_user(
                prompt=q["label"],
                hint=q["hint"],
                step_id=q["id"],
                context={"step": i, "total": len(QUESTIONS)},
            )
            answers[q["id"]] = answer.strip() or "(no answer)"

        summary_lines = ["User's project intake answers:\n"]
        for i, q in enumerate(QUESTIONS, start=1):
            summary_lines.append(f"Q{i}. {q['label']}")
            summary_lines.append(f"A{i}. {answers[q['id']]}")
            summary_lines.append("")

        return AgentResult(
            output="\n".join(summary_lines),
            status="success",
            metadata={"answers": answers},
        )

    async def _run_prompt(self, task_input: Any) -> AgentResult:
        """Generic HITL: forward the task context to the user and capture one answer."""
        context = {"task_input": task_input} if task_input else {}
        prompt = (
            str(task_input).strip() if task_input else "The workflow needs your input."
        )
        user_input = await self.ask_user(
            prompt=prompt,
            step_id="prompt",
            context=context,
        )
        return AgentResult(output=user_input.strip(), status="success")
