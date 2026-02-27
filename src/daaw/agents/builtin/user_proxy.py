"""UserProxyAgent — refactored from questionnaire.py. Human-in-the-loop agent."""

from __future__ import annotations

import asyncio
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
        """Walk the user through the 7-question intake."""
        print("\n" + "=" * 60)
        print("  WORKFLOW AUTOMATION — Project Intake")
        print("=" * 60)
        print("Answer the following questions to help us understand")
        print("what you want to automate. Take your time.\n")

        answers: dict[str, str] = {}

        for i, q in enumerate(QUESTIONS, start=1):
            print(f"Question {i} of {len(QUESTIONS)}")
            print(f"  {q['label']}")
            print(f"  Hint: {q['hint']}")
            print()

            while True:
                answer = (
                    await asyncio.to_thread(input, "  Your answer: ")
                ).strip()
                if answer:
                    break
                print("  (Answer can't be empty, please type something)\n")

            answers[q["id"]] = answer
            print()

        # Format summary
        summary_lines = ["User's project intake answers:\n"]
        for i, q in enumerate(QUESTIONS, start=1):
            summary_lines.append(f"Q{i}. {q['label']}")
            summary_lines.append(f"A{i}. {answers[q['id']]}")
            summary_lines.append("")

        summary = "\n".join(summary_lines)
        return AgentResult(output=summary, status="success")

    async def _run_prompt(self, task_input: Any) -> AgentResult:
        """Generic HITL: display context, collect user input."""
        print("\n" + "=" * 60)
        print("  HUMAN INPUT REQUIRED")
        print("=" * 60)
        if task_input:
            print(f"\nContext:\n{task_input}\n")
        print("Please provide your input:")
        user_input = (await asyncio.to_thread(input, "  [You]: ")).strip()
        return AgentResult(output=user_input, status="success")
