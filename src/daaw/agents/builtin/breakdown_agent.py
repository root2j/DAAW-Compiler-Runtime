"""BreakdownAgent — refactored from break_task.py. Task breakdown via LLM."""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.llm.base import LLMMessage
from daaw.schemas.results import AgentResult

TASK_BREAKDOWN_SYSTEM_PROMPT = """\
You are a Senior Workflow Analyst and Systems Thinker. Your job is to take a project brief \
written by a Project Manager and break it down into a precise, ordered list of subtasks that \
a technical automation system will later use to build an n8n workflow.

You do NOT write code. You do NOT know or care about specific n8n nodes yet.
You think purely from a PROCESS and OPERATIONS perspective.

For every major step in the project brief, you must:

1. Identify ALL the micro-actions involved:
   - Authentication / credential checks
   - API calls or data fetches
   - Data transformations
   - Conditional logic
   - External service interactions
   - Error scenarios
   - Final outputs or side effects

2. Group subtasks that can happen TOGETHER (in parallel or as one logical unit).

3. For sequential subtasks, make dependencies explicit.

Output format:
## Workflow Title: [derive from brief]
## Overview: [2-3 sentences]
## Trigger: [what starts it]

## Subtask Breakdown
### Phase N: [Name]
**Type:** Sequential | Grouped
**Depends on:** Nothing | Phase X
**Subtasks:** [detailed list]
**Output of this phase:** [what it produces]

## Conditional Logic & Edge Cases
## Data Flow Summary
## Assumptions Made"""

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"


@register_agent("breakdown")
class BreakdownAgent(BaseAgent):
    """Breaks a project brief into detailed subtasks."""

    async def run(self, task_input: Any) -> AgentResult:
        provider = self.config.get("provider", DEFAULT_PROVIDER)
        model = self.config.get("model", DEFAULT_MODEL)
        final_draft = task_input if isinstance(task_input, str) else str(task_input)

        print("\n" + "=" * 60)
        print("TASK BREAKDOWN AGENT — Analyzing workflow...")
        print("=" * 60 + "\n")

        user_message = (
            "Here is the approved project brief from the Project Manager.\n"
            "Please break this down into detailed subtasks as instructed.\n\n"
            "--- PROJECT BRIEF ---\n\n"
            f"{final_draft}"
        )

        messages = [
            LLMMessage(role="system", content=TASK_BREAKDOWN_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message),
        ]

        resp = await self.llm_client.chat(
            provider, messages, model=model, temperature=0.4, max_tokens=8192
        )

        subtask_document = resp.content
        print(subtask_document)

        print("\n" + "=" * 60)
        print("TASK BREAKDOWN COMPLETE")
        print("=" * 60)

        return AgentResult(output=subtask_document, status="success")
