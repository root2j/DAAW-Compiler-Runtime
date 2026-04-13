"""PMAgent — refactored from refine.py. Project Manager agent with clarification + drafting."""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.llm.base import LLMMessage
from daaw.schemas.results import AgentResult

PM_SYSTEM_PROMPT = """\
You are an experienced Project Manager and Stakeholder who deeply understands software projects.
Your job is to take a user's project description, ask smart clarifying questions to fully \
understand their intent, and then produce a clear, structured project brief/draft.

Your behavior:
1. When given a project description, first ask 2-4 focused clarifying questions to fill any gaps.
   Keep questions concise and directly relevant — don't ask things already answered.
2. Once you have enough context (after the user answers your questions), produce a detailed draft.
   The draft should include:
   - Project Overview
   - Goals & Success Criteria
   - Key Features / Scope
   - Out of Scope (assumptions)
   - Open Questions / Risks
3. When refining, take the user's feedback seriously. Only rewrite the sections that need changing.
   Keep what was already approved implicitly. Be concise about what changed.

Always be professional but conversational. You're a smart collaborator, not a bureaucrat."""

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.1-8b-instant"


@register_agent("pm")
class PMAgent(BaseAgent):
    """PM agent: clarification → draft → refinement loop."""

    async def run(self, task_input: Any) -> AgentResult:
        provider = self.config.get("provider", DEFAULT_PROVIDER)
        model = self.config.get("model", DEFAULT_MODEL)
        user_description = task_input if isinstance(task_input, str) else str(task_input)

        # ── Clarification phase
        print("\n" + "=" * 60)
        print("PM AGENT — Clarification Phase")
        print("=" * 60)

        messages = [
            LLMMessage(role="system", content=PM_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Here is the project description from the user:\n\n"
                    f"{user_description}\n\n"
                    "Please ask your clarifying questions now."
                ),
            ),
        ]

        resp = await self.llm_client.chat(
            provider, messages, model=model, temperature=0.7, max_tokens=2048
        )
        agent_questions = resp.content
        messages.append(LLMMessage(role="assistant", content=agent_questions))

        user_answers = (
            await self.ask_user(
                prompt=agent_questions,
                hint="Answer the PM's clarifying questions.",
                step_id="pm_clarifications",
            )
        ).strip()
        messages.append(LLMMessage(role="user", content=user_answers))

        # ── Draft generation
        messages.append(
            LLMMessage(
                role="user",
                content=(
                    "Thank you for the answers. Now please produce the detailed "
                    "project draft based on everything discussed."
                ),
            )
        )

        resp = await self.llm_client.chat(
            provider, messages, model=model, temperature=0.7, max_tokens=2048
        )
        current_draft = resp.content

        # ── Refinement loop
        max_rounds = int(self.config.get("max_refinement_rounds", 5))
        for _ in range(max_rounds):
            user_input = (
                await self.ask_user(
                    prompt=current_draft,
                    hint=(
                        "Type 'yes' to approve, or describe what to change. "
                        "Empty reply also approves."
                    ),
                    choices=["yes", "revise"],
                    step_id="pm_approval",
                )
            ).strip()

            if not user_input or user_input.lower() in ("yes", "y", "approve"):
                return AgentResult(output=current_draft, status="success")

            refine_messages = [
                LLMMessage(role="system", content=PM_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=f"Original project description:\n\n{user_description}",
                ),
                LLMMessage(role="assistant", content=current_draft),
                LLMMessage(
                    role="user",
                    content=f"Please refine the draft based on this feedback:\n\n{user_input}",
                ),
            ]
            resp = await self.llm_client.chat(
                provider, refine_messages, model=model, temperature=0.7, max_tokens=2048
            )
            current_draft = resp.content

        # Exhausted rounds — return the latest draft so the pipeline can move on.
        return AgentResult(
            output=current_draft,
            status="success",
            metadata={"note": f"max_refinement_rounds ({max_rounds}) reached"},
        )
