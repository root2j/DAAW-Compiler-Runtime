"""GenericLLMAgent — catch-all agent for planner-created tasks."""

from __future__ import annotations

from typing import Any

from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.llm.base import LLMMessage
from daaw.schemas.results import AgentResult

DEFAULT_PROVIDER = "groq"


@register_agent("generic_llm")
class GenericLLMAgent(BaseAgent):
    """Generic LLM-powered agent — uses system prompt and task input as user message."""

    async def run(self, task_input: Any) -> AgentResult:
        provider = self.config.get("provider", DEFAULT_PROVIDER)
        model = self.config.get("model")
        system_prompt = self.config.get("system_prompt_override", "")

        messages: list[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        content = task_input if isinstance(task_input, str) else str(task_input)
        messages.append(LLMMessage(role="user", content=content))

        resp = await self.llm_client.chat(
            provider, messages, model=model, temperature=0.7, max_tokens=4096
        )

        return AgentResult(output=resp.content, status="success")
