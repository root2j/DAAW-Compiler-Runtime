"""Compiler — transforms a user goal string into a validated WorkflowSpec."""

from __future__ import annotations

import json
import uuid
from typing import Any

from daaw.agents.registry import AGENT_REGISTRY
from daaw.compiler.prompts import PLANNER_REFINEMENT_PROMPT, PLANNER_SYSTEM_PROMPT
from daaw.config import AppConfig
from daaw.llm.base import LLMMessage
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.workflow import WorkflowSpec
from daaw.tools.registry import tool_registry


class Compiler:
    """Goal → WorkflowSpec via LLM with validation and retry."""

    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        config: AppConfig,
        provider: str = "groq",
        model: str | None = None,
    ):
        self._llm = llm_client
        self._config = config
        self._provider = provider
        self._model = model

    def _build_system_prompt(self) -> str:
        available_roles = ", ".join(AGENT_REGISTRY.keys()) or "generic_llm"
        available_tools = ", ".join(
            t.name for t in tool_registry._tools.values()
        ) or "(none)"
        return PLANNER_SYSTEM_PROMPT.format(
            available_roles=available_roles,
            available_tools=available_tools,
        )

    async def compile(self, user_goal: str) -> WorkflowSpec:
        """Compile a user goal into a WorkflowSpec with retry on parse failure."""
        system_prompt = self._build_system_prompt()
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=f"Create a workflow plan for this goal:\n\n{user_goal}",
            ),
        ]

        last_error = ""
        for attempt in range(self._config.max_planner_retries):
            if last_error:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            f"The previous response had an error: {last_error}\n"
                            "Please fix and respond with valid JSON only."
                        ),
                    )
                )

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(resp.content)
                if "id" not in data:
                    data["id"] = str(uuid.uuid4())
                spec = WorkflowSpec.model_validate(data)
                return spec
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)
                messages.append(LLMMessage(role="assistant", content=resp.content))

        raise RuntimeError(
            f"Compiler failed after {self._config.max_planner_retries} attempts: {last_error}"
        )

    async def refine(
        self, current_spec: WorkflowSpec, user_feedback: str
    ) -> WorkflowSpec:
        """Refine an existing WorkflowSpec based on user feedback."""
        system_prompt = self._build_system_prompt()
        current_json = current_spec.model_dump_json(indent=2)

        refinement_prompt = PLANNER_REFINEMENT_PROMPT.format(
            current_plan_json=current_json,
            user_feedback=user_feedback,
        )

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=refinement_prompt),
        ]

        last_error = ""
        for attempt in range(self._config.max_planner_retries):
            if last_error:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            f"Parse error: {last_error}\n"
                            "Please fix and respond with valid JSON only."
                        ),
                    )
                )

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(resp.content)
                # Preserve workflow ID
                data["id"] = current_spec.id
                spec = WorkflowSpec.model_validate(data)
                return spec
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)
                messages.append(LLMMessage(role="assistant", content=resp.content))

        raise RuntimeError(
            f"Compiler refinement failed after {self._config.max_planner_retries} attempts: {last_error}"
        )
