"""Critic — evaluates task outputs against success criteria via LLM."""

from __future__ import annotations

import json

from daaw.config import AppConfig
from daaw.critic.prompts import CRITIC_EVALUATION_PROMPT, CRITIC_SYSTEM_PROMPT
from daaw.llm.base import LLMMessage
from daaw.llm.unified import UnifiedLLMClient
from daaw.schemas.enums import PatchAction
from daaw.schemas.events import PatchOperation, WorkflowPatch
from daaw.schemas.results import TaskResult
from daaw.schemas.workflow import TaskSpec


class Critic:
    """Evaluates task results and optionally produces a WorkflowPatch."""

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

    async def evaluate(
        self, task: TaskSpec, result: TaskResult
    ) -> tuple[bool, WorkflowPatch | None]:
        """Evaluate a task result. Returns (passed, patch_or_none)."""
        # Skip if no success criteria
        if not task.success_criteria:
            return True, None

        # Auto-generate retry patch for failures
        if result.agent_result.status == "failure":
            patch = WorkflowPatch(
                operations=[
                    PatchOperation(
                        action=PatchAction.RETRY,
                        target_task_id=task.id,
                        feedback=(
                            f"Task failed with error: {result.agent_result.error_message}. "
                            "Please retry."
                        ),
                    )
                ],
                reasoning=f"Task '{task.id}' failed — automatic retry.",
            )
            return False, patch

        # Ask LLM to evaluate
        output_str = str(result.agent_result.output)[:4000]
        eval_prompt = CRITIC_EVALUATION_PROMPT.format(
            task_id=task.id,
            task_name=task.name,
            task_description=task.description,
            success_criteria=task.success_criteria,
            status=result.agent_result.status,
            attempt=result.attempt,
            elapsed=result.elapsed_seconds,
            task_output=output_str,
        )

        messages = [
            LLMMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
            LLMMessage(role="user", content=eval_prompt),
        ]

        last_error = ""
        for attempt in range(self._config.max_critic_retries):
            if last_error:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=f"Invalid JSON: {last_error}. Please fix.",
                    )
                )

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.2,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(resp.content)
                verdict = data.get("verdict", "pass")
                passed = verdict == "pass"

                patch = None
                if not passed and data.get("patch"):
                    patch = WorkflowPatch.model_validate(data["patch"])

                return passed, patch
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)
                messages.append(LLMMessage(role="assistant", content=resp.content))

        # Fail-open: if critic can't produce valid output, pass the task
        print(f"  [CRITIC] Could not parse verdict for {task.id} — fail-open (pass)")
        return True, None
