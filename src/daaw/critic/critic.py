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
    ) -> tuple[bool, WorkflowPatch | None, str]:
        """Evaluate a task result. Returns (passed, patch_or_none, reasoning)."""
        # Skip if no success criteria
        if not task.success_criteria:
            return True, None, "No success criteria defined — auto-pass."

        # Auto-generate retry patch for failures
        if result.agent_result.status == "failure":
            reasoning = f"Task '{task.id}' failed — automatic retry."
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
                reasoning=reasoning,
            )
            return False, patch, reasoning

        # Ask LLM to evaluate — keep output short for local models
        MAX_OUTPUT_CHARS = 800
        raw_output = str(result.agent_result.output)
        output_str = raw_output[:MAX_OUTPUT_CHARS]
        eval_prompt = CRITIC_EVALUATION_PROMPT.format(
            task_name=task.name,
            task_description=task.description,
            success_criteria=task.success_criteria,
            status=result.agent_result.status,
            elapsed=result.elapsed_seconds,
            task_output=output_str,
        )

        last_error = ""
        for attempt in range(self._config.max_critic_retries):
            # Fresh messages each attempt — don't accumulate context
            messages = [
                LLMMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
                LLMMessage(role="user", content=eval_prompt),
            ]

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.2,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(resp.content)
                verdict = str(data.get("verdict", "pass")).lower().strip()
                passed = verdict == "pass"
                reasoning = data.get("reasoning", "(no reasoning provided)")

                patch = None
                if not passed and data.get("patch"):
                    try:
                        patch = WorkflowPatch.model_validate(data["patch"])
                    except Exception:
                        pass  # bad patch format — skip it

                return passed, patch, reasoning
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)

        # Critic produced no parseable JSON verdict. Don't silently pass
        # (hides real failures) AND don't blindly fail (punishes good task
        # output when the critic LLM is flaky — common with local models).
        # Decide from the evidence we have:
        #
        # 1. Try to salvage a verdict from the prose.
        # 2. If prose has no signal, fall back to agent evidence:
        #      agent success AND non-empty output      → heuristic PASS
        #      agent failure OR empty/placeholder      → FAIL
        last_content = locals().get("resp").content if "resp" in locals() else ""
        salvaged = _salvage_verdict(last_content)
        if salvaged is not None:
            reason = (
                f"Critic JSON unparseable; verdict '{salvaged.upper()}' "
                f"salvaged from prose. Parse error: {last_error[:120]}"
            )
            print(f"  [CRITIC] Salvaged '{salvaged}' for {task.id} from prose")
            return salvaged == "pass", None, reason

        # No parseable verdict, no prose signal — use the task output itself.
        agent = result.agent_result
        output_str = str(agent.output or "").strip()
        has_real_output = (
            agent.status == "success"
            and len(output_str) >= 20
            and "[upstream task produced no usable" not in output_str
        )
        if has_real_output:
            reason = (
                f"Critic could not produce a verdict (parse error: "
                f"{last_error[:120]}). Falling back to agent evidence: "
                f"task reported success with {len(output_str)}-char output."
            )
            print(f"  [CRITIC] Fallback PASS for {task.id} (agent succeeded, "
                  f"output present)")
            return True, None, reason

        reason = (
            f"Critic could not produce a verdict and agent output is "
            f"missing / short / degenerate. Parse error: {last_error[:120]}"
        )
        print(f"  [CRITIC] Fallback FAIL for {task.id} (no evidence of success)")
        return False, None, reason


def _salvage_verdict(content: str) -> str | None:
    """Best-effort verdict extraction when the critic's JSON is malformed.

    Returns 'pass' / 'fail' if the text strongly signals one, else None.
    Deliberately conservative — only act on unambiguous language so a
    rambling response doesn't accidentally pass a bad task.
    """
    if not content:
        return None
    text = content.lower()
    # Explicit JSON-ish signals first.
    import re
    m = re.search(r'"verdict"\s*:\s*"(pass|fail)"', text)
    if m:
        return m.group(1)
    # Then clear prose signals near the start of the response.
    head = text[:400]
    has_pass = any(
        k in head for k in
        ("verdict: pass", "verdict pass", "result: pass",
         "passed the criteria", "meets the success criteria",
         "the task passes", "overall: pass")
    )
    has_fail = any(
        k in head for k in
        ("verdict: fail", "verdict fail", "result: fail",
         "fails the criteria", "does not meet", "task fails",
         "overall: fail")
    )
    if has_pass and not has_fail:
        return "pass"
    if has_fail and not has_pass:
        return "fail"
    return None
