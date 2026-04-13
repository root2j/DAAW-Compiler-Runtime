"""Compiler — transforms a user goal string into a validated WorkflowSpec."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator, Callable

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
        available_tools = ", ".join(
            t.name for t in tool_registry._tools.values()
        ) or "(none)"
        return PLANNER_SYSTEM_PROMPT.format(
            available_tools=available_tools,
        )

    async def compile(self, user_goal: str) -> WorkflowSpec:
        """Compile a user goal into a WorkflowSpec with retry on parse failure."""
        system_prompt = self._build_system_prompt()

        last_error = ""
        for attempt in range(self._config.max_planner_retries):
            # Build messages fresh each attempt to keep context small.
            # Retries include only the error, not the full bad response
            # (which can push small models OOM).
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        f"Create a workflow plan for this goal:\n\n{user_goal}"
                        + (f"\n\nPrevious attempt had error: {last_error[:200]}\nFix and respond with valid JSON." if last_error else "")
                    ),
                ),
            ]

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(_repair_json(resp.content))
                # Top-level defaults for fields small models often skip
                data.setdefault("id", str(uuid.uuid4()))
                data.setdefault("name", user_goal[:50])
                data.setdefault("description", user_goal)
                data.setdefault("tasks", [])
                data.setdefault("metadata", {})
                _fixup_json_structure(data)
                _fixup_agent_roles(data)
                spec = WorkflowSpec.model_validate(data)
                # Empty spec is always invalid — retry.
                if len(spec.tasks) == 0:
                    last_error = (
                        "No tasks generated. The workflow must have at "
                        "least one task."
                    )
                    continue
                # Single-task spec is valid but less interesting — if we
                # still have retries left, nudge the model to decompose.
                # If the final attempt still returns one task, accept it
                # rather than hard-failing: the DAG executor handles
                # single-task workflows fine, and refusing to compile
                # wastes the API call the user already paid for.
                if len(spec.tasks) == 1 and attempt < self._config.max_planner_retries - 1:
                    last_error = (
                        f"Only 1 task generated. Break the goal into "
                        f"2-3 tasks with dependencies so the steps can "
                        f"be reviewed and retried independently."
                    )
                    continue
                return spec
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)

        raise RuntimeError(
            f"Compiler failed after {self._config.max_planner_retries} attempts: {last_error}"
        )

    async def compile_stream(
        self, user_goal: str, *,
        on_token: Callable[[str, str], None] | None = None,
    ) -> WorkflowSpec:
        """Compile with real-time token streaming.

        Behaves exactly like :meth:`compile` (same retry loop, same
        validation, same repair pipeline) but emits each token through
        ``on_token(delta, full_so_far)`` as it arrives so the UI can
        render a live view of the plan being written.

        Returns the final validated ``WorkflowSpec``.
        """
        system_prompt = self._build_system_prompt()
        last_error = ""
        for attempt in range(self._config.max_planner_retries):
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        f"Create a workflow plan for this goal:\n\n{user_goal}"
                        + (f"\n\nPrevious attempt had error: {last_error[:200]}\nFix and respond with valid JSON." if last_error else "")
                    ),
                ),
            ]

            accumulated = ""
            async for chunk in self._llm.chat_stream(
                self._provider,
                messages,
                model=self._model,
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            ):
                if chunk.delta and on_token is not None:
                    try:
                        on_token(chunk.delta, chunk.full_content)
                    except Exception:
                        # UI callback must never break compilation.
                        pass
                if chunk.done:
                    accumulated = chunk.full_content

            try:
                data = json.loads(_repair_json(accumulated))
                data.setdefault("id", str(uuid.uuid4()))
                data.setdefault("name", user_goal[:50])
                data.setdefault("description", user_goal)
                data.setdefault("tasks", [])
                data.setdefault("metadata", {})
                _fixup_json_structure(data)
                _fixup_agent_roles(data)
                spec = WorkflowSpec.model_validate(data)
                if len(spec.tasks) == 0:
                    last_error = (
                        "No tasks generated. The workflow must have at "
                        "least one task."
                    )
                    continue
                if len(spec.tasks) == 1 and attempt < self._config.max_planner_retries - 1:
                    last_error = (
                        "Only 1 task generated. Break the goal into "
                        "2-3 tasks with dependencies so the steps can "
                        "be reviewed and retried independently."
                    )
                    continue
                return spec
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)

        raise RuntimeError(
            f"Compiler failed after {self._config.max_planner_retries} "
            f"attempts: {last_error}"
        )

    async def refine(
        self, current_spec: WorkflowSpec, user_feedback: str
    ) -> WorkflowSpec:
        """Refine an existing WorkflowSpec based on user feedback."""
        system_prompt = self._build_system_prompt()
        # Use compact JSON to save tokens
        current_json = current_spec.model_dump_json()

        refinement_prompt = PLANNER_REFINEMENT_PROMPT.format(
            current_plan_json=current_json,
            user_feedback=user_feedback,
        )

        last_error = ""
        for attempt in range(self._config.max_planner_retries):
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(
                    role="user",
                    content=(
                        refinement_prompt
                        + (f"\n\nPrevious error: {last_error[:200]}\nFix and respond with valid JSON." if last_error else "")
                    ),
                ),
            ]

            resp = await self._llm.chat(
                self._provider,
                messages,
                model=self._model,
                temperature=0.4,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            try:
                data = json.loads(_repair_json(resp.content))
                data["id"] = current_spec.id
                _fixup_json_structure(data)
                _fixup_agent_roles(data)
                spec = WorkflowSpec.model_validate(data)
                return spec
            except (json.JSONDecodeError, Exception) as e:
                last_error = str(e)

        raise RuntimeError(
            f"Compiler refinement failed after {self._config.max_planner_retries} attempts: {last_error}"
        )


# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# JSON text repair
# ─────────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """Best-effort repair of malformed JSON from small LLMs.

    Handles: markdown fences, trailing commas, unclosed brackets,
    leading/trailing garbage text.
    """
    t = text.strip()

    # Strip markdown fences: ```json ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # Find the outermost JSON object
    start = t.find("{")
    if start == -1:
        return t
    t = t[start:]

    # Remove trailing commas before ] or }
    t = re.sub(r",\s*([}\]])", r"\1", t)

    # Try parsing as-is
    try:
        json.loads(t)
        return t
    except json.JSONDecodeError:
        pass

    # Truncated JSON — close unclosed brackets
    open_b = t.count("{") - t.count("}")
    open_a = t.count("[") - t.count("]")

    # Remove any trailing partial key/value (incomplete string)
    # Find the last complete value by looking for the last , or : before EOF
    if open_b > 0 or open_a > 0:
        # Try to find the last valid comma or closing bracket and truncate there
        for i in range(len(t) - 1, max(len(t) - 200, 0), -1):
            if t[i] in (",", "}", "]"):
                candidate = t[: i + 1]
                # Remove trailing commas
                candidate = re.sub(r",\s*$", "", candidate)
                ob = candidate.count("{") - candidate.count("}")
                oa = candidate.count("[") - candidate.count("]")
                candidate += "]" * oa + "}" * ob
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        # Simple fallback: just close everything
        t = re.sub(r",\s*$", "", t)
        t += "]" * max(0, open_a) + "}" * max(0, open_b)

    return t


# ─────────────────────────────────────────────────────────────
# JSON structure auto-repair
# ─────────────────────────────────────────────────────────────

# Fields that belong at the task level, NOT inside agent.
_TASK_LEVEL_FIELDS = {
    "dependencies", "input_filter", "success_criteria",
    "timeout_seconds", "max_retries", "id", "name", "description",
}

# Fields that belong inside agent.
_AGENT_FIELDS = {"role", "tools_allowed", "system_prompt_override", "model_config_override"}


def _fixup_json_structure(data: dict) -> None:
    """Normalize task structure from any format the LLM produces.

    Handles:
    1. FLAT format (from lean prompt): role/tools_allowed at task level → wrap into agent{}
    2. NESTED format (old prompt): agent{} exists → move misplaced task fields out
    3. MIXED format: some fields in agent, some at task level → normalize
    4. TRUNCATED tasks missing id/name/description → dropped (small models
       sometimes hallucinate an empty stub at the end of the array).
    """
    # First, drop tasks missing the three truly-required fields. Auto-
    # filling with placeholders here would give the executor a fake task
    # to run; dropping is cleaner and matches the "be lenient, not wrong"
    # policy used elsewhere in the compiler.
    tasks_in = data.get("tasks", []) or []
    tasks_out: list[dict] = []
    for t in tasks_in:
        if not isinstance(t, dict):
            continue
        has_id = isinstance(t.get("id"), str) and t["id"].strip()
        has_name = isinstance(t.get("name"), str) and t["name"].strip()
        has_desc = (
            isinstance(t.get("description"), str) and t["description"].strip()
        )
        if has_id and has_name and has_desc:
            tasks_out.append(t)
    data["tasks"] = tasks_out

    for task in data.get("tasks", []):
        agent = task.get("agent")

        if agent and isinstance(agent, dict):
            # Nested or mixed format — move task fields out of agent
            for field in list(agent.keys()):
                if field in _TASK_LEVEL_FIELDS and field not in task:
                    task[field] = agent.pop(field)
                elif field not in _AGENT_FIELDS:
                    if field not in task:
                        task[field] = agent.pop(field)
                    else:
                        del agent[field]
            agent.setdefault("tools_allowed", [])
        else:
            # Flat format — reconstruct agent from task-level fields
            task["agent"] = {
                "role": task.pop("role", "generic_llm"),
                "tools_allowed": task.pop("tools_allowed", []),
            }
            # Remove system_prompt_override if at task level (shouldn't be)
            spo = task.pop("system_prompt_override", None)
            if spo:
                task["agent"]["system_prompt_override"] = spo

        # Ensure required defaults
        task.setdefault("dependencies", [])
        task.setdefault("input_filter", [])
        task.setdefault("success_criteria", "Task completes successfully")
        task.setdefault("timeout_seconds", 300)
        task.setdefault("max_retries", 2)

        # Fix dependencies: small models often emit ["task_001"] instead of
        # [{"task_id": "task_001"}].  Normalize both formats.
        deps = task.get("dependencies", [])
        fixed_deps = []
        for d in deps:
            if isinstance(d, str):
                fixed_deps.append({"task_id": d})
            elif isinstance(d, dict):
                fixed_deps.append(d)
        task["dependencies"] = fixed_deps


# Roles that exist in the registry but should NOT be used as task agents.
# The compiler prompt tells the LLM to use "generic_llm" for everything,
# but smaller models sometimes ignore this.
_INTERNAL_ONLY_ROLES = {"critic", "planner", "pm", "breakdown"}


def _fixup_agent_roles(data: dict) -> None:
    """Auto-correct agent roles the LLM got wrong.

    Fixes:
      - Tool name in role: {"role":"web_search"} → generic_llm + tools_allowed
      - Internal role used: {"role":"critic"} → generic_llm (critic is a placeholder)
      - Unknown role: → generic_llm
      - Strips system_prompt_override if it just restates the task (saves tokens)
    """
    tool_names = set()
    try:
        tool_names = {t.name for t in tool_registry._tools.values()}
    except Exception:
        pass

    for task in data.get("tasks", []):
        agent = task.get("agent")
        if not agent or not isinstance(agent, dict):
            continue
        role = agent.get("role", "")
        tools_allowed = agent.get("tools_allowed") or []

        # Fix: internal-only roles used as task agents
        if role in _INTERNAL_ONLY_ROLES:
            agent["role"] = "generic_llm"

        # Fix: tool name used as role
        elif role in tool_names:
            if role not in tools_allowed:
                tools_allowed.append(role)
            agent["tools_allowed"] = tools_allowed
            agent["role"] = "generic_llm"

        # Fix: completely unknown role
        elif role not in AGENT_REGISTRY:
            agent["role"] = "generic_llm"

        # Token optimization: drop system_prompt_override if it's just noise.
        # Short overrides (<20 chars) or ones that just say "you are an agent"
        # add tokens without value.
        spo = agent.get("system_prompt_override")
        if spo and len(spo) < 20:
            agent["system_prompt_override"] = None
