# DAAW Architecture

## High-Level Concept

DAAW follows a **Compiler-Runtime paradigm** — a shift from "chatty agents" to deterministic, structured execution.

| Component | Role |
|---|---|
| **Compiler** (Planner) | Converts a fuzzy user goal into a strict execution graph (WorkflowSpec) |
| **Runtime** (Engine) | A deterministic machine that executes the graph — it follows orders, not intuition |
| **Workers** (Agents) | Ephemeral processes instantiated only when needed to perform specific tasks |
| **Debugger** (Critic) | A loop that checks outputs against criteria and can patch the workflow at runtime |

---

## System Flow

```
[User Goal]
     │
     ▼
┌──────────┐    WorkflowSpec (JSON)    ┌─────────────┐
│ Compiler │ ────────────────────────► │ Plan Review │
│ (Planner)│ ◄──── feedback ────────── │   (Human)   │
└──────────┘                           └──────┬──────┘
                                              │ approved
                                              ▼
                                       ┌─────────────┐
                                       │  DAG Engine  │
                                       │  (Executor)  │
                                       └──────┬──────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         ┌─────────┐    ┌─────────┐    ┌─────────┐
                         │ Agent A │    │ Agent B │    │ Agent C │
                         └────┬────┘    └────┬────┘    └────┬────┘
                              │              │              │
                              ▼              ▼              ▼
                         ┌──────────────────────────────────────┐
                         │          Artifact Store              │
                         │    (Namespaced Key-Value Bus)        │
                         └──────────────────┬───────────────────┘
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │   Critic     │
                                     │  (Debugger)  │
                                     └──────┬──────┘
                                            │
                                    pass? ──┤── fail?
                                    │              │
                                    ▼              ▼
                               Continue      WorkflowPatch
                                            (retry / insert / remove)
```

---

## The Five-Stage Pipeline

### Stage 1 — Compilation

The **Compiler** takes a natural language goal and uses an LLM with a specialized system prompt to generate a `WorkflowSpec` — a JSON DAG of tasks with dependencies, agent roles, and success criteria.

- Includes retry logic (max 3 attempts) if JSON parsing fails
- Supports plan refinement via user feedback loop

### Stage 2 — Review

The **Plan Reviewer** presents the generated plan to the user in a human-readable format. The user can:

- **Accept** — proceed to execution
- **Provide feedback** — refine the plan (loops back to the Compiler)
- **Abort** — cancel the workflow

### Stage 3 — Execution

The **DAG Executor** loads the approved WorkflowSpec into a graph and runs it:

1. Topological sort via Kahn's algorithm identifies ready tasks
2. Ready tasks execute in parallel using `asyncio`
3. DAG state updates atomically as tasks complete
4. Dependencies are respected automatically

### Stage 4 — Evaluation

The **Critic** runs after each task completion:

- Evaluates output against `success_criteria` from the TaskSpec
- Can auto-retry failed tasks with targeted feedback
- Can issue `WorkflowPatch` operations: `RETRY`, `INSERT`, `REMOVE`, `UPDATE_INPUT`

### Stage 5 — Summary

Final report with success/failure counts, elapsed time, and per-task results.

---

## Abstraction Layers

| Layer | Component | Description |
|---|---|---|
| **L0: Foundation** | LLM Client | Unified wrapper for Groq / Gemini / OpenAI / Anthropic. Handles retries and API errors. |
| | Tool Registry | System to register Python functions as tools callable by LLMs. |
| **L1: Schema** | Pydantic Models | Strict definitions of `WorkflowSpec`, `TaskSpec`, `AgentSpec`, `DependencySpec`. |
| **L2: State** | Artifact Store | Async-safe JSON key-value store with namespacing (`{task_id}.output`). |
| **L3: Logic** | Agent Factory | Takes an `AgentSpec`, returns a runnable `Agent` object via registry lookup. |
| | DAG Executor | The async loop that manages dependencies and runs tasks. |
| **L4: Prompts** | Planner System | Prompt engineering that forces the LLM to design good execution graphs. |
| | Critic System | Prompt engineering that makes the LLM a strict evaluator. |

---

## Component Breakdown

### The Contract — Data Structures

```
WorkflowSpec
├── name: str
├── description: str
└── tasks: list[TaskSpec]
        ├── id: str
        ├── name: str
        ├── description: str
        ├── agent: AgentSpec
        │     ├── role: str
        │     ├── tools_allowed: list[str]
        │     └── model_config: dict
        ├── dependencies: list[DependencySpec]
        │     └── task_id: str
        ├── success_criteria: str
        └── input_filter: list[str]
```

### Compiler (Planner Agent)

- **Input:** User goal + list of available tools
- **Output:** Strictly valid JSON `WorkflowSpec`
- **Behavior:** Designs the graph only — never executes code. Optimizes for parallelism by identifying which tasks can run simultaneously.
- **Location:** `src/daaw/compiler/compiler.py`

### Runtime (DAG Executor)

- **Mechanism:** DAG-based execution using topological sort
- **Flow:** Load spec → Build graph → Find tasks with 0 unmet dependencies → Run in parallel → Update graph → Repeat
- **Safety:** If an agent tries to access a tool not in its `AgentSpec`, the engine raises a `SecurityError`
- **Location:** `src/daaw/engine/executor.py`, `src/daaw/engine/dag.py`

### Agent Factory

- Agents are **not** long-lived servers — they are Python objects created for a task and destroyed after
- The factory looks up agent classes in `AGENT_REGISTRY` by role, injects dependencies (`UnifiedLLMClient`, `ArtifactStore`, config), and returns a runnable instance
- **Location:** `src/daaw/agents/factory.py`, `src/daaw/agents/registry.py`

### Artifact Store

- Async-safe JSON key-value store (persisted to `.daaw_store/artifacts.json`)
- Namespaced keys: `{task_id}.output`, `{task_id}.status`, `{task_id}.metadata`
- Acts as the "bus" moving data between agents
- **Location:** `src/daaw/store/artifact_store.py`

### Critic (Debugger)

- Runs after every task completion
- Evaluates: "Did the output match the `success_criteria` defined in the spec?"
- Issues `WorkflowPatch` operations on failure (retry with feedback, insert new tasks, remove tasks, update inputs)
- **Location:** `src/daaw/critic/critic.py`

---

## Safety Mechanisms

### Circuit Breaker

Tracks consecutive failures per task. Trips after N failures (default 3) to prevent infinite retry loops. Can be reset on success or explicit retry.

**Location:** `src/daaw/engine/circuit_breaker.py`

### Context Pruner

As the workflow grows, memory grows. Passing all memory to every agent is expensive and confusing. The `input_filter` on each `TaskSpec` ensures only relevant artifact keys are passed to the agent.

**Location:** `src/daaw/engine/context_pruner.py`

### Human-in-the-Loop (HITL)

A special `AgentSpec` type called `UserProxy`. When the DAG hits this node, the engine pauses and waits for actual human input via CLI before resuming.

---

## Built-in Agents

| Role | Purpose | File |
|---|---|---|
| `user_proxy` | Human-in-the-loop intake (7-question questionnaire) | `agents/builtin/user_proxy.py` |
| `pm` | Project manager — clarification + drafting | `agents/builtin/pm_agent.py` |
| `breakdown` | Task decomposition into subtasks | `agents/builtin/breakdown_agent.py` |
| `planner` | Compiler wrapper for use in workflows | `agents/builtin/planner_agent.py` |
| `critic` | Evaluation + patch generation | `agents/builtin/critic_agent.py` |
| `generic_llm` | Catch-all LLM-based agent for any role | `agents/builtin/generic_llm_agent.py` |

---

## Extension Points

### Custom Agents

Inherit from `BaseAgent`, implement `async run(task_input) -> AgentResult`, register with the `@register_agent("name")` decorator:

```python
from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.schemas.results import AgentResult

@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run(self, task_input):
        result = f"Processed: {task_input}"
        return AgentResult(output=result, metadata={}, status="success")
```

### Custom Tools

Use the tool registry decorator:

```python
from daaw.tools.registry import tool_registry

@tool_registry.register(name="my_tool", description="Does something", parameters={})
async def my_tool(**kwargs):
    return {"result": "done"}
```

### Custom LLM Providers

Inherit from `LLMProvider` in `src/daaw/llm/base.py` and implement the `chat()` method.

---

## Directory Structure

```
src/daaw/
├── __main__.py              # CLI entry point
├── config.py                # AppConfig dataclass + env loading
├── cli/
│   ├── main.py              # Pipeline orchestration (legacy / run / ui)
│   └── display.py           # Terminal formatting
├── compiler/
│   ├── compiler.py          # Goal → WorkflowSpec
│   ├── plan_reviewer.py     # Interactive review loop
│   └── prompts.py           # LLM system prompts
├── engine/
│   ├── executor.py          # Async DAG executor
│   ├── dag.py               # DAG data structure + validation
│   ├── circuit_breaker.py   # Failure tracking
│   └── context_pruner.py    # Input context filtering
├── agents/
│   ├── base.py              # BaseAgent ABC
│   ├── factory.py           # Agent instantiation
│   ├── registry.py          # Registration system
│   └── builtin/             # Built-in agent implementations
├── llm/
│   ├── base.py              # LLMProvider ABC + LLMMessage/LLMResponse
│   ├── unified.py           # Unified multi-provider dispatcher
│   └── providers/           # Groq, Gemini, OpenAI, Anthropic
├── critic/
│   ├── critic.py            # Task evaluation
│   ├── patch.py             # Runtime DAG patching
│   └── prompts.py           # Evaluation prompts
├── schemas/
│   ├── workflow.py          # WorkflowSpec, TaskSpec, AgentSpec
│   ├── enums.py             # TaskStatus, PatchAction
│   ├── results.py           # AgentResult, TaskResult
│   └── events.py            # WorkflowPatch, PatchOperation
├── store/
│   └── artifact_store.py    # Async JSON key-value store
├── tools/
│   ├── registry.py          # Tool registration
│   └── mock_tools.py        # Example tools
└── ui/
    ├── app.py               # Streamlit dashboard (5 tabs)
    └── demo_data.py         # Pre-loaded demo workflow
```
