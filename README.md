# DAAW — Compiler-Runtime for Autonomous Workflows

DAAW transforms natural language goals into executable DAG-based workflows. Instead of chatty back-and-forth agent interactions, it uses a **Compiler-Runtime paradigm**: plan first, review the plan, then execute deterministically.

**Goal → Plan → Review → Execute → Critique**

## How It Works

1. You describe a workflow goal in plain English
2. The **Compiler** (an LLM) generates a structured execution plan (a DAG of tasks)
3. You **review** the plan and refine it if needed
4. The **Runtime** executes tasks in parallel where possible, respecting dependencies
5. A **Critic** evaluates each task's output and can auto-retry or patch the workflow

## Quick Start

### Prerequisites

- Python 3.9+
- At least one LLM API key (Groq, Gemini, OpenAI, or Anthropic)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

Set the Python path:

```bash
export PYTHONPATH=src
```

### Run

```bash
# Full compiler-runtime pipeline
python -m daaw run --goal "Automate email classification" --provider groq

# Interactive mode (prompts you for the goal)
python -m daaw run --provider groq

# Legacy questionnaire pipeline
python -m daaw legacy

# Launch the Streamlit demo UI (no API keys needed)
python -m daaw ui
```

## Commands

| Command | Description |
|---|---|
| `python -m daaw run` | Full compiler-runtime pipeline: compile → review → execute → critique |
| `python -m daaw legacy` | Original questionnaire → PM → breakdown flow |
| `python -m daaw ui` | Streamlit web dashboard with interactive visualizations |

See [Usage.md](Usage.md) for detailed command documentation and examples.

## Architecture

The system is built around five abstraction layers:

| Layer | Components |
|---|---|
| **Foundation** | Unified LLM client (Groq, Gemini, OpenAI, Anthropic), Tool registry |
| **Schema** | Pydantic models — `WorkflowSpec`, `TaskSpec`, `AgentSpec` |
| **State** | Artifact Store — async key-value bus for inter-task communication |
| **Logic** | DAG Executor (parallel async), Agent Factory (just-in-time instantiation) |
| **Intelligence** | Compiler prompts (plan generation), Critic prompts (evaluation + patching) |

Key safety mechanisms: **Circuit Breaker** (prevents infinite retries), **Context Pruner** (limits memory passed to agents), **Human-in-the-Loop** nodes (pauses for human input).

See [architecture.md](architecture.md) for the full architectural breakdown.

## Extending DAAW

### Custom Agents

```python
from daaw.agents.base import BaseAgent
from daaw.agents.registry import register_agent
from daaw.schemas.results import AgentResult

@register_agent("my_agent")
class MyAgent(BaseAgent):
    async def run(self, task_input):
        return AgentResult(output="result", metadata={}, status="success")
```

### Custom Tools

```python
from daaw.tools.registry import tool_registry

@tool_registry.register(name="my_tool", description="Does something", parameters={})
async def my_tool(**kwargs):
    return {"result": "done"}
```

## Project Structure

```
src/daaw/
├── cli/          # CLI entry points and terminal display
├── compiler/     # Goal → WorkflowSpec compilation
├── engine/       # DAG executor, circuit breaker, context pruner
├── agents/       # Base agent, factory, registry, built-in agents
├── llm/          # Unified multi-provider LLM client
├── critic/       # Task evaluation and runtime patching
├── schemas/      # Pydantic data models
├── store/        # Artifact store (async key-value)
├── tools/        # Tool registry and mock tools
└── ui/           # Streamlit dashboard
```

## Supported LLM Providers

| Provider | Default Model |
|---|---|
| Groq | LLaMA-3.1 |
| Google Gemini | Gemini-2.5-flash |
| OpenAI | GPT-4o |
| Anthropic | Claude-3.5-sonnet |

## Testing

```bash
pytest
```

Tests cover DAG validation, executor parallelism, agent registry, artifact store, circuit breaker, compiler, critic, and context pruning.
