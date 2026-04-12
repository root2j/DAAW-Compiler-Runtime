# DAAW — Compiler-Runtime for Autonomous Workflows

DAAW transforms natural language goals into executable DAG-based workflows. Instead of chatty back-and-forth agent interactions, it uses a **Compiler-Runtime paradigm**: plan first, review the plan, then execute deterministically.

**Goal --> Plan --> Review --> Execute --> Critique**

## How It Works

1. You describe a workflow goal in plain English
2. The **Compiler** (an LLM) generates a structured execution plan (a DAG of tasks)
3. You **review** the plan and refine it if needed
4. The **Runtime** executes tasks in parallel (cloud APIs) or sequentially (local LLMs)
5. A **Critic** evaluates each task's output and can auto-retry or patch the workflow

## Quick Start

### Prerequisites

- Python 3.9+
- At least one LLM provider:
  - **Cloud**: Groq, Gemini, OpenAI, or Anthropic API key
  - **Local**: Ollama with any model (e.g. `gemma4:e4b`)

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
# Cloud providers (set whichever you have)
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Local LLM via Ollama (no API key needed)
GATEWAY_URL=http://localhost:11434/v1
GATEWAY_MODEL=gemma4:e4b
```

Set the Python path:

```bash
export PYTHONPATH=src
```

### Run

```bash
# Full compiler-runtime pipeline (cloud)
python -m daaw run --goal "Plan a trip to Japan with costs" --provider groq

# Full pipeline (local LLM via Ollama)
python -m daaw run --goal "Plan a trip to Japan with costs" --provider gateway

# Launch the Streamlit dashboard UI
python -m daaw ui

# Launch the Under-the-Hood demo UI
python -m daaw demo

# Interactive mode (prompts you for the goal)
python -m daaw run --provider groq

# Legacy questionnaire pipeline
python -m daaw legacy
```

## Commands

| Command | Description |
|---|---|
| `python -m daaw run` | Full pipeline: compile --> review --> execute --> critique |
| `python -m daaw legacy` | Original questionnaire --> PM --> breakdown flow |
| `python -m daaw ui` | Streamlit dashboard with interactive visualizations |
| `python -m daaw demo` | Under-the-Hood demo UI with step-by-step pipeline walkthrough |

See [Usage.md](Usage.md) for detailed command documentation and [INSTALL.md](INSTALL.md) for the Linux/macOS setup guide.

## Architecture

The system is built around five abstraction layers:

| Layer | Components |
|---|---|
| **Foundation** | Unified LLM client (6 providers), Tool registry |
| **Schema** | Pydantic models -- `WorkflowSpec`, `TaskSpec`, `AgentSpec` |
| **State** | Artifact Store -- async key-value bus for inter-task communication |
| **Logic** | DAG Executor (parallel/sequential), Agent Factory, Circuit Breaker |
| **Intelligence** | Compiler (plan generation), Critic (evaluation + patching) |

Key safety mechanisms: **Circuit Breaker** (prevents infinite retries), **Context Pruner** (limits context passed to agents), **Human-in-the-Loop** nodes (pauses for human input).

See [architecture.md](architecture.md) for the full architectural breakdown.

## Supported LLM Providers

| Provider | Default Model | Type |
|---|---|---|
| Groq | llama-3.3-70b-versatile | Cloud (free tier) |
| Google Gemini | gemini-2.5-flash | Cloud |
| OpenAI | gpt-4.1-mini | Cloud |
| Anthropic | claude-sonnet-4-6 | Cloud |
| **Gateway** | gemma4:e4b | **Local (Ollama, LM Studio, vLLM)** |

The Gateway provider works with any OpenAI-compatible endpoint. It auto-detects local model quirks (tool calling format, context limits, crash recovery) and adapts.

## Local LLM Support

DAAW works fully offline with local models via Ollama:

```bash
# Install Ollama and pull a model
ollama pull gemma4:e4b

# Configure in .env
GATEWAY_URL=http://localhost:11434/v1
GATEWAY_MODEL=gemma4:e4b

# Run
python -m daaw run --goal "Research AI trends" --provider gateway
```

Local LLM optimizations built-in:
- **Sequential execution** -- tasks run one at a time (local GPUs handle one request)
- **Token-optimized prompts** -- flat JSON schema, lean system prompts
- **Auto-repair** -- fixes malformed JSON, misplaced fields, wrong agent roles
- **Crash recovery** -- retries with health probes when model crashes/reloads
- **Native tool calling** -- uses Ollama's structured tool API

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
├── compiler/     # Goal --> WorkflowSpec compilation + JSON repair
├── engine/       # DAG executor, circuit breaker, context pruner
├── agents/       # Base agent, factory, registry, built-in agents
├── llm/          # Unified multi-provider LLM client
│   └── providers/  # Groq, Gemini, OpenAI, Anthropic, Gateway
├── critic/       # Task evaluation and runtime patching
├── schemas/      # Pydantic data models
├── store/        # Artifact store (async key-value)
├── tools/        # Tool registry, real tools, mock tools
├── integrations/ # Webhook notifications (Discord, Slack)
└── ui/           # Streamlit dashboard + demo UI
```

## Testing

```bash
# All unit tests (no API keys needed)
pytest tests/ -x -q -m "not integration"

# Full test suite (needs GROQ_API_KEY)
pytest tests/ -x -q
```

145+ tests covering DAG validation, executor parallelism, agent registry, artifact store, circuit breaker, compiler, critic, context pruning, tool registry, and LLM unified client.
