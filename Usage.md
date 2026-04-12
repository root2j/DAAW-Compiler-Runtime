# Usage

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your provider config:

   ```env
   # Cloud providers (set whichever you have)
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key

   # Local LLM via Ollama / LM Studio / vLLM (no API key needed)
   GATEWAY_URL=http://localhost:11434/v1
   GATEWAY_MODEL=gemma4:e4b

   # Optional: webhook notifications
   # NOTIFY_WEBHOOK_URL=https://discord.com/api/webhooks/...
   # NOTIFY_WEBHOOK_TYPE=discord
   ```

3. Set the Python path (needed every session, or add to your shell profile):

   ```bash
   export PYTHONPATH=src
   ```

---

## Commands

### `run` -- The Full Compiler-Runtime Pipeline

```bash
python -m daaw run --goal "Plan a multi-location trip to Japan with costs" --provider gateway
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--goal` | *(prompts you)* | The workflow you want to build |
| `--provider` | `groq` | LLM provider: `groq`, `gemini`, `openai`, `anthropic`, `gateway` |
| `--model` | *(provider default)* | Override the model (e.g. `gemma4:e4b`, `llama-3.3-70b-versatile`) |
| `--mock-tools` | `false` | Use fake tool responses instead of real execution |

**What happens:**

1. **Compile** -- LLM generates a WorkflowSpec (a DAG of tasks) from your goal
2. **Review** -- You see the plan and can:
   - Type `yes` to approve
   - Type `abort` to cancel
   - Type feedback to refine (e.g. "add an error handling step")
3. **Execute** -- Tasks run in parallel (cloud) or sequentially (local LLM)
4. **Critique** -- Each task's output is evaluated; failures auto-retry
5. **Summary** -- Final results with success/failure counts

**Examples:**

```bash
# Local LLM via Ollama (recommended for free usage)
python -m daaw run --goal "Research AI trends and write a summary" --provider gateway

# Groq (free tier, fast)
python -m daaw run --goal "Classify incoming emails" --provider groq

# Gemini (cheap)
python -m daaw run --goal "Plan a marketing campaign" --provider gemini

# Interactive mode
python -m daaw run --provider gateway
```

---

### `ui` -- Dashboard UI

Launches a Streamlit web app with interactive visualizations of the full pipeline:

```bash
python -m daaw ui
python -m daaw ui --port 8502
```

**6 Tabs:**

1. **Architecture** -- system flow diagram, layer cards, design patterns
2. **Compiler** -- compilation pipeline, task cards, WorkflowSpec JSON
3. **DAG Visualization** -- interactive Plotly graph of the workflow DAG
4. **Execution Timeline** -- Gantt chart showing parallel/sequential execution
5. **Critic & Results** -- per-task pass/fail verdicts and output previews
6. **Tools** -- registered tools, usage statistics, call distribution

**Demo Mode** loads a pre-built "Goa Trip Planner" pipeline (6 tasks) with real execution data -- no API key required.

**Live Mode** lets you select a provider/model, type a goal, and compile + execute real workflows.

---

### `demo` -- Under-the-Hood Demo UI

A presentation-grade UI that shows the pipeline step by step:

```bash
python -m daaw demo
python -m daaw demo --port 8503
```

**Features:**

- **Pipeline stepper** -- visual indicator showing current stage (Goal --> Compile --> Execute --> Critique --> Summary)
- **Chat interface** -- Claude-like conversation showing each pipeline component's messages
- **Live log console** -- terminal-style scrolling logs with timestamps
- **DAG visualization** -- interactive graph with color-coded task status
- **Task output viewer** -- inspect individual task results and tool calls
- **Source code peek** -- syntax-highlighted snippets from the actual codebase
- **Performance metrics** -- Gantt timeline, speedup calculation, tool distribution

**Demo Walkthrough** mode: 12-step clickable walkthrough with Next/Back/Play All controls.

**Live Mode**: real compilation and execution with any provider.

---

### `legacy` -- The Original Pipeline

Runs the questionnaire --> PM --> breakdown flow:

```bash
python -m daaw legacy
```

**What happens:**

1. **UserProxyAgent** -- walks you through intake questions
2. **PMAgent** -- asks clarifying questions, generates a project draft
3. **BreakdownAgent** -- breaks the approved draft into detailed subtasks

---

## Local LLM Setup (Ollama)

```bash
# Install Ollama (https://ollama.ai)
# Pull a model
ollama pull gemma4:e4b

# Verify it works
curl http://localhost:11434/v1/chat/completions \
  -d '{"model":"gemma4:e4b","messages":[{"role":"user","content":"Hi"}]}'

# Configure DAAW
echo 'GATEWAY_URL=http://localhost:11434/v1' >> .env
echo 'GATEWAY_MODEL=gemma4:e4b' >> .env

# Run
python -m daaw run --goal "Write a poem about nature" --provider gateway
```

**Recommended models:**

| Model | Size | Best for |
|---|---|---|
| `gemma4:e4b` | 9GB | Multi-task planning + execution (recommended) |
| `gemma4:e2b-it-q4_K_M` | 7GB | Simple single-task execution (faster) |
| `llama3.2` | 2GB | Very fast, limited planning ability |

**Local LLM optimizations:**

- Tasks run sequentially (one at a time) to avoid GPU contention
- Token-optimized prompts (flat JSON schema, ~300 tokens)
- Auto-repair for malformed JSON (misplaced fields, missing braces, string deps)
- Crash recovery with health probes (waits for model reload)
- Native Ollama tool calling (structured API, not text injection)
- Context pruner truncates dependency outputs to 2000 chars

---

## Quick Test (No API Keys Needed)

```bash
# Run unit tests
pytest tests/ -x -q -m "not integration"

# Launch demo UI with pre-loaded data
python -m daaw ui

# Verify imports
python -c "from daaw.schemas.workflow import WorkflowSpec; print('OK')"
```
