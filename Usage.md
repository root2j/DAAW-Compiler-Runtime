# Usage

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your API keys (at least one required):

   ```env
   GROQ_API_KEY=your_groq_key
   GEMINI_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

3. Set the Python path (needed every session, or add to your shell profile):

   ```bash
   export PYTHONPATH=src
   ```

---

## Commands

### `legacy` — The Original Pipeline

Runs the questionnaire → PM → breakdown flow through the new architecture:

```bash
python -m daaw legacy
```

**What happens:**

1. **UserProxyAgent** — walks you through 7 intake questions (what to automate, triggers, apps, etc.)
2. **PMAgent** — asks clarifying questions, generates a project draft, lets you refine it until you type `yes`
3. **BreakdownAgent** — takes the approved draft and breaks it into detailed subtasks

This uses Groq for the PM agent and Gemini for the breakdown agent (same as the original code). You need at least those two API keys.

---

### `run` — The Full Compiler-Runtime Pipeline

```bash
python -m daaw run --goal "Automate email classification" --provider groq
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--goal` | *(prompts you)* | The workflow you want to build |
| `--provider` | `groq` | LLM provider for the compiler/critic (`groq`, `gemini`, `openai`, `anthropic`) |
| `--model` | *(provider default)* | Override the model (e.g. `llama-3.1-70b-versatile`) |

**What happens:**

1. **Compile** — LLM generates a WorkflowSpec (a DAG of tasks) from your goal
2. **Review** — You see the plan and can:
   - Type `yes` to approve
   - Type `abort` to cancel
   - Type feedback to refine (e.g. "add an error handling step")
3. **Execute** — Tasks run in parallel where possible, sequentially where dependencies require
4. **Critique** — Each task's output is evaluated against its success criteria; failures auto-retry
5. **Summary** — Final results with success/failure counts

**Examples:**

```bash
# Let it prompt you for the goal interactively
python -m daaw run --provider groq

# Provide goal directly
python -m daaw run --goal "When a new row is added to Google Sheets, send a Slack notification and log to Notion" --provider gemini

# Use OpenAI with a specific model
python -m daaw run --goal "Classify incoming emails" --provider openai --model gpt-4o
```

---

### `ui` — Demo UI (No API Keys Needed)

Launches a Streamlit web app with interactive visualizations of the full pipeline:

```bash
python -m daaw ui
python -m daaw ui --port 8502
```

Or directly via Streamlit:

```bash
PYTHONPATH=src streamlit run src/daaw/ui/app.py
```

**5 Tabs:**

1. **Architecture** — system flow diagram and component cards
2. **Compiler** — compilation pipeline, task table, WorkflowSpec JSON
3. **DAG Visualization** — interactive Plotly graph of the workflow DAG
4. **Execution Timeline** — Gantt chart showing parallel task execution
5. **Critic & Results** — per-task pass/fail verdicts and output previews

**Demo Mode** loads a pre-built "E-commerce Order Processing Pipeline" (6 tasks, diamond DAG) with realistic results — no API key required.

**Live Mode** (requires an API key) lets you type a goal and compile it into a WorkflowSpec using any provider.

---

## Quick Test (No API Keys Needed)

Verify everything imports and the DAG engine works:

```python
python -c "
import asyncio
from daaw.schemas.workflow import *
from daaw.engine.dag import DAG

spec = WorkflowSpec(
    name='Test', description='test',
    tasks=[
        TaskSpec(id='a', name='A', description='d', agent=AgentSpec(role='pm')),
        TaskSpec(id='b', name='B', description='d', agent=AgentSpec(role='pm')),
        TaskSpec(id='c', name='C', description='d', agent=AgentSpec(role='pm'),
                 dependencies=[DependencySpec(task_id='a'), DependencySpec(task_id='b')]),
    ]
)
dag = DAG(spec)
print('Validation:', dag.validate())
print('Ready (should be a,b):', sorted(dag.get_ready_tasks()))
"
```

---

## Minimum Viable Test

If you only have a **Groq API key**, run:

```bash
python -m daaw legacy
```

This exercises the full architecture end-to-end (UserProxy → PM via Groq → Breakdown needs Gemini though).

If you have both **Groq + Gemini** keys, `legacy` runs the complete original pipeline. For `run`, just a single provider key is enough.
