# DAAW Compiler-Runtime

## Architecture

Compiler-Runtime paradigm for autonomous workflow orchestration:
- **Compiler (Planner)**: LLM converts fuzzy user goal into a strict JSON execution graph (WorkflowSpec)
- **Runtime (Engine)**: Deterministic DAG executor -- runs tasks in parallel or sequentially
- **Workers (Agents)**: Ephemeral processes instantiated just-in-time, destroyed after task completion
- **Debugger (Critic)**: Post-execution evaluator that checks outputs and patches the workflow at runtime

## Layer Architecture

| Layer | Components |
|-------|-----------|
| L0: Foundation | `llm/` (provider abstraction), `tools/` (registry + implementations) |
| L1: Schema | `schemas/` (Pydantic models: WorkflowSpec, TaskSpec, AgentSpec, results, events) |
| L2: State | `store/` (async key-value artifact store with namespacing) |
| L3: Logic | `agents/` (factory + registry + builtins), `engine/` (DAG executor, circuit breaker, context pruner) |
| L4: Intelligence | `compiler/` (goal-to-spec), `critic/` (evaluation + patching) |
| L5: Interface | `cli/` (pipeline orchestration), `ui/` (Streamlit dashboard + demo UI) |

## Key Paths

- Entry point: `src/daaw/cli/main.py` (CLI) or `src/daaw/ui/app.py` (Dashboard) or `src/daaw/ui/demo_app.py` (Demo)
- Agent base class: `src/daaw/agents/base.py`
- Agent registry: `src/daaw/agents/registry.py` (use `@register_agent("name")`)
- Tool registry: `src/daaw/tools/registry.py` (use `@tool_registry.register(...)`)
- Tool implementations: `src/daaw/tools/` (real_tools.py for actual tools, mock_tools.py for testing)
- LLM providers: `src/daaw/llm/providers/` (groq, gemini, openai, anthropic, gateway)
- Gateway provider: `src/daaw/llm/providers/gateway_provider.py` (Ollama/LM Studio/vLLM)
- Config: `src/daaw/config.py` (singleton from env vars, `reset_config()` for hot-reload)

## Commands

```bash
# Run full pipeline
python -m daaw run --goal "your goal" --provider groq
# Run with local LLM
python -m daaw run --goal "your goal" --provider gateway
# Run legacy pipeline
python -m daaw legacy
# Launch dashboard UI
python -m daaw ui
# Launch demo UI
python -m daaw demo
# Run tests
pytest tests/ -x -q
# Run unit tests only (no API keys needed)
pytest tests/ -x -q -m "not integration"
```

## Conventions

- All agents inherit from `BaseAgent` and implement `async def run(self, task_input) -> AgentResult`
- Tools are async functions registered via `@tool_registry.register()` decorator
- LLM providers implement `LLMProvider` ABC from `llm/base.py`
- Pydantic v2 for all data contracts -- strict validation at every boundary
- Async-native throughout -- use `asyncio.gather()` for parallelism
- Agent config dict contains `tools_allowed`, `system_prompt_override`, and optional provider/model overrides
- Tool execution in agents uses a tool-call loop: LLM requests tools -> agent dispatches -> results fed back -> LLM continues
- Gateway provider: merges system role for compatibility, caps max_tokens to 700 for local models, retries on crash/reload

## Local LLM Notes

- Gateway provider auto-detects Ollama/LM Studio and adapts
- Uses native Ollama tool calling API (structured, not text injection)
- Falls back to text-based tool call injection if native API rejected
- Parses 6 tool call formats: Llama XML, Qwen, Gemma E4B keyword, Gemma E2B compact, bracket, bare function
- Compiler uses flat JSON schema (no nested agent object) for small model compatibility
- `_fixup_json_structure()` auto-repairs: flat-to-nested conversion, misplaced fields, string deps
- `_fixup_agent_roles()` auto-corrects: tool-as-role, internal roles (critic/planner/pm/breakdown)
- `_repair_json()` fixes: markdown fences, trailing commas, unclosed brackets
- Sequential execution (`max_concurrent=1`) for local models to avoid GPU contention
- Context pruner truncates dependency outputs to 2000 chars
- Crash recovery with escalating waits (15-45s) and inference health probes

## Testing

- 145+ tests across 16 files in `tests/`
- Unit tests: no external deps needed
- Integration tests: need `GROQ_API_KEY` env var
- Use `pytest -m "not integration"` for fast local runs
