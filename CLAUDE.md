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
| L0: Foundation | `llm/` (provider abstraction + rate limiter + streaming), `tools/` (registry + implementations) |
| L1: Schema | `schemas/` (Pydantic models: WorkflowSpec, TaskSpec, AgentSpec, results, events) |
| L2: State | `store/` (async key-value artifact store with namespacing) |
| L3: Logic | `agents/` (factory + registry + builtins), `engine/` (DAG executor, circuit breaker, context pruner) |
| L4: Intelligence | `compiler/` (goal-to-spec + anti-hallucination prompt), `critic/` (tiered evaluation + patching) |
| L5: Interface | `cli/` (pipeline orchestration), `ui/` (dashboard + demo + **chat UI**) |
| L6: Quality | `interaction.py` (HITL handler), `llm/rate_limiter.py`, `llm/model_probe.py` |

## Key Paths

- Entry point: `src/daaw/cli/main.py` (CLI) or `src/daaw/ui/app.py` (Dashboard) or `src/daaw/ui/demo_app.py` (Demo) or `src/daaw/ui/chat_app.py` (**Chat**)
- Agent base class: `src/daaw/agents/base.py`
- Agent registry: `src/daaw/agents/registry.py` (use `@register_agent("name")`)
- Tool registry: `src/daaw/tools/registry.py` (use `@tool_registry.register(...)`)
- Tool implementations: `src/daaw/tools/` (real_tools.py for actual tools, mock_tools.py for testing)
- LLM providers: `src/daaw/llm/providers/` (groq, gemini, openai, anthropic, gateway)
- Gateway provider: `src/daaw/llm/providers/gateway_provider.py` (Ollama/LM Studio/vLLM)
- Rate limiter: `src/daaw/llm/rate_limiter.py` (per-provider RPM + token budget)
- Model probe: `src/daaw/llm/model_probe.py` (pre-flight JSON compatibility check)
- HITL handler: `src/daaw/interaction.py` (Stdin/Queue/AutoAnswer/Null)
- Config: `src/daaw/config.py` (singleton from env vars, `reset_config()` for hot-reload)
- Streaming: `src/daaw/ui/_streaming_display.py` (progressive JSON pretty-print)
- Eval runner: `scripts/run_eval.py` (batch benchmark, 21 realistic prompts)

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
# Launch chat UI (recommended — Swiss editorial design)
python -m daaw chat
# Run tests
pytest tests/ -x -q
# Run unit tests only (no API keys needed)
pytest tests/ -x -q -m "not integration"
# Run eval benchmark
python scripts/run_eval.py --provider groq --model llama-3.1-8b-instant
# Verify tools
python scripts/verify_tools.py
# Probe local models
python scripts/probe_local_models.py
```

## Tools (7 real + 4 aliases)

| Tool | Description | Since |
|------|------------|-------|
| `web_search` | DuckDuckGo search | v0.1 |
| `file_write` | Write to sandboxed file | v0.1 |
| `file_read` | Read from sandboxed file | v0.1 |
| `shell_command` | Execute shell (30s timeout, dangerous-cmd blocklist) | v0.1 |
| `notify` | Webhook notification (Discord/Slack/generic) | v0.2 |
| `http_request` | Generic HTTP client with SSRF protection | v0.5.0 |
| `python_exec` | Sandboxed Python subprocess (30s, blocked imports) | v0.5.0 |

Aliases (prevent LLM hallucination loops): `brave_search`, `search`, `google_search`, `WebSearch` → all forward to `web_search`.

## LLM Providers (6)

| Provider | Config env var | Notes |
|----------|---------------|-------|
| `groq` | `GROQ_API_KEY` | Free tier, fast, recommended for execution |
| `gemini` | `GEMINI_API_KEY` | 20 req/day free on flash-lite |
| `openai` | `OPENAI_API_KEY` | |
| `anthropic` | `ANTHROPIC_API_KEY` | $5 free credit on signup |
| `gateway` | `GATEWAY_URL` | Ollama / LM Studio / vLLM (OpenAI-compatible) |
| `claude_api` | `DAAW_CLAUDE_API_URL` | Claude Code API gateway (uses your subscription) |

## Split-Provider Compile

Use a strong model for planning, a cheap one for execution:

```bash
DAAW_COMPILER_PROVIDER=groq
DAAW_COMPILER_MODEL=llama-3.3-70b-versatile
# Execution uses whatever the CLI/UI selects (e.g. groq/llama-3.1-8b-instant)
```

The Compiler reads these at init and overrides the default provider for compile only.

## Conventions

- All agents inherit from `BaseAgent` and implement `async def run(self, task_input) -> AgentResult`
- Tools are async functions registered via `@tool_registry.register()` decorator
- LLM providers implement `LLMProvider` ABC from `llm/base.py`; optional `chat_stream()` for SSE
- Pydantic v2 for all data contracts -- strict validation at every boundary
- Async-native throughout -- use `asyncio.gather()` for parallelism
- Agent config dict contains `tools_allowed`, `system_prompt_override`, and optional provider/model overrides
- Tool execution in agents uses a tool-call loop: LLM requests tools -> agent dispatches -> results fed back -> LLM continues
- Anti-hallucination: tasks with `tools_allowed` must actually call a tool or status=failure; pseudo-tool-call text detected
- Data preservation: research-tool outputs auto-appended if agent writes a vague summary
- Gateway provider: merges system role for compatibility, caps max_tokens to 700 for local models, retries on crash/reload

## Local LLM Notes

- Gateway provider auto-detects Ollama/LM Studio and adapts
- Ollama-specific: sends `options.num_ctx` (default 4096) + `keep_alive` (default 5m)
- CUDA OOM auto-retry: halves num_ctx on 500 "CUDA error" responses, floors at 1024
- Uses native Ollama tool calling API (structured, not text injection)
- Falls back to text-based tool call injection if native API rejected
- Parses 6 tool call formats: Llama XML, Qwen, Gemma E4B keyword, Gemma E2B compact, bracket, bare function
- Compiler uses flat JSON schema (no nested agent object) for small model compatibility
- `_fixup_json_structure()` auto-repairs: flat-to-nested conversion, misplaced fields, string deps, dangling deps, truncated stubs
- `_fixup_agent_roles()` auto-corrects: tool-as-role, internal roles (critic/planner/pm/breakdown)
- `_repair_json()` fixes: markdown fences, trailing commas, unclosed brackets
- Sequential execution (`max_concurrent=1`) for local models to avoid GPU contention
- Context pruner truncates dependency outputs (configurable via `DAAW_MAX_DEP_CHARS`, default 2000)
- Reserved-token sanitizer strips `<unused*>`, `<tool|>`, `<bos>`, etc. from dependency outputs
- Crash recovery with escalating waits (15-45s) and inference health probes
- Temperature escalation on compile retries: 0.4 → 0.6 → 0.85 → 1.0
- Model compatibility probe: quick JSON test at startup, shown as sidebar badge

## Testing

- 234+ tests across 18+ files in `tests/`
- Unit tests: no external deps needed
- Integration tests: need `GROQ_API_KEY` env var
- Use `pytest -m "not integration"` for fast local runs
- CI: `.github/workflows/ci.yml` runs on Ubuntu + Windows × Python 3.10 + 3.11
- Eval benchmark: `scripts/run_eval.py` with 21 realistic prompts (see `docs/eval/`)
