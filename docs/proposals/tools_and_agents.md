# Tool & Agent Proposals

**Status:** proposal (no code yet) · **Target:** DAAW v0.4

Eight concrete additions to DAAW's tool/agent surface. Each entry has: what it
is, why it fits DAAW specifically, dependencies, complexity (S/M/L), and
priority. Ranked by impact-per-effort.

Context — what DAAW already has today:

- **Tools** (`@tool_registry.register`): `web_search`, `file_write`,
  `file_read`, `shell_command`, `notify` (webhook).
- **Agents** (`@register_agent`): `generic_llm` (only worker; per-task
  specialization via `tools_allowed` + `system_prompt_override`), plus
  pipeline-role agents `planner` / `pm` / `breakdown` / `user_proxy` /
  `critic`.

Everything below slots into those two registries — no framework changes.

---

## Priority 1 — high ROI, low risk

### 1. `http_request` tool  ·  S

**What:** Generic HTTP client — `method`, `url`, `headers?`, `json?`, `params?`,
`timeout=30`. Returns `{status, headers, body}` with a 64 KB body cap.

**Why it fits:** Right now any integration beyond "search the web" needs a
bespoke tool. One `http_request` unlocks GitHub, Stripe, Notion, weather APIs,
anything — the LLM writes the call, we validate and execute. Already a
de-facto standard in agent frameworks.

**Guardrails:** Deny-list private IPs (RFC1918, link-local, `metadata.google`,
`169.254.169.254`) to prevent SSRF. Max redirect depth 3. Strip `Authorization`
when redirecting cross-host. Env-var allowlist mode for production.

**Deps:** `httpx` (already a transitive dep via Anthropic SDK).

**Complexity:** Small. ~80 lines tool impl + ~120 lines tests covering SSRF
denial, redirect limits, timeouts, large-body truncation.

---

### 2. `python_exec` tool  ·  M

**What:** Run a short Python snippet in a sandboxed subprocess; return stdout,
stderr, and the value of the last expression (if any). Default 10 s timeout,
64 MB memory cap, no network.

**Why it fits:** DAAW already has `shell_command` — but asking an LLM to
compute `sum(123 * 0.08 for i in ...)` via bash is absurd. A dedicated code tool
is the single biggest unlock for quantitative tasks (data wrangling, regex,
JSON massage). Every serious agent framework (OpenAI, CrewAI, LangChain) ships
one.

**Guardrails:** Subprocess with `resource.setrlimit` on POSIX + job objects on
Windows. Deny imports of `subprocess`, `ctypes`, `socket` via AST pre-scan.
Temp working dir mapped under `$DAAW_SANDBOX_DIR/py_exec/<uuid>`.

**Deps:** stdlib. For stronger isolation, pluggable E2B / Firecracker backend
later.

**Complexity:** Medium. Needs careful tests for resource limits on both OSes.

---

### 3. `browser_fetch` tool  ·  M

**What:** `url` → markdown of the fully-rendered page. Runs headless Chromium
via Playwright, waits for `networkidle`, runs Mozilla Readability to strip
chrome, returns cleaned markdown + detected title + canonical URL.

**Why it fits:** `web_search` only returns DuckDuckGo snippets (2–3 sentences).
Half the time the answer is in the page body. Today the LLM can't read it
because modern sites are JS-rendered. This fixes that — the existing
`researcher` prompts can cite real content.

**Guardrails:** Reuse `http_request`'s SSRF denylist. 30 s hard timeout. 2 MB
body cap after readability extraction.

**Deps:** `playwright` (~80 MB for the browser; optional install extra
`daaw[browser]`). `readability-lxml` or Mozilla's `readability.js` via pyppeteer.

**Complexity:** Medium — mostly packaging. The tool logic is ~60 lines.

---

### 4. `vector_search` tool + artifact-store backing  ·  M

**What:** Two functions, `vector_upsert(namespace, id, text, metadata?)` and
`vector_search(namespace, query, k=5)`. Stores embeddings in a local Chroma
collection under `$DAAW_STORE_DIR/vectors/`. Embeddings via the configured
provider's embedding endpoint (Gemini, OpenAI, or a local SentenceTransformers
fallback).

**Why it fits:** DAAW has no long-term memory between workflows. Right now
task_002 gets a 2000-char truncated version of task_001's output — anything
older is lost. A vector store lets the critic/retry loop reference past
successes, and lets `researcher`-style agents deduplicate web hits across runs.
Directly addresses the "context pruning is lossy" limitation flagged in
MEMORY.md.

**Deps:** `chromadb` (pure-Python; ships with DuckDB + HNSWlib binaries).
Embedding provider abstraction reuses existing `UnifiedLLMClient`.

**Complexity:** Medium. Needs a clean namespace per workflow + a retention
policy (last 30 days by default).

---

## Priority 2 — high impact, medium scope

### 5. `sql_query` tool  ·  S

**What:** `sql_query(connection_name, sql, params?)` — read-only by default.
Connection strings loaded from env: `DAAW_SQL_{NAME}=sqlite:///path.db` or
`postgresql://...`. Returns rows as a list-of-dicts, capped at 1000 rows /
256 KB.

**Why it fits:** "Agent that can query our prod read-replica" is a very common
real-world use case that today requires a bespoke integration. Read-only
default + allowlist of connection names is safe; bumps agent utility massively
for anyone running DAAW against an internal DB.

**Guardrails:** Parser-level rejection of `UPDATE`/`INSERT`/`DELETE`/`DROP`
unless connection is explicitly marked `writable=true` via env. 10 s query
timeout. Never log query text at default verbosity.

**Deps:** `sqlalchemy` (already heavy but universal) or just `sqlite3` + `asyncpg`
behind a thin wrapper.

**Complexity:** Small-to-medium — SQL-injection surface is zero since the LLM
supplies both the query and parameters, but we can at least AST-parse to block
write statements in read-only mode.

---

### 6. `mcp_client` tool  ·  M

**What:** Connect to any [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio or SSE, list its tools, and expose each one inside DAAW as if
it were registered natively. Configured via a JSON file listing servers.

**Why it fits:** MCP is becoming the standard cross-framework tool protocol in
2026 — every major framework (LangChain, CrewAI, AutoGen, OpenAI SDK) now
supports it. Wiring up one MCP client gives DAAW free access to the existing
MCP server ecosystem (filesystem, GitHub, Slack, Postgres, Puppeteer,
100+ others) without writing bespoke adapters per service. Highest
leverage-to-effort ratio on this list.

**Deps:** `mcp` (official Python SDK).

**Complexity:** Medium — tool proxies are generated at startup by listing
`server.list_tools()` and wrapping each in a DAAW `@tool_registry.register`
call. Biggest work is graceful handling of server restarts.

---

## Priority 3 — specialist agents

### 7. `researcher` agent  ·  S

**What:** A pre-configured `BaseAgent` subclass that runs a 3-step loop:
`web_search` → `browser_fetch` the top 3 hits → emit a structured JSON answer
with citations (`[{claim, source_url, quote}]`). Own system prompt tuned for
non-hallucinating output.

**Why it fits:** Every DAAW workflow that starts with "look up X" currently
reinvents this. Cheap to add, immediately useful, and it's a good showcase for
the `tools_allowed` + `system_prompt_override` pattern — the implementation
is mostly just a wrapper that preconfigures `generic_llm` with the right tools
and prompt.

**Complexity:** Small. Ship once `browser_fetch` lands.

---

### 8. `coder` agent  ·  M

**What:** Specialist agent that iterates on a code task: write → `python_exec`
against inline tests → read stderr → fix. Max 5 iterations. Produces the final
file content plus a "tests pass" flag.

**Why it fits:** Most "write a script that X" tasks fail today because
`generic_llm` has no feedback loop beyond the critic. Pairing `python_exec`
with a purpose-built prompt + iteration budget is the cleanest way to get
reliable code generation without a full multi-agent framework.

**Deps:** tool #2 (`python_exec`).

**Complexity:** Medium. The iteration loop is easy; prompt engineering for
"generate code with assertions I can check" is the real work.

---

## Not proposed (considered and dropped)

- **`browser_use` full browser automation (click, type, navigate):** Powerful
  but a huge maintenance surface. `browser_fetch` covers ~80% of demand.
  Revisit if a user has a concrete "automate this SaaS" need.
- **Email send/receive:** Abuse magnet, heavy auth, narrow use case. If
  needed, users can already wire SMTP via `http_request` or a webhook.
- **Calendar tool:** Covered better by MCP (Google Calendar / Outlook MCP
  servers exist).
- **Custom embeddings model:** YAGNI; defer until `vector_search` proves the
  abstraction.

---

## Suggested rollout order

1. `http_request` (1 day) → immediate utility.
2. `mcp_client` (2 days) → unlocks dozens of existing tools for free.
3. `python_exec` (2 days) → enables `coder`.
4. `browser_fetch` (2 days) → enables `researcher`.
5. `vector_search` (3 days) → fixes the context-pruning limitation.
6. `researcher` agent (half day given 1–4).
7. `coder` agent (1 day given 1–3).
8. `sql_query` (1 day, can parallel anything).

**Total:** roughly two weeks of focused work for all eight.

---

## Sources

- [OpenAI Agents SDK — Tools](https://openai.github.io/openai-agents-python/tools/)
- [awesome-ai-agents (e2b-dev)](https://github.com/e2b-dev/awesome-ai-agents)
- [browser-use](https://github.com/browser-use/browser-use) —
  full browser-automation agent for cases beyond `browser_fetch`.
- [Model Context Protocol](https://modelcontextprotocol.io) — standard tool
  protocol adopted across all 2026 agent frameworks.
- [LangChain vs CrewAI vs AutoGen 2026 comparison](https://pecollective.com/blog/ai-agent-frameworks-compared/)
  for tool-surface benchmarking.
