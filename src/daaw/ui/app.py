"""DAAW Compiler-Runtime — Pitch-Ready Demo UI."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DAAW Compiler-Runtime",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from daaw.ui.demo_data import (
    DEMO_COMPILATION_LOG,
    DEMO_CRITIC_VERDICTS,
    DEMO_RESULTS,
    DEMO_SYSTEM_STATS,
    DEMO_WORKFLOW_SPEC,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#6C63FF",
    "success": "#00D26A",
    "failure": "#FF4B4B",
    "running": "#00B4D8",
    "pending": "#6B7280",
    "retrying": "#FFA726",
    "bg": "#0E1117",
    "card": "#1A1D23",
}

STATUS_COLORS = {
    "success": COLORS["success"],
    "failure": COLORS["failure"],
    "running": COLORS["running"],
    "pending": COLORS["pending"],
    "retrying": COLORS["retrying"],
    "skipped": "#9CA3AF",
    "needs_human": "#F59E0B",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 1rem 0;">
                <h1 style="color:{COLORS['primary']}; margin:0;">DAAW</h1>
                <p style="color:#9CA3AF; margin:0; font-size:0.85rem;">
                    Compiler-Runtime
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        mode = st.radio(
            "Mode",
            ["Demo Mode", "Live Mode"],
            index=0,
            help="Demo Mode uses pre-loaded data. Live Mode requires an API key.",
        )

        if mode == "Live Mode":
            st.warning("Live Mode requires an LLM API key.", icon="⚡")
            provider = st.selectbox(
                "Provider", ["groq", "gemini", "openai", "anthropic"]
            )
            goal = st.text_area(
                "Goal", placeholder="Describe your workflow goal...", height=100
            )
            compile_btn = st.button("Compile", type="primary", use_container_width=True)

            # Show Execute button only when a spec has been compiled
            has_spec = "live_spec" in st.session_state and st.session_state.live_spec is not None
            has_results = "live_results" in st.session_state and st.session_state.live_results
            execute_btn = st.button(
                "Execute Workflow",
                type="secondary",
                use_container_width=True,
                disabled=not has_spec or has_results,
            )
        else:
            provider = None
            goal = None
            compile_btn = False
            execute_btn = False

        st.divider()
        st.markdown("##### System Stats")
        stats = DEMO_SYSTEM_STATS
        col1, col2 = st.columns(2)
        col1.metric("Agents", stats["agents"]["registered"])
        col2.metric("Tools", stats["tools"]["registered"])
        col1.metric("Providers", len(stats["providers"]["available"]))
        col2.metric("Schemas", stats["schemas"]["count"])

    return mode, provider, goal, compile_btn, execute_btn


# ---------------------------------------------------------------------------
# Tab 1: Architecture Overview
# ---------------------------------------------------------------------------
def render_architecture_tab():
    st.markdown("### System Architecture")
    st.caption("How the DAAW Compiler-Runtime transforms a user goal into executed results.")

    # Architecture flow diagram
    fig = go.Figure()

    # Nodes
    nodes = [
        ("User Goal", 0, 2),
        ("Compiler", 1, 2),
        ("WorkflowSpec", 2, 2),
        ("DAG Executor", 3, 2),
        ("Results", 4, 2),
        ("Critic", 4, 0.8),
        ("Patches", 3, 0.8),
    ]

    node_x = [n[1] for n in nodes]
    node_y = [n[2] for n in nodes]
    node_labels = [n[0] for n in nodes]
    node_colors = [
        COLORS["primary"],  # User Goal
        COLORS["running"],  # Compiler
        COLORS["primary"],  # WorkflowSpec
        COLORS["running"],  # DAG Executor
        COLORS["success"],  # Results
        "#F59E0B",          # Critic
        COLORS["retrying"], # Patches
    ]

    # Edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # main flow
        (4, 5), (5, 6), (6, 3),           # critic feedback loop
    ]

    for src, dst in edges:
        fig.add_trace(go.Scatter(
            x=[node_x[src], node_x[dst]],
            y=[node_y[src], node_y[dst]],
            mode="lines",
            line=dict(color="#4B5563", width=2),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=40, color=node_colors, line=dict(color="#FAFAFA", width=1.5)),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=13, color="#FAFAFA"),
        hoverinfo="text",
        hovertext=[
            "Natural language goal from the user",
            "LLM-powered compilation: goal → DAG spec",
            "Pydantic-validated workflow specification",
            "Async parallel execution engine",
            "Agent outputs with metadata",
            "LLM evaluator: pass/fail per task",
            "DAG mutations: retry, insert, remove",
        ],
        showlegend=False,
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Component cards
    st.markdown("### Core Components")
    cards = [
        ("Schemas", "11 Pydantic models", "WorkflowSpec, TaskSpec, AgentSpec, DependencySpec, AgentResult, TaskResult, TaskStatus, PatchOperation, WorkflowPatch", "src/daaw/schemas/"),
        ("LLM Client", "4 providers", "Groq, Gemini, OpenAI, Anthropic — unified async interface with lazy init", "src/daaw/llm/"),
        ("Agent Registry", "6 agents", "planner, pm, breakdown, critic, user_proxy, generic_llm — DI via AgentFactory", "src/daaw/agents/"),
        ("Engine", "DAG + Executor", "DAG validation (Kahn's), async parallel execution, circuit breaker, context pruning", "src/daaw/engine/"),
        ("Critic", "Evaluate + Patch", "LLM-based evaluation against success criteria, 4 patch actions: retry, insert, remove, update_input", "src/daaw/critic/"),
        ("CLI", "2 commands", "run (full pipeline), legacy (hardcoded workflow) — interactive plan review", "src/daaw/cli/"),
    ]

    cols = st.columns(3)
    for i, (title, subtitle, desc, path) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:{COLORS['card']}; border-radius:8px; padding:1rem; margin-bottom:0.75rem; border-left: 3px solid {COLORS['primary']};">
                    <strong style="color:{COLORS['primary']};">{title}</strong>
                    <span style="color:#9CA3AF; font-size:0.8rem;"> — {subtitle}</span>
                    <p style="color:#D1D5DB; font-size:0.85rem; margin:0.5rem 0 0.25rem;">{desc}</p>
                    <code style="color:#6B7280; font-size:0.75rem;">{path}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 2: Compiler
# ---------------------------------------------------------------------------
def render_compiler_tab(spec):
    st.markdown("### Compilation Pipeline")
    st.caption("Goal → LLM → JSON → Pydantic Validation → WorkflowSpec")

    # Compilation steps
    steps = [
        ("1. Goal Input", "Natural language goal from user or API"),
        ("2. LLM Call", "System prompt with agent roles + tools → JSON response"),
        ("3. JSON Parse", "Parse LLM response as JSON, retry on failure"),
        ("4. Pydantic Validation", "Validate against WorkflowSpec schema"),
        ("5. DAG Validation", "Check for cycles, invalid refs (Kahn's algorithm)"),
    ]

    cols = st.columns(5)
    for i, (step, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div style="background:{COLORS['card']}; border-radius:8px; padding:0.75rem; text-align:center; min-height:120px;">
                    <div style="color:{COLORS['primary']}; font-weight:bold; font-size:0.9rem;">{step}</div>
                    <p style="color:#D1D5DB; font-size:0.78rem; margin-top:0.5rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Task table
    st.markdown("### Compiled Tasks")
    task_rows = []
    for t in spec.tasks:
        deps = ", ".join(d.task_id for d in t.dependencies) or "—"
        task_rows.append({
            "ID": t.id,
            "Name": t.name,
            "Agent": t.agent.role,
            "Dependencies": deps,
            "Criteria": t.success_criteria[:60] + ("..." if len(t.success_criteria) > 60 else ""),
            "Timeout": f"{t.timeout_seconds}s",
        })
    st.dataframe(task_rows, use_container_width=True, hide_index=True)

    # WorkflowSpec JSON
    with st.expander("Full WorkflowSpec JSON"):
        st.json(json.loads(spec.model_dump_json()))

    # Compilation log
    st.markdown("### Compilation Log")
    log_text = "\n".join(DEMO_COMPILATION_LOG)
    st.code(log_text, language="text")


# ---------------------------------------------------------------------------
# Tab 3: DAG Visualization
# ---------------------------------------------------------------------------
def render_dag_tab(spec, results):
    st.markdown("### Workflow DAG")
    st.caption("Interactive graph — hover on nodes for details, scroll to zoom.")

    # Build networkx graph
    G = nx.DiGraph()
    for task in spec.tasks:
        G.add_node(task.id, name=task.name, agent=task.agent.role)
        for dep in task.dependencies:
            G.add_edge(dep.task_id, task.id)

    # Compute topological layers for multipartite layout
    topo_order = list(nx.topological_sort(G))
    layers: dict[str, int] = {}
    for node in topo_order:
        preds = list(G.predecessors(node))
        if not preds:
            layers[node] = 0
        else:
            layers[node] = max(layers[p] for p in preds) + 1

    # Assign positions: x = layer, y = spread within layer
    layer_groups: dict[int, list[str]] = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in layer_groups.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            y = (i - (n - 1) / 2) * 1.5  # spread vertically
            pos[node] = (layer * 2.0, y)

    # Status for each task
    statuses = {}
    for tid in spec.task_ids():
        if tid in results:
            statuses[tid] = results[tid].agent_result.status
        else:
            statuses[tid] = "pending"

    # Edge traces
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#4B5563", width=2),
        hoverinfo="skip",
        showlegend=False,
    )

    # Arrow annotations
    annotations = []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        annotations.append(dict(
            ax=x0, ay=y0,
            x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#4B5563",
            opacity=0.7,
        ))

    # Node trace
    node_x = [pos[tid][0] for tid in spec.task_ids()]
    node_y = [pos[tid][1] for tid in spec.task_ids()]

    node_colors = [STATUS_COLORS.get(statuses[tid], COLORS["pending"]) for tid in spec.task_ids()]

    # Node size proportional to elapsed time
    max_elapsed = max((results[tid].elapsed_seconds for tid in results), default=1.0)
    node_sizes = []
    for tid in spec.task_ids():
        if tid in results:
            size = 25 + (results[tid].elapsed_seconds / max_elapsed) * 30
        else:
            size = 25
        node_sizes.append(size)

    # Hover text
    hover_texts = []
    for task in spec.tasks:
        deps = ", ".join(d.task_id for d in task.dependencies) or "none"
        elapsed = results[task.id].elapsed_seconds if task.id in results else 0
        status = statuses[task.id].upper()
        hover_texts.append(
            f"<b>{task.name}</b> ({task.id})<br>"
            f"Agent: {task.agent.role}<br>"
            f"Status: {status}<br>"
            f"Dependencies: {deps}<br>"
            f"Criteria: {task.success_criteria[:80]}<br>"
            f"Elapsed: {elapsed:.1f}s"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color="#FAFAFA", width=1.5),
        ),
        text=[spec.get_task(tid).name for tid in spec.task_ids()],
        textposition="top center",
        textfont=dict(size=11, color="#FAFAFA"),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=450,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
    )

    # Legend
    for status, color in [("SUCCESS", COLORS["success"]), ("RUNNING", COLORS["running"]),
                           ("PENDING", COLORS["pending"]), ("FAILURE", COLORS["failure"]),
                           ("RETRYING", COLORS["retrying"])]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=status,
            showlegend=True,
        ))

    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
        font=dict(color="#FAFAFA", size=11),
        bgcolor="rgba(0,0,0,0)",
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Task detail panel
    st.markdown("### Task Details")
    selected = st.selectbox(
        "Select a task",
        spec.task_ids(),
        format_func=lambda tid: f"{tid} — {spec.get_task(tid).name}",
    )
    if selected:
        task = spec.get_task(selected)
        result = results.get(selected)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {task.name}")
            st.markdown(f"**Agent:** `{task.agent.role}`")
            st.markdown(f"**Timeout:** {task.timeout_seconds}s")
            st.markdown(f"**Max Retries:** {task.max_retries}")
            deps = ", ".join(f"`{d.task_id}`" for d in task.dependencies) or "None"
            st.markdown(f"**Dependencies:** {deps}")
        with col2:
            st.markdown(f"**Success Criteria:** {task.success_criteria}")
            if result:
                st.markdown(f"**Status:** {result.agent_result.status.upper()}")
                st.markdown(f"**Elapsed:** {result.elapsed_seconds:.1f}s")
                st.markdown(f"**Attempt:** {result.attempt}")
        if result:
            with st.expander("Task Output"):
                st.json(result.agent_result.output)


# ---------------------------------------------------------------------------
# Tab 4: Execution Timeline
# ---------------------------------------------------------------------------
def render_timeline_tab(spec, results):
    st.markdown("### Execution Timeline")
    st.caption("Gantt chart showing parallel task execution.")

    # Compute start/finish times from dependency order and elapsed times
    task_starts: dict[str, float] = {}
    topo = _topo_sort(spec)

    for tid in topo:
        task = spec.get_task(tid)
        dep_ends = []
        for dep in task.dependencies:
            dep_id = dep.task_id
            if dep_id in task_starts and dep_id in results:
                dep_ends.append(task_starts[dep_id] + results[dep_id].elapsed_seconds)
        task_starts[tid] = max(dep_ends) if dep_ends else 0.0

    # Build dataframe for plotly timeline
    base_time = datetime(2025, 6, 15, 10, 0, 0)
    timeline_data = []
    for task in spec.tasks:
        tid = task.id
        result = results.get(tid)
        if not result:
            continue
        start = task_starts.get(tid, 0.0)
        finish = start + result.elapsed_seconds
        timeline_data.append({
            "Task": f"{task.name}",
            "Start": base_time + timedelta(seconds=start),
            "Finish": base_time + timedelta(seconds=finish),
            "Status": result.agent_result.status.upper(),
            "Agent": task.agent.role,
            "Elapsed": f"{result.elapsed_seconds:.1f}s",
        })

    color_map = {
        "SUCCESS": COLORS["success"],
        "FAILURE": COLORS["failure"],
        "RUNNING": COLORS["running"],
        "PENDING": COLORS["pending"],
    }

    if not timeline_data:
        st.info("No execution results available. Run the workflow to see the timeline.")
        return

    df = pd.DataFrame(timeline_data)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Status",
        color_discrete_map=color_map,
        hover_data=["Agent", "Elapsed"],
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        xaxis=dict(
            title="Time",
            gridcolor="#1F2937",
            color="#FAFAFA",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            color="#FAFAFA",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
            font=dict(color="#FAFAFA"),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Execution metrics
    st.markdown("### Execution Metrics")
    total_elapsed = sum(r.elapsed_seconds for r in results.values())
    # Parallel wall-clock time = max finish time
    wall_clock = max(
        task_starts.get(tid, 0) + results[tid].elapsed_seconds
        for tid in results
    )
    success_count = sum(1 for r in results.values() if r.agent_result.status == "success")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wall-Clock Time", f"{wall_clock:.1f}s")
    c2.metric("Parallel Speedup", f"{total_elapsed / wall_clock:.1f}x" if wall_clock > 0 else "—")
    c3.metric("Success Rate", f"{success_count}/{len(results)}")
    c4.metric("Tasks/min", f"{len(results) / (wall_clock / 60):.1f}" if wall_clock > 0 else "—")

    # Per-task breakdown table
    st.markdown("### Per-Task Breakdown")
    rows = []
    for task in spec.tasks:
        result = results.get(task.id)
        if not result:
            continue
        rows.append({
            "Task": task.name,
            "Agent": task.agent.role,
            "Status": result.agent_result.status.upper(),
            "Elapsed": f"{result.elapsed_seconds:.1f}s",
            "Attempt": result.attempt,
            "Start": f"{task_starts.get(task.id, 0):.1f}s",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Critic & Results
# ---------------------------------------------------------------------------
def render_critic_tab(spec, results, verdicts):
    st.markdown("### Critic Evaluation")
    st.caption("LLM-based evaluation of each task against its success criteria.")

    # Summary metrics
    pass_count = sum(1 for v in verdicts if v["verdict"] == "PASS")
    fail_count = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    total = len(verdicts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Passed", pass_count)
    c2.metric("Failed", fail_count)
    c3.metric("Total", total)

    st.markdown("---")

    # Per-task verdict cards
    for verdict in verdicts:
        tid = verdict["task_id"]
        task = spec.get_task(tid)
        result = results.get(tid)
        is_pass = verdict["verdict"] == "PASS"
        badge_color = COLORS["success"] if is_pass else COLORS["failure"]
        badge_text = verdict["verdict"]

        st.markdown(
            f"""
            <div style="background:{COLORS['card']}; border-radius:8px; padding:1rem; margin-bottom:0.75rem;
                        border-left: 3px solid {badge_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong style="color:#FAFAFA;">{tid}: {verdict['task_name']}</strong>
                    <span style="background:{badge_color}; color:#FAFAFA; padding:0.15rem 0.6rem;
                                 border-radius:4px; font-size:0.8rem; font-weight:bold;">{badge_text}</span>
                </div>
                <p style="color:#D1D5DB; font-size:0.85rem; margin:0.5rem 0 0;">{verdict['reasoning']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if verdict.get("patch"):
            st.warning(f"Patch: {verdict['patch']}", icon="🔧")

    # Final outputs
    st.markdown("---")
    st.markdown("### Task Outputs")
    for task in spec.tasks:
        result = results.get(task.id)
        if not result:
            continue
        with st.expander(f"{task.id}: {task.name} — output"):
            st.json(result.agent_result.output)


# ---------------------------------------------------------------------------
# Live Mode: compile, execute, evaluate
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Run an async coroutine from sync context, even if an event loop exists."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_live_infra(provider: str):
    """Shared setup for live mode — returns (config, llm) or None on failure."""
    from daaw.config import get_config
    from daaw.llm.unified import UnifiedLLMClient

    config = get_config()
    llm = UnifiedLLMClient(config)

    available = llm.available_providers()
    if provider not in available:
        st.error(
            f"Provider '{provider}' not available. Set the API key env var. "
            f"Available: {available}"
        )
        return None
    return config, llm


def _patch_user_proxy_for_ui(spec):
    """Replace user_proxy agents with generic_llm so execution doesn't block on input()."""
    for task in spec.tasks:
        if task.agent.role == "user_proxy":
            task.agent = task.agent.model_copy(update={
                "role": "generic_llm",
                "system_prompt_override": (
                    "You are simulating user input for a demo. "
                    "Generate realistic, plausible data for the task described. "
                    "Return a JSON object with the relevant fields."
                ),
            })


def run_live_compile(provider: str, goal: str):
    """Compile a user goal into a WorkflowSpec."""
    try:
        from daaw.compiler.compiler import Compiler

        infra = _get_live_infra(provider)
        if infra is None:
            return None
        config, llm = infra

        compiler = Compiler(llm, config, provider=provider)
        spec = _run_async(compiler.compile(goal))
        return spec
    except Exception as e:
        st.error(f"Compilation failed: {e}")
        return None


def run_live_execute(spec, provider: str):
    """Execute a compiled WorkflowSpec and return (results, verdicts)."""
    try:
        # Import agent registrations so the factory can find them
        import daaw.agents.builtin.breakdown_agent  # noqa: F401
        import daaw.agents.builtin.critic_agent  # noqa: F401
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        import daaw.agents.builtin.planner_agent  # noqa: F401
        import daaw.agents.builtin.pm_agent  # noqa: F401
        import daaw.agents.builtin.user_proxy  # noqa: F401
        import daaw.tools.mock_tools  # noqa: F401
        from daaw.agents.factory import AgentFactory
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.store.artifact_store import ArtifactStore

        infra = _get_live_infra(provider)
        if infra is None:
            return None, None
        config, llm = infra

        # Swap user_proxy → generic_llm so execution doesn't block on input()
        _patch_user_proxy_for_ui(spec)

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        factory = AgentFactory(llm, store)
        executor = DAGExecutor(factory, store, cb)

        # Execute
        results = _run_async(executor.execute(spec))

        # Critique each task
        critic = Critic(llm, config, provider=provider)
        verdicts = []
        for task in spec.tasks:
            if task.id not in results:
                continue
            result = results[task.id]
            passed, patch = _run_async(critic.evaluate(task, result))
            verdicts.append({
                "task_id": task.id,
                "task_name": task.name,
                "verdict": "PASS" if passed else "FAIL",
                "reasoning": patch.reasoning if patch else (
                    "Task passed evaluation against success criteria."
                ),
                "patch": str(patch.operations) if patch else None,
            })

        return results, verdicts
    except Exception as e:
        st.error(f"Execution failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _topo_sort(spec) -> list[str]:
    """Topological sort of task IDs from the spec."""
    G = nx.DiGraph()
    for task in spec.tasks:
        G.add_node(task.id)
        for dep in task.dependencies:
            G.add_edge(dep.task_id, task.id)
    return list(nx.topological_sort(G))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mode, provider, goal, compile_btn, execute_btn = render_sidebar()

    # --- Session state for Live Mode persistence across reruns ---
    if "live_spec" not in st.session_state:
        st.session_state.live_spec = None
    if "live_results" not in st.session_state:
        st.session_state.live_results = {}
    if "live_verdicts" not in st.session_state:
        st.session_state.live_verdicts = []

    # Handle Live Mode actions
    if mode == "Live Mode":
        if compile_btn and goal:
            with st.spinner("Compiling workflow..."):
                live_spec = run_live_compile(provider, goal)
            if live_spec:
                st.session_state.live_spec = live_spec
                st.session_state.live_results = {}
                st.session_state.live_verdicts = []
                st.success("Compilation successful! Click **Execute Workflow** to run it.")
            else:
                st.session_state.live_spec = None

        if execute_btn and st.session_state.live_spec is not None:
            with st.spinner("Executing workflow & running critic..."):
                results, verdicts = run_live_execute(
                    st.session_state.live_spec, provider
                )
            if results is not None:
                st.session_state.live_results = results
                st.session_state.live_verdicts = verdicts or []
                st.success("Execution complete!")

    # Determine active data
    if mode == "Live Mode" and st.session_state.live_spec is not None:
        spec = st.session_state.live_spec
        results = st.session_state.live_results
        verdicts = st.session_state.live_verdicts
    else:
        spec = DEMO_WORKFLOW_SPEC
        results = DEMO_RESULTS
        verdicts = DEMO_CRITIC_VERDICTS

    # Title
    st.markdown(
        f"""
        <h2 style="color:{COLORS['primary']}; margin-bottom:0;">
            DAAW Compiler-Runtime
        </h2>
        <p style="color:#9CA3AF; margin-top:0;">
            Directed Acyclic Agent Workflow — Compile, Execute, Evaluate
        </p>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Demo Mode":
        st.info(
            f"**Demo Mode** — Showing: *{spec.name}* (pre-loaded data, no API key needed)",
            icon="🎯",
        )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Architecture",
        "Compiler",
        "DAG Visualization",
        "Execution Timeline",
        "Critic & Results",
    ])

    with tab1:
        render_architecture_tab()
    with tab2:
        render_compiler_tab(spec)
    with tab3:
        render_dag_tab(spec, results)
    with tab4:
        render_timeline_tab(spec, results)
    with tab5:
        render_critic_tab(spec, results, verdicts)


if __name__ == "__main__":
    main()
