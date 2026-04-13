"""DAAW Compiler-Runtime — Production-Ready Dashboard UI."""

from __future__ import annotations

import asyncio
import html
import io
import json
import queue
import re
import threading
import time
import traceback
import zipfile
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

# Re-read .env on every Streamlit rerun so config picks up changes
# without needing a full server restart.
from daaw.config import reset_config as _reset_config
_reset_config()

# Register real_tools at startup (real DuckDuckGo HTTP search).
# Falls back to mock_tools if real_tools import fails.
try:
    import daaw.tools.real_tools  # noqa: F401
except Exception:
    try:
        import daaw.tools.mock_tools  # noqa: F401
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
C = {
    "primary": "#6C63FF",
    "primary_dim": "#4B44CC",
    "accent": "#A78BFA",
    "success": "#00D26A",
    "success_dim": "#00A854",
    "failure": "#FF4B4B",
    "failure_dim": "#CC3C3C",
    "running": "#00B4D8",
    "pending": "#6B7280",
    "retrying": "#FFA726",
    "skipped": "#9CA3AF",
    "human": "#F59E0B",
    "bg": "#0E1117",
    "card": "#161B22",
    "card_hover": "#1C2333",
    "surface": "#21262D",
    "border": "#30363D",
    "text": "#E6EDF3",
    "text_dim": "#8B949E",
    "text_muted": "#6E7681",
}

STATUS_COLORS = {
    "success": C["success"],
    "failure": C["failure"],
    "running": C["running"],
    "pending": C["pending"],
    "retrying": C["retrying"],
    "skipped": C["skipped"],
    "needs_human": C["human"],
}

STATUS_ICONS = {
    "success": "&#10004;",
    "failure": "&#10008;",
    "running": "&#9679;",
    "pending": "&#9675;",
    "retrying": "&#8635;",
    "skipped": "&#8212;",
    "needs_human": "&#9888;",
}

# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------
_GLOBAL_CSS = f"""
<style>
    .card {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }}
    .card:hover {{
        border-color: {C['primary']};
    }}
    .card-accent {{
        border-left: 3px solid {C['primary']};
    }}
    .card-success {{
        border-left: 3px solid {C['success']};
    }}
    .card-failure {{
        border-left: 3px solid {C['failure']};
    }}
    .badge {{
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}
    .badge-success {{
        background: {C['success_dim']};
        color: #FFF;
    }}
    .badge-failure {{
        background: {C['failure_dim']};
        color: #FFF;
    }}
    .badge-primary {{
        background: {C['primary_dim']};
        color: #FFF;
    }}
    .badge-muted {{
        background: {C['surface']};
        color: {C['text_dim']};
    }}
    .section-title {{
        color: {C['text']};
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}
    .dim {{
        color: {C['text_dim']};
        font-size: 0.82rem;
    }}
    .mono {{
        font-family: 'SFMono-Regular', Consolas, monospace;
        font-size: 0.82rem;
        color: {C['accent']};
    }}
    .flow-step {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
        min-height: 110px;
    }}
    .flow-step-active {{
        border-color: {C['primary']};
        box-shadow: 0 0 8px {C['primary']}33;
    }}
    .tool-card {{
        background: {C['surface']};
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-left: 2px solid {C['running']};
    }}
    .verdict-card {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }}
    .stat-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {C['text']};
        line-height: 1;
    }}
    .stat-label {{
        font-size: 0.75rem;
        color: {C['text_dim']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }}
    .log-line {{
        font-family: 'SFMono-Regular', Consolas, monospace;
        font-size: 0.8rem;
        line-height: 1.6;
        color: {C['text_dim']};
    }}
    .log-timestamp {{
        color: {C['primary']};
    }}
    .pipeline-badge {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        vertical-align: middle;
    }}
    .task-card {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }}
    .task-card:hover {{
        border-color: {C['primary']};
    }}
    .tool-pill {{
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'SFMono-Regular', Consolas, monospace;
        background: {C['surface']};
        color: {C['running']};
        margin-right: 0.3rem;
        margin-bottom: 0.2rem;
    }}
    .dep-pill {{
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'SFMono-Regular', Consolas, monospace;
        background: {C['surface']};
        color: {C['accent']};
        margin-right: 0.3rem;
    }}
    .error-box {{
        background: #1a0000;
        border: 1px solid {C['failure']};
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-top: 0.5rem;
        font-family: 'SFMono-Regular', Consolas, monospace;
        font-size: 0.8rem;
        color: {C['failure']};
    }}
    .sidebar-section {{
        background: {C['card']};
        border: 1px solid {C['border']};
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }}
</style>
"""


def _card(content: str, border_color: str | None = None, extra_class: str = ""):
    cls = f"card {extra_class}"
    style = f"border-left: 3px solid {border_color};" if border_color else ""
    return f'<div class="{cls}" style="{style}">{content}</div>'


def _badge(text: str, variant: str = "primary"):
    return f'<span class="badge badge-{variant}">{_esc(text)}</span>'


def _stat_card(value: str, label: str, color: str = C["text"]):
    return f"""
    <div class="card" style="text-align:center; padding:1rem;">
        <div class="stat-value" style="color:{color};">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def _esc(text: str) -> str:
    return html.escape(str(text))


def _safe_filename(name: str) -> str:
    """Sanitize a task id into a filesystem-safe filename stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return stem or "task"


def _render_gateway_probe_badge(model: str) -> None:
    """Show a cached compatibility badge under the gateway model selector.

    First render triggers a single lightweight JSON probe; subsequent reruns
    read from the in-memory cache so we don't hammer the gateway.
    """
    from daaw.config import get_config
    from daaw.llm.model_probe import get_cached_probe, probe_model as _probe

    gateway_url = get_config().gateway_url or "http://127.0.0.1:11434/v1"
    cached = get_cached_probe(gateway_url, model)

    # Track per-model "probe started" flag so Streamlit rerun loops don't
    # retrigger in-flight work.
    key = f"probe_inflight_{gateway_url}_{model}"
    if cached is None:
        if not st.session_state.get(key):
            st.session_state[key] = True
            with st.spinner(f"Probing {model} for JSON compatibility..."):
                try:
                    asyncio.new_event_loop().run_until_complete(
                        _probe(gateway_url, model)
                    )
                except Exception:
                    pass
            st.session_state[key] = False
            cached = get_cached_probe(gateway_url, model)

    if cached is None:
        st.caption("Model compatibility: (probe failed to run)")
        return

    if cached.is_usable:
        st.caption(
            f"Model probe: :green[OK] — valid JSON in {cached.elapsed_seconds}s"
        )
    else:
        st.warning(
            f"**Model probe: {cached.badge}** — this model failed to emit "
            f"valid JSON in the compatibility test. The compiler will likely "
            f"fail. Try `gemma4:e2b-it-q4_K_M` (recommended for local) or a "
            f"cloud provider.\n\n"
            f"Probe response preview: `{cached.preview[:100]}`",
            icon="⚠",
        )


def _build_outputs_zip(spec, results) -> bytes:
    """Build a ZIP containing one JSON per task output plus a combined file.

    Each per-task file contains the raw output plus metadata (status, elapsed,
    agent role, tools_allowed, tool_calls). A trailing ``all_outputs.json``
    bundles every task keyed by task id.
    """
    buffer = io.BytesIO()
    combined: dict[str, dict] = {}
    task_by_id = {t.id: t for t in getattr(spec, "tasks", [])}
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for tid, r in results.items():
            task = task_by_id.get(tid)
            agent_role = getattr(getattr(task, "agent", None), "role", None)
            tools_allowed = list(getattr(getattr(task, "agent", None), "tools_allowed", []) or [])
            metadata = r.agent_result.metadata or {}
            payload = {
                "task_id": tid,
                "task_name": getattr(task, "name", tid),
                "agent_role": agent_role,
                "tools_allowed": tools_allowed,
                "status": r.agent_result.status,
                "elapsed_seconds": r.elapsed_seconds,
                "output": r.agent_result.output,
                "error_message": getattr(r.agent_result, "error_message", None),
                "tool_calls": metadata.get("tool_calls", []),
                "usage": metadata.get("usage"),
            }
            combined[tid] = payload
            fname = f"{_safe_filename(tid)}.json"
            zf.writestr(fname, json.dumps(payload, indent=2, default=str))
        zf.writestr("all_outputs.json", json.dumps(combined, indent=2, default=str))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding:1.25rem 0 0.5rem;">
                <div style="font-size:2.2rem; font-weight:800; color:{C['primary']};
                            letter-spacing:-0.02em; line-height:1;">DAAW</div>
                <div style="color:{C['text_dim']}; font-size:0.8rem; margin-top:0.25rem;
                            letter-spacing:0.1em; text-transform:uppercase;">Compiler-Runtime</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        mode = st.radio(
            "Mode",
            ["Demo Mode", "Live Mode"],
            index=0,
            help="Demo uses pre-loaded data. Live requires an API key.",
            horizontal=True,
        )

        use_mock_tools = True
        compile_btn = False
        execute_btn = False
        provider = None
        model = None
        goal = None

        if mode == "Live Mode":
            with st.expander("Live Mode Settings", expanded=True):
                provider = st.selectbox(
                    "LLM Provider",
                    ["groq", "gemini", "openai", "anthropic", "gateway"],
                    help="'gateway' uses any OpenAI-compatible endpoint (LiteLLM, Ollama, vLLM). Others require API key env vars.",
                )

                _PROVIDER_MODELS = {
                    "groq": [
                        "llama-3.3-70b-versatile",           # strong all-rounder (default)
                        "meta-llama/llama-4-scout-17b-16e-instruct",  # multimodal MoE
                        "qwen/qwen3-32b",                    # reasoning
                        "llama-3.1-8b-instant",              # fastest/cheapest
                        "moonshotai/kimi-k2-instruct-0905",  # best tool use (MoE 1T)
                    ],
                    "gemini": [
                        "gemini-2.5-flash",                  # fast/cheap (default)
                        "gemini-2.5-flash-lite",             # cheapest
                        "gemini-2.5-pro",                    # most capable stable
                        "gemini-3-flash-preview",            # next-gen fast (preview)
                        "gemini-3.1-pro-preview",            # next-gen best (preview)
                    ],
                    "openai": [
                        "gpt-4.1-mini",                      # fast/cheap (default)
                        "gpt-4.1-nano",                      # cheapest
                        "gpt-4.1",                           # capable
                        "o4-mini",                           # reasoning fast
                        "gpt-5.4-mini",                      # latest fast
                        "gpt-5.4-nano",                      # latest cheapest
                        "gpt-5.4",                           # latest flagship
                        "o3-mini",                           # reasoning
                    ],
                    "anthropic": [
                        "claude-sonnet-4-6",                 # best balance (default)
                        "claude-haiku-4-5-20251001",         # fastest/cheapest
                        "claude-sonnet-4-5",                 # previous gen
                        "claude-opus-4-6",                   # most capable
                    ],
                    "gateway": [
                        # e2b is ~100% reliable on the planner JSON workload,
                        # e4b ~25%. Ranked by empirical success rate from
                        # scripts/probe_local_models.py.
                        "gemma4:e2b-it-q4_K_M",          # Ollama Gemma E2B (recommended)
                        "gemma4:e4b",                    # Ollama Gemma E4B (flaky JSON output)
                        "default",                       # use GATEWAY_MODEL env var
                    ],
                }
                model_options = _PROVIDER_MODELS.get(provider, ["default"])
                model = st.selectbox(
                    "Model",
                    model_options,
                    help=f"Model to use with {provider}. First option is the default.",
                )

                # Probe gateway models: warn if they can't produce valid JSON.
                if provider == "gateway":
                    _render_gateway_probe_badge(model)

                goal = st.text_area(
                    "Goal",
                    placeholder="Describe your workflow goal...",
                    height=100,
                    help="Natural language description of the workflow to compile.",
                )
                use_mock_tools = st.checkbox(
                    "Use mock tools",
                    value=False,
                    help="Use fake tool responses instead of real execution.",
                )
                hitl_enabled = st.checkbox(
                    "Enable human-in-the-loop prompts",
                    value=True,
                    help=(
                        "When enabled, user_proxy / PM agents pause the workflow "
                        "and ask you questions here in the UI. Disable to auto-fill "
                        "with demo defaults."
                    ),
                )
                st.session_state["hitl_enabled"] = hitl_enabled

                compile_btn = st.button(
                    "Compile Workflow",
                    type="primary",
                    use_container_width=True,
                )

                has_spec = "live_spec" in st.session_state and st.session_state.live_spec is not None
                execute_btn = st.button(
                    "Execute Workflow",
                    type="secondary",
                    use_container_width=True,
                    disabled=not has_spec,
                )

                # Danger-styled reset button
                if has_spec:
                    st.divider()
                    st.markdown(
                        f'<div style="font-size:0.72rem; color:{C["text_muted"]}; margin-bottom:0.25rem;">'
                        f'Pipeline loaded &mdash; {len(st.session_state.live_spec.tasks)} tasks</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Reset Pipeline", use_container_width=True, key="danger_reset"):
                        st.session_state.live_spec = None
                        st.session_state.live_results = {}
                        st.session_state.live_verdicts = []
                        st.session_state.live_log = []
                        st.session_state.pipeline_stage = "idle"
                        st.rerun()

        st.divider()

        # System stats — live if possible
        stats = _get_live_system_stats()
        st.markdown(
            f"""
            <div style="padding:0 0.25rem;">
                <div style="color:{C['text_dim']}; font-size:0.72rem; text-transform:uppercase;
                            letter-spacing:0.08em; margin-bottom:0.5rem;">System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Agents", stats["agents"])
        c2.metric("Tools", stats["tools"])
        c1.metric("Providers", stats["providers"])
        c2.metric("Schemas", stats["schemas"])

        # Webhook notification status indicator
        try:
            from daaw.integrations.notifications import is_available as _notif_avail
            notif_ok = _notif_avail()
        except Exception:
            notif_ok = False
        st.markdown(
            f'<div style="font-size:0.72rem; color:{C["text_muted"]}; margin-top:0.5rem;">'
            f'Notifications: <span style="color:{C["success"] if notif_ok else C["failure"]};font-weight:600;">'
            f'{"configured" if notif_ok else "not configured"}</span></div>',
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.divider()
        from daaw.__version__ import BUILD_TAG as _BT, __version__ as _VER
        st.markdown(
            f'<div style="font-family:monospace;font-size:.7rem;'
            f'color:{C["text_muted"]};text-align:center">'
            f'DAAW v<strong style="color:{C["accent"]}">{_VER}</strong>'
            f' · <span style="color:{C["human"]}">{_BT}</span></div>',
            unsafe_allow_html=True,
        )

    return mode, provider, model, goal, compile_btn, execute_btn, use_mock_tools


def _get_live_system_stats() -> dict:
    try:
        from daaw.agents.registry import AGENT_REGISTRY
        from daaw.tools.registry import tool_registry
        agents = len(AGENT_REGISTRY)
        tools = len(tool_registry._tools)
    except Exception:
        agents = DEMO_SYSTEM_STATS["agents"]["registered"]
        tools = DEMO_SYSTEM_STATS["tools"]["registered"]
    try:
        from daaw.config import get_config as _gc
        from daaw.llm.unified import UnifiedLLMClient as _ULLM
        providers = len(_ULLM(_gc()).available_providers())
    except Exception:
        providers = 4
    return {
        "agents": agents,
        "tools": tools,
        "providers": providers,
        "schemas": 11,
    }


# ---------------------------------------------------------------------------
# Tab 1: Architecture Overview
# ---------------------------------------------------------------------------
def render_architecture_tab():
    st.markdown('<div class="section-title">System Architecture</div>', unsafe_allow_html=True)
    st.caption("How DAAW transforms a fuzzy goal into deterministic parallel execution.")

    # --- Flow diagram ---
    fig = go.Figure()

    nodes = [
        ("User Goal", 0, 2, C["accent"]),
        ("Compiler", 1, 2, C["running"]),
        ("WorkflowSpec", 2, 2, C["primary"]),
        ("DAG Executor", 3, 2, C["running"]),
        ("Results", 4, 2, C["success"]),
        ("Critic", 4, 0.7, C["human"]),
        ("Patches", 3, 0.7, C["retrying"]),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 3),
    ]
    edge_labels = {
        (0, 1): "natural language",
        (1, 2): "JSON DAG",
        (2, 3): "execute",
        (3, 4): "outputs",
        (4, 5): "evaluate",
        (5, 6): "patch ops",
        (6, 3): "retry/insert",
    }

    for src, dst in edges:
        x0, y0 = nodes[src][1], nodes[src][2]
        x1, y1 = nodes[dst][1], nodes[dst][2]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        label = edge_labels.get((src, dst), "")
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=C["border"], width=2),
            hoverinfo="skip", showlegend=False,
        ))
        if label:
            fig.add_annotation(
                x=mx, y=my, text=f"<i>{label}</i>",
                showarrow=False, font=dict(size=9, color=C["text_muted"]),
                bgcolor=C["bg"], borderpad=2,
            )

    fig.add_trace(go.Scatter(
        x=[n[1] for n in nodes],
        y=[n[2] for n in nodes],
        mode="markers+text",
        marker=dict(
            size=45,
            color=[n[3] for n in nodes],
            line=dict(color=C["text"], width=1.5),
        ),
        text=[n[0] for n in nodes],
        textposition="top center",
        textfont=dict(size=12, color=C["text"]),
        hoverinfo="text",
        hovertext=[
            "Natural language goal from the user",
            "LLM compiles goal into a strict JSON execution graph",
            "Pydantic-validated DAG: tasks, deps, agents, tools",
            "Async parallel executor with circuit breaker",
            "Agent outputs stored in artifact store",
            "LLM evaluator checks outputs against success criteria",
            "DAG mutations: retry, insert, remove, update_input",
        ],
        showlegend=False,
    ))

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 3]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Layer architecture ---
    st.markdown('<div class="section-title">Layer Architecture</div>', unsafe_allow_html=True)
    layers = [
        ("L0: Foundation", "LLM providers + Tool registry", C["text_muted"], "src/daaw/llm/, tools/"),
        ("L1: Schema", "Pydantic v2 data contracts", C["text_muted"], "src/daaw/schemas/"),
        ("L2: State", "Async artifact store", C["text_muted"], "src/daaw/store/"),
        ("L3: Logic", "Agent factory + DAG executor", C["primary"], "src/daaw/agents/, engine/"),
        ("L4: Intelligence", "Compiler + Critic", C["accent"], "src/daaw/compiler/, critic/"),
        ("L5: Interface", "CLI + Streamlit UI", C["running"], "src/daaw/cli/, ui/"),
    ]

    cols = st.columns(6)
    for i, (name, desc, color, path) in enumerate(layers):
        with cols[i]:
            st.markdown(
                f"""<div class="card" style="text-align:center; min-height:145px;
                    border-top: 3px solid {color};">
                    <div style="color:{color}; font-weight:700; font-size:0.82rem;">{name}</div>
                    <div class="dim" style="margin:0.4rem 0;">{desc}</div>
                    <div class="mono" style="font-size:0.7rem;">{path}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- Design patterns ---
    st.markdown('<div class="section-title">Design Patterns</div>', unsafe_allow_html=True)
    patterns = [
        ("Factory", "AgentFactory creates agents from AgentSpec with DI"),
        ("Registry", "@register_agent / @tool_registry.register decorators"),
        ("Strategy", "LLM providers implement common interface"),
        ("Circuit Breaker", "Per-task failure tracking, trips after threshold"),
        ("Observer/Patch", "Critic generates WorkflowPatch ops on live DAG"),
        ("Human-in-Loop", "UserProxy pauses execution for human input"),
    ]
    cols = st.columns(3)
    for i, (name, desc) in enumerate(patterns):
        with cols[i % 3]:
            st.markdown(
                _card(
                    f'<strong style="color:{C["accent"]};">{name}</strong>'
                    f'<div class="dim" style="margin-top:0.3rem;">{desc}</div>',
                    border_color=C["primary"],
                ),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 2: Compiler
# ---------------------------------------------------------------------------
def render_compiler_tab(spec, log_lines: list[str]):
    st.markdown('<div class="section-title">Compilation Pipeline</div>', unsafe_allow_html=True)
    st.caption("Goal -> LLM -> JSON -> Pydantic -> DAG Validation -> WorkflowSpec")

    steps = [
        ("1. Goal Input", "Natural language from user", False),
        ("2. LLM Call", "System prompt + agent roles + tools", True),
        ("3. JSON Parse", "Parse response, retry on failure", True),
        ("4. Pydantic", "Validate against WorkflowSpec", False),
        ("5. DAG Check", "Cycle detection (Kahn's)", False),
    ]

    cols = st.columns(5)
    for i, (step, desc, is_active) in enumerate(steps):
        cls = "flow-step flow-step-active" if is_active else "flow-step"
        with cols[i]:
            st.markdown(
                f"""<div class="{cls}">
                    <div style="color:{C['primary']}; font-weight:700; font-size:0.85rem;">{step}</div>
                    <div class="dim" style="margin-top:0.4rem;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    # Connecting arrows between steps
    st.markdown(
        f'<div style="text-align:center; color:{C["text_muted"]}; font-size:0.8rem; '
        f'margin:-0.5rem 0 1rem;">{"&nbsp;&nbsp;&rarr;&nbsp;&nbsp;" * 4}</div>',
        unsafe_allow_html=True,
    )

    # --- Workflow header ---
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(
            f'<div style="font-size:1.1rem; font-weight:600; color:{C["text"]};">'
            f'{_esc(spec.name)}</div>'
            f'<div class="dim">{_esc(spec.description)}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _stat_card(str(len(spec.tasks)), "Tasks", C["primary"]),
            unsafe_allow_html=True,
        )
    with col3:
        deps_count = sum(len(t.dependencies) for t in spec.tasks)
        st.markdown(
            _stat_card(str(deps_count), "Dependencies", C["accent"]),
            unsafe_allow_html=True,
        )

    # --- Task cards ---
    st.markdown("---")
    st.markdown('<div class="section-title">Tasks</div>', unsafe_allow_html=True)
    for t in spec.tasks:
        deps = [d.task_id for d in t.dependencies]
        tools = t.agent.tools_allowed
        criteria = t.success_criteria
        if len(criteria) > 120:
            criteria = criteria[:117] + "..."

        deps_html = (
            "".join(f'<span class="dep-pill">{_esc(d)}</span>' for d in deps)
            if deps
            else f'<span class="dim">none</span>'
        )
        tools_html = (
            "".join(f'<span class="tool-pill">{_esc(tool)}</span>' for tool in tools)
            if tools
            else f'<span class="dim">none</span>'
        )

        st.markdown(
            f"""<div class="task-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="mono">{_esc(t.id)}</span>
                        <strong style="color:{C['text']}; margin-left:0.5rem;">{_esc(t.name)}</strong>
                    </div>
                    <div>
                        {_badge(t.agent.role, "primary")}
                        <span class="badge badge-muted" style="margin-left:0.3rem;">{t.timeout_seconds}s</span>
                    </div>
                </div>
                <div style="margin-top:0.4rem;">
                    <span class="dim">Tools:</span> {tools_html}
                    <span class="dim" style="margin-left:0.75rem;">Deps:</span> {deps_html}
                </div>
                <div class="dim" style="margin-top:0.3rem;">{_esc(criteria)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # --- WorkflowSpec JSON ---
    col_j, col_l = st.columns(2)
    with col_j:
        with st.expander("WorkflowSpec JSON", expanded=False):
            st.json(json.loads(spec.model_dump_json()))
    with col_l:
        with st.expander("Compilation Log", expanded=False):
            log_html = ""
            for line in log_lines:
                # Highlight timestamp
                if line.startswith("["):
                    bracket_end = line.index("]") + 1
                    ts = line[:bracket_end]
                    rest = line[bracket_end:]
                    log_html += f'<div class="log-line"><span class="log-timestamp">{_esc(ts)}</span>{_esc(rest)}</div>'
                else:
                    log_html += f'<div class="log-line">{_esc(line)}</div>'
            st.markdown(log_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3: DAG Visualization
# ---------------------------------------------------------------------------
def render_dag_tab(spec, results):
    st.markdown('<div class="section-title">Workflow DAG</div>', unsafe_allow_html=True)
    st.caption("Interactive graph — hover for details, node size reflects execution time.")

    G = nx.DiGraph()
    for task in spec.tasks:
        G.add_node(task.id, name=task.name, agent=task.agent.role)
        for dep in task.dependencies:
            G.add_edge(dep.task_id, task.id)

    # Topological layering
    topo_order = list(nx.topological_sort(G))
    layers: dict[str, int] = {}
    for node in topo_order:
        preds = list(G.predecessors(node))
        layers[node] = (max(layers[p] for p in preds) + 1) if preds else 0

    layer_groups: dict[int, list[str]] = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes_in_layer in layer_groups.items():
        n = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            y = (i - (n - 1) / 2) * 1.8
            pos[node] = (layer * 2.5, y)

    statuses = {}
    for tid in spec.task_ids():
        statuses[tid] = results[tid].agent_result.status if tid in results else "pending"

    # Edge traces with curved paths
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color=C["border"], width=2),
        hoverinfo="skip",
        showlegend=False,
    )

    annotations = []
    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        annotations.append(dict(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowwidth=2, arrowcolor=C["border"], opacity=0.7,
        ))

    # Node trace
    max_elapsed = max((results[tid].elapsed_seconds for tid in results), default=1.0) or 1.0
    node_x, node_y, node_sizes, node_colors, hover_texts = [], [], [], [], []

    for tid in spec.task_ids():
        task = spec.get_task(tid)
        x, y = pos[tid]
        node_x.append(x)
        node_y.append(y)

        status = statuses[tid]
        node_colors.append(STATUS_COLORS.get(status, C["pending"]))

        if tid in results:
            size = 28 + (results[tid].elapsed_seconds / max_elapsed) * 28
        else:
            size = 28
        node_sizes.append(size)

        elapsed = results[tid].elapsed_seconds if tid in results else 0
        deps = ", ".join(d.task_id for d in task.dependencies) or "none"
        tools = ", ".join(task.agent.tools_allowed) or "none"
        tool_count = len((results[tid].agent_result.metadata or {}).get("tool_calls", [])) if tid in results else 0
        hover_texts.append(
            f"<b>{_esc(task.name)}</b> ({_esc(tid)})<br>"
            f"Agent: {_esc(task.agent.role)}<br>"
            f"Status: {STATUS_ICONS.get(status, '')} {_esc(status.upper())}<br>"
            f"Dependencies: {_esc(deps)}<br>"
            f"Tools allowed: {_esc(tools)}<br>"
            f"Tool calls made: {tool_count}<br>"
            f"Elapsed: {elapsed:.1f}s"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color=C["text"], width=1.5),
            opacity=0.9,
        ),
        text=[spec.get_task(tid).name for tid in spec.task_ids()],
        textposition="top center",
        textfont=dict(size=11, color=C["text"]),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # Status legend
    for status_name, color in [("SUCCESS", C["success"]), ("PENDING", C["pending"]),
                                ("FAILURE", C["failure"]), ("RUNNING", C["running"]),
                                ("RETRYING", C["retrying"])]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=status_name, showlegend=True,
        ))

    fig.update_layout(
        height=480,
        margin=dict(l=30, r=30, t=20, b=30),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=annotations,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5,
            font=dict(color=C["text"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Task detail panel ---
    st.markdown('<div class="section-title">Task Inspector</div>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Select task",
        spec.task_ids(),
        format_func=lambda tid: f"{tid} -- {spec.get_task(tid).name}",
    )
    if selected:
        _render_task_detail(spec, results, selected)


def _render_task_detail(spec, results, task_id: str):
    task = spec.get_task(task_id)
    result = results.get(task_id)
    status = result.agent_result.status if result else "pending"
    status_color = STATUS_COLORS.get(status, C["pending"])

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(
            _card(
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<strong style="color:{C["text"]};">{_esc(task.name)}</strong>'
                f'{_badge(status.upper(), "success" if status == "success" else "failure" if status == "failure" else "muted")}'
                f'</div>'
                f'<div class="dim" style="margin-top:0.5rem;">{_esc(task.description)}</div>',
                border_color=status_color,
            ),
            unsafe_allow_html=True,
        )
    with col2:
        deps = ", ".join(f"<code>{d.task_id}</code>" for d in task.dependencies) or "None"
        tools = ", ".join(f"<code>{t}</code>" for t in task.agent.tools_allowed) or "None"
        st.markdown(
            _card(
                f'<div class="dim"><strong>Agent:</strong> <code>{task.agent.role}</code></div>'
                f'<div class="dim"><strong>Dependencies:</strong> {deps}</div>'
                f'<div class="dim"><strong>Tools:</strong> {tools}</div>'
                f'<div class="dim"><strong>Timeout:</strong> {task.timeout_seconds}s &nbsp;|&nbsp; '
                f'<strong>Max Retries:</strong> {task.max_retries}</div>',
            ),
            unsafe_allow_html=True,
        )
    with col3:
        if result:
            elapsed = result.elapsed_seconds
            attempt = result.attempt
            st.markdown(
                _stat_card(f"{elapsed:.1f}s", "Elapsed", status_color),
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:center;" class="dim">Attempt {attempt}</div>',
                unsafe_allow_html=True,
            )

    if not result:
        st.info("Task has not been executed yet.")
        return

    # Success criteria
    if task.success_criteria:
        st.markdown(
            f'<div class="dim" style="margin-bottom:0.5rem;">'
            f'<strong>Success Criteria:</strong> {_esc(task.success_criteria)}</div>',
            unsafe_allow_html=True,
        )

    # Output + Tool calls side by side
    col_out, col_tools = st.columns([3, 2])
    with col_out:
        err = getattr(result.agent_result, "error_message", "") or ""
        if result.agent_result.status != "success" and err:
            with st.expander("Error", expanded=True):
                st.error(err)
        output = result.agent_result.output
        if output is not None or result.agent_result.status == "success":
            with st.expander("Task Output", expanded=True):
                if isinstance(output, (dict, list)):
                    st.json(output)
                else:
                    st.code(str(output), language="text")

    with col_tools:
        tool_calls = (result.agent_result.metadata or {}).get("tool_calls", [])
        with st.expander(f"Tool Calls ({len(tool_calls)})", expanded=bool(tool_calls)):
            if not tool_calls:
                st.markdown(f'<div class="dim">No tool calls recorded.</div>', unsafe_allow_html=True)
            else:
                for i, tc in enumerate(tool_calls, 1):
                    tool_name = tc.get("tool", "unknown")
                    args = tc.get("args", {})
                    res = str(tc.get("result", ""))
                    if len(res) > 250:
                        res = res[:247] + "..."
                    st.markdown(
                        f"""<div class="tool-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <strong style="color:{C['running']};">#{i} {_esc(tool_name)}</strong>
                                {_badge(tool_name, "primary")}
                            </div>
                            <div class="dim" style="margin-top:0.3rem;">
                                Args: <code>{_esc(json.dumps(args))}</code>
                            </div>
                            <div style="color:{C['text_dim']}; font-size:0.8rem; margin-top:0.3rem;
                                        background:{C['bg']}; padding:0.4rem 0.5rem; border-radius:4px;
                                        font-family:monospace;">
                                {_esc(res)}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Tab 4: Execution Timeline
# ---------------------------------------------------------------------------
def render_timeline_tab(spec, results):
    st.markdown('<div class="section-title">Execution Timeline</div>', unsafe_allow_html=True)
    st.caption("Gantt chart showing parallel task execution and performance metrics.")

    if not results:
        st.info("No execution results available. Compile and execute a workflow to see the timeline.")
        return

    task_starts = _compute_task_starts(spec, results)

    # --- Metrics row ---
    total_elapsed = sum(r.elapsed_seconds for r in results.values())
    wall_clock = max(
        task_starts.get(tid, 0) + results[tid].elapsed_seconds
        for tid in results
    )
    success_count = sum(1 for r in results.values() if r.agent_result.status == "success")
    fail_count = sum(1 for r in results.values() if r.agent_result.status == "failure")
    total_tool_calls = sum(
        len((r.agent_result.metadata or {}).get("tool_calls", []))
        for r in results.values()
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(_stat_card(f"{wall_clock:.1f}s", "Wall Clock", C["primary"]), unsafe_allow_html=True)
    speedup = f"{total_elapsed / wall_clock:.1f}x" if wall_clock > 0 else "---"
    c2.markdown(_stat_card(speedup, "Parallel Speedup", C["accent"]), unsafe_allow_html=True)
    c3.markdown(_stat_card(f"{success_count}/{len(results)}", "Success Rate", C["success"]), unsafe_allow_html=True)
    c4.markdown(_stat_card(str(fail_count), "Failures", C["failure"] if fail_count else C["text_dim"]), unsafe_allow_html=True)
    c5.markdown(_stat_card(str(total_tool_calls), "Tool Calls", C["running"]), unsafe_allow_html=True)

    st.markdown("---")

    # --- Gantt chart ---
    base_time = datetime(2025, 6, 15, 10, 0, 0)
    timeline_data = []
    for task in spec.tasks:
        tid = task.id
        result = results.get(tid)
        if not result:
            continue
        start = task_starts.get(tid, 0.0)
        finish = start + result.elapsed_seconds
        tool_count = len((result.agent_result.metadata or {}).get("tool_calls", []))
        timeline_data.append({
            "Task": task.name,
            "Start": base_time + timedelta(seconds=start),
            "Finish": base_time + timedelta(seconds=finish),
            "Status": result.agent_result.status.upper(),
            "Agent": task.agent.role,
            "Elapsed": f"{result.elapsed_seconds:.1f}s",
            "Tools": tool_count,
        })

    color_map = {
        "SUCCESS": C["success"],
        "FAILURE": C["failure"],
        "RUNNING": C["running"],
        "PENDING": C["pending"],
    }

    df = pd.DataFrame(timeline_data)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Status",
        color_discrete_map=color_map,
        hover_data=["Agent", "Elapsed", "Tools"],
    )
    fig.update_layout(
        height=max(250, len(timeline_data) * 50),
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        xaxis=dict(title="", gridcolor="#1F2937", color=C["text"]),
        yaxis=dict(title="", autorange="reversed", color=C["text"]),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
            font=dict(color=C["text"]), bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(color=C["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Per-task table ---
    st.markdown('<div class="section-title">Per-Task Breakdown</div>', unsafe_allow_html=True)
    rows = []
    for task in spec.tasks:
        result = results.get(task.id)
        if not result:
            continue
        tool_count = len((result.agent_result.metadata or {}).get("tool_calls", []))
        rows.append({
            "Task": task.name,
            "Agent": task.agent.role,
            "Status": result.agent_result.status.upper(),
            "Start": f"{task_starts.get(task.id, 0):.1f}s",
            "Elapsed": f"{result.elapsed_seconds:.1f}s",
            "Tool Calls": tool_count,
            "Attempt": result.attempt,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Critic & Results
# ---------------------------------------------------------------------------
def render_critic_tab(spec, results, verdicts):
    st.markdown('<div class="section-title">Critic Evaluation</div>', unsafe_allow_html=True)
    st.caption("LLM-based evaluation of each task output against its success criteria.")

    if not verdicts:
        st.info("No critic verdicts available. Execute the workflow first.")
        return

    # --- Summary ---
    pass_count = sum(1 for v in verdicts if v["verdict"] == "PASS")
    fail_count = sum(1 for v in verdicts if v["verdict"] == "FAIL")
    total = len(verdicts)
    pass_rate = f"{pass_count / total * 100:.0f}%" if total else "---"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_stat_card(str(pass_count), "Passed", C["success"]), unsafe_allow_html=True)
    c2.markdown(_stat_card(str(fail_count), "Failed", C["failure"] if fail_count else C["text_dim"]), unsafe_allow_html=True)
    c3.markdown(_stat_card(str(total), "Total", C["primary"]), unsafe_allow_html=True)
    c4.markdown(_stat_card(pass_rate, "Pass Rate", C["success"] if pass_count == total else C["human"]), unsafe_allow_html=True)

    st.markdown("---")

    # --- Verdict cards ---
    for verdict in verdicts:
        tid = verdict["task_id"]
        is_pass = verdict["verdict"] == "PASS"
        # Strip any stray HTML tags the LLM may have included in reasoning
        import re as _re
        verdict = dict(verdict)
        verdict["reasoning"] = _re.sub(r"<[^>]+>", "", verdict.get("reasoning", "")).strip()
        border = C["success"] if is_pass else C["failure"]
        badge_variant = "success" if is_pass else "failure"

        patch_html = ""
        if verdict.get("patch"):
            patch_html = (
                f'<div style="margin-top:0.5rem; padding:0.5rem; background:{C["surface"]}; '
                f'border-radius:6px; border-left:2px solid {C["retrying"]};">'
                f'<div class="dim"><strong style="color:{C["retrying"]};">Patch:</strong> '
                f'{_esc(str(verdict["patch"]))}</div></div>'
            )

        card_html = (
            f'<div class="verdict-card" style="border-left:3px solid {border};">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<div>'
            f'<strong style="color:{C["text"]};">{_esc(tid)}</strong>'
            f'<span class="dim">&nbsp;&mdash;&nbsp;{_esc(verdict["task_name"])}</span>'
            f'</div>'
            f'{_badge(verdict["verdict"], badge_variant)}'
            f'</div>'
            f'<div class="dim" style="margin-top:0.5rem;">{_esc(verdict["reasoning"])}</div>'
            f'{patch_html}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        # Show error details for failed tasks
        if not is_pass:
            result = results.get(tid)
            if result and result.agent_result.status == "failure":
                error_msg = result.agent_result.error_message
                output = result.agent_result.output
                if not error_msg and isinstance(output, dict) and "error" in output:
                    error_msg = str(output["error"])
                if error_msg:
                    st.markdown(
                        f'<div class="error-box" style="margin-top:-0.5rem; margin-bottom:0.75rem;">'
                        f'<strong>Error:</strong> {_esc(error_msg)}</div>',
                        unsafe_allow_html=True,
                    )

    # --- Task outputs ---
    st.markdown("---")
    st.markdown('<div class="section-title">Task Outputs</div>', unsafe_allow_html=True)
    for task in spec.tasks:
        result = results.get(task.id)
        if not result:
            continue
        status = result.agent_result.status
        icon = STATUS_ICONS.get(status, "")
        with st.expander(f"{icon} {task.id}: {task.name}"):
            err = getattr(result.agent_result, "error_message", "") or ""
            if status != "success" and err:
                st.error(err)
            output = result.agent_result.output
            if output is not None or status == "success":
                if isinstance(output, (dict, list)):
                    st.json(output)
                else:
                    st.code(str(output), language="text")


# ---------------------------------------------------------------------------
# Tab 6: Tools
# ---------------------------------------------------------------------------
def render_tools_tab(spec, results):
    st.markdown('<div class="section-title">Tool Registry & Usage</div>', unsafe_allow_html=True)
    st.caption("Registered tools and their usage across the workflow.")

    # --- Registered tools ---
    try:
        from daaw.tools.registry import tool_registry
        registered = tool_registry._tools
    except Exception:
        registered = {}

    if registered:
        st.markdown("**Registered Tools**")
        cols = st.columns(min(len(registered), 4))
        for i, (name, tool_def) in enumerate(registered.items()):
            with cols[i % len(cols)]:
                params = tool_def.parameters.get("properties", {})
                param_list = ", ".join(
                    f'{k}: {v.get("type", "any")}' for k, v in params.items()
                )
                required = tool_def.parameters.get("required", [])
                st.markdown(
                    _card(
                        f'<strong style="color:{C["running"]};">{_esc(name)}</strong>'
                        f'<div class="dim" style="margin:0.3rem 0;">{_esc(tool_def.description)}</div>'
                        f'<div class="mono" style="font-size:0.72rem;">({_esc(param_list)})</div>'
                        f'<div class="dim" style="margin-top:0.2rem;">Required: {", ".join(required) or "none"}</div>',
                        border_color=C["running"],
                    ),
                    unsafe_allow_html=True,
                )
    else:
        st.info("No tools registered. Import real_tools or mock_tools to see available tools.")

    st.markdown("---")

    # --- Tool usage across tasks ---
    st.markdown("**Tool Usage by Task**")
    if not results:
        st.info("Execute a workflow to see tool usage statistics.")
        return

    usage_data = []
    for task in spec.tasks:
        result = results.get(task.id)
        if not result:
            continue
        tool_calls = (result.agent_result.metadata or {}).get("tool_calls", [])
        for tc in tool_calls:
            usage_data.append({
                "Task": task.name,
                "Tool": tc.get("tool", "unknown"),
                "Arguments": json.dumps(tc.get("args", {})),
                "Result Preview": str(tc.get("result", ""))[:120],
            })

    if usage_data:
        st.dataframe(usage_data, use_container_width=True, hide_index=True)

        # Tool call distribution chart
        tool_counts = {}
        for d in usage_data:
            tool_counts[d["Tool"]] = tool_counts.get(d["Tool"], 0) + 1

        fig = go.Figure(data=[go.Bar(
            x=list(tool_counts.keys()),
            y=list(tool_counts.values()),
            marker_color=C["running"],
            text=list(tool_counts.values()),
            textposition="auto",
        )])
        fig.update_layout(
            title=dict(text="Tool Call Distribution", font=dict(color=C["text"], size=14)),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=C["bg"],
            plot_bgcolor=C["bg"],
            xaxis=dict(color=C["text"], gridcolor=C["border"]),
            yaxis=dict(color=C["text"], gridcolor=C["border"], title="Calls"),
            font=dict(color=C["text"]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tool calls were made during execution.")


# ---------------------------------------------------------------------------
# Live Mode: compile, execute, evaluate
# ---------------------------------------------------------------------------
def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_live_infra(provider: str):
    from daaw.config import get_config
    from daaw.llm.unified import UnifiedLLMClient

    config = get_config()
    llm = UnifiedLLMClient(config)

    available = llm.available_providers()
    if provider not in available:
        hint = "Set GATEWAY_URL env var." if provider == "gateway" else f"Set the {provider.upper()}_API_KEY env var."
        st.error(
            f"Provider '{provider}' not available. {hint} "
            f"Available: {available}"
        )
        return None
    return config, llm


def _patch_user_proxy_for_ui(spec):
    """Replace user_proxy tasks with pre-filled auto-answers.

    Legacy behaviour retained for users who opt out of interactive HITL
    (sidebar toggle). When HITL mode is enabled, leave user_proxy tasks
    intact so the UI can prompt the user live.
    """
    for task in spec.tasks:
        if task.agent.role == "user_proxy":
            task.agent = task.agent.model_copy(update={
                "role": "generic_llm",
                "system_prompt_override": (
                    "You are auto-filling user parameters for a demo. "
                    "Based on the task description, return ONLY a JSON object "
                    "with concrete values (dates, locations, budget numbers). "
                    "Do NOT generate a prompt or ask questions. Example: "
                    '{"travelers":2,"dates":"2025-07-01 to 2025-07-07",'
                    '"locations":["Tokyo","Kyoto"],"budget":"mid-range"}'
                ),
            })


def run_live_compile(provider: str, goal: str, model: str | None = None) -> tuple:
    log = []
    try:
        from daaw.compiler.compiler import Compiler
        import time

        t0 = time.monotonic()
        log.append(f"[{0:.2f}s] Compiler initialized -- provider: {provider}, model: {model or 'default'}")

        infra = _get_live_infra(provider)
        if infra is None:
            return None, log
        config, llm = infra

        log.append(f"[{time.monotonic() - t0:.2f}s] Sending goal to LLM...")
        compiler = Compiler(llm, config, provider=provider, model=model)
        # Stream tokens into a live code block so the compile phase is
        # visible instead of a silent 30-second spinner.
        stream_header = st.empty()
        stream_body = st.empty()
        stream_header.caption("Streaming compiler output...")

        from daaw.ui._streaming_display import prettify_partial_json

        def _on_compile_token(_delta: str, full: str) -> None:
            rendered, lang = prettify_partial_json(full)
            stream_body.code(rendered, language=lang)

        try:
            spec = _run_async(
                compiler.compile_stream(goal, on_token=_on_compile_token)
            )
        finally:
            stream_header.empty()
            stream_body.empty()
        log.append(f"[{time.monotonic() - t0:.2f}s] LLM response received")
        log.append(f"[{time.monotonic() - t0:.2f}s] WorkflowSpec validated -- {len(spec.tasks)} tasks")

        from daaw.engine.dag import DAG
        dag = DAG(spec)
        errors = dag.validate()
        if errors:
            log.append(f"[{time.monotonic() - t0:.2f}s] DAG validation FAILED: {errors}")
        else:
            log.append(f"[{time.monotonic() - t0:.2f}s] DAG validated -- no cycles detected")

        log.append(f"[{time.monotonic() - t0:.2f}s] Compilation complete")
        return spec, log
    except Exception as e:
        log.append(f"[ERROR] Compilation failed: {e}")
        st.error(f"Compilation failed: {e}")
        return None, log


def _send_completion_notification(spec, results) -> None:
    """Best-effort webhook notification on workflow completion."""
    try:
        import asyncio as _asyncio
        from daaw.integrations.notifications import notify_workflow_complete as _notify
        _success = sum(1 for r in results.values() if r.agent_result.status == "success")
        _elapsed = sum(r.elapsed_seconds for r in results.values())
        _loop = _asyncio.new_event_loop()
        _sent = _loop.run_until_complete(_notify(
            workflow_name=spec.name,
            total=len(results),
            passed=_success,
            failed=len(results) - _success,
            elapsed=_elapsed,
        ))
        _loop.close()
        if _sent:
            st.toast("Webhook notification sent", icon="bell")
    except Exception:
        pass


def _import_builtin_agents_and_tools(use_mock: bool) -> None:
    import daaw.agents.builtin.breakdown_agent  # noqa: F401
    import daaw.agents.builtin.critic_agent  # noqa: F401
    import daaw.agents.builtin.generic_llm_agent  # noqa: F401
    import daaw.agents.builtin.planner_agent  # noqa: F401
    import daaw.agents.builtin.pm_agent  # noqa: F401
    import daaw.agents.builtin.user_proxy  # noqa: F401

    if use_mock:
        import daaw.tools.mock_tools  # noqa: F401
    else:
        try:
            import daaw.tools.real_tools  # noqa: F401
        except ImportError:
            import daaw.tools.mock_tools  # noqa: F401


def run_live_execute(spec, provider: str, use_mock: bool = False, model: str | None = None):
    """Synchronous (no-HITL) execution path. Kept for the auto-fill demo mode."""
    try:
        _import_builtin_agents_and_tools(use_mock)

        from daaw.agents.factory import AgentFactory
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.store.artifact_store import ArtifactStore

        infra = _get_live_infra(provider)
        if infra is None:
            return None, None
        config, llm = infra

        _patch_user_proxy_for_ui(spec)

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        factory = AgentFactory(llm, store, default_provider=provider, default_model=model)
        max_conc = 1 if provider == "gateway" else None
        executor = DAGExecutor(factory, store, cb, max_concurrent=max_conc)

        results = _run_async(executor.execute(spec))

        critic = Critic(llm, config, provider=provider, model=model)
        verdicts = []
        for task in spec.tasks:
            if task.id not in results:
                continue
            result = results[task.id]
            try:
                passed, patch, reasoning = _run_async(critic.evaluate(task, result))
            except Exception:
                passed, patch, reasoning = False, None, "Critic crashed (model OOM/unavailable)"
            verdicts.append({
                "task_id": task.id,
                "task_name": task.name,
                "verdict": "PASS" if passed else "FAIL",
                "reasoning": reasoning,
                "patch": str(patch.operations) if patch else None,
            })

        return results, verdicts
    except Exception as e:
        st.error(f"Execution failed: {e}")
        st.code(traceback.format_exc(), language="text")
        return None, None


# ---------------------------------------------------------------------------
# HITL (human-in-the-loop) execution via background thread + queue bridge
# ---------------------------------------------------------------------------

def start_live_execute_hitl(
    spec,
    provider: str,
    use_mock: bool = False,
    model: str | None = None,
) -> dict | None:
    """Spin up a background thread running the full compile→execute→critic pipeline
    with a ``QueueInteractionHandler`` bridging any ``user_proxy`` prompts back
    to Streamlit via two ``queue.Queue`` objects.

    Returns a handle (dict) that should be stashed in ``st.session_state`` and
    driven by :func:`poll_live_execute_hitl` on each rerun.
    """
    try:
        _import_builtin_agents_and_tools(use_mock)

        from daaw.agents.factory import AgentFactory
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.interaction import QueueInteractionHandler
        from daaw.store.artifact_store import ArtifactStore

        infra = _get_live_infra(provider)
        if infra is None:
            return None
        config, llm = infra

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        questions: "queue.Queue" = queue.Queue()
        answers: "queue.Queue" = queue.Queue()
        handler = QueueInteractionHandler(questions, answers, timeout=1800.0)

        factory = AgentFactory(
            llm, store,
            default_provider=provider,
            default_model=model,
            interaction_handler=handler,
        )
        max_conc = 1 if provider == "gateway" else None
        executor = DAGExecutor(factory, store, cb, max_concurrent=max_conc)

        holder: dict = {
            "results": None,
            "verdicts": None,
            "error": None,
        }

        def _worker():
            try:
                loop = asyncio.new_event_loop()
                try:
                    results = loop.run_until_complete(executor.execute(spec))
                    critic = Critic(llm, config, provider=provider, model=model)
                    verdicts = []
                    for task in spec.tasks:
                        if task.id not in results:
                            continue
                        res = results[task.id]
                        try:
                            passed, patch, reasoning = loop.run_until_complete(
                                critic.evaluate(task, res)
                            )
                        except Exception:
                            passed, patch, reasoning = False, None, (
                                "Critic crashed (model OOM/unavailable)"
                            )
                        verdicts.append({
                            "task_id": task.id,
                            "task_name": task.name,
                            "verdict": "PASS" if passed else "FAIL",
                            "reasoning": reasoning,
                            "patch": str(patch.operations) if patch else None,
                        })
                    holder["results"] = results
                    holder["verdicts"] = verdicts
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                holder["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, name="daaw-live-exec", daemon=True)
        thread.start()

        return {
            "thread": thread,
            "questions": questions,
            "answers": answers,
            "holder": holder,
            "pending": None,
            "started_at": time.monotonic(),
        }
    except Exception as e:
        st.error(f"Execution failed to start: {e}")
        st.code(traceback.format_exc(), language="text")
        return None


def poll_live_execute_hitl(handle: dict) -> str:
    """Drive one poll cycle of the HITL executor.

    Returns one of:
      - ``"pending_question"`` — a prompt is waiting; caller should render it.
      - ``"running"`` — executor is still busy; caller should schedule a rerun.
      - ``"done"`` — executor finished; results/verdicts in ``holder``.
      - ``"error"`` — executor crashed; details in ``holder['error']``.
    """
    # If we already have a cached pending question, surface it again.
    if handle.get("pending") is not None:
        return "pending_question"

    # Try to drain the next question (non-blocking).
    try:
        req = handle["questions"].get_nowait()
        handle["pending"] = req
        return "pending_question"
    except queue.Empty:
        pass

    if handle["holder"].get("error"):
        return "error"

    if not handle["thread"].is_alive():
        # Thread finished; give any last-moment question a chance to land.
        try:
            req = handle["questions"].get_nowait()
            handle["pending"] = req
            return "pending_question"
        except queue.Empty:
            pass
        return "done"

    return "running"


def answer_live_execute_hitl(handle: dict, answer: str) -> None:
    """Feed a user answer back to the executor and clear the pending slot."""
    handle["answers"].put(str(answer))
    handle["pending"] = None


def render_hitl_prompt(handle: dict) -> bool:
    """Render a Streamlit form for the pending question. Returns True on submit."""
    req = handle.get("pending")
    if req is None:
        return False

    prompt_text = getattr(req, "prompt", str(req))
    hint = getattr(req, "hint", None)
    choices = getattr(req, "choices", None)
    agent_id = getattr(req, "agent_id", "agent")
    step_id = getattr(req, "step_id", None) or "prompt"
    ctx = getattr(req, "context", None) or {}

    st.markdown(
        f'<div style="background:{C["card"]}; border:1px solid {C["border"]}; '
        f'border-left:4px solid {C["human"]}; border-radius:8px; '
        f'padding:1rem 1.2rem; margin:0.5rem 0;">'
        f'<strong style="color:{C["human"]};">HUMAN INPUT REQUESTED</strong>'
        f'<div class="dim" style="margin-top:0.2rem; font-size:0.8rem;">'
        f'agent: <code>{_esc(agent_id)}</code> · step: <code>{_esc(step_id)}</code>'
        + (
            f' · Q {ctx.get("step")}/{ctx.get("total")}'
            if ctx.get("step") and ctx.get("total") else ""
        )
        + '</div></div>',
        unsafe_allow_html=True,
    )

    form_key = f"hitl_form_{id(req)}"
    with st.form(form_key, clear_on_submit=True):
        st.markdown(f"**{_esc(prompt_text)}**")
        if hint:
            st.caption(hint)
        if choices:
            value = st.radio(
                "Choose or type your own answer below:",
                options=choices,
                key=f"{form_key}_radio",
                horizontal=True,
            )
            freeform = st.text_area("Your answer (overrides choice if non-empty):",
                                    key=f"{form_key}_free", height=100)
        else:
            value = None
            freeform = st.text_area("Your answer:", key=f"{form_key}_free", height=120)

        submitted = st.form_submit_button("Submit answer", use_container_width=True)

    if submitted:
        answer = (freeform or "").strip() or (value or "")
        answer_live_execute_hitl(handle, answer)
        return True
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _topo_sort(spec) -> list[str]:
    G = nx.DiGraph()
    for task in spec.tasks:
        G.add_node(task.id)
        for dep in task.dependencies:
            G.add_edge(dep.task_id, task.id)
    return list(nx.topological_sort(G))


def _compute_task_starts(spec, results) -> dict[str, float]:
    topo = _topo_sort(spec)
    task_starts: dict[str, float] = {}
    for tid in topo:
        task = spec.get_task(tid)
        dep_ends = []
        for dep in task.dependencies:
            dep_id = dep.task_id
            if dep_id in task_starts and dep_id in results:
                dep_ends.append(task_starts[dep_id] + results[dep_id].elapsed_seconds)
        task_starts[tid] = max(dep_ends) if dep_ends else 0.0
    return task_starts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    mode, provider, model, goal, compile_btn, execute_btn, use_mock_tools = render_sidebar()

    # --- Session state ---
    if "live_spec" not in st.session_state:
        st.session_state.live_spec = None
    if "live_results" not in st.session_state:
        st.session_state.live_results = {}
    if "live_verdicts" not in st.session_state:
        st.session_state.live_verdicts = []
    if "live_log" not in st.session_state:
        st.session_state.live_log = []
    if "compile_success_msg" not in st.session_state:
        st.session_state.compile_success_msg = None
    if "pipeline_stage" not in st.session_state:
        st.session_state.pipeline_stage = "idle"

    # --- Show deferred compile success message (after st.rerun) ---
    if st.session_state.get("compile_success_msg"):
        st.success(st.session_state.compile_success_msg)
        st.session_state.compile_success_msg = None

    # --- Handle Live Mode actions ---
    if mode == "Live Mode":
        if compile_btn and goal:
            st.session_state.pipeline_stage = "compiling"
            with st.status("Compiling workflow...", expanded=True) as status:
                status.write("Initializing compiler...")
                live_spec, log = run_live_compile(provider, goal, model)
                st.session_state.live_log = log
                if live_spec:
                    status.write(f"Validated {len(live_spec.tasks)} tasks")
                    status.write("DAG validation passed")
                    status.update(label=f"Compiled {len(live_spec.tasks)} tasks", state="complete")
                else:
                    status.update(label="Compilation failed", state="error")
            if live_spec:
                st.session_state.live_spec = live_spec
                st.session_state.live_results = {}
                st.session_state.live_verdicts = []
                st.session_state.pipeline_stage = "compiled"
                st.session_state.compile_success_msg = (
                    f"Compiled {len(live_spec.tasks)} tasks. Click **Execute Workflow** to run."
                )
                st.rerun()  # Re-render so sidebar evaluates has_spec=True -> execute button enabled
            else:
                st.session_state.live_spec = None
                st.session_state.pipeline_stage = "idle"
                st.session_state.compile_success_msg = None

        hitl_enabled = st.session_state.get("hitl_enabled", True)

        # ── Kick off execution ────────────────────────────────────────────
        if execute_btn and st.session_state.live_spec is not None:
            if hitl_enabled:
                # Launch background HITL executor; the poller below drives it.
                handle = start_live_execute_hitl(
                    st.session_state.live_spec, provider, use_mock_tools, model,
                )
                if handle is not None:
                    st.session_state["live_exec_handle"] = handle
                    st.session_state.pipeline_stage = "executing"
                    st.rerun()
                else:
                    st.session_state.pipeline_stage = "compiled"
            else:
                # Legacy synchronous path: auto-fill user_proxy tasks.
                st.session_state.pipeline_stage = "executing"
                with st.status("Executing workflow...", expanded=True) as status:
                    status.write("Instantiating agents...")
                    status.write("Executing tasks in DAG order...")
                    results, verdicts = run_live_execute(
                        st.session_state.live_spec, provider, use_mock_tools, model
                    )
                    if results is not None:
                        success_count = sum(1 for r in results.values() if r.agent_result.status == "success")
                        status.write(f"Executed {len(results)} tasks ({success_count} succeeded)")
                        status.write("Critic evaluation complete")
                        status.update(
                            label=f"Done: {success_count}/{len(results)} tasks succeeded",
                            state="complete",
                        )
                    else:
                        status.update(label="Execution failed", state="error")
                if results is not None:
                    st.session_state.live_results = results
                    st.session_state.live_verdicts = verdicts or []
                    st.session_state.pipeline_stage = "done"
                    _send_completion_notification(st.session_state.live_spec, results)
                else:
                    st.session_state.pipeline_stage = "compiled"

        # ── HITL poller: runs on every rerun while a handle is active ────
        handle = st.session_state.get("live_exec_handle")
        if handle is not None:
            state = poll_live_execute_hitl(handle)
            if state == "pending_question":
                st.markdown("### Workflow paused — your input is needed")
                submitted = render_hitl_prompt(handle)
                if submitted:
                    # Give the worker a tick to pick up the answer, then rerun.
                    time.sleep(0.05)
                    st.rerun()
                else:
                    st.info(
                        "The executor is waiting for your reply. Answer above and "
                        "click **Submit answer** to continue.",
                        icon="⏸",
                    )
            elif state == "running":
                elapsed = int(time.monotonic() - handle["started_at"])
                st.info(f"Executing workflow... ({elapsed}s elapsed)", icon="⚙")
                time.sleep(1.0)
                st.rerun()
            elif state == "error":
                st.error("Execution crashed")
                st.code(handle["holder"]["error"], language="text")
                st.session_state.pipeline_stage = "compiled"
                st.session_state["live_exec_handle"] = None
            elif state == "done":
                results = handle["holder"].get("results") or {}
                verdicts = handle["holder"].get("verdicts") or []
                st.session_state.live_results = results
                st.session_state.live_verdicts = verdicts
                st.session_state.pipeline_stage = "done"
                st.session_state["live_exec_handle"] = None
                if results:
                    _send_completion_notification(st.session_state.live_spec, results)
                st.rerun()

    # --- Determine active data ---
    if mode == "Live Mode" and st.session_state.live_spec is not None:
        spec = st.session_state.live_spec
        results = st.session_state.live_results
        verdicts = st.session_state.live_verdicts
        log_lines = st.session_state.live_log
    else:
        spec = DEMO_WORKFLOW_SPEC
        results = DEMO_RESULTS
        verdicts = DEMO_CRITIC_VERDICTS
        log_lines = DEMO_COMPILATION_LOG

    # --- Header with pipeline stage badge ---
    stage = st.session_state.get("pipeline_stage", "idle")
    stage_colors = {
        "idle": C["text_muted"],
        "compiling": C["running"],
        "compiled": C["accent"],
        "executing": C["running"],
        "done": C["success"],
    }
    badge_color = stage_colors.get(stage, C["text_muted"])
    stage_label = stage if mode == "Live Mode" else "demo"
    badge_bg = C["primary_dim"] if mode == "Demo Mode" else badge_color

    st.markdown(
        f"""<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.25rem;">
            <h2 style="color:{C['primary']}; margin:0;">DAAW Compiler-Runtime</h2>
            <span class="dim">Dynamic Agentic AI Workflow</span>
            <span class="pipeline-badge" style="background:{badge_bg}; color:#fff;">{_esc(stage_label)}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- Mode banners ---
    if mode == "Demo Mode":
        st.markdown(
            f'<div style="background:{C["card"]}; border:1px solid {C["border"]}; border-left:3px solid {C["accent"]}; '
            f'border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.75rem;">'
            f'<strong style="color:{C["accent"]};">Demo Mode</strong> '
            f'<span class="dim">Showing pre-loaded data for <em>{_esc(spec.name)}</em>. '
            f'Switch to <strong>Live Mode</strong> in the sidebar to compile and execute your own workflows.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.live_spec and st.session_state.live_results:
        success = sum(1 for r in results.values() if r.agent_result.status == "success")
        total = len(results)
        color = C["success"] if success == total else C["human"]
        st.markdown(
            f'<div style="background:{C["card"]}; border:1px solid {C["border"]}; border-radius:8px; '
            f'padding:0.6rem 1rem; margin-bottom:0.5rem;">'
            f'<strong style="color:{color};">Live Results:</strong> '
            f'<span class="dim">{_esc(spec.name)} &mdash; {success}/{total} tasks passed</span></div>',
            unsafe_allow_html=True,
        )

    # --- Export row ---
    st.markdown("---")
    col_e1, col_e2, col_e3, col_e4 = st.columns([2, 2, 2, 1])
    with col_e1:
        spec_json = spec.model_dump_json(indent=2)
        st.download_button(
            "Download WorkflowSpec JSON",
            data=spec_json,
            file_name="workflow_spec.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e2:
        if results:
            results_export = {
                tid: {
                    "status": r.agent_result.status,
                    "output": r.agent_result.output,
                    "elapsed": r.elapsed_seconds,
                }
                for tid, r in results.items()
            }
            st.download_button(
                "Download Results JSON",
                data=json.dumps(results_export, indent=2, default=str),
                file_name="results.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Execute workflow to export results.")
    with col_e3:
        if results:
            zip_bytes = _build_outputs_zip(spec, results)
            st.download_button(
                "Download Output JSONs (ZIP)",
                data=zip_bytes,
                file_name="task_outputs.zip",
                mime="application/zip",
                use_container_width=True,
                help="One JSON file per task output, plus a combined all_outputs.json.",
            )
        else:
            st.caption("No outputs yet.")
    with col_e4:
        if verdicts:
            verdicts_json = json.dumps(verdicts, indent=2, default=str)
            st.download_button(
                "Download Verdicts",
                data=verdicts_json,
                file_name="verdicts.json",
                mime="application/json",
                use_container_width=True,
            )

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Architecture",
        "Compiler",
        "DAG Visualization",
        "Execution Timeline",
        "Critic & Results",
        "Tools",
    ])

    with tab1:
        render_architecture_tab()
    with tab2:
        render_compiler_tab(spec, log_lines)
    with tab3:
        render_dag_tab(spec, results)
    with tab4:
        render_timeline_tab(spec, results)
    with tab5:
        render_critic_tab(spec, results, verdicts)
    with tab6:
        render_tools_tab(spec, results)


if __name__ == "__main__":
    main()
