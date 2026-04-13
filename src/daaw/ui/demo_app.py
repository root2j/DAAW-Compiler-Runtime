"""DAAW — Under the Hood: Pipeline Demonstration UI.

A presentation-grade Streamlit dashboard that visualizes the DAAW
Compiler-Runtime pipeline step by step, with a chat interface,
live logs, DAG visualization, and performance metrics.

Launch: python -m daaw demo
"""
from __future__ import annotations

import asyncio
import html as html_lib
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

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAAW — Under the Hood",
    page_icon="🔬",
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
from daaw.config import reset_config as _reset_config
_reset_config()

try:
    import daaw.tools.real_tools  # noqa: F401
except Exception:
    try:
        import daaw.tools.mock_tools  # noqa: F401
    except Exception:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Design Tokens — "Deep Space" theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D = {
    "bg_deep": "#04060b",
    "bg": "#0a0e17",
    "bg_card": "#111827",
    "bg_surface": "#1e2535",
    "bg_input": "#0d1117",
    "border": "#1f2937",
    "border_active": "#374151",
    "border_glow": "#3b82f644",
    "blue": "#3b82f6",
    "blue_dim": "#2563eb",
    "blue_glow": "#3b82f622",
    "violet": "#8b5cf6",
    "cyan": "#06b6d4",
    "green": "#10b981",
    "green_dim": "#059669",
    "amber": "#f59e0b",
    "amber_dim": "#d97706",
    "red": "#ef4444",
    "red_dim": "#dc2626",
    "pink": "#ec4899",
    "text": "#f1f5f9",
    "text_dim": "#94a3b8",
    "text_muted": "#64748b",
    "stage_goal": "#8b5cf6",
    "stage_compile": "#3b82f6",
    "stage_execute": "#06b6d4",
    "stage_critique": "#f59e0b",
    "stage_summary": "#10b981",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
/* ── Streamlit overrides ── */
.stApp {{ background-color: {D['bg']} !important; color: {D['text']}; }}
section[data-testid="stSidebar"] {{ background-color: {D['bg_deep']} !important; border-right: 1px solid {D['border']}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0; background: {D['bg_card']}; border-radius: 10px; padding: 3px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px; font-family: 'DM Sans', sans-serif; font-size: 0.82rem; font-weight: 500; padding: 0.4rem 1rem; color: {D['text_dim']}; }}
.stTabs [aria-selected="true"] {{ background: {D['bg_surface']} !important; color: {D['text']} !important; }}

/* ── Pipeline stepper ── */
.stepper-wrap {{ display:flex; align-items:center; justify-content:center; padding:1.15rem 1.5rem; background:{D['bg_card']}; border:1px solid {D['border']}; border-radius:14px; margin-bottom:1.25rem; gap:0; flex-wrap:wrap; }}
.stepper-step {{ display:flex; align-items:center; gap:0.55rem; padding:0.5rem 1rem; border-radius:10px; font-family:'DM Sans',sans-serif; font-size:0.85rem; font-weight:600; color:{D['text_muted']}; transition:all .3s ease; }}
.stepper-step.active {{ color:{D['text']}; }}
.stepper-step.completed {{ color:{D['text_dim']}; }}
.stepper-step.pending {{ opacity:0.35; }}
.stepper-num {{ width:26px; height:26px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; font-family:'JetBrains Mono',monospace; border:2px solid {D['border']}; color:{D['text_muted']}; transition:all .3s ease; }}
.stepper-step.active .stepper-num {{ border-color:var(--sc); color:var(--sc); background:color-mix(in srgb, var(--sc) 12%, transparent); }}
.stepper-step.completed .stepper-num {{ border-color:{D['green']}; color:{D['green']}; background:{D['green']}11; }}
.stepper-conn {{ width:32px; height:2px; background:{D['border']}; margin:0 .15rem; border-radius:1px; }}
.stepper-conn.done {{ background:{D['green']}88; }}
.stepper-conn.on {{ background:linear-gradient(90deg, var(--pc), var(--nc)); }}

/* ── Chat ── */
.chat-wrap {{ background:{D['bg_deep']}; border:1px solid {D['border']}; border-radius:14px; padding:1rem 1.25rem; max-height:640px; overflow-y:auto; scroll-behavior:smooth; }}
.chat-msg {{ display:flex; gap:0.65rem; margin-bottom:1rem; animation:msgIn .35s ease-out; }}
.chat-msg.usr {{ flex-direction:row-reverse; }}
.chat-av {{ width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; margin-top:.2rem; }}
.chat-body {{ max-width:85%; display:flex; flex-direction:column; }}
.chat-msg.usr .chat-body {{ align-items:flex-end; }}
.chat-lbl {{ font-family:'DM Sans',sans-serif; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem; padding-left:0.1rem; }}
.chat-bub {{ padding:0.75rem 1rem; border-radius:14px; font-family:'DM Sans',sans-serif; font-size:0.88rem; line-height:1.65; color:{D['text']}; }}
.chat-bub.user {{ background:linear-gradient(135deg,{D['blue_dim']},{D['blue']}); border-bottom-right-radius:4px; color:#fff; }}
.chat-bub.compiler {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-bottom-left-radius:4px; border-left:3px solid {D['stage_compile']}; }}
.chat-bub.engine {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-bottom-left-radius:4px; border-left:3px solid {D['stage_execute']}; }}
.chat-bub.critic {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-bottom-left-radius:4px; border-left:3px solid {D['stage_critique']}; }}
.chat-bub.summary {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-bottom-left-radius:4px; border-left:3px solid {D['stage_summary']}; }}
.chat-bub code {{ background:{D['bg_surface']}; padding:0.1rem 0.35rem; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:{D['cyan']}; }}
.chat-bub strong {{ color:{D['text']}; font-weight:600; }}
.st-run {{ display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; background:{D['cyan']}22; color:{D['cyan']}; }}
.st-ok {{ display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; background:{D['green']}22; color:{D['green']}; }}
.st-fail {{ display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.75rem; font-weight:600; font-family:'JetBrains Mono',monospace; background:{D['red']}22; color:{D['red']}; }}
.st-pass {{ display:inline-block; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.75rem; font-weight:700; font-family:'JetBrains Mono',monospace; background:{D['green']}22; color:{D['green']}; }}

/* ── Log console ── */
.log-con {{ background:{D['bg_deep']}; border:1px solid {D['border']}; border-radius:14px; padding:1rem; font-family:'JetBrains Mono',monospace; font-size:0.78rem; max-height:520px; overflow-y:auto; line-height:1.9; scroll-behavior:smooth; }}
.log-con::before {{ content:'PIPELINE LOG'; display:block; font-size:0.65rem; font-weight:700; letter-spacing:0.12em; color:{D['text_muted']}; margin-bottom:0.75rem; padding-bottom:0.5rem; border-bottom:1px solid {D['border']}; }}
.ll {{ color:{D['text_dim']}; }}
.ll .ts {{ color:{D['blue']}; }}
.ll .ok {{ color:{D['green']}; }}
.ll .er {{ color:{D['red']}; }}
.ll .wr {{ color:{D['amber']}; }}
.ll .tg {{ color:{D['violet']}; }}
.log-cur {{ display:inline-block; width:7px; height:14px; background:{D['green']}; animation:blink 1s step-end infinite; vertical-align:middle; margin-left:2px; }}

/* ── Cards & badges ── */
.d-card {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-radius:12px; padding:1rem; margin-bottom:0.5rem; transition:border-color .2s; }}
.d-card:hover {{ border-color:{D['border_active']}; }}
.met-card {{ background:{D['bg_card']}; border:1px solid {D['border']}; border-radius:12px; padding:1rem 1.25rem; text-align:center; }}
.met-v {{ font-family:'Sora',sans-serif; font-size:1.75rem; font-weight:700; line-height:1; }}
.met-l {{ font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:600; color:{D['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-top:0.35rem; }}
.d-badge {{ display:inline-block; padding:0.12rem 0.5rem; border-radius:6px; font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:600; letter-spacing:0.03em; }}

/* ── Task mini-cards ── */
.task-m {{ display:flex; align-items:center; gap:0.6rem; padding:0.6rem 0.75rem; background:{D['bg_card']}; border:1px solid {D['border']}; border-radius:10px; margin-bottom:0.4rem; font-family:'DM Sans',sans-serif; font-size:0.82rem; transition:border-color .2s; }}
.task-m:hover {{ border-color:{D['border_active']}; }}
.task-dot {{ width:10px; height:10px; border-radius:3px; flex-shrink:0; }}
.task-id {{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{D['text_muted']}; }}
.task-nm {{ color:{D['text']}; font-weight:500; }}
.task-tm {{ margin-left:auto; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{D['text_muted']}; }}

/* ── Code peek ── */
.code-pk {{ background:{D['bg_deep']}; border:1px solid {D['border']}; border-radius:12px; padding:1rem; font-family:'JetBrains Mono',monospace; font-size:0.78rem; line-height:1.7; color:{D['text_dim']}; overflow-x:auto; }}
.code-pk .ck {{ color:{D['violet']}; }}
.code-pk .cf {{ color:{D['blue']}; }}
.code-pk .cc {{ color:{D['text_muted']}; font-style:italic; }}
.code-pk .cs {{ color:{D['green']}; }}
.code-pk .co {{ color:{D['amber']}; }}

/* ── Section header ── */
.sec-h {{ font-family:'Sora',sans-serif; font-size:1rem; font-weight:700; color:{D['text']}; margin-bottom:0.75rem; display:flex; align-items:center; gap:0.5rem; }}
.sec-h .acc {{ font-size:0.7rem; font-weight:600; padding:0.15rem 0.5rem; border-radius:5px; font-family:'JetBrains Mono',monospace; }}

/* ── Animations ── */
@keyframes msgIn {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0; }} }}
@keyframes pulse {{ 0%,100% {{ box-shadow:0 0 5px var(--gc); }} 50% {{ box-shadow:0 0 20px var(--gc); }} }}

/* ── Scrollbar ── */
.chat-wrap::-webkit-scrollbar, .log-con::-webkit-scrollbar {{ width:6px; }}
.chat-wrap::-webkit-scrollbar-track, .log-con::-webkit-scrollbar-track {{ background:transparent; }}
.chat-wrap::-webkit-scrollbar-thumb, .log-con::-webkit-scrollbar-thumb {{ background:{D['border']}; border-radius:3px; }}
.chat-wrap::-webkit-scrollbar-thumb:hover, .log-con::-webkit-scrollbar-thumb:hover {{ background:{D['border_active']}; }}

.demo-title {{ font-family:'Sora',sans-serif; font-weight:800; font-size:1.5rem; color:{D['text']}; letter-spacing:-0.02em; }}
.demo-sub {{ font-family:'DM Sans',sans-serif; font-size:0.85rem; color:{D['text_muted']}; }}
</style>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _met(value: str, label: str, color: str) -> str:
    return (
        f'<div class="met-card">'
        f'<div class="met-v" style="color:{color}">{value}</div>'
        f'<div class="met-l">{label}</div></div>'
    )


def _task_starts(spec, results) -> dict[str, float]:
    G = nx.DiGraph()
    for t in spec.tasks:
        G.add_node(t.id)
        for dep in t.dependencies:
            G.add_edge(dep.task_id, t.id)
    starts: dict[str, float] = {}
    for tid in nx.topological_sort(G):
        task = spec.get_task(tid)
        ends = [
            starts[d.task_id] + results[d.task_id].elapsed_seconds
            for d in task.dependencies
            if d.task_id in starts and d.task_id in results
        ]
        starts[tid] = max(ends) if ends else 0.0
    return starts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Code Snippets — syntax-highlighted source excerpts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE_SNIPPETS: dict[str, dict] = {
    "goal": {
        "title": "Entry Point",
        "file": "cli/main.py",
        "code": (
            '<span class="cc"># User provides a natural language goal</span>\n'
            '<span class="ck">goal</span> = <span class="cs">"Plan a multi-location Goa trip with costs"</span>\n'
            '<span class="ck">provider</span> = <span class="cs">"gemini"</span>\n\n'
            '<span class="cc"># Full pipeline: compile → execute → critique</span>\n'
            '<span class="ck">await</span> <span class="cf">run_full_pipeline</span>(goal, provider, model, config)'
        ),
    },
    "compile": {
        "title": "Compiler",
        "file": "compiler/compiler.py",
        "code": (
            '<span class="cc"># Compiler: fuzzy goal → strict JSON execution graph</span>\n'
            'compiler = <span class="cf">Compiler</span>(llm, config, provider=provider)\n'
            'spec = <span class="ck">await</span> compiler.<span class="cf">compile</span>(goal)\n\n'
            '<span class="cc"># Auto-fix: if LLM put tool name in role, correct it</span>\n'
            '<span class="cf">_fixup_agent_roles</span>(data)  <span class="cc"># web_search→generic_llm</span>\n\n'
            '<span class="cc"># Validate: Pydantic v2 + DAG cycle detection</span>\n'
            'spec = <span class="cf">WorkflowSpec</span>(**json_response)\n'
            'dag.<span class="cf">validate</span>()  <span class="cc"># Kahn\'s algorithm</span>'
        ),
    },
    "execute": {
        "title": "DAG Executor",
        "file": "engine/executor.py",
        "code": (
            '<span class="cc"># Parallel or sequential based on provider</span>\n'
            '<span class="ck">while not</span> dag.<span class="cf">is_complete</span>():\n'
            '    ready = dag.<span class="cf">get_ready_tasks</span>()  <span class="cc"># 0 unmet deps</span>\n'
            '    <span class="ck">if</span> max_concurrent == <span class="co">1</span>:  <span class="cc"># local LLM</span>\n'
            '        <span class="ck">for</span> tid <span class="ck">in</span> ready:\n'
            '            <span class="ck">await</span> self.<span class="cf">_run_task</span>(dag, tid)\n'
            '    <span class="ck">else</span>:  <span class="cc"># cloud API — full parallelism</span>\n'
            '        <span class="ck">await</span> asyncio.<span class="cf">gather</span>(\n'
            '            *[self.<span class="cf">_run_task</span>(dag, t) <span class="ck">for</span> t <span class="ck">in</span> ready]\n'
            '        )'
        ),
    },
    "critique": {
        "title": "Critic — Self-Healing",
        "file": "critic/critic.py",
        "code": (
            '<span class="cc"># Evaluate each task output vs success criteria</span>\n'
            '<span class="ck">for</span> task <span class="ck">in</span> spec.tasks:\n'
            '    passed, patch, reason = <span class="ck">await</span> critic.<span class="cf">evaluate</span>(\n'
            '        task, results[task.id]\n'
            '    )\n'
            '    <span class="ck">if not</span> passed <span class="ck">and</span> patch:\n'
            '        <span class="ck">await</span> <span class="cf">apply_patch</span>(patch, dag, executor)\n\n'
            '<span class="cc"># Patch types: RETRY | INSERT | REMOVE | UPDATE_INPUT</span>\n'
            '<span class="cc"># Circuit breaker trips after 3 consecutive failures</span>'
        ),
    },
    "summary": {
        "title": "Pipeline Summary",
        "file": "cli/main.py",
        "code": (
            '<span class="cc"># Results stored in artifact store</span>\n'
            '<span class="cc"># .daaw_store/artifacts.json</span>\n\n'
            'final_results = executor._results\n'
            '<span class="cf">display_execution_summary</span>(final_results)\n\n'
            '<span class="cc"># Key insight:</span>\n'
            '<span class="cc"># The LLM thinks ONCE (compilation).</span>\n'
            '<span class="cc"># The runtime executes DETERMINISTICALLY.</span>\n'
            '<span class="cc"># No chatty loops. No guesswork.</span>'
        ),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo Steps — pre-built walkthrough events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_demo_steps() -> list[dict]:
    spec = DEMO_WORKFLOW_SPEC
    verdicts = DEMO_CRITIC_VERDICTS
    pending_all = {t.id: "pending" for t in spec.tasks}

    return [
        # ── 0: Goal ──
        {
            "stage": "goal",
            "messages": [{"role": "user", "label": "You", "icon": "👤",
                          "color": D["blue"],
                          "content": "Plan a multi-location Goa trip with costs"}],
            "logs": [],
            "dag": {},
        },
        # ── 1: Compiler init ──
        {
            "stage": "compile",
            "messages": [{"role": "compiler", "label": "Compiler", "icon": "⚙",
                          "color": D["stage_compile"],
                          "content": (
                              "<strong>Compiler initialized</strong><br><br>"
                              "Provider: <code>gemini</code> &nbsp;|&nbsp; "
                              "Model: <code>gemini-2.5-flash</code><br>"
                              "Building system prompt with <strong>6 agent roles</strong> "
                              "and <strong>5 registered tools</strong>…"
                          )}],
            "logs": [
                {"ts": "0.00s", "text": "Compiler initialized — provider: gemini, model: gemini-2.5-flash", "lv": "info"},
                {"ts": "0.01s", "text": "Building system prompt with 6 agent roles, 5 tools", "lv": "info"},
            ],
            "dag": {},
        },
        # ── 2: LLM call ──
        {
            "stage": "compile",
            "messages": [{"role": "compiler", "label": "Compiler", "icon": "⚙",
                          "color": D["stage_compile"],
                          "content": (
                              "Sending goal to LLM for workflow generation…<br><br>"
                              f'<em style="color:{D["text_muted"]}">The LLM receives the goal along '
                              "with available agent roles and tool schemas. It generates a structured "
                              "JSON execution graph — not prose, not steps, a <strong>DAG</strong>.</em>"
                          )}],
            "logs": [
                {"ts": "0.02s", "text": "Sending goal to LLM for workflow generation…", "lv": "info"},
                {"ts": "2.41s", "text": "LLM response received (3,204 tokens)", "lv": "ok"},
            ],
            "dag": {},
        },
        # ── 3: Parse + validate ──
        {
            "stage": "compile",
            "messages": [{"role": "compiler", "label": "Compiler", "icon": "⚙",
                          "color": D["stage_compile"],
                          "content": (
                              "<strong>Response parsed and validated</strong><br><br>"
                              "&#10004; JSON parsed — 6 tasks found<br>"
                              "&#10004; Pydantic v2 validation passed<br>"
                              "&#10004; DAG validated — no cycles (Kahn's algorithm)<br>"
                              "&#10004; Topological order resolved"
                          )}],
            "logs": [
                {"ts": "2.42s", "text": "Parsing JSON response…", "lv": "info"},
                {"ts": "2.43s", "text": "JSON parsed — 6 tasks found", "lv": "ok"},
                {"ts": "2.45s", "text": "Pydantic validation passed — all schemas valid", "lv": "ok"},
                {"ts": "2.47s", "text": "DAG validated — no cycles detected (Kahn's algorithm)", "lv": "ok"},
            ],
            "dag": dict(pending_all),
        },
        # ── 4: WorkflowSpec summary ──
        {
            "stage": "compile",
            "messages": [{"role": "compiler", "label": "Compiler", "icon": "⚙",
                          "color": D["stage_compile"],
                          "content": (
                              "<strong>WorkflowSpec compiled — 6 tasks, 8 dependencies</strong><br><br>"
                              + "".join(
                                  f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">'
                                  f'<code style="font-size:0.75rem;color:{D["cyan"]}">{t.id}</code> '
                                  f'<span>{_esc(t.name)}</span> '
                                  f'<span class="d-badge" style="background:{D["bg_surface"]};color:{D["text_muted"]};font-size:0.68rem;">{t.agent.role}</span>'
                                  + (f' <span class="d-badge" style="background:{D["cyan"]}22;color:{D["cyan"]};font-size:0.68rem;">{", ".join(t.agent.tools_allowed)}</span>' if t.agent.tools_allowed else "")
                                  + "</div>"
                                  for t in spec.tasks
                              )
                              + f'<br><em style="color:{D["text_muted"]}">Parallel: task_003 + task_004 + task_005 run simultaneously</em>'
                          )}],
            "logs": [
                {"ts": "2.48s", "text": "Topo order: 001 → 002 → 003,004,005 (parallel) → 006", "lv": "info"},
                {"ts": "2.49s", "text": "Compilation complete — WorkflowSpec ready", "lv": "ok"},
            ],
            "dag": dict(pending_all),
            "show_spec": True,
        },
        # ── 5: Executor init + task_001 ──
        {
            "stage": "execute",
            "messages": [{"role": "engine", "label": "DAG Engine", "icon": "▶",
                          "color": D["stage_execute"],
                          "content": (
                              "<strong>DAG Executor initialized</strong><br>"
                              "Components: AgentFactory, ArtifactStore, CircuitBreaker<br>"
                              f'Strategy: Topological sort → parallel <code>asyncio.gather()</code><br><br>'
                              '<span class="st-run">▶ RUNNING</span> '
                              "<strong>task_001: Research Goa Locations</strong><br>"
                              "Agent: <code>generic_llm</code> | Tools: <code>web_search</code>"
                          )}],
            "logs": [
                {"ts": "0.00s", "text": "DAG executor initialized — 6 tasks queued", "lv": "info"},
                {"ts": "0.01s", "text": "[START] task_001: Research Goa Locations", "lv": "tg"},
            ],
            "dag": {**pending_all, "task_001": "running"},
        },
        # ── 6: task_001 done + task_002 start ──
        {
            "stage": "execute",
            "messages": [{"role": "engine", "label": "DAG Engine", "icon": "▶",
                          "color": D["stage_execute"],
                          "content": (
                              '<span class="st-ok">✓ DONE</span> <strong>task_001</strong> — 86.6s '
                              "| 1 tool call (web_search)<br>"
                              f'<em style="color:{D["text_dim"]}">Found 6 locations across North, Central, and South Goa</em><br><br>'
                              '<span class="st-run">▶ RUNNING</span> '
                              "<strong>task_002: Propose Itinerary</strong><br>"
                              "Depends on: <code>task_001</code>"
                          )}],
            "logs": [
                {"ts": "86.6s", "text": "[OK] task_001: Research Goa Locations (86.6s)", "lv": "ok"},
                {"ts": "86.6s", "text": "[START] task_002: Propose Itinerary", "lv": "tg"},
            ],
            "dag": {**pending_all, "task_001": "success", "task_002": "running"},
        },
        # ── 7: task_002 done + parallel group start ──
        {
            "stage": "execute",
            "messages": [{"role": "engine", "label": "DAG Engine", "icon": "▶",
                          "color": D["stage_execute"],
                          "content": (
                              '<span class="st-ok">✓ DONE</span> <strong>task_002</strong> — 43.2s<br>'
                              f'<em style="color:{D["text_dim"]}">6-day / 5-night itinerary: Anjuna → Panjim → Palolem</em><br><br>'
                              f'<div style="background:{D["cyan"]}11;border:1px solid {D["cyan"]}33;'
                              f'border-radius:8px;padding:0.6rem 0.8rem;margin:0.4rem 0;">'
                              f'<strong style="color:{D["cyan"]}">⚡ Parallel Execution</strong><br>'
                              f'<span style="color:{D["text_dim"]}">3 independent tasks via '
                              f'<code>asyncio.gather()</code></span><br><br>'
                              '<span class="st-run">▶</span> task_003: Accommodation Costs<br>'
                              '<span class="st-run">▶</span> task_004: Transport &amp; Activities<br>'
                              '<span class="st-run">▶</span> task_005: Food &amp; Misc Costs</div>'
                          )}],
            "logs": [
                {"ts": "129.8s", "text": "[OK] task_002: Propose Itinerary (43.2s)", "lv": "ok"},
                {"ts": "129.8s", "text": "3 tasks ready — launching parallel via asyncio.gather()", "lv": "info"},
                {"ts": "129.8s", "text": "[START] task_003 | task_004 | task_005", "lv": "tg"},
            ],
            "dag": {"task_001": "success", "task_002": "success",
                    "task_003": "running", "task_004": "running",
                    "task_005": "running", "task_006": "pending"},
        },
        # ── 8: parallel group done + task_006 start ──
        {
            "stage": "execute",
            "messages": [{"role": "engine", "label": "DAG Engine", "icon": "▶",
                          "color": D["stage_execute"],
                          "content": (
                              '<strong>Parallel group completed</strong><br><br>'
                              '<span class="st-ok">✓</span> <strong>task_003</strong> — 64.0s (3 tiers × 3 locations)<br>'
                              '<span class="st-ok">✓</span> <strong>task_004</strong> — 136.1s (transport + 8 activities)<br>'
                              '<span class="st-ok">✓</span> <strong>task_005</strong> — 80.9s (food tiers from Numbeo)<br><br>'
                              f'<em style="color:{D["text_dim"]}">Wall: 136.1s (longest). '
                              f'Sequential would be 280.9s → <strong>2.1× speedup</strong></em><br><br>'
                              '<span class="st-run">▶ RUNNING</span> '
                              '<strong>task_006: Compile Trip Plan</strong><br>'
                              'Deps: <code>002+003+004+005</code> | Tools: <code>file_write</code>'
                          )}],
            "logs": [
                {"ts": "193.8s", "text": "[OK] task_003: Accommodation (64.0s)", "lv": "ok"},
                {"ts": "210.7s", "text": "[OK] task_005: Food & Misc (80.9s)", "lv": "ok"},
                {"ts": "265.9s", "text": "[OK] task_004: Transport (136.1s)", "lv": "ok"},
                {"ts": "265.9s", "text": "[START] task_006: Compile Trip Plan", "lv": "tg"},
            ],
            "dag": {"task_001": "success", "task_002": "success",
                    "task_003": "success", "task_004": "success",
                    "task_005": "success", "task_006": "running"},
        },
        # ── 9: task_006 done ──
        {
            "stage": "execute",
            "messages": [{"role": "engine", "label": "DAG Engine", "icon": "▶",
                          "color": D["stage_execute"],
                          "content": (
                              '<span class="st-ok">✓ DONE</span> '
                              '<strong>task_006: Compile Trip Plan</strong> — 119.0s<br>'
                              f'<em style="color:{D["text_dim"]}">Written to goa-trip-plan.md (8,432 chars)</em><br><br>'
                              f'<div style="background:{D["green"]}11;border:1px solid {D["green"]}33;'
                              f'border-radius:8px;padding:0.6rem 0.8rem;">'
                              f'<strong style="color:{D["green"]}">✓ DAG Execution Complete</strong><br>'
                              f'<span style="color:{D["text_dim"]}">6/6 tasks | Wall: ~529s | '
                              f'Sequential: ~810s | '
                              f'<strong style="color:{D["green"]}">1.5× speedup</strong></span></div>'
                          )}],
            "logs": [
                {"ts": "384.9s", "text": "[OK] task_006: Compile Trip Plan (119.0s)", "lv": "ok"},
                {"ts": "384.9s", "text": "DAG execution complete — 6/6 succeeded", "lv": "ok"},
            ],
            "dag": {t.id: "success" for t in spec.tasks},
        },
        # ── 10: Critic ──
        {
            "stage": "critique",
            "messages": [{"role": "critic", "label": "Critic", "icon": "🔍",
                          "color": D["stage_critique"],
                          "content": (
                              "<strong>Critic evaluation — outputs vs success criteria</strong><br><br>"
                              + "".join(
                                  f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;">'
                                  f'<span class="st-pass">PASS</span> '
                                  f'<code style="font-size:0.75rem;color:{D["cyan"]}">{v["task_id"]}</code> '
                                  f'{_esc(v["task_name"])}</div>'
                                  for v in verdicts
                              )
                              + f'<br><div style="background:{D["green"]}11;border:1px solid {D["green"]}33;'
                              f'border-radius:8px;padding:0.5rem 0.7rem;">'
                              f'<strong style="color:{D["green"]}">6/6 passed</strong> — no patches needed</div>'
                              + f'<br><div style="background:{D["amber"]}11;border:1px solid {D["amber"]}33;'
                              f'border-radius:8px;padding:0.5rem 0.7rem;">'
                              f'<strong style="color:{D["amber"]}">🔄 Self-Healing Loop</strong><br>'
                              f'<span style="color:{D["text_dim"]};font-size:0.82rem;">'
                              f'On failure: Critic generates a <code>WorkflowPatch</code> '
                              f'(RETRY / INSERT / REMOVE / UPDATE_INPUT) → applied to live DAG → '
                              f're-executed → re-evaluated. Circuit breaker trips after 3 failures.</span></div>'
                          )}],
            "logs": [
                {"ts": "385.0s", "text": "Critic evaluation — 6 tasks to check", "lv": "info"},
                {"ts": "387.2s", "text": "[PASS] task_001: 6 locations (>4 required)", "lv": "ok"},
                {"ts": "389.1s", "text": "[PASS] task_002: 3 locations, logical flow", "lv": "ok"},
                {"ts": "391.4s", "text": "[PASS] task_003: 3 tiers × 3 locations", "lv": "ok"},
                {"ts": "393.2s", "text": "[PASS] task_004: transport + 8 activities", "lv": "ok"},
                {"ts": "395.0s", "text": "[PASS] task_005: 3-tier food + misc buffer", "lv": "ok"},
                {"ts": "397.3s", "text": "[PASS] task_006: comprehensive plan", "lv": "ok"},
                {"ts": "397.4s", "text": "Critic complete — 6/6 PASS, 0 patches", "lv": "ok"},
            ],
            "dag": {t.id: "success" for t in spec.tasks},
        },
        # ── 11: Summary ──
        {
            "stage": "summary",
            "messages": [{"role": "summary", "label": "Summary", "icon": "📊",
                          "color": D["stage_summary"],
                          "content": (
                              "<strong>Pipeline Complete</strong><br><br>"
                              f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin:0.5rem 0;">'
                              + "".join(
                                  f'<div style="text-align:center;background:{D["bg_surface"]};border-radius:8px;padding:0.5rem;">'
                                  f'<div style="font-family:Sora;font-size:1.1rem;font-weight:700;color:{c}">{v}</div>'
                                  f'<div style="font-size:0.65rem;color:{D["text_muted"]};text-transform:uppercase;letter-spacing:.05em">{l}</div></div>'
                                  for v, l, c in [
                                      ("6/6", "Tasks Passed", D["green"]),
                                      ("~529s", "Wall Clock", D["blue"]),
                                      ("1.5×", "Speedup", D["cyan"]),
                                      ("6", "Tool Calls", D["amber"]),
                                  ]
                              )
                              + "</div><br>"
                              f'<em style="color:{D["text_dim"]}">The LLM thought <strong>once</strong> '
                              f'(compilation). The runtime executed <strong>deterministically</strong>.</em>'
                          )}],
            "logs": [
                {"ts": "", "text": "═══ PIPELINE SUMMARY ═══", "lv": "info"},
                {"ts": "", "text": "Tasks: 6/6 | Wall: ~529s | Speedup: 1.5× | Tools: 6 calls", "lv": "ok"},
            ],
            "dag": {t.id: "success" for t in spec.tasks},
            "show_metrics": True,
        },
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rendering Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_STAGES = ["goal", "compile", "execute", "critique", "summary"]
_STAGE_NAMES = ["Goal", "Compiler", "DAG Engine", "Critic", "Summary"]
_STAGE_COLORS = [D["stage_goal"], D["stage_compile"], D["stage_execute"],
                 D["stage_critique"], D["stage_summary"]]


def render_stepper(current: str):
    idx = _STAGES.index(current) if current in _STAGES else -1
    h = ['<div class="stepper-wrap">']
    for i, (s, nm, cl) in enumerate(zip(_STAGES, _STAGE_NAMES, _STAGE_COLORS)):
        cls = "completed" if i < idx else "active" if i == idx else "pending"
        h.append(f'<div class="stepper-step {cls}" style="--sc:{cl}">')
        h.append(f'<div class="stepper-num">{i + 1}</div><span>{nm}</span></div>')
        if i < len(_STAGES) - 1:
            cc = "done" if i < idx else "on" if i == idx else ""
            h.append(f'<div class="stepper-conn {cc}" style="--pc:{_STAGE_COLORS[i]};--nc:{_STAGE_COLORS[i + 1]}"></div>')
    h.append("</div>")
    st.markdown("\n".join(h), unsafe_allow_html=True)


def render_chat(messages: list[dict]):
    h = ['<div class="chat-wrap">']
    if not messages:
        h.append(
            f'<div style="text-align:center;padding:3rem 2rem;">'
            f'<div style="font-size:2.5rem;margin-bottom:0.5rem;">🔬</div>'
            f'<div style="font-family:Sora;font-size:1.1rem;font-weight:600;color:{D["text"]}">Pipeline Visualization</div>'
            f'<div style="font-family:DM Sans;font-size:0.85rem;color:{D["text_muted"]};margin-top:0.5rem;">'
            f'Click <strong>Next Step</strong> to walk through the DAAW pipeline.</div></div>'
        )
    for m in messages:
        is_u = m["role"] == "user"
        mc = "usr" if is_u else ""
        bc = "user" if is_u else m["role"]
        cl = m.get("color", D["blue"])
        ic = m.get("icon", "👤")
        lb = m.get("label", "You")
        h.append(f'<div class="chat-msg {mc}">')
        h.append(f'<div class="chat-av" style="background:{cl}22;border:1px solid {cl}44">{ic}</div>')
        h.append(f'<div class="chat-body">')
        h.append(f'<div class="chat-lbl" style="color:{cl}">{_esc(lb)}</div>')
        h.append(f'<div class="chat-bub {bc}">{m["content"]}</div>')
        h.append("</div></div>")
    h.append("</div>")
    st.markdown("\n".join(h), unsafe_allow_html=True)


def render_logs(logs: list[dict]):
    h = ['<div class="log-con">']
    if not logs:
        h.append(f'<div class="ll" style="color:{D["text_muted"]}">Waiting for pipeline…</div>')
    for l in logs:
        ts = l.get("ts", "")
        ts_h = f'<span class="ts">[{_esc(ts)}]</span> ' if ts else ""
        lv = l.get("lv", "info")
        lc = {"ok": "ok", "error": "er", "warn": "wr", "tg": "tg"}.get(lv, "")
        txt = f'<span class="{lc}">{_esc(l["text"])}</span>' if lc else _esc(l["text"])
        h.append(f'<div class="ll">{ts_h}{txt}</div>')
    h.append('<span class="log-cur"></span></div>')
    st.markdown("\n".join(h), unsafe_allow_html=True)


def render_dag(spec, dag_state: dict[str, str]):
    if not dag_state:
        st.info("DAG appears after compilation.")
        return
    G = nx.DiGraph()
    for t in spec.tasks:
        G.add_node(t.id, name=t.name)
        for dep in t.dependencies:
            G.add_edge(dep.task_id, t.id)
    topo = list(nx.topological_sort(G))
    layers: dict[str, int] = {}
    for n in topo:
        preds = list(G.predecessors(n))
        layers[n] = (max(layers[p] for p in preds) + 1) if preds else 0
    lg: dict[int, list[str]] = {}
    for n, ly in layers.items():
        lg.setdefault(ly, []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    for ly, nodes in lg.items():
        nn = len(nodes)
        for i, n in enumerate(nodes):
            pos[n] = (ly * 3, (i - (nn - 1) / 2) * 2)

    sc = {"pending": D["text_muted"], "running": D["cyan"],
          "success": D["green"], "failure": D["red"]}
    ex, ey = [], []
    for s, d in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[d]
        ex.extend([x0, x1, None]); ey.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color=D["border_active"], width=2),
                             hoverinfo="skip", showlegend=False))
    for s, d in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[d]
        fig.add_annotation(ax=x0, ay=y0, x=x1, y=y1,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.5,
                           arrowwidth=2, arrowcolor=D["border_active"], opacity=0.7)

    tids = spec.task_ids()
    colors = [sc.get(dag_state.get(t, "pending"), D["text_muted"]) for t in tids]
    sizes = [44 if dag_state.get(t) == "running" else 36 for t in tids]
    short = [spec.get_task(t).name[:18] + ("…" if len(spec.get_task(t).name) > 18 else "") for t in tids]
    hover = [
        f"<b>{spec.get_task(t).name}</b><br>Agent: {spec.get_task(t).agent.role}<br>"
        f"Status: {dag_state.get(t, 'pending').upper()}"
        for t in tids
    ]
    fig.add_trace(go.Scatter(
        x=[pos[t][0] for t in tids], y=[pos[t][1] for t in tids],
        mode="markers+text", marker=dict(size=sizes, color=colors,
                                          line=dict(color=D["text"], width=1.5), opacity=0.9),
        text=short, textposition="top center",
        textfont=dict(size=10, color=D["text"], family="DM Sans"),
        hovertext=hover, hoverinfo="text", showlegend=False))
    # Legend
    for nm, cl in [("SUCCESS", D["green"]), ("RUNNING", D["cyan"]),
                   ("PENDING", D["text_muted"]), ("FAILURE", D["red"])]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=cl), name=nm))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=D["bg_deep"], plot_bgcolor=D["bg_deep"],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", y=-0.08, xanchor="center", x=0.5,
                    font=dict(color=D["text"], size=10), bgcolor="rgba(0,0,0,0)"),
        font=dict(color=D["text"]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_task_output(spec, results: dict, dag_state: dict):
    if not dag_state:
        st.info("Execute to see task outputs.")
        return
    done = [t for t, s in dag_state.items() if s in ("success", "failure") and t in results]
    if not done:
        st.info("Outputs appear as tasks complete.")
        return
    sel = st.selectbox("Task", done,
                       format_func=lambda t: f"{t} — {spec.get_task(t).name}")
    if not sel or sel not in results:
        return
    r = results[sel]; task = spec.get_task(sel)
    sc = D["green"] if r.agent_result.status == "success" else D["red"]
    st.markdown(
        f'<div class="d-card" style="border-left:3px solid {sc}">'
        f'<strong style="color:{D["text"]}">{_esc(task.name)}</strong> '
        f'<span class="d-badge" style="background:{sc}22;color:{sc}">{r.agent_result.status.upper()}</span>'
        f'<div style="font-size:.78rem;color:{D["text_muted"]};margin-top:.3rem">'
        f'Agent: {_esc(task.agent.role)} | {r.elapsed_seconds:.1f}s</div></div>',
        unsafe_allow_html=True)
    # Show the error message prominently when the task failed, so users
    # don't have to download the JSON to see why.
    err = getattr(r.agent_result, "error_message", "") or ""
    if r.agent_result.status != "success" and err:
        st.markdown(
            f'<div style="background:{D["red"]}11;border:1px solid {D["red"]}44;'
            f'border-left:4px solid {D["red"]};border-radius:8px;'
            f'padding:.7rem 1rem;margin:.4rem 0">'
            f'<strong style="color:{D["red"]};font-family:Sora">Error</strong>'
            f'<div class="mono" style="font-size:.78rem;color:{D["text"]};'
            f'margin-top:.3rem;white-space:pre-wrap;word-break:break-word">'
            f'{_esc(err)}</div></div>',
            unsafe_allow_html=True,
        )
    out = r.agent_result.output
    if out is None and r.agent_result.status != "success":
        pass  # error box above is sufficient; no "None" rendered
    elif isinstance(out, (dict, list)):
        st.json(out)
    else:
        st.code(str(out)[:2000], language="markdown")
    tcs = (r.agent_result.metadata or {}).get("tool_calls", [])
    if tcs:
        st.markdown(f'<div style="font-size:.82rem;font-weight:600;color:{D["text"]};margin:.5rem 0">Tool Calls ({len(tcs)})</div>', unsafe_allow_html=True)
        for tc in tcs:
            res_prev = str(tc.get("result", ""))[:150]
            st.markdown(
                f'<div style="background:{D["bg_surface"]};border-radius:8px;padding:.5rem .75rem;margin-bottom:.3rem;font-size:.8rem">'
                f'<strong style="color:{D["cyan"]}">{_esc(tc.get("tool", ""))}</strong>'
                f'<div style="color:{D["text_dim"]};margin-top:.2rem">{_esc(res_prev)}</div></div>',
                unsafe_allow_html=True)


def render_task_cards(spec, dag_state: dict, results: dict):
    sc = {"pending": D["text_muted"], "running": D["cyan"],
          "success": D["green"], "failure": D["red"]}
    for t in spec.tasks:
        s = dag_state.get(t.id, "pending")
        cl = sc.get(s, D["text_muted"])
        tm = f"{results[t.id].elapsed_seconds:.1f}s" if t.id in results else ""
        st.markdown(
            f'<div class="task-m">'
            f'<div class="task-dot" style="background:{cl}"></div>'
            f'<div><div class="task-nm">{_esc(t.name)}</div>'
            f'<div class="task-id">{_esc(t.id)} · {_esc(t.agent.role)}</div></div>'
            f'<div class="task-tm">{tm}</div></div>',
            unsafe_allow_html=True)


def render_code_peek(stage: str):
    sn = CODE_SNIPPETS.get(stage)
    if not sn:
        st.info("Source snippets update per pipeline stage.")
        return
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">'
        f'<span style="font-family:DM Sans;font-size:.82rem;font-weight:600;color:{D["text"]}">{sn["title"]}</span>'
        f'<span style="font-family:JetBrains Mono;font-size:.68rem;color:{D["text_muted"]}">{sn["file"]}</span></div>'
        f'<div class="code-pk"><pre style="margin:0;white-space:pre-wrap">{sn["code"]}</pre></div>',
        unsafe_allow_html=True)


def render_metrics(spec, results: dict):
    if not results:
        return
    ts = _task_starts(spec, results)
    total = sum(r.elapsed_seconds for r in results.values())
    wall = max(ts.get(t, 0) + results[t].elapsed_seconds for t in results)
    spd = total / wall if wall > 0 else 1
    ok = sum(1 for r in results.values() if r.agent_result.status == "success")
    tcs = sum(len((r.agent_result.metadata or {}).get("tool_calls", []))
              for r in results.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_met(f"{wall:.0f}s", "Wall Clock", D["blue"]), unsafe_allow_html=True)
    c2.markdown(_met(f"{spd:.1f}×", "Speedup", D["cyan"]), unsafe_allow_html=True)
    c3.markdown(_met(f"{ok}/{len(results)}", "Success", D["green"]), unsafe_allow_html=True)
    c4.markdown(_met(str(tcs), "Tool Calls", D["amber"]), unsafe_allow_html=True)

    st.markdown("---")
    # Gantt
    bt = datetime(2025, 6, 15, 10, 0, 0)
    rows = []
    for t in spec.tasks:
        r = results.get(t.id)
        if not r:
            continue
        s = ts.get(t.id, 0)
        rows.append({"Task": t.name,
                      "Start": bt + timedelta(seconds=s),
                      "Finish": bt + timedelta(seconds=s + r.elapsed_seconds),
                      "Status": r.agent_result.status.upper()})
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Status",
                      color_discrete_map={"SUCCESS": D["green"], "FAILURE": D["red"]})
    fig.update_layout(
        height=max(220, len(rows) * 42), margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=D["bg"], plot_bgcolor=D["bg"],
        xaxis=dict(title="", gridcolor=D["border"], color=D["text"]),
        yaxis=dict(title="", autorange="reversed", color=D["text"]),
        legend=dict(orientation="h", y=-0.35, font=dict(color=D["text"]), bgcolor="rgba(0,0,0,0)"),
        font=dict(color=D["text"]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Tool distribution
    tc_map: dict[str, int] = {}
    for r in results.values():
        for tc in (r.agent_result.metadata or {}).get("tool_calls", []):
            k = tc.get("tool", "?")
            tc_map[k] = tc_map.get(k, 0) + 1
    if tc_map:
        fig2 = go.Figure(data=[go.Bar(x=list(tc_map.keys()), y=list(tc_map.values()),
                                       marker_color=D["cyan"], text=list(tc_map.values()),
                                       textposition="auto")])
        fig2.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor=D["bg"], plot_bgcolor=D["bg"],
            xaxis=dict(color=D["text"], gridcolor=D["border"]),
            yaxis=dict(color=D["text"], gridcolor=D["border"], title="Calls"),
            font=dict(color=D["text"]))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _sys_stats() -> dict:
    try:
        from daaw.agents.registry import AGENT_REGISTRY
        from daaw.tools.registry import tool_registry
        a = len(AGENT_REGISTRY)
        t = len(tool_registry._tools)
    except Exception:
        a, t = 6, 5
    return {"agents": a, "tools": t, "providers": 5, "schemas": 11}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:1.25rem 0 .5rem">'
            f'<div style="font-family:Sora;font-size:2.2rem;font-weight:800;color:{D["blue"]};'
            f'letter-spacing:-.02em;line-height:1">DAAW</div>'
            f'<div style="font-family:DM Sans;color:{D["text_muted"]};font-size:.72rem;'
            f'margin-top:.3rem;letter-spacing:.12em;text-transform:uppercase">Under the Hood</div></div>',
            unsafe_allow_html=True)
        st.divider()

        mode = st.radio("Mode", ["Demo Walkthrough", "Live Mode"], index=0,
                        horizontal=True,
                        help="Demo: step through pre-loaded data. Live: real LLM.")

        step_fwd = step_back = reset = play_all = False
        provider = model = goal = None
        compile_btn = False

        if mode == "Demo Walkthrough":
            st.markdown(
                f'<div style="font-family:DM Sans;font-size:.82rem;color:{D["text_dim"]};margin:.5rem 0">'
                f'Step through each stage to see how DAAW works.</div>',
                unsafe_allow_html=True)
            total = len(_build_demo_steps())
            cur = st.session_state.get("demo_step", -1)
            c1, c2 = st.columns(2)
            with c1:
                step_fwd = st.button("Next Step ▶", use_container_width=True,
                                     type="primary", disabled=cur >= total - 1)
            with c2:
                step_back = st.button("◀ Back", use_container_width=True,
                                      disabled=cur < 0)
            play_all = st.button("▶▶ Play All", use_container_width=True)
            pct = (cur + 1) / total if cur >= 0 else 0
            st.progress(pct, text=f"Step {cur + 1 if cur >= 0 else 0} / {total}")
            reset = st.button("↺ Reset", use_container_width=True)
        else:
            with st.expander("Live Settings", expanded=True):
                provider = st.selectbox("Provider",
                                        ["groq", "gemini", "openai", "anthropic", "gateway"])
                _M = {
                    "groq": [
                        "llama-3.3-70b-versatile",
                        "llama-3.1-8b-instant",              # free, fastest
                        "meta-llama/llama-4-scout-17b-16e-instruct",
                        "qwen/qwen3-32b",
                    ],
                    "gemini": [
                        "gemini-2.5-flash-lite",              # cheapest
                        "gemini-2.5-flash",                   # cheap + capable
                        "gemini-2.5-pro",                     # most capable
                    ],
                    "openai": [
                        "gpt-4.1-nano",                       # cheapest
                        "gpt-4.1-mini",                       # cheap
                        "gpt-4.1",                            # capable
                        "o4-mini",                            # reasoning
                    ],
                    "anthropic": [
                        "claude-haiku-4-5-20251001",          # cheapest
                        "claude-sonnet-4-6",                  # balanced
                        "claude-opus-4-6",                    # most capable
                    ],
                    "gateway": [
                        "gemma4:e4b",
                        "gemma4:e2b-it-q4_K_M",
                        "default",
                    ],
                }
                model = st.selectbox("Model", _M.get(provider, ["default"]))
                if provider == "gateway":
                    st.markdown(
                        f'<div style="font-size:.75rem;color:{D["amber"]};background:{D["amber"]}11;'
                        f'border:1px solid {D["amber"]}33;border-radius:6px;padding:.4rem .6rem;margin:.3rem 0">'
                        f'Local LLM — tasks run <strong>sequentially</strong> (one at a time)</div>',
                        unsafe_allow_html=True)
                goal = st.text_area("Goal", placeholder="Describe your workflow goal…", height=80)
                hitl_enabled = st.checkbox(
                    "Enable human-in-the-loop prompts",
                    value=True,
                    help=(
                        "When on, user_proxy / PM tasks pause the workflow and "
                        "ask you questions in the UI. Off = auto-fill with demo defaults."
                    ),
                )
                st.session_state["hitl_enabled"] = hitl_enabled
                compile_btn = st.button("Compile & Execute", type="primary",
                                        use_container_width=True)

        st.divider()
        ss = _sys_stats()
        st.markdown(
            f'<div style="font-family:DM Sans;font-size:.65rem;font-weight:600;'
            f'color:{D["text_muted"]};text-transform:uppercase;letter-spacing:.1em;'
            f'margin-bottom:.5rem">System</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Agents", ss["agents"]); c2.metric("Tools", ss["tools"])
        c1.metric("Providers", ss["providers"]); c2.metric("Schemas", ss["schemas"])

        st.divider()
        from daaw.__version__ import BUILD_TAG as _BT, __version__ as _VER
        st.markdown(
            f'<div style="font-family:JetBrains Mono;font-size:.68rem;'
            f'color:{D["text_muted"]};text-align:center;margin-top:.25rem">'
            f'DAAW v<strong style="color:{D["cyan"]}">{_VER}</strong>'
            f' · <span style="color:{D["amber"]}">{_BT}</span></div>',
            unsafe_allow_html=True,
        )

    return mode, step_fwd, step_back, reset, play_all, provider, model, goal, compile_btn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State Management
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _apply_step(steps: list[dict], idx: int):
    s = steps[idx]
    st.session_state.demo_msgs.extend(s.get("messages", []))
    st.session_state.demo_logs.extend(s.get("logs", []))
    if s.get("dag"):
        st.session_state.demo_dag.update(s["dag"])
    st.session_state.demo_stage = s["stage"]
    if s.get("show_metrics"):
        st.session_state.demo_show_met = True
    if s.get("show_spec"):
        st.session_state.demo_show_spec = True


def _rebuild(steps: list[dict], up_to: int):
    st.session_state.demo_msgs = []
    st.session_state.demo_logs = []
    st.session_state.demo_dag = {}
    st.session_state.demo_stage = "goal"
    st.session_state.demo_show_met = False
    st.session_state.demo_show_spec = False
    for i in range(up_to + 1):
        _apply_step(steps, i)


def _init_state():
    for k, v in [("demo_step", -1), ("demo_msgs", []), ("demo_logs", []),
                 ("demo_dag", {}), ("demo_stage", "goal"),
                 ("demo_show_met", False), ("demo_show_spec", False),
                 ("live_msgs", []), ("live_logs", []), ("live_dag", {}),
                 ("live_stage", "goal"), ("live_spec", None),
                 ("live_results", {}), ("live_verdicts", []),
                 ("live_exec_handle", None),
                 ("live_show_met", False), ("hitl_enabled", True)]:
        if k not in st.session_state:
            st.session_state[k] = v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Live Mode helpers — exports + HITL bridge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _safe_filename(name: str) -> str:
    """Sanitize a task id into a filesystem-safe stem."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return stem or "task"


def _build_outputs_zip(spec, results) -> bytes:
    """Build a ZIP with one JSON per task output plus a combined all_outputs.json."""
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
            zf.writestr(f"{_safe_filename(tid)}.json",
                        json.dumps(payload, indent=2, default=str))
        zf.writestr("all_outputs.json", json.dumps(combined, indent=2, default=str))
    return buffer.getvalue()


def render_export_row(spec, results, verdicts: list[dict] | None = None) -> None:
    """Render download buttons for WorkflowSpec / Results / per-task ZIP / verdicts."""
    if spec is None:
        return
    verdicts = verdicts or []
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        st.download_button(
            "Download WorkflowSpec JSON",
            data=spec.model_dump_json(indent=2),
            file_name="workflow_spec.json",
            mime="application/json",
            use_container_width=True,
            key="demo_dl_spec",
        )
    with col2:
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
                key="demo_dl_results",
            )
        else:
            st.caption("Execute to export results.")
    with col3:
        if results:
            st.download_button(
                "Download Output JSONs (ZIP)",
                data=_build_outputs_zip(spec, results),
                file_name="task_outputs.zip",
                mime="application/zip",
                use_container_width=True,
                help="One JSON per task output plus a combined all_outputs.json.",
                key="demo_dl_zip",
            )
        else:
            st.caption("No outputs yet.")
    with col4:
        if verdicts:
            st.download_button(
                "Verdicts",
                data=json.dumps(verdicts, indent=2, default=str),
                file_name="verdicts.json",
                mime="application/json",
                use_container_width=True,
                key="demo_dl_verdicts",
            )


def _auto_fill_user_proxy(spec) -> None:
    """Swap user_proxy tasks for an auto-filling generic_llm agent (HITL disabled)."""
    for t in spec.tasks:
        if t.agent.role == "user_proxy":
            t.agent = t.agent.model_copy(update={
                "role": "generic_llm",
                "system_prompt_override": (
                    "You are auto-filling user parameters for a demo. "
                    "Based on the task description, return ONLY a JSON object "
                    "with concrete values (dates, locations, budget numbers). "
                    "Do NOT generate a prompt or ask questions."
                )})


# ── HITL bridge: background executor thread + queue-based prompt relay ─────

def _start_exec_hitl(spec, provider: str, model: str) -> dict | None:
    """Spin the execute + critic phases onto a background thread with a
    ``QueueInteractionHandler``. Returns a handle driven by ``_poll_exec_hitl``.
    """
    try:
        import daaw.agents.builtin.breakdown_agent  # noqa: F401
        import daaw.agents.builtin.critic_agent  # noqa: F401
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        import daaw.agents.builtin.planner_agent  # noqa: F401
        import daaw.agents.builtin.pm_agent  # noqa: F401
        import daaw.agents.builtin.user_proxy  # noqa: F401
        try:
            import daaw.tools.real_tools  # noqa: F401
        except ImportError:
            import daaw.tools.mock_tools  # noqa: F401

        from daaw.agents.factory import AgentFactory
        from daaw.config import get_config
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.interaction import QueueInteractionHandler
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.store.artifact_store import ArtifactStore

        config = get_config()
        llm = UnifiedLLMClient(config)
        is_local = provider == "gateway"

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        questions: "queue.Queue" = queue.Queue()
        answers: "queue.Queue" = queue.Queue()
        handler = QueueInteractionHandler(questions, answers, timeout=1800.0)

        factory = AgentFactory(
            llm, store,
            default_provider=provider, default_model=model,
            interaction_handler=handler,
        )
        executor = DAGExecutor(factory, store, cb,
                               max_concurrent=1 if is_local else None)

        holder: dict = {"results": None, "verdicts": None, "error": None,
                        "exec_time": 0.0}

        def _worker():
            try:
                loop = asyncio.new_event_loop()
                try:
                    t0 = time.monotonic()
                    results = loop.run_until_complete(executor.execute(spec))
                    holder["exec_time"] = time.monotonic() - t0
                    critic = Critic(llm, config, provider=provider, model=model)
                    verdicts: list[dict] = []
                    for task in spec.tasks:
                        if task.id not in results:
                            continue
                        try:
                            passed, patch, reason = loop.run_until_complete(
                                critic.evaluate(task, results[task.id])
                            )
                        except Exception as crit_err:
                            passed, patch, reason = False, None, (
                                f"Critic error: {str(crit_err)[:60]}"
                            )
                        verdicts.append({
                            "task_id": task.id,
                            "task_name": task.name,
                            "verdict": "PASS" if passed else "FAIL",
                            "reasoning": reason,
                            "patch": str(patch.operations) if patch else None,
                        })
                    holder["results"] = results
                    holder["verdicts"] = verdicts
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                holder["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, name="daaw-demo-exec", daemon=True)
        thread.start()

        return {
            "thread": thread, "questions": questions, "answers": answers,
            "holder": holder, "pending": None, "started_at": time.monotonic(),
        }
    except Exception as e:
        st.error(f"Failed to start HITL execution: {e}")
        st.code(traceback.format_exc(), language="text")
        return None


def _poll_exec_hitl(handle: dict) -> str:
    """Return ``pending_question`` / ``running`` / ``done`` / ``error``."""
    if handle.get("pending") is not None:
        return "pending_question"
    try:
        handle["pending"] = handle["questions"].get_nowait()
        return "pending_question"
    except queue.Empty:
        pass
    if handle["holder"].get("error"):
        return "error"
    if not handle["thread"].is_alive():
        try:
            handle["pending"] = handle["questions"].get_nowait()
            return "pending_question"
        except queue.Empty:
            return "done"
    return "running"


def _answer_exec_hitl(handle: dict, answer: str) -> None:
    handle["answers"].put(str(answer))
    handle["pending"] = None


def _render_hitl_prompt_demo(handle: dict) -> bool:
    """Render the pending HITL question. Returns True if the user just submitted."""
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
        f'<div style="background:{D["bg_card"]};border:1px solid {D["border"]};'
        f'border-left:4px solid {D["amber"]};border-radius:10px;padding:.9rem 1.1rem;'
        f'margin:.5rem 0;">'
        f'<strong style="color:{D["amber"]};font-family:Sora">HUMAN INPUT REQUESTED</strong>'
        f'<div style="font-size:.72rem;color:{D["text_muted"]};margin-top:.2rem">'
        f'agent: <code>{_esc(agent_id)}</code> · step: <code>{_esc(step_id)}</code>'
        + (
            f' · Q {ctx.get("step")}/{ctx.get("total")}'
            if ctx.get("step") and ctx.get("total") else ""
        )
        + '</div></div>',
        unsafe_allow_html=True,
    )

    with st.form(f"hitl_form_{id(req)}", clear_on_submit=True):
        st.markdown(f"**{_esc(prompt_text)}**")
        if hint:
            st.caption(hint)
        choice_val = None
        if choices:
            choice_val = st.radio("Pick one (or type below):",
                                  options=choices, horizontal=True,
                                  key=f"hitl_radio_{id(req)}")
        freeform = st.text_area("Your answer:",
                                key=f"hitl_text_{id(req)}", height=110)
        submitted = st.form_submit_button("Submit answer", use_container_width=True)
    if submitted:
        answer = (freeform or "").strip() or (choice_val or "")
        _answer_exec_hitl(handle, answer)
        return True
    return False


def _finalize_exec_hitl(spec, handle: dict) -> None:
    """Write results/verdicts from a finished handle back into session_state."""
    holder = handle["holder"]
    results = holder.get("results") or {}
    verdicts = holder.get("verdicts") or []
    exec_time = holder.get("exec_time") or 0.0
    st.session_state.live_results = results
    st.session_state.live_verdicts = verdicts

    exec_lines = []
    for t in spec.tasks:
        r = results.get(t.id)
        if not r:
            continue
        s = r.agent_result.status
        st.session_state.live_dag[t.id] = s
        is_ok = s == "success"
        st.session_state.live_logs.append(
            {"ts": f"{r.elapsed_seconds:.1f}s",
             "text": f"[{'OK' if is_ok else 'FAIL'}] {t.id}: {t.name}",
             "lv": "ok" if is_ok else "er"})
        tcs = len((r.agent_result.metadata or {}).get("tool_calls", []))
        tc_str = f" | {tcs} tool calls" if tcs else ""
        tag = "st-ok" if is_ok else "st-fail"
        label = "✓" if is_ok else "✗"
        exec_lines.append(
            f'<span class="{tag}">{label}</span> '
            f'<strong>{_esc(t.id)}</strong> — {r.elapsed_seconds:.1f}s{tc_str}')

    ok = sum(1 for r in results.values() if r.agent_result.status == "success")
    fail = len(results) - ok
    st.session_state.live_msgs.append(
        {"role": "engine", "label": "DAG Engine", "icon": "▶",
         "color": D["stage_execute"],
         "content": (
             f"<strong>Execution complete</strong> — {exec_time:.1f}s wall clock<br><br>"
             + "<br>".join(exec_lines)
             + f'<br><br><span class="st-ok">{ok} passed</span>'
             + (f' <span class="st-fail">{fail} failed</span>' if fail else "")
         )})

    pass_count = sum(1 for v in verdicts if v.get("verdict") == "PASS")
    verdict_lines = []
    for v in verdicts:
        passed = v.get("verdict") == "PASS"
        st.session_state.live_logs.append(
            {"ts": "", "text": f"[{v.get('verdict', '?')}] {v.get('task_id', '')}: "
                                f"{(v.get('reasoning') or '')[:80]}",
             "lv": "ok" if passed else "er"})
        tag = "st-pass" if passed else "st-fail"
        label = v.get("verdict", "?")
        verdict_lines.append(
            f'<span class="{tag}">{label}</span> '
            f'<code style="font-size:0.75rem;color:{D["cyan"]}">{_esc(v.get("task_id", ""))}</code> '
            f'{_esc(v.get("task_name", ""))}')
    st.session_state.live_stage = "critique"
    if verdicts:
        st.session_state.live_msgs.append(
            {"role": "critic", "label": "Critic", "icon": "🔍",
             "color": D["stage_critique"],
             "content": (
                 "<strong>Critic evaluation</strong><br><br>"
                 + "<br>".join(verdict_lines)
                 + f'<br><br><strong>{pass_count}/{len(results)} tasks passed</strong>'
             )})

    total_elapsed = sum(r.elapsed_seconds for r in results.values())
    st.session_state.live_stage = "summary"
    st.session_state.live_show_met = True
    st.session_state.live_msgs.append(
        {"role": "summary", "label": "Summary", "icon": "📊",
         "color": D["stage_summary"],
         "content": (
             "<strong>Pipeline complete</strong><br><br>"
             f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin:0.5rem 0;">'
             + "".join(
                 f'<div style="text-align:center;background:{D["bg_surface"]};border-radius:8px;padding:0.5rem;">'
                 f'<div style="font-family:Sora;font-size:1.1rem;font-weight:700;color:{c}">{v}</div>'
                 f'<div style="font-size:0.65rem;color:{D["text_muted"]};text-transform:uppercase;letter-spacing:.05em">{la}</div></div>'
                 for v, la, c in [
                     (f"{pass_count}/{len(results)}", "Passed", D["green"]),
                     (f"{exec_time:.0f}s", "Wall Clock", D["blue"]),
                     (f"{total_elapsed:.0f}s", "Total CPU", D["cyan"]),
                 ]
             )
             + "</div>"
         )})
    st.session_state.live_logs.append(
        {"ts": "", "text": f"Pipeline done — {pass_count}/{len(results)} passed, "
                           f"{exec_time:.0f}s wall", "lv": "ok"})


def _run_live(provider: str, model: str, goal: str, hitl: bool = True):
    st.session_state.live_msgs = [
        {"role": "user", "label": "You", "icon": "👤",
         "color": D["blue"], "content": _esc(goal)}]
    st.session_state.live_logs = []
    st.session_state.live_dag = {}
    st.session_state.live_stage = "compile"
    st.session_state.live_spec = None
    st.session_state.live_results = {}
    st.session_state.live_verdicts = []
    st.session_state.live_show_met = False
    st.session_state["live_exec_handle"] = None

    try:
        import time as _time
        from daaw.compiler.compiler import Compiler
        from daaw.config import get_config
        from daaw.llm.unified import UnifiedLLMClient

        config = get_config()
        llm = UnifiedLLMClient(config)
        avail = llm.available_providers()
        if provider not in avail:
            hint = "Set GATEWAY_URL env var." if provider == "gateway" else f"Set {provider.upper()}_API_KEY."
            st.error(f"Provider '{provider}' not available ({hint}). Available: {avail}")
            return

        is_local = provider == "gateway"
        exec_mode = "sequential (local LLM)" if is_local else "parallel (cloud API)"

        st.session_state.live_logs.append(
            {"ts": "0.0s", "text": f"Compiler init — {provider}/{model}", "lv": "info"})
        st.session_state.live_msgs.append(
            {"role": "compiler", "label": "Compiler", "icon": "⚙",
             "color": D["stage_compile"],
             "content": (
                 f"<strong>Compiler initialized</strong><br>"
                 f"Provider: <code>{_esc(provider)}</code> | "
                 f"Model: <code>{_esc(model)}</code><br>"
                 f"Execution mode: <code>{exec_mode}</code>"
             )})

        t0 = _time.monotonic()
        compiler = Compiler(llm, config, provider=provider, model=model)
        spec = _run_async(compiler.compile(goal))
        compile_time = _time.monotonic() - t0

        st.session_state.live_spec = spec
        st.session_state.live_dag = {t.id: "pending" for t in spec.tasks}

        # Build task list for chat
        task_lines = "".join(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0;">'
            f'<code style="font-size:0.75rem;color:{D["cyan"]}">{t.id}</code> '
            f'{_esc(t.name)} '
            f'<span class="d-badge" style="background:{D["bg_surface"]};color:{D["text_muted"]};font-size:0.68rem;">{t.agent.role}</span>'
            + (f' <span class="d-badge" style="background:{D["cyan"]}22;color:{D["cyan"]};font-size:0.68rem;">{", ".join(t.agent.tools_allowed)}</span>' if t.agent.tools_allowed else "")
            + "</div>"
            for t in spec.tasks
        )

        st.session_state.live_msgs.append(
            {"role": "compiler", "label": "Compiler", "icon": "⚙",
             "color": D["stage_compile"],
             "content": (
                 f"<strong>Compiled {len(spec.tasks)} tasks</strong> "
                 f"in {compile_time:.1f}s<br><br>{task_lines}"
             )})
        st.session_state.live_logs.append(
            {"ts": f"{compile_time:.1f}s", "text": f"Compiled {len(spec.tasks)} tasks", "lv": "ok"})

        # ── Execute ──
        st.session_state.live_stage = "execute"
        if hitl:
            # HITL path: spawn the background executor and return.
            # The main() poll loop will drive the prompt form and finalization.
            st.session_state.live_logs.append(
                {"ts": "", "text": "HITL executor starting (user_proxy prompts live in UI)",
                 "lv": "info"})
            handle = _start_exec_hitl(spec, provider, model)
            if handle is None:
                st.session_state.live_logs.append(
                    {"ts": "", "text": "Failed to start HITL executor", "lv": "error"})
                return
            st.session_state["live_exec_handle"] = handle
            return  # main() continues via _poll_exec_hitl → _finalize_exec_hitl

        # Legacy synchronous path (HITL disabled): auto-fill user_proxy tasks.
        import daaw.agents.builtin.breakdown_agent  # noqa: F401
        import daaw.agents.builtin.critic_agent  # noqa: F401
        import daaw.agents.builtin.generic_llm_agent  # noqa: F401
        import daaw.agents.builtin.planner_agent  # noqa: F401
        import daaw.agents.builtin.pm_agent  # noqa: F401
        import daaw.agents.builtin.user_proxy  # noqa: F401

        try:
            import daaw.tools.real_tools  # noqa: F401
        except ImportError:
            import daaw.tools.mock_tools  # noqa: F401

        from daaw.agents.factory import AgentFactory
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.store.artifact_store import ArtifactStore

        _auto_fill_user_proxy(spec)

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        factory = AgentFactory(llm, store, default_provider=provider, default_model=model)
        max_conc = 1 if is_local else None
        executor = DAGExecutor(factory, store, cb, max_concurrent=max_conc)

        st.session_state.live_logs.append(
            {"ts": "", "text": f"DAG executor init — max_concurrent={max_conc or 'unlimited'}", "lv": "info"})

        t0 = _time.monotonic()
        results = _run_async(executor.execute(spec))
        exec_time = _time.monotonic() - t0
        st.session_state.live_results = results

        # Per-task results in chat + logs
        exec_lines = []
        for t in spec.tasks:
            r = results.get(t.id)
            if not r:
                continue
            s = r.agent_result.status
            st.session_state.live_dag[t.id] = s
            is_ok = s == "success"
            st.session_state.live_logs.append(
                {"ts": f"{r.elapsed_seconds:.1f}s",
                 "text": f"[{'OK' if is_ok else 'FAIL'}] {t.id}: {t.name}",
                 "lv": "ok" if is_ok else "er"})
            tcs = len((r.agent_result.metadata or {}).get("tool_calls", []))
            tc_str = f" | {tcs} tool calls" if tcs else ""
            tag = "st-ok" if is_ok else "st-fail"
            label = "✓" if is_ok else "✗"
            exec_lines.append(
                f'<span class="{tag}">{label}</span> '
                f'<strong>{_esc(t.id)}</strong> — {r.elapsed_seconds:.1f}s{tc_str}')

        ok = sum(1 for r in results.values() if r.agent_result.status == "success")
        fail = len(results) - ok
        st.session_state.live_msgs.append(
            {"role": "engine", "label": "DAG Engine", "icon": "▶",
             "color": D["stage_execute"],
             "content": (
                 f"<strong>Execution complete</strong> — {exec_time:.1f}s wall clock<br><br>"
                 + "<br>".join(exec_lines)
                 + f'<br><br><span class="st-ok">{ok} passed</span>'
                 + (f' <span class="st-fail">{fail} failed</span>' if fail else "")
             )})

        # ── Critique ──
        # Each evaluation is wrapped in try/except because local models
        # can crash mid-sequence (OOM after many sequential requests).
        st.session_state.live_stage = "critique"
        critic = Critic(llm, config, provider=provider, model=model)
        pass_count = 0
        verdict_lines = []
        for t in spec.tasks:
            if t.id not in results:
                continue
            try:
                passed, patch, reason = _run_async(critic.evaluate(t, results[t.id]))
            except Exception as crit_err:
                passed, reason = False, f"Critic error: {str(crit_err)[:60]}"
                st.session_state.live_logs.append(
                    {"ts": "", "text": f"[SKIP] {t.id}: critic crashed, skipping", "lv": "wr"})
            if passed:
                pass_count += 1
            st.session_state.live_logs.append(
                {"ts": "", "text": f"[{'PASS' if passed else 'FAIL'}] {t.id}: {(reason or '')[:80]}",
                 "lv": "ok" if passed else "er"})
            tag = "st-pass" if passed else "st-fail"
            label = "PASS" if passed else "FAIL"
            verdict_lines.append(
                f'<span class="{tag}">{label}</span> '
                f'<code style="font-size:0.75rem;color:{D["cyan"]}">{t.id}</code> '
                f'{_esc(t.name)}')

        st.session_state.live_msgs.append(
            {"role": "critic", "label": "Critic", "icon": "🔍",
             "color": D["stage_critique"],
             "content": (
                 "<strong>Critic evaluation</strong><br><br>"
                 + "<br>".join(verdict_lines)
                 + f'<br><br><strong>{pass_count}/{len(results)} tasks passed</strong>'
             )})

        # ── Summary ──
        total_elapsed = sum(r.elapsed_seconds for r in results.values())
        st.session_state.live_stage = "summary"
        st.session_state.live_show_met = True
        st.session_state.live_msgs.append(
            {"role": "summary", "label": "Summary", "icon": "📊",
             "color": D["stage_summary"],
             "content": (
                 "<strong>Pipeline complete</strong><br><br>"
                 f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin:0.5rem 0;">'
                 + "".join(
                     f'<div style="text-align:center;background:{D["bg_surface"]};border-radius:8px;padding:0.5rem;">'
                     f'<div style="font-family:Sora;font-size:1.1rem;font-weight:700;color:{c}">{v}</div>'
                     f'<div style="font-size:0.65rem;color:{D["text_muted"]};text-transform:uppercase;letter-spacing:.05em">{la}</div></div>'
                     for v, la, c in [
                         (f"{pass_count}/{len(results)}", "Passed", D["green"]),
                         (f"{exec_time:.0f}s", "Wall Clock", D["blue"]),
                         (f"{total_elapsed:.0f}s", "Total CPU", D["cyan"]),
                     ]
                 )
                 + "</div>"
             )})
        st.session_state.live_logs.append(
            {"ts": "", "text": f"Pipeline done — {pass_count}/{len(results)} passed, {exec_time:.0f}s wall", "lv": "ok"})

    except Exception as e:
        st.session_state.live_msgs.append(
            {"role": "summary", "label": "Error", "icon": "❌",
             "color": D["red"],
             "content": f"<strong>Failed:</strong> <code>{_esc(str(e))}</code>"})
        st.session_state.live_logs.append(
            {"ts": "", "text": f"ERROR: {e}", "lv": "error"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    mode, step_fwd, step_back, reset, play_all, provider, model, goal, compile_btn = render_sidebar()

    # ── Demo controls ──
    if mode == "Demo Walkthrough":
        steps = _build_demo_steps()
        if reset:
            st.session_state.demo_step = -1
            _rebuild(steps, -1)  # clears everything
            st.session_state.demo_msgs = []
            st.session_state.demo_logs = []
            st.session_state.demo_dag = {}
            st.rerun()
        if step_fwd and st.session_state.demo_step < len(steps) - 1:
            st.session_state.demo_step += 1
            _apply_step(steps, st.session_state.demo_step)
            st.rerun()
        if step_back and st.session_state.demo_step >= 0:
            st.session_state.demo_step = max(-1, st.session_state.demo_step - 1)
            if st.session_state.demo_step >= 0:
                _rebuild(steps, st.session_state.demo_step)
            else:
                st.session_state.demo_msgs = []
                st.session_state.demo_logs = []
                st.session_state.demo_dag = {}
                st.session_state.demo_stage = "goal"
                st.session_state.demo_show_met = False
                st.session_state.demo_show_spec = False
            st.rerun()
        if play_all:
            st.session_state.demo_step = len(steps) - 1
            _rebuild(steps, len(steps) - 1)
            st.rerun()

        spec = DEMO_WORKFLOW_SPEC
        results = DEMO_RESULTS
        msgs = st.session_state.demo_msgs
        logs = st.session_state.demo_logs
        dag_st = st.session_state.demo_dag
        stage = st.session_state.demo_stage
        show_met = st.session_state.demo_show_met
        show_spec = st.session_state.demo_show_spec

    elif mode == "Live Mode":
        hitl_enabled = st.session_state.get("hitl_enabled", True)

        if compile_btn and goal:
            with st.status("Running pipeline…", expanded=True) as status:
                status.write("Compiling workflow…")
                _run_live(provider, model, goal, hitl=hitl_enabled)
                if st.session_state.get("live_exec_handle") is None:
                    # Legacy synchronous path already ran to completion.
                    status.update(label="Pipeline complete", state="complete")
                else:
                    status.update(
                        label="Compiled — HITL executor started, see prompt below",
                        state="running",
                    )
            st.rerun()

        # ── HITL poller (runs every rerun while a handle is active) ──
        handle = st.session_state.get("live_exec_handle")
        if handle is not None:
            state = _poll_exec_hitl(handle)
            if state == "pending_question":
                st.markdown("### Workflow paused — your input is needed")
                submitted = _render_hitl_prompt_demo(handle)
                if submitted:
                    time.sleep(0.05)
                    st.rerun()
                else:
                    st.info(
                        "The executor is waiting for your reply. "
                        "Answer above and click **Submit answer** to resume.",
                        icon="⏸",
                    )
            elif state == "running":
                elapsed = int(time.monotonic() - handle["started_at"])
                st.info(f"Executing workflow… ({elapsed}s elapsed)", icon="⚙")
                time.sleep(1.0)
                st.rerun()
            elif state == "error":
                st.error("Execution crashed")
                st.code(handle["holder"]["error"], language="text")
                st.session_state["live_exec_handle"] = None
            elif state == "done":
                _finalize_exec_hitl(st.session_state.live_spec, handle)
                st.session_state["live_exec_handle"] = None
                st.rerun()

        spec = st.session_state.live_spec
        results = st.session_state.live_results
        msgs = st.session_state.live_msgs
        logs = st.session_state.live_logs
        dag_st = st.session_state.live_dag
        stage = st.session_state.live_stage
        show_met = st.session_state.live_show_met
        show_spec = spec is not None
    else:
        return

    # ── Header ──
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.25rem">'
        f'<span class="demo-title">Under the Hood</span>'
        f'<span class="demo-sub">— Pipeline step-by-step visualization</span></div>',
        unsafe_allow_html=True)

    # ── Pipeline stepper ──
    render_stepper(stage)

    # ── Main: Chat + Inspection ──
    col_chat, col_inspect = st.columns([3, 2])

    with col_chat:
        st.markdown(
            f'<div class="sec-h">Pipeline Flow '
            f'<span class="acc" style="background:{D["blue"]}22;color:{D["blue"]}">Chat</span></div>',
            unsafe_allow_html=True)
        render_chat(msgs)

    with col_inspect:
        tab_logs, tab_dag, tab_out, tab_src = st.tabs(["Logs", "DAG", "Output", "Source"])
        with tab_logs:
            render_logs(logs)
        with tab_dag:
            if spec:
                render_dag(spec, dag_st)
            else:
                st.info("Compile a workflow to see the DAG.")
        with tab_out:
            vis_results = {t: results[t] for t in dag_st
                           if dag_st.get(t) in ("success", "failure") and t in (results or {})} if results else {}
            if spec and vis_results:
                render_task_output(spec, vis_results, dag_st)
            else:
                st.info("Task outputs appear as tasks complete.")
        with tab_src:
            render_code_peek(stage)

    # ── Task status cards ──
    if dag_st:
        st.markdown("---")
        st.markdown(
            f'<div class="sec-h">Task Status</div>', unsafe_allow_html=True)
        vis_r = {t: results[t] for t in dag_st
                 if dag_st.get(t) in ("success", "failure") and t in (results or {})} if results else {}
        render_task_cards(spec, dag_st, vis_r)

    # ── Metrics ──
    if show_met and results:
        st.markdown("---")
        st.markdown(
            f'<div class="sec-h">Performance Metrics '
            f'<span class="acc" style="background:{D["green"]}22;color:{D["green"]}">Results</span></div>',
            unsafe_allow_html=True)
        render_metrics(spec, results)

    # ── Exports (downloads) ──
    if show_spec and spec and mode == "Live Mode":
        st.markdown("---")
        st.markdown(
            f'<div class="sec-h">Exports '
            f'<span class="acc" style="background:{D["cyan"]}22;color:{D["cyan"]}">Downloads</span></div>',
            unsafe_allow_html=True)
        render_export_row(
            spec,
            results or {},
            st.session_state.get("live_verdicts") or [],
        )

    # ── WorkflowSpec JSON ──
    if show_spec and spec:
        with st.expander("WorkflowSpec JSON", expanded=False):
            st.json(json.loads(spec.model_dump_json()))


if __name__ == "__main__":
    main()
