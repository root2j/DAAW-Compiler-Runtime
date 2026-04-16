"""DAAW — Chat UI.

A ChatGPT / Claude–style interface that hides every moving part of DAAW
behind a single chat input. The user types a goal; the assistant bubble
streams back:

    1. "Compiling plan..."     (live JSON tokens as the planner writes)
    2. "Task 1 / Research..."  (live text tokens for each task that
                                doesn't rely on tool-calling — tool
                                rounds still run, just without a token
                                stream during the tool-use turn)
    3. A final natural-language synthesis section.

Under the hood it's the same compile -> execute -> critic pipeline; the
UI simply hides the DAG, the artifact store, the circuit breaker, etc.
A collapsible "Details" block per assistant message exposes the raw
task outputs, tool calls, verdicts, and the WorkflowSpec JSON.

Launch: ``python -m daaw chat`` (also works standalone: ``streamlit run
src/daaw/ui/chat_app.py``).
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

# Must be the first Streamlit call.
st.set_page_config(
    page_title="DAAW Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from daaw.config import reset_config as _reset_config
_reset_config()

from daaw.__version__ import BUILD_TAG as _BT, __version__ as _VER
from daaw.ui._streaming_display import prettify_partial_json


# ---------------------------------------------------------------------------
# Styling — Swiss editorial: warm paper, deep ink, one vermilion accent,
# Fraunces display + IBM Plex Sans body. Flight-deck status for tasks,
# choreographed stage transitions with CSS width animations.
# ---------------------------------------------------------------------------
C = {
    "paper":   "#F5F3EE",   # warm off-white
    "paper_2": "#FFFEFA",   # card / raised surface
    "ink":     "#1A1815",   # warm near-black
    "ink_2":   "#4A4638",   # body muted
    "ink_3":   "#8A8470",   # meta / secondary
    "rule":    "#D8D1BF",   # hairline
    "accent":  "#D84315",   # vermilion (one and only)
    "accent_soft": "#FBEBE4",
    "success": "#3D5F3A",   # forest (not bright)
    "fail":    "#8B2421",   # oxblood (not bright)
    "warn":    "#9C6F1E",   # antique gold
    "muted":   "#8A8470",   # alias for ink_3 (kept for convenience)
    "border":  "#D8D1BF",   # alias for rule
    "text":    "#1A1815",   # alias for ink
    "card":    "#FFFEFA",   # alias for paper_2
    "danger":  "#8B2421",   # alias for fail
}

_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT@0,9..144,300..900,0..100;1,9..144,300..900,0..100&family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ---------- canvas ---------- */
  .stApp {{
    background: {C['paper']};
    color: {C['ink']};
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    font-feature-settings: 'ss01', 'ss02';
  }}
  .block-container {{
    padding-top: 2.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 780px;
  }}
  p, div, span {{ color: {C['ink']}; }}

  /* ---------- editorial typography ---------- */
  h1, h2, h3 {{ font-family: 'Fraunces', Georgia, serif !important; font-weight: 400 !important; letter-spacing: -0.01em; }}

  /* ---------- masthead ---------- */
  .masthead {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: baseline;
    padding: 0.5rem 0 2.5rem;
    border-bottom: 1px solid {C['rule']};
    margin-bottom: 3rem;
    position: relative;
  }}
  .masthead::after {{
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: -5px;
    height: 1px;
    background: {C['rule']};
  }}
  .masthead-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 300;
    font-variation-settings: 'opsz' 144, 'SOFT' 50;
    letter-spacing: -0.03em;
    color: {C['ink']};
    line-height: 1;
  }}
  .masthead-title em {{
    font-style: italic;
    font-variation-settings: 'opsz' 144, 'SOFT' 100;
    color: {C['accent']};
    font-weight: 400;
  }}
  .masthead-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: {C['ink_3']};
    text-align: right;
    line-height: 1.4;
    letter-spacing: 0.02em;
  }}
  .masthead-meta strong {{ color: {C['ink']}; font-weight: 500; }}

  /* ---------- small labels (section headers, categories) ---------- */
  .label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {C['ink_3']};
  }}
  .label-accent {{ color: {C['accent']}; }}

  /* ---------- stage bar (flight-deck pipeline) ---------- */
  .stage-bar {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 1px solid {C['rule']};
    background: {C['paper_2']};
    margin: 1rem 0 2rem;
    position: relative;
    overflow: hidden;
  }}
  .stage-bar::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: var(--progress, 0%);
    background: {C['accent']};
    transition: width 0.5s cubic-bezier(0.65, 0, 0.35, 1);
    z-index: 0;
  }}
  .stage-seg {{
    padding: 0.7rem 1rem;
    border-right: 1px solid {C['rule']};
    position: relative;
    z-index: 1;
    transition: color 0.4s ease;
  }}
  .stage-seg:last-child {{ border-right: none; }}
  .stage-seg-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: {C['ink_3']};
    display: block;
    margin-bottom: 0.15rem;
  }}
  .stage-seg-label {{
    font-family: 'Fraunces', serif;
    font-size: 0.95rem;
    font-weight: 400;
    color: {C['ink_2']};
  }}
  .stage-seg.past .stage-seg-num,
  .stage-seg.past .stage-seg-label {{ color: {C['paper']}; }}
  .stage-seg.active .stage-seg-label {{
    color: {C['accent']};
    font-weight: 500;
    font-style: italic;
  }}
  .stage-seg.active .stage-seg-num {{ color: {C['accent']}; }}
  .stage-seg.active::after {{
    content: '';
    display: inline-block;
    width: 0.4em;
    height: 0.9em;
    background: {C['accent']};
    margin-left: 0.3em;
    vertical-align: text-bottom;
    animation: blink 1s steps(1) infinite;
  }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}

  /* ---------- task row (flight-deck status) ---------- */
  .task-row {{
    display: grid;
    grid-template-columns: 2.5rem 1fr auto;
    column-gap: 1.2rem;
    padding: 1.1rem 0;
    border-top: 1px solid {C['rule']};
    position: relative;
    animation: slideIn 0.4s cubic-bezier(0.2, 0.8, 0.2, 1) backwards;
  }}
  .task-row:last-child {{ border-bottom: 1px solid {C['rule']}; }}
  .task-row.running {{
    background: linear-gradient(90deg, {C['accent_soft']} 0%, transparent 60%);
    box-shadow: inset 3px 0 0 {C['accent']};
  }}
  @keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-8px); }}
    to {{ opacity: 1; transform: translateX(0); }}
  }}
  .task-num {{
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    font-weight: 300;
    font-variation-settings: 'opsz' 144;
    color: {C['ink_3']};
    line-height: 1;
    letter-spacing: -0.02em;
  }}
  .task-row.success .task-num {{ color: {C['ink']}; }}
  .task-row.failure .task-num {{ color: {C['fail']}; }}
  .task-row.running .task-num {{ color: {C['accent']}; }}
  .task-title {{
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: {C['ink']};
    line-height: 1.3;
    margin-bottom: 0.2rem;
    letter-spacing: -0.005em;
  }}
  .task-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: {C['ink_3']};
    letter-spacing: 0.05em;
    text-transform: lowercase;
  }}
  .task-meta strong {{ color: {C['ink_2']}; font-weight: 500; }}
  .task-status {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 500;
    white-space: nowrap;
    padding-top: 0.6rem;
  }}
  .task-status.pending {{ color: {C['ink_3']}; }}
  .task-status.running {{ color: {C['accent']}; }}
  .task-status.success {{ color: {C['success']}; }}
  .task-status.failure {{ color: {C['fail']}; }}
  .task-status .dot {{
    display: inline-block;
    width: 0.45rem;
    height: 0.45rem;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
  }}
  .task-status.pending .dot {{ background: {C['ink_3']}; }}
  .task-status.running .dot {{ background: {C['accent']}; animation: pulse 1.4s ease-in-out infinite; }}
  .task-status.success .dot {{ background: {C['success']}; }}
  .task-status.failure .dot {{ background: {C['fail']}; }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
  }}

  /* ---------- editorial prompt ( user message ) ---------- */
  .prompt-block {{
    padding: 1rem 0 2rem;
    border-top: 2px solid {C['ink']};
    margin-top: 2.5rem;
    position: relative;
  }}
  .prompt-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {C['ink_3']};
    margin-bottom: 0.5rem;
    font-weight: 500;
  }}
  .prompt-text {{
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 300;
    font-variation-settings: 'opsz' 144, 'SOFT' 50;
    color: {C['ink']};
    line-height: 1.35;
    letter-spacing: -0.015em;
  }}
  .prompt-text::first-letter {{
    font-size: 2.4em;
    float: left;
    line-height: 0.9;
    padding: 0.08em 0.1em 0 0;
    font-weight: 400;
    font-style: italic;
    color: {C['accent']};
  }}

  /* ---------- assistant response frame ---------- */
  .response-block {{ padding: 0.5rem 0 1rem; }}
  .response-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {C['accent']};
    margin-bottom: 0.6rem;
    font-weight: 600;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }}
  .response-label-right {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: {C['ink_3']};
    letter-spacing: 0.08em;
    text-transform: none;
    font-weight: 400;
  }}

  /* ---------- compile stream — editorial drafting frame ---------- */
  .compile-frame {{
    border: 1px solid {C['rule']};
    background: {C['paper_2']};
    padding: 0.5rem 0.9rem;
    margin: 0.8rem 0 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    line-height: 1.55;
    color: {C['ink_2']};
    position: relative;
  }}
  .compile-frame-label {{
    position: absolute;
    top: -0.55rem;
    left: 0.8rem;
    background: {C['paper']};
    padding: 0 0.4rem;
    font-size: 0.56rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {C['ink_3']};
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
  }}

  /* ---------- final answer (editorial body treatment) ---------- */
  .final-answer {{
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.6;
    color: {C['ink']};
    padding: 1.2rem 0;
    border-top: 1px solid {C['rule']};
    margin-top: 1.5rem;
  }}
  .final-answer h1, .final-answer h2, .final-answer h3 {{ font-size: 1.15rem !important; }}

  /* ---------- HITL prompt — inline editorial callout ---------- */
  .hitl-callout {{
    border-left: 3px solid {C['accent']};
    padding: 0.8rem 1rem;
    margin: 1.2rem 0;
    background: {C['accent_soft']};
  }}
  .hitl-callout-label {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: {C['accent']};
    font-weight: 600;
    margin-bottom: 0.3rem;
  }}

  /* ---------- chat input — clean editorial bar ---------- */
  [data-testid="stChatInput"] {{
    background: {C['paper']} !important;
    border-top: 2px solid {C['ink']} !important;
    border-radius: 0 !important;
    padding: 0.8rem 0 !important;
  }}
  [data-testid="stChatInput"] > div {{
    background: {C['paper_2']} !important;
    border: 1px solid {C['rule']} !important;
    border-radius: 0 !important;
    box-shadow: none !important;
  }}
  [data-testid="stChatInput"] textarea {{
    font-family: 'Fraunces', serif !important;
    font-size: 1.05rem !important;
    font-weight: 400 !important;
    background: transparent !important;
    color: {C['ink']} !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 0.9rem !important;
  }}
  [data-testid="stChatInput"] textarea::placeholder {{
    color: {C['ink_3']} !important;
    font-style: italic;
    font-family: 'Fraunces', serif !important;
  }}
  /* Send button */
  [data-testid="stChatInput"] button {{
    background: {C['ink']} !important;
    color: {C['paper']} !important;
    border-radius: 0 !important;
    border: none !important;
  }}
  [data-testid="stChatInput"] button:hover {{
    background: {C['accent']} !important;
  }}
  [data-testid="stChatInput"] button svg {{
    fill: {C['paper']} !important;
  }}

  /* ---------- streamlit top bar / header override ---------- */
  [data-testid="stHeader"] {{
    background: {C['paper']} !important;
    border-bottom: 1px solid {C['rule']} !important;
  }}
  /* Hide the deploy button / hamburger in top bar */
  [data-testid="stHeader"] [data-testid="stToolbar"] {{
    display: none !important;
  }}

  /* ---------- bottom padding for fixed chat input ---------- */
  [data-testid="stBottomBlockContainer"] {{
    background: {C['paper']} !important;
  }}

  /* ---------- streamlit chat_message restyle ---------- */
  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"] {{ display: none !important; }}
  [data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
  }}
  [data-testid="stChatMessage"] > div:first-child {{ display: none !important; }}

  /* ---------- sidebar (editorial pamphlet) ---------- */
  [data-testid="stSidebar"] {{
    background: {C['paper_2']} !important;
    border-right: 1px solid {C['rule']} !important;
  }}
  [data-testid="stSidebar"] .stRadio label {{
    font-family: 'Fraunces', serif !important;
    font-size: 0.95rem !important;
    color: {C['ink']} !important;
  }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {{
    color: {C['ink_2']} !important;
  }}
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {C['ink_3']} !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
  }}

  /* ---------- buttons ---------- */
  .stButton > button {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: 1px solid {C['ink']} !important;
    background: {C['paper']} !important;
    color: {C['ink']} !important;
    border-radius: 0 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
  }}
  .stButton > button:hover {{
    background: {C['ink']} !important;
    color: {C['paper']} !important;
  }}

  /* ---------- expanders ---------- */
  [data-testid="stExpander"] {{
    border: 1px solid {C['rule']} !important;
    border-radius: 0 !important;
    background: transparent !important;
    margin: 0.5rem 0 !important;
  }}
  [data-testid="stExpander"] summary {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: {C['ink_2']} !important;
    font-weight: 500 !important;
  }}

  /* ---------- status captions ---------- */
  .status-whisper {{
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: {C['ink_3']};
    padding: 0.4rem 0;
    font-weight: 300;
  }}
  .status-whisper::before {{
    content: '—  ';
    color: {C['accent']};
    font-style: normal;
  }}

  /* ---------- files saved footer ---------- */
  .files-saved {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    color: {C['ink_3']};
    padding: 0.6rem 0;
    border-top: 1px solid {C['rule']};
    margin-top: 0.8rem;
    letter-spacing: 0.03em;
  }}
  .files-saved strong {{ color: {C['ink_2']}; font-weight: 500; }}

  /* ---------- verdict row ---------- */
  .verdict-row {{
    display: grid;
    grid-template-columns: 4rem 1fr;
    padding: 0.5rem 0;
    border-top: 1px solid {C['rule']};
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    line-height: 1.4;
  }}
  .verdict-row:last-child {{ border-bottom: 1px solid {C['rule']}; }}
  .verdict-tag {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
  }}
  .verdict-tag.pass {{ color: {C['success']}; }}
  .verdict-tag.fail {{ color: {C['fail']}; }}
  .verdict-body {{ color: {C['ink_2']}; }}
  .verdict-body code {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: {C['rule']};
    padding: 0 0.3em;
    color: {C['ink']};
  }}
</style>
"""


# ---------------------------------------------------------------------------
# Conversation data model
# ---------------------------------------------------------------------------
@dataclass
class TaskTrace:
    task_id: str
    name: str
    status: str = "pending"           # pending / running / success / failure
    output: Any = None
    elapsed: float = 0.0
    tool_calls: int = 0
    streaming_text: str = ""          # accumulated live tokens
    error: str = ""
    tool_events: list[str] = field(default_factory=list)  # "web_search(query)", etc.


@dataclass
class Turn:
    role: str                          # "user" | "assistant"
    content: str = ""                  # final text answer
    stage: str = "idle"                # compile / execute / critic / done / error / hitl_prompt
    compile_stream: str = ""           # live JSON during compile
    tasks: list[TaskTrace] = field(default_factory=list)
    error: str = ""
    spec_json: str = ""
    verdicts: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    hitl_request: Any = None           # InteractionRequest when waiting for user
    # Cached once when the turn transitions to "done" so we don't re-scan
    # the sandbox directory on every Streamlit rerun.
    output_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []  # list[Turn]
    if "chat_handle" not in st.session_state:
        st.session_state.chat_handle = None
    if "chat_provider" not in st.session_state:
        st.session_state.chat_provider = "gateway"
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = ""


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------
def _start_pipeline(
    goal: str,
    provider: str,
    model: str | None,
    compile_provider: str | None = None,
    compile_model: str | None = None,
) -> dict | None:
    """Kick off compile -> execute -> critic on a background thread.

    ``compile_provider`` / ``compile_model`` override which LLM is used for
    the compile (planner) phase. When set, the Compiler uses those instead
    of the default execution provider. This enables the "Claude plans,
    Groq executes" split without touching env vars.
    """
    try:
        # Lazy imports so import-time cost of chat_app stays minimal.
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
        from daaw.compiler.compiler import Compiler
        from daaw.config import get_config
        from daaw.critic.critic import Critic
        from daaw.engine.circuit_breaker import CircuitBreaker
        from daaw.engine.executor import DAGExecutor
        from daaw.llm.unified import UnifiedLLMClient
        from daaw.store.artifact_store import ArtifactStore

        config = get_config()
        llm = UnifiedLLMClient(config)

        # For Local (Ollama) mode, ensure the "gateway" provider points at
        # Ollama (port 11434) even if .env's GATEWAY_URL points elsewhere.
        if provider == "gateway" and "gateway" not in llm.available_providers():
            from daaw.llm.providers.gateway_provider import GatewayProvider
            llm._providers["gateway"] = GatewayProvider(
                gateway_url="http://localhost:11434/v1",
                default_model=model or "gemma4:e2b-it-q4_K_M",
            )
        # Same for compile_provider if it needs gateway
        if compile_provider == "gateway" and "gateway" not in llm.available_providers():
            from daaw.llm.providers.gateway_provider import GatewayProvider
            llm._providers["gateway"] = GatewayProvider(
                gateway_url="http://localhost:11434/v1",
                default_model=compile_model or "gemma4:e2b-it-q4_K_M",
            )

        if provider not in llm.available_providers():
            return {"error": (
                f"Provider '{provider}' not configured. "
                f"Available: {llm.available_providers()}"
            )}

        events: "queue.Queue[dict]" = queue.Queue()

        def _send(kind: str, **data):
            events.put({"kind": kind, **data})

        def _on_compile_token(delta: str, full: str):
            _send("compile_token", delta=delta, full=full)

        def _on_agent_token(task_id: str, delta: str, full: str):
            _send("agent_token", task_id=task_id, delta=delta, full=full)

        # HITL: queue-based interaction handler so user_proxy tasks can
        # prompt the user in the chat UI.
        from daaw.interaction import QueueInteractionHandler
        hitl_questions: "queue.Queue" = queue.Queue()
        hitl_answers: "queue.Queue" = queue.Queue()
        hitl_handler = QueueInteractionHandler(
            hitl_questions, hitl_answers, timeout=1800.0,
        )

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        factory = AgentFactory(
            llm, store,
            default_provider=provider, default_model=model,
            on_agent_token=_on_agent_token,
            interaction_handler=hitl_handler,
        )
        is_local = provider == "gateway"
        executor = DAGExecutor(factory, store, cb,
                               max_concurrent=1 if is_local else None)

        holder: dict = {"spec": None, "results": None, "verdicts": None,
                        "error": None, "elapsed": 0.0}

        def _worker():
            try:
                loop = asyncio.new_event_loop()
                try:
                    t0 = time.monotonic()
                    _cp = compile_provider or provider
                    _cm = compile_model or model
                    compiler = Compiler(llm, config, provider=_cp, model=_cm)
                    _send("stage", stage="compile")
                    spec = loop.run_until_complete(
                        compiler.compile_stream(goal, on_token=_on_compile_token),
                    )
                    holder["spec"] = spec
                    _send("spec", spec_json=spec.model_dump_json(indent=2),
                          tasks=[{"id": t.id, "name": t.name,
                                  "tools": list(t.agent.tools_allowed or [])}
                                 for t in spec.tasks])
                    _send("stage", stage="execute")
                    results = loop.run_until_complete(executor.execute(spec))
                    holder["results"] = results

                    # Emit per-task final status with tool call details.
                    for tid, r in results.items():
                        meta = r.agent_result.metadata or {}
                        tc_list = meta.get("tool_calls", []) or []
                        tc_summaries = [
                            f"{tc.get('tool', '?')}({', '.join(f'{k}={v!r}' for k,v in (tc.get('args') or {}).items())[:60]})"
                            for tc in tc_list[:10]
                        ]
                        _send("task_done",
                              task_id=tid,
                              status=r.agent_result.status,
                              output=r.agent_result.output,
                              elapsed=r.elapsed_seconds,
                              tool_calls=len(tc_list),
                              tool_summaries=tc_summaries,
                              error=getattr(r.agent_result, "error_message", "") or "")

                    _send("stage", stage="critic")
                    critic = Critic(llm, config, provider=provider, model=model)
                    verdicts = []
                    for task in spec.tasks:
                        if task.id not in results:
                            continue
                        try:
                            passed, _patch, reason = loop.run_until_complete(
                                critic.evaluate(task, results[task.id]),
                            )
                        except Exception as e:  # noqa: BLE001
                            passed, reason = False, f"Critic error: {e}"
                        verdicts.append({
                            "task_id": task.id, "task_name": task.name,
                            "verdict": "PASS" if passed else "FAIL",
                            "reasoning": reason,
                        })
                    holder["verdicts"] = verdicts
                    _send("verdicts", verdicts=verdicts)

                    holder["elapsed"] = round(time.monotonic() - t0, 2)
                    _send("stage", stage="done", elapsed=holder["elapsed"])
                finally:
                    loop.close()
            except Exception as e:  # noqa: BLE001
                holder["error"] = f"{type(e).__name__}: {e}"
                _send("error", message=holder["error"],
                      traceback=traceback.format_exc())

        thread = threading.Thread(target=_worker, daemon=True,
                                  name="daaw-chat-pipeline")
        thread.start()
        return {"thread": thread, "events": events, "holder": holder,
                "started_at": time.monotonic(),
                "hitl_questions": hitl_questions,
                "hitl_answers": hitl_answers}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to start: {e}"}


def _drain_events(handle: dict, turn: Turn) -> bool:
    """Pull events off the worker's queue and fold them into ``turn``.

    Returns True if anything changed (so the caller can decide to rerun).
    """
    changed = False
    while True:
        try:
            evt = handle["events"].get_nowait()
        except queue.Empty:
            break
        changed = True
        kind = evt.get("kind")
        if kind == "stage":
            turn.stage = evt["stage"]
            if "elapsed" in evt:
                turn.elapsed = evt["elapsed"]
        elif kind == "compile_token":
            turn.compile_stream = evt["full"]
        elif kind == "spec":
            turn.spec_json = evt["spec_json"]
            turn.tasks = [TaskTrace(task_id=t["id"], name=t["name"])
                          for t in evt["tasks"]]
        elif kind == "agent_token":
            for t in turn.tasks:
                if t.task_id == evt["task_id"]:
                    t.status = "running"
                    t.streaming_text = evt["full"]
                    break
        elif kind == "task_done":
            for t in turn.tasks:
                if t.task_id == evt["task_id"]:
                    t.status = evt["status"]
                    t.output = evt["output"]
                    t.elapsed = evt["elapsed"]
                    t.tool_calls = evt["tool_calls"]
                    t.tool_events = evt.get("tool_summaries", [])
                    t.error = evt["error"]
                    break
        elif kind == "verdicts":
            turn.verdicts = evt["verdicts"]
        elif kind == "error":
            turn.error = evt["message"]
            turn.stage = "error"
    # Check HITL queue: if user_proxy agent is waiting for input, surface it.
    hitl_q = handle.get("hitl_questions")
    if hitl_q is not None and turn.hitl_request is None:
        try:
            req = hitl_q.get_nowait()
            turn.hitl_request = req
            turn.stage = "hitl_prompt"
            changed = True
        except queue.Empty:
            pass
    return changed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_STAGES = ("compile", "execute", "critic", "done")
_STAGE_LABEL = {"compile": "Planning", "execute": "Executing",
                "critic": "Reviewing", "done": "Done",
                "hitl_prompt": "Waiting for input"}


def _render_stage_bar(current: str):
    """Flight-deck progress bar. The accent fill animates across as stages advance."""
    current_idx = _STAGES.index(current) if current in _STAGES else -1
    # Fill % = (current_idx + 1) / len(_STAGES). "done" fills 100%.
    if current == "done":
        progress = 100
    elif current == "error":
        # Fill up to the stage that failed, in muted tone.
        progress = int((max(current_idx, 0) + 0.5) * 100 / len(_STAGES))
    elif current == "hitl_prompt":
        # Treat as paused mid-execute.
        progress = int(1.5 * 100 / len(_STAGES))
    else:
        progress = int((current_idx + 0.5) * 100 / len(_STAGES)) if current_idx >= 0 else 0

    segments = []
    for i, s in enumerate(_STAGES):
        if current == "done":
            cls = "past"
        elif current_idx > i:
            cls = "past"
        elif current_idx == i and current != "done":
            cls = "active"
        else:
            cls = ""
        segments.append(
            f'<div class="stage-seg {cls}">'
            f'<span class="stage-seg-num">0{i+1}</span>'
            f'<span class="stage-seg-label">{_STAGE_LABEL[s]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="stage-bar" style="--progress: {progress}%">'
        + "".join(segments) +
        '</div>',
        unsafe_allow_html=True,
    )


def _render_task(task: TaskTrace, index: int):
    """Flight-deck status row: numbered entry, title, live metadata, status."""
    status_label = {
        "pending": "AWAITING",
        "running": "IN FLIGHT",
        "success": "COMPLETE",
        "failure": "FAILED",
    }.get(task.status, task.status.upper())

    elapsed_str = f"{task.elapsed:.1f}s" if task.elapsed else "—"
    tools_str = f"{task.tool_calls} calls" if task.tool_calls else "no tools"

    st.markdown(
        f'<div class="task-row {task.status}" style="animation-delay: {index * 60}ms">'
        f'  <div class="task-num">{index + 1:02d}</div>'
        f'  <div>'
        f'    <div class="task-title">{_esc(task.name)}</div>'
        f'    <div class="task-meta">'
        f'      <strong>{task.task_id}</strong> &nbsp;·&nbsp; '
        f'      {elapsed_str} &nbsp;·&nbsp; {tools_str}'
        f'    </div>'
        f'  </div>'
        f'  <div class="task-status {task.status}">'
        f'    <span class="dot"></span>{status_label}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Live streaming body during run — inline code frame, editorial-scoped.
    if task.status == "running" and task.streaming_text:
        preview = task.streaming_text[-1000:]
        st.markdown(
            f'<div class="compile-frame" style="margin-top:-.2rem">'
            f'<div class="compile-frame-label">Streaming</div>'
            f'<pre style="margin:0; white-space:pre-wrap; '
            f'font-family:inherit; color:inherit;">{_esc(preview)}</pre>'
            f'</div>', unsafe_allow_html=True,
        )

    # Tool call log — only shown if the task used tools.
    if task.tool_events:
        with st.expander(f"Tool transcript · {len(task.tool_events)} call{'s' if len(task.tool_events) != 1 else ''}", expanded=False):
            for te in task.tool_events:
                st.code(te, language="text")

    # Final output (compact expander; the hero answer is rendered separately).
    if task.status == "success" and task.output is not None:
        out = task.output
        with st.expander("Task output", expanded=False):
            if isinstance(out, (dict, list)):
                st.json(out)
            else:
                st.markdown(str(out)[:3000])
    elif task.status == "failure" and task.error:
        st.markdown(
            f'<div style="padding:.5rem .8rem;border-left:2px solid {C["fail"]};'
            f'background:#FDF5F4;font-family:\'JetBrains Mono\',monospace;'
            f'font-size:.72rem;color:{C["fail"]};margin:.4rem 0;">'
            f'{_esc(task.error[:500])}</div>',
            unsafe_allow_html=True,
        )


def _esc(text) -> str:
    """HTML-escape helper for editorial-styled rendering."""
    import html
    return html.escape(str(text))


def _render_hitl_form(turn: Turn) -> bool:
    """Editorial callout for a pending HITL prompt. Returns answer on submit."""
    req = turn.hitl_request
    if req is None:
        return False
    prompt_text = getattr(req, "prompt", str(req))
    hint = getattr(req, "hint", None)
    choices = getattr(req, "choices", None)
    agent_id = getattr(req, "agent_id", "agent")

    st.markdown(
        f'<div class="hitl-callout">'
        f'<div class="hitl-callout-label">Query &mdash; from {_esc(agent_id)}</div>'
        f'<div style="font-family:\'Fraunces\',serif;font-size:1.05rem;'
        f'line-height:1.35;color:{C["ink"]};font-weight:400;">'
        f'{_esc(prompt_text)}</div>'
        + (f'<div style="font-family:\'IBM Plex Sans\',sans-serif;'
           f'font-size:.78rem;color:{C["ink_2"]};margin-top:.4rem;'
           f'font-style:italic;">{_esc(hint)}</div>' if hint else '') +
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.form(f"hitl_{id(req)}", clear_on_submit=True):
        choice_val = None
        if choices:
            choice_val = st.radio("Pick one:", choices, horizontal=True,
                                  key=f"hitl_r_{id(req)}")
        freeform = st.text_area("Your answer:", key=f"hitl_t_{id(req)}",
                                height=100, label_visibility="collapsed",
                                placeholder="Type your response…")
        submitted = st.form_submit_button("Respond", use_container_width=True)
    if submitted:
        answer = (freeform or "").strip() or (choice_val or "")
        return answer  # type: ignore[return-value]
    return False


def _render_turn(turn: Turn, is_live: bool = False):
    # User prompt: editorial quote block with oversized drop-cap.
    if turn.role == "user":
        st.markdown(
            f'<div class="prompt-block">'
            f'<div class="prompt-label">You &nbsp;&mdash;&nbsp; Prompt</div>'
            f'<div class="prompt-text">{_esc(turn.content)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Assistant turn — full editorial treatment.
    # Right-side byline shows elapsed time and agent identity.
    elapsed_hint = (
        f"Elapsed {turn.elapsed:.1f}s" if turn.elapsed else "Working"
    )
    st.markdown(
        f'<div class="response-block">'
        f'<div class="response-label">'
        f'<span>DAAW &nbsp;&mdash;&nbsp; Response</span>'
        f'<span class="response-label-right">{elapsed_hint}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    _render_stage_bar(turn.stage)

    if turn.error:
        st.markdown(
            f'<div style="padding:1rem;border:1px solid {C["fail"]};'
            f'background:#FDF5F4;font-family:\'Fraunces\',serif;'
            f'color:{C["fail"]};font-style:italic;">'
            f'{_esc(turn.error)}</div>',
            unsafe_allow_html=True,
        )
        return

    # ---- Compile phase ----
    if turn.stage == "compile":
        if turn.compile_stream:
            rendered, lang = prettify_partial_json(turn.compile_stream)
            st.markdown(
                f'<div class="compile-frame">'
                f'<div class="compile-frame-label">Plan &mdash; live draft</div>'
                f'<pre style="margin:0; white-space:pre-wrap; '
                f'font-family:inherit; color:inherit;">{_esc(rendered)}</pre>'
                f'</div>', unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-whisper">Drafting the workflow&hellip;</div>',
                unsafe_allow_html=True,
            )

    # ---- Execute phase ----
    if turn.stage == "execute" and turn.tasks:
        done_count = sum(1 for t in turn.tasks
                         if t.status in ("success", "failure"))
        st.markdown(
            f'<div class="status-whisper">'
            f'Executing &nbsp;<span style="font-family:\'JetBrains Mono\',monospace;'
            f'color:{C["ink_2"]};font-style:normal;">'
            f'{done_count:02d} &nbsp;/&nbsp; {len(turn.tasks):02d}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # ---- HITL paused ----
    if turn.stage == "hitl_prompt":
        st.markdown(
            '<div class="status-whisper">Paused &mdash; your input is needed below.</div>',
            unsafe_allow_html=True,
        )

    # ---- Task list — always shown once spec is known ----
    if turn.tasks:
        st.markdown(
            '<div class="label" style="margin:1.2rem 0 .2rem;">Tasks</div>',
            unsafe_allow_html=True,
        )
        for i, t in enumerate(turn.tasks):
            _render_task(t, i)

    # ---- Critic phase ----
    if turn.stage == "critic":
        st.markdown(
            '<div class="status-whisper">Reviewing outputs against success criteria&hellip;</div>',
            unsafe_allow_html=True,
        )

    # ---- Final answer (editorial body) ----
    if turn.stage == "done":
        final_task = next(
            (t for t in reversed(turn.tasks) if t.status == "success"
             and t.output is not None),
            None,
        )
        if final_task is not None:
            out = final_task.output
            st.markdown(
                '<div class="label label-accent" style="margin-top:1.5rem;">'
                'Answer</div>',
                unsafe_allow_html=True,
            )
            if isinstance(out, (dict, list)):
                st.markdown('<div class="final-answer">',
                            unsafe_allow_html=True)
                st.json(out)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                text = str(out)[:4000]
                st.markdown(
                    f'<div class="final-answer">{text}</div>',
                    unsafe_allow_html=True,
                )
            turn.content = str(out)[:4000]

        # Cached sandbox listing.
        if not turn.output_files:
            import os
            sandbox = os.environ.get(
                "DAAW_SANDBOX_DIR",
                os.path.join(os.getcwd(), ".daaw_sandbox"),
            )
            if os.path.isdir(sandbox):
                try:
                    turn.output_files = sorted(os.listdir(sandbox))[-5:]
                except OSError:
                    turn.output_files = []
        if turn.output_files:
            files_html = " &middot; ".join(
                f'<strong>{_esc(f)}</strong>' for f in turn.output_files
            )
            st.markdown(
                f'<div class="files-saved">'
                f'Files written &nbsp;→&nbsp; {files_html}</div>',
                unsafe_allow_html=True,
            )

        # Details — verdicts + spec (editorial treatment).
        with st.expander("Details", expanded=False):
            if turn.verdicts:
                st.markdown(
                    '<div class="label" style="margin-bottom:.5rem;">Verdicts</div>',
                    unsafe_allow_html=True,
                )
                for v in turn.verdicts:
                    cls = "pass" if v["verdict"] == "PASS" else "fail"
                    reasoning = v.get("reasoning", "")[:200]
                    st.markdown(
                        f'<div class="verdict-row">'
                        f'<div class="verdict-tag {cls}">{v["verdict"]}</div>'
                        f'<div class="verdict-body">'
                        f'<code>{_esc(v["task_id"])}</code> '
                        f'{_esc(v["task_name"])} &nbsp;&mdash;&nbsp; '
                        f'<em>{_esc(reasoning)}</em>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            if turn.spec_json:
                st.markdown(
                    '<div class="label" style="margin:1rem 0 .5rem;">'
                    'WorkflowSpec</div>',
                    unsafe_allow_html=True,
                )
                st.json(json.loads(turn.spec_json))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    # Editorial masthead — like a magazine issue number.
    from datetime import datetime as _dt
    today = _dt.now().strftime("%d.%m.%Y")
    st.markdown(
        f'<div class="masthead">'
        f'  <div class="masthead-title">DAAW <em>chat</em></div>'
        f'  <div class="masthead-meta">'
        f'    <strong>v{_VER}</strong> &nbsp;&middot;&nbsp; {_BT}<br>'
        f'    Issue &nbsp; {today}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: 4 simple modes
    #
    # Each mode defines: compile provider+model, execute provider+model,
    # and whether execution is local (sequential) or cloud (parallel).
    _MODES = {
        "Groq (Cloud, Free)": {
            "desc": "Fast cloud inference. Groq free tier.",
            "compile_provider": "groq",
            "compile_model": "llama-3.3-70b-versatile",
            "exec_provider": "groq",
            "exec_model": "llama-3.1-8b-instant",
            "is_local": False,
        },
        "Local (Ollama)": {
            "desc": "Fully offline. Runs on your GPU.",
            "compile_provider": "gateway",
            "compile_model": "gemma4:e2b-it-q4_K_M",
            "exec_provider": "gateway",
            "exec_model": "gemma4:e2b-it-q4_K_M",
            "is_local": True,
            "gateway_url": "http://localhost:11434/v1",
        },
        "Claude + Groq (Recommended)": {
            "desc": "Claude plans, Groq executes. Best quality + speed.",
            "compile_provider": "claude_api",
            "compile_model": "claude-sonnet-4-5-20250929",
            "exec_provider": "groq",
            "exec_model": "llama-3.1-8b-instant",
            "is_local": False,
        },
        "Claude (Full)": {
            "desc": "Claude for everything. Highest quality, slowest.",
            "compile_provider": "claude_api",
            "compile_model": "claude-sonnet-4-5-20250929",
            "exec_provider": "claude_api",
            "exec_model": "claude-sonnet-4-5-20250929",
            "is_local": False,
        },
    }
    _MODE_NAMES = list(_MODES.keys())

    with st.sidebar:
        st.caption("Mode")
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = _MODE_NAMES[2]  # default: Claude + Groq
        st.session_state.chat_mode = st.radio(
            "Select mode",
            _MODE_NAMES,
            index=_MODE_NAMES.index(st.session_state.chat_mode)
            if st.session_state.chat_mode in _MODE_NAMES else 2,
            label_visibility="collapsed",
        )
        mode = _MODES[st.session_state.chat_mode]
        st.caption(mode["desc"])

        st.divider()

        # Manual override: separate planner and executor model selection
        with st.expander("Advanced: Manual model override", expanded=False):
            _all_providers = ["groq", "gemini", "openai", "anthropic",
                              "gateway", "claude_api"]
            _model_hints = {
                "groq": "llama-3.3-70b-versatile",
                "gemini": "gemini-2.5-flash-lite",
                "openai": "gpt-4.1-mini",
                "anthropic": "claude-sonnet-4-6",
                "gateway": "gemma4:e2b-it-q4_K_M",
                "claude_api": "claude-sonnet-4-5-20250929",
            }
            st.caption("Planner (compile)")
            manual_cp = st.selectbox(
                "Compile provider", _all_providers,
                index=_all_providers.index(mode["compile_provider"])
                if mode["compile_provider"] in _all_providers else 0,
                key="manual_cp",
            )
            manual_cm = st.text_input(
                "Compile model",
                value=mode["compile_model"],
                placeholder=_model_hints.get(manual_cp, ""),
                key="manual_cm",
            )
            st.caption("Executor (tasks)")
            manual_ep = st.selectbox(
                "Execute provider", _all_providers,
                index=_all_providers.index(mode["exec_provider"])
                if mode["exec_provider"] in _all_providers else 0,
                key="manual_ep",
            )
            manual_em = st.text_input(
                "Execute model",
                value=mode["exec_model"],
                placeholder=_model_hints.get(manual_ep, ""),
                key="manual_em",
            )
            # Override mode with manual selections if changed
            mode = dict(mode)  # copy so we don't mutate the preset
            mode["compile_provider"] = manual_cp
            mode["compile_model"] = manual_cm or _model_hints.get(manual_cp, "")
            mode["exec_provider"] = manual_ep
            mode["exec_model"] = manual_em or _model_hints.get(manual_ep, "")

        # Show resolved config
        st.markdown(
            f'<div style="font-family:JetBrains Mono;font-size:.7rem;'
            f'color:{C["muted"]};line-height:1.6">'
            f'Compile: <strong style="color:{C["accent"]}">'
            f'{mode["compile_provider"]}</strong> / {mode["compile_model"]}<br>'
            f'Execute: <strong style="color:{C["accent"]}">'
            f'{mode["exec_provider"]}</strong> / {mode["exec_model"]}'
            f'</div>', unsafe_allow_html=True,
        )

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_turns = []
            st.session_state.chat_handle = None
            st.rerun()

        # Stash resolved provider/model in session_state so the input
        # handler can read them without re-parsing the mode dict.
        st.session_state.chat_provider = mode["exec_provider"]
        st.session_state.chat_model = mode["exec_model"]
        st.session_state._compile_provider = mode["compile_provider"]
        st.session_state._compile_model = mode["compile_model"]

    # History
    for t in st.session_state.chat_turns[:-1]:
        _render_turn(t)

    # Current / in-flight turn (last, may be live)
    live_turn = (
        st.session_state.chat_turns[-1]
        if st.session_state.chat_turns else None
    )
    if live_turn is not None:
        _render_turn(live_turn, is_live=(live_turn.stage not in ("done", "error")))

    # Drive the pipeline if one is running.
    handle = st.session_state.chat_handle
    if handle is not None:
        if "error" in handle and handle.get("events") is None:
            live_turn.error = handle["error"]  # type: ignore[union-attr]
            live_turn.stage = "error"  # type: ignore[union-attr]
            st.session_state.chat_handle = None
            st.rerun()
        changed = _drain_events(handle, live_turn)  # type: ignore[arg-type]

        # HITL: if a user_proxy task is waiting for input, render a form.
        if live_turn and live_turn.stage == "hitl_prompt":
            answer = _render_hitl_form(live_turn)
            if answer:
                handle["hitl_answers"].put(str(answer))
                live_turn.hitl_request = None
                live_turn.stage = "execute"  # resume
                time.sleep(0.05)
                st.rerun()
            # Don't auto-rerun while waiting for HITL — wait for form submit.
        elif live_turn.stage in ("done", "error"):  # type: ignore[union-attr]
            st.session_state.chat_handle = None
            if changed:
                st.rerun()
        else:
            # Poll for more events. If anything actually changed this
            # cycle, rerun immediately so the UI reflects the update.
            # Otherwise wait a beat — avoids hammering CPU at 4 Hz with
            # zero data changes during long LLM calls.
            if changed:
                st.rerun()
            else:
                time.sleep(0.5)
                st.rerun()

    # Input
    prompt = st.chat_input("What would you like DAAW to do?")
    if prompt:
        st.session_state.chat_turns.append(Turn(role="user", content=prompt))
        assistant = Turn(role="assistant", stage="compile")
        st.session_state.chat_turns.append(assistant)
        handle = _start_pipeline(
            prompt,
            st.session_state.chat_provider,
            st.session_state.chat_model or None,
            compile_provider=st.session_state.get("_compile_provider"),
            compile_model=st.session_state.get("_compile_model"),
        )
        if handle and "error" in handle:
            assistant.error = handle["error"]
            assistant.stage = "error"
            st.session_state.chat_handle = None
        else:
            st.session_state.chat_handle = handle
        st.rerun()


if __name__ == "__main__":
    main()
