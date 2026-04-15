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
# Styling
# ---------------------------------------------------------------------------
C = {
    "bg": "#0B0D12",
    "card": "#14171F",
    "border": "#232836",
    "text": "#E6E8EC",
    "muted": "#8D94A3",
    "accent": "#60A5FA",      # blue for assistant
    "user": "#A78BFA",        # violet for user
    "success": "#34D399",
    "warn": "#FBBF24",
    "danger": "#F87171",
}

_CSS = f"""
<style>
  .stApp {{ background: {C['bg']}; color: {C['text']}; }}
  .block-container {{ padding-top: 1.2rem; max-width: 880px; }}
  .hdr {{ display:flex; align-items:center; justify-content:space-between;
         padding:.25rem 0 1rem; border-bottom: 1px solid {C['border']};
         margin-bottom: 1rem; }}
  .hdr-title {{ font-family: 'Sora', sans-serif; font-size: 1.1rem;
               font-weight:700; color:{C['text']}; }}
  .hdr-ver {{ font-family: 'JetBrains Mono', monospace; font-size:.72rem;
             color:{C['muted']}; }}
  .stage-pill {{ display:inline-block; font-family:'JetBrains Mono',
                monospace; font-size:.68rem; padding:.1rem .45rem;
                border-radius:4px; margin-right:.4rem;
                background:{C['border']}; color:{C['muted']}; }}
  .stage-pill.active {{ color:{C['accent']}; background:{C['accent']}22; }}
  .stage-pill.ok {{ color:{C['success']}; background:{C['success']}22; }}
  .stage-pill.fail {{ color:{C['danger']}; background:{C['danger']}22; }}
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


def _render_stage_pills(current: str):
    bits = []
    current_idx = _STAGES.index(current) if current in _STAGES else -1
    for i, s in enumerate(_STAGES):
        cls = "ok" if (current_idx > i or s == "done" and current == "done") else ""
        if s == current and current != "done":
            cls = "active"
        if current == "error":
            cls = "fail" if i <= max(0, current_idx) else ""
        bits.append(
            f'<span class="stage-pill {cls}">{_STAGE_LABEL[s]}</span>'
        )
    st.markdown(" ".join(bits), unsafe_allow_html=True)


def _render_task(task: TaskTrace):
    icon = {
        "pending": "·", "running": "●",
        "success": "✓", "failure": "✗",
    }.get(task.status, "·")
    color = {
        "pending": C["muted"], "running": C["accent"],
        "success": C["success"], "failure": C["danger"],
    }.get(task.status, C["muted"])
    tc = f" · {task.tool_calls} tool calls" if task.tool_calls else ""
    elapsed = f" · {task.elapsed:.1f}s" if task.elapsed else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.5rem;'
        f'margin:.3rem 0;padding:.3rem .5rem;background:{C["card"]};'
        f'border-radius:6px;border-left:3px solid {color}">'
        f'<span style="color:{color};font-weight:700;font-size:1rem">'
        f'{icon}</span>'
        f'<div><strong style="color:{C["text"]}">{task.name}</strong>'
        f'<span style="color:{C["muted"]};font-size:.75rem;margin-left:.5rem">'
        f'{task.task_id}{elapsed}{tc}</span></div>'
        f'</div>', unsafe_allow_html=True,
    )
    # Live streaming body during run.
    if task.status == "running" and task.streaming_text:
        st.code(task.streaming_text[-1200:], language="text")
    elif task.status == "running":
        st.caption("Running...")
    # Tool call log (shown for completed tasks that used tools).
    if task.tool_events:
        with st.expander(f"Tool calls ({len(task.tool_events)})", expanded=False):
            for te in task.tool_events:
                st.code(te, language="text")
    # Final output.
    if task.status == "success" and task.output is not None:
        out = task.output
        if isinstance(out, (dict, list)):
            with st.expander("Output", expanded=False):
                st.json(out)
        else:
            with st.expander("Output", expanded=False):
                st.markdown(str(out)[:3000])
    elif task.status == "failure" and task.error:
        st.error(task.error[:500])


def _render_hitl_form(turn: Turn) -> bool:
    """Render an inline HITL prompt form. Returns True on submit."""
    req = turn.hitl_request
    if req is None:
        return False
    prompt_text = getattr(req, "prompt", str(req))
    hint = getattr(req, "hint", None)
    choices = getattr(req, "choices", None)
    agent_id = getattr(req, "agent_id", "agent")

    st.markdown(
        f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
        f'border-left:4px solid {C["warn"]};border-radius:8px;'
        f'padding:.8rem 1rem;margin:.5rem 0">'
        f'<strong style="color:{C["warn"]}">Input needed</strong>'
        f' <span style="color:{C["muted"]};font-size:.75rem">'
        f'from {agent_id}</span></div>',
        unsafe_allow_html=True,
    )
    with st.form(f"hitl_{id(req)}", clear_on_submit=True):
        st.markdown(f"**{prompt_text}**")
        if hint:
            st.caption(hint)
        choice_val = None
        if choices:
            choice_val = st.radio("Pick one:", choices, horizontal=True,
                                  key=f"hitl_r_{id(req)}")
        freeform = st.text_area("Your answer:", key=f"hitl_t_{id(req)}",
                                height=100)
        submitted = st.form_submit_button("Submit", use_container_width=True)
    if submitted:
        answer = (freeform or "").strip() or (choice_val or "")
        return answer  # type: ignore[return-value]
    return False


def _render_turn(turn: Turn, is_live: bool = False):
    if turn.role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn.content)
        return

    with st.chat_message("assistant", avatar="💬"):
        _render_stage_pills(turn.stage)

        if turn.error:
            st.error(turn.error)
            return

        # Live compile view
        if turn.stage == "compile" and turn.compile_stream:
            rendered, lang = prettify_partial_json(turn.compile_stream)
            st.caption("Planning the workflow...")
            st.code(rendered, language=lang)
        elif turn.stage == "compile":
            st.caption("Compiling plan...")

        # Execution progress header
        if turn.stage == "execute" and turn.tasks:
            done_count = sum(1 for t in turn.tasks
                            if t.status in ("success", "failure"))
            st.caption(f"Executing tasks... ({done_count}/{len(turn.tasks)} complete)")

        # HITL prompt
        if turn.stage == "hitl_prompt":
            # Rendered in the main loop, not here — just show status.
            st.caption("Waiting for your input...")

        # Task list — always shown once spec is known.
        if turn.tasks:
            for t in turn.tasks:
                _render_task(t)

        # Critic phase indicator
        if turn.stage == "critic":
            st.caption("Evaluating results...")

        # Final synthesis: the last task's output is the headline answer.
        if turn.stage == "done":
            final_task = next(
                (t for t in reversed(turn.tasks) if t.status == "success"
                 and t.output is not None),
                None,
            )
            if final_task is not None:
                st.markdown("---")
                st.markdown("**Answer**")
                out = final_task.output
                if isinstance(out, (dict, list)):
                    st.json(out)
                else:
                    st.markdown(str(out)[:4000])
                turn.content = str(out)[:4000]

            # File output hint — cached on the Turn so we don't re-scan
            # the sandbox on every Streamlit rerun.
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
                st.caption(
                    "Files saved: "
                    + ", ".join(f"`{f}`" for f in turn.output_files)
                )

            # Details expander: spec + verdicts + per-task outputs.
            with st.expander("Details"):
                if turn.verdicts:
                    st.markdown("**Critic verdicts**")
                    for v in turn.verdicts:
                        color = (C["success"] if v["verdict"] == "PASS"
                                 else C["danger"])
                        st.markdown(
                            f'<span style="color:{color}">'
                            f'{v["verdict"]}</span> · `{v["task_id"]}` '
                            f'{v["task_name"]} — {v.get("reasoning", "")[:200]}',
                            unsafe_allow_html=True,
                        )
                if turn.spec_json:
                    st.markdown("**WorkflowSpec**")
                    st.json(json.loads(turn.spec_json))
                st.caption(f"Total elapsed: {turn.elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    # Header
    st.markdown(
        f'<div class="hdr">'
        f'<div class="hdr-title">DAAW · Chat</div>'
        f'<div class="hdr-ver">v{_VER} · {_BT}</div>'
        f'</div>', unsafe_allow_html=True,
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
