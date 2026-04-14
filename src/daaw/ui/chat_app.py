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


@dataclass
class Turn:
    role: str                          # "user" | "assistant"
    content: str = ""                  # final text answer
    stage: str = "idle"                # compile / execute / critic / done / error
    compile_stream: str = ""           # live JSON during compile
    tasks: list[TaskTrace] = field(default_factory=list)
    error: str = ""
    spec_json: str = ""
    verdicts: list[dict] = field(default_factory=list)
    elapsed: float = 0.0


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
def _start_pipeline(goal: str, provider: str, model: str | None) -> dict | None:
    """Kick off compile -> execute -> critic on a background thread.

    All streaming callbacks write into a single thread-safe queue that the
    Streamlit render loop drains each rerun; no direct st.* calls from the
    worker thread (Streamlit isn't thread-safe for widgets).
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

        store = ArtifactStore(config.artifact_store_dir)
        cb = CircuitBreaker(threshold=config.circuit_breaker_threshold)
        factory = AgentFactory(
            llm, store,
            default_provider=provider, default_model=model,
            on_agent_token=_on_agent_token,
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
                    compiler = Compiler(llm, config, provider=provider, model=model)
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

                    # Emit per-task final status.
                    for tid, r in results.items():
                        _send("task_done",
                              task_id=tid,
                              status=r.agent_result.status,
                              output=r.agent_result.output,
                              elapsed=r.elapsed_seconds,
                              tool_calls=len(
                                  (r.agent_result.metadata or {})
                                  .get("tool_calls", []) or []),
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
                "started_at": time.monotonic()}
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
                    t.error = evt["error"]
                    break
        elif kind == "verdicts":
            turn.verdicts = evt["verdicts"]
        elif kind == "error":
            turn.error = evt["message"]
            turn.stage = "error"
    return changed


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_STAGES = ("compile", "execute", "critic", "done")
_STAGE_LABEL = {"compile": "Planning", "execute": "Executing",
                "critic": "Reviewing", "done": "Done"}


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
        f'margin:.3rem 0"><span style="color:{color};font-weight:700">'
        f'{icon}</span><strong style="color:{C["text"]}">{task.name}</strong>'
        f'<span style="color:{C["muted"]};font-size:.78rem">{elapsed}{tc}</span>'
        f'</div>', unsafe_allow_html=True,
    )
    # Live streaming body during run; final output when done.
    if task.status == "running" and task.streaming_text:
        st.code(task.streaming_text[-1500:], language="text")
    elif task.status == "success" and task.output is not None:
        out = task.output
        if isinstance(out, (dict, list)):
            with st.expander("Output", expanded=False):
                st.json(out)
        else:
            with st.expander("Output", expanded=False):
                st.markdown(str(out))
    elif task.status == "failure" and task.error:
        st.error(task.error[:500])


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

        # Task list — always shown once spec is known.
        if turn.tasks:
            for t in turn.tasks:
                _render_task(t)

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
                    st.markdown(str(out))
                turn.content = str(out)[:4000]

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

    # Sidebar: provider / model (kept tiny — not the focus)
    with st.sidebar:
        st.caption("Settings")
        providers = ["groq", "gemini", "openai", "anthropic", "gateway", "claude_api"]
        st.session_state.chat_provider = st.selectbox(
            "Provider", providers,
            index=providers.index(st.session_state.chat_provider)
            if st.session_state.chat_provider in providers else 0,
        )
        _default_models = {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.5-flash-lite",
            "openai": "gpt-4.1-mini",
            "anthropic": "claude-sonnet-4-6",
            "gateway": "gemma4:e2b-it-q4_K_M",
            "claude_api": "claude-sonnet-4-5-20250929",
        }
        st.session_state.chat_model = st.text_input(
            "Model",
            value=st.session_state.chat_model
            or _default_models.get(st.session_state.chat_provider, ""),
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_turns = []
            st.session_state.chat_handle = None
            st.rerun()

        # Show split-provider info if configured.
        from daaw.config import get_config as _gc
        _cfg = _gc()
        if _cfg.compiler_provider:
            st.caption(
                f"Compile: **{_cfg.compiler_provider}** / "
                f"{_cfg.compiler_model or 'default'}  \n"
                f"Execute: **{st.session_state.chat_provider}** / "
                f"{st.session_state.chat_model or 'default'}"
            )

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
        if live_turn.stage in ("done", "error"):  # type: ignore[union-attr]
            st.session_state.chat_handle = None
            if changed:
                st.rerun()
        else:
            # Keep pumping the queue on a timer.
            time.sleep(0.25)
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
