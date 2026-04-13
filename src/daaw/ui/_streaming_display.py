"""Streaming-display helpers shared by every DAAW Streamlit UI.

Providers like Ollama stream JSON with no whitespace, so rendering the raw
stream in ``st.code(..., language='json')`` produces one very long line.
``prettify_partial_json`` incrementally reformats the accumulator so the
user sees a structured, readable view even mid-stream.
"""

from __future__ import annotations

import json
import re

# Repair hooks reused from the compiler. Importing lazily avoids a cycle
# since compiler.py depends on LLM stack which depends on this module's
# neighbours, but compiler.py doesn't import ui.
_REPAIR_PATTERNS = [
    (re.compile(r",\s*(\}|\])"), r"\1"),           # trailing comma before } / ]
    (re.compile(r"```json\s*"), ""),                # markdown fence open
    (re.compile(r"```\s*$", re.MULTILINE), ""),     # markdown fence close
]


def _best_effort_repair(text: str) -> str:
    """Cheap syntactic fixes that often let a partial JSON stream parse.

    Closes unbalanced brackets in the correct nesting order by scanning
    the text and tracking the open-bracket stack (outside string literals).
    """
    out = text
    for pat, repl in _REPAIR_PATTERNS:
        out = pat.sub(repl, out)

    # Stack-based bracket balancer. Ignores chars inside string literals so
    # a `{` that appears inside "..." doesn't corrupt the stack.
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in out:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or
                          (ch == "]" and stack[-1] == "[")):
                stack.pop()

    # If we ended inside a string, close it first.
    if in_string:
        out = out + '"'

    # Close remaining opens in reverse order (innermost first).
    for opener in reversed(stack):
        out = out + ("}" if opener == "{" else "]")

    return out


def prettify_partial_json(accumulator: str, *, max_chars: int = 2400) -> tuple[str, str]:
    """Try to pretty-print the accumulator as JSON; fall back to raw text.

    Returns ``(rendered_text, language_hint)``. The language hint is either
    ``"json"`` (safe to render with JSON highlighting) or ``"text"``
    (unparseable or not yet valid — the raw stream is shown with word wrap
    so the user still sees structure via line breaks we inject at natural
    boundaries).
    """
    if not accumulator:
        return "", "text"

    # Tail-slice to keep the live box from growing unbounded. Slice BEFORE
    # pretty-printing so we don't re-parse a huge buffer each token.
    tail = accumulator if len(accumulator) <= max_chars else accumulator[-max_chars:]

    # 1. Happy path: maybe the stream is already valid JSON.
    candidate = _best_effort_repair(tail)
    try:
        data = json.loads(candidate)
        return json.dumps(data, indent=2, ensure_ascii=False), "json"
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Fallback: split on commas outside strings so the user sees line
    #    breaks even though the JSON is mid-flight and unparseable.
    #    This is deliberately simple — getting JSON lexing right at every
    #    byte is more work than it's worth for a preview.
    soft = _soft_break(tail)
    return soft, "text"


# Rough tokens that deserve a break-before marker. These are common JSON
# field separators; inserting a newline before them gives a readable preview.
_BREAK_BEFORE = re.compile(r'(,\s*)"')


def _soft_break(text: str) -> str:
    """Insert a newline before every ``,\"`` pair at the top level.

    We don't track quote/escape state, so this may over-break inside
    long string values — acceptable for a mid-stream preview where
    readability beats fidelity.
    """
    return _BREAK_BEFORE.sub(',\n"', text)


__all__ = ["prettify_partial_json"]
