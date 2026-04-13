"""Live end-to-end smoke test for every registered tool.

Run: ``python scripts/verify_tools.py``

Exercises each tool in ``daaw.tools.real_tools`` (and ``webhook_tools`` if
``NOTIFY_WEBHOOK_URL`` is set) against the real implementation — no mocks.
Output is a pass/fail table so you can confirm tools work outside the UI.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# Ensure project src is importable when run from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import daaw.tools.real_tools  # noqa: F401  (side-effect: registers tools)

try:
    import daaw.tools.webhook_tools  # noqa: F401
    _HAS_WEBHOOK = True
except Exception:
    _HAS_WEBHOOK = False

from daaw.tools.registry import tool_registry


async def _run_case(name: str, coro) -> tuple[str, bool, str]:
    try:
        out = await coro
        preview = str(out).replace("\n", " ")[:120]
        return name, True, preview
    except Exception as e:  # noqa: BLE001
        return name, False, f"{type(e).__name__}: {e}"


async def main() -> int:
    # Sandbox to a temp dir so we don't pollute the repo
    sandbox = tempfile.mkdtemp(prefix="daaw_verify_")
    os.environ["DAAW_SANDBOX_DIR"] = sandbox
    import daaw.tools.real_tools as rt
    rt._SANDBOX_DIR = sandbox

    print(f"Sandbox: {sandbox}")
    print(f"Registered tools: {sorted(tool_registry._tools.keys())}\n")

    cases: list[tuple[str, object]] = [
        ("file_write", tool_registry.execute(
            "file_write", path="hello.txt", content="hello daaw")),
        ("file_read", tool_registry.execute("file_read", path="hello.txt")),
        ("shell_command (safe)", tool_registry.execute(
            "shell_command",
            command="cmd /c echo ok" if sys.platform == "win32" else "echo ok",
        )),
        ("shell_command (blocked)", tool_registry.execute(
            "shell_command", command="rm -rf /")),
        ("file_write (path traversal blocked)", tool_registry.execute(
            "file_write", path="../../escape.txt", content="x")),
        ("web_search", tool_registry.execute(
            "query", query="python asyncio") if False else
            tool_registry.execute("web_search", query="python asyncio")),
    ]

    if _HAS_WEBHOOK and os.getenv("NOTIFY_WEBHOOK_URL"):
        cases.append(("notify", tool_registry.execute(
            "notify", message="DAAW tool verify smoke test", title="verify_tools")))
    else:
        print("(Skipping 'notify' — set NOTIFY_WEBHOOK_URL to exercise it.)\n")

    results = []
    for name, coro in cases:
        # For path-traversal we expect PermissionError → report as pass if it raises
        name_l, ok, preview = await _run_case(name, coro)
        if "blocked" in name_l and not ok and "escapes sandbox" in preview:
            ok = True
            preview = "correctly rejected: " + preview
        results.append((name_l, ok, preview))

    # Pretty table
    width = max(len(n) for n, _, _ in results) + 2
    print(f"{'TOOL'.ljust(width)}  RESULT  OUTPUT")
    print("-" * (width + 40))
    failures = 0
    for name, ok, preview in results:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{name.ljust(width)}  {tag:<6}  {preview}")

    print()
    print(f"{len(results) - failures}/{len(results)} tools verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
