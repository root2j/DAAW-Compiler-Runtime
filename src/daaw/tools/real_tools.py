"""Real tool implementations for production use."""

from __future__ import annotations

import os
from pathlib import Path

from daaw.tools.registry import tool_registry

# Sandboxed base directory for file operations
_SANDBOX_DIR = os.environ.get("DAAW_SANDBOX_DIR", os.path.join(os.getcwd(), ".daaw_sandbox"))


def _resolve_sandboxed(path: str) -> Path:
    """Resolve a path within the sandbox directory. Prevents directory traversal."""
    sandbox = Path(_SANDBOX_DIR).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    resolved = (sandbox / path).resolve()
    if not str(resolved).startswith(str(sandbox)):
        raise PermissionError(f"Path escapes sandbox: {path}")
    return resolved


@tool_registry.register(
    name="web_search",
    description="Search the web for information using DuckDuckGo",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def real_web_search(query: str) -> str:
    import httpx

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.post(url, data={"q": query}, headers=headers)
        resp.raise_for_status()

    text = resp.text
    results = []
    import re
    snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', text, re.DOTALL)
    for i, snippet in enumerate(snippets[:5], 1):
        clean = re.sub(r"<.*?>", "", snippet).strip()
        if clean:
            results.append(f"{i}. {clean}")

    if results:
        return f"Search results for '{query}':\n" + "\n".join(results)
    return f"No results found for: {query}"


@tool_registry.register(
    name="file_write",
    description="Write content to a file (sandboxed to workspace)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to sandbox)"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    },
)
async def real_file_write(path: str, content: str) -> str:
    resolved = _resolve_sandboxed(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {resolved.relative_to(Path(_SANDBOX_DIR).resolve())}"


@tool_registry.register(
    name="file_read",
    description="Read content from a file (sandboxed to workspace)",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path (relative to sandbox)"},
        },
        "required": ["path"],
    },
)
async def real_file_read(path: str) -> str:
    resolved = _resolve_sandboxed(path)
    if not resolved.exists():
        return f"File not found: {path}"
    content = resolved.read_text(encoding="utf-8")
    if len(content) > 10000:
        return content[:10000] + f"\n... (truncated, {len(content)} total characters)"
    return content


@tool_registry.register(
    name="shell_command",
    description="Run a shell command and return its output (sandboxed, timeout 30s)",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        },
        "required": ["command"],
    },
)
async def real_shell_command(command: str) -> str:
    import asyncio
    import shlex

    # Block dangerous patterns
    blocked = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb", ">/dev/sd",
               "chmod -R 777 /", "wget|sh", "curl|sh"]
    cmd_lower = command.lower().replace(" ", "")
    for b in blocked:
        if b.replace(" ", "") in cmd_lower:
            return f"Blocked dangerous command: {command}"

    # Ensure sandbox exists
    Path(_SANDBOX_DIR).mkdir(parents=True, exist_ok=True)

    try:
        import sys
        if sys.platform == "win32":
            # On Windows, use shell=True via create_subprocess_shell for correct
            # PATH resolution and built-in command support (echo, dir, etc.)
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_SANDBOX_DIR,
            )
        else:
            args = shlex.split(command)
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_SANDBOX_DIR,
            )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        if stderr:
            output += "\nSTDERR: " + stderr.decode(errors="replace")
        if len(output) > 5000:
            output = output[:5000] + "\n... (truncated)"
        return output or "(no output)"
    except asyncio.TimeoutError:
        proc.kill()
        return "Command timed out after 30 seconds"
    except Exception as e:
        return f"Command failed: {e}"
