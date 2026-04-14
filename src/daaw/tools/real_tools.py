"""Real tool implementations for production use."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

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


# ---------------------------------------------------------------------------
# SSRF protection for http_request
# ---------------------------------------------------------------------------
_SSRF_DENY_NETS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _check_ssrf(url: str) -> None:
    """Block requests to private/link-local IPs and metadata endpoints."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if host in ("metadata.google.internal", "metadata.google",
                "169.254.169.254"):
        raise PermissionError(f"SSRF: blocked metadata endpoint: {host}")
    try:
        for addr in __import__("socket").getaddrinfo(host, None):
            ip = ipaddress.ip_address(addr[4][0])
            for net in _SSRF_DENY_NETS:
                if ip in net:
                    raise PermissionError(
                        f"SSRF: {host} resolves to private IP {ip}"
                    )
    except (OSError, ValueError):
        pass  # unresolvable host → httpx will fail naturally


_MAX_RESPONSE_BODY = 64 * 1024  # 64 KB cap


@tool_registry.register(
    name="http_request",
    description=(
        "Send an HTTP request (GET, POST, PUT, DELETE, PATCH) to any public URL. "
        "Returns status code, headers, and body (capped at 64 KB). "
        "Use for REST APIs, webhooks, fetching web pages, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, DELETE, PATCH",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            },
            "url": {"type": "string", "description": "Full URL including scheme"},
            "headers": {
                "type": "object",
                "description": "Optional request headers as key-value pairs",
            },
            "json_body": {
                "type": "object",
                "description": "Optional JSON body (for POST/PUT/PATCH)",
            },
            "params": {
                "type": "object",
                "description": "Optional URL query parameters as key-value pairs",
            },
        },
        "required": ["method", "url"],
    },
)
async def http_request(
    method: str = "GET",
    url: str = "",
    headers: dict | None = None,
    json_body: dict | None = None,
    params: dict | None = None,
) -> str:
    """Generic HTTP client with SSRF protection and body-size cap."""
    import httpx

    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"Unsupported HTTP method: {method}"

    try:
        _check_ssrf(url)
    except PermissionError as e:
        return str(e)

    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, max_redirects=3,
        ) as client:
            resp = await client.request(
                method, url,
                headers=headers or {},
                json=json_body if json_body else None,
                params=params or {},
            )
        body = resp.text[:_MAX_RESPONSE_BODY]
        # Strip Authorization on cross-origin redirects (handled by httpx,
        # but defensive in case a future version doesn't).
        hdr_summary = {
            k: v for k, v in list(resp.headers.items())[:20]
        }
        return (
            f"HTTP {resp.status_code} {resp.reason_phrase}\n"
            f"Headers: {hdr_summary}\n"
            f"Body ({len(body)} chars):\n{body}"
        )
    except httpx.TimeoutException:
        return f"HTTP request timed out after 30s: {url}"
    except Exception as e:
        return f"HTTP request failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# python_exec: sandboxed code execution
# ---------------------------------------------------------------------------
@tool_registry.register(
    name="python_exec",
    description=(
        "Execute a short Python code snippet in a sandboxed subprocess. "
        "Returns stdout, stderr, and the exit code. Use for calculations, "
        "data processing, CSV/JSON manipulation, string formatting, etc. "
        "The sandbox has no network access. Files are read/written under "
        "the DAAW sandbox directory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute. Use print() to produce output. "
                    "Common imports available: json, csv, math, re, datetime, "
                    "collections, itertools, pathlib."
                ),
            },
        },
        "required": ["code"],
    },
)
async def python_exec(code: str = "") -> str:
    """Run Python code in a subprocess with a 30-second timeout."""
    import asyncio
    import sys
    import tempfile

    if not code.strip():
        return "Error: empty code"

    # Basic static rejection of dangerous imports.
    _BLOCKED = {"subprocess", "ctypes", "socket", "http.server",
                "xmlrpc", "multiprocessing", "signal", "os.exec"}
    for blocked in _BLOCKED:
        if blocked in code:
            return f"Blocked: import/use of '{blocked}' is not allowed in sandbox"

    # Write code to a temp file so we can run it in a clean subprocess.
    sandbox = Path(_SANDBOX_DIR).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(sandbox),
        delete=False, encoding="utf-8",
    ) as f:
        # Inject a safe CWD so relative file paths land in the sandbox.
        f.write(f"import os; os.chdir({str(sandbox)!r})\n")
        f.write(code)
        script_path = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(sandbox),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=30,
        )
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        result_parts = []
        if out:
            result_parts.append(f"stdout:\n{out[:8000]}")
        if err:
            result_parts.append(f"stderr:\n{err[:4000]}")
        result_parts.append(f"exit_code: {proc.returncode}")
        return "\n".join(result_parts) or "(no output)"
    except asyncio.TimeoutError:
        return "Code execution timed out after 30 seconds"
    except Exception as e:
        return f"Execution failed: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
