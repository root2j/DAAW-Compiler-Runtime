"""Tests for real tool implementations — unit tests, no external APIs."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from daaw.tools.registry import ToolRegistry


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def sandbox_dir(tmp_path, monkeypatch):
    """Set DAAW_SANDBOX_DIR to a temp directory for all tests."""
    sandbox = str(tmp_path / "sandbox")
    monkeypatch.setenv("DAAW_SANDBOX_DIR", sandbox)
    # Patch the module-level _SANDBOX_DIR
    import daaw.tools.real_tools as rt
    monkeypatch.setattr(rt, "_SANDBOX_DIR", sandbox)
    return sandbox


@pytest.fixture
def registry(sandbox_dir):
    """Import real_tools into a fresh state (tools register on the global singleton)."""
    import daaw.tools.real_tools  # noqa: F401
    from daaw.tools.registry import tool_registry
    return tool_registry


class TestRealFileWrite:
    def test_write_creates_file(self, registry, sandbox_dir):
        result = run(registry.execute("file_write", path="test.txt", content="hello world"))
        assert "11 characters" in result
        assert os.path.exists(os.path.join(sandbox_dir, "test.txt"))

    def test_write_nested_dirs(self, registry, sandbox_dir):
        result = run(registry.execute("file_write", path="sub/dir/file.txt", content="nested"))
        assert "6 characters" in result
        assert os.path.exists(os.path.join(sandbox_dir, "sub", "dir", "file.txt"))

    def test_write_path_traversal_blocked(self, registry):
        with pytest.raises(PermissionError, match="escapes sandbox"):
            run(registry.execute("file_write", path="../../etc/passwd", content="hack"))

    def test_write_dotdot_in_middle_blocked(self, registry):
        with pytest.raises(PermissionError, match="escapes sandbox"):
            run(registry.execute("file_write", path="a/../../outside.txt", content="x"))


class TestRealFileRead:
    def test_read_existing_file(self, registry, sandbox_dir):
        # Write a file first
        os.makedirs(sandbox_dir, exist_ok=True)
        filepath = os.path.join(sandbox_dir, "readme.txt")
        with open(filepath, "w") as f:
            f.write("hello from file")
        result = run(registry.execute("file_read", path="readme.txt"))
        assert "hello from file" in result

    def test_read_nonexistent_file(self, registry):
        result = run(registry.execute("file_read", path="nope.txt"))
        assert "not found" in result.lower()

    def test_read_path_traversal_blocked(self, registry):
        with pytest.raises(PermissionError, match="escapes sandbox"):
            run(registry.execute("file_read", path="../../../etc/shadow"))

    def test_read_truncates_large_file(self, registry, sandbox_dir):
        os.makedirs(sandbox_dir, exist_ok=True)
        filepath = os.path.join(sandbox_dir, "big.txt")
        with open(filepath, "w") as f:
            f.write("x" * 20000)
        result = run(registry.execute("file_read", path="big.txt"))
        assert "truncated" in result
        assert len(result) < 20000


class TestRealShellCommand:
    def test_echo_command(self, registry, sandbox_dir):
        os.makedirs(sandbox_dir, exist_ok=True)
        # Use 'cmd /c echo' on Windows, 'echo' on Unix
        import sys
        if sys.platform == "win32":
            result = run(registry.execute("shell_command", command="cmd /c echo hello"))
        else:
            result = run(registry.execute("shell_command", command="echo hello"))
        assert "hello" in result

    def test_blocked_dangerous_command(self, registry):
        result = run(registry.execute("shell_command", command="rm -rf /"))
        assert "Blocked" in result

    def test_blocked_mkfs(self, registry):
        result = run(registry.execute("shell_command", command="mkfs /dev/sda"))
        assert "Blocked" in result

    def test_timeout_handling(self, registry, sandbox_dir):
        """Long-running commands should timeout."""
        os.makedirs(sandbox_dir, exist_ok=True)
        import sys
        # Use platform-appropriate long-running command
        if sys.platform == "win32":
            # 'timeout /t N /nobreak' waits N seconds on Windows
            # Use ping as a more reliable cross-cmd sleep alternative
            cmd = "ping -n 120 127.0.0.1"
        else:
            cmd = "sleep 120"
        result = run(registry.execute("shell_command", command=cmd))
        assert "timed out" in result.lower() or "failed" in result.lower()


class TestToolRegistration:
    def test_real_tools_register_on_global(self):
        """Real tools should register with the same names as mock tools."""
        import daaw.tools.real_tools  # noqa: F401
        from daaw.tools.registry import tool_registry

        assert tool_registry.get("web_search") is not None
        assert tool_registry.get("file_write") is not None
        assert tool_registry.get("file_read") is not None
        assert tool_registry.get("shell_command") is not None

    def test_tool_schemas_have_correct_format(self):
        import daaw.tools.real_tools  # noqa: F401
        from daaw.tools.registry import tool_registry

        schemas = tool_registry.list_tools(allowed=["web_search", "file_write", "file_read", "shell_command"])
        for schema in schemas:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
