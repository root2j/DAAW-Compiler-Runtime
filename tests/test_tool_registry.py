"""Tests for the tool registry."""

import asyncio

import pytest

from daaw.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()

        @registry.register("greet", "Says hello")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = registry.get("greet")
        assert tool is not None
        assert tool.name == "greet"
        assert tool.description == "Says hello"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nope") is None

    def test_execute(self):
        registry = ToolRegistry()

        @registry.register("add", "Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        result = asyncio.get_event_loop().run_until_complete(
            registry.execute("add", a=2, b=3)
        )
        assert result == 5

    def test_execute_nonexistent_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="not registered"):
            asyncio.get_event_loop().run_until_complete(
                registry.execute("missing")
            )

    def test_list_tools_all(self):
        registry = ToolRegistry()

        @registry.register("t1", "Tool 1", {"type": "object", "properties": {"x": {"type": "string"}}})
        async def t1(x: str) -> str:
            return x

        @registry.register("t2", "Tool 2")
        async def t2() -> str:
            return ""

        schemas = registry.list_tools()
        assert len(schemas) == 2
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "t1"

    def test_list_tools_filtered(self):
        registry = ToolRegistry()

        @registry.register("allowed_tool", "Allowed")
        async def allowed() -> str:
            return ""

        @registry.register("blocked_tool", "Blocked")
        async def blocked() -> str:
            return ""

        schemas = registry.list_tools(allowed=["allowed_tool"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "allowed_tool"


class TestMockTools:
    """Test mock tool implementations directly (not via global registry).

    We test the mock handler functions directly to avoid import-order races
    with real_tools which may overwrite the same registry keys.
    """

    def test_mock_tools_registered(self):
        # Import mock_tools and verify functions are available
        import daaw.tools.mock_tools as mt
        assert callable(mt.mock_web_search)
        assert callable(mt.mock_file_write)
        assert callable(mt.mock_file_read)

    def test_mock_web_search(self):
        from daaw.tools.mock_tools import mock_web_search

        result = asyncio.get_event_loop().run_until_complete(
            mock_web_search(query="test query")
        )
        assert "[MOCK]" in result
        assert "test query" in result

    def test_mock_file_write(self):
        from daaw.tools.mock_tools import mock_file_write

        result = asyncio.get_event_loop().run_until_complete(
            mock_file_write(path="/tmp/test.txt", content="hello")
        )
        assert "[MOCK]" in result
        assert "5 chars" in result

    def test_mock_file_read(self):
        from daaw.tools.mock_tools import mock_file_read

        result = asyncio.get_event_loop().run_until_complete(
            mock_file_read(path="/tmp/test.txt")
        )
        assert "[MOCK]" in result
        assert "/tmp/test.txt" in result
