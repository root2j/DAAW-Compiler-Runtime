"""Mock tool implementations for testing and development."""

from daaw.tools.registry import tool_registry


@tool_registry.register(
    name="web_search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def mock_web_search(query: str) -> str:
    return f"[MOCK] Search results for: {query}"


@tool_registry.register(
    name="file_write",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    },
)
async def mock_file_write(path: str, content: str) -> str:
    return f"[MOCK] Wrote {len(content)} chars to {path}"


@tool_registry.register(
    name="file_read",
    description="Read content from a file",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File path"}},
        "required": ["path"],
    },
)
async def mock_file_read(path: str) -> str:
    return f"[MOCK] Contents of {path}: <placeholder>"
