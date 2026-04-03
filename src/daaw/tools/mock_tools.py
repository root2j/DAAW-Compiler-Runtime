"""Mock tool implementations for testing and development.

Includes aliases for commonly LLM-hallucinated tool names (brave_search, search, etc.)
so the agent doesn't get stuck calling unregistered tools in mock mode.
"""

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


@tool_registry.register(
    name="shell_command",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Shell command to run"}},
        "required": ["command"],
    },
)
async def mock_shell_command(command: str) -> str:
    return f"[MOCK] Executed: {command}"


# ── Aliases for commonly LLM-hallucinated tool names ─────────────────────
# LLMs sometimes call tools by names from their training data rather than
# the ones actually provided in the schema.  Register thin wrappers so
# the agent doesn't spin in an error loop.

@tool_registry.register(
    name="brave_search",
    description="Search the web for information (alias for web_search)",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def mock_brave_search(query: str) -> str:
    return await mock_web_search(query)


@tool_registry.register(
    name="search",
    description="Search the web for information (alias for web_search)",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def mock_search(query: str) -> str:
    return await mock_web_search(query)


@tool_registry.register(
    name="google_search",
    description="Search the web for information (alias for web_search)",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    },
)
async def mock_google_search(query: str) -> str:
    return await mock_web_search(query)
