"""Synchronous adapter over `dispatch_tool_call`, for graph nodes.

`dispatch_tool_call` is `async def` only because the MCP SDK's `on_call_tool` handler
signature requires it — its body contains no `await` (see its own module docstring: the
idempotency reserve/effect/complete sequence must run atomically with respect to the event
loop, which is only true because nothing in it yields). The purchase graph is synchronous
end to end, so this runs the coroutine to completion with `anyio.run` rather than making
every node async for a call that never actually suspends.
"""

from __future__ import annotations

from typing import Any

import anyio

from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.dispatch import dispatch_tool_call


def call_tool(deps: ToolDeps, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call MCP tool `name` and return its envelope: `{"ok": bool, "data": ..., "error": ...}`."""
    result = anyio.run(dispatch_tool_call, deps, name, arguments)
    return result.structured_content
