"""The MCP adapter: two handlers over `dispatch_tool_call`, and nothing else.

This is the only module in the package that imports `mcp.server`, which keeps every other module
testable without a transport.

It uses the **low-level** `Server` rather than `MCPServer`, for three reasons that all point the
same way (see docs/adr/0003 for the full argument):

1. `MCPServer.add_tool` has no schema parameter — `Tool.from_function` derives `parameters` from the
   function signature unconditionally, so a hand-authored strict schema cannot be supplied.
2. Its generated argument model has no `extra="forbid"`, so unknown arguments are **silently
   dropped**. Advertising `additionalProperties: false` while ignoring extras would be worse than
   not advertising it: a caller sending an extra `amount_cents` would get no error and believe it
   was honoured.
3. `MCPServer._handle_call_tool` echoes `str(e)` verbatim into the result, and a pydantic
   `ValidationError` string embeds the rejected input — an I2 leak if a caller pastes a PAN into a
   field. Owning the envelope means caller input never round-trips.

The cost is these ~40 lines. `Server.streamable_http_app()` stays available on the same object if the
eval harness later wants HTTP, so nothing is given up by choosing stdio now.
"""

from __future__ import annotations

import mcp_types as types
from mcp.server.lowlevel import Server

from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.descriptions import SERVER_INSTRUCTIONS
from payagent.mcp_server.dispatch import dispatch_tool_call
from payagent.mcp_server.registry import list_tools_result

SERVER_NAME = "payagent-guard"
SERVER_VERSION = "0.1.0"


def build_server(
    deps: ToolDeps, *, name: str = SERVER_NAME, version: str = SERVER_VERSION
) -> Server:
    """Wire the six tools onto a low-level MCP server, with `deps` bound by closure.

    No env reads and no client construction here — same dependency-injection discipline as
    `Retriever.__init__` and `make_retrieve_node`, so a test can hand this fully faked collaborators.
    """

    async def on_list_tools(
        ctx: object, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        # Pure: the tool list depends on nothing but the registry, so there is no pagination and
        # nothing to cache.
        return list_tools_result()

    async def on_call_tool(
        ctx: object, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        return await dispatch_tool_call(deps, params.name, params.arguments)

    return Server(
        name,
        version=version,
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )
