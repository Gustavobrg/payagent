"""The six MCP tools from Architecture.md, over the official `mcp` SDK.

`search_catalog`, `get_quote`, `create_intent_mandate`, `create_payment_mandate`,
`execute_settlement`, `refund`. Strict schema on all of them: `additionalProperties: false`, money as
integer cents, explicit ISO currency, `idempotency_key` required on the four with side effects.

Layered like `rag/`: `handlers.py` holds every invariant as plain synchronous functions, `dispatch.py`
is the one cross-cutting funnel, and `server.py` is the only module that imports `mcp.server`.

Note that `search_catalog` here is the **external** interface. It shares the vector store and the
untrusted-content wrapper with the retrieval sub-agent's tool of the same name, but nothing else —
this package never imports `payagent.rag.tools`.
"""

from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.dispatch import dispatch_tool_call
from payagent.mcp_server.registry import TOOL_SPECS, TOOL_SPECS_BY_NAME, list_tools_result
from payagent.mcp_server.server import build_server

__all__ = [
    "TOOL_SPECS",
    "TOOL_SPECS_BY_NAME",
    "ToolDeps",
    "build_server",
    "dispatch_tool_call",
    "list_tools_result",
]
