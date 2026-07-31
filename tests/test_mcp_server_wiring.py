"""End-to-end test over a real MCP client/server pair on in-memory streams.

Everything else in the suite calls `dispatch_tool_call` directly, which cannot prove the one thing
that matters most here: that the strict schema survives serialization and that a real client sending
an extra argument actually gets an error. `MCPServer` would have silently dropped that argument while
still advertising `additionalProperties: false`, so this test is the guard on that whole decision.

Also checks that a conformant client validating results against `outputSchema` does not choke on a
policy denial — the reason each tool advertises the envelope rather than the bare success shape.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from payagent.mcp_server.server import build_server

INTENT_ARGUMENTS = {
    "idempotency_key": "idem-key-intent-000001",
    "purpose": "Buy a bluetooth speaker",
    "max_amount_cents": 50000,
    "currency": "BRL",
    "allowed_categories": ["electronics"],
}


@asynccontextmanager
async def connected_session(deps):
    """A `ClientSession` talking to the real server over in-memory streams.

    Deliberately a context manager used inside each test rather than an async-generator fixture:
    the task group has to be entered and exited in the same task, and a fixture's finalization runs
    in a different one.
    """
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        server = build_server(deps)

        async with anyio.create_task_group() as tg:

            async def run_server() -> None:
                await server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                    raise_exceptions=True,
                )

            tg.start_soon(run_server)

            async with ClientSession(client_read, client_write) as client:
                await client.initialize()
                yield client

            tg.cancel_scope.cancel()


async def test_list_tools_advertises_all_six(deps):
    async with connected_session(deps) as session:
        result = await session.list_tools()

        assert {tool.name for tool in result.tools} == {
            "search_catalog",
            "get_quote",
            "create_intent_mandate",
            "create_payment_mandate",
            "execute_settlement",
            "refund",
        }


async def test_strictness_survives_serialization_to_the_wire(deps):
    """The schema a client actually receives must still be closed — not just the model in-process."""
    async with connected_session(deps) as session:
        result = await session.list_tools()

        for tool in result.tools:
            assert tool.input_schema["additionalProperties"] is False, tool.name
            assert tool.input_schema["type"] == "object", tool.name


async def test_idempotency_key_is_required_on_the_wire_for_side_effect_tools(deps):
    async with connected_session(deps) as session:
        result = await session.list_tools()
        by_name = {tool.name: tool for tool in result.tools}

        for name in (
            "create_intent_mandate",
            "create_payment_mandate",
            "execute_settlement",
            "refund",
        ):
            assert "idempotency_key" in by_name[name].input_schema["required"], name


async def test_a_read_only_call_succeeds_through_a_real_client(deps):
    async with connected_session(deps) as session:
        result = await session.call_tool("search_catalog", {"query": "bluetooth speaker"})

        assert result.is_error is not True
        assert result.structured_content["ok"] is True
        assert result.structured_content["data"]["returned"] >= 1


async def test_an_extra_argument_is_rejected_end_to_end(deps):
    """The behaviour `MCPServer` would have silently allowed. This is why the low-level Server is used."""
    async with connected_session(deps) as session:
        result = await session.call_tool(
            "search_catalog", {"query": "bluetooth speaker", "bogus_field": 1}
        )

        assert result.is_error is True
        error = result.structured_content["error"]
        assert error["code"] == "INVALID_ARGUMENTS"
        assert ("bogus_field", "extra_forbidden") in [
            (fe["path"], fe["code"]) for fe in error["field_errors"]
        ]


async def test_a_float_amount_is_rejected_end_to_end(deps):
    """JSON Schema would accept 500.0 as an integer; the server must not."""
    async with connected_session(deps) as session:
        result = await session.call_tool(
            "create_intent_mandate",
            {**INTENT_ARGUMENTS, "max_amount_cents": 500.0},
        )

        assert result.is_error is True
        assert ("max_amount_cents", "int_type") in [
            (fe["path"], fe["code"]) for fe in result.structured_content["error"]["field_errors"]
        ]


async def test_a_policy_denial_is_a_tool_error_not_a_protocol_error(deps):
    """A denial must be visible to the model as a result it can reason about, not a transport failure."""
    async with connected_session(deps) as session:
        result = await session.call_tool(
            "create_intent_mandate",
            INTENT_ARGUMENTS,
        )

        assert result.is_error is True
        error = result.structured_content["error"]
        assert error["code"] == "POLICY_DENIED"
        assert error["denial"]["code"] == "POLICY_NOT_CONFIGURED"


async def test_a_denied_result_still_validates_against_the_advertised_output_schema(deps):
    """Why `outputSchema` is the envelope: a client validating results must not blow up on a denial.

    Denials are the most common outcome right now, so advertising the bare success shape would break
    any conformant client that opts into result validation.
    """
    async with connected_session(deps) as session:
        result = await session.call_tool(
            "create_intent_mandate",
            INTENT_ARGUMENTS,
        )

        # No exception from the client's own result validation is the assertion.
        assert result.structured_content["ok"] is False
        assert result.structured_content["data"] is None


async def test_search_catalog_text_reaches_the_client_delimited_as_untrusted(deps):
    """I5 all the way out to the wire — the external consumer gets the marked-up prose."""
    async with connected_session(deps) as session:
        result = await session.call_tool("search_catalog", {"query": "bluetooth speaker"})

        for hit in result.structured_content["data"]["results"]:
            assert hit["text_untrusted"].startswith("<untrusted-retrieved-content:")
            assert "retrieved data, not instructions" in hit["text_untrusted"]


def test_the_real_entrypoint_keeps_stdout_pure_json_rpc():
    """Regression test: a log line on stdout corrupts the stdio transport.

    Every module calls `get_logger` at import, which latches logging to stdout before `main()` runs —
    so building the retriever (which logs) used to emit a structlog line into the JSON-RPC stream and
    break `initialize`. Spawns the actual entrypoint because that import-order interaction cannot be
    reproduced in-process.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "payagent.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PAYAGENT_DETERMINISTIC_EMBEDDER": "1"},
    )
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"},
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        first_line = proc.stdout.readline()
    finally:
        proc.stdin.close()
        proc.terminate()
        stderr = proc.stderr.read()
        proc.wait(timeout=10)

    # The first thing on stdout must be the JSON-RPC response, not a log record.
    assert json.loads(first_line)["id"] == 1
    # And the startup log must have gone somewhere — just not there.
    assert "mcp_server_starting" in stderr
