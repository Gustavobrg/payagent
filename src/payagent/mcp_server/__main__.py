"""stdio entrypoint: `uv run python -m payagent.mcp_server`.

The **only** module in this package that reads configuration. Everything below the entrypoint takes
its collaborators as arguments, which is what lets the whole contract suite run offline against an
in-memory Qdrant, a fake clock, and an in-process merchant.

Transport is stdio: the consumer is the LangGraph agent running locally, so there is no listening
socket and no auth surface to get wrong. Architecture.md puts APIM in front of the *agent*, not in
front of the tool server, so exposing HTTP here would add an unprotected money-moving surface for no
gain.

Note what this wires by default: `DenyAllPolicyEngine` and `UnsignedMandateAuthority`. The server
starts, advertises all six tools, and **cannot settle anything** — the real policy rules and mandate
signing land in later blocks. That is fail-closed working as intended, and it is loud rather than
silent.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

import anyio
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from payagent.mandates import InMemoryMandateStore, UnsignedMandateAuthority
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.idempotency import InMemoryIdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import InMemoryQuoteStore
from payagent.mcp_server.registry import TOOL_SPECS
from payagent.mcp_server.server import build_server
from payagent.observability.logging import configure_logging, get_logger
from payagent.policy import DenyAllPolicyEngine
from payagent.rag.retriever import Retriever


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_id_factory(prefix: str) -> str:
    """Unguessable handles.

    Mandate and settlement IDs are capability-like: knowing one lets you name it in a later call.
    Sequential IDs would be enumerable, so production never uses the counter factory the tests do.
    """
    return f"{prefix}-{uuid4().hex[:16]}"


def build_deps_from_env() -> ToolDeps:
    """Read configuration once, here, and inject everything else."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    use_deterministic = os.environ.get("PAYAGENT_DETERMINISTIC_EMBEDDER") == "1"

    return ToolDeps(
        retriever=Retriever(
            qdrant_url=qdrant_url, use_deterministic_embedder=use_deterministic
        ),
        # Fail-closed defaults. Replaced when the real rules and signing authority land.
        policy=DenyAllPolicyEngine(),
        mandate_authority=UnsignedMandateAuthority(),
        mandate_store=InMemoryMandateStore(),
        quotes=InMemoryQuoteStore(clock=_utc_now),
        merchant=MerchantSim(clock=_utc_now, id_factory=_uuid_id_factory),
        idempotency=InMemoryIdempotencyStore(),
        clock=_utc_now,
        id_factory=_uuid_id_factory,
    )


async def _serve(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    load_dotenv()
    # Redirect logs to stderr before building anything: stdout belongs to the JSON-RPC transport, and
    # constructing the retriever emits a log line. Passing an explicit stream reconfigures even
    # though importing this module's dependencies already configured logging with stdout.
    configure_logging(stream=sys.stderr)
    logger = get_logger(__name__)

    deps = build_deps_from_env()
    logger.info(
        "mcp_server_starting",
        transport="stdio",
        tools=len(TOOL_SPECS),
        policy_engine=type(deps.policy).__name__,
        mandate_authority=type(deps.mandate_authority).__name__,
    )
    anyio.run(_serve, build_server(deps))


if __name__ == "__main__":
    main()
