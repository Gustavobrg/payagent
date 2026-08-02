"""stdio entrypoint: `uv run python -m payagent.mcp_server`.

The **only** module in this package that reads configuration. Everything below the entrypoint takes
its collaborators as arguments, which is what lets the whole contract suite run offline against an
in-memory Qdrant, a fake clock, and an in-process merchant.

Transport is stdio: the consumer is the LangGraph agent running locally, so there is no listening
socket and no auth surface to get wrong. Architecture.md puts APIM in front of the *agent*, not in
front of the tool server, so exposing HTTP here would add an unprotected money-moving surface for no
gain.

Configuration failure is fail-closed, not a crash: if `POLICY_MAX_AMOUNT_CENTS` and friends are
missing or malformed, `_build_policy_engine` falls back to `DenyAllPolicyEngine` rather than
starting with a half-configured `RulesPolicyEngine`; if `MANDATE_SIGNING_KEY_PATH` cannot be read
or written, `_build_mandate_authority` falls back to `UnsignedMandateAuthority`, which verifies
nothing. Either fallback means the server starts, advertises all six tools, and simply cannot
settle anything — loud in its effect, never silent about why.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dotenv import load_dotenv
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from payagent.mandates import (
    Ed25519MandateAuthority,
    InMemoryMandateSignatureStore,
    InMemoryMandateStore,
    MandateAuthority,
    UnsignedMandateAuthority,
)
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.idempotency import InMemoryIdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import InMemoryQuoteStore
from payagent.mcp_server.registry import TOOL_SPECS
from payagent.mcp_server.server import build_server
from payagent.mcp_server.step_up import InMemoryStepUpVerifier
from payagent.observability.logging import configure_logging, get_logger
from payagent.observability.tracing import configure_tracing
from payagent.policy import DenyAllPolicyEngine, PolicyEngine, RulesPolicyEngine
from payagent.rag.retriever import Retriever

logger = get_logger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _uuid_id_factory(prefix: str) -> str:
    """Unguessable handles.

    Mandate and settlement IDs are capability-like: knowing one lets you name it in a later call.
    Sequential IDs would be enumerable, so production never uses the counter factory the tests do.
    """
    return f"{prefix}-{uuid4().hex[:16]}"


def _parse_int_env(name: str) -> int | None:
    """`None` for unset, blank, or non-integer — the caller decides what that means."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _parse_list_env(name: str) -> tuple[str, ...] | None:
    """Comma-separated env var to a tuple. `None` for unset or blank, not an empty tuple —
    an *empty* allowlist/restricted-category-set is a legitimate config (allow no merchants,
    restrict no categories), distinct from "not configured at all"."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_policy_engine() -> PolicyEngine:
    """POL-001/002/003/004/005 from the environment, or a fail-closed deny-all.

    All five values must be present and well-formed for `RulesPolicyEngine` to be used — a
    partially configured engine (e.g. a merchant allowlist with no amount ceiling) is worse than
    no engine, so any single missing/malformed value falls the whole thing back to `DenyAllPolicyEngine`.
    """
    max_amount_cents = _parse_int_env("POLICY_MAX_AMOUNT_CENTS")
    stepup_threshold_cents = _parse_int_env("POLICY_STEPUP_THRESHOLD_CENTS")
    daily_aggregate_limit_cents = _parse_int_env("POLICY_DAILY_AGGREGATE_LIMIT_CENTS")
    merchant_allowlist = _parse_list_env("POLICY_MERCHANT_ALLOWLIST")
    restricted_categories = _parse_list_env("POLICY_RESTRICTED_CATEGORIES")

    if (
        max_amount_cents is None
        or stepup_threshold_cents is None
        or daily_aggregate_limit_cents is None
        or merchant_allowlist is None
        or restricted_categories is None
    ):
        logger.warning("policy_engine_not_configured_falling_back_to_deny_all")
        return DenyAllPolicyEngine()

    return RulesPolicyEngine(
        max_amount_cents=max_amount_cents,
        stepup_threshold_cents=stepup_threshold_cents,
        daily_aggregate_limit_cents=daily_aggregate_limit_cents,
        merchant_allowlist=merchant_allowlist,
        restricted_categories=restricted_categories,
    )


def _load_or_generate_signing_key(path: Path) -> Ed25519PrivateKey:
    """Load the Ed25519 key at `path`, or generate and persist one if it doesn't exist yet.

    Dev convenience only, per `.env`'s own comment on `MANDATE_SIGNING_KEY_PATH`: production
    reads the key from Key Vault instead of generating one on first run.
    """
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)

    private_key = Ed25519PrivateKey.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return private_key


def _build_mandate_authority() -> MandateAuthority:
    """Ed25519 JWS signing from `MANDATE_SIGNING_KEY_PATH`, or a fail-closed unsigned authority."""
    key_path = os.environ.get("MANDATE_SIGNING_KEY_PATH")
    if not key_path:
        logger.warning("mandate_signing_key_not_configured_falling_back_to_unsigned")
        return UnsignedMandateAuthority()

    try:
        private_key = _load_or_generate_signing_key(Path(key_path))
    except Exception as exc:
        logger.error("mandate_signing_key_load_failed", error=str(exc))
        return UnsignedMandateAuthority()

    return Ed25519MandateAuthority(
        private_key=private_key, signatures=InMemoryMandateSignatureStore()
    )


def build_deps_from_env() -> ToolDeps:
    """Read configuration once, here, and inject everything else."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    use_deterministic = os.environ.get("PAYAGENT_DETERMINISTIC_EMBEDDER") == "1"

    return ToolDeps(
        retriever=Retriever(
            qdrant_url=qdrant_url, use_deterministic_embedder=use_deterministic
        ),
        policy=_build_policy_engine(),
        mandate_authority=_build_mandate_authority(),
        mandate_store=InMemoryMandateStore(),
        quotes=InMemoryQuoteStore(clock=_utc_now),
        merchant=MerchantSim(clock=_utc_now, id_factory=_uuid_id_factory),
        idempotency=InMemoryIdempotencyStore(),
        clock=_utc_now,
        id_factory=_uuid_id_factory,
        step_up=InMemoryStepUpVerifier(id_factory=_uuid_id_factory),
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
    configure_tracing()

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
