"""Manual smoke test of the MCP server, driven by a real MCP client.

Three things worth seeing with your own eyes, since none are obvious from reading the code:

  1. The six tools list with a readable, strict schema — what an MCP Inspector would show.
  2. Calling `execute_settlement` twice with the same `idempotency_key` produces exactly ONE
     charge in the simulated merchant.
  3. Every side-effect tool is denied today, because the policy stub is fail-closed.

This is a genuine MCP client over in-memory streams (`mcp.shared.memory`), not a direct call into
`dispatch_tool_call` — so everything below crosses a real JSON-RPC boundary, which is the only way
to see that the strict schema survives serialization.

Runs offline: an in-memory Qdrant seeded from the real `evals/datasets/catalog.json` with the
deterministic embedder. No Docker, no OPENROUTER_API_KEY.

    uv run python scratch_search.py
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from qdrant_client import QdrantClient

from payagent.mandates import (
    InMemoryMandateStore,
    MandateVerification,
    UnsignedMandateAuthority,
)
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.idempotency import InMemoryIdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import InMemoryQuoteStore
from payagent.mcp_server.server import build_server
from payagent.observability.logging import configure_logging
from payagent.policy import (
    Allow,
    DenyAllPolicyEngine,
    EffectGrant,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
)
from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    create_collection_if_missing,
    ingest_chunks,
    load_catalog,
)
from payagent.rag.retriever import Retriever

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
SEARCH_QUERY = "wireless headphones"


# --------------------------------------------------------------------------------- local doubles
# Defined here rather than imported from `tests/` so this script stands alone. They exist only to
# make check 2 reachable: with the shipped defaults nothing can settle, which is check 3's point.


class AllowAllPolicyEngine(PolicyEngine):
    """Grants everything. Stands in for the real rules that land in a later block."""

    def decide(self, action: PolicyAction, context: PolicyContext):
        return Allow(
            grant=EffectGrant(
                action=action,
                amount_cents=context.amount_cents,
                currency=context.currency,
                merchant_id=context.merchant_id,
                decided_at=context.now,
            )
        )


class AlwaysVerifyingAuthority(UnsignedMandateAuthority):
    """Stands in for the JWS/Ed25519 authority that lands in a later block."""

    def verify_for_settlement(self, mandate, *, now) -> MandateVerification:
        return MandateVerification(verified=True)


def counter_id_factory():
    counter = {"n": 0}

    def factory(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:016d}"

    return factory


def build_retriever() -> Retriever:
    """In-memory Qdrant seeded from the real catalog dataset — 200 SKUs, no network."""
    client = QdrantClient(":memory:")
    create_collection_if_missing(client, CATALOG_COLLECTION)
    ingest_chunks(client, CATALOG_COLLECTION, chunk_catalog(load_catalog()), DeterministicEmbedder())
    return Retriever(client=client, use_deterministic_embedder=True)


def build_deps(retriever: Retriever, *, permissive: bool, **shared) -> ToolDeps:
    """`permissive=False` is the real production wiring: deny-all policy, unverifiable mandates."""
    stores = {
        "mandate_store": InMemoryMandateStore(),
        "quotes": InMemoryQuoteStore(clock=lambda: FIXED_NOW),
        "merchant": MerchantSim(clock=lambda: FIXED_NOW, id_factory=counter_id_factory()),
        "idempotency": InMemoryIdempotencyStore(),
    }
    return ToolDeps(
        retriever=retriever,
        policy=AllowAllPolicyEngine() if permissive else DenyAllPolicyEngine(),
        mandate_authority=AlwaysVerifyingAuthority() if permissive else UnsignedMandateAuthority(),
        clock=lambda: FIXED_NOW,
        id_factory=counter_id_factory(),
        **{**stores, **shared},
    )


@asynccontextmanager
async def mcp_client(deps: ToolDeps):
    """A real `ClientSession` speaking JSON-RPC to the real server over in-memory streams."""
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
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session
            tg.cancel_scope.cancel()


# ------------------------------------------------------------------------------------- reporting


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe_field(name: str, prop: dict) -> str:
    """One readable line per field, resolving the `anyOf: [T, null]` shape optionals render as."""
    if "anyOf" in prop:
        non_null = [branch for branch in prop["anyOf"] if branch.get("type") != "null"]
        prop = non_null[0] if non_null else {}

    parts = [str(prop.get("type", "?"))]
    if "const" in prop:
        parts.append(f"const={prop['const']!r}")
    if "enum" in prop:
        parts.append(f"enum={prop['enum']}")
    if "minimum" in prop or "maximum" in prop:
        parts.append(f"range=[{prop.get('minimum', '-inf')}..{prop.get('maximum', 'inf')}]")
    if "pattern" in prop:
        parts.append(f"pattern={prop['pattern']}")
    elif "minLength" in prop or "maxLength" in prop:
        parts.append(f"len=[{prop.get('minLength', 0)}..{prop.get('maxLength', 'inf')}]")
    return f"{name:<22} {' '.join(parts)}"


def outcome(result) -> str:
    """`ok` or the structured error code — the one line that matters per call."""
    payload = result.structured_content
    if payload.get("ok"):
        return "ok"
    error = payload["error"]
    denial = error.get("denial")
    return f"{error['code']}" + (f" (denial={denial['code']})" if denial else "")


def data(result) -> dict:
    payload = result.structured_content
    if not payload.get("ok"):
        raise SystemExit(f"unexpected failure: {payload['error']}")
    return payload["data"]


# ---------------------------------------------------------------------------------------- checks


async def check_tool_listing(session: ClientSession) -> None:
    rule("CHECK 1 — the six tools, as an MCP client sees them")
    result = await session.list_tools()

    print(f"tools advertised: {len(result.tools)}\n")
    for tool in result.tools:
        schema = tool.input_schema
        hints = tool.annotations
        effect = (
            "read-only"
            if hints.read_only_hint
            else ("IRREVERSIBLE" if hints.destructive_hint else "side effect")
        )
        required = set(schema.get("required", []))

        print(f"--- {tool.name}  [{effect}]")
        print(f"    additionalProperties: {schema['additionalProperties']}   idempotent hint: {hints.idempotent_hint}")
        for name, prop in schema["properties"].items():
            marker = "*" if name in required else " "
            print(f"    {marker} {describe_field(name, prop)}")
        print()

    closed = all(t.input_schema["additionalProperties"] is False for t in result.tools)
    keyed = sorted(
        t.name for t in result.tools if "idempotency_key" in t.input_schema.get("required", [])
    )
    money_fields = sorted(
        {
            name
            for t in result.tools
            for name in t.input_schema["properties"]
            if name.endswith("_cents")
        }
    )
    print("(* = required)")
    print(f"every schema closed to unknown fields : {closed}")
    print(f"idempotency_key required on           : {keyed}")
    print(f"all monetary request fields           : {money_fields}")
    print("  -> none of them can set an amount to be moved: expected_* are assertions against a")
    print("     server-side record, max_amount_cents is a mandate ceiling, min/max_price are filters.")


async def check_idempotency(session: ClientSession, deps: ToolDeps) -> dict:
    rule("CHECK 2 — same idempotency_key twice => ONE effect in the merchant")

    # Walk the real sequence rather than fabricating records.
    search = await session.call_tool("search_catalog", {"query": SEARCH_QUERY, "top_k": 3})
    hit = data(search)["results"][0]
    print(f"search_catalog  -> {hit['sku']} at {hit['price_cents']} {hit['currency']}")

    quote = data(
        await session.call_tool(
            "get_quote", {"sku": hit["sku"], "quantity": 1, "currency": hit["currency"]}
        )
    )
    print(f"get_quote       -> {quote['quote_id']}  amount={quote['amount_cents']}")

    intent = data(
        await session.call_tool(
            "create_intent_mandate",
            {
                "idempotency_key": "demo-intent-key-0001",
                "purpose": "Demo purchase",
                "max_amount_cents": 500000,
                "currency": quote["currency"],
                "allowed_categories": [quote["category"]],
            },
        )
    )
    mandate = data(
        await session.call_tool(
            "create_payment_mandate",
            {
                "idempotency_key": "demo-payment-key-001",
                "intent_mandate_id": intent["intent_mandate_id"],
                "quote_id": quote["quote_id"],
                "expected_amount_cents": quote["amount_cents"],
                "expected_currency": quote["currency"],
            },
        )
    )
    print(f"payment mandate -> {mandate['payment_mandate_id']}  (parent {mandate['intent_mandate_id']})")

    settle_args = {
        "idempotency_key": "demo-settle-key-0001",
        "payment_mandate_id": mandate["payment_mandate_id"],
        "expected_amount_cents": mandate["amount_cents"],
        "expected_currency": mandate["currency"],
    }

    print("\ncalling execute_settlement TWICE with the same idempotency_key:")
    first = await session.call_tool("execute_settlement", settle_args)
    second = await session.call_tool("execute_settlement", settle_args)
    print(f"  call 1 -> {outcome(first):<4} settlement_id={data(first)['settlement_id']}")
    print(f"  call 2 -> {outcome(second):<4} settlement_id={data(second)['settlement_id']}")

    print(f"\n  identical payload?       {second.structured_content == first.structured_content}")
    print(f"  replay flagged in _meta? {second.meta}")
    print(f"  charges in the merchant: {len(deps.merchant.charges)}")
    for charge in deps.merchant.charges.values():
        print(f"    {charge.charge_id}  {charge.amount_cents} {charge.currency}  key={charge.idempotency_key}")
    total = sum(c.amount_cents for c in deps.merchant.charges.values())
    print(f"  total charged:           {total} cents (quote was {quote['amount_cents']})")

    # The abuse case: same key, different amount. Must refuse rather than replay the success — an
    # injected chunk that changed the amount would otherwise inherit the previous "settled".
    tampered = await session.call_tool(
        "execute_settlement", {**settle_args, "expected_amount_cents": 500000}
    )
    print(f"\n  same key + tampered amount -> {outcome(tampered)}")
    print(f"  charges after tampering:      {len(deps.merchant.charges)}  (unchanged)")

    return {
        "sku": quote["sku"],
        "quote_id": quote["quote_id"],
        "amount_cents": quote["amount_cents"],
        "currency": quote["currency"],
        "intent_mandate_id": mandate["intent_mandate_id"],
        "payment_mandate_id": mandate["payment_mandate_id"],
        "settlement_id": data(first)["settlement_id"],
    }


async def check_untrusted_wrapper(session: ClientSession) -> None:
    rule("CHECK 2b — retrieved product text arrives delimited as untrusted (I5 / P4)")
    hit = data(await session.call_tool("search_catalog", {"query": SEARCH_QUERY, "top_k": 1}))[
        "results"
    ][0]

    print(f"structured price, read from the Qdrant payload: {hit['price_cents']} {hit['currency']}")
    print("text_untrusted (the only textual field, and it is delimited):")
    for line in hit["text_untrusted"].splitlines():
        print(f"  | {line}")
    print("\n  the nonce is per-call, so injected content cannot forge a matching close tag.")


async def check_fail_closed(session: ClientSession, deps: ToolDeps, prepared: dict) -> None:
    rule("CHECK 3 — every side-effect tool is denied today (fail-closed)")
    print(f"policy engine    : {type(deps.policy).__name__}")
    print(f"mandate authority: {type(deps.mandate_authority).__name__}")
    print("\n(reusing the mandate/quote/settlement records from check 2, so each tool gets far")
    print(" enough to be *denied* instead of stopping at 'record not found')\n")

    calls = [
        (
            "create_intent_mandate",
            {
                "idempotency_key": "deny-intent-key-0001",
                "purpose": "Demo purchase",
                "max_amount_cents": 50000,
                "currency": "BRL",
                "allowed_categories": ["electronics"],
            },
        ),
        (
            "create_payment_mandate",
            {
                "idempotency_key": "deny-payment-key-001",
                "intent_mandate_id": prepared["intent_mandate_id"],
                "quote_id": prepared["quote_id"],
                "expected_amount_cents": prepared["amount_cents"],
                "expected_currency": prepared["currency"],
            },
        ),
        (
            "execute_settlement",
            {
                "idempotency_key": "deny-settle-key-0001",
                "payment_mandate_id": prepared["payment_mandate_id"],
                "expected_amount_cents": prepared["amount_cents"],
                "expected_currency": prepared["currency"],
            },
        ),
        (
            "refund",
            {
                "idempotency_key": "deny-refund-key-0001",
                "settlement_id": prepared["settlement_id"],
                "expected_amount_cents": prepared["amount_cents"],
                "expected_currency": prepared["currency"],
                "reason_code": "defective",
            },
        ),
    ]

    charges_before = len(deps.merchant.charges)
    for tool, arguments in calls:
        result = await session.call_tool(tool, arguments)
        print(f"  {tool:<23} isError={str(result.is_error):<5} {outcome(result)}")

    print(f"\n  merchant charges before: {charges_before}")
    print(f"  merchant charges after : {len(deps.merchant.charges)}  (no new effect)")
    print("\n  note: execute_settlement reports MANDATE_NOT_VERIFIED rather than POLICY_DENIED —")
    print("  it has two independent gates and hits the mandate one first (invariant I3). Even if")
    print("  the policy engine allowed everything, the shipped authority verifies nothing, so no")
    print("  code path in src/ can settle until real JWS signing lands.")

    print("\n  read-only tools still work — policy is never consulted for them (invariant I1):")
    for tool, arguments in [
        ("search_catalog", {"query": SEARCH_QUERY, "top_k": 1}),
        ("get_quote", {"sku": prepared["sku"], "quantity": 1, "currency": prepared["currency"]}),
    ]:
        result = await session.call_tool(tool, arguments)
        print(f"  {tool:<23} {outcome(result)}")


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # Logs to stderr so the readable report owns stdout. Passing an explicit stream reconfigures even
    # though the imports above already latched logging onto stdout via their `get_logger` calls.
    configure_logging(stream=sys.stderr)

    retriever = build_retriever()

    # Permissive wiring: the only way a settlement is reachable at all today, which is what makes
    # the idempotency check observable.
    permissive = build_deps(retriever, permissive=True)
    async with mcp_client(permissive) as session:
        await check_tool_listing(session)
        prepared = await check_idempotency(session, permissive)
        await check_untrusted_wrapper(session)

    # Now the real production wiring, sharing the stores so the records built above still resolve.
    production = build_deps(
        retriever,
        permissive=False,
        mandate_store=permissive.mandate_store,
        quotes=permissive.quotes,
        merchant=permissive.merchant,
    )
    async with mcp_client(production) as session:
        await check_fail_closed(session, production, prepared)

    print()


if __name__ == "__main__":
    anyio.run(main)
