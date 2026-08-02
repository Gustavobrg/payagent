"""One purchase, driven end-to-end and flushed to Langfuse Cloud, so you can see a real
session grouping graph-node, MCP-tool, and guardrail-rail spans in the dashboard.

Needs `OPENROUTER_API_KEY` (real `plan`/`retrieve` LLM calls and a real Llama Guard
classification) and `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` in `.env` — this hits real
network on both ends, unlike the rest of the test suite. Qdrant is in-memory, seeded from the
real catalog dataset (same pattern as `scratch_search.py` at the repo root), so it does NOT
need `docker compose up -d`.

Drives the graph step by step (`graph.step(...)`, mirroring `scripts/demo.py`/the eval
harness) rather than `run_guarded`, so it can auto-select the first retrieved SKU instead of
prompting — the point here is a complete, unattended trace (check_input rails -> plan ->
retrieve -> quote -> mandate -> confirm -> settle -> check_output rails), not the interactive
CLI flow. Policy/mandate wiring is a local permissive double, same convention as
`scratch_search.py`'s own — `AlwaysVerifyingAuthority` in particular is deliberately never
shipped in `src/` (invariant I3: no real code path can produce a verified mandate), so it's
defined here, standalone, for demonstration only.

    uv run python scripts/see_session_in_langfuse.py

Prints the session_id used; open your Langfuse project's Sessions view (or Traces, filtered
by that session ID) to see it.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from payagent.graph.graph import build_graph
from payagent.graph.respond import compose_response
from payagent.graph.state import PurchaseState
from payagent.guardrails.rails import build_rails, check_input, check_output
from payagent.mandates import (
    InMemoryMandateStore,
    MandateVerification,
    UnsignedMandateAuthority,
)
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.idempotency import InMemoryIdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import InMemoryQuoteStore
from payagent.mcp_server.step_up import InMemoryStepUpVerifier
from payagent.observability.logging import configure_logging
from payagent.observability.tracing import configure_tracing, flush, open_session_span, set_safe_io
from payagent.policy import Allow, EffectGrant, PolicyAction, PolicyContext, PolicyEngine
from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    create_collection_if_missing,
    ingest_chunks,
    load_catalog,
)
from payagent.rag.retriever import Retriever

REQUEST = "Quero comprar um fone de ouvido bluetooth"


class _AllowAllPolicyEngine(PolicyEngine):
    """Local demo double, not shipped in `src/` — grants everything so the run reaches
    `settled` and produces a full trace, mirroring `scratch_search.py`'s own convention."""

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


class _AlwaysVerifyingAuthority(UnsignedMandateAuthority):
    """Local demo double, not shipped in `src/` (invariant I3) — verifies every mandate so
    `settle` is reachable without real JWS signing."""

    def verify_for_settlement(self, mandate, *, now) -> MandateVerification:
        return MandateVerification(verified=True)


def _build_retriever() -> Retriever:
    client = QdrantClient(":memory:")
    create_collection_if_missing(client, CATALOG_COLLECTION)
    ingest_chunks(client, CATALOG_COLLECTION, chunk_catalog(load_catalog()), DeterministicEmbedder())
    return Retriever(client=client, use_deterministic_embedder=True)


def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY not configured. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
    )


def _build_deps(retriever: Retriever) -> ToolDeps:
    now = datetime.now(UTC)
    return ToolDeps(
        retriever=retriever,
        policy=_AllowAllPolicyEngine(),
        mandate_authority=_AlwaysVerifyingAuthority(),
        mandate_store=InMemoryMandateStore(),
        quotes=InMemoryQuoteStore(clock=lambda: now),
        merchant=MerchantSim(clock=lambda: now, id_factory=lambda prefix: f"{prefix}-demo"),
        idempotency=InMemoryIdempotencyStore(),
        clock=lambda: now,
        id_factory=lambda prefix: f"{prefix}-demo",
        step_up=InMemoryStepUpVerifier(id_factory=lambda prefix: f"{prefix}-demo"),
    )


def main() -> None:
    load_dotenv()
    configure_logging(stream=sys.stderr)
    configure_tracing()

    retriever = _build_retriever()
    deps = _build_deps(retriever)
    graph = build_graph(_build_llm(), retriever, deps)
    rails = build_rails()

    session_id = "demo-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"session_id: {session_id}", file=sys.stderr)

    with open_session_span("demo.see_session_in_langfuse", session_id=session_id) as root_span:
        input_result = check_input(rails, REQUEST)
        if not input_result.allowed:
            print(f"blocked by input rails: {input_result.refusal}", file=sys.stderr)
            set_safe_io(
                root_span,
                input=REQUEST,
                output={"status": "blocked_by_input_rails", "refusal": input_result.refusal},
            )
        else:
            state = PurchaseState(user_request=REQUEST)
            state = graph.step("plan", state)
            state = graph.step("retrieve", state)

            catalog_hits = [c for c in state.retrieved_chunks if c.source == "catalog"]
            if catalog_hits:
                state = state.model_copy(
                    update={"selected_sku": catalog_hits[0].chunk_id, "selected_quantity": 1}
                )
                state = graph.step("quote", state)
                state = graph.step("mandate", state)
                state = graph.step("confirm", state)
                state = graph.step("settle", state)

            response_text = compose_response(state)
            check_output(
                rails,
                user_message=REQUEST,
                bot_message=response_text,
                retrieved_chunks=state.retrieved_chunks,
            )
            print(f"final status: {state.status}", file=sys.stderr)
            set_safe_io(
                root_span,
                input=REQUEST,
                output={"status": state.status, "response": response_text},
            )

    flush()
    print(f"\nFlushed. Open Langfuse and look for session_id = {session_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
