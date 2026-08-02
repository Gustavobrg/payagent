"""End-to-end purchase demo, driven from the CLI.

Runs the real purchase graph (`payagent.graph.graph.build_graph`) against the real MCP tool
deps (`payagent.mcp_server.__main__.build_deps_from_env` — the same composition root the MCP
server itself uses, so this demo authorizes and settles through the identical policy engine,
mandate authority, and step-up state), printing every state-machine transition as it happens:
which node ran, the resulting status, and — whenever a node produced a policy decision —
whether it was Allow or Deny, with the structured reason. Nothing here decides anything the
graph itself doesn't decide; this script only drives it and narrates the result.

Two points require a human because the graph will not fabricate them:

- SKU selection. `retrieve` returns candidate chunks, never a single confirmed purchase — this
  script lists the catalog hits and asks which one to buy (invariant I5: retrieved prose is
  data, never an instruction, so nothing here infers a choice from it).
- Step-up resolution. If `confirm` pauses with `status="awaiting_step_up"`, that is
  `PolicyContext.step_up_satisfied` being false with the rule requiring it (POL-003). This
  script simulates the separate WebAuthn channel by prompting for approval and calling
  `StepUpVerifier.issue_challenge`/`resolve_challenge` directly — never by feeding anything
  back through a tool argument, which is exactly the P5 evasion the requirement exists to stop.

Usage:
    uv run python scripts/demo.py "<pedido>"

Requires Qdrant reachable at `QDRANT_URL` (see `docs/deployment.md`) with the catalog/policies
collections ingested, and `OPENROUTER_API_KEY` configured for the planning/retrieval LLM.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from payagent.graph.graph import build_graph
from payagent.graph.state import PurchaseState
from payagent.mcp_server.__main__ import build_deps_from_env
from payagent.observability.logging import configure_logging
from payagent.observability.tracing import configure_tracing, open_session_span

_OUTCOME_BY_STATUS = {
    "answered_directly": None,
    "quote_failed": "Deny",
    "mandate_denied": "Deny",
    "awaiting_step_up": "Deny",
    "settlement_denied": "Deny",
    "settled": "Allow",
}


def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not configured. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
    )


def _print_header(step: str) -> None:
    print(f"\n== {step} ==")


def _print_denial(denial: dict[str, Any] | None) -> None:
    if denial is None:
        return
    ref = f" [{denial['policy_ref']}]" if denial.get("policy_ref") else ""
    print(f"  Deny ({denial['code']}){ref}: {denial['reason']}")


def _report(step: str, state: PurchaseState) -> None:
    """Print this transition's status and, when the node produced one, its Allow/Deny result."""
    _print_header(step)
    print(f"  status: {state.status}")

    if step == "plan" and state.status == "answered_directly":
        print(f"  {state.direct_response}")
        return

    if step == "retrieve":
        print(
            f"  {len(state.retrieved_chunks)} chunk(s), confidence={state.retrieval_confidence}"
        )
        return

    if step == "quote":
        if state.status == "quote_failed":
            error = state.quote["error"]
            print(f"  Deny ({error['code']}): {error['message']}")
        else:
            data = state.quote["data"]
            print(
                f"  Allow: {data['quantity']} x {data['sku']} = "
                f"{data['amount_cents']} {data['currency']} (quote_id={data['quote_id']})"
            )
        return

    if step in ("mandate", "confirm"):
        outcome = _OUTCOME_BY_STATUS.get(state.status)
        if outcome == "Allow" or (outcome is None and state.policy_decision is None):
            print("  Allow")
        else:
            _print_denial(state.policy_decision)
        return

    if step == "settle":
        if state.status == "settled":
            data = state.settlement_result["data"]
            print(
                f"  Allow: settlement_id={data['settlement_id']} "
                f"{data['amount_cents']} {data['currency']}"
            )
        else:
            error = state.settlement_result["error"]
            _print_denial(error.get("denial"))
            if error.get("denial") is None:
                print(f"  Deny ({error['code']}): {error['message']}")


def _prompt_sku(state: PurchaseState) -> tuple[str, int]:
    catalog_hits = [c for c in state.retrieved_chunks if c.source == "catalog"]
    if not catalog_hits:
        raise SystemExit("Nothing found in the catalog for this request.")

    print("\nFound:")
    for i, chunk in enumerate(catalog_hits, start=1):
        print(f"  {i}. {chunk.chunk_id}")

    choice = input(f"Pick a number [1-{len(catalog_hits)}]: ").strip()
    index = int(choice) - 1 if choice else 0
    sku = catalog_hits[index].chunk_id

    quantity_raw = input("Quantity [1]: ").strip()
    quantity = int(quantity_raw) if quantity_raw else 1
    return sku, quantity


def _resolve_step_up_out_of_band(deps) -> None:
    """Simulate the separate WebAuthn channel — never a tool argument (P5)."""
    approved = input("Step-up required. Approve via passkey? [y/N] ").strip().lower() == "y"
    if not approved:
        return
    now = deps.clock()
    challenge = deps.step_up.issue_challenge(now=now)
    deps.step_up.resolve_challenge(challenge.challenge_id, now=now)


def run_demo(request: str) -> PurchaseState:
    deps = build_deps_from_env()
    llm = _build_llm()
    graph = build_graph(llm, deps.retriever, deps)

    # One demo invocation is one session (one purchase) — `run_guarded` isn't used here (this
    # script drives the graph step by step for the interactive SKU/step-up prompts), so the
    # root span/session are opened directly here via the same shared helper `run_guarded` uses.
    with open_session_span("demo.purchase"):
        state = PurchaseState(user_request=request)

        state = graph.step("plan", state)
        _report("plan", state)
        if state.status == "answered_directly":
            return state

        state = graph.step("retrieve", state)
        _report("retrieve", state)
        if not state.retrieved_chunks:
            return state

        sku, quantity = _prompt_sku(state)
        state = state.model_copy(update={"selected_sku": sku, "selected_quantity": quantity})

        state = graph.step("quote", state)
        _report("quote", state)
        if state.status != "in_progress":
            return state

        state = graph.step("mandate", state)
        _report("mandate", state)
        if state.status != "in_progress":
            return state

        while True:
            state = graph.step("confirm", state)
            _report("confirm", state)
            if state.status != "awaiting_step_up":
                break
            _resolve_step_up_out_of_band(deps)

        if state.status != "in_progress":
            return state

        state = graph.step("settle", state)
        _report("settle", state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one purchase through the full graph.")
    parser.add_argument("pedido", help="Natural-language purchase request.")
    args = parser.parse_args()

    load_dotenv()
    configure_logging(stream=sys.stderr)
    configure_tracing()

    run_demo(args.pedido)


if __name__ == "__main__":
    main()
