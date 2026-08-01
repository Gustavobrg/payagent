"""Tests for the `settle` node: exactly one `execute_settlement` attempt, no retry loop, and a
denial (policy or merchant) is terminal."""

from __future__ import annotations

import pytest
from doubles import AllowAllPolicyEngine, AlwaysVerifyingAuthority

from payagent.graph.nodes.confirm import make_confirm_node
from payagent.graph.nodes.mandate import make_mandate_node
from payagent.graph.nodes.quote import make_quote_node
from payagent.graph.nodes.settle import make_settle_node
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.merchant_sim import MerchantBehavior, MerchantSim
from payagent.policy import (
    Allow,
    DenialCode,
    Deny,
    EffectGrant,
    PolicyAction,
    PolicyDenial,
    PolicyEngine,
)
from payagent.rag.tools import ToolChunk


class _AllowExceptSettlement(PolicyEngine):
    """Allows mandate creation but denies `execute_settlement` — a settlement window that
    closed between mandate issuance and settlement, modeled directly rather than through
    `RulesPolicyEngine` (which would apply the same check at mandate time too, never letting
    a mandate reach this state in the first place — see `test_confirm_node.py`'s docstring)."""

    def decide(self, action, context):
        if action is PolicyAction.EXECUTE_SETTLEMENT:
            return Deny(PolicyDenial(DenialCode.POLICY_NOT_CONFIGURED, "settlement window closed"))
        return Allow(
            EffectGrant(
                action=action,
                amount_cents=context.amount_cents,
                currency=context.currency,
                merchant_id=context.merchant_id,
                decided_at=context.now,
            )
        )


def _confirmed_state(deps) -> PurchaseState:
    """A state that has passed through real `quote`, `mandate`, and `confirm`."""
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[ToolChunk(chunk_id="SKU-SPEAKER", source="catalog", score=0.9, text="<u/>")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )
    state = make_quote_node(deps)(state)
    state = make_mandate_node(deps)(state)
    return make_confirm_node(deps)(state)


def _mandated_state_without_confirm(deps) -> PurchaseState:
    """Like `_confirmed_state`, but stops right after `mandate` — for a policy engine that
    would deny `confirm`'s own pre-check too, so `settle`'s independent check can be observed
    on its own."""
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[ToolChunk(chunk_id="SKU-SPEAKER", source="catalog", score=0.9, text="<u/>")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )
    state = make_quote_node(deps)(state)
    return make_mandate_node(deps)(state)


def test_settles_and_records_the_result(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = _confirmed_state(deps)
    assert state.status == "in_progress"  # sanity: confirm let it through

    result = make_settle_node(deps)(state)

    assert result.status == "settled"
    assert result.settlement_result["ok"] is True
    assert result.settlement_result["data"]["amount_cents"] == 4999
    assert len(deps.merchant.charges) == 1


def test_policy_denial_is_terminal_and_settles_nothing(build_deps):
    """`settle` re-consults policy itself and can independently refuse, even though `mandate`
    (and, in this scenario, `confirm`) already allowed getting this far — the tool call is the
    authoritative check, not whatever an earlier node concluded."""
    deps = build_deps(
        policy=_AllowExceptSettlement(), mandate_authority=AlwaysVerifyingAuthority()
    )
    state = _mandated_state_without_confirm(deps)
    assert state.status == "in_progress"  # mandate succeeded — a real payment mandate exists

    result = make_settle_node(deps)(state)

    assert result.status == "settlement_denied"
    assert result.settlement_result["ok"] is False
    assert result.settlement_result["error"]["code"] == "POLICY_DENIED"
    assert len(deps.merchant.charges) == 0


def test_merchant_decline_is_terminal(build_deps, clock):
    merchant = MerchantSim(clock=clock, id_factory=lambda p: f"{p}-1", behavior=MerchantBehavior.DECLINE)
    deps = build_deps(
        policy=AllowAllPolicyEngine(),
        mandate_authority=AlwaysVerifyingAuthority(),
        merchant=merchant,
    )
    state = _confirmed_state(deps)

    result = make_settle_node(deps)(state)

    assert result.status == "settlement_denied"
    assert result.settlement_result["error"]["code"] == "MERCHANT_DECLINED"


def test_never_retries_automatically(build_deps):
    """One `call_tool` per invocation: calling the node function itself is the only way a
    second attempt could happen, and nothing inside it does that on its own."""
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = _confirmed_state(deps)

    make_settle_node(deps)(state)

    assert len(deps.merchant.calls) == 1


def test_raises_without_an_issued_mandate(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = PurchaseState(user_request="fone bluetooth", quote_amount_cents=4999)

    with pytest.raises(IllegalTransitionError):
        make_settle_node(deps)(state)


def test_is_a_no_op_while_awaiting_step_up(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = PurchaseState(user_request="fone bluetooth", status="awaiting_step_up")

    result = make_settle_node(deps)(state)

    assert result == state
    assert len(deps.merchant.calls) == 0
