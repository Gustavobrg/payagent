"""Tests for the `mandate` node: creates intent then payment mandate in sequence, and any
`Deny` — from either call, for any reason — is terminal with no workaround attempted."""

from __future__ import annotations

import pytest
from doubles import AllowAllPolicyEngine, AlwaysVerifyingAuthority, RecordingPolicyEngine

from payagent.graph.nodes.mandate import make_mandate_node
from payagent.graph.nodes.quote import make_quote_node
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.policy import DenyAllPolicyEngine
from payagent.rag.tools import ToolChunk


def _quoted_state(deps) -> PurchaseState:
    """A state that has already passed through a real `quote` node."""
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[ToolChunk(chunk_id="SKU-SPEAKER", source="catalog", score=0.9, text="<u/>")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )
    return make_quote_node(deps)(state)


def test_issues_intent_and_payment_mandate_when_policy_allows(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = _quoted_state(deps)

    result = make_mandate_node(deps)(state)

    assert result.status == "in_progress"
    assert result.mandate["ok"] is True
    assert result.mandate["data"]["payment_mandate_id"]
    assert result.mandate["data"]["amount_cents"] == 4999


def test_deny_on_intent_creation_is_terminal_and_never_attempts_payment_mandate(build_deps):
    engine = RecordingPolicyEngine(DenyAllPolicyEngine())
    deps = build_deps(policy=engine, mandate_authority=AlwaysVerifyingAuthority())
    state = _quoted_state(deps)

    result = make_mandate_node(deps)(state)

    assert result.status == "mandate_denied"
    assert result.mandate is None
    assert result.policy_decision["code"] == "POLICY_NOT_CONFIGURED"
    # Only one policy consultation happened — create_payment_mandate was never attempted.
    assert [action.value for action, _ in engine.calls] == ["create_intent_mandate"]


def test_deny_on_payment_mandate_creation_is_terminal(build_deps):
    """Intent succeeds (allowed), but the payment mandate itself is denied — e.g. step-up."""

    class DenyPaymentOnly(AllowAllPolicyEngine):
        def decide(self, action, context):
            from payagent.policy import DenialCode, Deny, PolicyAction, PolicyDenial

            if action is PolicyAction.CREATE_PAYMENT_MANDATE:
                return Deny(PolicyDenial(DenialCode.STEP_UP_REQUIRED, "needs step-up", "POL-003"))
            return super().decide(action, context)

    deps = build_deps(policy=DenyPaymentOnly(), mandate_authority=AlwaysVerifyingAuthority())
    state = _quoted_state(deps)

    result = make_mandate_node(deps)(state)

    assert result.status == "mandate_denied"
    assert result.mandate is None
    assert result.policy_decision["code"] == "STEP_UP_REQUIRED"


def test_raises_without_a_successful_quote(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = PurchaseState(user_request="fone bluetooth")

    with pytest.raises(IllegalTransitionError):
        make_mandate_node(deps)(state)


def test_is_a_no_op_when_the_purchase_already_resolved(build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    state = PurchaseState(user_request="oi", status="answered_directly", direct_response="Oi!")

    result = make_mandate_node(deps)(state)

    assert result == state


def test_mandate_amount_currency_merchant_category_always_come_from_the_quote(build_deps):
    """The node never invents scope: it is copied verbatim from `state.quote`."""
    engine = RecordingPolicyEngine(AllowAllPolicyEngine())
    deps = build_deps(policy=engine, mandate_authority=AlwaysVerifyingAuthority())
    state = _quoted_state(deps)

    make_mandate_node(deps)(state)

    intent_call = next(ctx for action, ctx in engine.calls if action.value == "create_intent_mandate")
    payment_call = next(ctx for action, ctx in engine.calls if action.value == "create_payment_mandate")
    assert intent_call.amount_cents == state.quote_amount_cents
    assert payment_call.amount_cents == state.quote_amount_cents
    assert payment_call.merchant_id == state.quote["data"]["merchant_id"]
    assert payment_call.category == state.quote["data"]["category"]
