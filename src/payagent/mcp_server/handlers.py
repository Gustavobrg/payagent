"""The six tool implementations: plain synchronous functions, `deps` first, zero `mcp` imports.

Mirrors the shape `rag/tools.py` established — business logic as plain functions, with the
framework adapter kept in a separate module (`dispatch.py`, `server.py`). Every invariant lives
here, so the contract suite can exercise all of it without a transport, a session, or an event loop.

Two boundaries are drawn deliberately:

**`policy/` decides what is *allowed*; this module decides what is *well-formed and bound*.** Some
checks therefore happen twice — a quote's expiry is checked here *and* will be checked by the real
policy engine. That duplication is intentional: it means an allow-all engine (a bug, or the real
engine mid-implementation) still cannot settle an expired quote or a mandate that does not match
its quote.

**Nothing here calls `engine.decide`.** Authorization goes through `payagent.policy.evaluate`,
which converts a raising, malformed, or mismatched decision into a denial. The effect helpers take
the resulting `EffectGrant` as a required positional argument, so an effect is unreachable from a
code path that never obtained an `Allow`.

`PolicyContext.step_up_satisfied` is hard-coded `False`. It must only ever be populated from
server-side session state — a request field asserting "step-up already done" is exactly the P5
evasion the step-up requirement exists to stop, which is why no request model has one.
"""

from __future__ import annotations

from datetime import timedelta

from payagent.mandates import IntentMandate, PaymentMandate, summarize
from payagent.mcp_server.catalog import lookup_sku, search_catalog_records
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.errors import ToolErrorCode, ToolErrorPayload, tool_error
from payagent.mcp_server.merchant_sim import (
    MerchantDeclined,
    MerchantTimeout,
    MerchantUnavailable,
)
from payagent.mcp_server.quotes import build_quote
from payagent.mcp_server.schemas import (
    CatalogHit,
    CreateIntentMandateRequest,
    CreateIntentMandateResult,
    CreatePaymentMandateRequest,
    CreatePaymentMandateResult,
    ExecuteSettlementRequest,
    ExecuteSettlementResult,
    GetQuoteRequest,
    GetQuoteResult,
    RefundRequest,
    RefundResult,
    SearchCatalogRequest,
    SearchCatalogResult,
)
from payagent.observability.logging import get_logger
from payagent.policy import Allow, EffectGrant, PolicyAction, PolicyContext, evaluate

logger = get_logger(__name__)

# Payment mandates are valid only as long as the quote they bind, so the mandate cannot outlive the
# price it authorizes.
_INDETERMINATE_MERCHANT_ERRORS = (MerchantTimeout,)


class HandlerFailure(Exception):
    """A structured, expected failure. Carries a payload; never a message built from caller input.

    `indeterminate` marks the case where the effect may or may not have landed (a merchant
    timeout). `dispatch.py` releases the idempotency reservation for those so a retry can proceed,
    and keeps it for deterministic failures so a denial replays instead of re-asking policy.
    """

    def __init__(self, payload: ToolErrorPayload, *, indeterminate: bool = False) -> None:
        super().__init__(payload.code.value)
        self.payload = payload
        self.indeterminate = indeterminate


def _fail(code: ToolErrorCode, *, indeterminate: bool = False, **kwargs) -> HandlerFailure:
    return HandlerFailure(tool_error(code, **kwargs), indeterminate=indeterminate)


def _authorize(
    deps: ToolDeps, action: PolicyAction, context: PolicyContext
) -> EffectGrant:
    """Evaluate policy and return the grant, or raise a structured denial.

    The only route to an `EffectGrant` in this module. Callers cannot reach an effect without one.
    """
    decision = evaluate(deps.policy, action, context)
    logger.info(
        "policy_decision",
        action=action.value,
        allowed=isinstance(decision, Allow),
        denial_code=None if isinstance(decision, Allow) else decision.denial.code.value,
    )
    if isinstance(decision, Allow):
        return decision.grant
    raise _fail(ToolErrorCode.POLICY_DENIED, denial=decision.denial)


def _translate_merchant_error(exc: Exception) -> HandlerFailure:
    """Map a merchant failure to a structured error, preserving whether the outcome is knowable."""
    if isinstance(exc, MerchantDeclined):
        return _fail(ToolErrorCode.MERCHANT_DECLINED)
    if isinstance(exc, MerchantUnavailable):
        return _fail(ToolErrorCode.MERCHANT_UNAVAILABLE)
    if isinstance(exc, _INDETERMINATE_MERCHANT_ERRORS):
        return _fail(ToolErrorCode.MERCHANT_TIMEOUT, indeterminate=True)
    return _fail(ToolErrorCode.INTERNAL_ERROR)


# ------------------------------------------------------------------------------ read-only tools


def handle_search_catalog(
    deps: ToolDeps, request: SearchCatalogRequest
) -> SearchCatalogResult:
    """Rank catalog entries. No policy consultation — this decides relevance, not permission."""
    records = search_catalog_records(
        deps.retriever,
        request.query,
        top_k=request.top_k,
        category=request.category,
        merchant_id=request.merchant_id,
        min_price_cents=request.min_price_cents,
        max_price_cents=request.max_price_cents,
    )

    hits = [
        CatalogHit(
            sku=record.sku,
            name=record.name,
            category=record.category,
            price_cents=record.price_cents,
            currency=record.currency,
            merchant_id=record.merchant_id,
            in_stock=record.in_stock,
            score=score,
            chunk_id=record.chunk_id,
            # The prose leaves this layer only wrapped (invariant I5). Structured fields above
            # come from the Qdrant payload, so poisoned text cannot influence a price.
            text_untrusted=record.as_untrusted_text(score=score),
        )
        for record, score in records
    ]
    return SearchCatalogResult(results=hits, returned=len(hits))


def handle_get_quote(deps: ToolDeps, request: GetQuoteRequest) -> GetQuoteResult:
    """Price one purchase from the catalog payload. The only authoritative source of an amount.

    Reuses an unexpired quote with the same fingerprint so a re-quote returns the same `quote_id`
    rather than silently refreshing an expiry the caller should have respected.
    """
    record = lookup_sku(deps.retriever, request.sku)
    if record is None:
        raise _fail(ToolErrorCode.UNKNOWN_SKU)
    if not record.in_stock:
        raise _fail(ToolErrorCode.OUT_OF_STOCK)
    if record.currency != request.currency:
        # An assertion mismatch, never a conversion.
        raise _fail(ToolErrorCode.CURRENCY_MISMATCH)

    now = deps.clock()
    quote = build_quote(
        sku=record.sku,
        name=record.name,
        category=record.category,
        quantity=request.quantity,
        unit_price_cents=record.price_cents,
        currency=record.currency,
        merchant_id=record.merchant_id,
        issued_at=now,
        ttl_seconds=deps.quote_ttl_seconds,
    )

    existing = deps.quotes.get(quote.quote_id)
    if existing is not None and not existing.is_expired(now):
        quote = existing
    else:
        deps.quotes.put(quote)

    return GetQuoteResult(
        quote_id=quote.quote_id,
        sku=quote.sku,
        name=quote.name,
        quantity=quote.quantity,
        unit_price_cents=quote.unit_price_cents,
        amount_cents=quote.amount_cents,
        currency=quote.currency,
        merchant_id=quote.merchant_id,
        category=quote.category,
        issued_at=quote.issued_at,
        expires_at=quote.expires_at,
    )


# ---------------------------------------------------------------------------- side-effect tools


def handle_create_intent_mandate(
    deps: ToolDeps, request: CreateIntentMandateRequest
) -> CreateIntentMandateResult:
    """Record a spending scope, if policy allows it."""
    now = deps.clock()
    grant = _authorize(
        deps,
        PolicyAction.CREATE_INTENT_MANDATE,
        PolicyContext(
            now=now,
            # An intent authorizes a ceiling; policy evaluates it as the amount at stake.
            amount_cents=request.max_amount_cents,
            currency=request.currency,
            intent_mandate_max_amount_cents=request.max_amount_cents,
            intent_mandate_allowed_categories=tuple(request.allowed_categories),
            intent_mandate_allowed_merchant_ids=(
                tuple(request.allowed_merchant_ids)
                if request.allowed_merchant_ids is not None
                else None
            ),
            # Populated from server-side session state only, never from a request field (P5).
            step_up_satisfied=False,
        ),
    )

    mandate = _issue_intent_mandate(deps, grant, request, now)
    summary = summarize(mandate)
    return CreateIntentMandateResult(
        intent_mandate_id=mandate.intent_mandate_id,
        status="issued",
        purpose=mandate.purpose,
        max_amount_cents=mandate.max_amount_cents,
        currency=mandate.currency,
        allowed_categories=list(mandate.allowed_categories),
        allowed_merchant_ids=(
            list(mandate.allowed_merchant_ids) if mandate.allowed_merchant_ids else None
        ),
        issued_at=mandate.issued_at,
        expires_at=mandate.expires_at,
        summary=summary.scope,
    )


def _issue_intent_mandate(
    deps: ToolDeps, grant: EffectGrant, request: CreateIntentMandateRequest, now
) -> IntentMandate:
    """Effect: mint and store an intent mandate. `grant` proves it was authorized."""
    mandate = IntentMandate(
        intent_mandate_id=deps.id_factory("IM"),
        purpose=request.purpose,
        max_amount_cents=request.max_amount_cents,
        currency=request.currency,
        allowed_categories=tuple(request.allowed_categories),
        allowed_merchant_ids=(
            tuple(request.allowed_merchant_ids)
            if request.allowed_merchant_ids is not None
            else None
        ),
        issued_at=now,
        expires_at=now + timedelta(seconds=request.expires_in_seconds),
    )
    issued = deps.mandate_authority.issue_intent(mandate)
    deps.mandate_store.put_intent(issued)
    return issued


def handle_create_payment_mandate(
    deps: ToolDeps, request: CreatePaymentMandateRequest
) -> CreatePaymentMandateResult:
    """Bind an intent mandate to a quote. The amount comes from the quote, always."""
    now = deps.clock()

    intent = deps.mandate_store.get_intent(request.intent_mandate_id)
    if intent is None:
        raise _fail(ToolErrorCode.INTENT_MANDATE_NOT_FOUND)
    if intent.is_expired(now):
        raise _fail(ToolErrorCode.INTENT_MANDATE_EXPIRED)

    quote = deps.quotes.get(request.quote_id)
    if quote is None:
        raise _fail(ToolErrorCode.QUOTE_NOT_FOUND)
    if quote.is_expired(now):
        raise _fail(ToolErrorCode.QUOTE_EXPIRED)

    # The `expected_` fields are assertions. A mismatch aborts rather than adjusting anything —
    # which is what turns an injected or hallucinated amount into a countable rejection (P2).
    if quote.currency != request.expected_currency:
        raise _fail(ToolErrorCode.CURRENCY_MISMATCH)
    if quote.amount_cents != request.expected_amount_cents:
        raise _fail(ToolErrorCode.AMOUNT_MISMATCH)

    grant = _authorize(
        deps,
        PolicyAction.CREATE_PAYMENT_MANDATE,
        PolicyContext(
            now=now,
            amount_cents=quote.amount_cents,
            currency=quote.currency,
            merchant_id=quote.merchant_id,
            category=quote.category,
            intent_mandate_id=intent.intent_mandate_id,
            intent_mandate_expires_at=intent.expires_at,
            intent_mandate_max_amount_cents=intent.max_amount_cents,
            intent_mandate_allowed_categories=intent.allowed_categories,
            intent_mandate_allowed_merchant_ids=intent.allowed_merchant_ids,
            quote_id=quote.quote_id,
            quote_expires_at=quote.expires_at,
            step_up_satisfied=False,
        ),
    )

    mandate = _issue_payment_mandate(deps, grant, intent, quote)
    return CreatePaymentMandateResult(
        payment_mandate_id=mandate.payment_mandate_id,
        intent_mandate_id=mandate.intent_mandate_id,
        quote_id=mandate.quote_id,
        status="issued",
        amount_cents=mandate.amount_cents,
        currency=mandate.currency,
        merchant_id=mandate.merchant_id,
        sku=mandate.sku,
        quantity=mandate.quantity,
        issued_at=mandate.issued_at,
        expires_at=mandate.expires_at,
        summary=summarize(mandate).scope,
    )


def _issue_payment_mandate(
    deps: ToolDeps, grant: EffectGrant, intent: IntentMandate, quote
) -> PaymentMandate:
    """Effect: mint and store a payment mandate. `grant` proves it was authorized."""
    mandate = PaymentMandate(
        payment_mandate_id=deps.id_factory("PM"),
        intent_mandate_id=intent.intent_mandate_id,  # POL-009 §3: mandatory parent reference
        quote_id=quote.quote_id,
        # Copied from the quote, never from a request field.
        amount_cents=quote.amount_cents,
        currency=quote.currency,
        merchant_id=quote.merchant_id,
        sku=quote.sku,
        quantity=quote.quantity,
        issued_at=quote.issued_at,
        # The mandate cannot outlive the price it authorizes.
        expires_at=quote.expires_at,
    )
    issued = deps.mandate_authority.issue_payment(mandate)
    deps.mandate_store.put_payment(issued)
    return issued


def handle_execute_settlement(
    deps: ToolDeps, request: ExecuteSettlementRequest
) -> ExecuteSettlementResult:
    """Move the money. Irreversible, and gated by every check invariant I3 names.

    Order matters: the mandate is resolved, re-bound to its quote, and verified *before* policy is
    consulted, and the merchant is only reached after both agree. Steps 2-4 re-derive facts the
    real policy engine will also check — deliberately, so a permissive engine still cannot settle
    an expired quote or a mandate that disagrees with its quote.
    """
    now = deps.clock()

    # 1. The mandate must exist and still be valid.
    mandate = deps.mandate_store.get_payment(request.payment_mandate_id)
    if mandate is None:
        raise _fail(ToolErrorCode.PAYMENT_MANDATE_NOT_FOUND)
    if mandate.is_expired(now):
        raise _fail(ToolErrorCode.PAYMENT_MANDATE_EXPIRED)

    # 2. POL-009 §3: without a resolvable parent intent there is no approved intent to point at.
    intent = deps.mandate_store.get_intent(mandate.intent_mandate_id)
    if intent is None:
        raise _fail(ToolErrorCode.INTENT_MANDATE_NOT_FOUND)

    # 3. POL-010: the quote is re-checked at settlement time, not just at mandate time.
    quote = deps.quotes.get(mandate.quote_id)
    if quote is None:
        raise _fail(ToolErrorCode.QUOTE_NOT_FOUND)
    if quote.is_expired(now):
        raise _fail(ToolErrorCode.QUOTE_EXPIRED)

    # 4. The mandate must still agree with the quote it references.
    if mandate.amount_cents != quote.amount_cents or mandate.currency != quote.currency:
        raise _fail(ToolErrorCode.MANDATE_SCOPE_MISMATCH)

    # 5. The caller's assertions must match the authoritative amount.
    if request.expected_currency != quote.currency:
        raise _fail(ToolErrorCode.CURRENCY_MISMATCH)
    if request.expected_amount_cents != quote.amount_cents:
        raise _fail(ToolErrorCode.AMOUNT_MISMATCH)

    # 6. Invariant I3: no settlement without a server-verified mandate. The authority shipped in
    #    `src/` verifies nothing, so this is where settlement stops until signing lands.
    verification = deps.mandate_authority.verify_for_settlement(mandate, now=now)
    if not verification.verified:
        logger.warning(
            "settlement_mandate_unverified",
            payment_mandate_id=mandate.payment_mandate_id,
            code=verification.code,
        )
        raise _fail(ToolErrorCode.MANDATE_NOT_VERIFIED)

    # 7. Only now is authorization asked for.
    grant = _authorize(
        deps,
        PolicyAction.EXECUTE_SETTLEMENT,
        PolicyContext(
            now=now,
            amount_cents=quote.amount_cents,
            currency=quote.currency,
            merchant_id=quote.merchant_id,
            category=quote.category,
            intent_mandate_id=intent.intent_mandate_id,
            intent_mandate_expires_at=intent.expires_at,
            intent_mandate_max_amount_cents=intent.max_amount_cents,
            intent_mandate_allowed_categories=intent.allowed_categories,
            intent_mandate_allowed_merchant_ids=intent.allowed_merchant_ids,
            payment_mandate_id=mandate.payment_mandate_id,
            quote_id=quote.quote_id,
            quote_expires_at=quote.expires_at,
            step_up_satisfied=False,
        ),
    )

    # 8. And only now is money moved.
    try:
        charge = deps.merchant.settle(
            grant,
            idempotency_key=request.idempotency_key,
            payment_mandate_id=mandate.payment_mandate_id,
            amount_cents=quote.amount_cents,
            currency=quote.currency,
            merchant_id=quote.merchant_id,
        )
    except Exception as exc:
        raise _translate_merchant_error(exc) from exc

    return ExecuteSettlementResult(
        settlement_id=charge.charge_id,
        payment_mandate_id=mandate.payment_mandate_id,
        status="settled",
        amount_cents=charge.amount_cents,
        currency=charge.currency,
        merchant_id=charge.merchant_id,
        settled_at=charge.created_at,
    )


def handle_refund(deps: ToolDeps, request: RefundRequest) -> RefundResult:
    """Reverse a settlement in full. No amount parameter — the settlement record decides it."""
    now = deps.clock()

    charge = deps.merchant.get_charge(request.settlement_id)
    if charge is None:
        raise _fail(ToolErrorCode.SETTLEMENT_NOT_FOUND)
    if deps.merchant.is_refunded(request.settlement_id):
        raise _fail(ToolErrorCode.SETTLEMENT_ALREADY_REFUNDED)

    # Assertions against the settlement record, exactly as elsewhere on this surface.
    if request.expected_currency != charge.currency:
        raise _fail(ToolErrorCode.CURRENCY_MISMATCH)
    if request.expected_amount_cents != charge.amount_cents:
        raise _fail(ToolErrorCode.AMOUNT_MISMATCH)

    grant = _authorize(
        deps,
        PolicyAction.REFUND,
        PolicyContext(
            now=now,
            amount_cents=charge.amount_cents,
            currency=charge.currency,
            merchant_id=charge.merchant_id,
            settlement_id=charge.charge_id,
            settled_amount_cents=charge.amount_cents,
            refund_reason_code=request.reason_code,
            step_up_satisfied=False,
        ),
    )

    try:
        refund = deps.merchant.refund(
            grant,
            idempotency_key=request.idempotency_key,
            charge_id=charge.charge_id,
            reason_code=request.reason_code,
        )
    except ValueError as exc:
        raise _fail(ToolErrorCode.SETTLEMENT_ALREADY_REFUNDED) from exc
    except KeyError as exc:
        raise _fail(ToolErrorCode.SETTLEMENT_NOT_FOUND) from exc
    except Exception as exc:
        raise _translate_merchant_error(exc) from exc

    return RefundResult(
        refund_id=refund.refund_id,
        settlement_id=charge.charge_id,
        status="refunded",
        amount_cents=refund.amount_cents,
        currency=refund.currency,
        reason_code=refund.reason_code,
        refunded_at=refund.created_at,
    )


HANDLERS = {
    "search_catalog": handle_search_catalog,
    "get_quote": handle_get_quote,
    "create_intent_mandate": handle_create_intent_mandate,
    "create_payment_mandate": handle_create_payment_mandate,
    "execute_settlement": handle_execute_settlement,
    "refund": handle_refund,
}
