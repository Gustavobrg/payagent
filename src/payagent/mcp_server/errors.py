"""Structured tool errors, with static messages that never echo caller input.

Every failure a tool can report has a code from `ToolErrorCode` and a **static** message authored
here. That is invariant I2 held at the source rather than by a redaction pass: if no message ever
interpolates caller data, no message can leak a PAN, a CVV, or anything else a user pasted into a
field.

Pydantic's own error rendering is the specific hazard. `ValidationError.errors()` includes the
rejected `input` by default, and for several error types the human-readable `msg` interpolates it
too — so `from_validation_error` drops `msg` entirely and keeps only the field path and the
machine-readable error type. A path plus a code carries every bit of diagnostic value a caller
needs to fix the call, and carries none of their text.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from payagent.policy import PolicyDenial


class ToolErrorCode(StrEnum):
    """Why a tool call failed. Stable strings — the eval harness switches on these."""

    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    UNKNOWN_SKU = "UNKNOWN_SKU"
    OUT_OF_STOCK = "OUT_OF_STOCK"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
    QUOTE_EXPIRED = "QUOTE_EXPIRED"
    INTENT_MANDATE_NOT_FOUND = "INTENT_MANDATE_NOT_FOUND"
    INTENT_MANDATE_EXPIRED = "INTENT_MANDATE_EXPIRED"
    PAYMENT_MANDATE_NOT_FOUND = "PAYMENT_MANDATE_NOT_FOUND"
    PAYMENT_MANDATE_EXPIRED = "PAYMENT_MANDATE_EXPIRED"
    MANDATE_NOT_VERIFIED = "MANDATE_NOT_VERIFIED"
    MANDATE_SCOPE_MISMATCH = "MANDATE_SCOPE_MISMATCH"
    POLICY_DENIED = "POLICY_DENIED"
    IDEMPOTENCY_KEY_REUSED = "IDEMPOTENCY_KEY_REUSED"
    IDEMPOTENCY_IN_PROGRESS = "IDEMPOTENCY_IN_PROGRESS"
    SETTLEMENT_NOT_FOUND = "SETTLEMENT_NOT_FOUND"
    SETTLEMENT_ALREADY_REFUNDED = "SETTLEMENT_ALREADY_REFUNDED"
    MERCHANT_DECLINED = "MERCHANT_DECLINED"
    MERCHANT_TIMEOUT = "MERCHANT_TIMEOUT"
    MERCHANT_UNAVAILABLE = "MERCHANT_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Static message per code. No entry may contain a placeholder or a format specifier — the test
# suite asserts that, because one interpolated message is all it takes to reopen the I2 leak.
_MESSAGES: dict[ToolErrorCode, str] = {
    ToolErrorCode.INVALID_ARGUMENTS: (
        "The arguments did not match this tool's schema. See field_errors for the paths and "
        "error types, then correct the call."
    ),
    ToolErrorCode.UNKNOWN_SKU: "No catalog entry exists for the requested SKU.",
    ToolErrorCode.OUT_OF_STOCK: "The requested SKU is out of stock.",
    ToolErrorCode.CURRENCY_MISMATCH: (
        "The expected currency does not match the server-side record. Currencies are never "
        "converted."
    ),
    ToolErrorCode.AMOUNT_MISMATCH: (
        "The expected amount does not match the server-side record. The recorded amount is "
        "authoritative and cannot be overridden."
    ),
    ToolErrorCode.QUOTE_NOT_FOUND: "No quote exists for the given quote_id. Request a fresh quote.",
    ToolErrorCode.QUOTE_EXPIRED: "The quote has expired. Request a fresh quote.",
    ToolErrorCode.INTENT_MANDATE_NOT_FOUND: "No intent mandate exists for the given identifier.",
    ToolErrorCode.INTENT_MANDATE_EXPIRED: "The intent mandate has expired.",
    ToolErrorCode.PAYMENT_MANDATE_NOT_FOUND: "No payment mandate exists for the given identifier.",
    ToolErrorCode.PAYMENT_MANDATE_EXPIRED: "The payment mandate has expired.",
    ToolErrorCode.MANDATE_NOT_VERIFIED: (
        "The mandate could not be verified server-side, so nothing was settled."
    ),
    ToolErrorCode.MANDATE_SCOPE_MISMATCH: (
        "The mandate does not match the quote it references, so nothing was settled."
    ),
    ToolErrorCode.POLICY_DENIED: (
        "The authorization policy denied this action. The denial is final and cannot be retried "
        "into an approval."
    ),
    ToolErrorCode.IDEMPOTENCY_KEY_REUSED: (
        "This idempotency_key was already used for a different request. Nothing was performed. "
        "Use a new key for a new operation, or resend the original arguments to retry."
    ),
    ToolErrorCode.IDEMPOTENCY_IN_PROGRESS: (
        "A call with this idempotency_key is still in progress. Retry shortly."
    ),
    ToolErrorCode.SETTLEMENT_NOT_FOUND: "No settlement exists for the given identifier.",
    ToolErrorCode.SETTLEMENT_ALREADY_REFUNDED: (
        "This settlement was already refunded in full. Nothing was performed."
    ),
    ToolErrorCode.MERCHANT_DECLINED: "The merchant declined the operation.",
    ToolErrorCode.MERCHANT_TIMEOUT: (
        "The merchant did not respond in time and the outcome is unknown. Retry with the same "
        "idempotency_key; it will not produce a second effect."
    ),
    ToolErrorCode.MERCHANT_UNAVAILABLE: "The merchant is unavailable. Retry later.",
    ToolErrorCode.INTERNAL_ERROR: "The tool failed for an internal reason. Nothing was performed.",
}

# Codes where retrying the identical call could plausibly succeed.
_RETRYABLE = frozenset(
    {
        ToolErrorCode.IDEMPOTENCY_IN_PROGRESS,
        ToolErrorCode.MERCHANT_TIMEOUT,
        ToolErrorCode.MERCHANT_UNAVAILABLE,
    }
)


class FieldError(BaseModel):
    """One schema violation: where it was, and what kind. Never the rejected value."""

    model_config = ConfigDict(extra="forbid")

    path: str
    code: str


class PolicyDenialPayload(BaseModel):
    """A policy denial, structured — never prose from a model."""

    model_config = ConfigDict(extra="forbid")

    code: str
    reason: str
    policy_ref: str | None = None


class ToolErrorPayload(BaseModel):
    """The error half of a tool result envelope."""

    model_config = ConfigDict(extra="forbid")

    code: ToolErrorCode
    message: str
    retryable: bool = False
    field_errors: list[FieldError] = []
    denial: PolicyDenialPayload | None = None


def tool_error(
    code: ToolErrorCode,
    *,
    field_errors: list[FieldError] | None = None,
    denial: PolicyDenial | None = None,
) -> ToolErrorPayload:
    """Build an error payload with the static message registered for `code`."""
    return ToolErrorPayload(
        code=code,
        message=_MESSAGES[code],
        retryable=code in _RETRYABLE,
        field_errors=field_errors or [],
        denial=(
            PolicyDenialPayload(
                code=denial.code.value, reason=denial.reason, policy_ref=denial.policy_ref
            )
            if denial is not None
            else None
        ),
    )


def from_validation_error(exc: ValidationError) -> ToolErrorPayload:
    """Convert a pydantic `ValidationError` into a structured payload with no caller input in it.

    `include_input=False` keeps the rejected value out, and `msg` is dropped rather than copied
    because pydantic interpolates the input into it for several error types.
    """
    field_errors = [
        FieldError(
            path=".".join(str(part) for part in entry["loc"]) or "(root)",
            code=entry["type"],
        )
        for entry in exc.errors(include_url=False, include_input=False, include_context=False)
    ]
    return tool_error(ToolErrorCode.INVALID_ARGUMENTS, field_errors=field_errors)
