"""Issuing and verifying mandates — the seam a real signing authority fills.

Invariant I3 requires four things checked server-side before settlement: signature, scope,
amount, expiration, plus the chain to the parent Intent Mandate (POL-009 §3). `Ed25519MandateAuthority`
answers all of them from material it alone holds:

- Signature and mandate expiry — verified from the JWS itself (Ed25519, EdDSA).
- Scope, amount, currency, merchant, category and the parent-intent chain — verified by decoding
  the *parent intent's own signed claims* and checking the payment mandate falls inside them, so
  the check cannot be fooled by a caller-supplied `IntentMandate` object that disagrees with what
  was actually signed.

`mcp_server/handlers.py` still separately re-derives scope and amount from the quote before this
is ever called — that duplication is intentional (see its module docstring) and stays even though
this authority now also checks scope: two independent code paths that must both agree is a
stronger property than either alone.

The signature itself never touches an `IntentMandate`/`PaymentMandate` field (P3) — it lives in a
`MandateSignatureStore`, keyed by mandate ID, injected into the authority. `mandates/` never
imports `mcp_server/`, so that store is defined in this package too (`mandates/store.py`).

`UnsignedMandateAuthority` remains shipped alongside `Ed25519MandateAuthority` as the fail-closed
fallback: if the signing key cannot be loaded, `mcp_server/__main__.py` falls back to it rather
than starting with a broken authority, so the honest answer to "is this mandate verified?" stays
"no" instead of silently becoming "yes".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from payagent.mandates.did_key import did_key_from_ed25519_public_key
from payagent.mandates.models import IntentMandate, PaymentMandate
from payagent.mandates.store import MandateSignatureStore

_JWS_ALGORITHM = "EdDSA"
# PyJWT's own `exp` check reads the real system clock. Every other expiry check in this codebase
# is against an injected `now` (the same reason `PolicyContext.now` and `ToolDeps.clock` exist),
# so that check is disabled here and done by hand against the caller's `now` instead.
_DECODE_OPTIONS = {"verify_exp": False, "verify_iat": False}


@dataclass(frozen=True, slots=True)
class MandateVerification:
    """Outcome of verifying a mandate. `verified=False` carries a code and a static reason."""

    verified: bool
    code: str | None = None
    reason: str | None = None


class MandateAuthority(Protocol):
    """Issues mandate records and verifies them before an irreversible effect."""

    def issue_intent(self, mandate: IntentMandate) -> IntentMandate:
        """Return the mandate as issued, attaching any authority-side material out of band."""
        ...

    def issue_payment(self, mandate: PaymentMandate) -> PaymentMandate:
        """Return the mandate as issued, attaching any authority-side material out of band."""
        ...

    def verify_for_settlement(
        self, mandate: PaymentMandate, *, now: datetime
    ) -> MandateVerification:
        """Verify signature and validity window. Scope and amount are the caller's to check."""
        ...


class UnsignedMandateAuthority:
    """Issues unsigned mandate records and verifies nothing.

    `verify_for_settlement` returns a hard negative for every input, including a well-formed,
    unexpired, amount-matching mandate: there is no signature to check, so nothing can honestly
    be called verified. Tests that need a settled purchase inject a verifying double, the same
    way `DeterministicEmbedder` stands in for the real embedder in `rag/` tests.
    """

    _UNVERIFIED_CODE = "MANDATE_SIGNATURE_UNVERIFIED"
    _UNVERIFIED_REASON = "The mandate carries no verifiable signature."

    def issue_intent(self, mandate: IntentMandate) -> IntentMandate:
        return mandate

    def issue_payment(self, mandate: PaymentMandate) -> PaymentMandate:
        return mandate

    def verify_for_settlement(
        self, mandate: PaymentMandate, *, now: datetime
    ) -> MandateVerification:
        return MandateVerification(
            verified=False,
            code=self._UNVERIFIED_CODE,
            reason=self._UNVERIFIED_REASON,
        )


def _intent_claims(mandate: IntentMandate, *, issuer: str) -> dict[str, Any]:
    return {
        "iss": issuer,
        "sub": mandate.intent_mandate_id,
        "typ": "intent_mandate",
        "intent_mandate_id": mandate.intent_mandate_id,
        "max_amount_cents": mandate.max_amount_cents,
        "currency": mandate.currency,
        "allowed_categories": list(mandate.allowed_categories),
        "allowed_merchant_ids": (
            list(mandate.allowed_merchant_ids) if mandate.allowed_merchant_ids is not None else None
        ),
        "iat": int(mandate.issued_at.timestamp()),
        "exp": int(mandate.expires_at.timestamp()),
    }


def _payment_claims(mandate: PaymentMandate, *, issuer: str) -> dict[str, Any]:
    return {
        "iss": issuer,
        "sub": mandate.payment_mandate_id,
        "typ": "payment_mandate",
        "payment_mandate_id": mandate.payment_mandate_id,
        "intent_mandate_id": mandate.intent_mandate_id,
        "quote_id": mandate.quote_id,
        "amount_cents": mandate.amount_cents,
        "currency": mandate.currency,
        "merchant_id": mandate.merchant_id,
        "category": mandate.category,
        "sku": mandate.sku,
        "quantity": mandate.quantity,
        "iat": int(mandate.issued_at.timestamp()),
        "exp": int(mandate.expires_at.timestamp()),
    }


class Ed25519MandateAuthority:
    """Signs mandates as Ed25519 JWTs (a JWS profile) and verifies them before settlement.

    Construction takes the private key and a `MandateSignatureStore` directly — never a file
    path or an environment variable name, matching every other collaborator in this codebase
    (`Retriever`, `MerchantSim`, ...). Reading `MANDATE_SIGNING_KEY_PATH` and loading the PEM
    file is `mcp_server/__main__.py`'s job, the one place allowed to do that kind of I/O.

    The `did:key` identifier (`self.did`) is derived from the public key and embedded as the
    JWT `iss` claim on everything this authority signs — decoding a token with this authority's
    public key already proves it was issued by this `did:key`, so no separate issuer lookup is
    needed.
    """

    def __init__(self, *, private_key: Ed25519PrivateKey, signatures: MandateSignatureStore) -> None:
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._did = did_key_from_ed25519_public_key(self._public_key)
        self._signatures = signatures

    @property
    def did(self) -> str:
        """The `did:key:z...` identifier this authority signs as."""
        return self._did

    def issue_intent(self, mandate: IntentMandate) -> IntentMandate:
        token = jwt.encode(
            _intent_claims(mandate, issuer=self._did), self._private_key, algorithm=_JWS_ALGORITHM
        )
        self._signatures.put(mandate.intent_mandate_id, token)
        return mandate

    def issue_payment(self, mandate: PaymentMandate) -> PaymentMandate:
        token = jwt.encode(
            _payment_claims(mandate, issuer=self._did), self._private_key, algorithm=_JWS_ALGORITHM
        )
        self._signatures.put(mandate.payment_mandate_id, token)
        return mandate

    def _decode(self, token: str) -> dict[str, Any] | None:
        """Decode and verify a stored token against this authority's own public key.

        Returns `None` for anything that does not verify: a bad signature (tampered payload,
        or signed by a different key entirely — the "belongs to another user/tenant" case),
        or a malformed token. Callers turn `None` into the appropriate structured code.
        """
        try:
            return jwt.decode(
                token, self._public_key, algorithms=[_JWS_ALGORITHM], options=_DECODE_OPTIONS
            )
        except jwt.PyJWTError:
            return None

    def verify_for_settlement(
        self, mandate: PaymentMandate, *, now: datetime
    ) -> MandateVerification:
        token = self._signatures.get(mandate.payment_mandate_id)
        if token is None:
            return MandateVerification(
                verified=False,
                code="MANDATE_SIGNATURE_MISSING",
                reason="The payment mandate carries no stored signature.",
            )

        claims = self._decode(token)
        if claims is None:
            return MandateVerification(
                verified=False,
                code="MANDATE_SIGNATURE_INVALID",
                reason="The payment mandate's signature does not verify.",
            )

        # Defense in depth: the signature alone already covers these fields, but a caller could
        # in principle pass a `PaymentMandate` object that disagrees with what was signed (e.g.
        # resolved from a different store than the one that issued it).
        if (
            claims.get("payment_mandate_id") != mandate.payment_mandate_id
            or claims.get("intent_mandate_id") != mandate.intent_mandate_id
            or claims.get("amount_cents") != mandate.amount_cents
            or claims.get("currency") != mandate.currency
            or claims.get("merchant_id") != mandate.merchant_id
            or claims.get("category") != mandate.category
        ):
            return MandateVerification(
                verified=False,
                code="MANDATE_SIGNATURE_INVALID",
                reason="The payment mandate does not match its stored signed claims.",
            )

        if mandate.is_expired(now):
            return MandateVerification(
                verified=False, code="MANDATE_EXPIRED", reason="The payment mandate has expired."
            )

        # POL-009 §3: the parent intent must resolve to a token this authority itself signed.
        parent_token = self._signatures.get(mandate.intent_mandate_id)
        parent_claims = self._decode(parent_token) if parent_token is not None else None
        if (
            parent_claims is None
            or parent_claims.get("typ") != "intent_mandate"
            or parent_claims.get("intent_mandate_id") != mandate.intent_mandate_id
        ):
            return MandateVerification(
                verified=False,
                code="MANDATE_CHAIN_BROKEN",
                reason="The parent intent mandate could not be resolved and verified.",
            )

        if now.timestamp() >= parent_claims["exp"]:
            return MandateVerification(
                verified=False,
                code="MANDATE_EXPIRED",
                reason="The parent intent mandate has expired.",
            )

        allowed_merchant_ids = parent_claims.get("allowed_merchant_ids")
        scope_ok = (
            mandate.amount_cents <= parent_claims["max_amount_cents"]
            and mandate.currency == parent_claims["currency"]
            and mandate.category in parent_claims["allowed_categories"]
            and (allowed_merchant_ids is None or mandate.merchant_id in allowed_merchant_ids)
        )
        if not scope_ok:
            return MandateVerification(
                verified=False,
                code="MANDATE_SCOPE_EXCEEDED",
                reason="The payment mandate falls outside the scope its intent mandate granted.",
            )

        return MandateVerification(verified=True)
