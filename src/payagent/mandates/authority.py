"""Issuing and verifying mandates — the seam a real signing authority will fill.

Invariant I3 requires four things checked server-side before settlement: signature, scope,
amount, expiration. Two of them can only be answered by an authority holding key material
(signature, and the mandate's own expiry as it was signed); the other two can only be answered
by whoever holds the quote and settlement records. So the split is deliberate:

- `MandateAuthority.verify_for_settlement` → signature and mandate expiry.
- `mcp_server/handlers.py` → scope and amount, by re-deriving them from the quote.

Keeping quote-binding out of here also keeps the dependency direction honest: `mandates/` never
imports `mcp_server/`.

The only implementation shipped in `src/` is `UnsignedMandateAuthority`, which issues records
but verifies **nothing**. Real JWS/Ed25519/did:key signing lands in a later block. Until then
the honest answer to "is this mandate verified?" is no, so settlement is impossible from any
code path in `src/` — I3 upheld by refusing rather than by pretending the check passed. That
failure mode is loud: settlement simply cannot happen. An HMAC placeholder was considered and
rejected for the opposite reason — it would look production-grade while having no key rotation
and no did:key resolution, and every test writable today would pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from payagent.mandates.models import IntentMandate, PaymentMandate


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
