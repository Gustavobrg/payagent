"""Where issued mandates live so a later step can resolve them by ID.

A miss returns `None` rather than raising: the caller turns it into a structured tool error
(`PAYMENT_MANDATE_NOT_FOUND`), and an exception here would end up in an error message echoed to
the client — the I2 path this project keeps closed by construction.
"""

from __future__ import annotations

from typing import Protocol

from payagent.mandates.models import IntentMandate, PaymentMandate


class MandateStore(Protocol):
    """Read/write access to issued mandates, keyed by their IDs."""

    def put_intent(self, mandate: IntentMandate) -> None: ...

    def put_payment(self, mandate: PaymentMandate) -> None: ...

    def get_intent(self, intent_mandate_id: str) -> IntentMandate | None: ...

    def get_payment(self, payment_mandate_id: str) -> PaymentMandate | None: ...


class InMemoryMandateStore:
    """Process-local mandate store. Durable storage is a later block's concern."""

    def __init__(self) -> None:
        self._intents: dict[str, IntentMandate] = {}
        self._payments: dict[str, PaymentMandate] = {}

    def put_intent(self, mandate: IntentMandate) -> None:
        self._intents[mandate.intent_mandate_id] = mandate

    def put_payment(self, mandate: PaymentMandate) -> None:
        self._payments[mandate.payment_mandate_id] = mandate

    def get_intent(self, intent_mandate_id: str) -> IntentMandate | None:
        return self._intents.get(intent_mandate_id)

    def get_payment(self, payment_mandate_id: str) -> PaymentMandate | None:
        return self._payments.get(payment_mandate_id)

    @property
    def intent_count(self) -> int:
        """Number of issued intent mandates — lets a test assert an effect did or didn't happen."""
        return len(self._intents)

    @property
    def payment_count(self) -> int:
        """Number of issued payment mandates — lets a test assert an effect did or didn't happen."""
        return len(self._payments)


class MandateSignatureStore(Protocol):
    """Where a mandate's JWS lives, keyed by mandate ID — never on the mandate record itself.

    P3 in storage form: `IntentMandate`/`PaymentMandate` have no field that can hold signing
    material (`mandates/models.py`), so the signature has to live somewhere else. This is that
    somewhere else. `MandateAuthority` implementations write here at issuance and read here at
    verification; nothing in `mcp_server/` reads this store directly, so a JWS can reach a
    caller only if some future code deliberately adds a field for it — the same "no field to
    put it in" property the models already have.
    """

    def put(self, mandate_id: str, jws: str) -> None: ...

    def get(self, mandate_id: str) -> str | None: ...


class InMemoryMandateSignatureStore:
    """Process-local signature store. Durable storage is a later block's concern."""

    def __init__(self) -> None:
        self._jws_by_mandate_id: dict[str, str] = {}

    def put(self, mandate_id: str, jws: str) -> None:
        self._jws_by_mandate_id[mandate_id] = jws

    def get(self, mandate_id: str) -> str | None:
        return self._jws_by_mandate_id.get(mandate_id)
