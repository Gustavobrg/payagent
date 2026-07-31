---
id: POL-010
title: "Quote Expiry"
category: mandate
---

# Purchase Policy for Autonomous Payment Agent

## Quote Validity
A quote obtained via the `get_quote` endpoint remains valid for a period of 30 minutes from the timestamp of issuance. During this window the quoted amount, currency, and terms are guaranteed and may be used to initiate settlement.

## Expiration Handling
If the quote expires before the settlement request is submitted, the agent must discard the expired quote and request a new `get_quote` invocation. Any settlement attempt referencing an expired quote will be rejected with a `QUOTE_EXPIRED` error code. The agent shall log the expiration event and notify the caller to obtain a fresh quote.

## Transaction Limits and Authentication
- The maximum limit per transaction is **BRL 500.00 (50000 cents)**. Any quote whose amount exceeds this limit cannot be settled and must be rejected at the quote validation stage.
- For transactions whose amount exceeds **BRL 1000.00 (100000 cents)**, step‑up authentication using WebAuthn/passkey is mandatory. Although the per‑transaction ceiling is lower than this threshold, the rule applies to any aggregated or split‑payment flows that may surpass the step‑up limit.

## Settlement Flow
1. Caller obtains a valid quote via `get_quote`.
2. Caller initiates settlement within the quote validity window.
3. Agent validates the quote timestamp, amount against the per‑transaction limit, and required authentication level.
4. If all checks pass, settlement proceeds; otherwise, the agent returns the appropriate error code (`QUOTE_EXPIRED`, `AMOUNT_EXCEEDS_LIMIT`, `STEP_UP_REQUIRED`).

## Compliance and Auditing
All quote issuance, expiration, and settlement events are recorded in an immutable audit log. The log captures the quote identifier, timestamps, amounts in cents, and authentication method applied. This log is retained for a minimum of 7 years to satisfy regulatory requirements.
