---
id: POL-005
title: "Merchant Allowlist"
category: allowlist
---

# Purchase Policy for Autonomous Payment Agent

## Merchant Allowlist Overview
The autonomous payment agent maintains a curated allowlist of merchant identifiers (merchant_id) that are authorized for settlement processing. Only merchants present on this list may receive funds through the agent’s settlement workflow. The allowlist is managed by the compliance team and updated in accordance with regulatory requirements and risk assessments.

## Authorized Settlement Merchant Verification
When a transaction is initiated, the agent validates the merchant_id against the current allowlist. If the merchant_id is found, the transaction proceeds to the settlement stage. Settlement amounts are subject to the maximum limit per transaction of **BRL 500.00 (50000 cents)**. For any settlement amount exceeding **BRL 1000.00 (100000 cents)**, a step‑up authentication using WebAuthn/passkey is mandatory before the settlement can be finalized.

## Handling of Unlisted merchant_id
If a transaction references a merchant_id that is not present on the allowlist, the agent will:
1. Reject the settlement request immediately.
2. Log the event with a reason code indicating "merchant_id not authorized" for audit purposes.
3. Notify the originating party (payer or upstream system) of the rejection via the standard error response format.
4. Escalate the incident to the compliance operations queue for review, should the merchant request inclusion on the allowlist.

No settlement will be performed for unlisted merchants, and no funds will be transferred until the merchant_id is formally added to the allowlist following the approved onboarding procedure.
