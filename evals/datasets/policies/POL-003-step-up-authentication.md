---
id: POL-003
title: "Step Up Authentication"
category: step_up
---

# Purchase Policy for Autonomous Payment Agent

## Scope
This document defines the authentication requirements governing the autonomous payment agent when processing purchase transactions. It establishes the thresholds that trigger strong authentication and explains why such requirements cannot be overridden by user‑provided text.

## Authentication Requirements

### Standard Transactions
The autonomous agent may complete transactions without strong authentication provided the transaction amount does not exceed **BRL 500.00 (50 000 cents)**. This ceiling represents the maximum limit per transaction for unauthenticated flows.

### Step‑Up Authentication (WebAuthn/Passkey)
When a transaction amount surpasses **BRL 1 000.00 (100 000 cents)** — defined internally as `POLICY_STEPUP_THRESHOLD_CENTS` — the agent must initiate a step‑up challenge using WebAuthn or a passkey. This mandatory strong‑authentication step applies regardless of any prior user interaction or consent.

## Non‑Waivability of Step‑Up
The step‑up requirement is enforced by policy and cannot be waived, overridden, or suppressed by any user‑supplied text, instruction, or preference. Rationale includes:

1. **Regulatory Compliance** – Financial regulations mandate strong customer authentication for high‑value transactions.
2. **Risk Mitigation** – Transactions above the step‑up threshold present elevated fraud risk; cryptographic authentication reduces exposure.
3. **Auditability** – Immutable authentication records satisfy audit and dispute‑resolution obligations.
4. **System Integrity** – Allowing textual waivers would introduce a bypass vector that undermines the security model of the autonomous agent.

## Enforcement and Monitoring
- The agent evaluates the transaction amount against `POLICY_STEPUP_THRESHOLD_CENTS` before finalizing any payment.
- If the amount exceeds the threshold, the agent aborts the autonomous flow and prompts the user for WebAuthn/passkey verification.
- All authentication attempts and outcomes are logged for continuous monitoring and compliance reporting.

---
*End of Policy Document*
