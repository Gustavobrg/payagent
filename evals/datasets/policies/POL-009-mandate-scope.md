---
id: POL-009
title: "Mandate Scope"
category: mandate
---

# Purchase Policy for Autonomous Payment Agent

## 1. Intent Mandate
An Intent Mandate represents the high‑level authorization granted by a user or account holder for a defined scope of spending activity. It captures the purpose, allowed merchant categories, time window, and overall budget ceiling. The Intent Mandate does not execute any financial transfer; it merely records the user’s consent for subsequent payment actions.

## 2. Payment Mandate
A Payment Mandate is a concrete, executable instruction that triggers a specific fund transfer. It contains the exact amount, payee details, currency, and settlement parameters. Each Payment Mandate must be derived from and linked to a pre‑existing Intent Mandate, ensuring that every transaction reflects a previously approved intent.

## 3. Mandatory Parent Reference
Every Payment Mandate must reference its parent Intent Mandate by identifier. This linkage guarantees traceability, enables audit trails, and prevents unauthorized or out‑of‑scope payments. Without a valid parent reference, the autonomous agent must reject the Payment Mandate and log the event for compliance review.

## 4. Transaction Limits and Step‑Up Authentication
The autonomous agent enforces a maximum limit per transaction of BRL 500.00 (50000 cents). Any Payment Mandate exceeding this amount is automatically declined unless it is covered by a higher‑level Intent Mandate that explicitly permits larger values. Additionally, for any transaction whose amount surpasses BRL 1000.00 (100000 cents), step‑up authentication using WebAuthn or a passkey is mandatory before the Payment Mandate can be executed.
