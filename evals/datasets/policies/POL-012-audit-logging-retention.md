---
id: POL-012
title: "Audit Logging Retention"
category: security
---

# Purchase Policy – Audit Trail Logging and Retention

## 1. Scope
This policy defines the data elements recorded in the audit trail for each transaction processed by the autonomous payment agent and specifies the retention period for those records.

## 2. Data Elements Logged per Transaction
The following information is captured for every transaction and written to the immutable audit log:

- **Transaction identifier** – unique system‑generated reference.
- **Timestamp** – date and time of processing in UTC (ISO 8601).
- **Amount** – value expressed in BRL and in cents (e.g., BRL 500.00 (50000 cents)).
- **Currency** – ISO 4217 code (BRL).
- **Payer identifier** – hashed or tokenised account reference.
- **Payee identifier** – hashed or tokenised merchant reference.
- **Authentication method** – primary credential type used (password, biometric, WebAuthn/passkey, etc.).
- **Step‑up authentication flag** – set to *true* when the transaction amount exceeds **BRL 1000.00 (100000 cents)**, triggering mandatory WebAuthn/passkey verification.
- **Applied transaction limit** – the per‑transaction ceiling enforced by the agent, **BRL 500.00 (50000 cents)**.
- **Transaction status** – authorized, declined, reversed, or pending.
- **Risk score** – numeric output of the fraud‑prevention engine at the moment of decision.
- **Device and network context** – IP address, device fingerprint, and geolocation (when collected).

All fields are stored in a tamper‑evident log with cryptographic chaining to guarantee integrity.

## 3. Retention Period
- **Active retention:** Audit records are kept online and searchable for **seven (7) calendar years** from the transaction date.
- **Archival retention:** After the active period, records are moved to a secure, read‑only archive for an additional **three (3) years**.
- **Destruction:** Upon expiry of the total ten‑year retention window, records are irreversibly erased in accordance with the data‑sanitisation procedure.

## 4. Access Controls and Governance
- Access to the audit trail is restricted to authorised compliance, security, and audit personnel.
- All read operations are logged and subject to periodic review.
- Modifications to the log are prohibited; only append‑only writes are permitted by the payment agent.

## 5. Compliance References
This policy aligns with applicable financial regulations (e.g., Central Bank of Brazil resolution 4.658) and international standards (ISO 27001, PCI‑DSS).

---
