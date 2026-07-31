---
id: POL-002
title: "Aggregate Limits"
category: limit
---

# Purchase Policy for Autonomous Payment Agent

## 1. Scope
This document defines the aggregate spending limits applied to each user of the autonomous payment agent. Limits are expressed in Brazilian reais (BRL) and are enforced independently of the per‑transaction ceiling.

## 2. Per‑Transaction Limit
The maximum amount permitted for any single payment is **BRL 500.00 (50000 cents)**. No transaction may exceed this value regardless of the user’s aggregate allowances.

## 3. Aggregate Daily Limit
Each user is assigned a daily spending cap that represents the total value of all successful transactions processed within a 24‑hour rolling window. The daily cap is configured per user profile and is evaluated after the per‑transaction check. If the sum of attempted transactions would surpass the daily cap, the excess transactions are declined.

## 4. Aggregate Monthly Limit
In addition to the daily cap, a monthly spending ceiling applies to the cumulative value of all successful transactions within a calendar month. The monthly limit operates independently of the daily limit; both must be satisfied for a transaction to be approved.

## 5. Stacking of Limits
Limits are enforced in the following order:
1. **Per‑transaction limit** – the transaction amount must not exceed BRL 500.00 (50000 cents).
2. **Daily aggregate limit** – the running total for the current day plus the transaction amount must not exceed the user’s daily cap.
3. **Monthly aggregate limit** – the running total for the current month plus the transaction amount must not exceed the user’s monthly cap.

A transaction is approved only when it satisfies all three checks simultaneously. The daily and monthly caps do not override the per‑transaction ceiling, nor does the per‑transaction ceiling relax the aggregate caps.

## 6. Step‑Up Authentication Threshold
Step‑up authentication (WebAuthn/passkey) is mandatory for any transaction whose amount exceeds **BRL 1000.00 (100000 cents)**. Because the per‑transaction ceiling is set at BRL 500.00 (50000 cents), this requirement is triggered only in exceptional cases where a transaction is authorized above the standard ceiling (e.g., manual override or policy exception).

## 7. Monitoring and Adjustments
Aggregate limits are monitored in real time. Administrative personnel may adjust daily or monthly caps per user based on risk assessment, regulatory requirements, or contractual agreements. Any change takes effect immediately for subsequent transactions.

## 8. Non‑Compliance Handling
Transactions that violate any of the defined limits are rejected and logged. Repeated violations may result in temporary suspension of the autonomous payment agent for the affected user pending review.
