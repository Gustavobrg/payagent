---
id: POL-001
title: "Transaction Limits"
category: limit
---

# Purchase Policy for Autonomous Payment Agent

## Maximum Transaction Limit (POLICY_MAX_AMOUNT_CENTS)

The autonomous payment agent enforces a hard ceiling on the value of any single transaction. The maximum permitted amount per transaction is **BRL 500.00 (50000 cents)**, defined as `POLICY_MAX_AMOUNT_CENTS`. Any purchase request whose amount exceeds this limit shall be rejected automatically with an appropriate error response.

## Handling of Transactions Exceeding the Maximum Limit

When a purchase request surpasses `POLICY_MAX_AMOUNT_CENTS`, the following actions occur:

1. The agent declines the transaction and returns a `TRANSACTION_AMOUNT_EXCEEDS_LIMIT` error code.
2. The caller may resubmit the purchase as multiple transactions, each not exceeding **BRL 500.00 (50000 cents)**.
3. No partial approvals or automatic splitting are performed by the agent.

## Step-Up Authentication for Exceptionally High Values

In scenarios where a transaction amount exceeds **BRL 1000.00 (100000 cents)**, the system mandates step‑up authentication via WebAuthn/passkey before any exception processing or manual override can be considered. This threshold is independent of the standard per‑transaction maximum and applies only to specially authorized high‑value flows.
