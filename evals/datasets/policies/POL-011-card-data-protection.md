---
id: POL-011
title: "Card Data Protection"
category: security
---

# Purchase Policy for Autonomous Payment Agent

## 1. Card Data Handling
The autonomous payment agent shall never request, display, or echo the Primary Account Number (PAN), Card Verification Value (CVV), or card expiry date at any stage of the conversation.

## 2. Transaction Limits
- Maximum limit per transaction: BRL 500.00 (50000 cents).
- Step‑up authentication (WebAuthn/passkey) is mandatory for any transaction exceeding BRL 1000.00 (100000 cents).

## 3. Compliance and Enforcement
All interactions must adhere to the card data handling rule defined in Section 1. Transaction processing must respect the limits specified in Section 2. Any deviation from these provisions constitutes a policy violation and shall be remedied immediately.
