---
id: POL-008
title: "Chargeback Dispute"
category: refund
---

# Purchase Policy for Autonomous Payment Agent: Dispute Process and Merchant Response Timelines

## 1. Scope
This document defines the procedures for disputing unauthorized charges processed by the autonomous payment agent and establishes the timelines within which merchants must respond to dispute notifications.

## 2. Definition of Unauthorized Charge
An unauthorized charge is any transaction executed without the explicit consent of the account holder, including but not limited to fraudulent use of payment credentials, processing errors, or merchant‑initiated charges that exceed agreed‑upon amounts.

## 3. Dispute Initiation Procedure
1. The account holder or authorized representative submits a dispute request through the designated dispute portal within 60 calendar days of the transaction posting date.
2. The request must include the transaction identifier, date, amount, and a concise description of the reason for the dispute.
3. Upon receipt, the payment agent logs the dispute, assigns a case number, and notifies the merchant of the claim.

## 4. Evidence Requirements
The disputing party shall provide supporting documentation such as:
- Communication records with the merchant
- Proof of cancellation or non‑receipt of goods/services
- Any authentication logs indicating lack of consent
The payment agent may request additional evidence if the initial submission is insufficient.

## 5. Merchant Notification and Response Timelines
1. The merchant receives an electronic notification of the dispute within 2 business days of case creation.
2. The merchant must acknowledge receipt and provide a preliminary response within 10 business days.
3. A full resolution, including any refund or justification for the charge, must be submitted within 30 calendar days from the date of notification.
4. Failure to respond within the stipulated periods results in an automatic provisional credit to the account holder pending final adjudication.

## 6. Resolution and Refund Process
- If the merchant accepts the dispute, a refund is processed to the original funding source within 5 business days.
- If the merchant contests the dispute, the payment agent conducts an impartial review based on submitted evidence and applicable regulations.
- The final decision is communicated to both parties within 15 business days of the review completion.

## 7. Transaction Limits and Authentication Requirements
- The autonomous payment agent enforces a maximum limit per transaction of **BRL 500.00 (50000 cents)**. Transactions exceeding this amount are rejected at the point of initiation and therefore are not subject to the dispute process.
- For any transaction amount above **BRL 1000.00 (100000 cents)**, step‑up authentication using WebAuthn/passkey is mandatory. This requirement reduces the likelihood of unauthorized high‑value charges and influences the evidentiary weight given to authentication logs during dispute evaluation.

## 8. Record Retention and Audit
All dispute records, merchant communications, and resolution outcomes are retained for a minimum of 7 years in compliance with applicable financial regulations and are available for audit upon request.
