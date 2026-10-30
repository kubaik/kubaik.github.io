# Settlement vs clearing: backend engineer's guide

A payment integration that works in staging can still produce a ledger that never balances in production. The usual cause is not a race condition or a missing index. It is that the system has collapsed two distinct phases — clearing and settlement — into a single state transition, and the business now depends on a distinction the schema cannot express.

## The one-paragraph version

Clearing is the process of matching, netting, and confirming what two parties owe each other. Settlement is the actual transfer of funds that discharges that obligation. They are separate steps, often run by different systems, on different schedules, with different failure modes. An engineer who has only built CRUD apps or integrated a hosted payment gateway can treat the distinction as academic until asked to reconcile a ledger that does not balance because a settlement file arrived late. The part that trips people up is that clearing produces an *obligation*, not money, and settlement produces *money*, not necessarily an obligation — and a typical database schema conflates the two.

## Why this concept confuses people

Most backend engineers encounter payments through a gateway SDK. A call creates a payment intent, a webhook arrives, and an order is marked paid. The gateway abstracts away everything between "customer clicked pay" and "money is in the bank account." That abstraction is a feature, but it hides a two-phase process that exists everywhere money moves between institutions.

In card payments, authorization, clearing, and settlement are three distinct phases. Authorization checks the card and holds funds. Clearing — often called presentment — is when the merchant's acquirer sends the transaction to the issuer for confirmation. Settlement is when the issuer transfers funds to the acquirer, who then pays the merchant. These happen on different timelines: authorization is real-time, clearing can be same-day or next-day, and settlement is typically T+1 or T+2.

A common failure mode: a developer builds an internal wallet where a user's balance is updated immediately on a "payment success" webhook. Finance then asks why bank reconciliation shows a persistent mismatch every month. The webhook fired on authorization, not settlement — and some authorizations never clear. The code treated an obligation as settled funds.

The confusion compounds because the words are used loosely in product specs. "Settle the payment" might mean "capture the charge" in one team's vocabulary and "transfer funds to the merchant's bank" in another's. Without a shared model, three services each think they own the source of truth.

## The mental model that makes it click

Think of clearing and settlement like a group dinner where everyone orders separately but the restaurant will not split the bill. Clearing is the part where everyone looks at the receipt, agrees on who had the risotto, and calculates that Alice owes Bob $14 and Bob owes Carol $9. Settlement is when Alice actually sends Bob the money. Until the transfer lands, the obligation exists but the money has not moved.

In financial infrastructure, the clearing house is the restaurant receipt — a central counterparty that nets positions. The settlement system is the payment rail — Fedwire, CHIPS, TARGET2, or a distributed ledger — that moves the actual value.

This separation exists for specific reasons:

1. **Netting reduces liquidity needs.** If Bank A owes Bank B $100M and Bank B owes Bank A $95M, clearing nets that to a $5M settlement. Without netting, both would need to move the full amounts.
2. **Risk isolation.** Clearing can fail (a counterparty disputes a trade) without triggering a settlement failure. Settlement can fail (a bank's connection drops) without invalidating the cleared obligation.
3. **Different regulatory regimes.** Clearing houses are often regulated as financial market utilities; settlement rails have their own rules. Mixing them in one system makes compliance harder.

For a backend engineer, the key insight is that **clearing is a state machine over obligations, and settlement is a state machine over money movement**. They communicate through a reconciliation process, not through a shared transaction.

Here is a simplified state machine for a single obligation:

```python
from enum import Enum

class ObligationState(Enum):
    PENDING = "pending"          # created, not yet cleared
    CLEARED = "cleared"          # matched and netted
    SETTLING = "settling"        # settlement instruction sent
    SETTLED = "settled"          # funds confirmed moved
    FAILED = "failed"            # settlement rejected

# A typical transition guard
def can_transition(current, target):
    allowed = {
        ObligationState.PENDING: {ObligationState.CLEARED, ObligationState.FAILED},
        ObligationState.CLEARED: {ObligationState.SETTLING, ObligationState.FAILED},
        ObligationState.SETTLING: {ObligationState.SETTLED, ObligationState.FAILED},
        ObligationState.SETTLED: set(),
        ObligationState.FAILED: set(),
    }
    return target in allowed[current]
```

Notice that you cannot go from `PENDING` directly to `SETTLED`. That is the invariant the database should enforce — and it is the one most homegrown systems violate.

## A concrete worked example

Consider a payout system for a marketplace. Sellers earn money when buyers purchase, and the platform wants to pay sellers daily.

A naive implementation updates a `seller_balance` table on every order and runs a cron job at midnight that sends a bank transfer for the balance. This collapses clearing and settlement into one step, and it breaks in at least three ways:

1. **Refunds and chargebacks.** A buyer disputes a charge 30 days later. If the seller was already settled, the platform is now chasing money.
2. **Netting.** If the seller also buys from the platform, payouts should be netted against purchases. The naive system sends two transfers instead of one.
3. **Settlement failures.** The bank transfer fails (invalid account number, closed account, daily limit). The `seller_balance` is now wrong and there is no record of the failed obligation.

The fix is to split the system:

- **Clearing service:** consumes order events, creates `Obligation` records, applies netting rules, and produces a `SettlementBatch` at the end of each day. This service owns the ledger of what is owed.
- **Settlement service:** takes a `SettlementBatch`, calls the bank API (or generates a NACHA file, or submits to SEPA), and records the result. This service owns the ledger of what actually moved.

In practice, a reconciliation job compares the two ledgers. Here is a simplified version in Python:

```python
import datetime
from decimal import Decimal

def reconcile(clearing_ledger, settlement_ledger, as_of: datetime.date):
    """
    Compare cleared obligations against settled transfers.
    Returns a list of discrepancies for manual review.
    """
    discrepancies = []

    cleared = {
        ob.id: ob.amount
        for ob in clearing_ledger.obligations(as_of)
        if ob.state == "cleared"
    }
    settled = {
        tx.obligation_id: tx.amount
        for tx in settlement_ledger.transfers(as_of)
        if tx.state == "settled"
    }

    for ob_id, amount in cleared.items():
        if ob_id not in settled:
            discrepancies.append((ob_id, "unsettled", amount))
        elif settled[ob_id] != amount:
            discrepancies.append(
                (ob_id, "amount_mismatch", amount - settled[ob_id])
            )

    for ob_id in settled.keys() - cleared.keys():
        discrepancies.append((ob_id, "settled_without_obligation", settled[ob_id]))

    return discrepancies
```

To interpret the output, classify each discrepancy by age. A cleared obligation with no matching settlement is expected on the day it clears if settlement is T+1. The same discrepancy after three days is a real problem. A settlement with no obligation is almost always a bug in the clearing service or a manual transfer that bypassed it.

### Measuring the mismatch rate

Rather than trusting a number quoted in an article, measure it. Instrument the reconciliation job to emit three counters per run: `cleared_unsettled`, `amount_mismatch`, and `settled_without_obligation`, each bucketed by age in days. Run the job hourly, not daily; the age buckets are only meaningful if the job runs more often than the settlement window.

Then compare against a control. Take one settlement batch, sum the obligations it claims to discharge, and compare that sum to the bank statement line for the same value date. If the two agree, the clearing ledger is internally consistent. If they disagree, the bug is upstream of reconciliation — usually in netting or in how partial captures are recorded. This comparison is the only one that matters, because the bank statement is the external source of truth.

## How this connects to things you already know

If you have built event-driven systems, you already know the pattern. Clearing is the write model — it records intent and validates it. Settlement is the read model that eventually reflects reality. The reconciliation job is the projector that catches drift.

If you have worked with distributed transactions, you know two-phase commit. Clearing and settlement resemble a domain-specific 2PC, but with a crucial difference: there is no rollback. Once funds settle, they are settled. The only remedy is compensation with a new transaction. This is why financial systems are so careful about the `CLEARED → SETTLING` transition — it is the point of no return.

If you have used a log-based messaging system, think of the clearing ledger as the log and the settlement ledger as the materialized view. The log is append-only and authoritative for obligations. The view is mutable and authoritative for money. They must be reconciled, not merged.

| Concept | General backend world | Financial world |
|---|---|---|
| Event log | Append-only topic | Clearing ledger |
| Materialized view | Read-optimized replica | Settlement ledger |
| Idempotency key | Request ID | Trade ID / UETR |
| Exactly-once | Transactional messaging | Settlement finality |
| Reconciliation | Data quality job | Nostro/vostro reconciliation |
| Rollback | Compensating transaction | Reversal / chargeback |

The mapping is not perfect, but it is close enough to make the domain legible. The biggest difference is that financial systems have legal finality — once a settlement is final, reversing it requires a new legal agreement, not just a database update.

## Common misconceptions, corrected

**Misconception 1: Settlement is just a slower clearing.** No. Clearing is about *what* is owed; settlement is about *how* it moves. A settlement can happen without clearing (a direct wire between two parties who trust each other) and clearing can happen without settlement (a netted obligation that never gets paid because the counterparty defaults).

**Misconception 2: Real-time payments eliminate clearing.** They compress it, but they do not eliminate it. Instant payment schemes still have a clearing step — it happens in milliseconds instead of hours. The clearing logic is still there; it is embedded in the payment rail's protocol.

**Misconception 3: A database transaction can atomically clear and settle.** It cannot, because settlement involves an external system (a bank, a ledger, another institution) that does not participate in the database's transaction. The best available pattern is an outbox with idempotent settlement calls plus a reconciliation job. Trying to force atomicity here is a common source of bugs — a typical symptom is a `SETTLED` record with no corresponding bank confirmation, which surfaces during month-end close.

**Misconception 4: Netting is an optimization, not a requirement.** For high-volume systems, netting is a regulatory requirement in many jurisdictions. Institutions moving money between each other may be legally required to net before settling.

**Misconception 5: ISO 20022 is just a message format.** It is a data model that encodes clearing and settlement semantics. The `pacs.008` message is a clearing instruction; the `camt.053` message is a settlement statement. Anything that talks to a bank will parse these, and the work is mostly schema-specific validation rather than XML plumbing.

## The advanced version

Once the two-ledger model is in place, the interesting problems are:

**Multi-currency netting.** Settling across currencies requires a netting algorithm that handles FX rates and settlement risk. The standard approach is to net within each currency pair, then settle the residual. This is the problem continuous linked settlement systems exist to solve: settling both legs of an FX trade simultaneously to eliminate principal risk.

**Settlement finality and legal risk.** In some jurisdictions, settlement is only final when confirmed by the central bank. Until then, the obligation can be unwound. A `SETTLED` state that is not legally final is a different state from one that is. Homegrown systems often miss this and end up in disputes.

**Liquidity management.** A settlement participant must fund its settlement account before the settlement window closes. This is a real-time optimization problem: hold as little liquidity as possible while never missing a settlement. Large institutions run dedicated treasury systems for this; a fintech typically integrates with one rather than building it.

**Reconciliation at scale.** A mid-size payment processor can handle millions of obligations per day. Reconciling that requires streaming comparison rather than batch. Any stream processing framework that supports keyed state and exactly-once sinks will work; the important properties are that reconciliation is idempotent and restartable, because it runs continuously rather than once a day.

**Regulatory reporting.** Derivatives reporting regimes require submission of trade data to a trade repository within a short window after execution. A clearing system needs to emit these reports as a first-class output, not an afterthought. In practice this means a separate reporting service that consumes the clearing ledger and produces the required formats.

## Quick reference

| Term | What it means | Who owns it | Typical latency |
|---|---|---|---|
| Authorization | Check and hold funds | Issuer | Real-time |
| Clearing | Match, net, confirm obligation | Clearing house | Minutes to hours |
| Settlement | Move actual funds | Settlement rail | T+0 to T+2 |
| Netting | Offset mutual obligations | Clearing house | Part of clearing |
| Reconciliation | Compare ledgers | Operations | Daily or streaming |
| Finality | Legally irreversible | Central bank / regulator | Varies by rail |

Common settlement rails and their characteristics:

- **Fedwire (US):** Real-time gross settlement; final when processed.
- **CHIPS (US):** Net settlement; final at end of day.
- **TARGET2 (EU):** RTGS; final when processed.
- **SWIFT (global):** Messaging, not settlement. It carries instructions between banks.
- **UPI (India):** Instant retail payments with deferred net settlement.

Check current operating hours against the operator's documentation before relying on them; they change.

## Frequently Asked Questions

**What is the difference between clearing and settlement in payments?**

Clearing is the process of matching transactions, calculating net obligations, and confirming what each party owes. Settlement is the actual transfer of funds that discharges those obligations. Clearing produces an obligation; settlement produces a change in account balances. They are separate steps because netting reduces liquidity needs and because settlement involves external systems that can fail independently.

**How does netting work in a clearing system?**

Netting offsets mutual obligations between parties. If Bank A owes Bank B $100 and Bank B owes Bank A $95, bilateral netting reduces that to a single $5 obligation. Multilateral netting extends this across many parties, often through a central counterparty. The result is a smaller set of settlement instructions, which reduces liquidity requirements and operational risk. A clearing system should produce netted settlement batches, not one instruction per original transaction.

**Why do settlement failures happen and how are they handled?**

Settlement failures happen when the settlement rail rejects a transfer — invalid account, insufficient funds, closed account, or a technical outage. The obligation should be marked `FAILED` and trigger a retry or manual review. Never assume a settlement instruction succeeded just because it was sent. Always reconcile against the settlement rail's confirmation. A common pattern is exponential backoff for transient failures and escalation to manual review after a fixed number of attempts.

**Can a blockchain be used for settlement instead of traditional rails?**

Yes, with trade-offs. Distributed ledgers can provide settlement finality and 24/7 operation. They typically have higher latency and lower throughput than traditional rails, and they introduce new risks: smart contract bugs, key management, and regulatory uncertainty. For most use cases, traditional rails are cheaper and faster. Ledger-based settlement makes sense when atomic cross-border settlement is required or when counterparties do not trust a central intermediary.

## A decision checklist before touching the schema

Before refactoring, answer these on paper:

1. Does any column named `balance`, `amount`, or `status` represent an obligation, a settled amount, or both? If both, it needs to be split.
2. Can a record reach a terminal "settled" state without an external confirmation ID? If yes, that state is a lie.
3. Is there a single job that compares the two ledgers, and does it run more often than the settlement window?
4. Are settlement calls idempotent, keyed on the obligation ID rather than a fresh request ID?
5. When a settlement fails, is there a durable record of the failure, or does the obligation silently revert to its previous state?
6. Is netting applied before settlement instructions are generated, or after?
7. Does the system distinguish "funds moved" from "funds moved and legally final"?

Any "no" is a concrete work item. None of them require a rewrite; most require one new table and one new job.

**Next step:** Open the payments schema and list every column whose name contains `balance`, `amount`, or `status`. For each, write one word next to it — obligation, settled, or unclear. Anything marked unclear is the bug. This takes about 15 minutes and tells you whether the system needs a two-ledger refactor.
