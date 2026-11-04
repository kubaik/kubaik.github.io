# PAPSS and mobile money: where integration breaks

## The scenario: a two-market wallet that needs to pay out across borders

Cross-border instant payments are often described as a plumbing problem: connect to PAPSS, connect to the mobile money APIs, and money moves. In practice, the plumbing is the easy part. The hard part is that PAPSS and mobile money interoperability solve different problems with different settlement models, different identifiers, and different failure semantics — and integrations that treat them as interchangeable end up with reconciliation gaps that take days to unwind.

A typical scenario: a fintech operates a wallet product in Ghana and Kenya. Users fund wallets via mobile money (MTN MoMo, Airtel Money, M-Pesa, depending on market) and can send to other wallet users instantly. The business now wants two new flows: wallet-to-bank-account in another African country, and wallet-to-mobile-money in a country where the fintech has no local entity.

The naive design treats both flows as "send money abroad." The team picks one rail — say PAPSS for the bank leg, and a mobile money aggregator for the wallet leg — and builds a single `transfer` service that calls whichever rail matches the destination. This works in testing. In production, three failure modes recur:

1. A PAPSS payment is accepted but the beneficiary bank rejects it hours later, after the sender's wallet has already been debited and the user has been told the money is on its way.
2. A mobile money payout succeeds at the telco but the aggregator's callback never arrives, so the fintech's ledger shows the payment as pending indefinitely.
3. A user in Kenya sends to a Ghanaian mobile money number, and the recipient's name on the telco side does not match the name the sender entered — but the telco's API accepts the transaction anyway because name matching is not enforced.

Each of these is a design failure, not an API failure. The rest of this article is about the mechanisms behind them.

## PAPSS and mobile money solve different problems

PAPSS — the Pan-African Payment and Settlement System — is a cross-border settlement infrastructure. It is operated by Afreximbank and is designed to let participating financial institutions settle intra-African payments in local currencies without routing through correspondent banks in a third currency. The key word is *settlement*. PAPSS moves value between institutions; it is not a consumer wallet, and it does not by itself give a fintech the ability to debit a customer in one country and credit a customer in another. That requires a participant bank or a licensed institution on each side, and it requires the fintech to integrate with those institutions.

Mobile money interoperability is a different layer. In markets like Kenya, Ghana, Tanzania, and Rwanda, telcos and regulators have built systems that let a wallet on one network pay a wallet on another, and increasingly let wallets pay bank accounts. These systems are domestic by design — they clear within a country's payment system. Cross-border mobile money transfers exist, but they are typically bilateral arrangements between telcos or corridors run by remittance companies, not a single pan-African instant rail.

The two rails differ on several axes that matter for integration design:

| Dimension | PAPSS | Mobile money interoperability |
|---|---|---|
| Primary unit | Interbank settlement in local currencies | Domestic wallet-to-wallet and wallet-to-bank clearing |
| Who the fintech talks to | A participating bank or licensed institution | Telco APIs or an aggregator |
| Identifier | Bank account / national account format | MSISDN (phone number), sometimes with a network prefix |
| Failure timing | Can be synchronous at submission, asynchronous at settlement | Usually synchronous at submission, asynchronous at callback |
| Reversibility | Reversal via the participating bank, subject to settlement windows | Reversal via telco or aggregator, often manual |
| Name resolution | Bank-side account name lookup, if the bank supports it | Telco-side subscriber name, often not exposed via API |

This table is the root of most integration bugs. A single `transfer` abstraction over both rails will inevitably model PAPSS as if it were a wallet payout, and model mobile money as if it were a bank transfer. Neither mapping holds.

## The mechanism at fault: modelling both rails as one payment state machine

The failure mode is almost always the same: a single payment state machine that assumes every rail behaves like a domestic card authorization. Card payments have a well-understood lifecycle — authorization, capture, settlement, chargeback — and the authorization response is a strong signal that the payment will settle. Cross-border bank settlement and mobile money payouts do not give that signal.

With PAPSS, the submission response from the participating bank tells you the instruction was accepted, not that the beneficiary bank will honour it. The beneficiary bank can reject on account status, name mismatch, sanctions screening, or limit breaches, and that rejection arrives later. If the fintech debits the sender at submission time and shows "sent," it has made a promise it cannot keep.

With mobile money, the telco API typically returns a transaction ID and a pending status, then sends a callback with the final result. The callback can be delayed, duplicated, or lost. Aggregators add another hop where callbacks can be dropped. A state machine that treats the submission response as final will mark payments successful that later fail, and a state machine that waits for a callback without a timeout will leave payments pending forever.

The correct model is a two-phase state machine per rail, with rail-specific semantics:

- **Submission phase:** the rail acknowledges the instruction. This is not success. The sender's funds should be held (reserved), not debited.
- **Resolution phase:** the rail reports a final outcome — settled, rejected, or expired. Only here does the sender's hold convert to a debit or release.

For mobile money, the resolution phase is driven by a callback, with a polling fallback and a hard timeout. For PAPSS, the resolution phase is driven by the participating bank's settlement confirmation, which may be a file, an API, or a daily statement depending on the bank. The two rails need different timeout values and different fallback paths, but they can share the same hold-and-resolve pattern.

## Designing the hold-and-resolve flow

The pattern is straightforward if you commit to it:

1. On submission, create a payment record with status `SUBMITTED` and place a hold on the sender's wallet for the principal plus fees.
2. Call the rail. If the call fails synchronously, release the hold and mark `FAILED`.
3. If the call succeeds, store the rail reference and move to `PENDING_RESOLUTION`.
4. For mobile money, wait for the callback. If no callback arrives within a rail-specific timeout (telcos commonly document callback expectations in hours, but treat any specific figure as something to confirm in your contract), poll the transaction status endpoint.
5. For PAPSS, wait for the participating bank's settlement advice. If the bank provides a daily file, your resolution phase is a reconciliation job, not a real-time callback.
6. On `SETTLED`, convert the hold to a debit. On `REJECTED` or `EXPIRED`, release the hold and notify the sender.

A minimal implementation of the hold-and-resolve state machine in TypeScript looks like this. It is deliberately rail-agnostic; the rail-specific parts are injected.

```typescript
type PaymentStatus =
  | 'SUBMITTED'
  | 'PENDING_RESOLUTION'
  | 'SETTLED'
  | 'REJECTED'
  | 'EXPIRED';

interface PaymentRecord {
  id: string;
  rail: 'PAPSS' | 'MOBILE_MONEY';
  railReference?: string;
  status: PaymentStatus;
  amount: number;
  currency: string;
  senderId: string;
  createdAt: Date;
  resolvedAt?: Date;
}

interface RailAdapter {
  submit(payment: PaymentRecord): Promise<{ railReference: string }>;
  poll(railReference: string): Promise<'SETTLED' | 'REJECTED' | 'PENDING'>;
}

interface Ledger {
  hold(senderId: string, amount: number, currency: string): Promise<void>;
  captureHold(senderId: string, amount: number, currency: string): Promise<void>;
  releaseHold(senderId: string, amount: number, currency: string): Promise<void>;
}

async function submitPayment(
  payment: PaymentRecord,
  adapter: RailAdapter,
  ledger: Ledger
): Promise<PaymentRecord> {
  await ledger.hold(payment.senderId, payment.amount, payment.currency);
  try {
    const { railReference } = await adapter.submit(payment);
    return { ...payment, railReference, status: 'PENDING_RESOLUTION' };
  } catch (err) {
    await ledger.releaseHold(payment.senderId, payment.amount, payment.currency);
    return { ...payment, status: 'REJECTED' };
  }
}

async function resolvePayment(
  payment: PaymentRecord,
  adapter: RailAdapter,
  ledger: Ledger
): Promise<PaymentRecord> {
  if (payment.status !== 'PENDING_RESOLUTION' || !payment.railReference) {
    return payment;
  }
  const outcome = await adapter.poll(payment.railReference);
  if (outcome === 'PENDING') return payment;
  if (outcome === 'SETTLED') {
    await ledger.captureHold(payment.senderId, payment.amount, payment.currency);
    return { ...payment, status: 'SETTLED', resolvedAt: new Date() };
  }
  await ledger.releaseHold(payment.senderId, payment.amount, payment.currency);
  return { ...payment, status: 'REJECTED', resolvedAt: new Date() };
}
```

The `Ledger` interface is where the important invariants live: a hold must be idempotent per payment ID, and a capture or release must be idempotent so that duplicate callbacks do not double-debit. If your ledger cannot enforce those invariants, fix that before adding another rail.

There is a subtle bug class in this pattern worth naming explicitly: the hold is keyed by sender and amount in the sketch above, but in production it must be keyed by payment ID. If two payments from the same sender for the same amount are in flight, a sender-and-amount key lets a release for one payment cancel the hold for the other. Carry the payment ID through every ledger call.

## The identifier problem: MSISDN is not an account number

Mobile money uses the phone number as the account identifier. That seems simple until you look at how phone numbers behave across borders. A Ghanaian MSISDN starts with the country code 233; a Kenyan one with 254. A wallet payout to a Ghanaian number must be routed to the correct telco network, and the network is determined by the number prefix — but number portability means a prefix no longer reliably identifies the network. In markets with mobile number portability, the same MSISDN can move between networks, and a payout routed on the old prefix will fail or, worse, land in a different wallet.

The fix is to resolve the network at payout time, not at onboarding time. Telcos and aggregators typically expose a lookup endpoint that returns the current network for an MSISDN. Use it on every payout, cache the result with a short TTL (minutes, not days), and handle the case where lookup fails by rejecting the payout rather than guessing. A payout routed to the wrong network is not a failed payment; it is a payment to the wrong person, and recovering it depends on the recipient's cooperation.

PAPSS has the opposite problem. Account identifiers are national formats — Ghana's bank account numbers, Kenya's bank codes plus account numbers, Nigeria's NUBAN — and they are not globally unique. A payment instruction that carries only an account number is ambiguous. The participating bank's routing information must be part of the instruction, and the beneficiary account name should be verified against the bank's records before submission. If the bank does not expose name lookup, the fintech should treat name mismatch as a risk it cannot eliminate and set limits accordingly.

A worked example makes the ambiguity concrete. Suppose a payment instruction carries account number `0123456789` and destination country Ghana. That number is valid at multiple Ghanaian banks, so the instruction is under-specified until it also carries the bank code. Now suppose the same number is sent to a Kenyan bank: Kenyan account formats differ, so the number may be rejected outright — or, if the receiving bank's validation is loose, accepted into an account that is not the intended one. The design rule is that a payment instruction is only complete when it carries country, institution identifier, and account identifier together, and the receiving institution has confirmed the account name.

## Reconciliation is the product, not the afterthought

Every cross-border instant payment integration eventually converges on the same conclusion: the real-time path is the easy part, and reconciliation is where the product lives or dies. The reason is structural. PAPSS settles between institutions on a settlement cycle; mobile money settles within a telco's own ledger. Neither rail guarantees that the fintech's view of a payment matches the rail's view at the same instant. Discrepancies are normal, and the question is how quickly they are detected and resolved.

A workable reconciliation design has three layers:

1. **Per-payment status reconciliation.** For every payment in `PENDING_RESOLUTION` beyond its expected resolution window, poll the rail and reconcile. This catches lost callbacks and delayed settlement advice.
2. **Daily position reconciliation.** For each rail and each currency, compare the fintech's net position against the rail's statement. For mobile money, this is the aggregator or telco settlement report; for PAPSS, it is the participating bank's advice. Breaks here indicate missing or duplicated payments.
3. **Exception queue with ownership.** Every break needs an owner and an SLA. A break that sits in a queue without an owner is a customer complaint waiting to happen.

The failure mode to watch for is reconciliation that only runs on settled payments. If your reconciliation job filters to `SETTLED`, it will never see the payments that are stuck in `PENDING_RESOLUTION` — which are exactly the ones that need attention. Reconcile the pending set first, then the settled set.

A second failure mode is reconciliation that compares totals only. If the fintech's total position matches the rail's total but individual payments are mismatched — one payment double-counted and another missing — the totals will agree and the break will go undetected until a customer complains. Reconcile at the payment level first, then aggregate; totals are a checksum, not a reconciliation.

## What to monitor after go-live

Once the hold-and-resolve flow is in place, the metrics that matter are not throughput or latency. They are the ones that reveal divergence between your ledger and the rails:

- **Pending age distribution.** Count of payments in `PENDING_RESOLUTION` bucketed by age. A growing tail beyond the expected resolution window is the earliest signal that callbacks or settlement advice are not arriving.
- **Resolution source.** Proportion of payments resolved by callback versus polling versus manual reconciliation. A rising manual share means the automated paths are degrading.
- **Reversal rate by rail and corridor.** Rejections after submission, grouped by destination bank or telco. A single corridor with an elevated reversal rate usually indicates a name-matching or identifier problem specific to that counterparty.
- **Hold duration.** Time between hold placement and capture or release. Long holds tie up customer funds and generate support tickets; they also indicate a resolution path that is not completing.
- **Reconciliation break count and age.** Number of open breaks and how long the oldest has been open. This is the metric that predicts operational pain.

Alert on the derivative, not the level. A pending count of 500 is meaningless without knowing whether it was 50 an hour ago. Rate-of-change alerts catch rail degradation before the absolute count looks alarming.

### How to measure these without inventing numbers

None of the metrics above need a benchmark to be useful; they need instrumentation. Concretely:

- **Pending age distribution:** a scheduled query against the payments table that buckets `now() - created_at` for rows in `PENDING_RESOLUTION`, grouped by rail. Run it every few minutes and emit the histogram to your metrics system. The useful output is the shape of the tail, not a single figure.
- **Resolution source:** add a `resolution_source` column populated with `callback`, `poll`, or `manual` at the moment of resolution. A simple `GROUP BY` over the last 24 hours gives the mix.
- **Hold duration:** record `hold_placed_at` and `hold_released_at` on the ledger entry and compute the difference. Compare the p50 and p99 per rail; a widening gap between them means a subset of payments is stalling.
- **Reconciliation breaks:** have the reconciliation job emit one counter per break found and one gauge for the age of the oldest open break. Alert on the gauge, not the counter.
- **Corridor reversal rate:** `rejected / submitted` grouped by destination institution over a rolling window. Compare each corridor against its own history rather than against other corridors, since base rates differ by rail and destination.

The point of each measurement is to detect change in your own system. External benchmarks are not required and are often misleading, because corridor behaviour depends on counterparties, cut-off times, and contract terms that vary per integration.

## A decision checklist before adding a rail

Before integrating a new rail or corridor, work through these questions. A "no" is a design gap, not a blocker — but it should be a known gap with an owner.

1. **Does the rail's submission response mean acceptance or settlement?** If acceptance only, the sender's funds must be held, not debited, until resolution.
2. **What is the resolution mechanism — callback, poll, file, or statement?** Each needs a different fallback path and a different timeout.
3. **What is the documented resolution window, and what happens at the boundary?** Payments that cross the window must move to an exception queue automatically.
4. **What identifier does the rail require, and is it globally unique?** If not, what additional routing fields complete the instruction?
5. **Can the rail confirm the beneficiary name before submission?** If not, what limits apply to first-time or high-value recipients?
6. **How does the rail report reversals, and how long do they take?** A reversal that arrives after the customer has been told the payment succeeded is a support incident.
7. **What does the rail's settlement report contain, and at what frequency?** If it is daily, real-time reconciliation is impossible and the pending set becomes the primary control.
8. **Is every ledger operation idempotent per payment ID?** Duplicate callbacks and retried polls are normal; double-debits are not.

## The broader lesson: instant is a settlement property, not an API property

The mistake that ties all of these failures together is treating "instant payment" as a property of the API call. It is not. Instant describes the settlement timeline between institutions, and even that is bounded by settlement windows and cut-off times. The API call is an instruction; the payment is not real until the rail says it is.

Teams that internalise this build hold-and-resolve state machines, rail-specific adapters, identifier resolution at payout time, and reconciliation that treats pending payments as first-class. Teams that do not build a single `transfer` service, discover the gaps in production, and spend the next quarter unwinding reconciliation breaks manually. The difference is not engineering skill; it is whether the design acknowledges that PAPSS and mobile money are different rails with different semantics, and refuses to flatten them into one abstraction before understanding both.

## Frequently Asked Questions

**Can a fintech connect directly to PAPSS?**

PAPSS is designed for participation by financial institutions, and access is typically through a participating bank or a licensed institution in each market. A fintech without a banking licence usually integrates with a participant rather than with PAPSS directly. The practical consequence is that your PAPSS integration is really an integration with your participating bank's API or file interface, and the bank's capabilities — name lookup, settlement advice format, cut-off times — determine what you can build.

**Why does a mobile money payout show as successful but the recipient says they did not receive it?**

The most common cause is a mismatch between the transaction status at the telco and the fintech's ledger. The telco may have accepted the instruction and later reversed it, or the payout may have been routed to a different network because the MSISDN was ported. Another cause is a callback that arrived but was not processed idempotently, leaving the ledger in a pending state while the telco shows success. Resolving this requires querying the telco's transaction status endpoint by the original reference and reconciling against the settlement report.

**How should name verification work for cross-border mobile money?**

Most telco APIs do not expose the subscriber's registered name for privacy reasons, so name verification is often limited to confirming the MSISDN is active and belongs to the expected network. Where an aggregator offers a name lookup, treat it as advisory rather than authoritative. The practical mitigation is to set lower limits for first-time recipients and to require confirmation of the recipient's name through an out-of-band channel when the amount exceeds a threshold.

**What is the right timeout for a mobile money callback?**

There is no universal value; it depends on the telco and the aggregator, and the documented expectation should be in your contract. A workable approach is to set the timeout based on the rail's stated callback window, then add a polling fallback that runs at a fixed interval until the timeout expires, after which the payment moves to the exception queue. The important design property is that the timeout is explicit and the fallback is automatic, not that the timeout has a particular value.

**Should PAPSS and mobile money share one payment table?**

They can share a table, but not a state machine. A shared table with a `rail` column and a `rail_reference` column is fine and simplifies reporting. The state machine, the resolution mechanism, and the timeout policy must be selected by rail. The anti-pattern is a single `resolve()` path that assumes callbacks for both rails, because PAPSS resolution may arrive as a file that no callback will ever trigger.

## Start with your pending set

The single most useful thing you can do in the next 30 minutes is to query your payments table for every record in a pending state older than its expected resolution window, grouped by rail and corridor. That query — not a new integration, not a new abstraction — will tell you whether your current design can see its own unresolved payments. If the query returns nothing because your schema has no pending state, that is the finding: your state machine has already flattened the rails, and the reconciliation breaks are waiting.
