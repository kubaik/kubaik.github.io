# Reconciling mobile money: edge cases banks ignore

Most guides to building on mobile money rails assume a clean environment and a patient timeline. Production provides neither. The failure modes that matter are not the happy path ones; they are duplicate confirmations, missing references, provider-specific rounding, and outages that arrive as empty files rather than error codes.

This article covers how to build a reconciliation layer for mobile money when the provider webhook is only an approximation of the truth, and the actual truth lives in a mobile network operator's ledger that you may only be able to reach by SFTP or a rate-limited API.

## The problem: approximate signals vs an exact ledger

A common first assumption is that mobile money providers behave like card networks: debit one account, credit another, and the ledger balances. In practice, the webhook stream is a set of signals, not a ledger. Typical failure modes include:

- **Duplicate confirmations.** A single successful transfer can trigger multiple identical webhooks within seconds because of retries and provider fallbacks.
- **Missing transaction IDs.** A confirmation may omit the original transaction reference, forcing a match on amount plus timestamp plus phone number, which collides on popular numbers.
- **Partial reversals.** A customer cancels shortly after initiation. The reversal webhook arrives after the original confirmation, and the provider's ledger shows the net credit while the raw events show both credit and debit.
- **Rounding differences.** The ledger may round to the nearest 5 or 10 units of currency while the webhook carries the exact amount.
- **Outages.** A provider outage can return an empty ledger file rather than an error, which looks like thousands of missing transactions.

These mismatches are often small in absolute terms — a few units of currency each — but they accumulate. The root problem is not missing events. It is time skew, partial failure, and provider-specific quirks. An event log that assumes causality (A then B then C) breaks when reality is A, then B or X or nothing, then C, where X can be a timeout retry, a fallback SMS confirmation, or an outage page.

A reconciliation system for this environment needs to:

- Handle asynchronous, multi-path confirmation flows.
- Attribute amounts to the correct merchant and customer accounts even when fields are missing.
- Detect silent failures that do not raise HTTP errors.
- Produce a human-readable variance report within a bounded time after settlement.

## Three approaches that fail, and why

### Approach 1: Idempotent keys only

The obvious first design is a unique index on `(provider, external_id)`. This works where the provider guarantees idempotency. Mobile money providers frequently do not. They reuse IDs across retries, so a unique index on the provider's ID either rejects legitimate events or, worse, silently drops them if the insert path swallows the conflict.

A second variant is hashing the entire event payload to detect duplicates. This fails when providers change their JSON schema, because semantically identical payloads produce different hashes:

```json
{"amount": 1000, "currency": "NGN", "reference": ""}
{"amount": 1000, "currency": "NGN", "reference": null}
```

Both mean the same thing. A payload hash treats them as different events. Canonicalising the payload before hashing helps, but it requires maintaining a per-provider field map that breaks whenever the provider adds a field.

### Approach 2: A queue per provider

Pushing every webhook to a FIFO queue with `MessageGroupId = provider` gives ordering per provider but loses global ordering. If provider A's success event lands before provider B's completion event for the same logical transfer, a naive reconciliation job sees the credit before the debit and flags a phantom overdraft. Adding a fixed delay to every message hides the problem while making dashboard latency unacceptable for merchants.

Queues also introduce orphaned messages. Ordering is not guaranteed across message groups, and a regional outage can strand messages for many minutes. If the consumer's visibility timeout is shorter than the outage, messages are redelivered or lost, leaving transactions in a zombie state that requires manual replay.

### Approach 3: Event sourcing with a single stream

Modelling every action as an immutable event (`TransferInitiated`, `TransferConfirmed`, `TransferReversed`, `WebhookReceived`) gives excellent auditability. It also explodes the stream count: one stream per transaction, with multiple confirmations per transaction, means tens of thousands of streams at modest volume. Category-based indexes degrade as stream counts grow, and many event stores do not support per-event TTL, so retention requires a compaction job that can lock the store while it runs.

The reconciliation logic becomes a saga with compensating transactions. Sagas are designed for distributed transactions across services. Reconciliation is a state machine with a single source of truth — your ledger — and does not need compensating writes. Worse, concurrent confirmation events for the same account can race, and a saga that aborts mid-flight leaves the balance inconsistent.

All three approaches share one assumption: **that the provider's confirmation is the ground truth**. In mobile money, the ground truth is the MNO's ledger, which is usually not directly accessible. Providers give approximations. The system has to reconcile approximations to an exact ledger.

## The approach that works: two stages

The core insight is to separate the **confirmation path** from the **reconciliation path**.

- Confirmation path: fast, provider-specific, may contain duplicates or omissions.
- Reconciliation path: slow, ledger-backed, treated as the single source of truth.

### Stage 1: Candidate aggregation

Every webhook becomes a candidate. Store it in a `candidates` table with fields such as `provider`, `external_id`, `amount`, `phone`, `timestamp`, `mno`, `status`, `raw_payload`. Deduplicate using a composite key over `(provider, mno, phone, amount, timestamp)` with a tolerance window.

A tolerance window can be expressed in PostgreSQL by bucketing the timestamp into fixed-width intervals:

```sql
-- PostgreSQL 15: bucket timestamps into 5-second windows for dedup
CREATE UNIQUE INDEX idx_candidates_dedup
ON candidates (
  provider,
  mno,
  phone,
  amount,
  to_timestamp(floor(EXTRACT(EPOCH FROM timestamp) / 5) * 5)
);
```

Note the caveat: bucketing into fixed windows means two events 4.9 seconds apart can land in different buckets, while two events 0.1 seconds apart across a boundary land in different buckets too. A windowed index reduces duplicates but does not eliminate them. Combine it with an application-level check that looks up recent candidates within the window before inserting, or accept a small residual duplicate rate and let the reconciliation stage absorb it.

Choosing the window is a measurement, not a guess. Instrument the following:

- Log every webhook with `(provider, external_id, amount, phone, timestamp)` into a staging table for a period long enough to cover peak traffic.
- Group by `(provider, phone, amount)` and compute the distribution of inter-arrival gaps for groups with more than one row.
- Plot the gap distribution. The duplicate cluster is usually a sharp spike at small gaps; legitimate repeat transfers to the same number produce a long tail.
- Pick the window at the point where the spike ends and the tail begins. Widening past that point increases false positives (legitimate transfers treated as duplicates).

The right window depends on your providers and traffic mix. Do not copy a number from an article; derive it from your own gap distribution.

### Stage 2: Ledger reconciliation

Periodically pull a ledger export from each MNO — a CSV over SFTP, or a paginated API — containing the actual credited and debited amounts per phone number. Load it into an `mno_ledgers` table keyed on `(mno, phone, transaction_date, amount)`.

Then match candidates to ledger rows:

1. Exact match on `(mno, phone, amount, transaction_date)`.
2. If no exact match, widen to `(mno, phone, amount within tolerance)` and `(mno, phone, timestamp within tolerance)`.
3. If still no match, flag as variance and generate a human-readable report.

A worked example of the tolerance arithmetic. Suppose a transfer of 12,345 NGN is sent to an MNO whose ledger rounds to the nearest 10 NGN, and whose webhook carries the exact amount.

- Ledger value: `round(12345 / 10) * 10 = 12350`.
- Absolute difference: `|12345 - 12350| = 5`.
- A 1% relative tolerance on the larger value: `0.01 * 12350 = 123.5`.
- The difference of 5 is well within 123.5, so the pair matches.

Now suppose the same transfer is off by 500 NGN because of a genuine discrepancy:

- Absolute difference: `500`.
- Relative tolerance: `123.5`.
- 500 > 123.5, so the pair is flagged as a variance.

The tolerance must be the **greater** of an absolute floor and a relative percentage, so that small transfers are not wrongly matched and large transfers are not wrongly flagged:

```sql
ABS(c.amount - l.amount) <= GREATEST(1, 0.05 * GREATEST(c.amount, l.amount))
```

The `GREATEST(1, ...)` floor prevents a zero tolerance on tiny amounts. The `0.05` relative term absorbs rounding on large amounts. Choose both constants from your own ledger data: compute the distribution of `amount` differences for pairs you have manually confirmed as the same transfer, and set the relative term above the 99th percentile of that distribution.

### A reconciliation query

The following CTE matches candidates to ledger rows, classifies each pair, and separates auto-clearable variances from ones needing review. It is deliberately brute-force and auditable.

```sql
WITH matched AS (
  SELECT
    c.id AS candidate_id,
    l.id AS ledger_id,
    c.amount AS candidate_amount,
    l.amount AS ledger_amount,
    CASE
      WHEN c.status = 'reversed' AND l.amount = 0 THEN 'matched_reversal'
      WHEN ABS(c.amount - l.amount)
           <= GREATEST(1, 0.05 * GREATEST(c.amount, l.amount)) THEN 'matched'
      ELSE 'variance'
    END AS match_status
  FROM candidates c
  JOIN mno_ledgers l
    ON c.mno = l.mno
   AND c.phone = l.phone
   AND c.timestamp::date = l.transaction_date
   AND ABS(c.amount - l.amount)
       <= GREATEST(1, 0.05 * GREATEST(c.amount, l.amount))
)
SELECT
  candidate_id,
  ledger_id,
  match_status,
  CASE
    WHEN match_status = 'variance'
     AND ABS(candidate_amount - ledger_amount) < 1 THEN 'auto_cleared'
    WHEN match_status = 'variance' THEN 'needs_review'
    ELSE 'no_action'
  END AS auto_action
FROM matched;
```

Two notes on correctness. First, the `auto_cleared` branch as written can only fire when `match_status = 'variance'`, which by definition means the difference exceeded the tolerance; if the tolerance floor is 1, a difference below 1 can never reach that branch. Either lower the floor or drop the branch. Second, the join is a fuzzy join: it can produce multiple matches for one candidate if several ledger rows fall within tolerance. Add a deterministic tie-breaker (nearest timestamp, then nearest amount) or aggregate to the best match per candidate, otherwise the report double-counts.

### Asynchronous ledger polling

Waiting for a nightly SFTP export means variance detection lags by hours. Polling each MNO's API more frequently reduces that lag but runs into rate limits. A typical pattern is a token bucket per provider: allow a fixed number of requests per minute, and back off exponentially on HTTP 429 responses. For example, if a provider documents a limit of 50 requests per minute, configure the bucket at 45 per minute with a 10-second retry delay to leave headroom for retries.

Measure the effect rather than assuming it. Instrument the timestamp of each ledger pull and the timestamp of the first matching candidate, and compute the distribution of `ledger_pull_time - candidate_time` per provider. That distribution is your reconciliation lag. Changing the poll interval shifts it predictably; changing the backoff parameters shifts it under load.

## Implementation notes

A workable stack for this problem:

- **Ingestion.** A serverless function receives webhooks from each provider, validates the provider's signature, and writes the raw payload to a durable queue or table. Payload validation should reject unsigned or malformed requests before they reach the candidate table.
- **Candidate storage.** A relational database with an index on the dedup key. Time-series partitioning helps if volume is high, but a plain table with the right indexes handles modest volumes fine. Partitioning is a decision to revisit when query latency degrades, not a default.
- **Ledger polling.** A separate worker polls each MNO's SFTP or HTTPS endpoint on a schedule, parses the export, and writes to the ledger table. Keep this worker independent from the reconciliation job so a slow or failed export does not block matching.
- **Reconciliation.** A scheduled job runs the matching query, writes a variance report, and notifies the team. The job should be idempotent: re-running it must not create duplicate variance rows.
- **Variance handling.** An API endpoint lets merchants view and approve variances, backed by a cache so that peak-hour reads do not hit the database.

Three architectural decisions are worth thinking through before committing:

1. **Time-series storage vs plain tables.** Time-series compression reduces storage cost for old rows, but migrating away later requires rewriting the data. Adopt it when raw storage cost is a measured problem, not preemptively.
2. **Shared database vs separate instances.** A single instance is cheaper but creates lock contention between ingestion and reconciliation. Splitting into a read replica for candidates and a writer for ledgers and reconciliation removes the contention. Merging back is expensive, so decide based on measured contention, not on the price difference alone.
3. **Polling vs push.** If a provider's webhook is undocumented or unreliable, polling is the only option, and it requires maintaining credentials and handling rate limits. Switching to push later requires the provider's cooperation, so treat polling as a long-term commitment if you start there.

One more lesson: mobile money reconciliation is usually **not** a real-time problem. Merchants reconcile at the end of the day, so a bounded lag is acceptable. A real-time dashboard that pushes updates on every webhook will show false variances during provider throttling and outages. Batch processing on a fixed interval is more resilient and cheaper to run.

### Handling provider outages

During an MNO outage, the ledger export may be an empty file rather than an error. A reconciliation job that treats an empty file as "no transactions" will flag every candidate as a variance. Add a `ledger_file_present` check: if the file is empty or missing, skip variance calculation and log an outage event instead. This prevents a burst of false alerts and false fraud signals.

The same principle applies to partial exports. If the row count is far below the trailing median, treat the export as suspect and skip rather than flag.

## What to measure

The following table lists metrics worth instrumenting and how to compute each one. It is a measurement plan, not a results table.

| Metric | How to compute it |
|---|---|
| Variance rate | Count of candidates with `match_status = 'variance'` divided by total candidates, per day |
| Manual review load | Count of variances routed to `needs_review` per week, and the time spent per review |
| Reconciliation lag | Distribution of `ledger_match_time - candidate_time` per provider |
| Duplicate rate | Count of candidates rejected by the dedup key divided by total webhooks |
| Provider error rate | Count of non-200 responses and empty exports per provider per day |
| False variance rate | Count of variances later resolved as matches, divided by total variances |

Track the false variance rate explicitly. A high variance rate that turns out to be mostly false positives is worse than a low variance rate, because it trains the team to ignore alerts.

## A decision checklist

Before shipping a reconciliation layer, work through these questions:

1. **What is your ground truth?** If it is the provider's webhook, you are building on sand. Identify the ledger source — MNO export, bank core, card network report — and treat it as authoritative.
2. **What is your dedup key, and how was the window chosen?** If the window came from a blog post rather than your own gap distribution, it is a guess.
3. **What is your tolerance, and how was it derived?** Compute the distribution of amount differences for confirmed matches and set the relative term above its 99th percentile.
4. **Are confirmation and reconciliation separate processes?** If they are the same process, you are conflating signals with facts.
5. **What happens on an empty or short ledger export?** If the answer is "flag everything," you will generate false variances during every outage.
6. **Is the reconciliation job idempotent?** Re-running it must not duplicate variance rows.
7. **Does the fuzzy join produce one match per candidate?** If not, add a tie-breaker or aggregate.
8. **What is the auto-clear threshold, and is it reachable given the tolerance floor?** A branch that can never fire is dead code.
9. **Who reviews variances, and how long does it take?** If the queue grows faster than it drains, the tolerance is wrong.
10. **What is the reconciliation lag, measured, not assumed?** Merchants will ask.

Score yourself honestly. Any "no" is a candidate for the next sprint.

## FAQ

**How should duplicate mobile money webhooks be handled?**
Check the provider's payload for an idempotency key. If present, use it as the dedup key. If not, use a composite of `(provider, mno, phone, amount, timestamp window)` and store rejected duplicates in a separate table with the raw payload for audit. Derive the window from your own inter-arrival gap distribution rather than assuming a fixed number of seconds.

**How do you match events to a ledger when the transaction reference is missing?**
Fuzzy match on `(mno, phone, amount within tolerance, timestamp within tolerance)`. Because a fuzzy join can match multiple ledger rows to one candidate, add a deterministic tie-breaker — nearest timestamp first, then nearest amount — or aggregate to the best match per candidate. Store the matched pair and the match reason for audit.

**Is event sourcing the right model for reconciliation?**
Event sourcing gives strong auditability but adds stream management, retention, and saga complexity. Reconciliation is a state machine with a single source of truth, so a candidate table plus a ledger table plus a scheduled matching job is usually sufficient. Adopt event sourcing only if you need per-event replay for reasons beyond reconciliation.

**How often should the reconciliation job run?**
As often as the ledger source allows. If the ledger is only available as a daily export, run once after the export lands. If the ledger is available via a rate-limited API, run at an interval that keeps you under the rate limit with headroom for retries. Measure the resulting lag and confirm it meets merchant expectations.

**What tolerance should be used for amount matching?**
The greater of an absolute floor and a relative percentage, with both constants derived from your own data. Compute the distribution of amount differences for pairs you have confirmed as the same transfer, and set the relative term above its 99th percentile. A common starting point is 1 unit of currency or 5%, whichever is larger, but validate it against your ledger.

## The broader lesson

The lesson generalises beyond mobile money. Most integrations assume the provider's confirmation is the ground truth. In reality, providers give approximate signals: webhooks that may be duplicated, delayed, or missing fields. The real ground truth is often inaccessible — an MNO ledger, a bank core, a card network report. The engineering job is to reconcile approximate signals to an exact ledger without introducing new approximations.

This applies to card payments (processor webhooks vs the network's ledger), bank transfers (aggregator webhooks vs the bank's core), crypto (explorer data vs exchange ledgers), and accounting (application entries vs bank statements). In each case, the edge cases are the core: duplicates, missing fields, rounding, and outages are the rule, not the exception.

The boring solution — batch processing, fuzzy matching, tolerance windows, and explicit outage handling — is the one that scales. Event sourcing, saga patterns, and real-time dashboards add complexity without addressing the central problem: reconciling approximate signals to an exact ledger.

## Do this in the next 30 minutes

Pick one provider you integrate with and pull the last 1,000 webhook payloads for it. Group them by `(phone, amount)` and compute the distribution of time gaps between rows in each group. That single histogram tells you whether your current dedup window is too narrow (duplicates slipping through) or too wide (legitimate transfers being collapsed), and it costs you one query and a few minutes of interpretation.
