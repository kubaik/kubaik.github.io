# Webhooks done right: retries that don’t explode in 2026

Webhook delivery looks trivial in a tutorial: receive an HTTP POST, do some work, return 200. In production it becomes a distributed systems problem with at-least-once semantics, unreliable downstreams, and clocks that disagree. This article covers the failure modes that actually bite, and the design patterns that contain them.

## The core contract: at-least-once means you must be idempotent

Webhook providers almost universally guarantee *at-least-once* delivery, not exactly-once. That means any event can arrive twice, three times, or after a delay measured in hours. Retries are not an edge case; they are the contract.

The implication is that every consumer must be idempotent. The standard approach is a deduplication key derived from the event itself, stored in a fast store with a TTL longer than your maximum retry window. Redis is a common choice because `SET key value NX EX ttl` is a single atomic operation:

```javascript
// Atomic check-and-set: returns "OK" if this is the first time we've seen the key.
// Returns null if the key already existed.
const result = await redis.set(`wh:${eventId}`, '1', 'NX', 'EX', 86400);
if (result === null) {
  // Already processed. Acknowledge and exit without side effects.
  return { statusCode: 200, body: 'duplicate' };
}
// First time: proceed with the side effect.
```

Two details matter more than people expect:

1. **The TTL must exceed your retry horizon.** If a provider retries for 24 hours, a 1-hour TTL will let duplicates through on hour two. Documented retry windows vary widely by provider, so read the specific one you integrate with.
2. **The key must be stable across retries.** If you derive the key from a timestamp you generate locally, every retry looks like a new event. Derive it from the provider's event identifier.

## Failure mode 1: HTTP 200 that means "slow down"

A surprisingly common failure mode is a downstream API that returns `200 OK` with a body or header indicating throttling. The HTTP status code says success; the payload says "I accepted your request but did not process it." A naive consumer marks the event done and moves on, and the event is silently lost.

This is not hypothetical. Several APIs in the wild use custom headers such as a remaining-quota counter to signal backpressure while still returning 2xx. The documented behavior is provider-specific, so the only reliable defense is to parse the response body and headers for the signals the provider documents, and treat "quota exhausted" as a retryable condition even when the status code is 200.

A defensive handler looks like this:

```javascript
const res = await fetch(downstreamUrl, { method: 'POST', body: payload });

// Treat an explicit throttle signal as retryable regardless of status code.
const remaining = res.headers.get('x-ratelimit-remaining');
if (res.status === 429 || remaining === '0') {
  throw new RetryableError('throttled');
}

if (!res.ok) {
  // 5xx and 408 are retryable; 4xx (except 408/429) usually are not.
  if (res.status >= 500 || res.status === 408) {
    throw new RetryableError(`upstream ${res.status}`);
  }
  throw new PermanentError(`upstream ${res.status}`);
}
```

The lesson generalizes: **never infer success from the status code alone.** Check the body for the provider's documented success semantics.

## Failure mode 2: idempotency keys that collide

A deduplication key is only as good as its uniqueness. If a provider reuses an event identifier for two genuinely different operations, an idempotency check will reject the second one as a duplicate — and a legitimate side effect is dropped.

This happens most often when the "event id" identifies a *resource* rather than a *state transition*. Two payment attempts on the same invoice might share an invoice id but represent different charges. The fix is to compose the key from enough dimensions to distinguish the operations:

```
wh:{provider_event_id}:{resource_id}:{attempt_timestamp_truncated_to_minute}
```

Truncating to the minute is a deliberate trade-off: it tolerates clock jitter and retry drift while still separating attempts that occur more than a minute apart. If two distinct attempts can occur within the same minute, truncate to the second instead — the correct granularity depends on how fast the upstream can generate distinct events.

## Failure mode 3: retry state that grows without bound

Retry loops that carry their state in the workflow input tend to accumulate. Each backoff iteration appends to the payload, and after a few hundred retries the state blob is large enough to hit platform limits on execution history or payload size. Managed workflow engines commonly impose a cap on the number of history events per execution (AWS Step Functions, for example, documents a limit of 25,000 history events per execution).

The fix is to keep the workflow input small and store retry counters, backoff state, and deduplication metadata in an external store keyed by the execution identifier:

```javascript
// Instead of mutating the workflow input, read/write retry state externally.
async function bumpRetry(executionArn) {
  const { Attributes } = await dynamo.updateItem({
    TableName: 'webhook_retry_state',
    Key: { executionArn },
    UpdateExpression: 'ADD attempts :one SET lastAttemptAt = :now',
    ExpressionAttributeValues: { ':one': 1, ':now': Date.now() },
    ReturnValues: 'UPDATED_NEW',
  });
  return Number(Attributes.attempts);
}
```

This keeps the workflow input constant-size regardless of how many retries occur, and it makes retry state observable from outside the workflow.

## Failure mode 4: clock skew and freshness windows

If a downstream service rejects events older than N minutes, and your workflow compares the event timestamp to the current time, clock skew between your compute and the downstream's clock will occasionally cause valid events to be dropped. Regional NTP skew is real and can be measured in seconds to minutes.

The mitigation is a safety margin: treat events as fresh if they are within `N - margin`, and log the observed skew as a metric so you can alert when it drifts. A 30-second margin is a common starting point, but the right value depends on your tolerance for late events versus your tolerance for dropped ones. Instrument both.

## Failure mode 5: DLQs full of recoverable messages

A dead-letter queue is often treated as a graveyard. In practice, many messages land there because a downstream was down for a bounded period, not because they are poison pills. Replaying them one at a time by hand is slow and error-prone.

A better pattern is a batch replay workflow that:

1. Reads the DLQ in small batches.
2. Deduplicates against the same idempotency store used in normal processing.
3. Re-queues only events not already processed.
4. Emits a count of replayed, skipped, and failed messages per batch.

The deduplication step is what makes this safe: replaying a message that was actually processed before the failure will be caught by the idempotency check and skipped, so the replay is idempotent by construction.

## Failure mode 6: cross-region replay

Redis Streams and similar in-region stores do not replicate across regions by default. If you need to replay events in a second region after a regional outage, you need an export path. The common pattern is periodic export of the stream to object storage, with the consumer in the second region reading from the export offset.

The trade-offs are latency (you can only replay up to the last export) and cost (storage plus egress). Measure both before committing to a recovery time objective. A five-minute export interval bounds your worst-case replay lag at five minutes plus replay throughput time.

## Failure mode 7: quota exhaustion during traffic spikes

Managed workflow engines and serverless platforms impose concurrency and start-rate quotas. During a traffic spike, you can hit the quota and start dropping events at the ingestion layer, before your retry logic ever runs. This is a *pre-retry* failure, and it is invisible to your workflow metrics because the workflow never started.

The defense is to monitor the platform's "started" metric against the documented quota, and alert at a threshold well below 100%. Some platforms allow quota increases via a support request; others require capacity planning. Either way, the alert must fire before the quota is hit, not after.

## Integrations: what to check before wiring one up

The specific products you integrate with will change; the questions to ask do not.

**For an incident-management or alerting service:** confirm whether it exposes a documented deduplication key (often called a `dedup_key` or similar) and whether its retry behavior is configurable. If it deduplicates on a key you control, you can suppress duplicate alerts from duplicate webhook deliveries. If it does not, you must deduplicate before the call.

**For a chat or messaging API:** confirm whether it documents rate-limit headers and what the reset semantics are. A handler that catches a rate-limit error and rethrows it with a computed retry delay is more robust than one that blindly retries on a fixed interval:

```javascript
try {
  await client.postMessage({ channel, text });
} catch (err) {
  const reset = Number(err?.response?.headers?.['x-ratelimit-reset']);
  if (err?.status === 429 && Number.isFinite(reset)) {
    // Surface the reset delay so the workflow can wait precisely.
    throw Object.assign(new Error('rate limited'), { retryAfter: reset });
  }
  throw err;
}
```

**For a payments API:** confirm how the provider's idempotency key is exposed on incoming webhook events, and forward it to your own downstream so idempotency is preserved end-to-end. Disable the SDK's own automatic retries if your workflow owns retries — otherwise you get nested retry loops with unpredictable timing.

## How to measure whether your retry design is working

Do not trust intuition here. Instrument these four things and compare before/after any change:

1. **Duplicate rate.** Count events that hit the idempotency check and were rejected, divided by total events received. This is your at-least-once tax.
2. **Silent failure rate.** Count events that returned 2xx but were later found to have no side effect. This requires a reconciliation job, but without it you are flying blind.
3. **Retry depth histogram.** Track how many retries each event needed. A long tail means your backoff is too aggressive or your downstream is flaky.
4. **Time-to-replay after an outage.** Measure from the moment the downstream recovers to the moment the backlog is drained. This is your recovery time objective in practice.

A simple way to get started: add a `duplicate_trace_id` field to every event you forward downstream, and emit a metric on every idempotency rejection. Then build a dashboard that plots duplicate rate and retry depth over time. You will learn more from one week of that dashboard than from any benchmark table.

## A decision checklist before you ship

- [ ] Is every consumer idempotent, with a key derived from the provider's event id?
- [ ] Does the idempotency TTL exceed the provider's documented maximum retry window?
- [ ] Does the handler parse the response body for throttle signals, not just the status code?
- [ ] Is retry state stored externally, so workflow input stays constant-size?
- [ ] Is there a clock-skew margin on any freshness check, with the skew logged?
- [ ] Can the DLQ be replayed in batches with deduplication, not one message at a time?
- [ ] Is there an export path for cross-region replay, with a measured worst-case lag?
- [ ] Are platform quotas monitored with an alert that fires before the quota is hit?
- [ ] Are duplicate rate, silent failure rate, retry depth, and time-to-replay all instrumented?

## Action for the next 30 minutes

Pick one webhook consumer in your system and add a single metric: the count of idempotency rejections, emitted with the provider's event id as a dimension. Deploy it, let it run for a day, and look at the number. If it is zero, either your deduplication is broken or your provider is not retrying — both are worth knowing. If it is non-zero, you now have a baseline to optimize against, and you have replaced guesswork with a measurement.
