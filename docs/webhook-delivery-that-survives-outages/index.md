# Webhook delivery that survives outages

Webhook delivery looks trivial until a receiver goes down mid-deploy. This article walks through the failure modes of common designs, then builds a durable pipeline from a transactional outbox, an at-least-once queue, and a delivery worker that owns its retry state.

## The problem, stated precisely

A webhook is a promise: when something happens on one side, the other side gets told. The promise is easy to keep when everything is up. The hard part is keeping it when the receiver is down, slow, or returning 500s for hours during its own deploy window.

A typical failure mode looks like this: an order service emits an `order.created` event, a webhook worker POSTs it to `https://customer.example.com/hooks/orders`, the request times out after 10 seconds, and the worker logs a warning and moves on. The event is gone. The customer's dashboard never shows the order. They file a support ticket two days later, and by then the event is unrecoverable because queue retention was 24 hours and nobody noticed.

That pattern — synchronous delivery, no durable retry state, no idempotency key — is the default shape of a webhook system built in an afternoon. It works for months. Then a downstream outage exposes it, and events have to be reconstructed from application logs.

The part that trips people up is that "retry" is not the hard problem. The hard problem is ordering, idempotency, and backpressure interacting at once.

## Four approaches that commonly fail

**In-process retry.** POST, catch the exception, sleep, retry a few times, give up. The retry budget is tied to the request lifecycle, so if the receiver is down for 30 minutes, three retries over three seconds accomplish nothing. Worse, the retries happen in-process, so a deploy or pod restart during the retry window kills the pending work. The symptom is a worker log full of connection errors followed by a permanent gap in delivered events.

**Database table as a queue.** Insert a row with `status='pending'`, poll with a cron job, POST, update the row. This is durable, which is a real improvement, but it breaks under concurrency: two pollers pick up the same row, both POST, and the receiver gets duplicate events. If the receiver isn't idempotent — most aren't unless you gave them a key — a duplicate delivery can become a duplicate charge. This is the classic `SELECT ... WHERE status='pending'` without `FOR UPDATE SKIP LOCKED` bug, and it shows up during high-throughput bursts.

**Managed queue with a consumer.** Closer to correct, but two details bite people. Standard queues (SQS standard, RabbitMQ without deduplication) are at-least-once, so duplicates are expected and idempotency is still required. And the visibility timeout is a fixed guess: if a POST takes longer than the visibility timeout, the message becomes visible again while the first consumer is still working, producing concurrent delivery of the same event. A typical misconfiguration is a 30-second visibility timeout with a receiver that occasionally takes 45 seconds.

**Push correctness onto the receiver.** Rely on the downstream being idempotent and retry aggressively. This transfers a correctness problem to a customer who didn't ask for it, and it fails the moment you integrate with a team that treats every POST as a new event.

| Approach | Durable? | Ordering | Duplicate risk | Operational cost |
|---|---|---|---|---|
| In-process retry | No | N/A | Low | Low |
| DB table + cron | Yes | Weak | High without `SKIP LOCKED` | Medium |
| Queue + consumer | Yes | None (standard) | Medium (at-least-once) | Medium |
| Outbox + ordered queue | Yes | Strong per key | Low with idempotency key | High |

## The shape that survives outages

The pattern that holds up is a transactional outbox plus a durable queue plus a delivery worker that owns retry state. Each part exists to solve a specific failure mode.

**Outbox.** When the application changes state (an order is created), it writes the event to an `outbox` table in the same database transaction as the state change. This is the key move: it guarantees that if the order exists, the event exists. There is no dual-write problem where the order commits but the event publish fails. The outbox row carries an ID, a payload, a destination, a status, and a `created_at`.

**Relay.** A separate process reads unpublished outbox rows and publishes them to a durable queue. The relay marks rows as published. If the relay crashes after publishing but before marking, the queue receives a duplicate, which is acceptable because the queue is at-least-once and the worker is idempotent.

**Delivery worker.** Consumes from the queue and POSTs to the customer endpoint. Critically, the worker does not retry in-process beyond a short timeout. If the POST fails, the worker records the failure and schedules a retry by re-enqueueing with a delay — a queue delay mechanism, a delayed-message exchange, or a `retry_at` column. The retry schedule is exponential with jitter: roughly 10s, 30s, 2m, 10m, 1h, 6h, 24h. That gives about 24 hours of retry coverage, which covers the majority of downstream outages.

**Idempotency.** Every event carries a stable `event_id` generated at outbox insert time. The worker sends it as an `Idempotency-Key` header. Well-behaved receivers dedupe on it. For receivers that don't, the stable identifier at least lets both sides reconcile when the same event arrives twice.

No single component holds delivery state in memory. The outbox holds the truth, the queue holds the pending work, and the worker holds only the current attempt. Restart any of them and the system resumes. A downstream can be down for six hours and events simply accumulate in the retry schedule.

## Implementation details

The outbox insert in Python, using SQLAlchemy 2.0 and PostgreSQL 15. The important detail is that the insert shares the transaction with the business write.

```python
# outbox.py — SQLAlchemy 2.0, PostgreSQL 15
import uuid
from datetime import datetime, timezone
from sqlalchemy import insert
from models import Order, OutboxEvent

def create_order(session, order_data: dict) -> Order:
    order = Order(**order_data)
    session.add(order)
    session.flush()  # get the order.id

    event = {
        "id": str(uuid.uuid4()),
        "event_type": "order.created",
        "aggregate_id": str(order.id),
        "payload": {"order_id": str(order.id), "amount": order.amount},
        "destination": order.webhook_url,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
    }
    session.execute(insert(OutboxEvent).values(**event))
    session.commit()
    return order
```

The relay reads pending rows with `FOR UPDATE SKIP LOCKED` so multiple relay instances don't fight over the same row. A common mistake is a plain `SELECT` followed by an `UPDATE` — under load, two relays pick the same row. `SKIP LOCKED` is the fix, and it has been in PostgreSQL since 9.5.

```sql
-- relay query, PostgreSQL 15
SELECT id, event_type, payload, destination
FROM outbox_events
WHERE status = 'pending'
ORDER BY created_at
LIMIT 100
FOR UPDATE SKIP LOCKED;
```

The delivery worker in Node 20 LTS using the AWS SDK v3 for SQS. The key detail is the visibility timeout: set it to at least 3x your p99 POST latency, and extend it with `ChangeMessageVisibility` for long-running requests rather than guessing.

```javascript
// worker.js — Node 20 LTS, @aws-sdk/client-sqs v3
import { SQSClient, ReceiveMessageCommand, DeleteMessageCommand } from "@aws-sdk/client-sqs";

const sqs = new SQSClient({ region: "eu-west-1" });
const QUEUE_URL = process.env.WEBHOOK_QUEUE_URL;

async function poll() {
  const { Messages } = await sqs.send(new ReceiveMessageCommand({
    QueueUrl: QUEUE_URL,
    MaxNumberOfMessages: 10,
    WaitTimeSeconds: 20,
    VisibilityTimeout: 90,
  }));

  for (const msg of Messages ?? []) {
    const event = JSON.parse(msg.Body);
    const attempt = Number(msg.Attributes?.ApproximateReceiveCount ?? 1);

    try {
      const res = await fetch(event.destination, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": event.id,
          "X-Webhook-Attempt": String(attempt),
        },
        body: JSON.stringify(event.payload),
        signal: AbortSignal.timeout(15_000),
      });

      if (res.status >= 200 && res.status < 300) {
        await sqs.send(new DeleteMessageCommand({
          QueueUrl: QUEUE_URL,
          ReceiptHandle: msg.ReceiptHandle,
        }));
      } else if (res.status >= 400 && res.status < 500 && res.status !== 429) {
        // 4xx (except 429) is a permanent failure — don't retry forever
        await moveToDeadLetter(msg, event, `HTTP ${res.status}`);
      }
      // 5xx and 429 fall through — message becomes visible again after timeout
    } catch (err) {
      // network errors also fall through to retry
    }
  }
}
```

The 4xx handling is important and often missed. A 400 means the customer's endpoint rejected the payload — retrying 24 times won't help. Move it to a dead-letter queue and surface it in a dashboard. A 429 means rate-limited, which is transient, so it should retry. A 5xx means their server broke, which is also transient.

## Measuring it on your own workload

The numbers that matter are workload-specific, so the useful thing is knowing what to instrument and what to compare. Each item below is a measurement you can run, not a benchmark to trust.

**Outbox insert overhead.** Time the transaction before and after adding the outbox insert, with the same index set. Compare p50 and p99 under your real write load. If the added latency is larger than you expect, count the indexes on the outbox table — every index is paid on insert.

**Relay throughput.** Run one relay process and count rows marked published per second while the queue is not the bottleneck. Then scale relay instances and confirm throughput rises roughly linearly; if it doesn't, check whether `SKIP LOCKED` is actually in the query plan and whether the queue is throttling.

**Delivery worker throughput.** This is dominated by downstream latency, not your code. Measure your p50 downstream POST latency, then compute concurrency needed: `required_concurrency = target_events_per_second × p50_latency_seconds`. With a 200 ms p50 and 10 concurrent requests per worker, one worker handles roughly 50 events/second. That is arithmetic from two measured inputs, not a benchmark.

**Retry coverage.** Count attempts in your schedule and sum the delays. A 10s, 30s, 2m, 10m, 1h, 6h, 24h schedule yields 7 attempts spanning about 24 hours. Decide explicitly what happens after the last attempt, because "retry forever" is not a policy.

**Duplicate rate.** Log every `event_id` the worker attempts and count repeats. Compare the rate in steady state against the rate during a visibility-timeout misconfiguration. The gap tells you how much of your duplicate traffic is a bug rather than the cost of at-least-once delivery.

**Queue cost.** Take your provider's published per-request price, multiply by requests per month (roughly one receive plus one delete per delivery, plus retries), and add data transfer. For most webhook volumes the queue is a rounding error next to engineering time.

## What to watch out for

**Retry storms.** When a downstream recovers after an outage, accumulated events fire at once. If 50,000 events queued during a three-hour outage, the recovery burst can knock the receiver over again. The fix is jitter — random noise added to each retry delay — plus a circuit breaker that pauses delivery to a destination after N consecutive failures and resumes gradually.

**Unbounded outbox growth.** If the relay falls behind or the queue throttles, the outbox table grows. A table with tens of millions of pending rows makes the `SKIP LOCKED` query slow unless there is a partial index on `status='pending'`. Add that index early.

**Dead-letter queue as graveyard.** A DLQ nobody looks at is the same as dropping events. You need a dashboard showing DLQ depth, an alert when it grows, and a replay tool that can re-enqueue a DLQ message with a corrected payload. Teams commonly discover a DLQ full of months-old messages only when a customer asks why they never got their events.

**Signature verification versus retries.** If payloads are signed with HMAC-SHA256 and the receiver validates signatures, a retry with a regenerated timestamp fails validation. Sign once at outbox insert time and store the signature with the event, or include the timestamp in the signed payload and have the receiver tolerate a window (5 minutes is a common choice).

**Ordering assumptions.** This pattern does not give global ordering across all events. Strict per-customer ordering requires a FIFO queue or a partitioned log, and that costs throughput. For most webhook use cases, per-aggregate ordering — all events for order 123 in order — is enough, and you get it by partitioning on `aggregate_id`.

## The broader lesson

Durability is a property of where state is stored, not how many times you retry. Every webhook system that survives outages does so because the pending work lives in a database or a queue that survives process death, and because the retry schedule is decoupled from the request lifecycle. Retries are cheap; losing events is expensive. The same pattern applies to sending emails or syncing to a third-party API: persist the intent, deliver asynchronously, retry with backoff, and make the receiver's job easy with a stable idempotency key.

The corollary is to design for the receiver being down, not for the receiver being up. Assume every endpoint will eventually return 500s for hours. If the system can't survive that without human intervention, it isn't a webhook system — it's best-effort notification, and customers should be told that.

## How to apply this to your situation

Check whether an outbox table exists. If not, add one and route event publishing through it. There is no need to migrate everything at once — pick the highest-value event (usually `order.created` or `payment.succeeded`) and move just that one to the outbox pattern. Run both paths in parallel for a week, compare delivery counts, then migrate the rest.

If an outbox already exists, audit the retry schedule. Most systems have a fixed three-retry policy covering about 30 seconds. Extend it to at least 24 hours with exponential backoff and jitter, and add a dead-letter queue with an alert on depth. That single change is usually the difference between "events were lost during the outage" and "everything was delivered once the downstream came back."

## Frequently Asked Questions

**How do I prevent duplicate webhook deliveries?**

You can't prevent them entirely with at-least-once delivery, so the goal is to make them harmless. Send a stable `Idempotency-Key` header on every attempt, generated once when the event is created and reused across retries. Document it for your customers and recommend they dedupe on it. On your side, use `FOR UPDATE SKIP LOCKED` in the relay and set visibility timeouts longer than your p99 delivery latency.

**What's the right visibility timeout?**

Set it to at least 3x your p99 POST latency, and use `ChangeMessageVisibility` to extend it for long-running requests. If your p99 is around 20–30 seconds, 90 seconds is a reasonable starting point. Too low and you get concurrent delivery of the same message; too high and failed messages take longer to retry.

**Should I use FIFO or standard queues for webhooks?**

Standard queues unless strict ordering is genuinely required. FIFO queues cap throughput per message group and cost more. For most webhook use cases, per-aggregate ordering is enough, and standard queues plus application-level partitioning on the aggregate ID deliver it.

**How long should I retry a failed webhook?**

About 24 hours is the common sweet spot. Beyond that, the event is usually stale and the customer would rather get a manual replay than a three-day-old event. After 24 hours, move to a dead-letter queue, alert on it, and provide a replay tool. Retrying forever hides failures rather than fixing them.

Open your webhook delivery code and find the retry policy. If it's a fixed three-retry loop, replace it with an exponential backoff schedule of 10s, 30s, 2m, 10m, 1h, 6h, 24h with jitter, and add a dead-letter queue with an alert on depth. That is a one-file change and the highest-leverage thing you can do in the next 30 minutes.
