# Webhook delivery that survives outages

It's easy to spend longer than expected on webhook delivery before the actual failure mode becomes clear. This is the writeup with the mistakes left in, not edited out. It's the kind of problem that's easy to reproduce and hard to explain.

## The problem, in general terms

A webhook is a promise you make to a customer: when something happens on your side, you'll tell them about it. The promise is easy to keep when everything is up. The hard part is keeping it when the receiver is down, slow, or returning 500s for three hours straight during their own deploy window. That's the scenario every integration eventually hits — and it's the one where most webhook systems quietly start dropping events.

The failure isn't usually dramatic. It looks like this: your order service emits an `order.created` event, your webhook worker POSTs it to `https://customer.example.com/hooks/orders`, the request times out after 10 seconds, and your worker logs a warning and moves on. The event is gone. The customer's dashboard never shows the order. They file a support ticket two days later, and by then the event is unrecoverable because your queue retention was 24 hours and nobody noticed.

That pattern — synchronous delivery, no durable retry state, no idempotency key — is the default shape of a webhook system built in an afternoon. It works for months. Then a downstream outage exposes it, and you're reconstructing events from application logs.

The part that trips people up is that "retry" is not the hard problem. The hard problem is ordering, idempotency, and backpressure all interacting at once — and that's what this post actually covers.

## The approaches that commonly fail, and why

The first approach is fire-and-forget with a retry loop in the request handler. You POST, catch the exception, sleep 1 second, retry three times, then give up. This fails because the retry budget is tied to the request lifecycle. If the receiver is down for 30 minutes, three retries over three seconds accomplish nothing. Worse, the retries happen in-process, so a deploy or a pod restart during the retry window kills the pending work entirely. A common symptom is a worker log full of `ConnectionError: HTTPSConnectionPool(host='customer.example.com', port=443): Max retries exceeded` followed by a gap in delivered events.

The second approach is a database table used as a queue. You insert a row with `status='pending'`, a cron job polls every minute, POSTs, and updates the row. This is better — it's durable — but it breaks under concurrency. Two pollers pick up the same row, both POST, and the receiver gets duplicate events. If the receiver isn't idempotent (most aren't, unless you gave them a key), you've now created a billing double-charge. This is the classic `SELECT ... WHERE status='pending'` without `FOR UPDATE SKIP LOCKED` bug, and it shows up in production as duplicate webhook deliveries during high-throughput bursts.

The third approach is a managed queue like SQS or RabbitMQ with a consumer that POSTs. This is closer to correct, but two details bite people. First, SQS standard queues are at-least-once, so duplicates are expected — you still need idempotency. Second, the visibility timeout is a fixed guess. If your POST takes longer than the visibility timeout, the message becomes visible again while the first consumer is still working, and you get concurrent delivery of the same event. A typical misconfiguration is a 30-second visibility timeout with a receiver that occasionally takes 45 seconds, producing sporadic duplicates that are hard to reproduce.

The fourth approach — and the one I think is most overrated — is relying on the downstream to be idempotent and just retrying aggressively. This pushes your correctness problem onto a customer who didn't ask for it. It works until you integrate with a team that treats every POST as a new event, and then you're the one explaining why their inventory went negative.

| Approach | Durable? | Ordering | Duplicate risk | Operational cost |
|---|---|---|---|---|
| In-process retry | No | N/A | Low | Low |
| DB table + cron | Yes | Weak | High without SKIP LOCKED | Medium |
| SQS + consumer | Yes | None (standard) | Medium (at-least-once) | Medium |
| Outbox + ordered queue | Yes | Strong per key | Low with idempotency key | High |

## The approach that works in practice

The pattern that survives real outages is the transactional outbox combined with a durable queue and a delivery worker that owns retry state. It has four parts, and each one exists to solve a specific failure mode.

First, the outbox. When your application changes state (an order is created), you write the event to an `outbox` table in the same database transaction as the state change. This is the key move: it guarantees that if the order exists, the event exists. No dual-write problem where the order commits but the event publish fails. The outbox row has an ID, a payload, a destination, a status, and a `created_at`.

Second, a relay. A separate process reads unpublished outbox rows and publishes them to a durable queue — SQS, RabbitMQ, or Kafka depending on your ordering needs. The relay marks rows as published. If the relay crashes after publishing but before marking, you get a duplicate in the queue, which is fine because the queue is at-least-once and the worker is idempotent.

Third, a delivery worker. This consumes from the queue and POSTs to the customer endpoint. Critically, the worker does not retry in-process beyond a short timeout. If the POST fails, the worker records the failure and schedules a retry by re-enqueueing with a delay — SQS delay queues, RabbitMQ delayed message exchange, or a `retry_at` column. The retry schedule is exponential with jitter: roughly 10s, 30s, 2m, 10m, 1h, 6h, 24h. That gives you roughly 24 hours of retry coverage, which covers the vast majority of downstream outages.

Fourth, idempotency. Every event carries a stable `event_id` (a UUID generated at outbox insert time). The worker sends it as an `Idempotency-Key` header. Well-behaved receivers dedupe on it. For receivers that don't, you at least have a stable identifier to reconcile with when they ask why they got the same event twice.

The reason this survives outages is that no single component holds delivery state in memory. The outbox holds the truth, the queue holds the pending work, and the worker holds only the current attempt. Restart any of them and the system resumes. Downstream can be down for six hours and events just accumulate in the retry schedule.

## Implementation details

Here's the outbox insert in Python, using SQLAlchemy 2.0 and PostgreSQL 15. The important detail is that the insert shares the transaction with the business write.

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

The relay reads pending rows with `FOR UPDATE SKIP LOCKED` so multiple relay instances don't fight over the same row. A common mistake here is using a plain `SELECT` and then an `UPDATE` — under load, two relays pick the same row. `SKIP LOCKED` is the fix, and it's been in PostgreSQL since 9.5, so there's no reason not to use it.

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
// worker.js — Node 20 LTS, @aws-sdk/client-sqs v3.500+
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

## Results — the numbers to expect, and their limits

Typical figures for a setup like this, based on public benchmarks and documented behavior of the components:

- Outbox insert overhead: roughly 1–3 ms added to the transaction, depending on index count on the outbox table. If you're adding more than 5 ms, you probably have too many indexes on the outbox.
- Relay throughput: a single relay process on a 2-vCPU instance can move roughly 500–1500 events/second with batch publishing. Scale horizontally with `SKIP LOCKED`.
- Delivery worker throughput: dominated by downstream latency, not your code. At 200 ms p50 downstream latency and 10 concurrent requests per worker, you get roughly 50 events/second per worker.
- Retry coverage: a 10s-to-24h exponential schedule with jitter gives roughly 7 attempts over 24 hours. For outages longer than 24 hours, you need a dead-letter queue and a manual replay tool — no automated system should retry forever.
- SQS costs: at $0.40 per million requests (standard queue, eu-west-1, as of early 2026), a system delivering 10 million webhooks/month costs roughly $4–8 in SQS fees, plus data transfer. The queue is not where your money goes.
- Duplicate rate: with at-least-once delivery, expect 0.01–0.1% duplicates in normal operation, spiking to 1%+ during visibility-timeout misconfigurations.

These numbers are illustrative, not measured on your workload. The point is the order of magnitude: the queue and outbox are cheap, the downstream latency dominates, and duplicate handling is a correctness requirement, not an edge case.

The limit to be honest about: this pattern does not give you global ordering across all events. If you need strict ordering per customer, you need a FIFO queue (SQS FIFO, or Kafka with a partition key) and you accept lower throughput — SQS FIFO caps at 300 messages/second per message group without batching, 3000 with batching. For most webhook use cases, per-aggregate ordering (all events for order 123 in order) is enough, and you get that by partitioning on `aggregate_id`.

## What to watch out for

The first trap is retry storms. When a downstream recovers after an outage, your retry schedule fires all accumulated events at once. If you had 50,000 events queued during a 3-hour outage, the recovery burst can knock the receiver over again. The fix is jitter — add random noise to each retry delay — and a circuit breaker that pauses delivery to a destination after N consecutive failures, resuming gradually. Without jitter, you get synchronized retries, which is exactly the thundering herd problem.

The second trap is unbounded outbox growth. If the relay falls behind or the queue is throttled, the outbox table grows. A table with 50 million pending rows will make your `SKIP LOCKED` query slow unless you have a partial index on `status='pending'`. Add that index early; it's cheap and it's the difference between a 5 ms and a 500 ms relay query.

The third trap is treating the dead-letter queue as a graveyard. A DLQ that nobody looks at is the same as dropping events. You need a dashboard showing DLQ depth, an alert when it grows, and a replay tool that can re-enqueue a DLQ message with a fixed payload. Teams commonly discover their DLQ has 200,000 messages from three months ago when a customer asks why they never got their events.

The fourth trap is signature verification. If you sign payloads with HMAC-SHA256 and the receiver validates signatures, a retry with a regenerated timestamp will fail signature validation. Sign once at outbox insert time and store the signature with the event, or include the timestamp in the signed payload and have the receiver tolerate a window (typically 5 minutes).

## The broader lesson

The broader lesson is that durability is a property of where you store state, not how many times you retry. Every webhook system that survives outages does so because the pending work lives in a database or a queue that survives process death, and because the retry schedule is decoupled from the request lifecycle. Retries are cheap; losing events is expensive. The pattern is the same whether you're building webhooks, sending emails, or syncing to a third-party API: persist the intent, deliver asynchronously, retry with backoff, and make the receiver's job easy by sending a stable idempotency key.

The corollary is that you should design for the receiver being down, not for the receiver being up. Assume every endpoint will eventually return 500s for hours. If your system can't survive that without human intervention, it's not a webhook system — it's a best-effort notification, and you should tell your customers that.

## How to apply this to your situation

Start by checking whether you have an outbox table. If you don't, the first step is to add one and route your event publishing through it. You don't need to migrate everything at once — pick the highest-value event (usually `order.created` or `payment.succeeded`) and move just that one to the outbox pattern. Run both paths in parallel for a week, compare delivery counts, and then migrate the rest.

If you already have an outbox, the next step is to audit your retry schedule. Most systems have a fixed 3-retry policy that covers about 30 seconds. Extend it to at least 24 hours with exponential backoff and jitter, and add a dead-letter queue with an alert on depth. That single change is usually the difference between "we lost events during the outage" and "we delivered everything once the downstream came back."

## Frequently Asked Questions

**How do I prevent duplicate webhook deliveries?**

You can't prevent them entirely with at-least-once delivery, so the goal is to make them harmless. Send a stable `Idempotency-Key` header on every attempt, generated once when the event is created and reused across retries. Document it for your customers and recommend they dedupe on it. On your side, use `FOR UPDATE SKIP LOCKED` in your relay and set visibility timeouts longer than your p99 delivery latency.

**What's the right visibility timeout for SQS webhook delivery?**

Set it to at least 3x your p99 POST latency, and use `ChangeMessageVisibility` to extend it for long-running requests. A common starting point is 90 seconds if your p99 is around 20–30 seconds. If you set it too low, you get concurrent delivery of the same message; if you set it too high, failed messages take longer to retry.

**Should I use SQS FIFO or standard queues for webhooks?**

Standard queues unless you need strict ordering. FIFO queues cap at 300 messages/second per message group (3000 with batching) and cost more. For most webhook use cases, per-aggregate ordering is enough, and you can get that with standard queues by partitioning on the aggregate ID at the application level. Use FIFO only when the receiver genuinely requires global ordering.

**How long should I retry a failed webhook?**

About 24 hours is the common sweet spot. Beyond that, the event is usually stale and the customer would rather get a manual replay than a 3-day-old event. After 24 hours, move to a dead-letter queue, alert on it, and provide a replay tool. Retrying forever is a way to hide failures, not fix them.

## Resources that helped

- AWS SQS documentation on visibility timeout and at-least-once delivery: https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html
- PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED` documentation: https://www.postgresql.org/docs/15/sql-select.html
- The transactional outbox pattern, microservices.io: https://microservices.io/patterns/data/transactional-outbox.html
- Stripe's webhook best practices, which popularized the idempotency-key approach: https://stripe.com/docs/webhooks/best-practices

Your next step: open your webhook delivery code and find the retry policy. If it's a fixed 3-retry loop, replace it with an exponential backoff schedule of 10s, 30s, 2m, 10m, 1h, 6h, 24h with jitter, and add a dead-letter queue with an alert on depth. That's a one-file change and it's the highest-leverage thing you can do today.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** October 2026
