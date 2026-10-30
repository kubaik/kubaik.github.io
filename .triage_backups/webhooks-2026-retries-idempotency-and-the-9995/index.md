# Webhook Delivery: Retries, Idempotency, and Durability

Most webhook guides assume a clean environment and a patient timeline. Production gives you neither. The hard part is rarely the HTTP call — it is durability, ordering, and duplicate suppression under partial failure. This article walks through the failure modes that show up in real integrations, the invariants worth writing down, and a reference architecture that satisfies them.

## The situation

An integration platform that ingests transaction events from a dozen or more partners tends to hit the same wall. Each partner provides a webhook endpoint, and the platform has to guarantee delivery of every event — even if those partner servers are down for hours. An initial estimate of two weeks is common. Three months of firefighting is just as common.

A typical first customer complaint looks like this: a large wire transfer vanishes from the dashboard because a partner's webhook endpoint returned HTTP 500 for 45 minutes straight. Their retries used exponential backoff, but their buffer overflowed after 1,024 attempts, dropping events on the floor. A connection pool issue that consumes three days of debugging is usually a single misconfigured timeout.

The goal in this pattern: guarantee at-least-once delivery for 99.95% of events, with no more than 10 seconds of end-to-end latency, and a cost ceiling of $1.20 per 1,000 events. Those numbers are illustrative targets, not measured results — pick your own, but write them down before you choose components.

## What teams try first and why it doesn't work

The first attempt is usually the classic "fire-and-forget" pattern copied from a tutorial. A Node 20 LTS server behind an Application Load Balancer, using Express with a single POST endpoint:

```javascript
app.post('/webhook/:partner', async (req, res) => {
  const partner = partners[req.params.partner];
  const event = req.body;
  try {
    await fetch(partner.url, {
      method: 'POST',
      body: JSON.stringify(event),
      headers: { 'Content-Type': 'application/json' },
    });
    res.status(200).send('OK');
  } catch (err) {
    console.error(err);
    res.status(500).send('Failed');
  }
});
```

Park this on an EC2 t3.medium instance and call it a day. Within a week, three problems show up:

1. **No retries.** The tutorial told you to return 200 even on failure so the partner wouldn't retry you. That's backwards; you need to tell them to retry you.
2. **No idempotency.** The same event can arrive twice if the partner's retry overlaps with a server restart.
3. **No buffer.** When one partner's endpoint goes down for 20 minutes, their 500 retries saturate the connection pool, starving other partners.

The next rewrite usually uses a local SQLite queue plus a naive retry loop with exponential backoff. The latency skyrockets: p99 jumps because the event loop is blocked waiting for each retry. Then comes a managed queue such as Amazon SQS. Push events into a standard queue and have Lambda consumers poll. Latency drops, but a new problem appears: **ordering.** SQS FIFO guarantees order only within a single message group, and financial events from the same partner can still arrive out of order if the group key is chosen poorly. Reordering events in application code adds latency and doubles invocations.

## The invariants worth writing down

Before picking components, write the invariants. A useful starting set:

| Invariant | Why it matters |
|-----------|----------------|
| At-least-once delivery for 99.95% of events | Regulatory requirement for financial data |
| End-to-end latency ≤ 10 s p99 | User experience for real-time dashboards |
| No duplicate processing | Prevent double charges |
| Regional failover < 60 s | Disaster recovery SLA |

Combine three pieces:

1. **A durable stream buffer** — shard streams by partner ID so each partner's events live in a separate stream. Consumer groups on Redis Streams give you acknowledgement and redelivery semantics without a separate broker. Redis Streams support `XADD`, `XREADGROUP`, `XACK`, and pending-entry listing via `XPENDING` and `XAUTOCLAIM`.
2. **A fast consumer runtime** — Lambda with SnapStart, or any runtime with pre-warmed workers. Cold starts matter because they consume part of your retry budget.
3. **A durable anchor** — PostgreSQL with logical replication. Mirror the events table to a read-replica so even if the primary database fails, the raw events are still available to replay.

The flow:

1. Partner sends webhook → load balancer → consumer writes to a Redis Stream.
2. Consumer group reads from the stream with a blocking read (`XREADGROUP BLOCK 5000`).
3. Consumer writes the event to PostgreSQL and leaves the message *pending* in the consumer group.
4. On success, the consumer marks the message acknowledged (`XACK`); on failure, the message stays pending and is redelivered.
5. A separate archiver writes raw payloads to object storage on a schedule.

Add an idempotency key: a SHA-256 hash of `partner_id + event_id + source_timestamp`. Before processing, check `SELECT 1 FROM processed_events WHERE idempotency_key = ?`. If it exists, skip processing but still ack the message. This lets you safely replay failed batches without duplicates.

## Implementation details

### Architecture

```
Partner Webhook → ALB → Consumer (SnapStart) → Redis Stream →
  Consumer Group → PostgreSQL (primary) →
  Logical replication → PostgreSQL replica (for failover) →
  Idempotency store in Redis (TTL 7 days)
```

### Redis Stream setup

Use consumer groups to shard load. Each partner gets its own stream:

```bash
# Create stream for partner "acme"
redis-cli -h redis.internal XGROUP CREATE acme_stream acme_group $ MKSTREAM

# Add consumer
redis-cli -h redis.internal XGROUP CREATECONSUMER acme_stream acme_consumer1
```

The consumer code:

```python
import redis
import hashlib
import psycopg2

r = redis.Redis(host='redis.internal', port=6379, decode_responses=True)
conn = psycopg2.connect("host=pg-primary dbname=events user=webhook")

consumer_name = "worker-1"
stream_key = f"{partner_id}_stream"
group_name = f"{partner_id}_group"

while True:
    messages = r.xreadgroup(
        group_name, consumer_name,
        {stream_key: '>'},  # '>' means new messages
        count=100,
        block=5000,
        noack=False
    )
    for _, message_list in messages:
        for msg_id, fields in message_list:
            event = fields['event']
            idempotency_key = hashlib.sha256(
                f"{partner_id}:{event['event_id']}:{event['timestamp']}".encode()
            ).hexdigest()

            # Idempotency check
            if r.exists(f"idemp:{idempotency_key}"):
                r.xack(stream_key, group_name, msg_id)
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO events (idempotency_key, payload) VALUES (%s, %s)",
                        (idempotency_key, event)
                    )
                conn.commit()
                r.xack(stream_key, group_name, msg_id)
                r.setex(f"idemp:{idempotency_key}", 604800, "1")  # 7 days TTL
            except Exception as e:
                # On error, message stays pending and will be redelivered
                print(f"Failed to process {msg_id}: {e}")
```

Note the ordering: commit to PostgreSQL *before* `XACK`. If the process dies between the two, the message is redelivered and the idempotency check absorbs the duplicate.

### PostgreSQL schema

```sql
CREATE TABLE events (
    idempotency_key TEXT PRIMARY KEY,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_events_received ON events(received_at);
CREATE INDEX idx_events_processed ON events(processed_at);
```

Use logical replication to mirror the `events` table to a read-replica:

```sql
-- On primary
CREATE PUBLICATION events_pub FOR TABLE events;

-- On replica
CREATE SUBSCRIPTION events_sub
CONNECTION 'host=pg-primary port=5432 dbname=events user=repl'
PUBLICATION events_pub;
```

Monitor replication lag. On the replica, `SELECT now() - pg_last_xact_replay_timestamp()` gives a usable estimate. If lag exceeds your threshold, route reads to the replica only for non-critical paths.

### Retry strategy

Fixed exponential backoff without jitter produces synchronized retry storms. Two things help:

- **Jittered delay.** Increase the base delay on each failure up to a ceiling, then add random jitter:

```python
import random

def get_retry_delay(failure_count, base_seconds=5, ceiling=30):
    base = min(base_seconds * failure_count, ceiling)
    jitter = random.uniform(0, base * 0.2)
    return base + jitter
```

- **Circuit breaker per partner.** After N consecutive failures within a window, stop retrying for that partner for a cooldown period and alert. This prevents burning through consumer concurrency when a partner's endpoint is truly down.

### Cost model

Do not copy someone else's cost table. Build your own from your provider's published rates and your measured per-event resource usage. The components that typically dominate:

| Component | What to measure |
|-----------|-----------------|
| Compute | Invocations × duration × memory, plus cold-start overhead |
| Buffer | Memory used by streams and pending entries, plus network egress |
| Database | Storage, IOPS, and replication bandwidth |
| Load balancer | Request count and processed bytes |
| Observability | Metric and log ingestion volume |

Run the numbers with a pricing calculator using your own event volume and payload size. Small changes in payload size or retention window change the answer more than component choice does.

## How to measure the guarantees

Every claim in this pattern is measurable. Instrument these:

- **Delivery rate.** Count events accepted at the edge versus events acknowledged in the store. The ratio is your delivery rate. Emit it as a counter, not a gauge, so restarts don't reset it.
- **Duplicate rate.** Count idempotency-key hits. A nonzero rate is expected; a *growing* rate means your retry policy is too aggressive or your ack path is broken.
- **Latency.** Record time from edge receipt to durable commit. Publish p50, p95, and p99. If p99 is far above p50, you have a queueing or retry problem, not a network problem.
- **Pending entries.** `XPENDING` tells you how many messages are claimed but unacknowledged. A steadily growing pending set means consumers are failing silently.
- **Replication lag.** Track it with the query above and alert on sustained increases, not single spikes.

A synthetic probe that sends a small number of test events per hour and asserts end-to-end completion catches most regressions before customers do. Keep the probe's events tagged so they can be filtered out of business metrics.

## Failure modes to expect

- **Ack before commit.** If you `XACK` before the database write commits, a crash loses the event. Always commit first.
- **Idempotency key drift.** If the key includes a field the partner changes between retries (for example, a re-generated timestamp), duplicates slip through. Hash only stable fields.
- **Pending entry starvation.** Messages that fail repeatedly sit in the pending list forever unless you claim them. Use `XAUTOCLAIM` to move stale pending entries to a live consumer.
- **Poison messages.** A malformed payload that always throws will be retried forever. After a bounded number of attempts, move it to a dead-letter stream and alert.
- **Partner clock skew.** If you sort by a partner-supplied timestamp, skew can reorder events. Sort by your own receipt time and treat the partner timestamp as data.
- **Replica cold start.** A read-replica that has been idle can be slow on first query after failover. Warm it with periodic reads, or use a serverless database tier that scales on demand.

## Decision checklist

1. **Buffer.** A stream with consumer groups if you want low latency and built-in acknowledgment. A managed queue if you want zero ops and can tolerate its ordering model. PostgreSQL alone if volume is low and you value SQL over throughput.
2. **Idempotency.** Decide the key format now: `partner_id:event_id:source_timestamp`. Store it with a TTL longer than your maximum retry window.
3. **Retry policy.** Bounded attempts, jittered delays, per-partner circuit breaker, dead-letter destination.
4. **Durability.** Commit before ack. Replicate the store. Archive raw payloads to object storage on a schedule.
5. **Observability.** Counters for delivery and duplicates, histograms for latency, a gauge for pending entries, and a lag metric for replication.

## FAQ

**How do I handle partners that don't support idempotency keys?**

Most partners will ignore your request to include one. Generate your own key from stable fields in the request body. If the partner sends no unique event ID, hash the payload minus volatile fields like timestamps.

**At-least-once versus exactly-once?**

At-least-once means the message is delivered one or more times but never lost. Exactly-once is at-least-once delivery plus idempotent processing: the system guarantees no duplicates even if the message is redelivered.

**How do I scale streams to thousands of partners?**

Shard by partner ID. Monitor memory per stream and archive old entries. If a single stream grows beyond your configured `MAXLEN` or memory budget, trim it and rely on the durable anchor for replay.

**Why not Kafka?**

Kafka gives ordering guarantees and long retention, but it requires operating a cluster. Choose it if you need strict global ordering or retention measured in weeks. A stream buffer plus a relational anchor covers most webhook workloads with less operational surface.

**How much storage do I need?**

Estimate from your own payload sizes and retention. As an illustrative example: if the average message is 500 bytes and you retain 1 million messages, that is roughly 500 MB before overhead. Measure actual memory with `MEMORY USAGE` on a sample key and multiply.

## Do this in the next 30 minutes

Run this query against your events table (or create the table first if it doesn't exist):

```sql
SELECT
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE processed_at IS NULL) AS pending_events,
    COUNT(*) FILTER (WHERE idempotency_key IS NULL) AS missing_keys
FROM events;
```

If you have pending events older than five minutes, or any rows with missing idempotency keys, you have found your next incident. Fix the missing keys first: they are the ones that will duplicate on the next replay.
