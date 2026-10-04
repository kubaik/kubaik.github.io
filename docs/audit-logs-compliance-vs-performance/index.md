# Audit logs: compliance vs performance

Audit logging is usually taught as a binary: block the request to guarantee durability, or fire-and-forget and hope the record lands. In practice the hard parts are ordering, schema evolution, retention, and the atomicity between the business write and the audit write. Treating audit events like debug output is how a team ends up with a compliance finding months later. This article separates the durability requirement from the performance requirement, and shows how to measure both.

## Why audit logs are not application logs

Application logs are best-effort. Losing a debug line during a deploy is acceptable. Audit logs carry different requirements:

- **Durability.** If the action happened, the record must exist. "Eventual consistency" is not an acceptable answer in many regulated environments.
- **Ordering.** Events for the same entity must be reconstructable in the order they occurred. A delete logged before the update that preceded it is logically impossible and will raise questions.
- **Immutability.** Records must not be alterable after the fact. Append-only storage or cryptographic chaining is the usual mechanism.
- **Retention.** Records must survive for a defined period, often measured in years, and must be retrievable on demand.

The conventional advice — ship audit events to a separate service asynchronously — addresses throughput and decoupling but not atomicity. It is incomplete, not wrong.

## What goes wrong with naive async shipping

A common pattern is to produce an audit event to a message queue and have a consumer persist it to object storage or a database. This works until the following failures appear.

**Lost events on crash.** The application acknowledges the user request after enqueuing the audit event. If the process crashes before the consumer persists it, the record is gone. The gap between enqueue and persist is the exposure window. Under load, that window is bounded by consumer lag, which can be milliseconds or seconds depending on backlog.

**Out-of-order writes.** If two events for the same entity are processed by different consumers, they can be written out of order. Partitioning by entity ID fixes this in systems that support partition keys; a shared consumer pool does not.

**Schema drift.** Audit events must remain queryable for years. Changing the schema means either versioning events or migrating old data. Teams that treat audit logs like application logs often end up with a mix of incompatible formats that no query can span.

**Unbounded local buffering.** If the shipper stalls, a local queue grows. Without a bound and a dead-letter path, the disk fills and the service fails in a way that also loses audit data.

A useful mental model is the transactional outbox: write the audit event to a durable local store in the same transaction as the business data, then ship asynchronously. This gives atomicity with the business operation and keeps the request-path write fast, because the local write is small and local.

## The pattern: local durable queue plus async shipper

The design has three parts:

1. **Request path.** Write the audit event to a local durable store in the same transaction as the business operation. This can be a table in the same database, a SQLite file, or a write-ahead-log-backed store.
2. **Shipper.** A background process reads events in order, batches them, and writes to the central audit store.
3. **Central store.** Append-only, immutable, with a retention policy. Object storage with object lock, or a ledger-style database.

Ordering is preserved per entity by reading the local queue in insertion order and by partitioning the central store by entity ID. If the shipper processes events strictly in sequence, per-entity order is maintained without any partition key at the central store.

The latency cost is the local write. The central store cost is amortized by batching.

## A worked example

Assume an order service that handles 1,000 requests per second at peak, and every order state change must be audited. Assume each audit event is 1 KB.

**Option A: synchronous write to a central database.**
Each request performs one extra remote write. If the round trip to the central database is 5–10 ms, the request path gains that much latency. The central database must sustain 1,000 writes per second on top of its existing load. That is achievable, but it requires capacity planning and indexing discipline.

**Option B: local write plus async batch shipping.**
Each request performs one local write. If the local write is 1–3 ms, the request path gains that much. The shipper reads batches of, say, 1,000 events and writes one object per batch.

Arithmetic, all illustrative:

- Events per second: 1,000
- Batch size: 1,000 events
- Central writes per second: 1,000 / 1,000 = 1
- Events per day: 1,000 × 86,400 = 86,400,000
- Bytes per day at 1 KB per event: 86,400,000 KB ≈ 82.4 GB per day
- Bytes per year: 82.4 GB × 365 ≈ 30,100 GB ≈ 30.1 TB per year

Those figures are illustrative and depend entirely on event size, event rate, and retention. They are included to show the arithmetic, not to predict any specific bill. Storage cost depends on the tier and the retention policy; the point is that batching moves the central write count from per-event to per-batch, which is the dominant lever.

## Reference implementation

The following is a minimal local-queue-plus-shipper pattern. It is intentionally simple; it is not production-hardened, and the failure modes are discussed below.

```python
import sqlite3
import json
import threading
import time
import boto3

# Local audit queue. check_same_thread=False allows the shipper thread
# to use the same connection; production code should use a connection
# per thread or a connection pool.
conn = sqlite3.connect('audit_queue.db', check_same_thread=False)
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('''CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
)''')

def log_audit_event(entity_id, event_type, payload):
    conn.execute(
        'INSERT INTO audit_events (event_time, entity_id, event_type, payload) '
        'VALUES (?, ?, ?, ?)',
        (time.time(), entity_id, event_type, json.dumps(payload))
    )
    conn.commit()

def shipper():
    s3 = boto3.client('s3')
    while True:
        time.sleep(1)
        rows = conn.execute(
            'SELECT id, event_time, entity_id, event_type, payload '
            'FROM audit_events ORDER BY id LIMIT 1000'
        ).fetchall()
        if not rows:
            continue
        lines = [
            json.dumps({
                'id': r[0],
                'event_time': r[1],
                'entity_id': r[2],
                'event_type': r[3],
                'payload': r[4],
            })
            for r in rows
        ]
        s3.put_object(
            Bucket='audit-logs',
            Key=f'batch-{int(time.time())}.jsonl',
            Body='\n'.join(lines),
        )
        conn.execute('DELETE FROM audit_events WHERE id <= ?', (rows[-1][0],))
        conn.commit()

threading.Thread(target=shipper, daemon=True).start()
```

Correctness notes on this snippet:

- `check_same_thread=False` disables SQLite's thread check. It does not make concurrent writes safe by itself. SQLite serializes writes internally, but sharing one connection across threads without a lock can interleave statements. A lock around `log_audit_event`, or a connection per thread, is the safer choice.
- `PRAGMA journal_mode=WAL` improves concurrent read/write behavior. It is set here for that reason.
- The `DELETE` after a successful `put_object` is the commit point. If the process dies between the upload and the delete, the batch is uploaded again on restart. That is at-least-once delivery, which is the correct trade-off for audit data: duplicates are preferable to gaps, and duplicates can be de-duplicated downstream by `id`.
- The table is unbounded. A real deployment needs a retention bound on the local queue and a dead-letter path for batches that repeatedly fail to upload.

## Failure-mode analysis

**Shipper dies.** Events accumulate in the local queue. On restart, the shipper resumes from the oldest unshipped row. No loss, provided the local disk survives. Monitor queue depth and alert on growth.

**Local disk dies.** Events that were written but not yet shipped are lost. Mitigations: replicate the local store, ship more frequently, or use a store with replication. This is the residual risk of the pattern and should be stated explicitly to auditors.

**Central store unavailable.** The shipper retries. The local queue grows. A bound and a dead-letter path are required so the service does not fail on disk exhaustion.

**Duplicate uploads.** Inherent to the delete-after-upload commit point. De-duplicate by event `id` at the central store.

**Clock skew.** Timestamps from different hosts may not be comparable. Use NTP, store UTC with millisecond precision, and consider a logical sequence number per entity in addition to wall-clock time.

**Schema change.** Version the event envelope from the start. Store a schema version field and keep old readers working.

## How to measure the overhead

Do not estimate; measure. The relevant metrics are:

- **Request latency.** Compare p50 and p99 latency with audit logging enabled and disabled. Instrument the audit write as its own span so its contribution is attributable.
- **Queue depth.** Expose the count of unshipped rows as a gauge. Alert on sustained growth.
- **Shipper lag.** Measure the time between an event's `event_time` and the time it becomes durable in the central store. Alert on a threshold.
- **Shipper error rate.** Count failed uploads and retries.
- **Disk usage of the local queue.** Alert before it fills.

A practical test: run a load generator against a staging instance, record baseline p99, enable audit logging, record p99 again. Then kill the shipper and confirm that queue depth grows and no events are dropped from the local store. Then restore the shipper and confirm the queue drains.

## When the conventional advice is right

- **Low throughput.** Below roughly 100 requests per second, and when 10–20 ms of added latency is acceptable, a synchronous write to a central store is simpler and easier to reason about.
- **No strict durability requirement.** If the logs are for debugging or operational visibility rather than regulatory audit, fire-and-forget to a log platform is fine.
- **Managed audit sources.** If the events of interest are API calls to a cloud provider, the provider's own audit service may already cover them. Its delivery latency and retention are documented by the provider and should be checked against the requirement.

## Decision checklist

1. Does the regulation require the audit record to exist before the action is considered complete? If yes, the local write must be in the same transaction as the business operation.
2. What is the latency budget? Sub-millisecond budgets rule out synchronous remote writes.
3. What is the peak event rate? Batching matters above roughly 1,000 events per second; below that, simplicity usually wins.
4. How long must records be retained? This drives the storage tier and the lifecycle policy.
5. What is the acceptable loss window? Any async design has one; state it explicitly.
6. Who monitors the shipper? Queue depth, lag, and error rate need owners and alerts.

## Comparison of approaches

| Approach | Latency overhead | Durability | Ordering | Complexity |
|---|---|---|---|---|
| Synchronous to central store | Highest | Strong | Strong with transactions | Low |
| Async fire-and-forget | Lowest | Weak; events can be lost | None guaranteed | Low |
| Local queue plus async shipper | Low | Strong while local disk survives | Strong if shipped in order | Medium |
| Memory-mapped queue with replication | Very low | Strong with replication | Strong | High |

Latency figures are omitted deliberately; they depend on storage, network, and hardware and should be measured on the target system.

## Objections and responses

**"Local queues add operational complexity."** They do. The complexity is a bounded queue, a retry path, and monitoring. The alternative is an unprovable claim that no events were lost.

**"Kafka solves durability and ordering."** Kafka provides durable, ordered, partitioned logs. It does not by itself make the audit write atomic with the business transaction. That still requires the outbox pattern or a transaction that spans both systems.

**"Auditors don't care about latency."** They care about completeness and integrity. They will ask for evidence that no events were lost. A local queue with monitoring is evidence; fire-and-forget is not.

**"This is over-engineering for a small app."** If there are no strict requirements, it is. If there are, the cost of the pattern is small relative to the cost of a finding.

## Schema, retention, and immutability

- **Schema.** Use a versioned envelope. Include event ID, timestamp, entity ID, event type, actor, and a schema version. Prefer a self-describing or schema-registry-backed format so old events remain readable.
- **Retention.** Retention periods are set by the applicable regulation, not by preference. Common figures cited for financial services and healthcare are years, not months, but the specific requirement should be confirmed with counsel. Use lifecycle policies to move older data to cheaper tiers.
- **Immutability.** Append-only storage with object lock, or a database table restricted to inserts, or cryptographic chaining where each event includes a hash of the previous event. Chaining makes tampering detectable without requiring special storage.

## FAQ

**How is immutability enforced?**
Append-only storage with object lock, insert-only database permissions, or a hash chain over events. The first two prevent modification; the third makes it detectable.

**How is ordering preserved across services?**
Partition by entity ID so all events for one entity are handled by one consumer, and ship from the local queue in insertion order. Avoid parallel processing for the same entity.

**How long should records be retained?**
Whatever the applicable regulation requires. Confirm with counsel rather than assuming.

**Can a cloud provider's audit service cover application events?**
Provider audit services typically capture control-plane and API activity, not business events inside an application. They can complement application-level audit logging but usually do not replace it.

**What metrics matter most?**
Queue depth, shipper lag, and shipper error rate. These three detect the failure modes that cause data loss or unbounded growth.

**How can loss be tested?**
Kill the shipper, fill the disk in a test environment, and partition the network to the central store. Confirm that events remain in the local queue and are eventually shipped, and that alerts fire.

**What about the right to erasure?**
Audit retention obligations and erasure requests can conflict. A common approach is to store pseudonymized identifiers in audit records and keep the mapping in a separate store that can be erased. Legal review is required.

**Does a message queue replace the local outbox?**
No. A message queue decouples consumers from producers but does not make the audit write atomic with the business transaction. The outbox is what provides that atomicity.

## The one thing to do in the next 30 minutes

Pick one write path in your service and instrument the audit write as its own timing span. Run your existing load test, record the p99 latency of that span, and check whether the audit write currently happens in the same transaction as the business write. That single measurement tells you whether you have a latency problem, a durability problem, or both.
