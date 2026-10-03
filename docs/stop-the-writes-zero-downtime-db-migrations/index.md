# Stop the writes — zero-downtime DB migrations

## The conventional playbook and where it breaks

The standard playbook for zero-downtime database migrations goes like this: start a background copy, add a trigger or application-level dual-write to capture changes, watch replication lag, then cut traffic once the two sides look consistent. Repeat per table. It is the approach taught in most migration guides, and it works often enough to be dangerous.

The gap is the assumption that the application can keep mutating the old schema while the copy runs. Background copy and dual-write are *convergence* mechanisms, not *consistency* mechanisms. They close the gap between two datasets eventually. They do not give you a point in time at which both sides are known to agree. That point has to be manufactured, and the only reliable way to manufacture it is to stop new writes to the old schema first.

A typical failure mode: a large table is copied in the background while the application still writes to the old schema. Replication lag looks acceptable on the dashboard because the dashboard measures average lag, not the age of the oldest unapplied change. Under peak load the lag grows, the cut-over proceeds anyway, and some in-flight writes land on the old side after the new side has been declared authoritative. The result is divergence that surfaces later as stale reads, duplicate records, or lost updates.

## Three failure modes worth knowing

**1. Lag that is green on the dashboard but red in reality.** Managed replication and logical decoding expose lag as a time delta between the last received and last applied change. That number hides the tail. A p50 lag of 200 ms can coexist with a p99 lag of 30 seconds, and it is the tail that decides whether your cut-over is safe. The fix is instrumentation, not a different tool: sample the lag metric at a high rate and alert on the maximum, not the mean.

**2. Race conditions between dual-write and the background copy.** When the application writes to both stores, those two writes are not atomic. A crash, a timeout, or an out-of-order retry between the two writes leaves the stores divergent. This is a well-documented property of dual-write, not a bug in any particular library. The mitigation is either an outbox pattern (write the change once, to a durable log, and let a single consumer apply it to both stores in order) or a drain phase that removes the window entirely.

**3. Table rewrites on the primary.** Adding a `NOT NULL` column with a non-null default forces a full table rewrite in most engines, and a rewrite takes an `ACCESS EXCLUSIVE` lock that blocks reads as well as writes. The safe sequence is: add the column as nullable, backfill in bounded batches, then add the constraint with validation. In PostgreSQL, `ALTER TABLE ... ADD CONSTRAINT ... NOT VALID` followed by `VALIDATE CONSTRAINT` avoids holding a long lock. The exact locking behavior is version-specific and should be checked against the documentation for your engine before running it in production.

In each case the root cause is not the migration tool. It is the assumption that the application can keep writing while the migration runs.

## A different mental model: drain, copy, flip

Treat the migration as a state machine with three phases and one invariant.

1. **Drain** — stop accepting new writes to the old schema. Reads continue.
2. **Copy** — perform the schema or data transformation, with the write path quiesced.
3. **Flip** — point the application at the new schema and resume writes.

The invariant: *the copy phase must run against a dataset that is not being mutated.* Everything else — dual-write, triggers, logical replication — is plumbing that lets you shorten the drain window, not replace it.

The drain phase is the part teams under-build. A drain is not "stop the app." It is:

- Set a flag that makes the write path reject new mutations for the affected tables (return a retryable error, not a 500).
- Wait for in-flight requests to complete, bounded by a deadline.
- Wait for the write queue to reach zero depth, confirmed by a metric the orchestrator can read.
- Emit a `drain-complete` event only after the queue has stayed empty for several consecutive checks.

The consecutive-check requirement matters. A single observation of zero depth can be a transient between two batches. Three consecutive checks at 500 ms intervals is a reasonable starting point; the right number depends on your producer's batching behavior.

Two details that are easy to get wrong:

- **Prefetch and acknowledgement semantics.** A consumer with a large prefetch holds messages that are not yet acknowledged. Queue depth metrics that count only ready messages will report zero while unacknowledged messages are still in flight. Drain on `ready + unacknowledged`, not on ready alone.
- **Publishers that retry.** If a producer retries on timeout, a write you thought you rejected can reappear after the drain. The write path must return a non-retryable status for the duration of the drain, or the producer must be given a drain-aware backoff.

## Instrumenting the drain

The drain gate is only as good as the metric behind it. What to measure and how:

- **Queue depth**: `ready` and `unacknowledged` separately, sampled at least once per second. For RabbitMQ, `rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers` or the equivalent HTTP API call. For Kafka, consumer group lag per partition via `kafka-consumer-groups --describe`.
- **Replication lag**: for PostgreSQL logical replication, the difference between `pg_current_wal_lsn()` on the primary and `pg_last_wal_replay_lsn()` on the subscriber, converted to bytes and then to time using the observed apply rate. Do not use the built-in lag view alone; it reports time since last commit, which is zero when there is no traffic.
- **In-flight request count**: from your application's own metrics, so you can confirm the drain deadline is not being hit by a long-running request.
- **Lock waits during backfill**: `pg_locks` filtered on `AccessExclusiveLock`, or the equivalent for your engine.

To validate a drain gate before you trust it, replay a recorded peak-load trace against a staging copy and record how long the queue takes to reach zero. That number is your drain budget. If it exceeds your acceptable cut-over window, the drain gate is not the problem — the write volume is, and you need to shard or batch the flip.

## Designing schemas so the flip is boring

If the flip is the risky part, the way to reduce risk is to make the schema change additive.

- Add new columns as nullable with no default. Backfill in batches. Add the constraint afterward, validated separately.
- Never rename a column in place. Add the new name, dual-read (not dual-write) during the transition, then drop the old column after the drain.
- Keep the old and new schemas readable by the same application version for one release. This lets you roll back the flip without rolling back the data.
- Bound backfill batches by rows *and* by time. A batch that takes longer than your lock timeout is a batch that will block.

A worked example of the batching arithmetic: suppose a table has 100 million rows and you want the backfill to finish in under an hour. That is 3,600 seconds, so you need roughly 27,800 rows per second. If a single-row update takes 0.2 ms on your hardware, a batch of 100 rows takes about 20 ms of work plus network round-trip. At 50 batches per second you get 5,000 rows per second, which is far short of the target. Either raise the batch size until the round-trip cost is amortized (a batch of 1,000 rows at 0.2 ms each is 200 ms of work, and 5 batches per second is 5,000 rows per second — still short), or accept a longer window. The point of the arithmetic is to show that backfill throughput is dominated by round-trips until batches are large, and that large batches are what cause lock contention. There is no batch size that is both fast and unobtrusive; you pick which problem you want.

## When the conventional playbook is fine

Draining is not free. It adds a gate, a metric, and a failure mode of its own. It is not worth building for every migration.

Use background copy with dual-write or triggers when all of the following hold:

- The table is read-heavy, with writes infrequent enough that the divergence window is measured in milliseconds.
- The table is small enough that the copy completes inside a single maintenance window.
- The data is not on a critical path: logs, audit trails, metrics, caches.
- Eventual consistency is acceptable for the duration of the migration.

Use a drain-copy-flip when any of these hold:

- The write rate is high enough that replication lag has a meaningful tail.
- The data is on a critical path: payments, sessions, entitlements, inventory.
- You cannot tolerate a divergent read at any point.
- The cut-over window must be bounded and provable, not best-effort.

## A decision checklist

Run through these before choosing an approach. Each answer should be a number or a yes/no, not a feeling.

1. **Peak write rate** to the affected tables, in rows per second, measured over the busiest hour you have data for.
2. **Tail replication lag** at that peak, p99 not p50.
3. **Acceptable cut-over window** in seconds, agreed with the stakeholders who own the SLA.
4. **Consistency requirement**: can a read observe the old value after the flip has been declared complete? If no, you need a drain.
5. **Rollback plan**: can you flip back without a data migration? If not, the flip is one-way and needs more rehearsal.
6. **Producer behavior on rejection**: does your write path retry on error, and with what backoff?
7. **Observability**: can the orchestrator read queue depth and replication lag as metrics, or does it have to poll a UI?

If you cannot answer 1, 2, and 3 with numbers, you are not ready to choose. Measure first.

## Common objections

**"Draining adds latency."** Draining adds latency only during the drain window, and only to writes. Reads are unaffected. The alternative — dual-write — adds latency to every write for the entire duration of the migration, not just the cut-over. If your write path has a tight latency budget, a short drain is usually cheaper than a long dual-write.

**"We can't pause writes; our availability target forbids it."** You are not pausing writes; you are rejecting them with a retryable status for a bounded window. Whether that counts against availability depends on how you define the SLO. If the SLO is measured on successful requests, a retryable rejection that the client retries successfully does not consume error budget. Decide this explicitly rather than assuming.

**"Our ORM can't handle a schema change without a restart."** Use a database-first approach: create the new structure, backfill, then update the ORM's mapping and deploy. The restart is of the application, not the database, and it can be a rolling deploy. Most ORMs also support reading a generated schema file rather than managing migrations themselves, which decouples the two.

**"Custom drainers are risky; use a managed service."** A managed replication service optimizes for throughput and convergence. It generally does not expose a "the queue is empty and stable" signal you can gate a cut-over on. A small drainer that reads your existing queue metrics and publishes a single event is a few hundred lines and has a narrow failure surface. The risk is in the gate logic, not the language.

## Operational notes

- **Connection pools.** A drain that empties the queue can leave the new store cold. Pre-warm the pool and, if the new store scales to zero when idle, keep a low-rate synthetic load running until the flip.
- **Idle scale-to-zero.** Serverless database tiers that scale to zero introduce a cold-start latency on the first write after the flip. Either disable scale-to-zero for the migration window or keep a heartbeat write running.
- **Client library defaults.** Default prefetch values vary and are not always what you expect. A prefetch of zero or unlimited changes the meaning of "queue depth" entirely. Set it explicitly and document it.
- **Clock skew.** If the drain gate compares timestamps from two hosts, skew can make a stable queue look unstable or vice versa. Compare sequence numbers or counts, not wall-clock times.

## Frequently asked questions

**How do I drain a queue without losing messages?**
Use manual acknowledgement with a small prefetch. Stop consuming new messages, let the in-flight ones be acknowledged or returned to the queue, then observe depth reaching zero. Messages that were never acknowledged remain available. Verify with a depth query rather than assuming.

**Can logical replication replace the drain?**
No. Logical replication is a copy-plus-stream mechanism. It converges, but it does not give you a point at which the subscriber is provably caught up under load, because the lag has a tail. It is a good *copy* mechanism; it is not a *cut-over gate*.

**What batch size should a backfill use?**
Start at 100 rows per transaction and measure. Increase until either throughput stops improving or lock waits appear, then back off. There is no universal number; it depends on row size, index count, and storage latency.

**Do indexes need rebuilding after a migration?**
Only if you changed the indexed columns or added an index. For new indexes on a live table, use the concurrent build path your engine provides, and expect reduced write throughput for the duration.

**How do I test the drain gate?**
Replay a recorded peak-load trace against a staging copy and measure time-to-zero. If that number exceeds your cut-over budget, the gate will not save you.

## Action for the next 30 minutes

Pick the busiest table you plan to migrate and measure its write queue depth and consumer lag right now, under current load. For RabbitMQ:

```bash
rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers
```

For Kafka, per partition:

```bash
kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group your-group
```

Write down two numbers: current depth and current lag. Then run a load test at your expected peak and write down the same two numbers again. If depth does not return to zero within your acceptable cut-over window, you need a drain gate — and you now have the baseline to size it.
