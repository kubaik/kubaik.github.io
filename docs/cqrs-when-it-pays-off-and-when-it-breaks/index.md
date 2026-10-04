# CQRS: when it pays off and when it breaks

## What CQRS actually changes

Command Query Responsibility Segregation separates the model that handles writes from the model that serves reads. In its simplest form that can mean two interfaces over one database. In its full form it means a write model that emits events or messages, a projection process that consumes them, and a read store shaped for queries rather than for invariants.

The second form is where the trouble lives. The moment the read model is updated asynchronously, the system is eventually consistent, and every read-after-write expectation in the product now depends on how fast the projection pipeline runs and how it behaves when a message arrives twice, out of order, or not at all.

The errors that follow are rarely caused by a single misconfiguration. They are caused by a mismatch between what the domain requires and what the architecture guarantees. The useful question is not "is CQRS good?" but "at what granularity, for which aggregates, and with what consistency contract?"

## Failure mode 1: no asymmetric load to justify the split

The usual motivation for CQRS is that reads and writes have different shapes and different volumes. If they do not, the pattern adds a broker, a projection engine, and a failure surface in exchange for nothing.

A typical symptom is a service that reads and writes roughly comparable volumes of simple records, yet has grown a message broker and a projection worker. The team now debugs two data paths and a lag metric for a workload that a single relational instance would serve comfortably.

**How to check.** Instrument both paths before deciding anything. Count queries per endpoint over a representative window, and record the ratio of read operations to write operations. Two practical ways:

- Application-level: increment a counter in the data access layer per operation type, then read the totals from your metrics backend over a week.
- Database-level: on PostgreSQL, `pg_stat_statements` gives per-statement call counts; sum the calls for `SELECT` versus `INSERT`/`UPDATE`/`DELETE` to get an empirical ratio.

The threshold is a judgement call, not a law. A ratio near 1:1 with simple CRUD is a strong signal that a read replica or even a single instance is sufficient. A ratio in the tens or hundreds to one, combined with a read shape that genuinely differs from the write shape, is where the pattern starts to earn its keep.

## Failure mode 2: using CQRS to avoid a transaction you still need

A recurring trap is adopting CQRS to remove lock contention, then discovering that the business rule requires the write to be visible to the very next read.

Consider a profile update. The user submits a change, the page reloads, and the user expects to see it. If the read model is updated asynchronously, the reload can return the old value. Teams then add compensating logic: a client-side optimistic update, a "read your writes" token that routes the user to the write model for a short window, or a synchronous projection for that one aggregate. Each of these is a workaround for a consistency requirement that a transaction would have satisfied directly.

**Decision test.** For each command, ask: must the effect be visible to a read issued by the same user immediately afterward? If yes for most commands, the domain wants strong consistency on that path, and CQRS is the wrong default. If the answer is no for most commands and yes for a small, enumerable set, a hybrid is viable: keep the asynchronous read model for the bulk, and serve the consistency-critical reads from the write model or a synchronously updated store.

## Failure mode 3: non-idempotent projections

This is the most common correctness bug in message-driven read models. Brokers commonly guarantee at-least-once delivery, which means duplicate delivery is normal, not exceptional. A projection that inserts on every message will eventually insert twice.

The classic failure is a unique constraint violation that halts the consumer. In some brokers, a poison message that keeps failing can block the partition or queue behind it, so one duplicate can stall the entire projection.

**The fix is to make every projection idempotent.** Two complementary techniques:

1. Upsert instead of insert, so a repeated message converges to the same row.
2. Carry a monotonically increasing version or sequence number in each event, and ignore any message whose version is not newer than what the read model already holds.

The following Node.js projection uses both. It assumes a `users_read` table with a primary key on `id` and a `version` column:

```javascript
async function projectUserCreated(event) {
  const { userId, name, version } = event;
  await db.query(
    `INSERT INTO users_read (id, name, version)
     VALUES ($1, $2, $3)
     ON CONFLICT (id) DO UPDATE
     SET name = EXCLUDED.name, version = EXCLUDED.version
     WHERE users_read.version < EXCLUDED.version`,
    [userId, name, version]
  );
}
```

The `WHERE` clause is the important part. Without it, a late-arriving older event would overwrite newer state. With it, out-of-order delivery is harmless and duplicate delivery is a no-op.

**How to verify.** Deliberately publish the same event twice against a test database and confirm the read model is unchanged after the second delivery. Then publish events out of order and confirm the final state matches the highest version. This is a cheap test that catches the majority of projection bugs before they reach production.

## Failure mode 4: latency budgets that assume synchronous reads

Eventual consistency has a latency cost, and that cost is environment-dependent. In a warm, co-located deployment the gap between a write and its appearance in the read model may be tens of milliseconds. In an environment with cold starts, separate compute for the projection, and a queue in between, the gap can grow into seconds.

The failure is not the lag itself; it is that the product was designed as if reads were synchronous. Users see stale data, retry, and sometimes issue the same command again.

**How to measure.** Record a timestamp when the write commits, propagate it through the event, and record the timestamp when the projection applies it. The difference is the projection lag. Emit it as a histogram, not an average, because the tail is what users experience.

For queue-based pipelines, also track consumer lag directly. For Kafka, `kafka-consumer-groups --describe --group <group>` reports the offset gap per partition. For RabbitMQ, the queue depth from the management API serves the same purpose. A depth or gap that grows without bound means consumers cannot keep up with producers.

**How to budget.** Decide the maximum acceptable staleness per read path, then compare it to the measured p99 lag. If the p99 exceeds the budget for a path that users perceive as immediate, that path needs a synchronous read or a client-side optimistic update, not a faster projection.

## Failure mode 5: treating the read model as disposable when it is not

If the read model can be rebuilt from the write side, it is disposable, and recovery from corruption is a replay. If it cannot, it is a second source of truth, and the system has two systems of record that can disagree.

The dangerous case is a read model that accumulates state not present in the events: counters incremented without a corresponding event, timestamps set at projection time, or fields derived from external calls. Replaying then produces a different result than the original run.

**Checklist for a rebuildable read model:**

- Every column is derivable from the event stream alone.
- Projection handlers are pure functions of the event plus the current read state, with no external calls that mutate.
- Any external enrichment is itself versioned and stored, so replay reads the same enrichment.
- There is a documented procedure to truncate and replay, and it has been exercised in a non-production environment.

If any of those fail, the read model is not disposable, and the operational story is much harder than the pattern's advocates usually admit.

## Comparing CQRS with simpler read scaling

The table below compares the full asynchronous pattern with read/write splitting on a single logical database. It is a qualitative comparison; the latency and cost figures are illustrative and depend entirely on your infrastructure.

| Factor | Full CQRS with async projections | Read replica / read-write splitting |
|---|---|---|
| Read/write ratio that justifies it | Strongly asymmetric, with differing read and write shapes | Any ratio |
| Consistency for reads after writes | Eventual; lag depends on the pipeline | Strong on the primary; bounded by replica lag |
| New infrastructure | Broker, projection workers, lag monitoring | One replica |
| Failure modes introduced | Duplicate, out-of-order, and lost messages; projection halts | Replica lag and failover |
| Read model shape | Can differ completely from the write model | Same schema |
| Operational burden | Higher: two data paths, replay procedures, lag budgets | Lower |

The point of the table is not that one column wins. It is that the right-hand column covers a large share of real read-scaling needs, and the left-hand column should be chosen for the cases it cannot cover: reads that need a fundamentally different shape, or a write model whose load characteristics genuinely diverge from the read model's.

## A worked sizing example

Suppose a service handles 200 writes per second at peak and 2,000 reads per second at peak, a 10:1 ratio. Assume each write is a small row update and each read is a primary-key lookup.

On a single PostgreSQL instance, a primary-key lookup is typically sub-millisecond of database time, and a small update is a few milliseconds including commit. At 2,000 reads per second, the reads consume a small fraction of one core if they are index lookups. The writes at 200 per second are the heavier side because each commit involves a durable write.

This arithmetic is illustrative, but the conclusion is general: at a 10:1 ratio with simple shapes, one instance plus a replica for read fan-out is usually adequate, and the replica removes the read load from the primary without introducing eventual consistency between a user's write and their own subsequent read, provided reads that must reflect the user's own write are routed to the primary or use a read-your-writes mechanism.

CQRS becomes interesting when the read shape diverges: full-text search across many fields, pre-aggregated dashboards over millions of rows, or a graph traversal that the write schema cannot serve efficiently. In those cases the read model is not a copy; it is a different representation, and building it asynchronously is the natural design.

## Verifying that a change actually helped

Whichever direction you move, verification follows the same shape.

1. **Instrument the write-to-read gap.** Log a correlation ID and timestamps at write commit and at read-model apply. Emit the difference as a histogram. Compare p50 and p99 before and after the change.
2. **Check for duplicates directly.** A query such as `SELECT id, COUNT(*) FROM users_read GROUP BY id HAVING COUNT(*) > 1;` returning zero rows is the expected result for a correctly idempotent projection.
3. **Watch consumer lag over time.** A flat or oscillating lag is healthy. A monotonically increasing lag means the consumers are under-provisioned or blocked.
4. **Replay in a test environment.** Truncate the read model, replay the full event stream, and compare the result to the previous state. Any difference indicates non-derivable state.
5. **Load test the write-then-read path.** Simulate a user writing and immediately reading. Count how often the read returns stale data. Compare that count against the staleness budget you set for that path.

The last step is the one teams skip, and it is the one that predicts user complaints.

## Prevention checklist

- Start with one database. Add a read replica before adding a broker.
- Adopt asynchronous projections only for read paths where staleness is acceptable, and write that contract down.
- Make every projection idempotent with an upsert plus a version guard.
- Keep the read model derivable from the event stream so it can be replayed.
- Set an explicit staleness budget per read path and alert on the p99 exceeding it.
- Design the UI for the consistency you actually have: optimistic updates or explicit "processing" states where reads may lag.
- Exercise the replay procedure before you need it.

## FAQ

**Is CQRS the same as event sourcing?**
No. CQRS separates read and write models. Event sourcing stores state as a sequence of events. They are often combined because an event stream is a convenient source for projections, but either can be used without the other.

**Can CQRS provide strong consistency?**
Yes, if the read model is updated synchronously in the same transaction as the write. At that point the read and write models are separate in code but not in consistency, which is a legitimate and often underrated middle ground.

**How do I handle a projection that has failed repeatedly?**
Route the failing message to a dead-letter queue with enough context to replay it, fix the handler, then replay the dead-lettered messages. Ensure the projection is idempotent first, or the replay will reproduce the failure.

**What replaces a distributed transaction across aggregates?**
Usually a saga: a sequence of local transactions with compensating actions for the steps that must be undone. Sagas are more complex than a single ACID transaction, so if the domain genuinely needs atomicity across aggregates, reconsider whether the aggregate boundaries are correct.

## Next step

Open the data access layer for one service and count read and write operations over a representative period, using application counters or `pg_stat_statements`. If the ratio is below roughly 10:1 and the read shape matches the write shape, you have a concrete candidate for replacing asynchronous projections with a read replica — and a measured number to justify the change.
