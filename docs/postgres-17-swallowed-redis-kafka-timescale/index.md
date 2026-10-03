# Postgres 17 swallowed Redis, Kafka, Timescale

## What the consolidation argument actually claims

Postgres has accumulated extensions that cover ground once owned by separate systems: in-database job scheduling, partitioned tables with automatic retention, approximate vector search, and queue-style message tables. The pitch is that a single Postgres deployment can absorb three categories of infrastructure — a cache, a lightweight event bus, and a time-series store — and that the reduction in moving parts outweighs the loss of specialised features.

That pitch is sometimes right. It is most often wrong when it is stated as a blanket rule. This article lays out what each substitution really costs, where each one breaks, and how to measure the trade-off on your own workload rather than trusting a comparison table.

The framing to keep in mind throughout: you are not choosing between "Postgres" and "Redis". You are choosing between one operational surface with weaker primitives, and three operational surfaces with stronger ones. The question is whether your workload actually depends on the stronger primitives.

## The three substitutions, stated precisely

Each of the three replacements is a different kind of bet.

**Cache.** Redis is an in-memory data structure server with its own eviction policies, TTL semantics, pub/sub, and persistence modes. Postgres has a buffer pool that caches disk blocks, plus unlogged tables and temporary tables. The substitution is not "Postgres is a cache"; it is "for read-mostly hot data, the buffer pool plus a small unlogged table may be fast enough that a network hop to a separate cache is not worth it."

**Event bus.** Kafka is a partitioned, replicated, ordered log with consumer groups, offset management, and configurable retention. Postgres queue extensions are tables with visibility timeouts and archive functions. The substitution is "for fire-and-forget work where per-key ordering and replay are not required, a table-backed queue is adequate."

**Time series.** A dedicated time-series database typically offers columnar compression, continuous aggregates, and retention policies tuned for append-heavy metric data. Postgres offers declarative partitioning plus a scheduler that can pre-create and drop partitions. The substitution is "for metrics where you control the retention window and query patterns, partitioned Postgres tables may be sufficient."

None of these substitutions is free. Each one moves complexity from infrastructure into SQL and schema design.

## Cache: what you gain and what you lose

### The mechanism

A common pattern is an unlogged table keyed by the entity you want to cache, written on read miss and read on subsequent hits, relying on the buffer pool to keep hot pages resident. Unlogged tables skip WAL for their contents, which makes writes cheaper at the cost of being truncated after a crash — acceptable for data you can rebuild.

```sql
CREATE UNLOGGED TABLE session_cache (
  session_id   uuid PRIMARY KEY,
  payload      jsonb NOT NULL,
  expires_at   timestamptz NOT NULL,
  last_access  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX session_cache_expires_idx ON session_cache (expires_at);

-- Upsert on write
INSERT INTO session_cache (session_id, payload, expires_at)
VALUES ($1, $2, now() + interval '30 minutes')
ON CONFLICT (session_id)
DO UPDATE SET payload = EXCLUDED.payload,
              expires_at = EXCLUDED.expires_at,
              last_access = now();
```

Expiry is enforced by a scheduled job rather than by the storage engine:

```sql
DELETE FROM session_cache WHERE expires_at < now();
```

If the `pg_cron` extension is installed and listed in `shared_preload_libraries`, that statement can be scheduled:

```sql
SELECT cron.schedule('expire-sessions', '* * * * *',
  $$DELETE FROM session_cache WHERE expires_at < now()$$);
```

### What this costs you

The expiry job is a full scan of the expiry index on every run. At small table sizes that is irrelevant; at large ones it becomes a background write load that competes with your foreground traffic. Redis expires keys lazily and in a background cycle without you writing the sweep.

Eviction is the sharper problem. Redis will evict under memory pressure according to a policy you choose. Postgres has no per-table eviction policy. When the table outgrows `shared_buffers`, reads start hitting disk, and a cache that hits disk is slower than no cache at all because you now pay both the lookup and the miss. A cache in front of Postgres usually degrades gracefully; a cache inside Postgres degrades into a disk read.

The failure mode to watch for: latency looks fine during testing, then a data growth event pushes the working set past `shared_buffers` and p99 read latency steps up without any error being logged. Instrument `pg_stat_user_tables` hit ratios and `pg_statio_user_tables` read counts for the cache table specifically, and alert when the hit ratio for that table drops below your threshold.

### How to measure whether it is worth it

Do not run a synthetic benchmark. Instrument the real path.

1. Record the current p50, p95 and p99 latency of the code path that talks to the cache, and the cache's own hit ratio.
2. Record the query count and rows returned for that path from `pg_stat_statements`.
3. Build the Postgres version behind a flag, and split traffic so both paths run concurrently.
4. Compare p99 and the number of disk reads per request for the cache table.

The decision rule that matters is not "is Postgres faster". It is "does the added p99 from disk reads exceed the p99 of the network round trip to the existing cache". On a same-host or same-cluster deployment the network hop is often sub-millisecond, which sets a low bar for the in-database version to clear.

## Event bus: what you gain and what you lose

### The mechanism

Table-backed queues generally work by inserting a row, marking rows as invisible for a visibility timeout when a consumer reads them, and either deleting or archiving the row when processing succeeds. The exact function names differ between implementations, but the shape is consistent.

```sql
CREATE TABLE outbound_events (
  id           bigserial PRIMARY KEY,
  topic        text NOT NULL,
  payload      jsonb NOT NULL,
  visible_at   timestamptz NOT NULL DEFAULT now(),
  attempts     int NOT NULL DEFAULT 0,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX outbound_events_ready_idx
  ON outbound_events (topic, visible_at)
  WHERE attempts < 5;
```

Consumers claim work with a locking read that skips rows already held by another transaction:

```sql
WITH claimed AS (
  SELECT id FROM outbound_events
  WHERE topic = $1 AND visible_at <= now() AND attempts < 5
  ORDER BY id
  FOR UPDATE SKIP LOCKED
  LIMIT 10
)
UPDATE outbound_events e
SET visible_at = now() + interval '30 seconds',
    attempts = attempts + 1
FROM claimed
WHERE e.id = claimed.id
RETURNING e.id, e.payload;
```

This is the core primitive. Everything else — retry backoff, dead-letter handling, metrics — is application code you now own.

### What this costs you

The queue and the business data share one resource. A long-running analytics query that holds snapshots for minutes will block vacuum on the queue table, and a queue table that is never vacuumed accumulates bloat until the index scan that claims work becomes the slowest query in the system. This is a real and common failure mode, and it does not exist when the queue lives in a separate system with its own storage.

Ordering is the second cost. `FOR UPDATE SKIP LOCKED` gives you no ordering guarantee across concurrent consumers. If two workers claim rows 5 and 6, row 6 may commit first. If your workflow requires that events for a given entity be processed in order, you must add a per-entity lock or a single-consumer-per-key scheme, and both reduce throughput.

Replay is the third. Kafka retains a log you can rewind. A table-backed queue that deletes processed rows cannot be replayed. If you archive instead of delete, you can replay, but the archive table grows without bound unless you partition it and drop old partitions on a schedule.

The failure mode to watch for: consumer lag grows during a traffic spike, the queue table bloats, autovacuum cannot keep up, and claim queries slow down, which reduces consumer throughput further. Instrument the claim query's duration and the table's dead tuple count together; a rising claim duration with a rising dead tuple count is this failure in progress.

### How to measure whether it is worth it

1. Instrument the current broker's publish-to-consume latency at p50 and p99, and the consumer lag in messages.
2. Instrument the claim query duration and the queue table's dead tuple ratio.
3. Run both paths concurrently with a traffic split, and compare the tail latency of the end-to-end workflow, not just the enqueue step.
4. Watch dead tuple count on the queue table during the test. If it climbs and does not recover between load pulses, the queue is not keeping up with its own garbage.

The decision rule: if your workflow tolerates out-of-order processing and at-least-once delivery, the table-backed queue is likely adequate. If it needs per-key ordering, replay, or fan-out to independent consumer groups with independent offsets, the broker is doing work you would otherwise have to build.

## Time series: what you gain and what you lose

### The mechanism

Declarative partitioning on a time column, plus a scheduler that pre-creates future partitions and drops expired ones, reproduces the retention half of a time-series database.

```sql
CREATE TABLE api_metrics (
  recorded_at timestamptz NOT NULL,
  service     text NOT NULL,
  route       text NOT NULL,
  latency_ms  numeric NOT NULL
) PARTITION BY RANGE (recorded_at);

CREATE TABLE api_metrics_2026_01
  PARTITION OF api_metrics
  FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

Queries that filter on `recorded_at` benefit from partition pruning, which is the main reason this works at all. A query without a time filter will scan every partition and will be slower than the same query against a single table, because the planner has more work to do.

Retention becomes a `DROP TABLE` on the oldest partition, which is a metadata operation and effectively instant, unlike a bulk `DELETE` that has to write WAL and wait for vacuum.

### What this costs you

Compression. A columnar time-series engine stores each column contiguously and compresses runs of similar values, which is why metrics compress so well. Postgres stores rows, and its built-in compression applies to variable-length values within a row. If your metric rows have many columns and you query only one or two of them, you are paying to read and decompress the others.

Continuous aggregates. A time-series engine can maintain a materialised rollup incrementally as data arrives. In Postgres you build this yourself with a materialised view and a refresh job. A full refresh recomputes the whole window; a concurrent refresh blocks less but still does the full computation. For high-cardinality rollups this becomes the dominant cost.

Cardinality. Partitioning by time does not help when the expensive part of the query is grouping by a high-cardinality label. The planner still has to aggregate across all matching rows in each partition.

The failure mode to watch for: partition count grows faster than expected because the interval is too fine, and planning time — not execution time — becomes the bottleneck. Every query against a partitioned table pays planning cost proportional to the number of partitions. Instrument planning time separately from execution time; if planning dominates on simple queries, widen the partition interval.

### How to measure whether it is worth it

1. Capture the ten most frequent metric queries and their current p99, with their row counts.
2. Recreate the same data in partitioned Postgres tables with the same retention window.
3. Compare p99 execution time, and separately compare planning time.
4. Measure the size on disk of both representations for the same data and retention window.
5. Measure the wall-clock time of the retention operation in both systems — dropping a partition versus whatever the existing system does.

The decision rule: if your metric queries are dominated by time-range filters and simple aggregations, partitioning is likely sufficient. If they depend on incremental rollups over high-cardinality dimensions, or on compression ratios that row storage cannot reach, the specialised engine is earning its keep.

## A worked sizing example

Here is the arithmetic for a hypothetical service, with every assumption stated so you can substitute your own.

Assume an event rate of 2,000 events per second, sustained. Each event row is 400 bytes including the payload and row overhead.

- Rows per day: 2,000 × 86,400 = 172,800,000.
- Bytes per day: 172,800,000 × 400 = 69,120,000,000 bytes ≈ 64 GiB per day before any index overhead.
- With a 30-day retention window: 64 × 30 = 1,920 GiB ≈ 1.9 TiB of table data.

Now add indexes. A primary key on a `bigserial` plus a composite index on `(topic, visible_at)` might add 30 to 50 bytes per row depending on fill factor and key width. Take 40 bytes:

- Index bytes per day: 172,800,000 × 40 = 6,912,000,000 bytes ≈ 6.4 GiB per day.
- Over 30 days: 6.4 × 30 = 192 GiB.

Total steady-state footprint is roughly 1.9 TiB plus 0.19 TiB, or about 2.1 TiB. That is the number you compare against the storage and memory of the broker you would otherwise run, and against the disk size you must provision for Postgres.

The more important number is the write path. At 2,000 rows per second with WAL enabled, you are generating WAL at roughly the row size plus WAL record overhead — call it 500 bytes per row, so about 1 MB per second, or roughly 86 GiB per day of WAL. Your archive strategy and checkpoint tuning have to sustain that, and your replica has to apply it. This is the constraint that usually decides the question, not storage.

Run the same arithmetic with your own event rate, row size and retention window before you commit to anything.

## A decision checklist

Work through these in order. A "no" early on is a strong signal to stop.

1. Does the workload tolerate at-least-once delivery with possible reordering? If not, keep the broker.
2. Is the cache working set comfortably smaller than the memory you can dedicate to Postgres, with headroom for growth? If not, keep the external cache.
3. Are the metric queries dominated by time-range filters over a fixed retention window? If not, keep the time-series engine.
4. Can your team write and review PL/pgSQL, and debug query plans? If not, the consolidation moves work to the people least equipped to absorb it.
5. Can you run both stacks concurrently behind a traffic split for long enough to observe a full traffic cycle, including a peak? If not, you cannot measure the trade-off, and you should not make the change.
6. Do you have monitoring on the specific failure modes named above — cache table hit ratio, queue dead tuple ratio, partition planning time? If not, build that first.

## FAQ

**Does an unlogged table survive a crash?**

No. Unlogged tables are truncated after an unclean shutdown. They are appropriate for data you can rebuild from a source of truth, and inappropriate for anything you would be upset to lose.

**Can a table-backed queue provide exactly-once processing?**

No. You can get at-least-once delivery plus idempotent consumers, which produces the same observable effect if every consumer operation is idempotent. That requires designing idempotency into the consumer, not configuring it in the queue.

**Why is planning time a problem with many partitions?**

The planner considers each partition when building the plan. With thousands of partitions, planning cost grows and can exceed execution time for simple queries. Widening the partition interval reduces the count at the cost of coarser retention granularity.

**Is a partitioned Postgres table as fast as a columnar store for analytics?**

For narrow time-range scans over a few columns, it can be competitive. For wide scans over many columns, or for queries that benefit from per-column compression, row storage reads more data than necessary and the columnar store wins. Measure with your own column count and query shape.

**What is the first thing to instrument before attempting this?**

The tail latency of the specific code path you intend to move, split by hit and miss where applicable. Without a baseline for that path alone, an aggregate system metric will not tell you whether the change helped or hurt.

## Do this in the next 30 minutes

Pick the single highest-traffic code path that currently talks to your external cache, broker or time-series store. Find it in `pg_stat_statements` if it already touches Postgres, or in your tracing if it does not. Write down its current p99 latency, its request rate, and the size of the data it reads. That one row of numbers is the baseline you need before any consolidation decision is worth making.
