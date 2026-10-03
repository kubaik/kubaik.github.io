# High-Volume Log Pipelines: Shipping 50k Events Per Second

At high event rates, log pipelines stop being an observability detail and become a distributed systems problem. The failure is rarely a single dramatic crash. It is a slow accumulation of merge pressure, shard imbalance, buffer exhaustion, and query latency that turns a dashboard refresh into an outage.

This article compares two broad architectures for shipping and querying tens of thousands of events per second (EPS), explains where each one breaks, and gives you the instrumentation to measure your own pipeline rather than trust someone else's numbers. All figures below are either documented defaults, arithmetic shown step by step from stated assumptions, or explicitly labelled illustrative.

## The two architectures

**Architecture A: collector + forwarder + columnar SQL store.** A vendor-neutral collector receives logs over OTLP, a lightweight host-level forwarder tails files and ships them, and a columnar analytical database stores and queries them. This is the "logs are a table" model: you write SQL, you control partitioning and sorting keys, and you own the merge behavior.

**Architecture B: shipper + search cluster + visualization layer.** An agent tails files and bulk-indexes into a distributed search engine with an inverted index, and a dashboard tool queries it. This is the "logs are documents" model: you get relevance scoring and a mature query UI, but you pay for refresh cycles, JVM heap management, and shard routing.

Neither is universally better. The decision hinges on four things: cardinality of your filter dimensions, your latency SLA, your team size, and whether full-text search is a first-class requirement.

## Constraint 1: cardinality

High-cardinality fields — `tenant_id`, `user_id`, `deployment_id`, `request_id` — are where the two models diverge most sharply.

In a columnar store, a skip index on a low-cardinality or well-ordered column lets the engine eliminate granules before reading them. A query like:

```sql
SELECT count(*)
FROM logs
WHERE tenant_id = 'acme'
  AND deployment_id = 'api-prod'
  AND region = 'eu-central-1'
  AND timestamp > now() - INTERVAL 5 MINUTE
```

can be answered by reading only the granules whose min/max metadata overlaps the predicate. The cost scales with the number of matching granules, not the total table size.

In an inverted-index search engine, the same query hits the term dictionary for each filter and intersects posting lists. That is fast when the terms are rare and the index is warm. It degrades when the filter matches a large fraction of documents, when the field is mapped as `keyword` with a large cardinality, or when the index is still refreshing. A `refresh_interval` of 1s (the default) means recently indexed documents are not yet searchable, and a forced refresh is expensive.

**How to measure it:** build a query that filters on your highest-cardinality dimension and run it against both backends with the same data volume. Instrument:

- The query planner output (in a columnar store, `EXPLAIN` and the `system.query_log` table; in a search engine, the profile API).
- Wall-clock p50 and p99 over at least 100 runs, not one.
- The number of rows/granules/documents scanned, not just the time.

If the search engine's profile shows it is visiting a large fraction of the index, cardinality is your bottleneck.

## Constraint 2: latency budget

End-to-end latency is the sum of every hop:

```
app write → forwarder read → collector receive → batch → store commit → query visible
```

Each hop has a configurable timeout, and each timeout is a latency floor. A batch timeout of 1s means a low-volume stream can sit for up to 1s before flushing. A search engine's refresh interval of 1s means a document is not queryable for up to 1s after it is indexed.

**How to measure it:** embed a monotonic timestamp at the application, carry it through as a field, and at query time compute `now() - app_timestamp`. This measures true end-to-end latency including queueing, which no component-level metric captures. Do this at p50, p95, and p99; the tail is where pipelines fail.

A useful arithmetic exercise: if your batch timeout is 1s and your refresh interval is 1s, your worst-case visibility latency is already ~2s before any queueing. Tightening either requires trading throughput for latency.

## Constraint 3: team size and operational surface

The operational surface of a pipeline is roughly the number of independently tunable components times the number of failure modes each one has.

Architecture A typically has three moving parts: the forwarder (stateless, restartable), the collector (stateless, horizontally scalable), and the store (stateful, needs capacity planning for merges and disk). The store is the only component that requires deep expertise.

Architecture B typically has three moving parts too, but the search cluster is a JVM application with heap sizing, garbage collection tuning, shard allocation awareness, and index lifecycle management. The visualization layer adds a reverse proxy and role-based access control. The failure modes are more numerous.

**A decision heuristic:** if your observability team is one or two engineers, minimize the number of components that require JVM-level tuning. If you already employ search engineers, that constraint does not apply.

## Constraint 4: schema evolution

Log schemas change. New fields appear weekly in active services.

In a columnar store, adding a column is a metadata operation:

```sql
ALTER TABLE logs ADD COLUMN IF NOT EXISTS user_agent LowCardinality(String);
```

This is near-instant because existing data files simply lack the column and read as default values. `LowCardinality(String)` is worth using for fields with fewer than roughly 10,000 distinct values; it stores a dictionary plus integer references instead of repeated strings.

In an inverted-index search engine, adding a field to an existing index requires either a reindex or a runtime field. A reindex copies and rebuilds the entire index, which is proportional to data volume and competes with live indexing for I/O.

**How to measure it:** time an `ALTER TABLE ... ADD COLUMN` on a representative table, and time a reindex of a representative index. The ratio is usually the deciding factor for teams that change schemas often.

## Storage and compression

Columnar stores compress well because values in a column share type and often share distribution. Compression codecs like ZSTD with a moderate level are common; the exact ratio depends entirely on your data.

**How to measure it:** write a representative day of logs, then compare `sum(bytes_on_disk)` from the store's system tables against the raw input size. Do not trust a ratio from someone else's dataset — log compression is highly sensitive to field repetition, timestamp regularity, and the number of distinct string values.

For search engines, compression is applied per segment and interacts with merge behavior. More aggressive compression means slower merges and higher CPU.

## Failure modes worth designing for

**Merge storms.** In any LSM-style or columnar store, background merges compete with ingestion and queries for CPU and I/O. If partitions are too large or the merge policy is too aggressive, merges can saturate the disk. Mitigation: partition by day, keep active partitions bounded, and monitor merge queue depth as a first-class metric.

**Buffer exhaustion.** A forwarder or collector with a memory-only buffer will drop or block when the downstream stalls. A filesystem buffer survives restarts but can fill the disk. The correct choice depends on whether you prefer backpressure (block the app) or data loss (drop oldest). Document the choice explicitly.

**Retry without backoff.** A forwarder that retries immediately on failure turns a brief downstream hiccup into a retry storm. Verify that your forwarder's retry policy includes exponential backoff and a maximum retry count, or implement it in a filter.

**Shard imbalance.** In a search cluster, uneven shard distribution after a node restart concentrates load on a few nodes. Monitor per-node shard counts, not just cluster health, which can report green while individual nodes are overloaded.

**Query-triggered write blocking.** A high-cardinality ad-hoc query can consume enough resources to delay indexing. This is the classic "one dashboard took down the cluster" failure. Mitigation: query isolation, separate read replicas, or a store whose query engine does not compete with ingestion for the same resources.

## A worked cost example (illustrative)

Cost comparisons are only meaningful with stated assumptions. Here is one, clearly labelled illustrative.

Assume 50,000 EPS, 1 KB per event, 30 days of retention, and a 4:1 compression ratio.

- Raw volume: 50,000 × 1,000 bytes = 50 MB/s.
- Per day: 50 MB/s × 86,400 s = 4,320,000 MB ≈ 4.3 TB/day.
- Over 30 days: ~130 TB raw.
- Compressed at 4:1: ~32 TB on disk.

At a hypothetical $0.08 per GB-month for block storage, 32 TB ≈ 32,768 GB × $0.08 ≈ $2,621/month in storage alone. Compute cost depends on the number of nodes required to sustain ingestion and merges, which you must measure on your own hardware.

The point of this exercise is not the number. It is that storage dominates at this scale, so compression ratio and retention policy matter more than per-node instance pricing. If your compression ratio is 2:1 instead of 4:1, you double the storage bill.

**How to measure your own ratio:** ingest one representative day, then query the store's system tables for bytes on disk and divide by the raw byte count of the input files.

## Instrumentation checklist

Before comparing stacks, instrument both with the same signals:

- **Ingestion rate:** records accepted per second at the collector and at the store.
- **Drop rate:** records rejected or dropped, by reason. This should be zero in steady state.
- **Queue depth:** buffer occupancy at every hop. A rising queue depth is the earliest warning of a stall.
- **End-to-end latency:** from an application-embedded timestamp, at p50/p95/p99.
- **Query latency:** by query shape, not just overall. A single slow query class can dominate.
- **Merge or compaction pressure:** queue depth and CPU time spent on background work.
- **Disk I/O wait:** the most common hidden bottleneck at high EPS.

If you cannot produce these seven numbers for your current pipeline, you are not ready to compare it against an alternative.

## Decision checklist

Lean toward a columnar SQL store when:

1. You filter on high-cardinality dimensions and need sub-second tail latency.
2. Your observability team is small and you want to avoid JVM tuning.
3. Schema changes are frequent and reindexing is unacceptable.
4. You are comfortable writing SQL and do not need relevance scoring.

Lean toward a search cluster when:

1. Full-text search on unstructured logs is a first-class requirement.
2. Your team is already trained on the query language and dashboards.
3. You need the ecosystem's mature visualization and alerting tooling.
4. Your retention window is short and merge overhead is not a concern.

## Common follow-up questions

**Can I run both?** Yes, but the query languages and data models differ enough that you will maintain two mental models. Teams that bolt a columnar store onto an existing search cluster often regret it because materialized views and saved searches do not map cleanly between them.

**Does the collector add latency?** It adds one hop. Measure it with the end-to-end timestamp method above; it is usually single-digit milliseconds on the same node.

**Is a filesystem buffer always better than a memory buffer?** No. A filesystem buffer survives restarts but can fill the disk and cause write failures. A memory buffer is faster but loses data on restart. Choose based on whether you prefer backpressure or data loss.

**How do I size partitions?** Start with daily partitions and keep the active partition small enough that merges complete within your lowest-traffic window. Measure merge duration, do not guess.

## Action for the next 30 minutes

Pick your highest-cardinality log field and run one query filtering on it against your current backend. Capture the query profile and the p99 wall-clock time over 100 runs. Then run the same query against a single-node columnar store loaded with one day of the same data. Compare granules or documents scanned, not just elapsed time. That single comparison will tell you more about which architecture fits than any benchmark table.

If the search backend scans a large fraction of the index while the columnar store prunes to a handful of granules, cardinality is your bottleneck and the architecture decision is effectively made.
