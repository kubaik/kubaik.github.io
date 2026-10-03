# Postgres vector search: what breaks first

## What this article covers

Keeping embeddings inside PostgreSQL is an attractive default: one datastore, one backup story, one connection string, and SQL for filtering and joins. That default holds up fine for prototypes and low-traffic semantic search. It stops holding up at a fairly predictable point, and the failure rarely announces itself as "vector search is slow." It shows up as connection pool exhaustion, autovacuum interference, index bloat, or a p99 that drifts upward over weeks.

This article describes the failure modes in the order they typically appear, how to measure each one with tools you already have, and what the decision looks like when you have to choose between staying in Postgres and running a separate vector service. It is written for teams already comfortable with PostgreSQL and now putting real query volume against an HNSW or IVFFlat index.

## The failure modes, in the order they usually appear

### 1. Connection pool exhaustion (before the database is the bottleneck)

The first thing that breaks is usually not the index. It is the pool.

Vector search endpoints tend to be called from application code that opens a connection per request or per worker, and the queries are longer than typical OLTP statements. Under a burst, connections are held longer, the pool saturates, and clients queue. The observable symptom is latency that rises before CPU or IO on the database host looks saturated.

PostgreSQL's `max_connections` default is 100. That number is a hard server-side ceiling, and every connection costs backend process memory. A common failure pattern is an application pool configured with a high `max_client_conn` pointed at a server whose `max_connections` was never raised, so under load the server refuses connections and the client retries with backoff. The retry storm adds latency to every request, including ones that would otherwise have been fast.

What to measure:

- `SELECT count(*), state FROM pg_stat_activity GROUP BY state;` during a load test, not after.
- `SHOW max_connections;` and your pooler's `max_client_conn` / `default_pool_size`.
- Client-side pool wait time. With PgBouncer, `SHOW POOLS;` exposes `maxwait` and `maxwait_us`; a sustained nonzero `maxwait` means requests are queuing for a connection.
- Connection churn: `SELECT sum(numbackends) FROM pg_stat_database;` sampled over time, plus the `tcp_established` count on the host.

If pool wait time is nonzero during your peak, fix that before you benchmark the index. Otherwise every latency number you collect is contaminated by queueing.

### 2. Autovacuum and index maintenance interfering with search

HNSW indexes in pgvector are graph structures stored in the normal PostgreSQL page format. Inserts and updates cause page splits and dead tuples, and the graph gradually becomes less cache-friendly. Two things follow:

- Query latency degrades over time even when query volume is flat.
- Maintenance operations (autovacuum, `REINDEX`) consume CPU and IO that the search path also needs.

A frequently reported pattern is a periodic latency spike correlated with autovacuum running on the table or index. On a busy instance, autovacuum is not something you tune once and forget; it competes with your query workload by design.

What to measure:

- `SELECT relname, n_dead_tup, last_autovacuum, last_autoanalyze FROM pg_stat_user_tables WHERE relname = 'your_table';`
- Index size over time: `SELECT pg_size_pretty(pg_relation_size('your_index'));` sampled daily. Steady growth with flat row count indicates bloat.
- Latency histogram sliced by time-of-day, so you can correlate spikes with vacuum windows.
- `log_autovacuum_min_duration = 0` temporarily, to see exactly when vacuum runs and how long it takes.

If your p99 spikes line up with vacuum, the fix is either tuning autovacuum aggressively for that table (lower `autovacuum_vacuum_scale_factor`, raise `autovacuum_vacuum_cost_limit`) or moving the workload off the instance. Both are legitimate; only one of them is cheap.

### 3. Recall and latency trade-offs under filtering

Vector search combined with a `WHERE` clause is where naive implementations fall over. There are two broad strategies and they have opposite failure modes:

- **Post-filter:** run the ANN search for `top_k`, then discard rows that fail the filter. If the filter is selective, you may discard most of the results and return fewer than `top_k` rows. Raising `top_k` to compensate increases latency roughly linearly.
- **Pre-filter:** restrict the candidate set first, then search. If the filter is selective, the ANN index may not be usable and the planner falls back to a sequential scan over the filtered rows, which is exact but slow.

Neither is wrong. The problem is that the query planner's choice can change as statistics drift, so a query that was fast last month becomes slow this month without any code change.

What to measure:

- `EXPLAIN (ANALYZE, BUFFERS)` on the actual production query shape, with realistic filter values, not on a `SELECT *` with no filter.
- Recall against a brute-force ground truth. For a sample of queries, compute exact nearest neighbors with a sequential scan and compare to the ANN result. Track recall@k over time; a falling number means the index is degrading or the data distribution has shifted.
- Latency as a function of filter selectivity. Run the same query with filters that match 1%, 10%, and 50% of rows.

### 4. Write amplification and re-index cost

Every insert into an HNSW index does graph maintenance work. For a write-heavy table, this can dominate. The practical consequence is that bulk loads and model migrations become expensive operations that need their own maintenance window, and the cost scales with collection size.

What to measure:

- Insert throughput with the index present versus dropped, on the same hardware, with the same batch size.
- Time to build the index from scratch on your actual data volume. This is the number that determines your migration window.
- Time for `REINDEX INDEX CONCURRENTLY`, which does not block writes but does consume significant IO and takes longer than the blocking version.

The build time on your data is the only number that matters here. It depends on dimension count, row count, and the `m` and `ef_construction` parameters you chose, so it cannot be borrowed from someone else's benchmark.

## How to benchmark this yourself

A benchmark you did not run is not evidence. The following procedure produces numbers that are defensible for your workload.

**Define the workload.** Fix the vector count, the dimension, the query rate, the burst profile, and the filter selectivity distribution. A burst profile matters more than a steady-state number, because that is where pools and vacuum interact.

**Set up ground truth.** For a sample of at least a few hundred queries, compute exact nearest neighbors with a sequential scan (`SET enable_indexscan = off;` for the query, or a separate table without the index). Store the true top-k. This gives you recall.

**Instrument four things, not one:**

- Latency percentiles (p50, p95, p99), recorded server-side and client-side. The gap between them is your queueing and network cost.
- Recall@k against the ground truth sample.
- CPU utilization and iowait on the database host, sampled at least every 15 seconds.
- Pool wait time and connection count.

A minimal Prometheus setup for the host metrics:

```promql
# CPU utilization (1 = fully busy)
1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

# Memory actually available
node_memory_MemAvailable_bytes

# Disk read latency, if you are IO-bound during index build
rate(node_disk_read_time_seconds_total[5m]) / rate(node_disk_reads_completed_total[5m])
```

A minimal load generator. The point is to reproduce your burst shape, not to hit a headline QPS number:

```python
import time
import random
from concurrent.futures import ThreadPoolExecutor

import httpx

DIM = 1024
TOP_K = 10

def one_query(client: httpx.Client) -> float:
    payload = {
        "query": [random.random() for _ in range(DIM)],
        "top_k": TOP_K,
        "filter": {},
    }
    start = time.perf_counter()
    client.post("http://localhost:8000/search", json=payload, timeout=30.0)
    return (time.perf_counter() - start) * 1000.0

def run(concurrency: int, duration_s: int) -> None:
    latencies: list[float] = []
    deadline = time.time() + duration_s
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while time.time() < deadline:
                futures = [pool.submit(one_query, client) for _ in range(concurrency)]
                latencies.extend(f.result() for f in futures)
    latencies.sort()
    n = len(latencies)
    print(f"n={n} p50={latencies[n // 2]:.1f}ms "
          f"p95={latencies[int(n * 0.95)]:.1f}ms "
          f"p99={latencies[int(n * 0.99)]:.1f}ms")

if __name__ == "__main__":
    run(concurrency=50, duration_s=300)
```

Ramp concurrency in steps and record the percentile at each step. The knee of that curve, where p99 starts rising faster than p50, is your practical capacity. That number is far more useful than a single peak-throughput figure.

## A worked example: deciding whether to move

Suppose a service handles 40,000 vector searches per day, averaging 0.5 queries/second, with a daily 15-minute burst at 10× average (5 queries/second). Vectors are 1,024-dimensional float32, 1.5 million rows.

Storage for the raw vectors alone: 1.5M × 1,024 × 4 bytes = 6.1 GB, before index overhead. An HNSW index typically adds a substantial multiple of that, so plan for the index and the table to both be resident in memory for predictable latency, or accept disk reads.

At 5 queries/second peak, with each query taking even 200 ms of database CPU, the database needs about 1 core dedicated to vector search at peak. That is comfortably within a small instance. At 50 queries/second peak, the same query cost needs about 10 cores, which is no longer a small instance and starts to compete with whatever else the database does.

This arithmetic is illustrative, but the method is not: (peak QPS) × (CPU seconds per query) = cores required at peak. Measure the second factor on your own hardware with `EXPLAIN (ANALYZE)` and `pg_stat_statements`, then decide. The crossover point where a dedicated service becomes cheaper than a larger database instance is a function of that product, not of a rule of thumb.

## When a dedicated vector service earns its cost

A separate service is justified when at least one of these is true:

- **Vector search is on the critical path with a strict p99 SLO** and you cannot afford latency spikes from vacuum or index maintenance.
- **Query volume is high enough** that the cores required for ANN search crowd out transactional work on the same instance.
- **You need index types or quantization** that pgvector does not offer, and the recall/latency trade-off matters for your product.
- **You need independent scaling** of the search tier, so a traffic spike in search does not degrade writes.

A separate service is not justified merely because it benchmarks faster on a synthetic workload. It adds a second datastore, a second backup and restore procedure, a second upgrade cadence, and a consistency problem: keeping the vector store in sync with the source of truth in Postgres. That last item is the one teams underestimate.

## The consistency problem nobody benchmarks

If vectors live in a separate service, Postgres remains the source of truth for the underlying rows, and the vector store is a derived index. That means you need a propagation mechanism: outbox table plus a worker, logical replication, or a change-data-capture pipeline. Each has a failure mode.

- **Outbox plus polling worker:** simple and debuggable, but adds latency between a row changing and its embedding being searchable. Under a backlog, search results are stale.
- **Logical replication:** lower latency, but schema changes and replication slot management become operational concerns, and a dropped slot can cause unbounded WAL growth.
- **CDC pipeline:** lowest latency, but the most moving parts.

Whichever you choose, you need a reconciliation job that periodically compares row counts and, ideally, a checksum of the embedding column against the vector store. Without it, silent divergence accumulates and you find out when search results stop matching reality.

The migration itself follows a standard pattern:

1. Stand up the new service alongside Postgres.
2. Backfill in batches, recording the highest primary key or `updated_at` processed per batch.
3. Start the change stream from that watermark.
4. Run both paths in parallel, comparing result sets for a sample of queries.
5. Shift read traffic gradually, watching recall and latency.
6. Keep the old path available for rollback until the new one has run through a full maintenance cycle.

Step 4 is the one people skip. Comparing result sets catches both propagation bugs and recall differences between the two index implementations, which are real and often larger than expected.

## Decision checklist

Work through this before committing either way.

1. What is your peak QPS, and what is the CPU cost per query on your hardware? Multiply them. Does the result fit on your current instance with headroom?
2. What is the p99 SLO, and does it survive a vacuum running concurrently with peak traffic? Test it, do not assume.
3. What is your recall@10 today, measured against brute force? Is it acceptable, and do you know how it changes as the collection grows?
4. How long does a full index build take on your data? Is that a window you can schedule?
5. If you move to a separate service, what is the propagation mechanism, and what is the maximum staleness it permits?
6. Who owns the reconciliation job, and what alerts when the vector store diverges from Postgres?
7. What is the rollback plan if the new service has a bad week?

If you cannot answer 5 through 7, the operational cost of moving is higher than the latency you are trying to fix, and you should spend another cycle on tuning and measurement first.

## FAQ

**Does pgvector get slower over time even with an HNSW index?**
Yes, it can. Inserts and updates fragment the graph, and query latency tends to drift upward as the index grows and becomes less cache-friendly. The mitigation is monitoring index size and recall over time and rebuilding when they degrade, not assuming the index is static.

**How do I know if the connection pool is the bottleneck rather than the query?**
Compare pool wait time against query execution time. If `SHOW POOLS;` in PgBouncer reports a sustained nonzero `maxwait`, or if client-side latency exceeds server-side `EXPLAIN (ANALYZE)` execution time by a wide margin, requests are waiting for a connection. Fix that before tuning the index.

**Should I use `REINDEX INDEX CONCURRENTLY`?**
It avoids blocking writes, which matters for a live service, but it takes longer and does more total IO than the blocking form. Schedule it in a low-traffic window and monitor disk saturation while it runs.

**Is IVFFlat or HNSW better?**
HNSW generally gives better recall at a given latency and supports incremental inserts more gracefully. IVFFlat builds faster and uses less memory, but recall depends heavily on the list count and it needs a representative sample to train. The right choice depends on whether your data is write-heavy and how much memory you can afford.

**Can I keep vectors in Postgres but move only the search?**
Yes. A common pattern is to keep the source of truth and the embeddings in Postgres and maintain a derived ANN index in a separate service, populated by an outbox worker. This keeps transactional consistency where it belongs and isolates the search tier for scaling.

## Do this in the next 30 minutes

Run this against your production database and look at the last two columns:

```sql
SELECT
  relname,
  n_live_tup,
  n_dead_tup,
  last_autovacuum,
  pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_stat_user_tables
WHERE relname IN ('your_vectors_table')
ORDER BY n_dead_tup DESC;
```

If `n_dead_tup` is a significant fraction of `n_live_tup`, or if `last_autovacuum` is more than a day old on a table that receives writes, your latency spikes are probably maintenance, not the index. That is a tuning problem, and it is cheaper to fix than a migration.
