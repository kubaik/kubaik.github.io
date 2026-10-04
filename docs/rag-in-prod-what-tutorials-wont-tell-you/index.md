# RAG in prod: what tutorials won’t tell you

Most RAG pipeline guides assume a clean environment and a patient timeline. Production gives you neither. The interesting failures in a retrieval-augmented generation feature rarely come from the model or the prompt. They come from the retrieval layer's interaction with write concurrency, memory limits, and data freshness.

This article walks through a common failure mode for a B2B SaaS copilot — a feature that answers questions like *"Why did my conversion drop 20% last week?"* using a customer's historical order data — and the two-tier retrieval design that resolves it. The scenario is representative rather than reported: the numbers below are labelled either as documented defaults, as arithmetic from stated assumptions, or as illustrative.

## The situation

A typical stack for this kind of build:

- Python 3.11
- A RAG orchestration framework (LangChain is common; version pinning matters less than knowing which components it lazy-loads)
- PostgreSQL 15 with pgvector for vector storage
- A hosted embedding API, or a local sentence-transformers model
- FastAPI for the HTTP layer, behind an ASGI server with multiple workers
- A change-data-capture (CDC) runner that refreshes embeddings nightly

A lean illustrative deployment might be two ARM-based application instances, one memory-optimised instance for PostgreSQL/pgvector, and one small general-purpose instance for the CDC runner.

Typical targets for this kind of feature:

- **P99 latency under 800ms** for retrieval plus generation
- **Cost under $0.002 per query** at 100k daily active users
- **No manual retraining** — embeddings refresh nightly from CDC

That setup looks ready. Then production traffic arrives, and the failure mode is almost always the same shape.

**The failure mode:** a nightly CDC job that refreshes embeddings takes hours to process a large table. During that window, the vector index is effectively unavailable for reads at normal latency, and retrieval latency spikes into the seconds. Users get timeouts. The connection-pool errors that dominate the debugging session are usually a symptom — the real cause is write contention on the index, hidden behind a timeout that looks like a pool exhaustion problem.

The diagnostic question is not "is my index fast?" It is "what happens to read latency while the index is being written?"

## Why the obvious fixes don't work

### Attempt 1: more PostgreSQL resources

Scaling the database instance up and raising `shared_buffers` is the first move most teams make. Latency improves modestly, but the index lock still happens during CDC, and the nightly job still takes the same wall-clock time. The bottleneck is not buffer cache; it is serialised writes to the index.

Parallelising the CDC job with more workers does not help either, because pgvector does not support concurrent writes to the same index. Inserts serialise regardless of how many workers you throw at them. This is documented behaviour rather than a bug, but it is easy to miss when the tutorial only shows a single-writer example.

**How to verify this for your own workload:** instrument the CDC job to log `pg_stat_activity.wait_event` for the embedding writer session, and log read latency percentiles from the application side on the same timeline. If read p99 spikes correlate with write-wait events, you have confirmed index-level write contention rather than a query-plan problem.

### Attempt 2: an in-memory index behind a cache

Moving the vector index into an in-memory library loaded into a cache cluster is the next common step. Pre-computing embeddings nightly and loading them into memory at process startup drops latency dramatically. Two costs appear immediately:

- **RAM.** The index must fit in memory on every node that serves it, and vector indexes are memory-hungry. Sizing is arithmetic: `num_vectors × dimensions × bytes_per_float`. For 2 million vectors at 1024 dimensions in float32, that is `2,000,000 × 1024 × 4 = 8.19 GB` before index overhead. HNSW graph structure adds more on top.
- **Invalidation.** When a customer updates their product catalog, the in-memory embeddings are stale until the next reload. There is no incremental update path unless you build one.

### Attempt 3: a dedicated vector database

A dedicated vector database promises concurrent writes and built-in batching, and usually delivers on both. The trade-offs that show up in practice:

- **CPU cost of the graph index.** HNSW search is more CPU-intensive than a flat or IVF index at the same recall, so latency per query can be higher than the in-memory alternative unless you tune `efSearch` down.
- **Operational maturity.** Memory behaviour under sustained load varies by implementation. Any service that holds a large graph in RAM needs a memory ceiling and a restart strategy, because an OOM kill during a query burst produces exactly the latency spikes you were trying to avoid.
- **Payload limits.** Vector databases impose per-point payload size caps. If your embeddings are built from joined order-plus-product-metadata text, long documents can exceed the cap, and truncation silently degrades retrieval quality.

The common resolution is to roll back to PostgreSQL/pgvector and conclude that the architecture, not the database, is wrong.

## The approach that works: two-tier retrieval

The fix is not optimising the vector index. It is treating embeddings as **ephemeral, disposable assets** and splitting retrieval by data temperature.

1. **Hot tier** — an in-memory index covering recent data (for example, the last 7 days), which is what most queries touch.
2. **Cold tier** — PostgreSQL/pgvector for older data, queried infrequently.

This split addresses three problems at once:

- **Concurrency.** The in-memory index handles its own writes; PostgreSQL handles bulk writes on a schedule that does not overlap peak read traffic.
- **Latency.** The hot tier serves the majority of queries without touching the database.
- **Cost.** The cache cluster's RAM requirement drops because it no longer holds the full index, only the hot slice plus cached answers.

### Routing queries

A simple heuristic is enough to start: if the query references a date within the hot window, route to the hot tier; otherwise route to the cold tier.

```python
from datetime import datetime, timedelta

def route_query(query: str) -> str:
    date_threshold = datetime.now() - timedelta(days=7)
    for token in query.split():
        try:
            date = datetime.strptime(token, "%Y-%m-%d")
            if date >= date_threshold:
                return "hot"
        except ValueError:
            continue
    return "cold"
```

This is deliberately naive. It only recognises ISO-formatted dates, and it fails on relative expressions like "last week". Before shipping it, log the routing decision alongside the query and review the misroutes weekly. A better long-term approach is to have the generation model emit a structured time filter as part of the query plan, then route on that.

The hot tier runs as a separate service instance with the in-memory index. A cache (Redis is the common choice) holds generated answers with a short TTL to avoid regenerating identical responses.

The cold tier stays in PostgreSQL/pgvector, but with **partial indexing** to shrink the nightly CDC window. Instead of rebuilding the entire vector index, only vectors for rows changed in the last 24 hours are updated.

## Implementation details

### Hot tier: in-memory index

Using `faiss-cpu` with AVX2 support, the index is built with an HNSW graph:

```python
import faiss

# Dimension after the embedding model
D = 1024
index = faiss.IndexHNSWFlat(D, 32)  # 32 is the M parameter for HNSW

# Add vectors in batches
batch_size = 10_000
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.add(batch)
```

Wrapping the index in a FastAPI service, with worker count matched to available cores, is standard. A cache client serves as the shared answer cache:

```python
import redis.asyncio as redis

r = redis.Redis(host="redis-hot-tier", port=6379, decode_responses=True)

async def generate_answer(query: str, store_id: str) -> str:
    cache_key = f"answer:{store_id}:{hash(query)}"
    cached = await r.get(cache_key)
    if cached:
        return cached

    # ... RAG logic here ...
    answer = rag_pipeline(query, store_id)

    await r.set(cache_key, answer, ex=300)  # 5 minutes TTL
    return answer
```

Two operational requirements are easy to skip and expensive to skip:

- **Cap index memory.** Monitor process RSS and set a hard ceiling at roughly 80% of the node's RAM. When the ceiling is reached, rebuild the index on a fresh process and swap traffic over, rather than growing in place.
- **Pre-warm after rebuild.** An HNSW index that has just been loaded has cold caches. A periodic dummy query against the index keeps latency stable. Without it, the first real query after a rebuild pays a visible penalty.

**How to measure the warm-up cost:** time a query immediately after index load, then time the same query again after 10, 100, and 1000 warm-up queries. Plot p50 against warm-up count. If the curve flattens after a few hundred queries, schedule that many synthetic queries before the process accepts traffic.

### Cold tier: PostgreSQL/pgvector with partial updates

Add a recency flag to the orders table and maintain it with a trigger:

```sql
CREATE OR REPLACE FUNCTION mark_recent_orders()
RETURNS TRIGGER AS $$
BEGIN
    NEW.is_recent = (NEW.updated_at >= NOW() - INTERVAL '24 hours');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER mark_recent_trigger
BEFORE UPDATE OR INSERT ON orders
FOR EACH ROW EXECUTE FUNCTION mark_recent_orders();
```

Note that a `BEFORE` trigger only fires on the row being written. Rows that were recent yesterday but are not recent today will not be re-flagged by this trigger alone. In practice you need either a scheduled job that clears stale flags, or a `WHERE updated_at >= NOW() - INTERVAL '24 hours'` predicate in the CDC query itself, which is simpler and avoids the trigger entirely for this purpose.

The nightly CDC job then only rebuilds vectors for recently changed rows:

```python
# Pseudocode for the CDC job
changed_orders = db.query("""
    SELECT id, order_data
    FROM orders
    WHERE updated_at >= NOW() - INTERVAL '24 hours'
""")

vectors = embedder.encode([order.order_data for order in changed_orders])

# Only update pgvector for these rows
for order_id, vector in zip([o.id for o in changed_orders], vectors):
    db.execute(
        "UPDATE order_vectors SET embedding = %s WHERE order_id = %s",
        vector, order_id
    )
```

The size of the win depends entirely on what fraction of rows change daily. If 5% of a 2-million-row table changes per night, the CDC job processes 100,000 rows instead of 2,000,000 — a 20x reduction in work. If 80% changes, the partial update buys you almost nothing and you should reconsider whether the cold tier is worth maintaining.

## Monitoring: what to instrument

The metrics that matter for this architecture, and how to read them:

| Metric | Where it comes from | What a rising value means |
|---|---|---|
| Hot tier query latency (p99) | Application histogram | Graph index degrading under concurrency, or memory pressure |
| Index write wait time | `pg_stat_activity.wait_event` on the writer session | CDC is contending with reads; move the window or shrink the batch |
| Cold tier lock duration | `pg_locks` | Partial updates are not partial enough |
| Answer cache hit rate | Cache server `INFO` stats | TTL too short, or query diversity higher than assumed |
| Process RSS vs. ceiling | `psutil` or the container runtime | Index rebuild is due |

Alerts belong on the first three. Cache hit rate is a tuning signal, not a page.

## A worked sizing example

Suppose the copilot serves 100k queries per day, and 80% of them reference data inside the 7-day hot window. That is 80,000 hot queries per day, or roughly 0.93 queries per second on average. Assume a 5x peak-to-average ratio, giving about 4.6 queries per second at peak.

If the hot tier holds 7 days of order data for all customers, and that comes to 500,000 vectors at 1024 dimensions in float32:

`500,000 × 1024 × 4 bytes = 2.05 GB` of raw vector data, plus HNSW graph overhead. Budget 1.5x to 2x for the graph, so 3-4 GB per serving instance. That fits comfortably on a memory-optimised instance and leaves headroom for the answer cache.

The cold tier holds the remaining 1.5 million vectors. At the same dimensions that is `1,500,000 × 1024 × 4 = 6.14 GB` of vector data in PostgreSQL. That is a real storage cost, and it is the reason the cold tier should only be queried when routing says the hot tier cannot answer.

These figures are illustrative. Substitute your own vector count, dimensions, and peak ratio; the arithmetic is the same.

## Common mistakes

1. **Optimising the index before measuring the write path.** The bottleneck is usually the CDC job's interaction with reads, not the search algorithm.
2. **Assuming the cache cluster can hold the whole index.** Sizing is arithmetic, and it is usually larger than expected.
3. **Using a `BEFORE` trigger for recency without a cleanup path.** Stale flags accumulate silently.
4. **Skipping the warm-up after a rebuild.** The cold-start penalty is real and shows up as a p99 spike, not a p50 one.
5. **Trusting default HNSW parameters.** The defaults are tuned for general use, not for your recall/latency trade-off. Measure recall against a labelled set before and after any parameter change.
6. **Letting an orchestration framework hide the retrieval path.** Framework overhead is measurable; if you cannot see where time goes inside the framework, you cannot tune it.

## FAQ

**How do I know if my RAG pipeline needs a two-tier system?**

Look at the distribution of query dates in your logs. Group queries by the age of the data they reference:

```sql
SELECT date_trunc('day', query_created_at - data_referenced_at) AS age_bucket,
       COUNT(*)
FROM queries
GROUP BY 1
ORDER BY 1;
```

If a large majority of queries reference the last few days, a hot tier is worth building. If the distribution is flat, the two-tier split adds complexity without benefit, and you should optimise the cold path instead.

**What's the worst mistake teams make with an in-memory index?**

Not capping memory usage. An in-memory index will allocate until the process is killed. Set a hard ceiling, monitor RSS, and plan a rolling rebuild rather than growing the index in place.

**How do I handle multilingual embeddings in production?**

Pick a multilingual sentence-transformer model and run it through an ONNX runtime for CPU inference. Verify the language coverage you actually need against the model card before committing — "multilingual" claims vary in how well they cover lower-resource languages. Quantising to FP16 reduces memory at a small, measurable accuracy cost that you should quantify on your own evaluation set rather than assume.

**What's the simplest way to cache RAG answers?**

A key-value cache with a short TTL, keyed on a hash of the query plus the tenant ID. Monitor the hit rate; if it is low, the TTL is too short or the queries are more diverse than assumed. Do not cache across tenants.

**Why might a team move from a hosted embedding API to a local model?**

Cost predictability and rate-limit independence. A hosted API charges per token and can throttle; a local model has a fixed infrastructure cost and no external dependency. The trade-off is operational: you now own model serving, batching, and version management. Measure both against your actual query volume before switching.

**What's the biggest surprise after going live?**

Users ask questions that require joining order data with product metadata, but the embeddings often only cover order text. Validate the embedding input against real user queries before shipping. If the retrieval corpus does not contain the fields the questions reference, no amount of index tuning will help.

## The broader lesson

A RAG pipeline is a distributed system with its own failure modes. The tutorials skip the unglamorous parts: write contention, cache invalidation, partial updates, and memory ceilings. Those are what determine whether the feature holds up under load.

Two principles carry most of the weight. First, treat embeddings as disposable — if an index is stale, rebuild it; if a cache misses too often, adjust the TTL. Second, measure before optimising: the bottleneck is almost never the vector index in isolation, it is the interaction between the write path and the read path.

## Action for the next 30 minutes

Run this against your query logs and look at the shape of the result:

```sql
SELECT date_trunc('day', NOW() - referenced_date) AS age_bucket,
       COUNT(*) AS queries
FROM query_log
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;
```

If the first bucket dominates, you have a case for a hot tier. If the distribution is flat, you do not — and you have saved yourself an architecture you would have had to maintain.
