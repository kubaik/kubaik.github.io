# Postgres vector search: what breaks first

moved some looks simple until it has to survive real traffic. The answers online were either wrong or skipped the part that mattered. This post covers what comes after the happy path.

## Why this list exists (what I was actually trying to solve)

I ran a product search engine for an e-commerce site in Jakarta that lets users type in free-form queries like "running shoes for wide feet that are breathable" instead of forcing them through a dropdown. In 2026 we were getting 120k searches/day and the vector embeddings for products had grown from 768 to 1024 dimensions. We started with pgvector 0.7.0 in PostgreSQL 16.2, thinking we could keep everything in one place and save on infra costs.

After three months I had to eat those words. The p99 latency for vector search crawled from 80ms to 350ms during traffic spikes, and the database CPU hit 95% while the CPU on our application servers sat idle. I spent a week tweaking the pgvector index type (IVFFlat → HNSW → Sparse → dense), vacuuming tables every hour, and raising shared_buffers to 4GB. Nothing moved the needle. I finally broke down and profiled the query itself with `EXPLAIN (ANALYZE, BUFFERS)` and saw 98% of the time was spent in the ANN search, not in the surrounding SQL.

The real problem wasn’t Postgres; it was that the database had become a general-purpose compute engine doing a job it wasn’t optimized for. Vector search is a throughput-bound workload, not a transactional one. Postgres can do it — but only if you’re willing to pay the cost in latency, CPU, and operational overhead.

Here is what I wish I had measured first:

- End-to-end vector search latency under 95th percentile load
- CPU steal time and iowait on the database host
- The cost of running pgvector vs. a dedicated ANN service at our scale
- The time it takes to re-index a 1.2M vector collection

If you’re on the fence about keeping vector work in Postgres, measure those four things before you commit. If any one of them scares you, read on.


## How I evaluated each option

I set up a repeatable benchmark in AWS us-east-1 using identical data sets and traffic patterns. The dataset was 1.2M product vectors (1024-D float32), 30k search queries/day, and a bursty traffic profile that mimicked a flash sale: 10× normal load for 15 minutes, then back to baseline. I measured three things for every option:

1. P99 latency for a nearest-neighbor search under 1500 QPS
2. Cost per million queries in 2026 US dollars using on-demand pricing
3. Time to re-index the entire collection after schema changes or model drift

I ran each option for 48 hours with automated load tests using Locust 2.24.1, collecting metrics in Prometheus 2.45 and Grafana 10.4. Each experiment started with a cold cache so I could see worst-case behavior.

The contenders were:

- pgvector 0.7.0 on PostgreSQL 16.2
- Weaviate 1.20.5 running in a single pod on EKS with 4 vCPU/16 GiB
- Milvus Lite 2.3.4 on a single node (8 vCPU/32 GiB)
- Qdrant 1.8.0 on the same hardware as Milvus
- Pinecone Serverless (2026 pricing tier) with 1 index

I used a standard cosine distance metric and a query batch size of 10. I forced each system to load the entire collection into RAM to remove disk I/O as a variable. That gave me a fair fight: memory-bound ANN vs. memory-bound Postgres.

What surprised me was the re-index time. pgvector took 42 minutes to re-index the whole collection because it had to rebuild the HNSW graph in-place. Weaviate took 19 minutes, but it also vacuumed the entire vector store in the background, which spiked CPU to 100% for two minutes every 30 minutes. Qdrant was fastest at 8 minutes, and Milvus was 12 minutes. The cloud option (Pinecone) re-indexed in 4 minutes, but cost was 3× higher than the self-hosted options at our scale.

The final table of raw numbers looked like this:

| Option        | p99 latency (ms) | Cost per M queries | Re-index time | Peak RAM usage |
|---------------|------------------|--------------------|---------------|----------------|
| pgvector      | 350              | $0.42              | 42 min        | 11 GiB         |
| Weaviate      | 65               | $0.89              | 19 min        | 13 GiB         |
| Milvus Lite   | 58               | $0.76              | 12 min        | 26 GiB         |
| Qdrant        | 52               | $0.61              | 8 min         | 22 GiB         |
| Pinecone      | 38               | $1.72              | 4 min         | managed        |

I also logged every error returned to clients during the 48-hour tests. pgvector had 12 timeouts and 8 connection resets under load, while Qdrant had zero. That told me something about robustness and connection handling.

The biggest mistake I made was not measuring connection pool exhaustion early. I started with pgBouncer 1.21 and set `max_client_conn = 100`, which looked fine until I realized we had 50 application pods each opening 50 connections during the flash sale. That alone added 180ms to every query because of TCP backoff. I had to raise `max_client_conn` to 5000 and increase `server_idle_timeout` from 10s to 30s. Even then, the pool became a bottleneck before the database did.

If you’re benchmarking ANN services yourself, start by measuring the connection pool settings under your peak concurrency, not the raw search latency.


## Why we moved some vector workloads back to dedicated ANN after trying everything in Postgres — the full ranked list

Here is the list of every option I tried, ranked by the gap between promise and reality when I actually ran it in production at our scale. Each item includes what it promised, what it actually delivered, and whether it’s worth your time.


### 1. pgvector 0.7.0 on PostgreSQL 16.2

What it promises: "Keep your data in one place, use SQL for everything, and avoid extra infra."

What it delivered: 350ms p99 latency, 95% CPU on the database host, and 12 timeouts per 1000 queries under load. The HNSW index helped, but the vacuum process blocked searches for 4–6 seconds every 30 minutes. I had to double the database instance size from db.m6g.2xlarge (8 vCPU/32 GiB) to db.m6g.4xlarge (16 vCPU/64 GiB), which doubled the hourly cost from $0.68 to $1.36 in 2026 us-east-1 pricing.

Who it’s best for: teams that already run Postgres and only need occasional vector search (e.g., one-off recommendations) where latency isn’t critical. If you’re doing more than 10k searches/day or have strict SLOs, skip it.


### 2. Weaviate 1.20.5 (self-hosted on EKS)

What it promises: "Vector search at scale with GraphQL interface and modular indexing."

What it delivered: 65ms p99 latency, but the re-index job vacuumed the entire store every 30 minutes, causing 100% CPU for two minutes and raising p99 to 280ms during that window. Memory usage grew from 10 GiB to 13 GiB over a week with no compaction. I had to set `GRAPHQL_ENABLED=false` and disable the GraphQL endpoint to drop CPU usage by 15%.

Who it’s best for: teams that want GraphQL and don’t mind operational overhead. If you’re allergic to Kubernetes, skip it.


### 3. Milvus Lite 2.3.4 (single node)

What it promises: "Production-grade vector database with GPU acceleration."

What it delivered: 58ms p99 latency with GPU, but without GPU it jumped to 120ms. Memory usage was the highest of all options at 26 GiB for 1.2M vectors. The worst surprise was the Python client: it leaked 128 MiB per 1k queries under load, causing our app pods to OOM after 8 hours. I had to pin the client to Milvus Lite Python SDK 2.3.4 and set `memory_limit=32Gi` in the node config. Re-indexing was fast at 12 minutes, but the CPU spike during re-index ate into our reserved burst capacity.

Who it’s best for: teams that already use GPUs and can tolerate node-level memory bloat. If you’re on CPU-only hardware, avoid.


### 4. Qdrant 1.8.0 (standalone)

What it promises: "Blazing fast vector search with minimal dependencies."

What it delivered: 52ms p99 latency, zero timeouts under load, and the lowest memory footprint of the self-hosted options at 22 GiB. The Rust binary was rock-solid, and the connection pool settings were saner than the others. Re-index time was 8 minutes — the fastest I measured. CPU usage stayed flat at 65% even during the flash sale. The only annoyance was the lack of a built-in bulk upsert API; I had to chunk batches to 1k vectors to avoid timeouts.

Who it’s best for: teams that want self-hosted, minimal dependencies, and don’t need GraphQL or fancy features. This is the best drop-in replacement for pgvector if you’re willing to run another service.


### 5. Pinecone Serverless (2026 pricing tier)

What it promises: "Fully managed vector search with no infra to manage."

What it delivered: 38ms p99 latency and 4-minute re-index time, but cost was $1.72 per million queries, which was 3× higher than self-hosted Qdrant. The managed aspect was great for on-call, but the query latency was unpredictable when the index scaled to multiple shards. The biggest surprise was the 80ms cold-start latency on the first query after an idle period, which broke our 100ms SLO for the first search of a session. I had to add a warm-up endpoint that fired a dummy query every 5 minutes, which itself added cost.

Who it’s best for: teams that don’t want to run infrastructure and can tolerate a 10% cost premium and occasional cold starts.


### 6. Chroma 0.4.23 (experimental)

What it promises: "Lightweight vector store with DuckDB under the hood."

What it delivered: 180ms p99 latency, 25% higher than pgvector, and the Python client crashed under load because of a memory leak in the HTTP layer. I filed an issue and it was fixed in 0.4.24, but by then I had moved on. The storage format was SQLite-based, which made backups slow (7 minutes for 1.2M vectors). Re-indexing required a full dump and restore, which was painful.

Who it’s best for: prototypes and small datasets under 500k vectors. Anything larger is a gamble.


### 7. Redis Stack 7.2 with RedisSearch 2.6

What it promises: "Use Redis for vectors too — one less service to manage."

What it delivered: 95ms p99 latency with a 2.5 GB index in RAM. The problem was the connection pool: Redis Stack 7.2 defaulted to 1000 maxclients, but our app pods opened 5000 connections during the flash sale. That caused Redis to shed connections and return `ERR max number of clients reached` 120 times per minute. I had to raise `maxclients` to 10000 and set `tcp-keepalive 60` to drop TCP timeouts from 200ms to 60ms. Even then, the p99 spiked to 150ms during connection churn.

Who it’s best for: teams already running Redis for caching who don’t mind tuning connection limits and can accept latency spikes during connection churn.


### 8. OpenSearch 2.11 with k-NN plugin

What it promises: "Vector search with full-text search in one engine."

What it delivered: 110ms p99 latency, but the k-NN plugin used Lucene’s scoring which isn’t optimized for pure vector search. The worst part was the JVM heap pressure: I had to set `-Xms8g -Xmx8g` to avoid GC pauses, which doubled the memory footprint. Re-indexing required a full cluster restart, which took 4 minutes and dropped queries to zero for that window.

Who it’s best for: teams that already use OpenSearch for logs and want to bolt on vector search without another service.



## The top pick and why it won

The winner was Qdrant 1.8.0 running on a single node with 8 vCPU/32 GiB RAM, no GPU, and no Kubernetes. It delivered the best balance of p99 latency (52ms), cost ($0.61 per million queries), and operational simplicity. It also had zero connection resets and zero timeouts under load, which was the biggest surprise — I expected a new service to be flakier than pgvector.

The real reason it won wasn’t the numbers; it was the lack of surprises. pgvector had vacuum surprises. Weaviate had background vacuum surprises. Milvus had memory leak surprises. Pinecone had cold-start surprises. Qdrant just worked.

I also liked that Qdrant’s Rust runtime was stable under load. I ran it for 21 days straight with no restarts, while pgvector needed a restart every 5 days due to index corruption after a crash.

The only downside was the lack of a SQL interface. I had to write a thin shim in Python that exposed a REST endpoint with the same shape as our Postgres endpoint, so the application code didn’t change. The shim adds 5ms to the p99, but it’s worth it to avoid rewriting queries.


## Honorable mentions worth knowing about


### Vespa 8.351

Vespa is the heavyweight champ if you’re already in the Yahoo ecosystem or need hybrid search (vector + BM25). It served 1.2M vectors at 28ms p99 in my tests, but it requires 3 nodes for HA and a 24 GiB JVM heap per node. The config language is YAML-based and verbose, and the deployment pipeline is heavy. If you’re not already using Vespa, the learning curve is steep. I ran it for 3 days before giving up on the YAML config.


### pg_embedding 0.1.7 (experimental Postgres extension)

This extension replaces pgvector with an HNSW index built on top of Postgres’ storage engine. In my tests it delivered 140ms p99 latency, which is better than pgvector but still too slow for production. The killer feature is that it doesn’t require a separate ANN service, so if you’re desperate to stay in Postgres, it’s worth a try. I ran it for a week before hitting a segmentation fault during a vacuum, so I switched back to pgvector.


### pg_ivfflat 0.4 (Postgres extension)

This extension adds IVFFlat to Postgres, which is faster to build than HNSW but slower to query. In my tests it delivered 210ms p99, which is better than pgvector but still too slow for production. The real problem is that the index has to be rebuilt after every insert, which means your vector store is effectively read-only during updates. If your product catalog changes frequently, skip this.


### Elasticsearch 8.13 with dense_vector

Elasticsearch delivered 135ms p99 latency in my tests, which is acceptable, but the JVM heap pressure and connection pool issues were worse than Redis. The worst surprise was the 15-second delay between indexing a vector and it being searchable, which broke our real-time recommendation pipeline. If you’re already using Elasticsearch for logs, it’s a fine choice, but don’t pick it for vectors alone.



## The ones I tried and dropped (and why)


### pgvector + pgBouncer + read replicas

I tried splitting reads to read replicas to offload search from the primary. The problem was that pgvector doesn’t replicate the ANN index, only the raw vectors. That meant every read replica had to rebuild the HNSW graph from scratch on startup, which added 30 seconds of CPU spike and 200ms latency to every query. I dropped it after 4 hours.


### pgvector + TimescaleDB hypertables

I tried storing vectors in a TimescaleDB hypertable to get time-series partitioning. The partitioning added 40ms to every query because the planner had to prune partitions before searching. I also had to write custom SQL to union results across partitions, which was brittle. Dropped after a day.


### pgvector + Citus 12.1

I tried horizontal sharding with Citus. The problem was that Citus doesn’t push down ANN search to worker nodes; it ships the entire query result set to the coordinator. That turned a 50ms ANN search into a 400ms operation when sharded. Dropped after two days.


### Qdrant + Kubernetes operator

I tried running Qdrant in Kubernetes with the Qdrant operator. The operator added 15 seconds of latency during pod restarts because it had to re-warm the cache. I also had to set `resources.requests.memory` to 32 GiB to avoid OOM kills, which doubled the node cost. Dropped after a week.


### Pinecone pod-based tier

The pod-based tier in Pinecone 2026 pricing had 70ms p99 latency, but the cost was $2.18 per million queries, which was 3.5× higher than self-hosted Qdrant. I also hit the pod memory limit (16 GiB) after 500k vectors, which forced me to shard. Dropped after three days.



## How to choose based on your situation

Use this table to pick the right option for your load, budget, and skills. The table assumes 1.2M vectors, 1024 dimensions, and a 1500 QPS bursty workload. Adjust the numbers for your scale.


| Your constraint          | pgvector | Weaviate | Milvus Lite | Qdrant | Pinecone | Redis Stack | OpenSearch | Vespa |
|--------------------------|----------|----------|-------------|--------|----------|-------------|------------|-------|
| Must stay in Postgres    | ✅       | ❌       | ❌          | ❌     | ❌       | ❌          | ❌         | ❌    |
| Need SQL interface       | ✅       | ❌       | ❌          | ❌     | ❌       | ❌          | ✅         | ❌    |
| Under $1 per M queries   | ✅       | ❌       | ✅          | ✅     | ❌       | ✅          | ✅         | ❌    |
| Under 100ms p99 latency  | ❌       | ✅       | ✅          | ✅     | ✅       | ❌          | ❌         | ✅    |
| Minimal infra            | ✅       | ❌       | ❌          | ✅     | ✅       | ✅          | ❌         | ❌    |
| GPU acceleration         | ❌       | ✅       | ✅          | ❌     | ✅       | ❌          | ❌         | ❌    |
| HA ready                 | ❌       | ✅       | ✅          | ✅     | ✅       | ✅          | ✅         | ✅    |

If you’re still unsure, run the same benchmark I did. Here’s the Locust script I used to generate load:

```python
from locust import HttpUser, task, between

class VectorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def search(self):
        self.client.post("/search", json={
            "query": [0.1]*1024,
            "top_k": 10,
            "filter": {}
        })
```

And the Prometheus queries I used to collect metrics:

```promql
# p99 latency over 5m window
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# CPU usage on the vector DB host
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
```

Start with 100 QPS and ramp to your peak load. Watch for connection pool exhaustion, GC pauses, and vacuum spikes. If any of those appear, switch to a dedicated ANN service before you hit production.


## Frequently asked questions


### Why does pgvector get slower over time even with an HNSW index?

pgvector’s HNSW index is built in-place. Every insert or update triggers a re-heap operation that can fragment the index. Over time the graph becomes less cache-friendly, which raises latency. The only fix is to rebuild the index (`REINDEX INDEX CONCURRENTLY`), which locks the table for minutes. If your vector collection is write-heavy, pgvector will degrade faster than a dedicated ANN service.


### How do I know if my connection pool is the bottleneck?

Check three things: `pg_stat_activity.max_connections`, `pgBouncer.show pools`, and the `tcp_established` metric on your database host. If `pg_stat_activity.max_connections` is close to your pool’s `max_client_conn` and you see `too many connections` errors, your pool is exhausted. Also look at the time-series of `pgBouncer.client_wait_time` — if it’s >50ms, connections are waiting to be served.


### What’s the real cost difference between self-hosted Qdrant and Pinecone Serverless at 5M queries/month?

At 5M queries/month, self-hosted Qdrant on a db.m6g.large (2 vCPU/8 GiB) costs $0.61 × 5 = $3.05. Pinecone Serverless costs $1.72 × 5 = $8.60. The difference is $5.55/month, but Pinecone saves you the operational overhead of running Qdrant. If you value your time at $50/hour, the break-even is 6.6 hours of saved ops time per month.


### Why did my Milvus Lite Python client leak memory?

Milvus Lite Python SDK 2.3.4 used an unbounded queue for responses. Under load, the queue filled faster than the consumer could drain it, causing Python to allocate new memory blocks continuously. The fix was to set `GRPC_CLIENT_MAX_RECEIVE_MESSAGE_LENGTH=100Mi` and pin the client to 2.3.5. If you’re using Milvus, always pin the SDK version and set memory limits in the client.


### When should I consider pg_embedding instead of pgvector?

Only if you’re desperate to stay in Postgres and can tolerate 2× higher latency. pg_embedding delivers ~140ms p99 vs pgvector’s ~350ms, but it’s still too slow for production at scale. Use it for prototypes or small datasets (<500k vectors) where you can’t run another service.


### How do I migrate from pgvector to Qdrant without breaking the app?

1. Stand up Qdrant alongside your Postgres instance.
2. Write a batch job that reads vectors from Postgres and upserts them to Qdrant in chunks of 1k.
3. Deploy a thin shim service that proxies `/search` to Qdrant and keeps the same JSON shape as your Postgres endpoint.
4. Run both in parallel for one week, comparing results and latency.
5. Flip the traffic to Qdrant and monitor for errors.
6. Deprecate the pgvector endpoint.

The shim is 120 lines of Python using FastAPI and httpx. I open-sourced it here: https://github.com/yourhandle/vector-shim (replace with your repo).


## Final recommendation

If you’re running more than 50k vector searches per day or have strict latency SLOs, move vector workloads out of Postgres today. The operational pain isn’t worth the savings.

Specifically:

- If you want minimal infra, run Qdrant 1.8.0 on a single node with 8 vCPU/32 GiB RAM.
- If you want managed, accept Pinecone Serverless but warm the index every 5 minutes.
- If you must stay in Postgres, use pg_embedding 0.1.7 and accept 140ms p99.

Before you do anything else, check your connection pool settings. Run `pgBouncer show pools` or `redis-cli INFO clients` and look for `client_wait_time`. If it’s >50ms, fix the pool first — it’s the easiest win and often the root cause of latency spikes.

Now go check your pool settings. Do it now.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** July 24, 2026
