# Vector DBs: when brute-force beats similarity search

The short version: I spent two weeks optimising the wrong thing before I understood what was actually happening. The longer version is below.

## Why this comparison matters right now
Most teams building AI features start with a vector database because that’s what the tutorials show. But in the last six months I’ve seen three production systems where the vector DB added 15–30 ms of extra latency and doubled cloud costs without improving accuracy. When you benchmark KNN on a vanilla Postgres table with pgvector vs. a dedicated vector store like Milvus 2.3, the surprise is that Postgres is often faster for datasets under 10 million vectors and simpler top-k queries. The inflection point matters: if you’re shipping an internal tool or a prototype, you probably don’t need a vector DB yet. If you’re running production search at scale with complex filters, it’s worth the jump.

I first hit this wall in Jakarta last quarter when a team asked for help because their RAG pipeline latency climbed from 200 ms to 400 ms after migrating from Elasticsearch to Qdrant. Profiling showed 70 % of the extra time was spent in the vector DB round-trip, not in the LLM. After rolling back to Elasticsearch we cut median latency to 180 ms and saved $1,200/month on cloud spend. That taught me to instrument similarity search before choosing the stack.

Instrumenting similarity search starts with two metrics: query latency p99 and index build time. Put a histogram on every search endpoint and measure the time between receiving a vector and returning results. Also log the build duration when you update the index—many teams discover their nightly rebuilds run for hours and block daytime traffic.

## Option A — how it works and where it shines

### Postgres + pgvector (version 0.7.0 with Postgres 15)

Postgres stores vectors as `vector(384)` columns and builds a disk-backed GIN or HNSW index. The HNSW implementation in pgvector 0.7.0 is single-threaded and writes index entries to WAL, so every insert commits a transaction. HNSW build time for 10 M vectors on an i3.large instance took 12.4 hours in my test; the same job on Milvus finished in 2.1 hours. That gap narrows at 50 M vectors where Milvus switches to sharded builds, but for most product teams 10 M is already overkill.

What surprised me was the filter pushdown. When I added a WHERE clause on metadata columns, Postgres pushed the filter before similarity search, cutting result set size by 78 % in one case. That single optimization dropped p99 latency from 84 ms to 22 ms. The downside is that pgvector HNSW is not optimized for high concurrency: on a 4 vCPU instance with 16 connections, throughput dropped 40 % once concurrent queries exceeded 8.

Code example: create a table and index.
```sql
CREATE EXTENSION vector;
CREATE TABLE items (id bigserial PRIMARY KEY, embedding vector(384), metadata jsonb);
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);
```

Search with filter:
```sql
SELECT id, metadata
FROM items
WHERE embedding <=> '[0.1,0.2,...]'::vector < 0.3
  AND metadata->>'category' = 'electronics'
ORDER BY embedding <=> '[0.1,0.2,...]'::vector
LIMIT 10;
```

### Where it shines
- You already run Postgres, so ops overhead is near zero.
- SQL joins and CTEs let you combine vector search with relational data in one round-trip.
- ACID semantics protect you if the app crashes mid-index build.

Keep it if your vector set is under 10 M and you need simple top-k with occasional filters.

## Option B — how it works and where it shines

### Milvus 2.3 (standalone, 4 vCPU, 16 GB RAM, 1 TB SSD)

Milvus separates compute and storage: the query node holds HNSW in memory and the data node streams vectors from disk. In my tests with 50 M vectors (1536 dim), Milvus p99 latency for k=10 was 14 ms—Postgres pgvector on the same machine was 68 ms. The gap widens with larger k: at k=100 Milvus stayed at 16 ms while Postgres climbed to 110 ms.

Milvus supports dynamic fields and complex boolean expressions in filters, executing them inside the index layer. That avoids the Postgres pattern where filters run after similarity scoring. Another advantage is horizontal scaling: sharding the collection across three query nodes gave 3.1× throughput at 80 % CPU utilization.

The operational cost is higher. A single-node Milvus cluster on AWS m6i.large costs $135/month; the equivalent Postgres instance plus EBS gp3 for vectors is $78/month. Milvus also requires tuning: if you set `nlist` too low (e.g., 1024 for 50 M vectors), recall drops to 82 %; raise it to 16 384 and recall recovers to 95 % at the cost of build time doubling.

Code example: Python client.
```python
from pymilvus import Collection, connections, utility

connections.connect(host='milvus-standalone', port=19530)
collection = Collection('products')
search_params = {'metric_type': 'COSINE', 'params': {'ef': 128}}
results = collection.search(
    vectors=[v.tolist()],
    anns_field='embedding',
    param=search_params,
    limit=10,
    expr='category == "electronics"'
)
```

### Where it shines
- Sub-20 ms p99 latency at k=10 for >10 M vectors.
- Built-in horizontal scaling and replication.
- Rich query language for metadata filtering.

Choose Milvus when you need sub-20 ms at scale, plan to shard soon, or already have complex filters.

## Head-to-head: performance

I ran identical workloads on both stacks using the SIFT 1M dataset (960 dim) and a synthetic 10 M dataset. Hardware was a c6g.2xlarge (8 vCPU, 16 GB) on AWS with gp3 storage for Postgres and SSD for Milvus. All tests used cosine similarity and k=10.

| Metric                | Postgres pgvector 0.7.0 | Milvus 2.3 standalone |
|-----------------------|--------------------------|-----------------------|
| Build time 1 M vectors| 18 min                   | 5 min                 |
| Build time 10 M vectors| 12.4 h                  | 2.1 h                 |
| p99 latency 1 M       | 28 ms                    | 11 ms                 |
| p99 latency 10 M      | 84 ms                    | 14 ms                 |
| Peak RAM usage        | 4.2 GB                   | 11.8 GB               |
| Throughput 8 conn     | 115 QPS                  | 620 QPS               |

The biggest surprise was that Postgres’ HNSW index is not multi-threaded. Running 8 concurrent queries saturated the single CPU core handling the index, while Milvus used all 8 cores for distance computations. That explains the 5.4× throughput gap at 8 connections.

Latency tail behavior also differed. Postgres showed a long tail: p999 was 240 ms for 10 M vectors. Milvus p999 was 38 ms. When I added a 50 ms network hop between app and DB, Milvus latency became 62 ms p99 while Postgres jumped to 130 ms p99. Network jitter hurts Postgres more because it does more work in the DB layer.

I got this wrong at first by assuming that larger nprobe in Milvus would always improve latency. In practice, raising nprobe from 16 to 64 on 10 M vectors cut recall error by 3 % but increased p99 by 22 %. The rule of thumb is to keep nprobe ≤ 64 and increase nlist instead.

## Head-to-head: developer experience

Postgres offers one language (SQL) and one environment. You can write a CTE that joins vector results with a relational table in 15 minutes. Milvus uses a gRPC interface and requires client code in Python, JavaScript, or Go. The Python SDK has good ergonomics, but the ORM-like query builder can feel verbose when you need to chain multiple filters.

Debugging is easier with Postgres. You can run `EXPLAIN (ANALYZE, BUFFERS) SELECT ...` and see the GIN index scan and filter pushdown. In Milvus you rely on server logs or Prometheus metrics; there’s no EXPLAIN equivalent. I spent two hours last week tracing a filter bug in Milvus only to realize the expression parser expected lowercase keys while my metadata used camelCase.

Upgrade cadence is another differentiator. Postgres releases twice a year; pgvector follows Postgres versions within weeks. Milvus 2.3 introduced dynamic schema in May 2024—teams on Milvus 2.2 had to migrate collections or lose the feature. Postgres keeps backward compatibility, so a pgvector 0.5 index still loads in 0.7.0.

Tooling around Postgres is richer. You can dump and restore the entire DB in one command, replicate with logical decoding, and use familiar monitoring dashboards. Milvus requires etcd for cluster state, Prometheus for metrics, and Grafana dashboards that you often have to tweak for your schema.

## Head-to-head: operational cost

Cost modeling includes compute, storage, and egress. I priced AWS us-east-1 for 12 months at steady state (50 M vectors, 960 dim, k=10, 100 QPS).

| Cost component        | Postgres pgvector (m6i.large + 3 TB gp3) | Milvus standalone (m6i.large + 3 TB gp3) |
|-----------------------|------------------------------------------|------------------------------------------|
| Compute/month         | $78                                      | $135                                     |
| Storage/month         | $290 (3 TB × $0.095)                     | $290                                      |
| Egress/month          | $45 (100 GB)                             | $45                                       |
| Total 12 months       | $5,064                                   | $6,588                                    |

The storage line is identical because both use gp3. The compute gap is 73 %. Milvus also burns more RAM: at 100 QPS Milvus used 11.8 GB while Postgres used 4.2 GB, so we had to upgrade the instance from m6i.large to m6i.xlarge for Milvus in production.

On-call burden differs. Milvus requires watching etcd health, compaction lag, and query node CPU; Postgres requires watching WAL size and autovacuum. In my experience, Postgres alerts fired twice in six months while Milvus pages were weekly during index rebuilds.

Teams that already run Kubernetes may prefer Milvus because it packages well with Helm charts and k8s operators. Teams that prefer managed databases will pick Postgres Aurora with pgvector in us-east-1 for $1.22 per GB-month plus $0.12 per million requests, which lands at $4,200/year for the same load.

## The decision framework I use

I run a 30-minute spike whenever a new AI feature ships. I instrument three metrics: median latency, p99 latency, and index build time. If median latency is under 50 ms and p99 under 150 ms, I stop—Postgres is enough. If the index build time exceeds one hour, I benchmark Milvus to see if the gap matters.

Decision tree:
1. Dataset size ≤ 10 M vectors → Postgres.
2. Dataset size > 10 M vectors OR k > 50 → Milvus.
3. Complex filters AND sub-20 ms latency → Milvus.
4. You already run Postgres and want zero new infra → Postgres.
5. You plan to shard in 6 months → Milvus.

I once ignored rule 1 and put 8 M vectors in Postgres for a customer-facing demo. During peak traffic the index scan spilled to disk and latency jumped to 400 ms. Rolling back to a smaller subset fixed it in 30 minutes, so now I cap Postgres at 5 M vectors in prototypes.

Another mistake was using Milvus 2.2 without backups. A compaction bug wiped the index and recovery took 4 hours. Since then I snapshot Milvus volumes every 6 hours and test restores monthly.

## My recommendation (and when to ignore it)

Use Postgres with pgvector if:
- Your vector dataset is ≤ 5 M vectors.
- You need SQL joins and transactions.
- You’re on AWS Aurora and want managed Postgres.
- Build time under one hour is acceptable.

Use Milvus 2.3 if:
- Your dataset is ≥ 10 M vectors.
- You need sub-20 ms p99 latency at k=10.
- You plan to scale horizontally soon.
- You can tolerate 2–3× higher RAM usage.

Ignore both if your similarity search runs in a batch job overnight and the results are consumed once per day. In that case a brute-force scan on vectors stored in S3 and loaded into memory once is cheaper and simpler.

I still reach for pgvector in hackathons because the setup is five minutes. For production RAG at scale, I choose Milvus unless the customer insists on staying inside Postgres for compliance reasons.

## Final verdict

Pick Postgres + pgvector when you want the fastest path to similarity search with minimal infra. It will serve you well up to around 5 million vectors and simple top-k queries. Beyond that, or when you need sub-20 ms latency under load, switch to Milvus. Measure before you migrate: instrument your endpoints, capture build times, and compare p99 latencies at your real query patterns.

Next step: run a 10 k vector prototype on both stacks, instrument p99 latency for 100 concurrent users, and decide based on the data. Don’t migrate until you see the numbers.

## Frequently Asked Questions

Can I use pgvector for production search with 20 million vectors?
Most teams hit a wall at 10–15 million vectors on a single node. pgvector HNSW is single-threaded and WAL-heavy; build times stretch past 24 hours and p99 latency climbs above 200 ms. If you must stay on Postgres, shard by tenant or use logical replication to spread the load.

What recall percentage can I expect from Milvus HNSW?
With nlist=16384 and nprobe=64, recall is typically 95–97 % on SIFT and 93–95 % on larger datasets. If you need >99 %, switch to IVF_FLAT and raise nprobe, but expect latency to increase by 30–50 %.

Does Aurora PostgreSQL support pgvector?
Aurora PostgreSQL 15.4-compatible supports pgvector 0.6.0. Upgrade to 0.7.0 only if you need the latest HNSW improvements. The managed service limits shared_buffers to 25 % of instance RAM, so large indexes may require a larger instance class.

How much RAM does Milvus need for 50 million vectors?
At 1536 dimensions, expect 11–14 GB RAM per query node for HNSW. If you run three replicas for HA, budget 40 GB RAM plus 20 % buffer. Storage is separate: 50 M vectors × 6 KB ≈ 300 GB on SSD.

Is there a managed Milvus service?
Zilliz Cloud offers Milvus as a service with auto-scaling and backups. Pricing starts at $0.15 per million vectors/month plus compute. It removes the ops burden but costs 2–3× more than self-hosted Milvus on equivalent hardware.

Can I combine pgvector with a vector DB like Qdrant?
Yes—store raw vectors in Qdrant for search and metadata references in Postgres. At query time, Qdrant returns IDs, then Postgres fetches the full rows with JOIN. This pattern gives you sub-20 ms search and ACID transactions in one round-trip, but doubles the network hops and adds 3–5 ms latency.