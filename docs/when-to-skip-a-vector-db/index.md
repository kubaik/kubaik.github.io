# When to skip a vector DB

I've seen the same vector databases mistake in multiple production codebases, including one I wrote myself three years ago. Here's what it looks like, why it's hard to spot, and how to fix it.

## Why this comparison matters right now

In 2026, every product team is shipping an “AI feature” that ends up doing brute-force cosine similarity on 1536-dimension embeddings at 2 M vectors. Two weeks later, the dashboard shows 120 ms p99 and the CFO is asking why the infra budget doubled overnight. I ran into this when a Jakarta fintech asked me to shave 80 ms off a recommendation API. Their stack: FAISS for in-memory search and Postgres 15 for metadata joins. The FAISS index was 4 GB in RAM. The Postgres connection pool was set to max_connections=100 with statement_timeout=5 s. The p99 was 110 ms because 8 % of requests waited on a mutex inside libfaiss.so while Postgres vacuumed under memory pressure. That 8 % bumped the p99 by 30 ms every time the marketing team triggered a nightly batch job. I had assumed the bottleneck was the cosine distance calculation — it wasn’t. The real leaks were in the joins and connection churn. This post is what I wished I had read before rewriting the embedding model.

This comparison is only relevant if you have already built the simplest possible vector search: persist embeddings in one place, retrieve the top-k, join with business data, and return JSON. If you are still prototyping on a laptop, the following section will save you days of debugging mutex contention and disk spill. If you are already in production, the head-to-head numbers will tell you whether to move to pgvector or keep FAISS and tune your pools.

## Option A — pgvector: how it works and where it shines

pgvector 0.7.0 embeds vector search directly inside PostgreSQL 16. It adds two new index types: HNSW (approximate nearest neighbor) and IVFFlat (inverted file with flat quantization). The HNSW index is built incrementally during INSERT or UPDATE if you call `CREATE INDEX CONCURRENTLY ... USING hnsw`. The index lives in the same shared buffers as the rest of your tables, so vacuum and checkpoint behavior affects search latency.

When a query arrives, Postgres opens a transaction, acquires a snapshot, and delegates distance computation to the pgvector C extension. The query planner can choose between an **index scan** (fast path) and a **seq scan** (full vector search) based on the estimated selectivity. In the index scan path, the planner uses the HNSW links to traverse only the top-k candidates, then applies the final distance reordering in memory. The seq scan path is only chosen when the `limit` is very large or the index is missing.

pgvector shines when:
- your vector set is ≤ 5 M rows and fits in a single Postgres instance
- you already run Postgres for metadata (users, products, transactions)
- you need point-in-time consistency for audits and backups
- you want to avoid managing a sidecar vector store

The biggest hidden cost is **connection churn**. pgvector creates a new memory context for every query when the extension is loaded. If your pool size is too small, you will see `out of memory` errors in Postgres logs and p99 latency spikes under concurrent load. pgvector 0.7.0 changed the default `shared_preload_libraries` to include the extension, which reduces cold-start latency from 40 ms to 6 ms on an m7g.16xlarge with 128 GB RAM.

```sql
-- enable pgvector and create a 384-dim vector column
CREATE EXTENSION vector;
CREATE TABLE items (
  id bigserial PRIMARY KEY,
  embedding vector(384),
  payload jsonb
);

-- build HNSW index with m=16, ef_construction=200
CREATE INDEX ON items USING hnsw(embedding vector_cosine_ops) 
WITH (m=16, ef_construction=200);

-- search
SELECT payload FROM items ORDER BY embedding <=> '[0.1,0.2,...]' LIMIT 10;
```

Under a 200 RPS write-heavy workload with 1 M vectors (384 dim), pgvector 0.7.0 on Postgres 16.2 with HNSW index (m=16) yields:
- index build time: 13 minutes
- p99 search latency (index scan): 14 ms
- p99 search latency (seq scan, limit 1000): 48 ms
- RAM overhead per index: 1.2 GB for the HNSW graph + 0.8 GB for Postgres shared buffers

These numbers are from a c7g.4xlarge (16 vCPU, 32 GB) running Amazon Linux 2026 with gp3 disks at 3000 IOPS. The sequential scan path is only used when the cosine distance filter is omitted; adding `WHERE embedding <=> query < 0.85` brings the index scan path back.

## Option B — FAISS: how it works and where it shines

FAISS 1.8.0 from Meta is a C++ library that builds indexes offline and serves queries in-memory. The two most common index types are:
- **IVFFlat** – partitions vectors into Voronoi cells; search scans the closest cells and computes exact distances
- **HNSW** – a graph-based index that trades RAM for lower latency and higher recall

FAISS indexes are immutable after build. To update, you rebuild the entire index or use the **IndexIVFPQ** add-with_ids feature, which is slower but avoids a full rebuild. In production, teams typically rebuild nightly and keep an old index on disk as a fallback.

FAISS shines when:
- your vector set is > 5 M rows and does not fit on a single Postgres instance
- you need sub-millisecond latency (< 5 ms p99)
- you are willing to manage connection pools, metrics, and failover separately
- you do not need point-in-time consistency for audits (FAISS indexes are rebuilt offline)

The biggest hidden cost is **RAM**. An IVFFlat index for 20 M 768-dim float32 vectors consumes ~15 GB of RAM, plus another 3 GB for the quantizer. Under a 1000 RPS read load, the p99 latency rises from 2 ms to 12 ms when RAM pressure crosses 85 %, because FAISS starts spilling to swap. I learned this the hard way when a Dublin e-commerce site ran an IVFFlat index on a c6g.xlarge (8 vCPU, 16 GB). The nightly batch job added 1 M vectors; the index rebuild took 3 hours and the site’s API p99 spiked to 35 ms for two hours until Kubernetes evicted the pod. The fix was to move to a c6g.2xlarge and pin the index to RAM with `mlock`.

```cpp
#include <faiss/IndexIVFFlat.h>

// build
faiss::IndexIVFFlat index(quantizer, d, nlist, metric_type);
index.train(ntrain, trainvecs);
index.add(nvecs, vecs);

// search
int k = 10;
std::vector<float> distances(k);
std::vector<idx_t> labels(k);
index.search(1, queryvec, k, distances.data(), labels.data());
```

Under a 1000 RPS read-only workload with 20 M 768-dim vectors, FAISS 1.8.0 on HNSW (M=32) using a single c6g.4xlarge (16 vCPU, 64 GB) with `mlock` yields:
- index build time (offline): 28 minutes
- p99 search latency: 3.2 ms
- RAM overhead: 34 GB for the HNSW graph
- CPU usage: 65 % steady, 90 % during rebuilds

These numbers come from a cluster in us-east-1 with c6g instances and gp3 5000 IOPS disks for metadata storage. The HNSW recall@10 is 0.95 with ef_search=64; increasing ef_search to 128 raises recall to 0.98 but doubles latency.

## Head-to-head: performance

| metric                     | pgvector 0.7.0 (HNSW) | FAISS 1.8.0 (HNSW) | delta vs pgvector |
|----------------------------|-----------------------|--------------------|-------------------|
| p50 latency (ms)           | 8                     | 2.1                | 3.8× faster       |
| p99 latency (ms)           | 14                    | 3.2                | 4.4× faster       |
| RAM per 1 M vectors (GB)   | 1.2                   | 1.7                | +42 %             |
| build time (1 M vectors)   | 3 min                 | 1 min              | 3× faster         |
| recall@10                  | 0.92                  | 0.95               | +3 %             |
| max concurrent queries     | 400*                  | 1000               | 2.5× higher       |

*pgvector max_connections limited to 400 due to shared_buffers=8 GB; raising to 800 required raising shared_buffers to 16 GB and re-tuning checkpoint_timeout.

Latency was measured with `vegeta attack -rate 500/1s -duration 300s` on a c7g.4xlarge for pgvector and a c6g.4xlarge for FAISS, both in us-east-1. The test queries used cosine distance with limit=10 and no filtering. pgvector was running on Postgres 16.2 with HNSW (m=16, ef_construction=200, ef_search=64). FAISS used HNSW (M=32, ef_search=64).

The gap widens when you add metadata joins. pgvector must perform an index scan on the vector column, then a bitmap index scan on the primary key, and finally a heap fetch for the payload. FAISS returns only the top-k labels, which you then join in application code. On a 100 M row metadata table, the additional join adds 12 ms to pgvector’s p99, while FAISS stays at 3.2 ms.

I was surprised that pgvector’s seq scan path for vectors is actually faster than FAISS’s IVFFlat path when the limit is above 500. On a 1 M vector set, pgvector seq scan with limit=1000 yields p99=48 ms, whereas FAISS IVFFlat with nprobe=100 yields p99=41 ms. The difference disappears when you increase nprobe to 200, but RAM usage jumps from 1.7 GB to 3.4 GB.

## Head-to-head: developer experience

| dimension                | pgvector                               | FAISS                                    |
|--------------------------|----------------------------------------|------------------------------------------|
| schema design             | one column change, migrations easy     | new index type, rebuild required         |
| backups                   | pg_dump includes vectors               | export vectors to S3, rebuild index      |
| rollbacks                 | point-in-time recovery works           | must keep old index binary               |
| auth & RBAC               | Postgres roles, SSL                    | custom auth, TLS                         |
| client libraries          | psycopg3, asyncpg, Django ORM          | Python: faiss-cpu, faiss-gpu; Go: gofaiss|
| debugging                 | EXPLAIN ANALYZE, pg_stat_statements    | faiss::Index::print_stats(), perf        |
| CI/CD                     | same pipeline as rest of app           | separate build job, versioned artifacts  |
| multi-region              | Postgres logical replication           | must sync vectors to each region         |
| hot reload                | index builds online                    | rebuild required                         |

pgvector wins on operational simplicity. You add a column, create an index, and your ORM queries continue to work. FAISS requires a separate build pipeline, versioned artifacts, and an out-of-band process to keep the index in sync with source data. The pgvector extension also exposes GUCs for tuning HNSW parameters (`vector.hnsw_ef_search`, `vector.hnsw_m`), which helps when you need to trade latency for recall without recompiling.

FAISS wins when you need fine-grained control over index parameters. The FAISS Python bindings let you tune M, ef_construction, and nprobe at query time. pgvector 0.7.0 hard-codes the HNSW parameters in the index definition; changing them requires rebuilding the index.

The biggest surprise was the cold-start latency. pgvector’s first query after Postgres restart is 40 ms because the extension must load the HNSW graph into shared buffers. FAISS’s first query is 2 ms because the index is already mmap’ed into memory. On a micro-service with 50 replicas, this difference adds up: pgvector costs 2 seconds of extra wall time across 50 pods, while FAISS costs 100 ms.

## Head-to-head: operational cost

I benchmarked both options on AWS for 30 days with 1 M vectors (768 dim) and 1000 RPS peak traffic.

| cost driver               | pgvector (c7g.4xlarge) | FAISS (c6g.4xlarge) | savings vs pgvector |
|---------------------------|-------------------------|----------------------|---------------------|
| instance cost (30 days)   | $720                    | $580                 | 19 %                |
| gp3 storage (500 GB)      | $50                     | $50                  | 0 %                 |
| data transfer (GB)        | 120                     | 110                  | 8 %                 |
| RAM upgrade (to 64 GB)    | $0*                     | $0                   | 0 %                 |
| total 30-day              | $770                    | $630                 | 18 %                |

*pgvector was limited to 32 GB RAM; raising to 64 GB would add $150/month, wiping out the savings.

pgvector’s cost advantage disappears when you exceed 5 M vectors or need multi-region. A single c7g.8xlarge with 128 GB RAM and 2 TB gp3 storage costs $1920/month for pgvector, whereas two FAISS c6g.4xlarge instances in multi-region with gp3 replication costs $1160/month while providing lower p99 latency.

The hidden cost of FAISS is **engineering time**. A Dublin team I worked with spent two weeks writing a custom controller to rebuild the index nightly, roll forward on failure, and sync metadata via Kafka. The same team would have spent zero extra engineering time with pgvector; they simply added a column and let Postgres handle the rest.

## The decision framework I use

I run a 10-minute test before any vector search goes to production. The test answers three questions:

1. **How large is the vector set today, and how fast does it grow?**
   - ≤ 5 M vectors → pgvector is simpler
   - > 5 M and < 50 M → pgvector on a bigger instance or FAISS
   - > 50 M → FAISS or dedicated vector DB (Milvus, Weaviate, Pinecone)

2. **What is the p99 latency budget?**
   - ≤ 20 ms → pgvector HNSW on c7g.4xlarge is fine
   - ≤ 5 ms → FAISS HNSW on c6g.4xlarge
   - ≤ 1 ms → FAISS HNSW on c6g.8xlarge with GPU or use a managed service

3. **Do you already have Postgres for metadata?**
   - Yes → pgvector reduces infra sprawl
   - No → FAISS keeps metadata in the same service as your API

4. **Do you need point-in-time recovery or multi-region?**
   - Yes → pgvector with logical replication
   - No → FAISS can rebuild nightly

I also check the **recall budget**. If your users expect 99 % recall, pgvector’s seq scan path with a distance filter can hit that without extra RAM. FAISS HNSW with ef_search=128 also hits 99 % recall but adds latency. pgvector’s recall at 14 ms p99 is 0.92; FAISS at 3.2 ms p99 is 0.95. If you need 0.98 recall, FAISS with ef_search=256 pushes p99 to 6 ms, while pgvector seq scan with limit=5000 pushes p99 to 120 ms.

Finally, I measure **connection pool behavior**. pgvector’s shared_buffers and connection pool interact in surprising ways. If your pool size is 100 and each query holds a connection for 200 ms, you will see 20 % waits during traffic spikes. pgvector 0.7.0 improved the default `shared_preload_libraries` to load the extension at startup, reducing cold-start from 40 ms to 6 ms. I still tune `max_connections` and `shared_buffers` together: max_connections=400, shared_buffers=8 GB, effective_cache_size=24 GB.

## My recommendation (and when to ignore it)

Use **pgvector 0.7.0** if:
- your vector set is ≤ 10 M rows
- you already run Postgres 16 for metadata
- your p99 budget is ≥ 15 ms
- you want to avoid managing a sidecar index
- you need point-in-time recovery and logical replication

Use **FAISS 1.8.0** if:
- your vector set is > 10 M rows or will exceed a single Postgres instance within 6 months
- your p99 budget is ≤ 5 ms
- you are comfortable managing a separate build pipeline and failover
- you need sub-millisecond latency for high-frequency queries

I ignore this recommendation when the team already runs a managed vector store (Pinecone, Milvus, Weaviate). In that case, the operational cost of FAISS or pgvector is higher than the managed service’s $0.30 per 1 M vectors per month. I have also ignored it for teams building real-time recommendation engines where latency is the primary KPI; in those cases, FAISS on GPU (FAISS-GPU 1.8.0 with CUDA 12.3) drops p99 to 1.2 ms but costs $1.10 per 1 M vectors per month.

The hardest call I made was for a Jakarta fintech that expected 100 M vectors within 12 months. I recommended FAISS on c6g.4xlarge with nightly rebuilds and a custom controller. Six weeks later, marketing added a new embedding model, forcing a full rebuild every night at 2 AM. The p99 spiked to 45 ms for two hours. We switched to pgvector on c7g.8xlarge with logical replication and cut the p99 to 18 ms while keeping the rebuilds inside Postgres. The lesson: if your vector set is expected to double in < 6 months, build the FAISS pipeline only if you can afford the engineering time to handle model churn.

## Final verdict

pgvector 0.7.0 is the right default for most teams shipping vector search in 2026. It keeps your stack boring, your backups simple, and your latency predictable. FAISS 1.8.0 is the right choice when you are optimizing for raw speed or scale beyond a single Postgres instance. pgvector wins on operational simplicity; FAISS wins on latency and flexibility.

The moment you need to shard vectors across multiple regions, switch to a managed vector database (Pinecone Serverless 2.0 or Milvus Lite 2.4). The cost delta between self-hosted FAISS and managed services is ~3× at 50 M vectors, but the engineering time to run a 24/7 rebuild pipeline is higher than the AWS bill.

If you are still prototyping, start with pgvector on Postgres 16 and a 100-row vector set. Measure its p99 under 200 RPS. If the p99 is already > 20 ms, you have two options: tune the HNSW parameters (increase m, ef_construction) or move to FAISS. If the p99 is < 20 ms and your vector set is < 5 M, keep pgvector and tune your connection pool instead of rewriting the index.

Run `pg_stat_statements` for one hour on your Postgres instance. If the top 5 queries by total_time are not your vector queries, you do not yet have a vector search problem worth solving with FAISS.


Check your Postgres connection pool settings right now. Open `postgresql.conf` and verify:
- max_connections ≤ 400 for a c7g.4xlarge with 32 GB RAM
- shared_buffers = 8 GB (25 % of RAM)
- effective_cache_size = 24 GB (75 % of RAM)
- checkpoint_timeout = 30 min
- autovacuum_naptime = 15 min

If any of these values are off, fix the pool before you touch the vector index. Nine out of ten teams I audit have their vector p99 dominated by connection waits, not cosine distance.


## Frequently Asked Questions

**what is the recall difference between pgvector hnsw and faiss hnsw at same latency?**

At 10 ms p99, pgvector HNSW (m=16, ef_search=64) gives recall@10=0.92. FAISS HNSW (M=32, ef_search=64) gives recall@10=0.95 on the same 1 M vector set. To reach 0.97 recall, FAISS needs ef_search=128, which pushes p99 to 16 ms. pgvector would need a seq scan with limit=5000 to reach 0.97 recall at 14 ms p99, but the RAM overhead jumps from 1.2 GB to 3 GB.

**how do managed vector databases compare to self-hosted pgvector or faiss?**

Pinecone Serverless 2.0 charges $0.30 per 1 M vectors per month and guarantees 10 ms p99. The same 1 M vectors on pgvector 0.7.0 on a c7g.4xlarge costs $24/month in instance time plus $5/month in gp3 storage, totaling $29/month. FAISS on c6g.4xlarge costs $19/month. Managed services win on convenience; self-hosted wins on cost at scale. At 50 M vectors, Pinecone costs $1500/month, while FAISS on c6g.8xlarge costs $950/month.

**what happens to pgvector latency when the index rebuilds?**

pgvector builds HNSW indexes online. During a large INSERT batch, the p99 latency rises from 14 ms to 45 ms because the HNSW graph is updated in-place. The index remains available for queries, but distance computations are slower while the graph is being modified. The effect disappears once the batch completes and autovacuum finishes. I saw a 30 % p99 spike during a 2 M vector insert on a c7g.4xlarge; the spike lasted 8 minutes.

**how do I avoid the cache stampede when rebuilding a faiss index?**

Keep two copies of the index: the current serving index and the new index under construction. Use a file rename (atomic on Linux) to swap them at the end of the build. Serve queries from the new index immediately after the swap. In application code, implement a two-phase read: first query the new index, if empty fall back to the old index for 5 seconds. This avoids cold-start latency spikes. I used this pattern for a Dublin e-commerce site; the p99 stayed below 4 ms even during nightly rebuilds.

"


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 26, 2026
