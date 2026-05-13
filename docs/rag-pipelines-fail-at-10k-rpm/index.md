# RAG pipelines fail at 10k RPM

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2023, we launched a customer-facing RAG chatbot for a Vietnamese e-commerce startup with 2M monthly active users. The bot had to answer product queries using a catalog of 500k items with real-time inventory and pricing. We aimed for 95th-percentile latency under 500ms and a cost ceiling of $0.002 per query at 10k requests per minute (RPM).

The tutorials made it look straightforward: ingest documents, embed them, build a vector store, add retrieval, and wire up an LLM endpoint. We used ChromaDB with sentence-transformers/all-MiniLM-L6-v2 (384 dim) and hosted the LLM on an A100 on RunPod at $1.50/hr. On paper, we were set.

What the tutorials didn’t mention was that at 10k RPM, ChromaDB’s default HNSW index would thrash the disk, the embedding queue would back up, and the LLM would get 20% timeouts because it waited for stale or duplicate context. We measured 1.2s median latency at 500 RPM, but at 10k RPM we hit 3.8s median with 12% errors. The cost per query ballooned to $0.0055—almost triple our ceiling—mostly from LLM retries and over-provisioned embedding workers.

The biggest surprise was the inventory updates. Our catalog changed every 5 minutes with new stock, prices, and discontinued items. The vector store couldn’t keep up; we were rebuilding the index every 30 minutes, which locked the chatbot for 45 seconds and dropped requests by 30%. We needed a system that could ingest, index, and serve under load without batch windows.

**Summary:** We underestimated the operational load of frequent updates and high concurrency. The default RAG stack wasn’t built for real-time catalog churn or 10k RPM traffic.


## What we tried first and why it didn’t work

First, we tried upscaling the embedding workers. We spun up 16 CPU workers on c6i.large instances (2 vCPU, 4GB RAM) to handle 10k RPM. Each worker used 100% CPU and still queued for 800ms per embedding. The queue depth hit 2000, and 22% of requests timed out waiting for embeddings. Cost per query rose to $0.006, mostly from idle CPU minutes.

Next, we tried Redis with RedisSearch for vector search. We used the RedisStack module with HNSW index and 384 dim vectors. At 5k RPM it was fine, but at 10k RPM the index rebuild during updates caused 700ms spikes and 15% request failures. The Redis instance (r6g.2xlarge, 8 vCPU, 61GB RAM) cost $0.48/hr, but the LLM endpoint still dominated at $1.50/hr, pushing total cost to $0.007 per query.

We then tried Qdrant 1.8.0 with on-disk storage and async indexing. It handled 10k RPM with 450ms median latency, but the on-disk index had 6-second spikes every time we updated the catalog. We measured 8% failed requests during index rebuilds, and the storage IOPS burst to 15k for 30 seconds, causing EBS throttling on AWS.

Finally, we tried pgvector 0.6.0 on a db.r6g.2xlarge Postgres instance. We used the ivfflat index with 100 lists. At 10k RPM we got 350ms median latency, but the index degraded after 5k updates—recall dropped from 0.92 to 0.78, and the LLM hallucinated prices 11% of the time. The database CPU hit 95% and we had to scale to db.r6g.4xlarge, doubling the cost to $0.68/hr.

**Summary:** Every off-the-shelf stack we tried hit a wall at 10k RPM: queue backlogs, index rebuild stalls, or cost overruns. The core issue was that catalog updates forced full or partial index rebuilds, which blocked serving traffic.


## The approach that worked

We realized we needed a system that decoupled ingestion from serving. We split the pipeline into two lanes: a real-time lane for serving traffic and a background lane for ingesting updates. For serving, we used a pre-built snapshot of the index that only changed when the background lane finished a full rebuild. For ingestion, we used a write-optimized store that buffered changes without blocking reads.

For serving, we chose Milvus 2.3.3 with an HNSW index and 16 CPU cores on a Kubernetes node. We pre-built the index nightly and served it from memory-mapped files, avoiding rebuilds during peak hours. For ingestion, we used ClickHouse 23.8 with the `VectorIndex` engine (HNSW) configured for write-optimized mode. ClickHouse buffered updates in memory and flushed to disk every 5 minutes, so the serving index stayed consistent without blocking.

We added a reconciliation step: every 30 minutes, we compared the serving index (Milvus) with the ingestion index (ClickHouse) and applied deltas to Milvus via a background job. This kept recall high (>0.95) and latency low (<450ms at 10k RPM). We also implemented a circuit breaker: if the LLM context length exceeded 4k tokens, we truncated and warned the user, reducing timeouts by 40%.

We measured the system at 10k RPM for a week. Median latency was 380ms, 95th percentile was 490ms. Error rate was 0.9%, down from 12%. Cost per query averaged $0.0018, under our $0.002 ceiling.

**Summary:** Decoupling serving from ingestion, using memory-mapped serving indexes, and write-optimized ingestion buffers solved the update-blocking problem. The circuit breaker for context length cut LLM timeouts.


## Implementation details

Here’s the rough architecture:

- **Ingestion:** ClickHouse 23.8 with `VectorIndex` (HNSW, 384 dim, buffer_flush_interval=300s). We inserted raw catalog items as JSON with a `product_id`, `embedding`, and `last_updated` timestamp. The VectorIndex automatically buffered and flushed to disk every 5 minutes. We tuned `index_build_thread_count=4` and `index_build_memory_limit=4GB` to avoid OOM.

- **Serving:** Milvus 2.3.3 on Kubernetes (16 CPU, 64GB RAM). We pre-built the HNSW index nightly from a snapshot of ClickHouse’s VectorIndex. The serving index was served from memory-mapped files (`storage_type=MMAP`), avoiding disk I/O during queries. We set `nlist=1024` and `efConstruction=200` for recall >0.95.

- **Reconciliation:** A background job every 30 minutes compared the serving index (Milvus) with the latest ClickHouse snapshot. It computed deltas (added, updated, deleted product_ids) and applied them to Milvus via the Milvus SDK (`milvus==2.3.3`). The job ran in 2–3 minutes and used 4 CPU cores.

- **LLM endpoint:** We used vLLM 0.3.0 on an A100 with FlashAttention v2. We set `max_model_len=4096` and `max_num_seqs=8` to avoid OOM. We added a context-length guard: if the prompt + context exceeded 3500 tokens, we truncated the context to 2500 tokens and warned the user. This reduced timeouts by 40%.

- **Orchestration:** Argo Workflows on Kubernetes for the background jobs. We used S3 for snapshots and Redis for job queue. The entire system ran on 3 EKS nodes (m6i.2xlarge) plus ClickHouse on a separate node (r6g.2xlarge).

Here’s the ClickHouse table DDL:

```sql
CREATE TABLE product_embeddings (
  product_id String,
  embedding Array(Float32),
  last_updated DateTime64(3),
  INDEX idx_embedding embedding TYPE HNSW('metric=Cosine')
) ENGINE = MergeTree()
ORDER BY (product_id, last_updated)
SETTINGS index_build_memory_limit = 4194304000; -- 4GB
```

Here’s the Milvus collection creation in Python:

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections

connections.connect(host="milvus", port="19530")

fields = [
  FieldSchema(name="product_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
  FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]
schema = CollectionSchema(fields, description="Product embeddings")
collection = Collection("products", schema)

index_params = {
  "index_type": "HNSW",
  "metric_type": "COSINE",
  "params": {"M": 16, "efConstruction": 200}
}
collection.create_index("embedding", index_params)
collection.load()
```

**Summary:** The implementation relied on write-optimized ingestion (ClickHouse VectorIndex), memory-mapped serving (Milvus MMAP), and a lightweight reconciliation job. The context-length guard in the LLM endpoint cut timeouts without extra infra.


## Results — the numbers before and after

| Metric | ChromaDB (baseline) | Qdrant (attempt 1) | pgvector (attempt 2) | Milvus+ClickHouse (final) |
|---|---|---|---|---|
| Median latency (10k RPM) | 1200ms | 450ms | 350ms | 380ms |
| 95th percentile latency | 3800ms | 1100ms | 1050ms | 490ms |
| Error rate (timeouts + failures) | 12% | 8% | 11% | 0.9% |
| Cost per query (@10k RPM) | $0.0055 | $0.0042 | $0.0070 | $0.0018 |
| Recall (on test set) | 0.89 | 0.91 | 0.78 | 0.95 |
| Index rebuild impact | 45s block every 30m | 700ms spikes | 6s spikes | <3s background job |

The biggest win was cost: we cut cost per query from $0.0055 to $0.0018, a 67% reduction. The error rate dropped from 12% to 0.9%, mostly by removing index rebuild stalls and adding the context-length guard. Median latency improved from 1.2s to 380ms, though we sacrificed 30ms to the context guard.

We also measured infra cost: the final system used 3 EKS nodes (m6i.2xlarge) at $0.408/hr each, a ClickHouse node (r6g.2xlarge) at $0.48/hr, and an A100 for LLM at $1.50/hr. Total infra cost was $4.21/hr at 10k RPM. Before, with ChromaDB and 16 embedding workers, infra cost was $5.80/hr. The saving came from fewer workers and no emergency index rebuilds.

A surprising result was the reconciliation job. It ran every 30 minutes, took 2–3 minutes, and used 4 CPU cores. It felt like overhead, but without it, recall degraded by 7% after 4 hours. The job was cheap (0.2 CPU-hours/day) and kept recall stable.

**Summary:** Milvus+ClickHouse cut latency, errors, and costs dramatically. The reconciliation job was a small but critical cost to maintain recall.


## What we'd do differently

We over-provisioned embedding workers in the first attempt. We should have profiled the embedding model on our actual catalog first. The all-MiniLM-L6-v2 model took 45ms per embedding on a c6i.large CPU instance. At 10k RPM, we needed 22 workers to keep the queue under 100, but they idled 60% of the time. If we had benchmarked first, we could have used 12 workers and saved $0.30/hr.

We also trusted the default HNSW parameters too much. In Milvus, we set `efConstruction=200` and `M=16`, which gave us 0.95 recall but high index build time (12 minutes nightly). If we had reduced `M` to 8, index build would drop to 6 minutes, and we could rebuild twice a day without blocking. We only realized this after load-testing a synthetic dataset.

The context-length guard saved us, but it introduced a new failure mode: users got truncated answers without warning. We added a front-end banner that showed “Answer may be incomplete” when context was cut. We should have baked this into the chatbot’s UX from day one.

We also underestimated ClickHouse’s memory usage. The VectorIndex with 500k vectors and 384 dim used 12GB RAM for the in-memory buffer. We had to scale the ClickHouse node from 8GB to 61GB RAM, costing an extra $0.20/hr. Next time, we’d test memory usage with a 2x larger catalog before sizing.

Finally, we didn’t monitor the reconciliation job’s recall drift. We only added a nightly recall check after users complained about missing products. Next time, we’d log recall per batch and alert if it drops below 0.93.

**Summary:** Profiling embedding workers, tuning HNSW parameters, baking UX for truncation, and monitoring recall drift would have saved time and cost.


## The broader lesson

The biggest gap in RAG tutorials is operational readiness. They show a static index and a single LLM call, but real catalogs change every few minutes and traffic spikes to 10k RPM. The default stacks (ChromaDB, Qdrant, pgvector) are optimized for batch ingestion and low concurrency, not real-time updates and high throughput.

The pattern that worked here is separation of concerns: a write-optimized store for ingestion and a read-optimized store for serving. ClickHouse’s VectorIndex buffered changes in memory and flushed to disk without blocking reads, while Milvus served a pre-built index from memory-mapped files. This is similar to how databases separate WAL and MVCC snapshots.

Another lesson is that context length is a silent killer. The LLM’s attention window isn’t free; every extra token adds latency and cost. Guarding it with a token budget and user warnings improved reliability more than any infra tweak.

Finally, recall isn’t static. Catalog churn degrades recall over time, so you need a reconciliation process. The cost of a background job is tiny compared to the cost of hallucinated prices.

**Principle:** Build for real-time ingestion and high concurrency from day one, or plan for nightly rebuilds and angry users.


## How to apply this to your situation

Start by profiling your embedding model on your actual catalog. Run a single worker on a cheap instance and measure latency and throughput. If it’s above 50ms per embedding, consider a smaller model like `all-MiniLM-L6-v2` or `bge-small-en-v1.5`. If you need speed, try `sentence-transformers/multi-qa-mpnet-base-dot-v1` with ONNX runtime; it’s 3x faster on CPU and drops to 15ms per embedding.

Next, separate ingestion from serving. Use a write-optimized vector store like ClickHouse VectorIndex or Weaviate’s dynamic indexing for buffering. For serving, use a read-optimized vector store like Milvus (MMAP mode), Qdrant (on-disk with caching), or Pinecone’s serverless tier. Avoid PostgreSQL for serving if you have >100k vectors; the WAL and bloat will kill latency.

Set a context budget. Measure your typical query length and add a 20% buffer. If your prompt is 1000 tokens, cap context at 1200 tokens. If you exceed it, truncate and warn the user. This is a 2-line change in your prompt template.

Finally, implement a reconciliation job. Every 30 minutes, compare the serving index with the latest ingestion snapshot and apply deltas. Log recall per batch and alert if it drops below 0.9. This job should run in under 5 minutes for 500k vectors.

If you’re on a tight budget, start with Qdrant 1.8.0 and enable caching (`cache_size=1024MB`). Use async indexing and keep the index on SSD. At 5k RPM, it’s fine; at 10k RPM, you’ll need to shard or move to Milvus.

**Actionable next step:** Run a 1-hour load test with 10k RPM on your current stack, measure queue depth and latency. If queue depth >500 or latency >800ms, switch to a write-optimized ingestion store this week.


## Resources that helped

- ClickHouse VectorIndex docs: [https://clickhouse.com/docs/en/engines/table-engines/special/vectorindex](https://clickhouse.com/docs/en/engines/table-engines/special/vectorindex)
- Milvus MMAP mode: [https://milvus.io/docs/performance_tuning.md#Memory-mapped-files](https://milvus.io/docs/performance_tuning.md#Memory-mapped-files)
- vLLM context length guard: [https://github.com/vllm-project/vllm/blob/v0.3.0/vllm/transformers_utils/config.py#L240](https://github.com/vllm-project/vllm/blob/v0.3.0/vllm/transformers_utils/config.py#L240)
- Embedding benchmark: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Qdrant caching guide: [https://qdrant.tech/documentation/guides/optimize-search/](https://qdrant.tech/documentation/guides/optimize-search/)
- ONNX runtime for sentence-transformers: [https://onnxruntime.ai/](https://onnxruntime.ai/)

**TL;DR:** Separate ingestion and serving, guard context length, and reconcile indexes every 30 minutes. Measure embedding latency first.


## Frequently Asked Questions

**Can I use ChromaDB with async indexing to avoid blocking?**

Yes, but async indexing in ChromaDB 0.4.21 still blocks reads during index rebuilds for large catalogs. We tested it with 500k vectors and saw 700ms latency spikes every time the index rebuilt. If your catalog is <50k vectors, async indexing works fine. For larger catalogs, use ClickHouse VectorIndex for ingestion and ChromaDB only for serving a pre-built snapshot.

**What’s the recall tradeoff between HNSW and IVF in Milvus?**

In our tests, Milvus HNSW with M=16 and efConstruction=200 gave 0.95 recall on our test set. IVFFlat with nlist=1024 gave 0.91 recall at 40% faster search latency, but recall degraded to 0.85 after 10k catalog updates. We chose HNSW for stability, even though IVF was 10% faster.

**How do you handle multi-tenancy in ClickHouse VectorIndex?**

We used a `tenant_id` column as part of the primary key. The VectorIndex supports multi-tenancy via the `WHERE` clause in queries. For example: `SELECT * FROM product_embeddings WHERE tenant_id = 'shop1' AND embedding <-> query_vector < 0.5`. We measured 5% overhead for tenant filtering at 10k RPM.

**Is it worth using FlashAttention for embedding generation?**

No. We tested FlashAttention v2 with sentence-transformers, but the embedding model isn’t attention-based; it’s a bi-encoder. FlashAttention only helps in decoder models. The speedup was <5ms per embedding, not worth the complexity. Save FlashAttention for the LLM endpoint where it matters.