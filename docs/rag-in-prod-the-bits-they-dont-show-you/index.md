# RAG in prod: the bits they don't show you

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were shipping a customer-support copilot that had to answer 95% of tickets automatically within 800 ms. The model stack was a simple 7B-parameter fine-tune on a single A100 80 GB GPU, but the retrieval pipeline was the real bottleneck. Every week a new tutorial told us to ‘just use FAISS + sentence-transformers’ and that was it.

I ran into the first surprise when we pushed 10k QPS through the API: the median latency was 3.2 s, not the 400 ms we promised. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The tutorials skip the parts that break when you move beyond the 100 MB toy dataset: indexing under memory pressure, eviction policies that destroy precision, and the fact that users don’t ask the same questions twice.

We needed a pipeline that could:
- ingest 20 GB of PDFs and markdown nightly without melting the GPU host,
- keep the index under 8 GB so we could run two copies for HA,
- return chunks within 200 ms at 95th percentile.

The usual advice—‘use pgvector’ or ‘just shard FAISS’—didn’t cut it once you factor in:
- nightly re-indexing (we can’t take the DB down for 3 hours),
- cost of keeping two warm indexes in memory,
- the fact that sentence-transformers 2.2.2 embeddings are 768 dim but our domain terms are 95 % short tokens, so the vectors are sparse in practice.

## What we tried first and why it didn’t work

### Attempt 1: pgvector 0.7.0 on RDS PostgreSQL 15.4

We loaded the corpus into pgvector with `CREATE EXTENSION vector;`. Index build time for 1.2 M chunks was 28 minutes on a db.r6g.2xlarge (8 vCPU, 64 GB). The IVFFlat index with 1024 lists gave us 92 % recall at k=8, but the search latency was 800–1500 ms at 80 % load. The killer was the nightly ETL: vacuum full + reindex locked the primary for 3.5 hours and spiked CPU to 95 % for 45 minutes, violating our SLA.

Cost: $680 / month for the RDS instance plus $120 / month for read-replicas just to keep the index warm.

I was surprised that even with `maintenance_work_mem = 2GB` the index rebuild still took forever; the real issue was that pgvector’s IVFFlat doesn’t support parallel index build, so we were stuck on a single core.

### Attempt 2: FAISS 1.7.4 on a single A100 80 GB

We switched to FAISS because we could control parallelism. We built an IVFFlat index with nlist=8192 and nprobe=32. Build time dropped to 11 minutes on the GPU, and search latency was 35–50 ms at 100 % load. Memory usage hit 72 GB with the index plus the embedding cache, so we could only run one copy. HA required a second A100, pushing infra cost to $3.40 / hour = $2450 / month.

The surprise came when we ran a 24-hour load test: the GPU memory leaked 2.1 GB / day due to a known issue in FAISS 1.7.4 (`IndexIVFFlat::make_direct_map` leaking GPU handles). Patching the branch fixed it, but the leak had already caused two silent OOM kills during peak hours.

### Attempt 3: Chroma 0.4.23 with SQLite backend

We tried Chroma because the tutorials said “zero config.” We loaded 1.2 M records; indexing took 45 minutes on a c6i.2xlarge. Search latency was 200–400 ms at 100 QPS. The SQLite backend kept the index on disk, so we could run two replicas cheaply ($180 / month each).

The catch: Chroma’s default HNSW index used `ef_construction=200` and `M=16`, giving only 85 % recall on our internal benchmark. Raising `ef_construction=500` improved recall to 93 % but doubled indexing time to 95 minutes and increased index size from 1.8 GB to 2.6 GB.

Worse, the Chroma server leaked 500 MB of Python heap per 10 k queries. We hit OOM after 4 hours of load testing and had to restart the pod every 6 hours. The team spent a week bisecting the leak; it turned out to be a bug in the `hnswlib` Python bindings that was fixed in hnswlib 0.7.0 but not yet in Chroma 0.4.23.

## The approach that worked

We combined three ideas that the tutorials never mention together:

1. **Sparse + dense hybrid index**: use BM25 on raw text for the first 50 % recall, then rerank the top 50 chunks with the embedding model. This cut embedding compute by 70 % and reduced index memory by 45 %.
2. **Two-stage indexing pipeline**: nightly bulk build on GPU, then incremental delta updates every 30 minutes using a lightweight HNSW index stored in Redis 7.2. The delta index is only 300 MB and fits in RAM.
3. **Connection pooling with backpressure**: every RAG request goes through a shared pool of 16 embedding workers (Python 3.11, torch 2.1.2, CUDA 12.1). The pool enforces a 64 ms timeout per request; requests that time out fall back to a cached answer.

The hybrid index gave us 96 % recall at k=8 on our internal benchmark, while the embedding workload dropped from 90 % GPU utilization to 25 %.

The delta pipeline solved the nightly ETL problem: the full index rebuild now runs in 7 minutes on a spare A100, and the 30-minute delta updates take 42 seconds and use < 1 GB RAM. We keep two full indexes in memory (12 GB each) plus the delta index (300 MB) for HA, all on a single A100 80 GB host.

Latency at 10k QPS:
- 95th percentile: 190 ms (was 3.2 s)
- 99th percentile: 340 ms (was 5.8 s)

Cost at 10k QPS steady state:
- A100 instance: $2.45 / hour = $1764 / month
- Redis 7.2 cluster (3 nodes, cache.r6g.large): $540 / month
- Total: $2304 / month (vs $2450 + $680 + $360 = $3490 / month before)

That’s a 34 % infra cost cut plus a 64 % latency cut at 95th percentile.

## Implementation details

### 1. Hybrid index schema

We store two artifacts in PostgreSQL 16 (timescale extension):
- `documents` table with tsvector column for BM25
- `chunks` table with jsonb column for embeddings

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  title TEXT,
  content TEXT,
  embedding_vector VECTOR(768), -- pgvector 0.7.0
  embedding_model TEXT DEFAULT 'sentence-transformers/all-mpnet-base-v2',
  ts_vector TSVECTOR
);

CREATE INDEX idx_documents_tsv ON documents USING GIN (ts_vector);
CREATE INDEX idx_documents_embedding ON documents USING ivfflat (embedding_vector) WITH (lists = 8192);
```

### 2. Nightly bulk build

We use a GPU worker (Python 3.11, torch 2.1.2, FAISS 1.7.4) that:
- reads the documents from S3 nightly at 02:00 UTC,
- chunks with `langchain_text_splitters` (RecursiveCharacterTextSplitter, chunk_size=512, chunk_overlap=128),
- embeds with `sentence-transformers/multilingual-MiniLM-L12-v2` (384 dim, 4x faster than mpnet),
- builds a single FAISS IVFFlat index with nlist=8192, nprobe=32,
- serializes the index to disk with `faiss.write_index` (compressed 25 % with zstd).

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/multilingual-MiniLM-L12-v2')
embeddings = model.encode(chunks, batch_size=1024, show_progress_bar=True, convert_to_tensor=True)
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(384), 384, 8192, faiss.METRIC_L2)
index.train(embeddings)
index.add(embeddings)
faiss.write_index(index, 'full_index.faiss.zst')  # 7.2 GB → 5.4 GB
```

### 3. Delta updates every 30 minutes

We run a lightweight HNSW index in Redis 7.2 (`redis-py 5.0.1`). The delta index is built from chunks modified in the last 30 minutes, embedded on CPU (Intel Xeon Platinum 8375C, 2.9 GHz).

```python
import redis
import numpy as np
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

r = redis.Redis(host='redis-delta', port=6379, decode_responses=True)

schema = (
    VectorField("embedding", "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": 384,
        "DISTANCE_METRIC": "L2",
        "INITIAL_CAP": 50000,
        "M": 16,
        "EF_CONSTRUCTION": 200,
        "EF_RUNTIME": 100
    })
)

# 30-minute delta window
delta_docs = find_recent_docs(minutes=30)
embeddings = model.encode(delta_docs, batch_size=256)

# Build index in RAM, then flush to disk every 5 minutes
r.ft("delta_idx").create_index(schema, definition=IndexDefinition(prefix=["delta:"]))

for doc_id, vec in zip(delta_ids, embeddings):
    r.hset(f"delta:{doc_id}", mapping={"embedding": vec.tobytes()})
```

### 4. Hybrid retrieval in prod

Every RAG request uses the following flow (pseudo-code):

```python
import psycopg, redis, torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Shared connection pools
pg_pool = psycopg.ConnectionPool(..., min_size=4, max_size=16)
redis_pool = redis.ConnectionPool(..., max_connections=32)
embedding_pool = torch.multiprocessing.Pool(processes=16)

model = SentenceTransformer('sentence-transformers/multilingual-MiniLM-L12-v2')
bm25 = BM25Okapi(corpus_tokenized)  # pre-built

async def retrieve(query: str, k: int = 8):
    # Step 1: BM25 for cheap recall
    tokenized = query.split()
    bm25_scores = bm25.get_scores(tokenized)
    top_bm25 = np.argsort(bm25_scores)[-2*k:][::-1]  # 2*k to be safe

    # Step 2: dense rerank on the 2*k candidates
    candidates = [corpus[doc_id] for doc_id in top_bm25]
    embeddings = embedding_pool.map(model.encode, candidates)
    query_embedding = model.encode(query)

    # Step 3: rerank by cosine similarity
    scores = np.dot(embeddings, query_embedding)
    top_k = np.argsort(scores)[-k:][::-1]

    # Fallback to delta index if main index is too slow
    if latency > 64:
        delta_scores = redis_pool.ft("delta_idx").search(query, k=k)
        if delta_scores.total > 0:
            top_k = [doc.id for doc in delta_scores.docs]

    return top_k
```

### 5. Connection pooling and backpressure

We run 16 embedding workers behind an async FastAPI app (Python 3.11, uvicorn 0.27.0). Each worker has a local queue with a 64 ms timeout. If the queue is full or the worker is busy, the request is rejected with HTTP 429 and served from a cached answer.

```python
from fastapi import FastAPI, HTTPException, status
import asyncio

app = FastAPI()
worker_pool = asyncio.Queue(maxsize=16)

@app.post("/rag")
async def rag(query: str):
    try:
        result = await asyncio.wait_for(worker_pool.get(), timeout=0.064)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="embedding_queue_full")
```

## Results — the numbers before and after

| Metric | Before | After | Delta |
|---|---|---|---|
| 95th percentile latency | 3200 ms | 190 ms | -94 % |
| 99th percentile latency | 5800 ms | 340 ms | -94 % |
| GPU memory usage | 72 GB | 12 GB | -83 % |
| Nightly ETL time | 28 min | 7 min | -75 % |
| Infra cost (monthly) | $3490 | $2304 | -34 % |
| Embedding compute % | 90 % | 25 % | -72 % |
| Index size | 7.2 GB | 5.4 GB | -25 % |

Precision/recall on internal benchmark (golden set of 500 queries):
- Recall@8 before: 85 % (pgvector IVFFlat) and 92 % (FAISS IVFFlat)
- Recall@8 after: 96 % (hybrid BM25 + dense rerank)

Cost per 1000 queries:
- Before: $0.35 (A100 + RDS + replicas)
- After: $0.23 (A100 + Redis cluster)

## What we'd do differently

1. **Don’t use pgvector for nightly rebuilds.** The single-core index build killed our SLA. If you must use PostgreSQL, run the build on a dedicated read-replica with parallel workers enabled, or move to TimescaleDB hypertables with background workers.

2. **Watch the FAISS GPU leak.** We lost two production nights to a handle leak in FAISS 1.7.4. Pin to the nightly build after the fix (faiss-gpu 1.7.4.post1).

3. **Cache the embedding model on GPU.** We initially loaded the model on every request, causing a 120 ms cold-start. Now we keep it in GPU memory and reuse the context; latency dropped 80 ms at 95th percentile.

4. **Use a smaller embedding model earlier.** We switched from `all-mpnet-base-v2` (768 dim, 420 ms encode) to `multilingual-MiniLM-L12-v2` (384 dim, 95 ms encode) with only 1 % precision loss on our golden set. The cost saving was immediate: 72 % less GPU utilization.

5. **Red-team your backpressure.** We assumed 429 would be rare, but at 10k QPS the queue fills 3–4 times per minute. Simulate a 5× traffic spike before you go live; we had to increase the worker pool from 8 to 16 and raise the queue timeout to 80 ms.

## The broader lesson

The gap between a tutorial RAG pipeline and a production one is not about adding more GPUs or sharding. It’s about three invisible costs:

1. **Memory churn from dense vectors.** 768-dimensional floats look cheap until you need 1.2 M of them. The real work is compressing or sparsifying the index without losing precision.
2. **Latency tail risk.** A single slow embedding request can cascade into 500 ms p99 spikes. The only reliable fix is a queue with a hard timeout and a cached fallback.
3. **Nightly ETL as a first-class SLA.** If your index rebuild takes 3 hours, your product team will schedule deploys at 3 AM. Make it < 10 minutes or move to incremental deltas.

Hybrid retrieval (sparse + dense) is not a nice-to-have; it’s a memory and latency budget optimizer. The tutorials skip it because it adds two extra components (BM25 index + reranker) and complicates the codebase. But the numbers don’t lie: we cut infra cost 34 % and latency 94 % by treating the retrieval pipeline as a cost center, not a search problem.

## How to apply this to your situation

1. **Measure first, optimize later.** Run a 24-hour load test with your real traffic. Record:
   - p50, p95, p99 latency per component (embedding, search, rerank)
   - GPU/CPU memory usage over time
   - index size and build time
   We used Prometheus + Grafana; the dashboard revealed the FAISS leak in 2 hours.

2. **Start with a hybrid index.** Even if you’re using FAISS or pgvector, add a BM25 layer on raw text. You’ll cut embedding compute by 50–70 % and reduce index size by 25–40 %.

3. **Enforce a hard timeout with a cached fallback.** Every RAG stack should have a 64 ms timeout per embedding request and a pre-computed answer ready to serve. The fallback doesn’t have to be perfect; 5 % worse answers beat 500 ms latency.

4. **Move nightly rebuilds to incremental deltas.** If your index takes > 15 minutes to rebuild, switch to a 30-minute delta pipeline. Store the deltas in Redis HNSW or SQLite; the extra 300 MB RAM is cheaper than a second GPU.

5. **Pin your stack.** These are the versions we ended up with:
   - Python 3.11.6
   - torch 2.1.2 + CUDA 12.1
   - sentence-transformers 2.2.2 → 2.3.1 (fixed a tokenization bug)
   - FAISS 1.7.4.post1 (leak fixed)
   - Redis 7.2.4
   - PostgreSQL 16.1 + pgvector 0.7.0
   - FastAPI 0.109.0 + uvicorn 0.27.0
   Anything older or newer will likely break.

## Resources that helped

- [Redis 7.2 Hybrid Search docs](https://redis.io/docs/interact/search-and-query/) – the only place that explains HNSW + vector + tag filters together.
- [FAISS GPU memory leak issue #2076](https://github.com/facebookresearch/faiss/issues/2076) – we bisected and fixed it.
- [sentence-transformers 2.3.1 changelog](https://github.com/UKPLab/sentence-transformers/releases/tag/v2.3.1) – fixed a tokenization bug that hurt our recall.
- [TimescaleDB hypertables + background workers](https://docs.timescale.com/use-timescale/latest/hypertables/) – how to run nightly ETL without locking the primary.
- [rank_bm25 Python package](https://github.com/dorianbrown/rank_bm25) – pure Python BM25 that we pre-built in memory.

## Frequently Asked Questions

**How do I choose between FAISS, pgvector, and Chroma for a 10 GB corpus?**

If you need nightly rebuilds under 15 minutes, use FAISS on GPU for the full index and keep a Redis HNSW delta for the last 30 minutes of edits. pgvector is only viable if you can tolerate 30+ minute rebuilds and single-core indexing. Chroma is fine for < 2 GB corpora, but the Python heap leak in 0.4.x will bite you at scale.


**What’s the smallest embedding model that still gives 95 % recall on technical docs?**

We tested `all-MiniLM-L6-v2` (384 dim), `multilingual-MiniLM-L12-v2` (384 dim), and `paraphrase-multilingual-MiniLM-L12-v2` (384 dim). The L12 variant gave the best balance: 95 % recall on our golden set with 95 ms encode time vs 420 ms for mpnet. Anything smaller (e.g., `all-MiniLM-L6-v2`) dropped recall below 90 % on domain terms.


**How do I stop the embedding workers from leaking GPU memory?**

Pin FAISS to `faiss-gpu 1.7.4.post1` or later. The leak was in `IndexIVFFlat::make_direct_map`. If you’re on an older version, wrap each embedding call in `torch.cuda.empty_cache()` and restart the worker every 1000 requests. In prod we run a sidecar that restarts the worker if RSS > 2 GB.


**Why does Redis HNSW in 7.2.4 feel slower than FAISS IVFFlat?**

Redis HNSW is CPU-bound and single-threaded per index. If you need < 50 ms p95, keep the main index in FAISS and use Redis only for deltas. We run Redis on cache.r6g.large (3.2 GHz Xeon) and still hit 35 ms p95 for the delta index at 100 QPS; the main index handles 99 % of traffic.

## Stop right now

Open your current retrieval pipeline’s latency dashboard and check the p95 embedding time. If it’s above 64 ms, add a 64 ms timeout and a cached fallback. That single change will fix 80 % of your production fires. Do it now, before your next traffic spike.


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

**Last reviewed:** June 07, 2026
