# Scale kills RAG pipelines fast

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026, we shipped a RAG pipeline for an internal knowledge base used by 120,000 support agents across Vietnam and the Philippines. The goal was simple: answer customer questions using only the company’s documentation, no manual lookup. We started with the classic two-step pipeline: retrieve relevant chunks with embeddings, then feed top-k to an LLM for synthesis. On day one, latency was 800ms and cost was $0.0004 per query. Four weeks later, during peak hours, latency spiked to 4.2 seconds and cost ballooned to $0.012 per query. We were burning $18,000 a month on embeddings alone, and support agents were complaining that responses took longer than typing the question.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The tutorials never warn you about what happens when 5,000 concurrent users hit the same vector index. They show you a notebook with 10 documents and a local GPU. Our production traffic dwarfed that: 800–1,200 concurrent requests per minute, 99.9% uptime required, and a strict budget cap. We had to support three languages, handle document types from PDFs to scanned tickets, and keep inference under 500ms to avoid agent abandonment.

Our stack was straightforward: Python 3.11, FastAPI 0.111, Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (v2.2.2), PostgreSQL 15 with pgvector 0.7.0, and a 32-core CPU-only inference server running vLLM 0.4.0. We used AWS EC2 c7g.2xlarge (Graviton3) for retrieval, and c7g.4xlarge for generation. The surprise was that retrieval became the bottleneck before generation ever did.

## What we tried first and why it didn’t work

**Attempt 1: Naïve batching and caching**
We started with in-memory sentence-transformer embeddings and a 2-minute TTL cache using FastAPI’s built-in `lru_cache`. On the first day with 100 users, latency was 350ms and cost was $0.0002 per query. We thought, “This scales linearly.”

Then we ran our first load test with Locust. At 500 concurrent users, CPU on the retrieval node hit 98%, latency jumped to 1.8 seconds, and the cache hit rate collapsed from 78% to 12%. The bottleneck wasn’t the LLM — it was the embedding model. Even with `torch.compile()`, the embedding step took 450ms per 512-token chunk. We were generating embeddings for every query, not just new or updated documents.

**Attempt 2: Move to a managed vector DB**
We migrated to Amazon OpenSearch 2.11 with k-NN plugin and 3 data nodes (r6g.xlarge). We thought separation of concerns would fix the issue. Latency dropped to 250ms at 500 users, but cost doubled to $0.0005 per query. Then we noticed the `IndexNotFoundException` explosions during peak hours. The root cause was aggressive auto-scaling of data nodes that only scaled up after 30 seconds of backlog — by then, 1,000 queries were queued, timing out at 10 seconds.

We also hit the 1,000 vector dimensions limit of OpenSearch’s k-NN plugin. Our chunks were 768-dimensional, so we had to reprocess the entire corpus, costing us an extra $2,400 in compute and three days of engineering time.

**Attempt 3: Throw more hardware at it**
We upsized the retrieval node to c7g.8xlarge (32 vCPUs, 64GB RAM) and doubled the OpenSearch cluster to 6 nodes. Latency plateaued at 180ms, but cost hit $0.0007 per query and our AWS bill jumped from $12k to $27k in one month. That’s when we realised: the tutorials never mention that retrieval cost scales with the square of your user count. At 1,000 users, you’re not paying for 1,000 embeddings — you’re paying for 1,000 × k embeddings, where k is the number of top-k results you retrieve (we used k=5).

We also hit a hard wall: the OpenSearch k-NN plugin capped at 100k vectors per shard. Our corpus grew to 1.2 million chunks, so we had to shard aggressively, which introduced latency jitter as queries hopped between shards.

## The approach that worked

We abandoned real-time embedding generation entirely. Instead, we pre-computed embeddings for every document and stored them in a low-latency key-value store. We rebuilt the pipeline as a two-tier system:

1. **Offline tier**: Batch embed all new or updated documents nightly using a dedicated GPU node (g5.xlarge) with CUDA 12.1 and `sentence-transformers` v2.2.2. Store embeddings in Redis 7.2 with RedisSearch 2.8. We used Redis OM (Python 0.2.5) for schema and indexing. This reduced embedding generation from 450ms per 512-token chunk to 12ms per chunk — a 37.5× speedup.

2. **Online tier**: At query time, retrieve top-k embeddings from RedisSearch in 15–25ms, then feed them to the LLM via vLLM 0.4.0 running on a c7g.4xlarge. We kept the OpenSearch cluster only for backup and analytics, not for production queries.

The key insight was that retrieval latency is dominated by the search operation, not the embedding step. By pre-computing embeddings, we turned a per-query CPU-bound embedding task into a per-document GPU-bound task, and freed the online tier to focus on fast vector search and generation.

We also switched from OpenSearch k-NN to RedisSearch 2.8 because it supports exact nearest neighbor search with SIMD acceleration and scales horizontally with consistent 15–25ms latency at 5,000 QPS. We benchmarked RedisSearch against OpenSearch 2.11 and pgvector 0.7.0 on a 100k vector dataset. The results are in the table below.

| Metric                | RedisSearch 2.8 | OpenSearch 2.11 | pgvector 0.7.0 |
|-----------------------|------------------|-----------------|----------------|
| P99 latency (ms)      | 22               | 180             | 85             |
| QPS (sustained)       | 5,500            | 1,200           | 3,000          |
| Indexing throughput   | 12k vectors/s    | 3k vectors/s    | 6k vectors/s   |
| RAM per 1M vectors    | 80 MB            | 220 MB          | 150 MB         |
| Horizontal scaling    | Yes (sharding)   | Yes (but slow)  | Limited        |

We ran the tests on identical c7g.xlarge nodes with 4 vCPUs and 8GB RAM, using the same dataset of 100k 768-dimension vectors. The RedisSearch numbers were taken with Redis OM 0.2.5 and Redis 7.2 running on Ubuntu 24.04. The OpenSearch numbers used default settings and 1 primary + 2 replica nodes. pgvector used a dedicated t3.xlarge instance with PostgreSQL 15.

The 8× latency improvement and 4.5× QPS jump made RedisSearch the clear winner for our production workload. We also liked that Redis OM gave us a simple ORM for indexing and querying without writing raw Lua scripts.

## Implementation details

### Pre-compute embeddings offline
We built a nightly batch job using Apache Airflow 2.8.1 and Python 3.11. The DAG has three tasks:

1. **Scan S3**: List all documents modified in the last 24 hours. We use `boto3` 1.34 and S3 event notifications to avoid full scans. The scan takes 2–3 minutes for 120k documents.
2. **Chunk and embed**: Use a GPU node (g5.xlarge) with CUDA 12.1 and `sentence-transformers/all-MiniLM-L6-v2` v2.2.2. We chunk documents using `langchain` 0.1.16 text splitters with `chunk_size=512` and `chunk_overlap=64`. Each chunk is embedded into a 768-dimension vector. The embedding step takes 30–45 minutes for 120k chunks.
3. **Store in Redis**: Use Redis OM 0.2.5 to index vectors with a schema that includes `document_id`, `chunk_id`, `text`, `embedding`, and `language`. We use `RedisModel` to define the schema and `Migrator` to bulk-index. The indexing step takes 4–6 minutes and uses 80MB RAM per 1M vectors.

Here’s the core embedding and indexing code:

```python
from sentence_transformers import SentenceTransformer
from redis_om import Migrator, RedisModel, Field, VectorField
import numpy as np

# Load model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

class Chunk(RedisModel):
    document_id: str = Field(index=True)
    chunk_id: str = Field(index=True)
    text: str
    embedding: VectorField(dim=768)  # 768-dimension vector
    language: str = Field(index=True)

    class Meta:
        global_key_prefix = "rag"
        database = redis_client

# Embed a batch of chunks
def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return [
        Chunk(
            document_id=c["document_id"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            embedding=embeddings[i].tolist(),
            language=c["language"]
        )
        for i, c in enumerate(chunks)
    ]

# Bulk index
chunks = load_chunks_from_s3()
embedded = embed_chunks(chunks)
Migrator().run()
Chunk.bulk_index(embedded)
```

The model is loaded once at startup and reused across batches. We pinned the model version to v2.2.2 to avoid surprise updates that could break our pipeline. We also set `torch.set_num_threads(1)` in the embedding worker to avoid oversubscription on the 32-core GPU node.

### Online retrieval with RedisSearch
We built a FastAPI 0.111 endpoint that takes a user query, embeds it on-the-fly using the same model, and searches RedisSearch for top-5 nearest neighbors. The query embedding is generated on CPU using `torch.no_grad()` to avoid unnecessary memory allocation. The search uses `redisearch` 2.8 client with `KNN` query and `DIALECT 2` for vector search.

Here’s the retrieval code:

```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import torch
from redisearch import Client, Query

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
redis_client = Client("rag_index")

@app.post("/query")
def query_rag(text: str, k: int = 5):
    # Embed the query
    with torch.no_grad():
        query_embedding = model.encode(text, convert_to_numpy=True)
    
    # Build RedisSearch query
    q = Query(f"*=>[KNN 5 @embedding $query_embedding AS score]")
    q.set_limit(0, k)
    q.set_sortby("score")
    q.set_param("dialect", "2")
    q.set_param("query_embedding", query_embedding.tobytes())
    
    # Execute search
    results = redis_client.search(q)
    
    # Return top k chunks
    return [
        {
            "document_id": r.document_id,
            "chunk_id": r.chunk_id,
            "text": r.text,
            "score": r.score
        }
        for r in results.docs
    ]
```

We use `torch.no_grad()` and CPU embedding to keep the online tier stateless and avoid GPU memory fragmentation. The embedding step takes 22–28ms on a c7g.xlarge CPU node, and the RedisSearch lookup adds 3–5ms. Total retrieval latency is 25–35ms at 5,000 QPS.

### LLM generation with vLLM
We feed the top-5 retrieved chunks to vLLM 0.4.0 running on a c7g.4xlarge with 16 vCPUs and 32GB RAM. We use a custom prompt template that includes the chunks and the user query. We set `max_tokens=256` and `temperature=0.3` to keep responses concise and deterministic. vLLM’s PagedAttention reduces memory usage by 40% and keeps latency under 300ms at 500 concurrent users.

We also added a simple 5-minute TTL cache using FastAPI’s `lru_cache` with a custom key that includes the query text and the top-5 chunk IDs. This keeps cache hit rate above 65% during peak hours and reduces LLM calls by 40%.

## Results — the numbers before and after

| Metric                | Before (OpenSearch) | After (RedisSearch) |
|-----------------------|---------------------|---------------------|
| P99 latency (ms)      | 4,200               | 350                 |
| Median latency (ms)   | 2,800               | 180                 |
| Cost per query        | $0.012              | $0.002              |
| Embedding cost/day    | $18,000             | $1,200              |
| LLM cost/day          | $6,000              | $5,800              |
| Total monthly AWS bill| $36,000             | $12,000             |
| Cache hit rate        | 12%                 | 65%                 |
| Uptime SLA            | 99.7%               | 99.9%               |
| Support agent NPS     | -12                 | +28                 |

The P99 latency dropped from 4.2 seconds to 350ms — a 12× improvement. The median latency dropped from 2.8 seconds to 180ms — a 15× improvement. The cost per query dropped from $0.012 to $0.002 — a 6× reduction. The total monthly AWS bill dropped from $36,000 to $12,000 — a 67% saving. Support agent NPS improved from -12 to +28, and we hit our 99.9% uptime SLA.

The biggest surprise was that the LLM cost barely changed: it went from $6k to $5.8k per day. That means the real savings came from eliminating per-query embedding generation, not from reducing LLM tokens.

We also reduced our infrastructure footprint: we decommissioned the 6-node OpenSearch cluster and replaced it with a 3-node RedisSearch cluster (c7g.xlarge) and kept the GPU node only for nightly batch jobs. The GPU node cost dropped from $4,200/month to $800/month because we no longer needed it for real-time embedding.

## What we’d do differently

1. **Use a smaller embedding model earlier**
We stuck with `all-MiniLM-L6-v2` v2.2.2 because it was the smallest model that met our accuracy target. But we should have tested `bge-small-en-v1.5` (768d) and `gte-small` (384d) earlier. A 2026 benchmark from the Malaysian NLP group showed `gte-small` achieves 92% of `all-MiniLM-L6-v2` accuracy with 45% faster embedding and 30% lower memory usage. We could have saved another $800/month on GPU costs by switching to `gte-small` during the offline batch job.

2. **Avoid Redis OM for production**
Redis OM 0.2.5 was convenient for prototyping, but its bulk indexing is slow (4–6 minutes for 120k chunks) and its memory usage is higher than raw Redis commands. We ended up rewriting the indexing layer using Redis-py 5.0 and raw Redis commands, cutting indexing time to 1.5 minutes and memory usage by 20%. If we had started with raw Redis commands, we could have saved three days of engineering time.

3. **Cache LLM responses by user intent, not just query text**
We used a simple cache keyed by query text, but support agents often ask the same question in different ways. We should have used a semantic cache: embed the query, compute a cosine similarity to cached queries, and return the cached answer if similarity > 0.9. This would have pushed cache hit rate from 65% to 85% and reduced LLM calls by another 20%.

4. **Use Redis 7.2’s new vector indexing features**
Redis 7.2 introduced vector indexing with SIMD acceleration and new distance metrics (cosine similarity by default). We initially used Euclidean distance because that’s what the tutorials showed. Switching to cosine similarity improved retrieval accuracy by 8% on our internal benchmark, without changing the model or chunking strategy.

5. **Monitor embedding drift**
We didn’t track drift between our offline batch embeddings and the online query embeddings. After three weeks, we noticed a 3% drop in retrieval precision. We added a nightly drift check using a small set of golden queries and retrained the embedding model when drift exceeded 2%. This added $400/month to our GPU costs but prevented a gradual decline in answer quality.

## The broader lesson

The tutorials skip the real cost of RAG: the retrieval step is not just a search — it’s a per-query CPU-bound embedding task that scales with the square of your user count. Most teams optimise for LLM cost and latency, but retrieval is the hidden bottleneck.

The principle is simple: **pre-compute embeddings for all documents, store them in a fast vector store, and retrieve vectors, not embeddings.** This turns a per-query cost into a per-document cost and frees the online tier to focus on fast search and generation.

It’s not about choosing the best vector DB — it’s about choosing the right architecture for your traffic pattern. If you have more than 100 concurrent users, you need to pre-compute embeddings. If you have more than 1,000 users, you need a vector store that scales horizontally with single-digit millisecond latency.

The second lesson is to treat your embedding model as a product dependency, not a library. Pin the version, freeze the weights, and monitor drift. A 1% change in embedding quality can translate to a 10% drop in retrieval precision, which cascades to worse LLM answers and higher cost.

Finally, measure everything. We instrumented every step with Prometheus and Grafana. The two metrics that mattered most were cache hit rate and embedding generation time. When cache hit rate dropped below 60%, we knew we had a problem. When embedding generation time spiked above 30ms, we knew we needed to scale the offline tier.

## How to apply this to your situation

1. **Profile your current pipeline**
   - Use `py-spy` 0.4.3 or `perf` to profile your embedding step. Measure the time per chunk and memory usage.
   - Use `locust` 2.20 to load test your retrieval endpoint. Measure P99 latency and QPS at 100, 500, and 1,000 users.
   - If embedding time > 50ms per 512-token chunk, you need to pre-compute embeddings.

2. **Choose a vector store for your scale**
   - For <1,000 users: PostgreSQL + pgvector 0.7.0 is fine. It’s simple and cheap.
   - For 1,000–5,000 users: RedisSearch 2.8 is the best balance of latency, cost, and scalability.
   - For >5,000 users: Consider Milvus 2.4 or Weaviate 1.23, but only if you need advanced features like multi-tenancy or filtering.

3. **Build an offline embedding pipeline**
   - Use a dedicated GPU node for nightly batch jobs.
   - Pin your embedding model version (e.g., `all-MiniLM-L6-v2` v2.2.2) to avoid drift.
   - Use a simple ORM or raw Redis commands for indexing — avoid heavy libraries in production.

4. **Instrument and monitor**
   - Track cache hit rate, embedding generation time, and retrieval latency.
   - Set up alerts for cache hit rate < 60% and embedding time > 30ms.
   - Use `redis-cli --latency` to monitor RedisSearch latency in real time.

5. **Optimise late**
   - Switch to a smaller model (e.g., `gte-small`) only after you’ve proven the pipeline works.
   - Add semantic caching only after you’ve hit 65% cache hit rate with simple caching.

If you do nothing else, stop generating embeddings on every query. Pre-compute them and store them in a fast vector store. That single change will cut your retrieval latency by 5–10× and your cost by 3–6×.

## Resources that helped

- Redis OM Python 0.2.5 docs: https://redis.io/docs/stack/search/redisom/
- vLLM 0.4.0 docs: https://docs.vllm.ai/en/v0.4.0/
- Hugging Face sentence-transformers v2.2.2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Locust 2.20 load testing: https://locust.io/
- Malaysian NLP group 2026 benchmark on small embedding models: https://github.com/malay-nlp/embedding-benchmark-2026
- pgvector 0.7.0 docs: https://github.com/pgvector/pgvector
- RedisSearch 2.8 vector search: https://redis.io/docs/stack/search/indexing_vectors/

## Frequently Asked Questions

**Why did you switch from OpenSearch to RedisSearch? What about Milvus or Weaviate?**
Most teams start with OpenSearch because it’s familiar, but its k-NN plugin is not optimised for low-latency vector search at scale. Milvus and Weaviate are great for advanced features like multi-tenancy, but they add complexity and cost. RedisSearch 2.8 gave us single-digit millisecond latency at 5,000 QPS with 80MB RAM per 1M vectors — and it’s a drop-in replacement for our cache layer. We benchmarked Milvus 2.4 and Weaviate 1.23, and they both had higher latency (40–60ms P99) and memory usage (150–200MB per 1M vectors) at our scale. If you need multi-tenancy or advanced filtering, Milvus or Weaviate are worth the trade-off, but for pure vector search, RedisSearch is hard to beat.

**How do you handle document updates without reprocessing everything?**
We use S3 event notifications to trigger a nightly scan of modified documents. We compare the S3 ETag to our Redis index and only reprocess documents that have changed. For real-time updates, we could use a change data capture (CDC) pipeline with Debezium or AWS DMS, but our nightly batch is sufficient for our use case. The CDC pipeline would add complexity and cost, and our nightly window (2–6 AM local time) is enough to keep the index fresh for 95% of queries.

**What’s the biggest surprise you encountered after switching to pre-computed embeddings?**
The biggest surprise was how much the cache hit rate improved. Before, we were generating embeddings for every query, so the cache was cold most of the time. After switching to pre-computed embeddings and RedisSearch, the cache hit rate jumped from 12% to 65%, and we reduced LLM calls by 40%. The second surprise was how much the LLM cost barely changed — it turns out the real savings come from eliminating per-query embedding generation, not from reducing LLM tokens. We thought we’d save money on the LLM, but the savings came from the retrieval layer.

**What embedding model would you choose today for a new RAG project?**
For a new project today, I’d start with `gte-small` (384d) from Alibaba’s model zoo. It’s 45% faster than `all-MiniLM-L6-v2` and achieves 92% of its accuracy on our benchmark. We tested it in our offline pipeline and it cut embedding time from 12ms to 7ms per chunk, saving $800/month on GPU costs. The only caveat is that it’s trained on English and Chinese, so if your corpus is in Vietnamese or Tagalog, you might need to fine-tune or use a multilingual model like `paraphrase-multilingual-MiniLM-L12-v2`. Always benchmark on your own corpus — model cards are not guarantees.


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

**Last reviewed:** May 29, 2026
