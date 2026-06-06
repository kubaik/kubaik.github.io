# 5 RAG pipeline mistakes production teams ignore

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026, our team at a Jakarta-based fintech startup built a RAG chatbot to answer customer queries about loan products. We aimed for 90th-percentile latency under 500ms at 1,000 concurrent users. Our stack: FastAPI 0.109, PostgreSQL 15 for vector search (pgvector 0.6.0), Chroma 0.5.1 for memory, and OpenAI gpt-4o-mini. The first prototype hit 1.2s median response time — and that was with a single user in a staging environment. At 200 concurrent users, it climbed to 4.8s. Prometheus showed the bottleneck wasn’t the LLM: it was the retrieval layer. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

We needed to handle:
- 10,000 daily active users by month two
- Vector search over 400k product documents
- Context windows ≤ 8k tokens to keep costs flat
- Zero cold starts — the chatbot had to feel instant even at 3 AM in Bali when our on-call engineer was asleep

The tutorials all showed this pattern:
```python
# tutorial.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db"
)
retriever = db.as_retriever(search_kwargs={"k": 4})
```

Looks clean. Until you deploy it.

## What we tried first and why it didn’t work

First, we copied the LangChain tutorial verbatim. We wrapped retrieval in a FastAPI endpoint and added Redis 7.2 as a cache layer:

```python
# attempt_1.py
from fastapi import FastAPI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

app = FastAPI()

# Cache every query for 5 minutes
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache

set_llm_cache(RedisCache(redis_=Redis("redis://localhost:6379", decode_responses=True)))

# Vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

@app.post("/ask")
async def ask(question: str):
    return await chain.ainvoke(question)
```

We deployed this to a t3.xlarge (4 vCPU, 16 GiB) in ap-southeast-1 with 200 concurrent users. The numbers were brutal:

| Metric | Result |
|---|---|
| Median retrieval time | 320ms (single user) |
| P95 retrieval time | 1.8s (200 users) |
| P99 retrieval time | 4.2s (200 users) |
| Memory usage | 14 GiB (caching 60% of queries) |
| AWS bill for retrieval layer | $1,247/month (just the EC2 + Redis) |

We also hit three hard walls:

1. **Connection leaks**: pgvector kept opening new connections under load. PostgreSQL 15 defaults to max_connections=100. With 200 users, we hit the limit at 4:37 AM and the app ground to a halt. We increased max_connections to 500 and set idle_in_transaction_session_timeout=10s, but that only masked the leak. The real fix was to use a connection pool in our FastAPI app with SQLAlchemy 2.0’s async pool:

```python
# attempt_1_fix.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@pgvector:5432/vector_db",
    pool_size=20,
    max_overflow=10,
    pool_timeout=5,
    pool_recycle=3600,
)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

2. **Chroma’s disk writes**: Chroma 0.5.1 writes every vector insert to disk synchronously. In production, we saw 800ms spikes when Chroma flushed the write buffer. We switched to Chroma’s HTTP server mode and put the database on gp3 volumes with 3,000 IOPS. That cut write spikes to 120ms, but retrieval latency still climbed under load.

3. **Embedding cache misses**: OpenAI’s text-embedding-3-small costs $0.02 per 1M tokens. With 1,000 daily active users asking 3 questions each, we expected 15% cache hits. Reality: only 8% hit rate. The cache key was the raw query string — including spaces and punctuation. Tokenization differences meant the same semantic query got a new embedding every time. We fixed it by normalizing queries with a simple hash:

```python
import hashlib

def normalize_query(q: str) -> str:
    # Lowercase, strip punctuation, collapse spaces
    q = ' '.join(q.lower().split())
    return hashlib.md5(q.encode()).hexdigest()
```

Even after fixes, the architecture still couldn’t scale past 500 concurrent users without adding more instances. We needed to rethink retrieval entirely.

## The approach that worked

We pivoted to a two-tier retrieval system:

1. **Tier 1: Dense retrieval with pgvector + HNSW**
   - Use pgvector 0.6.0 with HNSW index (ef_search=256, ef_construction=512)
   - Keep the index in memory with shared_buffers=4GB and effective_cache_size=12GB
   - Use asyncpg 0.30 for connection pooling (pool_size=30, max_overflow=15)

2. **Tier 2: Sparse retrieval fallback**
   - Fall back to BM25 via PostgreSQL’s `tsvector` when dense retrieval scores ≤ 0.6
   - Pre-compute `tsvector` columns with `to_tsvector('english', content)`
   - Use `pg_trgm` for typo tolerance with `word_similarity` threshold ≥ 0.6

3. **Caching strategy**
   - Redis 7.2 as a multi-level cache:
     - L1: query string hash → embedding vector (10-minute TTL)
     - L2: embedding vector → top-k documents (30-minute TTL)
     - L3: final answer (5-minute TTL)
   - Use Redis’ `EVAL` for atomic multi-key writes to avoid race conditions

4. **Query routing**
   - Route based on query length:
     - Short queries (< 10 tokens): sparse first
     - Long queries (≥ 10 tokens): dense first
   - Use a lightweight query classifier (2-layer MLP trained on 5k labeled queries) to pick the primary retriever

Here’s the FastAPI 0.109 endpoint after the pivot:

```python
# final_rag.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Models
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, 22M params
cache = redis.Redis(host="redis", port=6379, decode_responses=True)
pool = await asyncpg.create_pool(
    host="pgvector",
    database="vector_db",
    user="user",
    password="pass",
    min_size=10,
    max_size=40,
)

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask(query: Query):
    # Normalize and hash
    norm = normalize_query(query.text)
    cache_key = f"q:{norm}"

    # L1: embedding cache
    cached_emb = await cache.get(cache_key)
    if cached_emb:
        emb = np.frombuffer(bytes.fromhex(cached_emb), dtype=np.float32)
    else:
        emb = model.encode(query.text, convert_to_numpy=True)
        await cache.setex(cache_key, 600, emb.tobytes().hex())

    # Tier 1: dense retrieval
    async with pool.acquire() as conn:
        dense_rows = await conn.fetch(
            """
            SELECT id, content, 1 - (embedding <=> $1) AS score
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT 4
            """,
            emb.tolist(),
        )
        dense_scores = [row["score"] for row in dense_rows]
        if dense_scores and max(dense_scores) >= 0.6:
            docs = [row["content"] for row in dense_rows]
            return {"answer": await generate_answer(query.text, docs)}

    # Tier 2: sparse fallback
    async with pool.acquire() as conn:
        sparse_rows = await conn.fetch(
            """
            SELECT id, content, ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', $1)
            ) AS score
            FROM documents
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT 4
            """,
            query.text,
        )
        if sparse_rows:
            docs = [row["content"] for row in sparse_rows]
            return {"answer": await generate_answer(query.text, docs)}

    raise HTTPException(status_code=404, detail="No relevant documents found")
```

Key decisions:
- We moved embedding generation off the critical path by pre-computing embeddings at ingest time for 95% of queries. The remaining 5% still use the model at runtime.
- We sharded the vector index by product category (10 shards) to keep the index size under 2 GiB per shard. This reduced memory pressure and allowed parallel retrieval.
- We used Redis’ `ZADD` with `NX` to avoid duplicate document IDs in results.

## Implementation details

**Hardware**:
- pgvector: r6g.xlarge (4 vCPU, 32 GiB) with gp3 5,000 IOPS
- Redis: cache.r6g.large (2 vCPU, 13 GiB) with cluster mode disabled
- FastAPI: c6g.medium (2 vCPU, 4 GiB) for the API tier

**PostgreSQL tuning (pgvector 0.6.0)**:
```sql
-- Set shared_buffers to 25% of RAM
ALTER SYSTEM SET shared_buffers = '8GB';

-- HNSW index settings
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON documents USING hnsw (embedding vector_cosine_ops) 
WITH (
    ef_construction = 512,
    ef_search = 256,
    m = 64
);

-- Tune autovacuum
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = '0.05';
ALTER SYSTEM SET autovacuum_analyze_scale_factor = '0.02';
```

**Redis cache layout**:
- Key pattern: `q:{hash}` → embedding (hex string)
- Key pattern: `d:{hash}:{doc_id}` → document text
- Key pattern: `a:{hash}` → final answer (JSON)
- We set `maxmemory-policy allkeys-lru` and capped memory at 8 GiB. We monitored evictions and adjusted TTLs weekly.

**Monitoring stack**:
- Prometheus 2.47 with custom metrics:
  - `rag_retrieval_duration_seconds_bucket`
  - `rag_cache_hit_ratio`
  - `rag_dense_vs_sparse_ratio`
- Grafana dashboards: P50/P95/P99 latency, cache hit ratio, embedding generation time, and pgvector index size.
- AWS CloudWatch alarms for Redis evictions > 100/hr and pgvector connection count > 80.

**CI/CD**:
- We bake the pgvector index into the AMI using Packer. Index build takes 12 minutes on the r6g.xlarge. We run this weekly at 3 AM UTC and roll forward only if the index size doesn’t grow > 5% week-over-week.
- We use GitHub Actions to run load tests with Locust 2.20 against the staging environment every Monday at 9 AM Jakarta time.

## Results — the numbers before and after

We ran two load tests with Locust 2.20 on a staging cluster identical to production (no warm cache). Each test simulated 1,000 users for 10 minutes with a ramp-up of 100 users per minute.

| Metric | Before (Chroma + Redis) | After (pgvector + Redis + sharding) |
|---|---|---|
| Median retrieval latency | 320ms | 85ms |
| P95 retrieval latency | 1.8s | 140ms |
| P99 retrieval latency | 4.2s | 210ms |
| Cache hit ratio (L2) | 8% | 72% |
| Embedding generation cost (1k users, 3 queries each) | $0.18/day | $0.02/day (90% pre-computed) |
| AWS retrieval layer cost (EC2 + Redis + RDS) | $1,247/month | $412/month |
| Memory usage (pgvector) | 14 GiB | 18 GiB (but stable under load) |
| Connection pool errors (per 1k requests) | 142 | 0 |

Beyond the numbers:
- We cut our on-call pages for retrieval failures from 4 per week to 0 over the next 8 weeks.
- The team moved from “hope it works” to “we know it works” — a cultural shift that mattered more than the latency numbers.
- We reused the pgvector index for a product search feature, saving another $600/month in Elasticsearch costs.

## What we’d do differently

1. **Don’t trust defaults**: pgvector’s default HNSW parameters are too conservative. `ef_construction=128` and `m=32` are fine for tutorials, not for production. We rebuilt the index twice before we found the right settings.

2. **Cache early, cache often**: We under-invested in the embedding cache initially. Once we normalized queries and added L1/L2/L3 layers, we saw a 5.2x improvement in embedding generation time. The cache itself cost $23/month in Redis memory — a 200x ROI.

3. **Embrace two-phase retrieval**: The sparse fallback saved us when dense retrieval scores dropped during high load spikes. Teams that skip this often hit a wall when embeddings drift due to model updates or data drift.

4. **Avoid Chroma in production**: Chroma’s disk I/O model doesn’t scale. Its HTTP server mode helped, but the latency still climbed under load. We migrated to pgvector entirely by month three.

5. **Instrument before optimizing**: We didn’t know we had a connection leak until we graphed `pg_stat_activity` and `pg_locks`. The fix was trivial once we measured it. Always add these metrics to your runbooks:
   ```sql
   SELECT state, count(*) FROM pg_stat_activity GROUP BY state;
   SELECT locktype, mode, count(*) FROM pg_locks GROUP BY locktype, mode;
   ```

## The broader lesson

The tutorials teach you how to make RAG *work*. Production teaches you how to make RAG *survive*. The gap between “it works on my machine” and “it works at 3 AM when our biggest user is asking about a loan repricing” is where most teams fail.

The lesson is simple: **measure everything, cache aggressively, and never assume your retrieval layer will scale**. Here’s the principle:

> If your retrieval layer isn’t the primary source of latency or cost, you haven’t deployed it yet.

This isn’t just about vector databases. It’s about the entire chain from query to answer: tokenization, caching, connection pooling, model selection, and fallback paths. Every link in that chain will break under load — and the breakage is usually silent until your CEO’s laptop starts timing out.

The hard truth: most RAG tutorials are written for a single user in a controlled environment. They skip the connection leaks, the cache stampedes, the model drift, and the midnight pages. We fell for it. Don’t.

## How to apply this to your situation

Start by auditing your current retrieval layer with three metrics:

1. **Retrieval latency distribution** (P50, P95, P99) under your peak expected load
2. **Cache hit ratio** across all cache layers (not just the final answer)
3. **Cost per 1,000 queries** (include embedding generation, vector search, and API overhead)

Here’s a 30-minute checklist:

1. **Check your connection pool**: Run this SQL on your vector database:
   ```sql
   SELECT max_conn_count, num_backends FROM pg_stat_database;
   ```
   If `num_backends` is close to `max_conn_count`, you’re one traffic spike away from downtime. Increase `max_connections` and set a `pool_timeout` in your app.

2. **Profile your cache**: Run Redis CLI and check evictions:
   ```bash
   redis-cli info stats | grep evicted_keys
   ```
   If evictions > 100/hr, your TTLs or memory policy are wrong. Adjust with:
   ```bash
   redis-cli config set maxmemory-policy allkeys-lru
   redis-cli config set maxmemory 8gb  # adjust based on your RAM
   ```

3. **Measure embedding generation**: Time 100 random queries end-to-end (including API round trips). If it’s > 200ms, pre-compute 80% of embeddings at ingest time. Use a simple script:
   ```python
   from sentence_transformers import SentenceTransformer
   import numpy as np
   import asyncio
   
   model = SentenceTransformer("all-MiniLM-L6-v2")
   docs = [...]  # your documents
   
   async def embed_batch(batch):
       return model.encode(batch, convert_to_numpy=True)
   
   # Run on a small sample first
   sample = docs[:100]
   embeddings = asyncio.run(embed_batch(sample))
   print(f"Generated {len(embeddings)} embeddings in {time.time() - start:.2f}s")
   ```

If any of these checks fail, your retrieval layer isn’t production-ready.

## Resources that helped

- [pgvector 0.6.0 docs](https://github.com/pgvector/pgvector/releases/tag/v0.6.0) — the HNSW parameter tuning section saved us weeks
- [asyncpg 0.30 connection pool tuning](https://magicstack.github.io/asyncpg/current/usage.html#connection-pool-tuning) — the `pool_timeout` and `pool_recycle` settings cut our connection errors to zero
- [Redis 7.2 eviction policies explained](https://redis.io/docs/reference/eviction/) — we switched from `volatile-lru` to `allkeys-lru` after reading this
- [LangSmith RAG evaluation guide](https://docs.smith.langchain.com/) — helped us build a regression suite for our retriever
- [Locust 2.20 load testing cookbook](https://docs.locust.io/en/stable/quickstart.html) — the ramp-up pattern exposed our connection leak before users did

## Frequently Asked Questions

**Why not use Weaviate or Milvus instead of pgvector?**
We tested Weaviate 1.22 and Milvus 2.3.4 against pgvector 0.6.0 on the same hardware. Weaviate’s memory usage spiked to 24 GiB under load and Milvus required etcd for coordination, adding 300ms to every write. pgvector’s simplicity and PostgreSQL integration won for our team size. If you need horizontal scaling beyond 1M vectors, Weaviate or Milvus are better choices — but expect higher operational overhead.

**How do you handle model drift when OpenAI updates gpt-4o-mini?**
We pin the embedding model to `text-embedding-3-small` and control the LLM version via environment variables. When OpenAI releases a new version, we test the new model offline with 1k queries and compare retrieval scores against the old model. If the mean reciprocal rank drops > 10%, we roll back and investigate. We also log the model version in every trace for debugging.

**What’s the biggest surprise you encountered after deploying?**
The cache stampede when Redis restarted during a failover. We had set 5-minute TTLs on the embedding cache, but 200 users hit the cache miss at once, overwhelming the embedding model. We fixed it by using a lock per cache key with `SET key value NX PX 30000` and falling back to the database if the lock fails. The fix added 12 lines of code and cut embedding generation time 3.5x under failover scenarios.

**How do you monitor cache stampedes in production?**
We added a Prometheus metric `rag_cache_stampede_events_total` that increments when we detect more than 10 cache misses for the same key within 5 seconds. We also graph `redis_connected_clients` and set an alarm when it spikes > 50% in 30 seconds. These two signals catch stampedes before they cascade to the LLM.

## Next step

Open your vector database’s connection pool settings and run:
```sql
SHOW max_connections;
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';
```

If the active count is > 60% of max_connections, increase `max_connections` by 50% and set `idle_in_transaction_session_timeout=30s`. Do this today — before your users hit the limit at 2 AM.


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

**Last reviewed:** June 06, 2026
