# RAG pipelines: the load tests fail

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In early 2026 we launched a Vietnamese-language Q&A chatbot for a fintech startup’s 1.2 million users. The goal: cut support tickets by 30% by letting users self-serve answers on loans, fees, and account issues. We picked a standard RAG pipeline—split documents into chunks, embed with `bge-small-en-v1.5`, store in `pgvector 0.7.0`, and query with cosine similarity. Latency targets: < 200 ms p95 for the full stack, including retrieval + generation.

We started with the textbook tutorials. They all showed the same clean flow: ingest, index, retrieve, generate. What they never mentioned was what happens when 15 000 users hammer the chatbot at 11 p.m. during a new loan promotion. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The real problem wasn’t accuracy; it was stability under load. After 100 concurrent queries, PostgreSQL connections saturated at 50, and our `pgvector` index scans slowed from 25 ms to 800 ms. The API error rate hit 8% and stayed there until we restarted the pods. That’s when I knew the tutorials had skipped the production edge cases: connection pooling, chunk sizing, eviction policies, and index fragmentation.

We also discovered our Vietnamese prompts were token-heavy. The first embedding model we used (`bge-small-en-v1.5`) produced 384-dim vectors, but Vietnamese sentences averaged 140 tokens. Each request was pushing 56 kB over the wire just for the embedding payload. With 12 000 daily active users, that added 672 MB/day in unnecessary bandwidth—before we even counted the LLM input tokens.

Finally, the vector store indexing strategy was naive. We started with a single `pgvector` index on the `embedding` column using HNSW with `ef_construction=200`. After two weeks, insert latency for new documents climbed from 120 ms to 2.1 s because the index was rebuilding under every INSERT. Our background job that refreshed the knowledge base every 6 hours locked the table for 4–6 minutes, causing timeout cascades across the API fleet.

By month two we had three hard constraints to satisfy:
- Keep 95th-percentile latency under 200 ms during peak load (15 k concurrent users).
- Hold AWS costs below $1 800/month (we were burning $2 400 on RDS `db.m6g.2xlarge` and three `r7g.xlarge` embedders).
- Keep error rate under 1% during traffic spikes.

The tutorials promised 80% accuracy on the first try. Production asked for 99% uptime and a bill we could justify to the CFO.

## What we tried first and why it didn't work

Our first attempt was to throw more hardware at the problem. We moved from `db.m6g.2xlarge` (4 vCPU, 16 GB) to `db.r6g.4xlarge` (16 vCPU, 128 GB) and doubled the embedder cluster to six `g5.xlarge` instances running `text-embedding-3-small` via SageMaker endpoints. Cost jumped to $3 200/month, but latency crept up instead of down. The issue wasn’t CPU; it was connection exhaustion and index contention.

We also tried sharding the vector store by document type. Each shard got its own `pgvector` table and HNSW index. The idea was to reduce index rebuild locks during knowledge-base refreshes. What we didn’t plan for was the cross-shard joins needed for multi-topic queries. A user asking about "early repayment fees and credit score impact" had to hit three shards. With 15 concurrent users, the round-trip time ballooned to 340 ms p95—exceeding our SLA.

Next we switched the embedding model to `bge-multilingual-gemma2-2b` hosted on vLLM 0.5.0 with tensor parallelism across two A10G GPUs. Accuracy improved from 72% to 84%, but latency rose from 120 ms to 780 ms because the model ran at 3.8 tokens/sec on quantized weights. We also hit a new problem: GPU OOM errors when batching more than 8 requests. Our autoscaler kept thrashing between 0 and 2 pods, causing 1–2% 5xx errors every scaling event.

We then tried Redis 7.2 as a caching layer. We cached embeddings by raw text hash with 5-minute TTL. The cache hit rate was 68%, but the miss penalty was brutal: 320 ms to compute the embedding plus 180 ms to retrieve the top-k from `pgvector`. Worse, Redis memory climbed to 42 GB because we weren’t evicting stale keys aggressively enough. We set `maxmemory-policy allkeys-lru`, but the LRU wasn’t keeping up with our 10 k new documents per day. We had to restart the cluster every 36 hours to reclaim memory.

The final blind alley was query rewriting. We tried expanding user queries with synonyms using a small Vietnamese thesaurus. Synonym expansion increased recall, but also token length. Average prompt length grew from 140 tokens to 205 tokens. Our LLM wrapper (vLLM 0.5.0) started dropping requests because the total token budget (prompt + max_new_tokens=128) exceeded the model’s context window—even though the actual question was tiny.

Each of these attempts solved one symptom while worsening another. The root cause kept shifting: connection pool exhaustion, index rebuild locks, GPU batching limits, cache churn, and prompt bloat. The tutorials never warned us that fixing one layer could break another in subtle ways.

## The approach that worked

We stepped back and treated the RAG pipeline like a distributed system, not a script. The key insight: treat every stage as a resource with limits—connections, memory, GPU batch slots, and disk I/O. We started with four principles:

1. **Bounded concurrency everywhere.** Each stage (embedding, retrieval, generation) has a fixed-size worker pool and a backpressure mechanism.
2. **Lazy index updates.** Never rebuild an index under write load; instead, stream new chunks to a sidecar indexer that merges asynchronously.
3. **Chunk sizing as a first-class constraint.** Vietnamese sentences split into 3–7 words per chunk gave us 78% recall with 128 tokens per prompt. Anything longer added latency without meaningful accuracy gains.
4. **Cost-aware caching.** Cache only the top-k vectors, not raw embeddings, and evict aggressively using a size-based policy capped at 20 GB per node.

Step one was to move the vector store to a read-only replica for queries and a dedicated writer for inserts. The replica ran `pgvector 0.7.0` on a smaller instance (`db.m6g.large`) and used HNSW with `ef_construction=400` and `m=16`. We set `maintenance_work_mem` to 1 GB to speed up index builds. Inserts went to a separate writer (`db.r6g.xlarge`) that rebuilt the index every 6 hours during off-peak (2 a.m.). The rebuild now took 2 minutes instead of 6, and the replica never locked during queries.

Step two was embedding optimization. We switched to `bge-reranker-v2-m3` for retrieval and kept `bge-small-en-v1.5` for the final top-100 rerank. Reranking reduced the prompt token count by 30% because we filtered irrelevant context early. We ran the embedder on CPU with `ONNX Runtime 1.17` and 4 threads, hitting 1.2 k req/sec with 40 ms latency. Cost dropped from $1 100/month (SageMaker) to $80/month (EC2 `c7g.4xlarge`).

Step three was bounded concurrency. We used `Redis 7.2` as a rate-limiting cache with a sliding window of 100 req/min per user. We also capped the embedding worker pool at 8 and the retrieval worker pool at 16. When the pools were full, the API returned a 429 with Retry-After instead of queuing indefinitely. Error rate fell from 8% to 0.3% during spikes.

Step four was prompt sizing. We rewrote the prompt template to include only the top-10 most relevant chunks (1 280 tokens max) and forced the LLM to answer in ≤ 50 tokens. We used `vLLM 0.5.0` with `max_tokens=128` and `temperature=0.1` to reduce variance. The prompt now fit comfortably within the 8 k context window of our 7B model, and the generation stage stayed below 180 ms p95.

Step five was async knowledge updates. We moved the document ingestion pipeline to an SQS queue fed by S3 event notifications. A Lambda (`python 3.11`) split chunks, computed embeddings, and wrote them to a staging table. A separate ECS service (`Fargate 1.4`) merged the staging table into the main `pgvector` table every 6 hours using a merge statement that avoided full index rebuilds. The merge took 30 s and added < 10 ms latency to queries.

We also added a circuit breaker in the API layer: after 5 consecutive 5xx errors, the endpoint returned a cached default response for 30 s. This prevented thundering herds during deployments and index rebuilds.

The result was a pipeline that looked deceptively simple on paper but survived Black Friday traffic without human intervention.

## Implementation details

Here’s the stack we ended up with after six weeks of tuning:

| Component               | Tool / Version           | Instance / Config                          | Monthly cost |
|-------------------------|--------------------------|--------------------------------------------|--------------|
| Document splitter       | `langchain-text-splitters 0.2.1` | Chunk size 256, overlap 32, Vietnamese tokenizer `vinai/phobert-base` | $12          |
| Embedding               | `ONNX Runtime 1.17`      | `c7g.4xlarge` (4 vCPU Graviton), 4 threads | $80          |
| Vector store (replica)  | `pgvector 0.7.0`         | `db.m6g.large` (2 vCPU, 8 GB), HNSW `ef_construction=400`, `m=16` | $120         |
| Vector store (writer)   | `pgvector 0.7.0`         | `db.r6g.xlarge` (4 vCPU, 32 GB), rebuild every 6h at 2 a.m. | $320         |
| Retrieval               | `Redis 7.2`              | 3 nodes `cache.r7g.large`, `maxmemory 20 GB`, `maxmemory-policy allkeys-lru` | $180         |
| Ranking                 | `bge-reranker-v2-m3`      | ONNX Runtime 1.17 on CPU, top-100 filtered to top-10 | —            |
| LLM                     | `vLLM 0.5.0`             | `g5.xlarge` (1x A10G), `Qwen2-7B-Instruct`, `max_tokens=128`, `temperature=0.1` | $520         |
| API load balancer       | `Traefik 2.11`           | 2 pods behind ALB, circuit breaker after 5 failures | $40          |
| Async indexer           | `python 3.11` + `psycopg 3.1.18` | Lambda + ECS Fargate 1.4, SQS for backpressure | $60          |

The embedding service is a FastAPI app with connection pooling:

```python
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np
from contextlib import asynccontextmanager
from psycopg_pool import AsyncConnectionPool

pool = AsyncConnectionPool("postgresql://user:pass@replica:5432/db", min_size=2, max_size=8)

@asynccontextmanager
async def lifespan(app: FastAPI):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession("bge-small.onnx", sess_options)
    app.state.embedding_session = sess
    yield
    await pool.close()

app = FastAPI(lifespan=lifespan)

@app.post("/embed")
async def embed(text: str):
    emb = app.state.embedding_session.run(None, {"input_ids": ...})[0]
    async with pool.connection() as conn:
        await conn.execute("INSERT INTO embeddings (text_hash, embedding) VALUES (%s, %s)", hash(text), emb.tobytes())
    return {"embedding": emb.tolist()}
```

The retrieval pipeline uses Redis for rate limiting and top-k caching:

```javascript
// Node 20 LTS + redis 4.6
import { createClient } from 'redis4';
const redis = createClient({ url: 'redis://cache:6379' });

const limiter = redis.pipeline()
  .incr(`rate:${userId}`)
  .expire(`rate:${userId}`, 60)
  .exec();

if (limiter[0][1] > 100) {
  return new Response('Too many requests', { status: 429 });
}

const cacheKey = `topk:${hash(query)}`;
let vectors = await redis.json.get(cacheKey);
if (!vectors) {
  vectors = await pg.query(
    `SELECT embedding FROM chunks ORDER BY embedding <=> $1 LIMIT 100`, [queryEmbedding]
  );
  await redis.json.set(cacheKey, '$', vectors, { EX: 300 });
}
```

The async indexer uses a merge strategy that avoids full index rebuilds:

```sql
-- PostgreSQL 16
BEGIN;
-- Stage new chunks
INSERT INTO new_chunks (chunk_id, embedding)
SELECT id, embedding FROM staging_chunks;

-- Merge in batches of 5 000 to avoid lock contention
DELETE FROM chunks WHERE id IN (SELECT chunk_id FROM new_chunks WHERE chunk_id BETWEEN $1 AND $2);
INSERT INTO chunks SELECT * FROM new_chunks WHERE chunk_id BETWEEN $1 AND $2;

-- Update HNSW index concurrently (no lock)
CREATE INDEX CONCURRENTLY IF NOT EXISTS ivfflat_chunks_idx 
  ON chunks USING hnsw (embedding vector_cosine_ops);
COMMIT;
```

We also added Prometheus metrics to watch queue depths, pool sizes, and error rates:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rag-pipeline'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000', 'embedding:8001', 'retrieval:8002']
    scrape_interval: 5s
```

All components run in EKS with Karpenter provisioner tuned for low-latency pods (startup < 2 s) and spot instances for non-critical workers (embedder, async indexer).

## Results — the numbers before and after

| Metric                               | Before (naive stack) | After (tuned stack) | Change     |
|--------------------------------------|----------------------|---------------------|------------|
| 95th-percentile latency (full stack) | 800 ms               | 145 ms              | –82%       |
| Error rate during peak (15 k users)   | 8%                   | 0.3%                | –96%       |
| Monthly AWS bill                     | $2 400               | $1 232              | –48.7%     |
| Embedding throughput (req/sec)        | 450                  | 1 200               | +167%      |
| Knowledge-base refresh downtime       | 4–6 min every 6 h    | 30 s every 6 h      | –92%       |
| Memory usage (Redis)                 | 42 GB (peak)         | 18 GB (steady)      | –57%       |
| Prompt token count (avg)              | 205 tokens           | 130 tokens          | –37%       |
| Accuracy (human-labeled test set)    | 72%                  | 85%                 | +13%       |

During the October 2026 loan promotion, we served 112 k chat sessions in 72 hours. The 95th-percentile latency stayed at 150 ms even when traffic doubled the normal peak. The bill for the entire month came in at $1 232—$480 below our $1 800 target.

The most surprising win was the prompt token reduction. By reranking early and capping context to the top-10 chunks, we cut the LLM input by 37%. That alone saved us $180/month in LLM tokens and reduced generation latency from 210 ms to 105 ms.

We also discovered that Vietnamese text splits better with a 256-token window and 32-token overlap when using a Vietnamese-specific tokenizer (`vinai/phobert-base`). The recall on our test set jumped 9 percentage points compared to the generic `sentence-transformers` tokenizer.

The circuit breaker reduced cascading failures to zero. Before, a single 500 ms spike in the DB could trigger 100 retries per user and melt the API. After, the breaker kicked in after 5 failures and returned a cached response for 30 s. That alone cut error rate from 8% to under 1% during spikes.

Cost-wise, the biggest lever was moving from SageMaker embeddings ($1 100/month) to ONNX on Graviton ($80/month). Accuracy stayed the same, but latency improved 40 ms because we eliminated the network hop to SageMaker. We also shrank the instance from 16 vCPU to 4, proving that CPU-bound workloads don’t always need more cores.

## What we'd do differently

1. Start with Vietnamese-specific tokenization earlier. We wasted two weeks tuning chunk sizes before realizing the tokenizer was the bottleneck. A quick test with `vinai/phobert-base` on a 100-sentence sample showed an immediate 6% recall lift.

2. Measure cache hit rate before tuning cache size. We set Redis memory to 40 GB based on a gut feel. After two weeks we saw a 68% hit rate and 12 GB steady usage. We could have started with 20 GB and saved $90/month from day one.

3. Test index rebuilds under load before going to production. Our first rebuild script locked the table for 6 minutes and dropped the API p99 from 150 ms to 2.1 s. We should have run a chaos test: `pgbench --time 300 --jobs 50` while rebuilding the index concurrently.

4. Use async/await consistently from day one. Our first FastAPI embedder used synchronous `psycopg2`. Under 500 req/sec it leaked connections until we hit the pool limit. Switching to `psycopg 3.1.18` async pool cut connection churn by 70%.

5. Set circuit breaker thresholds before traffic spikes. We tuned the breaker after the first outage. We should have done it during load testing so we knew the exact failure threshold.

6. Avoid mixing staging and prod in the same vector DB. We ran staging on the same `pgvector` instance for a week, and the staging index scans fragmented the prod HNSW graph. Separate instances cost $40/month but saved us hours of debugging.

7. Monitor GPU memory, not just GPU utilization. Our first vLLM deployment crashed with OOM errors even though utilization was only 30%. Adding `max_model_len=2048` and `gpu_memory_utilization=0.7` prevented crashes.

The biggest lesson: production RAG isn’t about the model or the index; it’s about the resource boundaries. Every stage—embedding, retrieval, generation—has a concurrency limit and a memory budget. Ignore them and your pipeline will crumble under load.

## The broader lesson

The core mistake in most RAG tutorials is treating the pipeline as a pipeline, not as a distributed system. A RAG pipeline is five loosely coupled services glued together: document splitter, embedder, vector store, reranker, and generator. Each service has its own resource envelope—CPU threads, GPU memory, connection slots, disk I/O, and network bandwidth. When you exceed any envelope, the entire chain breaks.

The second mistake is assuming the vector index is immutable. In production, the index is a living data structure that rebuilds, fragments, and grows. If you don’t bound the rebuild time and lock duration, the query path will stall during every knowledge-base refresh. The fix is to separate writes (offline rebuilds) from reads (replica queries), and to use concurrent index builds when possible.

The third mistake is caching the wrong thing. Most teams cache raw embeddings or full documents, but the expensive step is retrieval—not embedding. Cache the top-k vector IDs or the reranked chunks, not the embeddings. Evict aggressively using size-based policies, not time-based TTLs.

Finally, measure the pipeline end to end, not per stage. A 40 ms embedding stage followed by a 160 ms retrieval stage gives a 200 ms total—until the vector index locks and retrieval jumps to 800 ms. The only way to catch these cascades is continuous latency histograms and error-rate alerts per user, not per pod.

Production RAG is less about accuracy and more about stability under load. The tutorials skip the resource limits and the operational details because they assume a single user and a cold start. In reality, you’re running a public API that must survive Black Friday and a CFO who wants the bill justified.

## How to apply this to your situation

1. Profile your current pipeline under synthetic load. Use `k6` to hit 2× expected peak for 30 minutes. Watch connection pool depth, index scan latency, and GPU memory. If any metric spikes beyond 80% of capacity, you’ve found your first bottleneck.

2. Split your vector store into read-only replica for queries and a dedicated writer for inserts. Use concurrent index builds (`CREATE INDEX CONCURRENTLY`) and rebuild during off-peak hours. Measure the rebuild time and set alerts if it exceeds 2 minutes.

3. Implement bounded concurrency at every stage. Use Hystrix-style circuit breakers, fixed-size worker pools, and backpressure via queues. Return 429 early instead of queuing indefinitely. Your error rate will drop from 8% to under 1% during spikes.

4. Cache the top-k vector IDs, not raw embeddings. Use Redis with `maxmemory 20 GB` and `allkeys-lru`. Evict aggressively. Expect a 60–70% cache hit rate; if it’s lower, shrink the cache size and redirect savings to faster hardware.

5. Tokenize early and language-specifically. If you’re serving Vietnamese, use `vinai/phobert-base` for splitting. If you’re serving Thai, use `wangchanberta`. The tokenizer can lift recall by 6–10 points without changing the model.

6. Measure prompt token count and generation length. Cap both. A 200-token prompt with 128 max tokens will save you 30–40% on LLM costs and cut generation latency in half.

7. Add Prometheus metrics for queue depth, pool size, and error rate per user. Set alerts at 80% of capacity. The first alert will tell you exactly which stage is breaking.

Do these seven steps in order. Skip any stage and you’ll hit the same wall we did—connection exhaustion, index locks, or OOM crashes—just later in the project.

## Resources that helped

- `pgvector` tuning guide: https://github.com/pgvector/pgvector/blob/master/docs/tuning.md (accessed 2026-05-15)
- ONNX Runtime docs for embedding models: https://onnxruntime.ai/docs/ (v1.17)
- vLLM 0.5.0 release notes on batching and memory: https://github.com/vllm-project/vllm/releases/tag/v0.5.0
- Vietnamese tokenizer comparison: https://github.com/VinAIResearch/PhoBERT (2026-03-10)
- Redis 7.2 memory policies explained: https://redis.io/docs/manual/eviction/
- Karpenter tuning for low-latency pods: https://karpenter.sh/docs/ (2026-04-15)
- Chaos engineering for RAG pipelines (case study from Grab): https://grab.github.io/ (2026-01-22)

## Frequently Asked Questions

**Why did you move from SageMaker embeddings to ONNX on Graviton? What was the latency difference?**

SageMaker endpoints added a 30–50 ms network hop and cost $1 100/month for the single endpoint we used. ONNX Runtime on a 4-vCPU `c7g.4xlarge` handled 1 200 req/sec with 40 ms median latency. The switch saved $1 020/month and cut 40 ms off every request.

**What chunk size and overlap did you settle on for Vietnamese text? How did you measure recall?**

We tested 128, 256, and 512 token windows with 16, 32, and 64 token overlaps. The best balance was 256 tokens with 32 overlap using `vinai/phobert-base` tokenizer. We measured recall@10 on a 500-question test set. 256/32 gave 85% recall vs 76% for 128/16 and 79% for 512/32.

**How did you handle the Vietnamese diacritics and accents in search? Did you normalize before embedding?**

We normalized to NFC form and removed tone marks in the tokenizer (`vinai/phobert-base` does this internally). This reduced noise in the embedding space. We also added a Vietnamese stopword filter before splitting. Without normalization, recall dropped 5 percentage points.

**What alert thresholds did you set for your Prometheus metrics? What was the first alert that fired during production?**

We set alerts at 80% of capacity: embedding pool depth > 6, retrieval pool depth > 12, Redis memory > 16 GB, and error rate > 1% per 5-minute window. The first alert was Redis memory approaching 16 GB after we ingested 10 k new documents in one hour. We caught it before it started evicting keys aggressively and tuned the merge batch size.

**Why did you choose `bge-reranker-v2-m3` instead of keeping the same embedding model for retrieval and reranking?**

The reranker uses a cross-encoder architecture that scores all pairs of query and chunk. It’s slower (30 ms per rerank) but more accurate than cosine similarity. By using the small reranker on the top-100 chunks, we cut the prompt token count by 30% and lifted accuracy 13 points without changing the final LLM.

**What was the biggest surprise after moving to the async indexer?**


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

**Last reviewed:** May 30, 2026
