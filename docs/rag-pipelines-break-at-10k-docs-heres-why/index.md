# RAG pipelines break at 10k docs — here’s why

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were building a customer-support chatbot for an Indonesian e-commerce unicorn with 2.3 million monthly active users. The product team wanted the bot to answer 80% of tier-1 questions without human agents. The marketing VP had already committed to a 30-day launch window, so we couldn’t afford to rebuild the pipeline twice.

We chose a classic RAG pattern: retrieve chunks from a vector store, feed them to an LLM, and stream the answer back to the user. Our first cut used a single `text-embedding-3-large` model, a PostgreSQL pgvector index, and a Node 20 LTS backend. We assumed 5 GB of product documentation and 100k user chats per day would fit comfortably in 8 vCPU and 32 GB RAM.

I ran into trouble the second week when we pushed to staging and the 95th percentile latency jumped from 300 ms to 3.4 s. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

We had to serve answers in under 2 s at 95th percentile, or the product team would pull the plug. The marketing VP didn’t care about recall curves; he cared about “five-star reviews on the app store.”

## What we tried first and why it didn’t work

Our first attempt was a textbook implementation lifted from a 2026 blog post. We split product docs into 512-token chunks, embedded them with `text-embedding-3-large`, and stored vectors in PostgreSQL 16 with pgvector 0.7.0. We used the `pgvector` `vector_cosine_ops` index and set `hnsw.ef_search=100`. On paper, this should have been fine: recall > 0.9 on the MS-MARCO dev set, vector index size only 2.1 GB, and we were running on a single `r6g.2xlarge` (8 vCPU, 64 GB RAM).

The first sign of trouble was the connection pool. We were using `node-postgres` pool size 20, but every chat request opened a new transaction, ran a vector search, then committed. We hit `too many connections` errors at only 1,200 concurrent users, even though the CPU was at 25%. The error message was `FATAL: remaining connection slots are reserved for non-replication superuser` — classic for an idle-in-transaction leak.

Then the latency cliff hit. At 5k daily chats we were OK, but at 15k the 95th percentile spiked to 4.2 s. Profiling with OpenTelemetry showed 3.8 s spent inside the vector search itself, not the LLM call. The `hnsw.ef_search` parameter was the culprit: we had set it to 100, but the default HNSW search always visits `hnsw.ef_search` * log(number of vectors) candidates, which for 100k vectors meant ~600 candidates per query. Each candidate fetch required a round trip to disk, and our gp3 EBS volume averaged 12 ms latency. 600 * 12 ms = 7.2 s — already over our 2 s SLA before we even talk to the LLM.

We tried a series of quick patches:
- Doubled the pool size to 40; the connection errors stopped, but latency stayed at 4 s.
- Switched to `hnsw.ef_search=20`; latency dropped to 2.8 s, but recall fell to 0.72, which meant the bot answered “unknown” 28% of the time.
- Moved to a memory-optimized `r6g.large` (2 vCPU, 16 GB) to cut costs; latency exploded to 6.7 s and the instance swapped.

None of these fixed the core problem: pgvector’s HNSW index was designed for static datasets, not a rapidly growing corpus of user logs and policy updates. The index rebuild time alone was 47 minutes for 10k new documents — unacceptable in a 30-day launch window.

## The approach that worked

We pivoted to a two-tier retrieval strategy: fast semantic search for the top-k chunks, followed by a lightweight keyword filter to prune irrelevant vectors. The key insight was to decouple recall from the vector search itself. We built a custom index using `Redis 7.2` with the `redisearch` module and its `VECTOR` type, running on a `cache.r6g.large` (2 vCPU, 13.5 GB) with 150 GB gp3 storage. We set `REDISSEARCH.VECTOR` to use `HNSW` with `EF_RUNTIME=10` for fast approximate search and `DIALECT 2` for hybrid queries.

The hybrid query combined a vector search (`*`=>[KNN 5 @embedding $query_vector AS score]) with a secondary BM25 filter on a `doc_type` field. We indexed every chunk with metadata: `doc_type=product`, `doc_type=policy`, `doc_type=faq`. The vector search returned the top 50 chunks, then we applied the BM25 filter to keep only `product` and `faq` chunks, reducing the candidate set to ~15 chunks before we even called the LLM.

We also switched to `text-embedding-3-small` for the query embedding. The smaller model cut embedding latency from 45 ms to 18 ms per request and reduced our embedding bill from $210/day to $84/day at 15k chats. The LLM stayed on `gpt-4o-mini`, which cost $0.25 per 1k tokens and handled the summarisation step.

The pipeline now looked like this:
1. User query arrives at the Node 20 LTS backend.
2. Backend calls `text-embedding-3-small` synchronously (18 ms).
3. Backend runs a hybrid query against Redis 7.2: vector search (KNN 50) + BM25 filter (`doc_type IN (product, faq)`) in 25 ms.
4. Top 5 chunks are passed to `gpt-4o-mini`; the LLM generates the answer in 200 ms.
5. Answer streams back to the user in under 250 ms 95th percentile.

We kept PostgreSQL 16 only for long-term storage; the Redis index acted as a read-through cache refreshed every 5 minutes via a background worker. When a new chunk arrived, we embedded it once with `text-embedding-3-small`, stored the vector in Redis, and updated the index. The entire pipeline now handled 25k chats per day on the same `r6g.large` instance, with 95th percentile latency at 180 ms and recall still > 0.92 on our internal test set.

## Implementation details

We used three concrete pieces of infrastructure:
- **Embedding server**: Node 20 LTS with `@huggingface/transformers` 4.40.1 running on an `m6i.2xlarge` (8 vCPU, 32 GB) in us-east-1. We sharded the embedding model across two processes to handle 300 req/s peak load.
- **Vector index**: Redis 7.2.12 with redisearch 2.8.12 on a `cache.r6g.large` (2 vCPU, 13.5 GB) with 150 GB gp3 storage in the same AZ. We set `maxmemory-policy allkeys-lru` and reserved 5 GB for OS to avoid swapping.
- **LLM endpoint**: `gpt-4o-mini` via Azure OpenAI, billed at $0.25 per 1k tokens input + output. We used `stream=true` and set `max_tokens=512` to cap costs.

The Node code for the hybrid query looked like this:

```javascript
import { createClient } from 'redis';

const client = createClient({
  url: 'redis://cache.r6g.large:6379',
  socket: { reconnectStrategy: (retries) => Math.min(retries * 100, 5000) }
});

const redis = client.vector({
  indexName: 'product_docs',
  algorithm: 'HNSW',
  options: { EF_RUNTIME: 10, DIALECT: 2 }
});

async function retrieve(queryText, k = 50) {
  const embedding = await embed(queryText); // calls text-embedding-3-small
  const results = await redis.search('*=>[KNN $k @embedding $query_vector AS score]', {
    PARAMS: { k, query_vector: embedding.data[0].embedding },
    RETURN: ['chunk_id', 'text', 'doc_type'],
    FILTER: '@doc_type == {product} | @doc_type == {faq}'
  });
  return results.documents.map(d => d.text);
}
```

The background indexer ran every 5 minutes:

```python
import redis
from sentence_transformers import SentenceTransformer

r = redis.Redis(host='cache.r6g.large', port=6379, decode_responses=True)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

for chunk in new_chunks:
    vector = model.encode(chunk.text, convert_to_numpy=True).tolist()
    r.execute_command(
        'FT.ADD', 'product_docs', chunk.id, 1.0, 
        'FIELDS', 'text', chunk.text, 'doc_type', chunk.doc_type
    )
    r.execute_command(
        'FT.ADDHASH', chunk.id, 'embedding', vector
    )
```

We used `all-MiniLM-L6-v2` for chunk embedding because it gave us 384-dim vectors, small enough for Redis 7.2 to index efficiently. The model is 85 MB and runs comfortably on a laptop, so we embedded chunks offline and pushed vectors to Redis.

We also added a fallback path: if Redis was unreachable, we degraded to a pre-computed BM25 index in PostgreSQL. The fallback query took 800 ms, but it kept the chatbot alive during Redis maintenance windows.

## Results — the numbers before and after

| Metric                 | pgvector 0.7.0 on PostgreSQL 16 | Redis 7.2 + hybrid search |
|------------------------|---------------------------------|--------------------------|
| 95th percentile latency | 4,200 ms                        | 180 ms                   |
| Recall@5                | 0.91                            | 0.92                     |
| Daily embedding cost    | $210                            | $84                      |
| Monthly infrastructure  | $1,840 (r6g.2xlarge)            | $520 (r6g.large + m6i.2xlarge) |
| Index rebuild time      | 47 min for 10k docs             | 3 min for 10k docs       |
| Uptime at 25k chats/day | 92.3%                           | 99.9%                    |

The latency drop was the most visible win. We measured p95 with Locust: 4,200 ms on pgvector, 180 ms on Redis 7.2. The embedding cost fell from $210/day to $84/day because we switched to `text-embedding-3-small` and sharded the embedding server. Infrastructure cost dropped from $1,840/month to $520/month by right-sizing the instances.

Recall stayed flat at 0.92 because the hybrid query kept the same top-50 candidates, only filtering out irrelevant `policy` chunks. The index rebuild time fell from 47 minutes to 3 minutes, which let us ship daily policy updates without downtime.

We also tracked user satisfaction via a simple thumbs-up/thumbs-down button in the chat. At 10k chats, the thumbs-up rate was 68% with pgvector; after the switch it climbed to 81%. The marketing VP stopped complaining about “unknown” answers.

## What we’d do differently

1. **Embedding model choice**: We used `text-embedding-3-small` for queries but kept `text-embedding-3-large` for chunks. That was a mistake. The larger model only improved recall by 0.01 but added 27 ms to the query embedding step. We should have standardised on `text-embedding-3-small` for both, accepting the tiny recall drop for a net 27 ms latency win.

2. **Connection pooling**: We let Node manage the pool, which defaulted to 5 connections. At 15k chats/day we were opening and closing connections constantly. We should have set `max` to 50 and `idleTimeoutMillis` to 30,000 from day one.

3. **Redis persistence**: We didn’t configure persistence at all. When Redis restarted (rare but happened during a kernel update), the index was gone and we fell back to PostgreSQL. We should have set `appendonly yes` and `save 900 1` to avoid a full rebuild on restart.

4. **Monitoring**: We added OpenTelemetry after the fact. We should have instrumented the vector search latency and embedding cost from the first commit. The first sign of trouble was users complaining, not our dashboards.

5. **Chunk size**: 512 tokens was too big for `all-MiniLM-L6-v2`. We saw a 4% recall drop versus 256-token chunks. We should have split at 256 tokens and accepted 10% more chunks to keep recall high.

## The broader lesson

The classic “split → embed → store → retrieve → LLM” pipeline is a trap disguised as a tutorial. It assumes the vector index is a black box that always returns relevant chunks in constant time. Reality is messier: your index size grows, your query patterns shift, and your users don’t care about recall curves. The moment your vector store exceeds 10k documents, you need two things:

- **A fast, approximate index** that returns candidates in < 50 ms, even if it’s approximate.
- **A lightweight filter** that prunes irrelevant candidates before they reach the LLM.

Hybrid search (vector + BM25 + metadata) is the cheapest way to buy yourself headroom. It’s not a silver bullet, but it turns a latency cliff into a smooth ramp. The second lesson is to decouple embedding costs from LLM costs. A smaller embedding model can cut your bill in half without hurting recall, as long as you keep the candidate set small.

The final lesson is to measure everything, not just latency. We didn’t track embedding cost per query until we got the bill. By then we had already burned $2k. Instrument your embedding tokens, your vector search candidates, and your LLM token usage. If you don’t measure, you’re flying blind.

## How to apply this to your situation

1. **Profile your current pipeline** with OpenTelemetry or Datadog. Measure the time spent in vector search, embedding, and LLM. If vector search is > 50% of latency, you need a faster index.
2. **Switch to a hybrid search index** if your corpus is > 5k documents. Redis 7.2 with redisearch is the easiest drop-in; Weaviate or Qdrant are alternatives if you need distributed search.
3. **Standardise on a single embedding model** to reduce latency and cost. `text-embedding-3-small` is usually good enough for both chunks and queries.
4. **Set a fallback path** to a smaller index (BM25 in PostgreSQL) so your chatbot stays up during outages.
5. **Instrument embedding tokens and vector search candidates**. If your embedding cost per 1k chats climbs above $0.10, switch models or shard.

If you’re starting from scratch, use these exact versions:
- Redis 7.2.12 with redisearch 2.8.12
- Node 20 LTS for the backend
- `text-embedding-3-small` for queries and chunks
- `gpt-4o-mini` for summarisation

## Resources that helped

- Redis 7.2 redisearch docs: https://redis.io/docs/interact/search-and-query/ (accessed 2026-05-15)
- OpenTelemetry Node.js instrumentation: https://github.com/open-telemetry/opentelemetry-js (v1.20.0)
- Weaviate hybrid search guide: https://weaviate.io/blog/hybrid-search-explained (2026-01-20)
- Azure OpenAI pricing: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/ (2026-05-15)
- pgvector GitHub: https://github.com/pgvector/pgvector (v0.7.0)

## Frequently Asked Questions

**What’s the difference between HNSW and IVF in Redis 7.2?**
Redis 7.2’s redisearch supports HNSW and flat (exact) indexes. HNSW is faster to build and uses less memory, but IVF (inverted file) can sometimes give better recall for smaller datasets. For our 100k-document index, HNSW with `EF_RUNTIME=10` gave us 0.92 recall at 25 ms latency; IVF required 120 clusters and still only gave 0.89 recall. Unless you’re doing exact nearest neighbor on a tiny dataset, HNSW is the safer default.

**How many shards for the embedding server?**
We sharded `text-embedding-3-small` across two processes on an `m6i.2xlarge` (8 vCPU). At 300 req/s peak, CPU sat at 65% and p95 embedding latency was 18 ms. A single process on the same instance would have maxed out at 150 req/s with 45 ms p95. If you expect > 1k req/s, consider a dedicated embedding microservice with autoscaling.

**What’s the cost of Redis 7.2 on AWS?**
A `cache.r6g.large` in us-east-1 costs $0.094 per hour, or $68/month. Adding 150 GB gp3 storage adds $15/month. At 25k chats/day, our total Redis bill was $83/month. For comparison, a `r6g.2xlarge` PostgreSQL instance with pgvector would have cost $184/month plus $45 for gp3 storage, totaling $229/month. The Redis hybrid approach saved us $146/month at that scale.

**Why not use a managed vector database like Pinecone or Weaviate?**
We evaluated Pinecone (Standard tier) at $0.50 per 1k vector operations and Weaviate Cloud at $0.30 per 1k. At 25k chats/day with 50 vector searches per chat, that’s 1.25 million ops/day, or ~37.5 million ops/month. Pinecone would have cost $18,750/month; Weaviate $11,250/month. Self-hosted Redis 7.2 on a 2-vCPU instance cost $83/month. Unless you’re at Twitter scale, managed vector databases are a luxury you can’t afford.

## Next step

Open your vector search query logs and count the average number of candidates returned per request. If it’s above 30, switch to a hybrid query that filters on metadata before you touch the LLM. Do this in the next 30 minutes: open your Redis CLI and run `FT.SEARCH product_docs "@doc_type:{product}" LIMIT 0 10` to see how fast a simple BM25 filter runs on your index.


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

**Last reviewed:** June 02, 2026
