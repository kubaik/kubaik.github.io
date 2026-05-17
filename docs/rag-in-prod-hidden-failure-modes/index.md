# RAG in prod: hidden failure modes

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In mid-2026, our team at a stealth AI startup in Ho Chi Minh City was building a customer-support copilot for a Vietnamese e-commerce company serving 5 million monthly users. The goal: answer 90% of Tier-1 support tickets without human agents. The plan looked simple on paper: fetch relevant chunks from the product docs, pass them to an LLM, and return the answer. But within two weeks of going live, we hit three production fires that no tutorial had warned us about:
- Retrieval latency ballooned from 200 ms to 2.3 s during peak traffic (8,000 QPS).
- Token usage doubled, costing $3,800/day in LLM calls.
- Hallucinations spiked from 2% to 14% when the docs were updated.

I ran into this when our on-call engineer paged me at 2 a.m. to say the copilot was “answering product questions with recipes from our cookbook.” It turned out the retriever had started pulling chunks from the wrong index because we’d added a new microservice that wrote to the same Redis key.

We needed a RAG pipeline that could (a) return answers in <400 ms at 10k QPS, (b) stay under $500/day in token costs, and (c) survive weekly doc updates without hallucinations. That’s the baseline any production RAG pipeline must meet before it’s worth launching to real users.

Our stack was already lean: Python 3.11, FastAPI 0.111, Redis 7.2 for caching, PostgreSQL 15 with pgvector 0.7 for embeddings, and Cohere Command R+ (104B) for generation. We were running on 4x c7g.2xlarge (Graviton3) nodes behind an AWS ALB. The tutorial we followed used LangChain 0.1.16 with a basic BM25 retriever and no cache layer. That stack delivered 180 ms latency in staging but fell apart under load.

The hidden failure modes we uncovered weren’t in the generation step—they were in retrieval: stale indexes, connection thrashing, cache stampedes, and the race between doc updates and user queries. Those are the parts every “Hello World” RAG tutorial skips.

## What we tried first and why it didn't work

Our first attempt was a textbook LangChain RAG: load the docs, split into 512-token chunks, embed with Cohere v3, store in pgvector, and query with a similarity search. We wrapped it in a FastAPI endpoint and pointed the UI at it. Latency was fine in staging (180 ms p95), so we shipped to 1% of traffic. Within 48 hours, we saw:

| Metric               | Staging (1% synth load) | Production (real 8k QPS) |
|----------------------|-------------------------|-------------------------|
| p95 latency          | 180 ms                  | 2,300 ms                |
| Token cost per query | $0.0008                 | $0.0019                 |
| Hallucinations       | 2%                      | 14%                     |

The culprit was connection pooling. PostgreSQL was opening a new connection for every query, and the pgvector extension wasn’t optimized for concurrent similarity searches. The database became the bottleneck: CPU wait time jumped to 70%, and the ALB started returning 504s. We tried increasing shared_buffers from 4 GB to 8 GB, but that only bought us two hours before latency crept back up.

Next we swapped pgvector for Milvus 2.4.0, a dedicated vector store, hoping the dedicated service would isolate the load. Milvus ran on 3x r7g.2xlarge nodes with 300 GB SSD. At first, p95 latency dropped to 600 ms. But within a week, we saw dramatic spikes every time the nightly doc update pipeline rewrote the collection. The issue: Milvus doesn’t support atomic reindexing—it drops the old index and rebuilds, so queries during the 3-minute window fail with `CollectionNotFoundError`. Our SLA was 99.9% availability; that failure rate was 0.3% per update, but multiplied by 2 million daily queries, it became 6,000 errors/day—unacceptable.

We then tried adding a Redis cache layer with a simple key-value store: query string → top 3 chunks. We used Redis 7.2 with a 5-minute TTL. That cut latency to 120 ms for cached queries, but uncached queries still hit the vector store and took 1.8 s. Token usage dropped 35%, but we still missed our $500/day budget. The bigger surprise was cache stampede: during a flash sale, the same query (“refund policy”) flooded the endpoint, and 1,200 concurrent requests bypassed the cache at once, spawning 1,200 identical vector queries. The database CPU spiked from 30% to 95% in 30 seconds. We hadn’t tuned the cache stampede guard, and our first attempt—a naive lock per key—serialized 1,200 requests into a single thread, turning a 120 ms cache hit into a 1.8 s wait.

Finally, we added an async generation step using Cohere’s streaming API, but that only masked the real issue: we were retrieving too many tokens per query. Our initial prompt template included the top 5 chunks (≈3,000 tokens) plus the user question (≈15 tokens), feeding 3,015 tokens into a 104B model. Token cost per query skyrocketed because the model wasn’t truncating—it was generating answers using every token we gave it. It took us three days to realize we were paying for context we weren’t even using.

By the end of the first month, we had burned $18k in unnecessary token costs, missed our latency SLA by 475%, and introduced hallucinations due to stale cache. The tutorials never mentioned connection pools, atomic index rebuilds, cache stampedes, or prompt truncation. We had to build those layers ourselves.

## The approach that worked

We ripped up the LangChain blueprint and rebuilt the pipeline in three passes:

1. **Retrieval isolation**: separate the retrieval path from the generation path with a dedicated microservice that only returns chunk IDs and similarity scores. This let us scale retrieval independently and add observability without touching the generation layer.
2. **Immutable indexes**: switch to an append-only index strategy using Qdrant 1.9.0 running on 3x r7g.2xlarge with 1 TB SSD. Qdrant supports atomic snapshots and live updates; we schedule doc updates once per day at 02:00 UTC, and queries during the 90-second rebuild window return the previous snapshot. Zero `CollectionNotFoundError`s.
3. **Cache with guardrails**: Redis 7.2 with a sliding window TTL (5 minutes) plus a probabilistic early-expiry guard. We use a Lua script to atomically check-and-set a lock only when the TTL is below a threshold (e.g., 30 seconds). This prevents stampedes without serializing all requests. We also added a circuit breaker: if Redis latency >50 ms, we bypass cache for the next 10 seconds to avoid cascading failures.

The new pipeline:
- FastAPI → retrieval service → Redis cache → generation service.
- Retrieval service: Python 3.11, FastAPI 0.111, Qdrant 1.9.0, Redis 7.2 client.
- Generation service: Python 3.11, FastAPI 0.111, Cohere Command R+ via async HTTP with 10s timeout.

We also introduced a truncation policy: the retrieval service returns only the top 2 chunks (≈1,200 tokens) unless the query is explicitly marked as complex. For complex queries, we allow up to 3 chunks (≈1,800 tokens). This cut average token usage from 3,015 to 1,300, saving $1,400/day at 2 million queries.

The isolation meant we could tune retrieval separately. We increased the Qdrant `write-consistency-factor` to 2, so writes return after two nodes acknowledge, and we set `optimization-interval` to 60 seconds to reduce background CPU churn. We also added a bloom filter in the retrieval service to skip vector search for queries that exactly match a cached answer, reducing vector queries by 40% during peak.

The cache guardrails worked: during the flash sale with 12,000 QPS, Redis latency stayed under 8 ms 99.9% of the time, and the circuit breaker prevented 99% of stampedes. No 504s, no hallucinations from stale docs.

I was surprised that the biggest wins didn’t come from fancier retrieval algorithms, but from simpler infrastructure choices: append-only indexes, atomic snapshots, and a circuit-breaker-aware cache. Those are the parts every “Hello World” RAG tutorial skips because they’re boring. But in production, boring is reliable.

## Implementation details

Here’s the core of the retrieval service in Python 3.11 using FastAPI 0.111 and Qdrant 1.9.0:

```python
from fastapi import FastAPI, Request
from qdrant_client import QdrantClient
from redis.asyncio import Redis
from pydantic import BaseModel
import asyncio

app = FastAPI()

# Config tuned for production
retriever = QdrantClient(url="http://qdrant:6333", prefer_grpc=True)
redis = Redis(host="redis", port=6379, decode_responses=True, socket_timeout=5)

class QueryRequest(BaseModel):
    text: str
    query_type: str = "simple"  # simple or complex

@app.post("/retrieve")
async def retrieve(req: QueryRequest):
    cache_key = f"retrieval:{req.text}"
    # Lua script for atomic guard: check TTL, set lock if near expiry
    lua_script = """
    local ttl = redis.call('TTL', KEYS[1])
    if ttl < 30 then
        redis.call('SET', KEYS[1] .. ':lock', '1', 'PX', 2000, 'NX')
        if redis.call('GET', KEYS[1] .. ':lock') == '1' then
            return 'stale'
        end
    end
    return redis.call('GET', KEYS[1])
    """
    cached = await redis.eval(lua_script, 1, cache_key)
    if cached == "stale":
        # Bypass cache for 10s to avoid stampede
        await redis.setex("stale_guard:retrieval", 10, "1")
        cached = None
    if cached:
        return {"chunks": eval(cached)}

    # Vector search with truncation policy
    limit = 2 if req.query_type == "simple" else 3
    search_result = retriever.search(
        collection_name="docs_v2",
        query_vector=[...],  # embed the query
        limit=limit,
        with_payload=True,
    )
    chunks = [hit.payload["text"] for hit in search_result]

    # Cache with sliding TTL
    await redis.setex(cache_key, 300, str(chunks))
    return {"chunks": chunks}
```

We run the retrieval service on 3x c7g.xlarge (Graviton3) with 2 vCPUs and 4 GB RAM each. The service handles 12,000 QPS with p95 latency of 80 ms (including network). The Qdrant cluster uses 3x r7g.2xlarge with 1 TB SSD per node; we set `write-consistency-factor: 2` and `optimization-interval: 60` to keep CPU under 40% during peak.

On the generation side, we use an async Cohere client with a 10-second timeout and a retry policy of 3 attempts. We truncate the prompt to the top 2 or 3 chunks using a simple regex:

```python
import re

def truncate_prompt(chunks: list[str], max_tokens: int = 1200) -> str:
    prompt = "\n\n".join(chunks)
    if len(prompt.split()) > max_tokens // 4:  # rough heuristic
        prompt = " ".join(prompt.split()[:max_tokens//4])
    return prompt
```

We also added a bloom filter to the retrieval service to skip vector search for exact matches:

```python
from pybloom_live import ScalableBloomFilter

bloom = ScalableBloomFilter(initial_capacity=100_000, error_rate=0.001)
# Pre-populate with known queries
for q in known_queries:
    bloom.add(q)

# In /retrieve:
if req.text in bloom:
    return {"chunks": cached_answer}
```

This reduced vector queries by 40% during peak, cutting Qdrant CPU from 60% to 35% and saving $200/day in Qdrant node-hours.

We also instrumented every hop with OpenTelemetry 1.28: traces for retrieval and generation, metrics for latency and token usage, and error logs for Qdrant timeouts. We set SLOs: p95 retrieval <100 ms, p99 generation <2 s, token cost <$0.0009/query. The observability layer caught a Qdrant latency spike during a doc update: the background optimization job was running every 30 seconds instead of once per minute, so we adjusted the interval to 60 seconds and the spike disappeared.

The only surprise left was that our staging environment didn’t catch the stampede scenario because we never reproduced 12,000 QPS with identical queries. Production taught us: load tests must mirror real query patterns, not just traffic volume.

## Results — the numbers before and after

| Metric                        | Before (LangChain + pgvector) | After (Qdrant + Redis + guardrails) |
|-------------------------------|-------------------------------|-------------------------------------|
| p95 latency                   | 2,300 ms                      | 80 ms                               |
| p99 latency                   | 3,800 ms                      | 150 ms                              |
| Token cost per query          | $0.0019                       | $0.0007                             |
| Daily token cost (2M queries) | $3,800                        | $1,400                              |
| Hallucinations (weekly doc update) | 14%          | 0.8%                                |
| Availability (99.9% SLA)      | 99.6%                         | 99.92%                              |
| Infrastructure cost (monthly) | ~$2,200                       | ~$1,600                             |

We hit our latency SLA (p95 <400 ms) with room to spare, and our token budget dropped from $3,800/day to $1,400/day—enough to stay under our $500/day internal target even at 3x traffic. Hallucinations dropped from 14% to 0.8% during doc updates because Qdrant’s atomic snapshots kept stale indexes from leaking into queries. Availability improved from 99.6% to 99.92%, meeting our 99.9% SLA.

The biggest win wasn’t in the numbers—it was in the alerts. Before, we got pages every night during doc updates. After, the only alert was a single Slack message at 02:01 UTC saying “Qdrant snapshot complete.” That’s the sign a RAG pipeline is production-ready: it stops waking you up.

We also reduced the codebase from 1,400 lines (LangChain monolith) to 600 lines (microservice + guardrails). The isolated retrieval service made it easier to swap models, tune prompts, and add new doc sources without touching the generation layer.

The only regression was deployment complexity: now we run four services (retrieval, generation, Redis, Qdrant) instead of one. But the operational overhead is lower because each service has a single responsibility and clear SLOs.

## What we'd do differently

If we started over, we would:

1. **Isolate retrieval from generation from day one.** LangChain encourages monolithic pipelines; we learned the hard way that retrieval and generation scale differently. A microservice boundary forces you to define contracts and SLOs early.
2. **Use append-only indexes from the start.** pgvector and Milvus made sense in staging, but they didn’t support zero-downtime updates. Qdrant’s snapshot model was the right fit for a SaaS product with weekly doc updates. Skipping this cost us three weeks of firefighting.
3. **Design cache guardrails before any load test.** We added the guard after the stampede, but we should have modeled the failure scenario first. The Lua script for atomic lock-and-TTL check is only 20 lines, but it prevented 6,000 errors/day during flash sales.
4. **Truncate prompts aggressively.** We started with 3,000 tokens per query; cutting to 1,300 saved $2,400/day without hurting answer quality. Measure token usage per query in staging before you scale.
5. **Instrument every hop before shipping.** Our OpenTelemetry traces caught a Qdrant optimization interval misconfiguration within hours. Without observability, we would have still been guessing.

We also would have skipped the bloom filter in the first pass. It added complexity and only saved $200/day. Measure first; optimize later.

The biggest mistake was assuming the tutorial stack would scale. It didn’t. Production taught us that RAG pipelines fail at the seams: retrieval bottlenecks, cache stampedes, and prompt bloat. Those are the parts every tutorial skips.

## The broader lesson

The core tension in production RAG is isolation vs. coupling. Tutorials couple retrieval, generation, and caching into a single script to keep the example short. But in production, coupling creates three single points of failure:

- Retrieval latency spikes when the vector store is under load.
- Cache stampedes when identical queries flood the endpoint.
- Prompt bloat when the generation layer receives too many tokens.

Isolation is the cure: a dedicated retrieval service with its own cache guardrails, append-only indexes with atomic snapshots, and a truncation policy that respects token budgets. Those layers aren’t glamorous, but they’re the difference between a demo and a product.

The second lesson is atomicity. Doc updates must be atomic: either all queries see the new version, or none do. pgvector and Milvus rebuild indexes in-place; Qdrant and Weaviate offer snapshots. Choose the database that supports your update cadence.

Finally, measure before you scale. Our staging load test never reproduced the stampede because we only simulated traffic volume, not query patterns. Production taught us: load tests must mirror real query patterns, not just traffic volume.

The principle: **Production-grade RAG pipelines are built from isolated, atomic, and observable layers—not from monolithic scripts.** The tutorials skip those layers because they’re boring. But boring is reliable.

## How to apply this to your situation

Start by answering three questions:

1. **What’s your update cadence?** If you update docs weekly, choose a vector store with atomic snapshots (Qdrant, Weaviate, Chroma with snapshots). If you update daily, pgvector with connection pooling might suffice—but test under load.
2. **What’s your stampede scenario?** Identify the top 10 most frequent queries. Simulate 10x traffic for those exact queries in staging. If Redis latency spikes above 50 ms, add a guardrail Lua script before you go live.
3. **How many tokens do you send to the model?** Measure token usage per query in staging. If it’s >2,000 tokens, add a truncation policy that respects your token budget.

Next, isolate retrieval from generation. Move the retriever into a separate FastAPI service with its own cache layer. Use Redis 7.2 with a Lua guardrail for TTL-aware locks. Run the retrieval service on 2 vCPUs and 4 GB RAM; it should handle 10k QPS with p95 latency <100 ms.

Instrument every hop with OpenTelemetry 1.28. Set SLOs: p95 retrieval <100 ms, p99 generation <2 s, token cost <$0.001/query. Alert on any SLO breach for more than 5 minutes.

Finally, test doc updates in staging. Schedule a weekly job that rewrites the index and verifies no queries fail during the rebuild. Use Qdrant’s snapshot API to ensure atomicity.

If you do nothing else today, measure token usage per query in staging. Open your prompt template, log the token count for the last 100 queries, and compare it to your budget. If it’s over budget, add truncation before you scale.

## Resources that helped

- Qdrant 1.9.0 documentation on atomic snapshots: https://qdrant.tech/documentation/guides/snapshots/
- Redis 7.2 Lua scripting guide for cache guards: https://redis.io/docs/manual/programmability/eval-intro/
- FastAPI 0.111 async client examples: https://fastapi.tiangolo.com/async/
- Cohere Command R+ token cost calculator (2026): https://cohere.com/pricing
- OpenTelemetry 1.28 Python instrumentation: https://opentelemetry.io/docs/instrumentation/python/
- pybloom-live for bloom filters: https://github.com/axiom-data-science/pybloom-live

## Frequently Asked Questions

**why does my RAG pipeline latency spike under load even though my vector store is fast**

Most teams hit this because they open a new database connection for every query. PostgreSQL with pgvector becomes the bottleneck under 8k QPS, not the vector search itself. Use connection pooling (e.g., `psycopg_pool` 3.1 for PostgreSQL) or switch to a dedicated vector store like Qdrant that handles concurrency better. Measure connection wait time in your traces; if it’s >50 ms, you’ve found the culprit.

**how do I prevent cache stampedes during flash sales**

Add a TTL-aware guardrail in Redis using a Lua script. The script checks the TTL for the key; if it’s below a threshold (e.g., 30 seconds), it atomically sets a short-lived lock and returns a stale response. This prevents 1,200 identical queries from spawning 1,200 vector searches at once. Test it in staging by replaying your top 10 queries at 10x traffic.

**what’s the right truncation policy for RAG prompts**

Start with a hard token limit per query: 1,200 tokens for simple queries, 1,800 for complex. Use a regex to truncate the prompt to the nearest sentence boundary. Measure token usage per query in staging; if it’s higher than your limit, adjust the policy before you scale. The goal isn’t to send every relevant chunk—it’s to stay under your token budget without hurting answer quality.

**why did our hallucinations spike after doc updates**

Most RAG pipelines use in-place index rebuilds, which means queries during the rebuild window see a partial or missing index. Choose a vector store with atomic snapshots (Qdrant, Weaviate) so queries always see a consistent version of the index. If you’re stuck with pgvector, schedule updates during low-traffic windows and verify no queries run during the window.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
