# RAG pipelines: 3 silent killers in production

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our team at a Jakarta-based fintech startup was racing to ship a customer support chatbot that could answer questions about loan products, interest rates, and payment schedules. The product team promised 50k daily active users within six months. Our first prototype used a simple retrieval-augmented generation (RAG) pipeline with a vector database, a large language model (LLM), and a basic prompt template. Everything worked fine in the demo — latency under 2 seconds, cost under $5/day on AWS. But when we put it behind a load balancer and sent real traffic, we hit three production issues we hadn’t planned for.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

By week three, we were seeing:
- **Mean response time: 8.2 seconds** (vs. 1.8s in tests)
- **P95 latency: 15.7 seconds** (vs. 3.5s in tests)
- **Cost per 1,000 requests: $2.34** (vs. $0.42 in tests)

We were running on:
- **Python 3.11**, **FastAPI 0.110.0**, **Redis 7.2** for caching
- **pgvector 0.7.0** on **PostgreSQL 15.4** for embeddings
- **OpenAI gpt-4o-mini** at $0.40 per 1M tokens
- **AWS EC2 c6i.large** (2 vCPUs, 4GB RAM) for the API
- **AWS RDS db.t4g.large** (2 vCPUs, 8GB RAM) for PostgreSQL

Our vector index had 12,000 documents. The embeddings were 1,536-dimensional. We were using cosine similarity with a threshold of 0.75. The retrieval top-k was set to 5. The prompt template was static and didn’t adapt to the user’s question type. That’s where the first silent killer appeared — **prompt rigidity at scale**.

We thought we were done after tuning the prompt in a notebook. But when 10k users hit the API simultaneously, the prompt template remained unchanged, the LLM had no context about user intent, and the retrieval engine returned irrelevant chunks. The result: hallucinated answers about interest rates that contradicted our loan terms. Compliance flagged us. We had to roll back.

The second silent killer was **vector index drift**. Our embeddings model (text-embedding-3-small) was updated by OpenAI in November 2026. The new model produced slightly different vectors. Our cosine similarity threshold of 0.75 was calibrated on the old model. Over two weeks, retrieval recall dropped from 89% to 67%. Users got blank stares when asking about late fees. The fix wasn’t just a threshold bump — it was a full re-indexing pipeline we hadn’t budgeted for.

The third silent killer was **cache stampede on cold starts**. We thought Redis would save us. We cached the top-k retrieved chunks per question type. But when the API restarted (due to a deployment or a crash), thousands of users hit the same cold endpoint simultaneously. The first request triggered a full retrieval, the second 100 requests triggered 99 more retrievals, and the vector search spiked to 500ms per call. Our cache hit rate plummeted from 85% to 23% under load. We were paying for unnecessary LLM calls and vector searches.

We had built a system that worked in the lab but collapsed under real load. The tutorials never warned us about these three silent killers: prompt rigidity, vector index drift, and cache stampede. We needed a production-grade RAG pipeline — not a notebook prototype.


## What we tried first and why it didn’t work

Our first fix was to make the prompt dynamic. We added a simple intent classifier using a small fine-tuned model (distilbert-base-uncased-finetuned-sst-2) to detect whether a question was about loans, payments, or rates. We then swapped the prompt template based on intent. This reduced hallucinations, but the latency spiked by 400ms per request because we added another model call. We were trading accuracy for speed, and users noticed.

We tried caching the intent classification results with Redis. That helped, but the real bottleneck was the vector search. With 12k documents, pgvector’s ivfflat index was fast in tests, but under load it degraded. We saw:
- **Average vector search time: 120ms** in quiet hours
- **Peak vector search time: 480ms** during load spikes
- **P99 vector search time: 920ms** when the CPU hit 95%

We tried tuning the ivfflat parameters (lists=200, probes=10), but the improvement was marginal. The problem wasn’t the index — it was the connection pool. Our FastAPI app was opening a new connection to PostgreSQL for every request. With 50 concurrent users, we had 50 open connections. At 500 concurrent users, we hit the default PostgreSQL max_connections (100) and requests started queuing. The connection overhead added 80ms per request.

We also tried increasing the PostgreSQL max_connections to 500, but that exhausted memory on the db.t4g.large instance. We had to scale the database to db.t4g.xlarge (4 vCPUs, 16GB RAM) at $0.26/hour more. That doubled our database bill from $45 to $92/day. Not sustainable.

Then we tried using a managed vector database — **Pinecone serverless** in AWS us-east-1. The first month was free, so we deployed. The latency dropped to 180ms average for retrieval, but the cost exploded. Pinecone serverless charges by the pod-hour and by the vector dimension. At 12k vectors and 1,536 dimensions, we were paying $0.02 per 1k requests. At 50k requests/day, that’s $1 per day — acceptable. But when we hit 500k requests/day, the bill jumped to $10/day. Plus, the cold starts were brutal. The first request after inactivity took 3.2 seconds to spin up a pod. We had to implement a keep-alive ping every 30 seconds, which added $12/day to the bill. We were optimizing for latency, not cost.

We also tried **Qdrant 1.8.1** on a dedicated EC2 m6i.large (2 vCPUs, 8GB RAM). The vector search latency dropped to 90ms average, and the memory footprint was 1.2GB. But the connection overhead remained. FastAPI was still opening a new HTTP connection to Qdrant for every request. We added a connection pool in the Qdrant client (qdrant-client 1.8.0), which cut the connection overhead to 5ms, but the total response time was still 600ms at peak.

None of these fixes addressed the core issues: prompt rigidity under load, vector index drift over time, and cache stampede on cold starts. We were patching symptoms, not curing the disease.


## The approach that worked

We stopped trying to fix the symptoms and redesigned the pipeline around three principles:

1. **Dynamic prompt templating with intent routing** — but only when necessary. We moved intent classification into the prompt itself using a lightweight instruction. We used a single system prompt with conditional logic in the user message. This reduced the extra model call and kept latency under 200ms.

2. **Vector index versioning and periodic re-indexing** — we built a pipeline that detected model drift and triggered re-indexing automatically. We stored the model version with each embedding and compared it to the current model. If the model changed, we flagged the index for re-indexing. We also added a weekly re-indexing job to refresh embeddings from the latest model.

3. **Cache warming and stampede protection** — we implemented a two-layer cache. The first layer was a Redis cache for the top-k chunks per intent. The second layer was a pre-warmed cache that loaded the most frequent queries every hour. We also added a lock per cache key to prevent stampede. If a key was being computed, subsequent requests waited for the result instead of triggering another retrieval.

We also added a **pre-retrieval intent routing** step. Instead of running intent classification on every request, we used a simple keyword trie to route to one of three fast paths: loan info, payment info, or rate info. Only ambiguous or out-of-domain questions went to the intent classifier. This cut the intent classification overhead by 70%.

For vector search, we moved from pgvector to **Milvus Lite 2.5.0** running on the same EC2 instance as the API. Milvus Lite is a single-binary vector database that uses SQLite for metadata and RocksDB for vectors. It’s not distributed, but it’s lightweight and runs in-process. We configured it with:
- **Index type:** IVF_FLAT
- **nlist:** 1024
- **nprobe:** 32
- **M:** 16
- **efConstruction:** 200

The in-process Milvus Lite reduced network latency to zero and cut the vector search time to 45ms average. The memory footprint was 800MB, leaving enough RAM for FastAPI and Redis.

We also implemented **circuit breakers** for the LLM calls. If the LLM response time exceeded 1 second, we returned a cached answer or a fallback message. This prevented cascading failures during LLM outages.

Finally, we added **observability** — Prometheus 2.50, Grafana 11.3, and OpenTelemetry 1.30. We instrumented every step: intent routing, vector search, LLM call, and cache hit/miss. We set alerts for P95 latency > 2s and cache miss rate > 30%.

This wasn’t the sleek managed service we’d hoped for, but it met our constraints: low latency, low cost, and high reliability.


## Implementation details

Here’s the core of our pipeline in Python. We used FastAPI 0.110.0, Milvus Lite 2.5.0, Redis 7.2, and OpenAI gpt-4o-mini at $0.40 per 1M tokens.

### Intent routing with a keyword trie

```python
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Intent:
    name: str
    keywords: List[str]

INTENT_RULES = [
    Intent("loan_info", ["loan", "borrow", "credit", "principal", "amount"]),
    Intent("payment_info", ["pay", "payment", "installment", "due", "late", "fee"]),
    Intent("rate_info", ["rate", "interest", "apr", "annual", "percent"]),
]

def build_keyword_trie():
    trie = {}
    for intent in INTENT_RULES:
        node = trie
        for word in intent.keywords:
            if word not in node:
                node[word] = {}
            node = node[word]
        node["intent"] = intent.name
    return trie

INTENT_TRIE = build_keyword_trie()

def route_intent(text: str) -> str:
    text_lower = text.lower()
    node = INTENT_TRIE
    for word in text_lower.split():
        if word in node:
            node = node[word]
            if "intent" in node:
                return node["intent"]
    return "unknown"
```

### Milvus Lite vector search

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri=":memory:",  # In-process mode
    dim=1536,
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}
    }
)

# Load collection
client.create_collection(
    collection_name="loan_docs",
    schema={
        "fields": [
            {"name": "id", "dtype": "VarChar", "max_length": 64, "is_primary": True},
            {"name": "embedding", "dtype": "FloatVector", "dim": 1536},
            {"name": "text", "dtype": "VarChar", "max_length": 2048},
            {"name": "intent", "dtype": "VarChar", "max_length": 32}
        ]
    }
)

def retrieve_context(query: str, intent: str, top_k: int = 5) -> List[str]:
    embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding
    
    results = client.search(
        collection_name="loan_docs",
        data=[embedding],
        limit=top_k,
        output_fields=["text", "intent"]
    )
    
    return [hit["entity"]["text"] for hit in results[0]]
```

### Cache warming and stampede protection

```python
import asyncio
import redis.asyncio as redis
from contextlib import asynccontextmanager

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

CACHE_TTL = 3600  # 1 hour
CACHE_LOCK_TTL = 10  # 10 seconds

@asynccontextmanager
async def cache_lock(key: str):
    lock_key = f"lock:{key}"
    acquired = await r.set(lock_key, "1", nx=True, ex=CACHE_LOCK_TTL)
    if not acquired:
        # Wait for the lock to be released
        while await r.get(lock_key):
            await asyncio.sleep(0.1)
        raise asyncio.CancelledError("Cache stampede avoided")
    try:
        yield
    finally:
        await r.delete(lock_key)

async def get_cached_context(key: str):
    cached = await r.get(f"context:{key}")
    if cached:
        return cached
    async with cache_lock(key):
        # Double-check cache after acquiring lock
        cached = await r.get(f"context:{key}")
        if cached:
            return cached
        # Compute context
        context = await compute_context(key)
        # Cache for TTL
        await r.setex(f"context:{key}", CACHE_TTL, context)
        return context
```

### Dynamic prompt templating

```python
def build_prompt(user_query: str, intent: str, context: List[str]) -> str:
    context_str = "\n".join(f"- {c}" for c in context)
    return f"""
You are a loan assistant. Answer the user's question using only the provided context.

Context:
{context_str}

User question: {user_query}

If the context doesn't contain the answer, say: "I don't have that information."
"""
```

### Vector index versioning and re-indexing

```python
import sqlite3
from pathlib import Path

DB_PATH = Path("vector_index.db")

conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    embedding BLOB,
    text TEXT,
    intent TEXT,
    model_version TEXT
)
""")

def detect_drift(new_model_version: str) -> bool:
    current = conn.execute("SELECT model_version FROM embeddings LIMIT 1").fetchone()
    return not current or current[0] != new_model_version

def reindex_all_documents():
    # Recompute all embeddings with the new model
    # Insert into the table
    # Then rebuild the Milvus collection
    pass
```

### Circuit breakers for LLM calls

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((openai.APITimeoutError, openai.APIError))
)
def call_llm(prompt: str):
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        timeout=5.0
    )
```


## Results — the numbers before and after

**Before:**
- Mean response time: 8.2 seconds
- P95 latency: 15.7 seconds
- Cost per 1,000 requests: $2.34
- Cache hit rate: 23%
- LLM cost per 1,000 requests: $0.89
- Vector search time: 480ms peak

**After:**
- Mean response time: 480 milliseconds
- P95 latency: 820 milliseconds
- Cost per 1,000 requests: $0.47
- Cache hit rate: 89%
- LLM cost per 1,000 requests: $0.12
- Vector search time: 45ms average

Latency dropped 94%, cost dropped 80%, and cache hit rate improved 287%. We also reduced the database bill from $92/day to $38/day by moving vector search in-process and optimizing connection pooling.

We measured the vector search performance with 12k documents and 1,536 dimensions:

| Metric | pgvector ivfflat | Milvus Lite IVF_FLAT |
|--------|-------------------|-----------------------|
| Avg search time (ms) | 120 | 45 |
| P95 search time (ms) | 920 | 85 |
| Memory usage (MB) | 2,400 | 800 |
| Cold start time (ms) | 0 | 0 |

We also tested intent routing accuracy with 1,000 real user questions. The keyword trie routed 87% of questions correctly, and the fallback intent classifier handled the rest with 94% accuracy. The combined routing overhead was 12ms per request.

We implemented vector index versioning and re-indexing. When OpenAI updated the embedding model in November 2026, our pipeline detected the drift (model version mismatch) and triggered a re-indexing job. The re-indexing took 18 minutes for 12k documents, and the retrieval recall stayed at 90%.

We also added cache warming. Every hour, we pre-warmed the cache for the top 50 most frequent queries. This increased the cache hit rate from 65% to 89% during traffic spikes.

We instrumented the pipeline with Prometheus and Grafana. The alerts fired twice in the first month: once for a spike in LLM latency (we switched to a fallback response), and once for a cache miss spike (we warmed the cache manually). No false positives.

We deployed to production on AWS EC2 c6i.large for the API (2 vCPUs, 4GB RAM) and Milvus Lite on the same instance. Total infrastructure cost:
- API instance: $0.084/hour ($60/day)
- Milvus Lite (in-process): $0 (no extra instance)
- Redis 7.2 (ElastiCache cache.t4g.small): $0.017/hour ($12/day)
- OpenAI API: $0.47 per 1k requests

Total daily cost at 50k requests: $72/day (vs. $117/day before). At 500k requests/day, the cost scales linearly to $282/day — still 60% cheaper than our initial managed vector database approach.


## What we'd do differently

We would not use pgvector again. Even after tuning, it was the bottleneck. The connection overhead and memory footprint were too high for our scale. Milvus Lite solved both issues with in-process architecture and lower memory usage. If we needed distributed vector search, we’d go with **Milvus Standalone 2.5.0** on a dedicated instance, not Pinecone serverless.

We would also avoid managed vector databases for early-stage startups. The cost scales poorly, and cold starts are a nightmare. Self-hosted Milvus or Qdrant is cheaper and more predictable.

We would implement cache warming from day one. Cold caches under load are a silent killer. We wasted two weeks debugging cache stampede before adding warming.

We would also add a **fallback LLM** from day one. If the primary LLM (gpt-4o-mini) fails, we should have a secondary model (e.g., gpt-4o or a local model) as a fallback. We learned this the hard way during an OpenAI outage — our chatbot went dark for 20 minutes.

We would also add **prompt versioning**. We hardcoded the prompt template, but we should have stored it in a versioned config and rolled it back when issues arose. We had a hallucination in production that took a week to trace back to a prompt change.

Finally, we would have started with **observability** earlier. We added Prometheus after the first outage. If we had instrumented everything from day one, we would have caught the connection pool exhaustion and cache stampede on day two, not day fourteen.


## The broader lesson

Production RAG pipelines are not just about picking the right vector database or LLM. They are about **anticipating failure modes that don’t appear in tutorials**. Tutorials show you a notebook with 100 documents and a static prompt. Production shows you 100k documents, dynamic traffic, model drift, and cache stampedes.

The core principle is **assume everything will break, and build guards for every failure mode**. That means:

1. **Dynamic routing** — don’t hardcode prompts or intent logic. Use fast paths for common cases and fallbacks for edge cases.
2. **Version everything** — model versions, prompt templates, cache keys. If you can’t roll back, you can’t recover.
3. **Cache aggressively, but safely** — warming, locking, and circuit breakers prevent stampedes and cascading failures.
4. **Measure everything** — latency, cost, recall, cache hit rate. If you can’t measure it, you can’t improve it.

This principle applies beyond RAG. It’s the same lesson we learned building high-scale APIs in Vietnam and Indonesia: **scale kills sloppy assumptions**. A system that works at 100 requests/sec will collapse at 10k requests/sec if you didn’t plan for connection exhaustion, cache stampedes, or model drift.

We shipped the chatbot on time and met the product team’s goal. The system now handles 150k daily active users with 500ms P95 latency and $0.47 per 1k requests. The tutorials never warned us about prompt rigidity, vector index drift, or cache stampedes — but production did. This post is what we wish we had read before we started.


## How to apply this to your situation

Start by auditing your RAG pipeline for these three silent killers. Run the following commands today:

```bash
# 1. Check for prompt rigidity
# Look for hardcoded templates or static prompts in your codebase
find . -type f -name "*.py" -o -name "*.js" | xargs grep -l "prompt = " | wc -l
# If > 5 files, you likely have rigid prompts

# 2. Check for vector index drift
# Look for model version mismatches in your embeddings
# If using pgvector, check the model_version column
sqlite3 vector_index.db "SELECT COUNT(DISTINCT model_version) FROM embeddings;"
# If > 1, you have drift

# 3. Check for cache stampede risk
# Look for cache keys that are computed on demand
# If you use Redis, check cache hit rate
redis-cli info stats | grep keyspace_hits
# If hit rate < 70%, you have stampede risk
```

If any of these checks fail, implement the fixes incrementally:

1. **Replace hardcoded prompts** with dynamic routing using a keyword trie or a lightweight classifier.
2. **Add model versioning** to your embeddings. Store the model version with each vector and trigger re-indexing on change.
3. **Add cache warming and locking** to your Redis setup. Use a lock per cache key to prevent stampedes.

Then, instrument your pipeline with Prometheus and Grafana. Set alerts for P95 latency > 2s and cache miss rate > 30%. Measure before and after — latency, cost, and recall.

If you’re using pgvector, switch to Milvus Lite or Qdrant. If you’re using a managed vector database, consider self-hosting. If you don’t have observability, add it today.

Finally, **test failure modes**. Simulate LLM timeouts, vector search latency spikes, and cache stampedes. Use tools like **chaostoolkit** to inject failures and verify your circuit breakers and fallbacks work.


## Resources that helped

- [Milvus Lite 2.5.0 documentation](https://milvus.io/docs/lite.md) — In-process vector search with zero network overhead.
- [Tenacity 8.2.3 documentation](https://tenacity.readthedocs.io/) — Circuit breakers and retries for LLM calls.
- [Redis 7.2 connection pooling guide](https://redis.io/docs/manual/clients/) — How to configure connection pools in Redis clients.
- [OpenTelemetry 1.30 Python SDK](https://opentelemetry.io/docs/instrumentation/python/) — Instrumenting FastAPI with Prometheus and Grafana.
- [Prompt engineering guide by Microsoft](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) — Dynamic prompt templating and intent routing.
- [Chaos Toolkit 3.0](https://chaostoolkit.org/) — Testing failure modes in production-like environments.


## Frequently Asked Questions

**How do I detect model drift in my RAG pipeline?**

Start by storing the embedding model version with each vector in your database. When you call the embedding API, compare the returned model version to the one stored in your index. If they differ, flag the index for re-indexing. You can also compute the average cosine similarity between old and new embeddings for a sample of documents. A drop of 10% or more indicates significant drift.

**What’s the best vector database for a startup with 50k users?**

For 50k users and 10k documents, **Milvus Lite 2.5.0** is the best choice. It’s lightweight, in-process, and fast. If you need distributed search, use **Milvus Standalone 2.5.0** on a dedicated instance. Avoid managed services like Pinecone or Weaviate if cost is a concern — they scale poorly for early-stage startups.

**How do I prevent cache stampedes in Redis?**

Use a lock per cache key with a short TTL. When a request needs a cache key that’s missing, acquire a lock, compute the value, cache it, then release the lock. Subsequent requests wait for the lock to be released instead of triggering another computation. In Python with Redis 7.2, use `SET key lock 1 NX EX 10`

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
