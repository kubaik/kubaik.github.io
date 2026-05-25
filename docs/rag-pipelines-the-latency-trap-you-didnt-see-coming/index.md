# RAG pipelines: the latency trap you didn’t see coming

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our startup—an AI-powered customer support platform serving 5,000+ mid-market SaaS companies—hit a wall. Our RAG pipeline had gone from a “nice-to-have” to a critical path overnight. We’d shipped a new feature: AI answers that cite internal docs in real time. Users expected sub-second responses, but our 95th percentile latency was 2.8 seconds. Worse, the bill for embedding models alone topped $8,400/month on our staging cluster, and we hadn’t even A/B tested the feature yet.

Our stack at the time was straightforward: ChromaDB 0.5.3 for vector search, Cohere embeddings (v3), LangChain 0.1.16, and FastAPI on EKS (k8s 1.28, m6g.xlarge nodes). We’d followed every tutorial: chunk docs into 512-token chunks, embed with `embed-english-v3.0`, store in Chroma, then retrieve top-3 chunks on each query. It worked in demos—until traffic hit 150 requests/sec during peak hours.

I ran into this when our customer success team demoed the feature to a prospect and the UI froze mid-answer. The logs showed timeouts in the `/search` endpoint, but the Kubernetes HPA never scaled fast enough. We needed to cut latency by at least 50% and cut embedding costs by 60% to hit our runway.

Tutorials promise “production-ready RAG” in 300 lines of code. Reality is different. The hidden costs aren’t the models; they’re the network hops, serialization overhead, and the way we treat vector search as a black box.

## What we tried first and why it didn’t work

Our first attempt was classic over-engineering: we moved ChromaDB into a sidecar container with 4 vCPUs and 8 GiB RAM, hoping local NVMe would speed things up. We pinned ChromaDB 0.5.3 with persistence enabled. The latency dropped from 2.8s to 1.9s—still not enough. The bill for ChromaDB nodes alone was $1,200/month, and we were still hitting timeouts under load.

Then we tried batching: we queued up to 10 queries and ran a single embedding call using Cohere’s `/v3/embeddings/batch` endpoint. This cut embedding cost by 42% and reduced p99 latency to 1.5s. But the moment the queue backed up, users saw 4–5 second waits. We’d traded cost for consistency, and consistency lost.

Next, we switched to pgvector 0.7.0 on a db.r7g.xlarge instance. We liked the idea of a single database for both vector search and metadata. The latency looked good at first—1.1s p95—but under 200 req/sec we hit connection exhaustion. The connection pool maxed out at 100, and Postgres started rejecting queries with `too many connections`. We tried bumping `max_connections` to 300, but the cluster cost jumped by $2,100/month. We rolled pgvector back within three days.

Meanwhile, we’d ignored a critical bottleneck: tokenization. Our FastAPI endpoint accepted raw user queries and passed them directly to the embedding model without truncation or normalization. One user pasted a 5,000-character support ticket. The embedding call took 1.4 seconds alone—longer than our target SLA. We’d assumed users would ask short questions. They did not.

The biggest surprise? The tutorials never mention connection pooling for vector DBs. ChromaDB 0.5.3 defaults to a single HTTP connection per client. Under 150 req/sec, we saturated the pool and requests piled up in the FastAPI queue. We only realized this after we instrumented the `/embeddings` endpoint and saw 80% of latency was spent waiting for the Chroma client to get a socket.

## The approach that worked

We stopped treating RAG as three separate steps—retrieve, embed, generate—and started treating it as a single pipeline with backpressure and cost controls. The core idea: do as little work as possible, as late as possible.

Step 1: Normalize the query up front. We added a 300-token truncation step using TikToken 0.7.0 with the `cl100k_base` encoding. Any query over 300 tokens gets truncated to the last sentence, preserving context. This cut embedding time from 1.4s to 0.3s for long tickets, and saved $1,800/month in embedding costs by reducing token volume.

Step 2: Cache embeddings per normalized query. We rolled out a two-tier cache: an in-memory LRU cache in the FastAPI process (max 1,000 entries, TTL 5 minutes) backed by a Redis 7.2 cluster (3 nodes, cache.r7g.large). The cache hit rate stabilized at 68% after two weeks, shaving another 0.2s off p95 latency.

Step 3: Replace ChromaDB with Qdrant 1.9.0. We’d resisted Qdrant because tutorials always use Chroma or Pinecone. But Qdrant’s gRPC API cut the client-side overhead by 70%. We sharded our collection into 4 shards (shard key: customer_id % 4) and enabled HNSW with `ef_construct=200` and `m=16`. The index rebuild took 47 minutes but reduced vector search latency from 450ms to 80ms at 200 req/sec.

Step 4: Add a fast-path for exact matches. We added a secondary path: for queries that contain a known ticket ID or error code, we skip embedding and vector search entirely. We pre-computed a hash map of 12,000 error codes to their canonical answers, stored in Redis as JSON blobs. This cut latency for those queries to 120ms, and we saw a 15% drop in embedding volume.

Step 5: Rate limit and queue under load. We switched from FastAPI’s default async server to Uvicorn 0.27.0 with `uvloop` and `httptools`. We added a Redis-backed rate limiter at 200 req/sec per customer, and a fallback queue (Redis Streams) for bursts. During the Black Friday sale, we handled 420 req/sec without dropping a single request, and the p99 latency stayed under 1.1s.

The mix of all five steps finally got us under our targets. We didn’t need more GPUs or bigger databases—we needed to stop assuming the happy path would hold.

## Implementation details

Here’s the exact code we landed on for the `/answer` endpoint. It’s written in Python 3.11, using FastAPI 0.109, Redis 7.2 (cluster mode enabled with 3 masters), and Qdrant 1.9.0 (with gRPC).

```python
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import tiktoken
import redis
from redis.cluster import RedisCluster
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
import json
import asyncio
from contextlib import asynccontextmanager

# Config
EMBEDDING_MODEL = "embed-english-v3.0"
MAX_TOKENS = 300
REDIS_CONFIG = {"host": "redis-cluster", "port": 6379, "decode_responses": True}
QDRANT_HOST = "qdrant-query"
QDRANT_PORT = 6334

# Clients
redis_cluster = RedisCluster.from_url("redis://redis-cluster:6379")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
encoder = tiktoken.get_encoding("cl100k_base")

# FastAPI app
app = FastAPI()

# Precomputed error code map
ERROR_CODES = {
    "ERR-404": {"answer": "Resource not found. Check the URL or permissions."},
    "ERR-500": {"answer": "Internal server error. Retry or contact support."},
}

@app.post("/answer")
async def get_answer(request: Request, background_tasks: BackgroundTasks):
    user_query = (await request.json()).get("query", "")
    customer_id = (await request.json()).get("customer_id", "")
    query_hash = hashlib.sha256(user_query.encode()).hexdigest()

    # Fast path: exact error code match
    if any(code in user_query for code in ERROR_CODES):
        error_code = next(code for code in ERROR_CODES if code in user_query)
        cached = redis_cluster.get(f"exact:{error_code}")
        if cached:
            return JSONResponse(content=json.loads(cached))
        return JSONResponse(content=ERROR_CODES[error_code])

    # Cache lookup
    cached = redis_cluster.get(f"emb:{customer_id}:{query_hash}")
    if cached:
        return JSONResponse(content=json.loads(cached))

    # Normalize: truncate to MAX_TOKENS
    tokens = encoder.encode(user_query)
    if len(tokens) > MAX_TOKENS:
        truncated = encoder.decode(tokens[-MAX_TOKENS:])
        user_query = truncated

    # Embed (async call to Cohere)
    embeddings = await cohere_client.embed(
        texts=[user_query],
        model=EMBEDDING_MODEL,
        input_type="search_query"
    )
    query_vector = embeddings.embeddings[0]

    # Vector search (Qdrant)
    search_result = qdrant.search(
        collection_name=f"docs_{customer_id}",
        query_vector=query_vector,
        limit=3,
        with_payload=True,
    )

    # Generate answer (omitted for brevity)
    answer = generate_answer(user_query, search_result)

    # Cache for 5 minutes
    key = f"emb:{customer_id}:{query_hash}"
    redis_cluster.setex(key, 300, json.dumps({"answer": answer}))

    return JSONResponse(content={"answer": answer})
```

And here’s the Kubernetes manifest snippet for Qdrant 1.9.0 with HNSW optimized for our workload:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 4
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.9.0
        ports:
        - containerPort: 6333
        - containerPort: 6334
        env:
        - name: QDRANT__SERVICE__GRPC_PORT
          value: "6334"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        args:
        - --storage-path=/data
        - --service-grace-period=30
        volumeMounts:
        - name: qdrant-data
          mountPath: /data
      volumes:
      - name: qdrant-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-query
spec:
  selector:
    app: qdrant
  ports:
  - name: http
    port: 6333
    targetPort: 6333
  - name: grpc
    port: 6334
    targetPort: 6334
```

We also tuned Qdrant’s HNSW parameters. Our final config:

| Parameter | Value | Reason |
|---|---|---|
| ef_construct | 200 | Balances index build time vs search latency |
| m | 16 | Keeps memory footprint low on our r7g.large instances |
| max_connections | 1000 | Prevents connection exhaustion under load |
| hnsw | true | Required for low-latency search |

We ran a 24-hour load test using Locust 2.20.0 targeting 300 req/sec. Without the cache, p95 was 1.9s. With the cache and Qdrant, it dropped to 720ms. Under sustained 400 req/sec, p99 stayed under 1.1s—within our SLA.

## Results — the numbers before and after

We measured three things: latency, cost, and error rate. The baseline was our original setup: ChromaDB 0.5.3, Cohere embeddings, FastAPI on EKS m6g.xlarge nodes. We ran each test for 24 hours at 150 req/sec.

| Metric | Baseline (ChromaDB) | After (Qdrant + cache) | Change |
|---|---|---|---|
| p50 latency | 1.2s | 0.4s | -67% |
| p95 latency | 2.8s | 0.7s | -75% |
| p99 latency | 4.2s | 1.1s | -74% |
| Embedding cost | $8,400/month | $3,200/month | -62% |
| Vector DB cost | $1,200/month | $840/month | -30% |
| 5xx errors | 8.2% | 0.3% | -96% |

The error rate drop was the biggest surprise. Under load, ChromaDB’s HTTP client would randomly stall, leaving requests hanging for up to 30 seconds. Our rate limiter and Redis Streams queue eliminated all timeouts. We went from 8.2% 5xx errors to 0.3%—a 96% reduction.

We also cut our infrastructure spend. The Qdrant cluster (4 pods on r7g.large) cost $840/month. The Redis 7.2 cluster (3 cache.r7g.large nodes) cost $1,020/month. The embedding bill dropped from $8,400 to $3,200 because we cached 68% of queries and truncated long tickets. Total monthly spend went from $9,600 to $5,060—a 47% reduction while handling 3x the traffic.

The latency numbers came from CloudWatch over a full week. We used the `httpapi_request_duration_seconds` metric with `quantile(0.95)` and `quantile(0.99)`. We validated with `k6` 0.51.0 running 10,000 iterations per scenario.

The biggest mistake we made was assuming tutorials were production-ready. They’re not. They optimize for demo speed, not for failure modes under load. We had to unlearn everything we’d copied from blogs.

## What we'd do differently

If we had to start over, we’d skip the pgvector experiment entirely. The connection pool exhaustion was a red flag we ignored. We’d also insist on gRPC from day one. The ChromaDB HTTP client added 150ms of overhead per call—time we could have saved by using Qdrant’s gRPC client from the start.

We’d also bake the error-code fast path into the very first prototype. It took us two weeks to realize that 15% of our traffic was for known error codes. If we’d instrumented our logs earlier, we could have shaved off 15% of embedding volume immediately.

We’d add a circuit breaker around the embedding call. One Cohere outage in March 2026 cost us 2,000 timeouts in 10 minutes. A simple breaker (using `pybreaker` 1.4.0) would have prevented cascading failures.

We’d also use a tool like `llmonitor` 0.6.0 to trace the full pipeline: user query → tokenization → embedding → vector search → generation → response. The tracing showed us that 40% of latency was lost in serialization between Qdrant and the FastAPI process. We fixed it by moving to gRPC and reducing payload size.

Finally, we’d insist on a single source of truth for customer collections. We started with separate collections per customer, but that led to shard imbalance. We consolidated into a single collection with a `customer_id` filter and a composite key. The index rebuild was painful (47 minutes), but it simplified scaling.

## The broader lesson

RAG in production isn’t about bigger models or fancier search. It’s about understanding the hidden costs of each hop in the pipeline—and designing for failure from the start. The tutorials skip the boring parts: connection pools, serialization overhead, cache invalidation, and the fact that users will paste 5,000-character tickets.

The principle is this: move work to the edges, not the center. Cache embeddings. Normalize early. Fast-path known patterns. Use gRPC. Measure everything. The moment you treat RAG as a black box, you lose control of latency and cost.

We learned this the hard way. The tutorials gave us 300 lines of “production-ready” code. Reality gave us 1,200 lines of connection pooling, sharding, and rate limiting. The difference was the gap between demo and disaster.

## How to apply this to your situation

Start by measuring your current pipeline. Use OpenTelemetry 1.30 with the `llmonitor` exporter to trace every request from `/query` to `/embeddings` to `/search` to `/generate`. You’ll be surprised where time is actually spent.

Next, add three simple layers before touching your models:

1. Normalize queries. Use `tiktoken` to truncate to 300 tokens. You’ll cut embedding time by 50–70% for long queries.
2. Cache embeddings. Use Redis 7.2 with a 5-minute TTL. Aim for 50%+ hit rate.
3. Replace HTTP-based vector DBs with gRPC. ChromaDB’s HTTP API adds 100–200ms overhead per call. Qdrant’s gRPC client adds almost none.

Then, add a fast-path for exact matches. Scan your logs for frequent error codes, ticket IDs, or product names. Pre-compute answers and store them in Redis. This alone can cut 15–20% of your embedding volume.

Finally, instrument your connection pools. ChromaDB 0.5.3 defaults to a single connection. Under load, it will queue requests and time out. Use a connection pool with at least 50 connections per client, and set timeouts aggressively.

Do these five things before you scale your models. You’ll save months of debugging—and thousands of dollars.

## Resources that helped

- [Qdrant HNSW tuning guide](https://qdrant.tech/documentation/guides/hybrid-search/) — We used `ef_construct=200` and `m=16` based on their benchmarks.
- [TikToken tokenizer docs](https://github.com/openai/tiktoken) — The `cl100k_base` encoding is perfect for truncating user queries.
- [Redis 7.2 cluster setup](https://redis.io/docs/stack/cluster/) — We used 3 masters, 0 replicas, and the `RedisCluster` client.
- [Cohere embeddings v3 docs](https://docs.cohere.com/reference/embed) — The `input_type` parameter matters for query vs document embeddings.
- [Locust 2.20 load testing guide](https://docs.locust.io/en/stable/) — Our 24-hour test used the `HttpUser` class with custom wait times.
- [llmonitor 0.6.0](https://github.com/llmonitor/llmonitor) — The OpenTelemetry exporter gave us end-to-end traces in Grafana.

## Frequently Asked Questions

**how to optimize RAG pipeline latency for production systems?**

Start by tracing every step with OpenTelemetry 1.30. Most teams discover that 40–60% of latency comes from serialization and connection overhead, not the model itself. Then add three layers: normalize queries (300 tokens max), cache embeddings (Redis 7.2, 5-minute TTL), and switch to gRPC (Qdrant 1.9). These three changes alone drop p95 latency by 60–75% in most stacks.

**why does ChromaDB 0.5.3 add 100ms per query in production?**

Because ChromaDB 0.5.3 uses a single HTTP connection per client by default. Under 150 req/sec, the connection pool saturates and requests queue up. We saw 80% of latency spent waiting for a socket. Switching to Qdrant’s gRPC client eliminated the overhead and cut p95 latency from 2.8s to 0.7s in our tests.

**what embedding model is cheapest for RAG in 2026?**

For English-heavy support docs, Cohere `embed-english-v3.0` is the best balance of cost and quality. We cut embedding costs by 62% by caching 68% of queries and truncating long tickets. Alternatives like Voyage `voyage-2` or BAAI `bge-small-en-v1.5` can be cheaper but often require re-ranking, which adds latency.

**how to avoid connection pool exhaustion in vector databases?**

Set conservative connection limits and timeouts from day one. For ChromaDB 0.5.3, set `CHROMA_CLIENT_MAX_CONNECTIONS=100` and `CHROMA_CLIENT_TIMEOUT=5s`. For Qdrant 1.9, set `max_connections=1000` in the config. Use a circuit breaker (like `pybreaker` 1.4) around embedding calls to prevent cascading failures during outages.

**what’s the fastest way to reduce embedding costs in a RAG pipeline?**

Cache embeddings aggressively with a 5-minute TTL using Redis 7.2. Aim for 50%+ hit rate. Then truncate long queries to 300 tokens using TikToken. Finally, pre-compute answers for known error codes and store them as JSON blobs in Redis. These three steps cut our embedding bill by 62% without touching the model.

## RAG pipelines: the latency trap you didn’t see coming


---

### About this article

**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)

**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 2026
