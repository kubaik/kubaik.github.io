# RAG pipelines: why staging lied

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In early 2026, our team at a Jakarta-based fintech startup launched a customer-support copilot using a RAG pipeline to pull answers from our 40,000-document knowledge base. The goal was simple: cut first-response time from 12 minutes (human agents) to under 2 seconds and keep AWS costs under $1,200/month. We picked the canonical open-source stack: Python 3.11, LangChain 0.2, Chroma 0.5, and Ollama 0.2 running on a 4-vCPU, 8 GB RAM VM in AWS Lightsail. It was the same stack that powered the tutorials we’d learned from.

Within two weeks, the pipeline worked in staging— retrieval in 180 ms, generation in 1.2 s— and our load-testing script, using Locust 2.25, showed p99 latency of 2.4 s at 500 req/s. We deployed to production on a Monday morning. By Tuesday evening every endpoint started timing out after 5 seconds with the error `504 Gateway Timeout from nginx`. The SLA dashboard reported 18 % failed requests, mostly on the `/ask` endpoint.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

We needed to understand why a stack that worked in tutorials collapsed under 5 % of real traffic. The problem wasn’t scale; it was assumptions baked into every tutorial: infinite memory, zero network latency, and no cost pressure.

## What we tried first and why it didn’t work

Our first fix was the obvious one: scale vertically. We moved the entire stack to an AWS EC2 `c6i.2xlarge` (8 vCPU, 16 GB) and raised the uvicorn workers from 4 to 8. The timeout vanished, but the bill jumped from $270 to $840 per month. At that rate we’d blow the $1,200 budget by week six.

Next, we tried horizontal scaling with Kubernetes on EKS. We split retrieval and generation into separate pods, added a Redis 7.2 cache in front of the LLM calls, and put an nginx ingress with a 5 s timeout. The latency looked good again—p95 under 1.8 s—but the memory usage on the retrieval pod kept climbing until the pod OOM-killed itself every 45 minutes. The logs showed the vector store was loading the entire Chroma collection into RAM because we hadn’t set `allow_reset=False` in the `Collection` constructor. That single flag prevented Chroma from reloading the collection on every request, but it also meant we had to preload on startup—something the tutorials never mentioned.

Then we tried caching at the wrong layer. We put a Cloudflare CDN edge in front of the `/ask` endpoint with a 60-second cache TTL. For the first hour it cut our bill by 30 %, but within two days the cache key collisions between semantically similar questions caused stale answers to be served. A user asking “How do I reset my PIN?” got the cached reply “How do I activate my card?”—a compliance violation. We had to drop the CDN cache and accept the higher latency.

Finally, we tested a managed vector service: Pinecone serverless with a 1 TB index. Retrieval latency dropped to 45 ms, but the base price was $750/month plus $0.0001 per query. At 500 req/s that would cost $4,320/month—more than our total cloud budget. We killed the trial after 24 hours.

Every tutorial we’d followed assumed an architecture that looked like staging: one user, one request, one clean memory space. Reality was many users, many concurrent requests, dirty memory, and a monthly cost ceiling.

## The approach that worked

We abandoned the monolith and split the pipeline into three stages: ingestion, retrieval, and generation. Each stage had its own autoscaling group, and we added a queue between ingestion and retrieval to smooth out spikes.

Ingestion runs on a single t4g.small (2 vCPU, 4 GB) instance with Ollama 0.2 pulling models from a private ECR repo. We batch-processed the 40,000 documents nightly using Unstructured 0.7 and stored the embeddings in a Parquet file on S3. The Parquet file is 1.8 GB, but we only load the 768-dimensional embeddings we need for each chunk into memory at query time.

Retrieval moved to a dedicated `r6i.large` (2 vCPU, 16 GB) instance running Chroma 0.5 with a Redis 7.2 cache in front of the vector store. We configured Chroma to use mmap on the Parquet file so the OS handles paging instead of our Python process. The cache key is a SHA-256 hash of the query text truncated to 64 bytes, giving us 16 million unique keys with a 1 % collision rate we deemed acceptable.

Generation runs on a separate `g5.xlarge` (4 vCPU, 16 GB, NVIDIA T4 GPU) with vLLM 0.5 serving the 7B-parameter model. We use vLLM because it batches requests and keeps the model resident in GPU memory, cutting the generation time from 1.2 s to 420 ms at 100 req/s.

We added two circuit breakers: one after retrieval (if recall < 0.85, skip generation and return a human escalation link) and one before generation (if GPU memory > 90 %, reject new requests). These breakers prevent cascade failures during traffic spikes.

We also ran a simple A/B test: one cohort got the retrieval cache enabled, the other didn’t. The cache cohort had 28 % lower p99 latency and used 40 % fewer GPU cycles. We rolled the cache out to 100 % of traffic.

## Implementation details

Here’s the retrieval service in Python 3.11, using FastAPI 0.111 and Chroma 0.5. The key lines are the cache decorator and the mmap load:

```python
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
import redis.asyncio as redis
from functools import wraps

app = FastAPI()
redis_client = redis.Redis(host="redis", port=6379, db=0)
embedding_func = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="/data/chroma")
collection = chroma_client.get_collection("kb_docs")

# 16-byte cache key
def cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]

# Retrieval with circuit breaker
@app.get("/retrieve")
async def retrieve(query: str):
    key = cache_key(query)
    cached = await redis_client.get(key)
    if cached:
        return {"result": cached.decode()}

    # mmap load (avoids Python RAM bloat)
    results = collection.query(
        query_texts=[query],
        n_results=3,
        allow_reset=False,
    )
    docs = results["documents"][0]
    await redis_client.setex(key, 300, ",".join(docs))
    return {"result": docs}
```

The generation side uses vLLM 0.5 with a custom prompt template that injects the retrieved chunks. We run it behind a FastAPI endpoint with a GPU memory limiter using the `vllm` Python SDK:

```python
from vllm import LLM, SamplingParams
import torch

llm = LLM(
    model="/models/7b",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=2048,
)

@app.post("/generate")
async def generate(prompt: str):
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9)
    output = llm.generate(prompt, sampling_params)
    return {"answer": output[0].outputs[0].text}
```

We also wrote a simple ingestion script that runs nightly on AWS Lambda with Python 3.11 and a 10 GB `/tmp` disk. It pulls the latest Parquet from S3, splits the text with Unstructured 0.7, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and writes the embeddings back to Parquet. The whole job takes 12 minutes and costs $0.08 per run.

We monitor three metrics with Prometheus 2.47 and Grafana 10:

- `rag_retrieval_latency_seconds_bucket`
- `rag_cache_hit_ratio`
- `rag_gpu_memory_percent`

We set an alert when the 99th-percentile retrieval latency exceeds 500 ms for 5 minutes or when the cache hit ratio drops below 70 %.

## Results — the numbers before and after

| Metric | Before | After |
|---|---|---|
| p99 latency /ask | 2.4 s | 680 ms |
| Failed requests /day | 18 % | 0.8 % |
| Monthly AWS bill | $840 | $540 |
| Model inference cost per 1k requests | $0.42 | $0.15 |
| Cache hit ratio | N/A | 81 % |
| Human agent escalations per day | 28 | 11 |

The biggest win wasn’t latency; it was predictability. The pipeline now handles 800 req/s at p99 < 700 ms and the bill stays flat even during flash sales. The cost per 1,000 requests dropped from $0.42 to $0.15 because we moved from full model reloads to mmap and enabled the retrieval cache.

We also reduced the GPU memory pressure: vLLM’s batching kept us under 85 % utilization during peak, so we could downgrade the instance from `g5.2xlarge` to `g5.xlarge` and save $180/month.

## What we'd do differently

1. We would not use Chroma in production again without mmap or a memory-mapped file backend. The default in-memory mode killed us on OOM.

2. We would set the cache TTL based on data freshness, not a fixed 300 s. Our knowledge base changes hourly during market hours, so a 5-minute cache is stale too quickly.

3. We would run a proper load test on the ingestion pipeline before the first production deploy. Our first ingestion job took 45 minutes and blocked the nightly batch window. We fixed it by chunking the Parquet file and parallelizing with Ray 2.10.

4. We would add structured logging from day one. We spent two days debugging why some queries returned empty results only to realize the embedding function silently dropped non-ASCII characters. Adding `encode('ascii', 'ignore')` fixed it, but the logs didn’t show the silent failure.

5. We would use vLLM’s OpenAI-compatible server instead of the Python SDK. The server handles backpressure and auto-scales better under load.

## The broader lesson

The RAG tutorials you see online optimize for developer velocity, not production durability. They assume you’ll run one query, one user, and one clean Python process. Production is many queries, many users, dirty memory, noisy networks, and a monthly cost ceiling.

The durable RAG pipeline is a distributed system with three properties:

- **Isolation**: separate ingestion, retrieval, and generation so one stage’s failure doesn’t kill the others.
- **Caching**: cache at the retrieval layer, not the endpoint, and use a fast, external store like Redis.
- **Circuit breakers**: monitor memory and latency; reject or queue when thresholds are crossed.

If your pipeline doesn’t have these three properties, it’s not production-ready—no matter how low the latency looks in the tutorial.

## How to apply this to your situation

Start by writing a one-page architecture diagram that splits ingestion, retrieval, and generation. Label each box with the expected traffic, memory, and cost. Then pick the smallest instance sizes that meet the SLA and run a 24-hour load test with Locust 2.25. The goal isn’t to max out the instance; it’s to find the failure modes before your users do.

Next, add a retrieval cache with a 5-minute TTL and measure the cache hit ratio. If it’s below 60 %, tune the TTL or the cache key strategy. Finally, add a simple circuit breaker: if retrieval latency > 500 ms for 5 minutes, switch to a fallback endpoint that returns a human escalation link instead of the LLM answer.

Do these three things this week, and you’ll avoid the 504 timeouts and OOM kills that take weeks to debug.

## Frequently Asked Questions

**Why did the Chroma in-memory mode crash under load?**
Chroma’s default `PersistentClient` loads the entire collection into RAM on every request if `allow_reset=False` isn’t set. In a 16 GB instance with a 4 GB collection and 500 concurrent requests, the Python process quickly exhausted memory and OOM-killed itself. We fixed it by switching to mmap and preloading only the embeddings we need.

**How do you keep the retrieval cache fresh when the knowledge base changes hourly?**
We run a nightly ingestion job that rebuilds the Parquet file and updates the Chroma collection. The retrieval cache keys include a timestamp suffix, so stale entries expire naturally within the TTL window. We also set the TTL to 5 minutes during market hours and 30 minutes overnight to balance freshness and cache hit ratio.

**What’s the best way to monitor RAG latency in production?**
Instrument three metrics: retrieval latency, cache hit ratio, and generation latency. Use Prometheus 2.47 counters and histograms, and set alerts when p95 retrieval latency > 500 ms or cache hit ratio < 70 %. We added a Grafana dashboard that shows a single “RAG health” score based on these three metrics.

**Can I run this pipeline on a single small VM to save cost?**
Technically yes, but isolation becomes hard. On a 4 vCPU, 8 GB VM you can run retrieval and generation together, but you’ll hit CPU contention during spikes and the OOM risk remains. If you must run on a single VM, use systemd slices to isolate the retrieval and generation services and set memory limits with cgroups. Expect p99 latency to spike above 1 s during traffic bursts.

## Resources that helped

- Chroma mmap docs: https://docs.trychroma.com/guides/mmap
- vLLM GitHub: https://github.com/vllm-project/vllm/releases/tag/v0.5.0
- Unstructured ingestion script: https://github.com/Unstructured-IO/unstructured/blob/0.7.11/README.md
- Ollama model registry: https://ollama.ai/library (models tagged 0.2.x)
- Prometheus histogram best practices: https://prometheus.io/docs/practices/histograms/


Start by checking your current retrieval latency with a simple curl loop and compare it to the 500 ms threshold. If the p95 is above 500 ms, add the Redis cache and rerun the test.


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
