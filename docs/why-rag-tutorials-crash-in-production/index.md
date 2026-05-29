# Why RAG tutorials crash in production

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were building a customer-support chatbot for a Southeast Asian fintech with 1.2 million monthly active users. The bot had to answer questions about transaction limits, card fees, and dispute procedures. The stakes weren’t academic: wrong answers meant chargebacks and regulator escalations.

Our first architecture was a textbook RAG pipeline: user query → embedding model (text-embedding-3-small) → vector search against a 15 GB chunked knowledge base → reranker (bge-reranker-base) → LLM (Qwen2.5–72B-Instruct) → final answer. We picked the smallest embedding model to keep latency low, but we didn’t account for how fast the knowledge base would grow. By month two, the index had ballooned to 40 GB, and 95th-percentile response time hit 2.8 s—double the SLA.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout. This post is what I wished I had found then.

The real problem wasn’t just speed; it was the hidden cost of scale. At 20 queries/sec, our AWS bill for the embedding model alone was $6.8 k/month. That’s 42 % of our entire AI budget. We needed answers that stayed under 1.2 s and cost under $2 k/month, or the project would get killed.

## What we tried first and why it didn’t work

### Attempt 1: Bigger embedding model, fewer dimensions

We switched to text-embedding-3-large and increased dimension from 1536 to 3072 to capture more nuance. The hope was that higher quality embeddings would shrink the reranker’s candidate set, so we could drop the index size. Result: reranker latency doubled, total cost rose to $11 k/month, and recall actually dropped 8 % because the larger index had more false positives. We rolled back after a week.

### Attempt 2: Smaller reranker and no reranking at all

Next, we tried removing the reranker entirely and just taking the top-5 chunks from the vector search. Precision crashed: 34 % of answers were flat-out wrong on factual questions. Support tickets spiked 180 %. We added a lightweight BERT reranker (distilbert-base-uncased), but it added 320 ms and still gave 12 % incorrect answers.

### Attempt 3: Sharding the index by region

We split the knowledge base into 5 regional shards, hoping to reduce search space. The good news: 95th-percentile latency fell to 1.5 s. The bad news: every query now required 5 parallel searches, and our embedding bill jumped to $9.2 k/month because we were running 5× embeddings in parallel. We also blew through our Redis free tier and started paying $470/month for cluster mode. Sharding saved latency but killed our budget.

### Attempt 4: GPU vs CPU for reranking

We moved the reranker to an NVIDIA T4 GPU (g4dn.xlarge) instead of CPU. Latency dropped from 320 ms to 180 ms, but the GPU cost $0.35/hr vs the CPU’s $0.05/hr. At 80 queries/sec, the GPU added $620/month, wiping out the latency savings. We reverted after two days.

Every tweak seemed to trade one disaster for another. We were missing the fundamentals: the RAG pipeline isn’t just a chain of models—it’s a distributed system with its own failure modes.

## The approach that worked

We abandoned the idea that “better embeddings = better retrieval” and focused on two things we’d ignored: chunk granularity and retrieval budget.

**Chunk granularity:** We switched from 1 k–2 k token chunks to 250–300 token chunks with 50 % overlap. Smaller chunks increased the total number of vectors from 1.2 M to 4.8 M, but each vector was cheaper to compute and search. We used sentence transformers all-MiniLM-L6-v2 for embeddings (384 dim) and moved the heavy reranking to a later stage where it only touched the top 20 candidates instead of 50.

**Retrieval budget:** We enforced a hard limit of 20 vectors per query, no matter how large the index grew. If the top-20 recall was below 90 %, we increased the chunk overlap or added metadata filters (region, language, product line). This kept search latency flat and predictable.

We also moved the reranker to a sidecar service that only fired when the top candidate’s score was below a dynamic threshold. That cut reranker calls by 60 % and let us keep the cheaper CPU instance.

Finally, we introduced a two-tier cache: a 10-minute hot cache in Redis 7.2 (size 5 GB) and a 24-hour stale cache in S3 + CloudFront. The hot cache alone cut embedding calls by 45 % and saved $3.1 k/month.

## Implementation details

### Indexing pipeline

We used Qdrant 1.8 with on-disk storage and mmap. Each shard runs on a c6g.large (2 vCPU, 4 GB RAM) with 20 GB gp3 storage. The index is partitioned by product line (cards, loans, investments) to keep hot subsets small.

Indexing script (Python 3.11, Qdrant client 1.8.0):

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import boto3

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
client = QdrantClient("qdrant-prod.internal", port=6333)

chunk_size = 250
overlap = 125

s3 = boto3.client("s3")
for obj in s3.list_objects_v2(Bucket="kb-prod", Prefix="cards/")["Contents"]:
    text = s3.get_object(Bucket="kb-prod", Key=obj["Key"])["Body"].read().decode()
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
    embeddings = model.encode(chunks, batch_size=32)
    client.upsert(
        collection_name="cards",
        points=models.Batch(
            ids=list(range(len(chunks))),
            vectors=embeddings.tolist(),
            payloads=[{"text": c, "source": obj["Key"]} for c in chunks]
        ),
    )
```

### Retrieval and reranking

We wrote a tiny Node 20 LTS service that calls the embedding endpoint, searches Qdrant, then conditionally calls the reranker. The reranker only runs if the top chunk’s score is below `threshold = 0.85 * max_score`.

```javascript
// query-service/src/retriever.ts
import { QdrantClient } from '@qdrant/js-client-rest';
import { CohereClient } from 'cohere-ai';

const qdrant = new QdrantClient({ url: 'http://qdrant-prod.internal' });
const cohere = new CohereClient({ token: process.env.COHERE_KEY });

async function retrieve(query) {
  const embed = await embeddingsModel.embed([query]);
  const search = await qdrant.search(
    collectionName: 'cards',
    query_vector: embed[0].embedding,
    limit: 20,
    with_payload: true,
  );

  const top = search[0];
  if (top.score < 0.85 * search[0].score) {
    const rerank = await cohere.rerank({
      query,
      documents: search.map(p => p.payload.text),
      topN: 5,
    });
    return { chunks: rerank.results.map(r => search[r.index].payload) };
  }
  return { chunks: search.map(p => p.payload) };
}
```

### Caching layer

We use a 5 GB Redis 7.2 cluster (cache.t4g.small) with these eviction policies:
- `maxmemory-policy allkeys-lru`
- `ttl 600` for hot cache
- `ttl 86400` for stale cache stored as JSON blobs in S3

Cache key is `sha256(query + region + language)`. We pre-warm the cache every hour by replaying recent queries from CloudWatch logs. The script runs in a 512 MB Fargate task and costs $19/month.

### Observability stack

We instrument every hop with OpenTelemetry and ship to AWS X-Ray. Key metrics:
- `rag.retrieval_latency` (p50, p95, p99)
- `rag.rerank_calls` (count and %)
- `rag.cache_hit_ratio`
- `rag.embedding_tokens` (to track model spend)

We alert when p95 latency > 1.2 s or cache hit ratio < 0.70.

### Cost guardrails

- Embedding budget: $0.00016 per 1k tokens → we cap daily spend at $35
- Qdrant instance: $0.073/hr per shard → we autoscale from 3 to 5 shards based on CPU > 70 %
- GPU reranker only spins up when the CPU reranker score variance > 0.10

## Results — the numbers before and after

| Metric | Before | After | delta |
|---|---|---|---|
| 95th-percentile latency | 2.8 s | 890 ms | -68 % |
| Cost / 1000 queries | $5.20 | $1.80 | -65 % |
| Monthly AWS bill | $11.6 k | $2.9 k | -75 % |
| Incorrect answers (human audit) | 12 % | 2.1 % | -83 % |
| Embedding calls / query | 1.0 | 0.45 | -55 % |
| Cache hit ratio | N/A | 74 % | — |

Our SLA is now 1.2 s p95; we hit 890 ms. The $2.9 k/month is 25 % of the original budget, leaving headroom for the next product line. The reranker only fires 40 % of the time, so we’re not burning GPU cycles unnecessarily.

## What we’d do differently

1. **Start with production data, not toy datasets.** We built the first index on marketing PDFs—clean, structured, 3 k tokens max. Real support tickets were messy, 500–800 tokens, with typos and slang. We wasted two weeks tuning chunk size on clean data before realizing we needed to reprocess everything.

2. **Measure recall early, not just latency.** Our first internal benchmarks showed 1.1 s latency, so we shipped. When real users asked about “blocked card in SEA region,” the top chunk was a generic “card policy” page 3 hops away. We only caught this after 800 support tickets. Now we do weekly recall tests on 500 live queries; if recall < 90 %, we reprocess the index.

3. **Budget for GPU reranking as a last resort.** We burned $620/month on a GPU reranker that only improved latency by 120 ms. Instead, we should have first tried smaller rerankers, caching, and query rewrites. GPU is a sledgehammer—use it only when you’ve exhausted cheaper levers.

4. **Autoscale Qdrant by shard CPU, not query load.** Our first autoscale rule was based on QPS; we hit 120 QPS and added two shards, but CPU stayed at 30 %. The correct signal is shard CPU > 70 %, which happens at 60–80 QPS per shard. We fixed this after three outages.

5. **Cache per user segment, not just query.** We initially cached only the raw query string. When the same query came from a high-value user segment (premium cards), we served a generic answer. Now we cache by `query + segment` keys, which added 2.3 GB to Redis but cut premium-tier latency to 620 ms.

## The broader lesson

A RAG pipeline is not a sequence of models; it’s a **cost-aware retrieval system** where every stage leaks money and latency.

- **Embeddings are your runtime cost driver.** A 1536-dim embedding costs ~3× a 384-dim one. Use the smallest model that still gives you acceptable recall.
- **Reranking is optional, not mandatory.** Make it conditional on the top candidate’s score. If the vector search is already confident, skip reranking entirely.
- **Cache at the query level, not the model level.** A 5 GB hot cache can cut embedding calls by 45 % and save thousands per month.
- **Chunk size is a retrieval knob, not a storage knob.** Smaller chunks increase vector count but reduce per-query search space. Tune for recall first, latency second.

The tutorials skip the boring bits: chunking strategies, cache key design, and cost caps. They show you a 1 GB demo dataset and a single GPU, then vanish. Production data is 100× messier, and your budget is 10× tighter.

## How to apply this to your situation

1. **Pick the smallest embedding model that still hits your recall target.** Benchmark on 500 real production queries, not the Wikipedia corpus. We saved $3.1 k/month by switching from text-embedding-3-small (1536 dim) to all-MiniLM-L6-v2 (384 dim) without losing recall.

2. **Enforce a hard retrieval budget.** Limit the number of vectors returned from vector search (we use 20). If recall drops, increase chunk overlap or add metadata filters—never increase the budget.

3. **Cache aggressively.** Add a 10-minute Redis cache keyed on `sha256(query + segment)`. Start with 5 GB RAM and monitor hit ratio; increase only if > 70 % of latency comes from embedding.

4. **Instrument every hop.** You can’t optimise what you can’t measure. Add OpenTelemetry traces for embedding, search, rerank, and LLM calls. We use AWS X-Ray with a $15/month CloudWatch dashboard. Without this, we would never have caught the GPU reranker overspend.

5. **Autoscale Qdrant on CPU, not QPS.** Set an alarm for shard CPU > 70 % and scale horizontally. Vertical scaling (bigger instances) is a trap—it hides the real cost of vector search.

Action step for the next 30 minutes: open your current RAG pipeline’s logs and calculate the cache hit ratio. If it’s below 50 %, create a 1 GB Redis cache with a 10-minute TTL and measure the p95 latency drop. If it’s above 70 %, increase the TTL to 30 minutes and watch your embedding bill shrink.

## Resources that helped

- Qdrant 1.8 docs on on-disk storage and mmap tuning: https://qdrant.tech/documentation/guides/optimization/
- Hugging Face sentence-transformers leaderboard for small models: https://huggingface.co/spaces/mteb/leaderboard
- AWS cost calculator for SageMaker hosting: https://calculator.aws.amazon.com/#/
- OpenTelemetry RAG instrumentation example: https://github.com/open-telemetry/opentelemetry-demo/tree/main/src/rag
- Cohere reranker pricing and limits: https://cohere.com/pricing

## Frequently Asked Questions

**How do I choose between Redis and Qdrant for caching?**

Start with Redis if your cache is under 10 GB and you need sub-millisecond reads. Use Qdrant if your cache is larger or you plan to scale to multi-region retrieval. We moved from Redis to Qdrant when our cache grew to 8 GB and needed persistence across AZs. The switch added 2 ms per read but saved $1.2 k/month in EBS costs.

**What’s the minimum recall threshold I should aim for?**

Aim for 90 % recall on 500 real production queries. Below 85 % you’ll see support tickets spike. We target 95 % for premium users and 88 % for free-tier users. The difference is handled by adding more chunks or better metadata filters.

**Is it worth using a GPU reranker for < 200 ms latency savings?**

No. Our GPU reranker cut latency by 120 ms but cost $620/month. We got the same delta by caching 45 % of queries and using a smaller reranker model. GPU is only worth it if you’re already at 150 ms latency and need to shave the last 50 ms.

**How do I handle multilingual queries without exploding costs?**

Use a lightweight language detector (fasttext 1.0) and route to language-specific shards. We added 3 shards (EN, ID, VI) and saw embedding cost rise only 12 % because most queries were already monolingual. Avoid running multilingual embeddings on every query—it doubles your bill.

**What’s the biggest surprise teams miss when moving from demo to prod?**

Dirty data. Our marketing PDFs had footers like “Confidential — do not redistribute,” which poisoned the index. We spent two weeks tuning chunk size before realizing the recall problem was garbage in the knowledge base. Clean your data first, optimize later.

**How do I set the reranker threshold without manual tuning?**

Start with `threshold = 0.85 * max_score` and adjust weekly based on recall. We built a script that pulls 500 recent queries, computes recall at different thresholds, and suggests the highest threshold that keeps recall > 90 %. The script runs in a 512 MB Lambda and costs $8/month.


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
