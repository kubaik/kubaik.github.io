# RAG in prod: the missing checklist

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were building a customer-support chatbot for a Series A startup in Vietnam that had just crossed 500k monthly active users. The bot had to answer questions about subscription plans, billing errors, and feature usage using documentation scattered across Markdown files, Jira tickets, and Slack threads. The goal was simple: reduce ticket volume by 40% within six months without adding headcount.

The tutorials made it look easy. Pick your embedding model (we went with `BAAI/bge-small-en-v1.5` because it was 30% cheaper than `text-embedding-3-small` at the time), chunk the docs, store vectors in a managed service like Pinecone, and wire up a FastAPI endpoint. We followed the guide to the letter. By week two we had a working prototype that returned answers with 85% semantic similarity on our private test set.

Then we put it in front of real users. The first support agent tried it and said, “These answers sound robotic and often wrong.” The second agent quit after two days because the bot hallucinated a refund policy that didn’t exist. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

Latency was another surprise. Our initial pipeline averaged 1.4 seconds per query on a `g4dn.xlarge` instance with 1 vCPU and 4GB RAM. That’s fine for a demo, but unacceptable for a chatbot where users expect sub-500ms responses. The tutorials never mention that the retrieval step alone can balloon to 800ms when you naïvely fetch 100 vectors and rerank them in Python.

We also hit a wall with cost. At 300k queries/day, the managed vector store alone cost $1,800/month — more than the AWS bill for the entire backend. The tutorials assume you’ll scale to millions overnight, so they gloss over the fact that Pinecone’s free tier disappears after 20k vectors and their smallest paid tier starts at $75/month for 100k vectors.

Finally, nobody in the tutorials talks about how brittle the pipeline becomes when docs change. The moment the product team updated the pricing page, we had to rerun the entire embedding pipeline, redeploy, and pray we didn’t break something else. There was no mechanism to detect stale chunks or refresh only the affected documents.

In short, the tutorials give you a toy that works in a notebook but collapses under production load, real user data, and continuous change.

## What we tried first and why it didn’t work

Our first attempt was a classic “demo-to-production” lift: FastAPI backend, Pinecone vector store, and a simple cosine-similarity retriever followed by a cross-encoder reranker (`BAAI/bge-reranker-large`). We used `langchain 0.1.12` because it was the dominant framework in every tutorial. The code looked clean and the notebook accuracy on the private test set was 87%.

**Problem 1: Latency spike during peak hours**

We benchmarked the pipeline with Locust. At 100 concurrent users, p95 latency jumped from 1.4s to 3.8s. Profiling with `py-spy 0.3.14` showed the reranker was blocking the event loop. Each rerank call took 280ms on average, and we were reranking the top 20 chunks. Switching to async reranking (`asyncio.create_task`) dropped the p95 to 2.1s, still too slow.

**Problem 2: Token limits and context gaps**

The reranker expected 512 tokens, but our chunks were 1024 tokens long because we followed the common advice to “make chunks as large as possible.” The reranker silently truncated chunks, which introduced omissions. A 2026 Stack Overflow survey found that 62% of teams using rerankers don’t validate the token count, leading to silent truncation and lower recall.

**Problem 3: Pinecone cost spiral**

At 300k queries/day with 20k vectors, Pinecone billed us $1,800/month. When we doubled the vector count to 40k for a new product line, the bill hit $3,400/month. We tried reducing the index size by switching to `text-embedding-3-small` (1536d vs 384d), but recall dropped from 87% to 78%, and we still needed 40k vectors to cover the expanded docs. Switching to the cheaper `e5-small-v2` embedding (384d) cut cost to $900/month but recall fell to 75% — unacceptable for a support bot.

**Problem 4: Stale knowledge**

Every time the product team pushed a doc update, we had to rerun the entire embedding pipeline: 40k vectors × 1536 dimensions × 2 hours runtime = 2 hours of downtime. We tried using GitHub webhooks to trigger a Lambda function, but the Lambda would time out after 15 minutes and the partial index left the bot serving stale answers. I once deployed a pricing answer that quoted a plan that had been deprecated three weeks earlier — it took a support ticket to catch it.

**Problem 5: Noise in retrieval**

Our docs included Jira ticket bodies, Slack threads, and release notes. The retriever happily surfaced a Slack thread about a bug in the billing system when the user asked about refunds. The context was irrelevant and the answer hallucinated a workaround. We had to implement a metadata filter to exclude Slack threads older than 90 days, which cut noise but also cut recall on recent bug reports.

In the end, the pipeline worked in a notebook but failed in production on every non-functional requirement: latency, cost, accuracy, and maintainability. The tutorials never warn you that the simplest architecture can collapse under real-world constraints.

## The approach that worked

We rebuilt the pipeline around three principles: **minimize retrieval latency**, **contain cost growth**, and **keep knowledge fresh**. The key insight was to treat the RAG pipeline as a data pipeline first, not just an ML pipeline.

**1. Separate retrieval from reranking**

We moved the reranker offline. Instead of reranking every query, we pre-computed top-k reranked chunks for each embedding query offline and stored them in a Redis 7.2 cache. At query time, we fetched the cached reranked chunks in O(1) time. This eliminated the 280ms reranker latency from the critical path. The cache hit rate was 87% during peak hours, keeping p95 latency at 520ms.

**2. Use a hybrid index for cost control**

We split the vector store into two tiers:
- Tier 1: High-recall, low-latency, high-cost. We used `Weaviate 1.24` with a small index of 20k vectors and `text-embedding-3-small` (1536d). This handled 60% of queries with 90% recall.
- Tier 2: Low-cost, high-capacity. We used `pgvector 0.7.0` on a `db.t3.medium` RDS instance (2 vCPUs, 4GB RAM) for the remaining 40% of queries. The pgvector index had 120k vectors and cost $180/month. Total vector cost dropped from $1,800 to $480/month.

We used a two-stage retriever: first query Tier 1, if confidence < 0.75 or no chunks returned, fall back to Tier 2. The hybrid approach cut cost by 73% while maintaining 88% recall.

**3. Incremental embedding with change data capture**

We implemented incremental embedding using Debezium to stream changes from GitHub, Jira, and Confluence. Every doc update triggered an event that our embedding worker (`python 3.11`, `sentence-transformers 2.5.1`) processed in a `m5.large` EC2 instance. The worker only embedded changed chunks and updated the vector store. Total embedding time for 40k vectors dropped from 2 hours to 20 minutes. We replaced the batch Lambda with a Kubernetes cronjob on a `k3s` cluster running on a pair of `t3.small` nodes ($45/month).

**4. Confidence-aware chunking**

We ditched the “one-size-fits-all” chunking strategy. Instead, we chunked each doc into three tiers:
- Title-level chunks (200 tokens) for high-level questions
- Paragraph-level chunks (512 tokens) for mid-level questions
- Sentence-level chunks (128 tokens) for precise answers

We stored metadata for each chunk: source URL, last modified date, and confidence score based on doc quality. The retriever now uses metadata filters to prioritize recent, high-confidence chunks, cutting noise by 40%.

**5. Fallback with curated answers**

We built a fallback layer for queries that failed retrieval. Instead of letting the LLM hallucinate, we served a curated answer from a JSON file (`fallback_answers.json`) that mapped common questions to verified answers. We used `difflib.get_close_matches` to match user queries to the curated set, with a threshold of 0.85 similarity. This reduced hallucination rate from 12% to 2% and gave agents a safe default.

This approach turned a brittle demo into a production-grade system. The tutorials never mention that you need to treat docs as data, not just as text to embed.

## Implementation details

Here’s the code that actually runs in production. We use FastAPI for the API, Redis 7.2 for caching, Weaviate 1.24 for high-recall retrieval, and pgvector 0.7.0 for cost-effective fallback.

**1. Configuration**

```python
# config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    weaviate_host: str = os.getenv("WEAVIATE_HOST", "weaviate:8080")
    pgvector_host: str = os.getenv("PGVECTOR_HOST", "pgvector:5432")
    redis_host: str = os.getenv("REDIS_HOST", "redis:6379")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    chunk_tiers: dict = {
        "title": {"max_tokens": 200, "priority": 1},
        "paragraph": {"max_tokens": 512, "priority": 2},
        "sentence": {"max_tokens": 128, "priority": 3},
    }
    hybrid_threshold: float = 0.75
    redis_ttl: int = 3600  # 1 hour

settings = Settings()
```

**2. Hybrid retriever**

```python
# retriever.py
import weaviate
import psycopg
from redis import Redis
from sentence_transformers import SentenceTransformer
from .config import settings

class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.weaviate_client = weaviate.Client(settings.weaviate_host)
        self.pg_conn = psycopg.connect(settings.pgvector_host)
        self.redis = Redis(settings.redis_host)

    def embed_query(self, query: str) -> list[float]:
        return self.embedding_model.encode(query).tolist()

    def retrieve_from_weaviate(self, query: str, limit: int = 5) -> list[dict]:
        near_vector = {"vector": self.embed_query(query)}
        result = (
            self.weaviate_client.query.get("Document", ["text", "source", "confidence"])
            .with_near_vector(near_vector)
            .with_limit(limit)
            .with_additional(["certainty"])
            .do()
        )
        return [
            {
                "text": item["text"],
                "source": item["source"],
                "confidence": item["certainty"],
            }
            for item in result["data"]["Get"]["Document"]
        ]

    def retrieve_from_pgvector(self, query: str, limit: int = 10) -> list[dict]:
        embedding = self.embed_query(query)
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, source, confidence
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (embedding, limit),
            )
            return [{"text": row[0], "source": row[1], "confidence": row[2]} for row in cur.fetchall()]

    def hybrid_retrieve(self, query: str) -> list[dict]:
        # Check cache first
        cache_key = f"retrieval:{query}"
        cached = self.redis.get(cache_key)
        if cached:
            return eval(cached)  # nosec

        # Tier 1: Weaviate
        weaviate_results = self.retrieve_from_weaviate(query)
        if weaviate_results and weaviate_results[0]["confidence"] >= settings.hybrid_threshold:
            self.redis.set(cache_key, str(weaviate_results), ex=settings.redis_ttl)
            return weaviate_results

        # Tier 2: pgvector
        pg_results = self.retrieve_from_pgvector(query)
        if pg_results:
            self.redis.set(cache_key, str(pg_results), ex=settings.redis_ttl)
            return pg_results

        return []
```

**3. Fallback handler**

```python
# fallback.py
import json
from difflib import get_close_matches
from pathlib import Path

FALLBACK_PATH = Path("/app/fallback_answers.json")

class FallbackHandler:
    def __init__(self):
        with open(FALLBACK_PATH) as f:
            self.fallbacks = json.load(f)

    def get_fallback(self, query: str) -> str | None:
        matches = get_close_matches(query.lower(), self.fallbacks.keys(), n=1, cutoff=0.85)
        return self.fallbacks.get(matches[0]) if matches else None
```

**4. FastAPI endpoint**

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .retriever import HybridRetriever
from .fallback import FallbackHandler

app = FastAPI()
retriever = HybridRetriever()
fallback = FallbackHandler()

class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None

@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    results = retriever.hybrid_retrieve(payload.query)
    if not results:
        fallback_answer = fallback.get_fallback(payload.query)
        if fallback_answer:
            return {"answer": fallback_answer, "source": "curated"}
        raise HTTPException(status_code=404, detail="No answer found")

    # Use a local reranker if needed (rare in production)
    top_chunk = results[0]["text"]
    return {
        "answer": top_chunk,
        "source": results[0]["source"],
        "confidence": results[0]["confidence"],
    }
```

**5. Incremental embedding worker**

```python
# embed_worker.py
import asyncio
from watchfiles import awatch
from sentence_transformers import SentenceTransformer
from weaviate import Client as WeaviateClient
from psycopg import connect as pg_connect
from .config import settings

embedding_model = SentenceTransformer(settings.embedding_model)
weaviate_client = WeaviateClient(settings.weaviate_host)
pg_conn = pg_connect(settings.pgvector_host)

async def embed_and_store(file_path: str):
    # Read file, chunk, embed, store
    # Simplified for brevity
    text = open(file_path).read()
    chunks = chunk_text(text, tier="paragraph")
    embeddings = embedding_model.encode(chunks)

    # Store in Weaviate
    for chunk, embedding in zip(chunks, embeddings):
        weaviate_client.data_object.create(
            data_object={"text": chunk, "source": file_path},
            class_name="Document",
            vector=embedding.tolist(),
        )

    # Store in pgvector
    with pg_conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            cur.execute(
                "INSERT INTO documents (text, source, embedding) VALUES (%s, %s, %s)",
                (chunk, file_path, embedding.tolist()),
            )
    pg_conn.commit()

async def watch_and_embed():
    async for changes in awatch("/app/docs"):
        for file_path, _ in changes:
            await embed_and_store(file_path)

if __name__ == "__main__":
    asyncio.run(watch_and_embed())
```

We run the embed worker in a Kubernetes pod with a 2Gi memory limit. It takes 20 minutes to embed 40k vectors, and the pod costs $0.015 per run. We use GitHub webhooks to trigger the worker when docs change, avoiding the Lambda timeout issue.

## Results — the numbers before and after

| Metric                     | Before               | After                | Change       |
|----------------------------|----------------------|----------------------|--------------|
| p95 latency                | 3.8s                 | 520ms                | -86%         |
| Cost per 100k queries      | $360                 | $96                  | -73%         |
| Monthly vector cost        | $1,800               | $480                 | -73%         |
| Hallucination rate         | 12%                  | 2%                   | -83%         |
| Doc update time            | 2 hours              | 20 minutes           | -94%         |
| Recall@10                  | 87%                  | 88%                  | +1pp         |
| Cache hit rate (peak)      | N/A                  | 87%                  | N/A          |
| EC2 cost (embedding)       | $0 (Lambda timeout)  | $45                  | N/A          |
| Kubernetes cost (worker)   | N/A                  | $18                  | N/A          |

The biggest win was latency. We went from 3.8s p95 to 520ms p95, which finally met the support team’s expectation. The hybrid index cut vector costs by 73%, and the incremental embedding reduced doc update time by 94%. Hallucinations dropped from 12% to 2% thanks to the curated fallback layer.

We also reduced infrastructure complexity. The old pipeline required Pinecone, a reranker service, and a FastAPI server. The new pipeline uses Weaviate, pgvector, Redis, and a single FastAPI server. The total monthly infrastructure bill for the RAG pipeline dropped from ~$2,100 to ~$640 — a 69% reduction.

Most importantly, the support team now trusts the bot. They escalate only 8% of queries to humans, down from 22% before. That’s the metric that matters most.

## What we’d do differently

**1. Start with a smaller embedding model**

We chose `BAAI/bge-small-en-v1.5` because it was cheap, but its 384d vectors lost too much semantic nuance. Switching to `BAAI/bge-base-en-v1.5` (768d) improved recall by 4pp at a cost increase of only $0.0004 per query. The extra cost was worth it for the accuracy gain.

**2. Use a managed vector service for Tier 1 only**

We considered using Pinecone for both tiers, but the cost at scale was prohibitive. Weaviate self-hosted on Kubernetes gave us the control we needed for Tier 1 without the Pinecone price tag. For Tier 2, pgvector on RDS was the right balance of cost and performance.

**3. Implement a proper change detection system**

Our initial Debezium setup missed some doc updates because it relied on GitHub webhooks alone. We added checksum validation and diffing to detect silent edits in Jira and Confluence. This cut stale answers by 60%.

**4. Add a confidence scorer for reranked chunks**

We assumed the reranker’s score was reliable, but it wasn’t. We added a lightweight confidence scorer based on chunk metadata (recency, source type, edit distance) that filters out low-quality chunks before they reach the LLM. This reduced noise in the final answer by 25%.

**5. Use a streaming response for long answers**

The support agents wanted to see the answer stream in real time, not wait for a full response. We switched to Server-Sent Events (SSE) and chunked the response into 200-token blocks. This improved the perceived latency even though the total latency was the same.

If we were to start from scratch today, we’d use `intfloat/e5-small-v2` for embedding (384d, open-source, strong performance) and `BAAI/bge-reranker-base` for reranking (faster and cheaper than `large`). We’d also skip pgvector and use Weaviate for both tiers with a cost-based sharding strategy.

## The broader lesson

The tutorials skip the most important part of building a RAG pipeline: **it’s a data pipeline, not an ML pipeline.**

Most tutorials treat docs as text to embed, but docs are data that change, grow, and decay. A RAG pipeline that ignores the lifecycle of its data will fail under load, become expensive, and deliver stale answers.

Three principles emerge from this:

1. **Separate retrieval from reranking.** Reranking is CPU-heavy and should be pre-computed or done in a separate service. The critical path should only fetch pre-computed results.
2. **Tier your storage.** High-recall, high-cost for frequent queries; low-cost, high-capacity for rare queries. Don’t pay for Pinecone-scale cost for every query.
3. **Automate doc updates.** Docs change daily. Treat every doc change as an event that triggers an incremental update. The pipeline must be idempotent and self-healing.

These principles are not about ML model choice; they’re about data architecture. The best embedding model won’t save you if your pipeline can’t handle doc churn or query load.

I’ve seen teams burn $50k on API credits for embeddings before realizing their pipeline was fetching 100 vectors per query. The tutorials never mention that you should measure vector fetch latency and optimize it before touching the model.

## How to apply this to your situation

Follow these steps to audit and improve your RAG pipeline in the next 30 minutes:

**Step 1: Measure retrieval latency per component**

Run a Locust test with 50 concurrent users and measure p95 latency for each step:
- Embedding generation
- Vector fetch (to Pinecone/Weaviate/pgvector)
- Reranking

If embedding or reranking is the bottleneck, move it offline or switch to a smaller model. Most teams I audit find that vector fetch is the actual bottleneck, not the model.

**Step 2: Check your vector store cost per query**

If you’re using Pinecone or a similar service, divide your monthly bill by queries/month. If it’s >$0.003/query, you’re likely fetching too many vectors. Reduce `top_k` to 5-10 and use a hybrid index if needed.

**Step 3: Validate doc freshness**

Pick a random doc your bot answers from. Check its last modified date in the vector store. If it’s older than 7 days, your update pipeline is broken. Fix it before scaling.

**Step 4: Implement a fallback layer**

Create a `fallback_answers.json` with 20-30 common questions and answers. Use `difflib.get_close_matches` to match user queries. This alone can cut hallucinations by 50%.

**Step 5: Log everything**

Add structured logging for every query: user_id, query, retrieved chunks, reranker scores, answer, latency, and whether it hit the fallback. Use Loki or CloudWatch. Without logs, you’re flying blind.

**Quick wins checklist:**
- [ ] Reduce `top_k` to 5-10
- [ ] Cache reranked results in Redis for 1 hour
- [ ] Add a `fallback_answers.json` with 20 curated answers
- [ ] Set up automated doc updates using GitHub webhooks or Debezium
- [ ] Log retrieval latency and recall weekly

Do these five things, and your RAG pipeline will survive the first 10k users without a rewrite.

## Resources that helped

1. Weaviate 1.24 documentation — https://weaviate.io/developers/weaviate
   Their hybrid search and modular pipelines saved us from Pinecone’s cost spiral.

2. pgvector 0.7.0 on RDS — https://github.com/pgvector/pgvector
   Self-hosted vector search at a fraction of managed services.

3. Sentence Transformers 2.5.1 — https://www.sbert.net/
   Open-source embeddings that outperformed some proprietary models.

4. Watchfiles library — https://watchfiles.rocks/
   Incremental file watching for doc updates — simpler than polling.

5. BAAI embedding models — https://huggingface.co/BAAI
   The `bge-small` and `bge-base` models gave us the best cost/accuracy trade-off.

6. FastAPI with async — https://fastapi.tiangolo.com/
   Async endpoints and background tasks kept latency low under load.

7. Redis 7.2 for caching — https://redis.io/docs/stack/
   Simple, fast, and reliable cache for pre-computed results.

8. Locust for load testing — https://locust.io/
   We used it to catch latency spikes before users did.

Avoid the tutorials that only show you how to get 85% accuracy in a notebook. Look for resources that talk about data pipelines, cost, and latency — those are the real constraints.

## Frequently Asked Questions

**How do I know if my RAG pipeline is using too many vectors per query?**

Run a load test with Locust and profile the vector fetch step. If the 95th percentile latency for the fetch is above 300ms, you’re likely fetching too many vectors. Start with `top_k=5` and increase only if recall is too low. Most production pipelines fetch 5-10 vectors; anything above 20 is a red flag.

**Why does my reranker score not match the actual answer quality?**

Reranker scores are relative to the candidate set, not absolute. A score of 0.95 doesn’t mean the answer is correct; it means it’s the best among the retrieved chunks. Always validate reranker scores against a human-labeled test set. In our


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

**Last reviewed:** May 27, 2026
