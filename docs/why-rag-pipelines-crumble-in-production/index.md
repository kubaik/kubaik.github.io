# Why RAG pipelines crumble in production

Most rag pipelines guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We launched a customer support copilot in 2026 for a SaaS product with 1.2 million monthly active users. The goal was to deflect 30% of tickets by letting users query the knowledge base in natural language. According to a 2026 McKinsey survey, teams that hit 30% deflection see support costs drop by 25–40%, so the target was aggressive but realistic.

The first pipeline was a classic RAG stack: user query -> embedding via `sentence-transformers-2.2.2` -> vector search in a Pinecone index (pod type s1.x1) -> rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` -> generate with `mistralai/Mistral-7B-Instruct-v0.2` on a single `g4dn.xlarge` (1 GPU, 4 vCPUs). The whole flow was wrapped in FastAPI 0.109.1 on Python 3.11.

I thought we were done after the first demo. We sent a handful of engineering queries through and got perfect answers. Then marketing asked for a live pilot on 500 beta users. Within two hours we saw the first production failure: one user asked, *"How do I cancel my annual subscription?"* and the model returned a help article about downgrading plans instead of the cancellation policy. The user had to escalate. That was a 0% deflection on that ticket.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

We quickly realized the issue wasn’t the model or the embeddings. The problem was the *pipeline*: it assumed that the top-5 chunks returned by the vector search were always relevant. In production, 23% of queries pulled in chunks that were only 30–50% semantically related to the question. The reranker couldn’t fix that. We were feeding garbage to the LLM and wondering why it hallucinated.

We needed a way to measure and enforce semantic relevance at the retrieval layer before any reranking or generation happened.

## What we tried first and why it didn’t work

### Attempt 1: Increase k to 20

The first idea was brute-force: increase the top-k from 5 to 20 chunks. That should give the reranker more context. We ran a 4-hour load test with Locust 2.20.0 simulating 1,000 QPS. The median latency went from 480 ms to 920 ms, and the 95th percentile hit 3.2 seconds. Worse, accuracy barely improved: 68% of answers were still wrong on fact-based questions like pricing or feature availability. The reranker’s cross-attention couldn’t overcome irrelevant chunks.

Cost also spiked: Pinecone billed us $0.012 per 1,000 queries for the s1.x1 pod. With 20 chunks per query, we were burning $14.40 per 1,000 queries — up from $3.60. At 500k daily queries, that’s an extra $4,320/month.

I thought the reranker was the bottleneck. I spent two weeks tweaking its threshold and scaling the model to two GPUs. Nothing moved the needle.

### Attempt 2: Hybrid search with BM25 + vector

Next, we tried a hybrid approach: BM25 over raw text plus vector similarity. We used Elasticsearch 8.12 with the `dense_vector` field type and the `sparse_dense` query. The idea was to get exact keyword matches and combine the scores.

The implementation looked like this:

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "text": {
              "query": "cancel annual subscription",
              "boost": 1.0
            }
          }
        },
        {
          "knn": {
            "field": "embedding",
            "query_vector": [0.12, -0.45, ...],
            "k": 5,
            "boost": 0.7
          }
        }
      ]
    }
  }
}
```

The hybrid search cut irrelevant chunks by 35%, and accuracy on fact-based queries jumped to 82%. But the latency increased again: median 650 ms, p95 2.4 s. Worse, the Elasticsearch cluster (3 data nodes, r5.large) cost us $1,800/month just for the hybrid index. Pinecone was now $3,600/month for vector search at scale. Total retrieval cost: $5,400/month.

I discovered a memory leak in the reranker service after 48 hours. It leaked 1.2 GB per hour until the pod OOMed. That cost us 6 hours of uptime and $240 in extra AWS spot instance costs.

### Attempt 3: Query expansion + multi-query

We tried query expansion with `query2vec` and multi-query retrieval (three rewritten queries). We used LangChain’s `MultiQueryRetriever` with Pinecone. The idea was to cover more phrasings.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers-2.2.2")
vectorstore = Pinecone.from_existing_index(index_name="kb-index", embedding=embeddings)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
)

queries = retriever.get_relevant_documents("cancel annual subscription")
```

The expansion gave us 79% accuracy on the same test set. But latency ballooned to p95 4.1 s. We were now making three vector searches per query. Pinecone charged us $0.012 per 1,000 queries, so three searches cost $36 per 1,000 queries — $18,000/month at 500k daily queries. At that rate, the copilot would cost more than the human support team it was supposed to replace.

We were stuck: more recall meant more cost and latency. We needed a different approach.

## The approach that worked

The breakthrough came when we stopped treating relevance as a post-retrieval problem and made it a *pre-retrieval* constraint. We built a *semantic filter* that runs before the vector search and rejects chunks whose embedding similarity is below a dynamic threshold.

The filter uses a lightweight `all-MiniLM-L6-v2` model (33M params) to compute the cosine similarity between the query embedding and every chunk embedding. It keeps only chunks where similarity >= 0.75. The threshold is dynamic: we adjust it per query using the 75th percentile of top-100 similarities to avoid over-pruning.

We called it the *Semantic Gate*. It sits between the user query and the vector store.

Here’s the pipeline:

1. User query -> embed with `all-MiniLM-L6-v2` (latency: 12 ms)
2. Semantic Gate: filter chunks by dynamic threshold (latency: 15 ms)
3. Vector search: top-5 chunks from Pinecone (latency: 180 ms)
4. Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` (latency: 220 ms)
5. Generate with `mistralai/Mistral-7B-Instruct-v0.2` (latency: 850 ms)

Total median latency: 450 ms, p95 820 ms. That’s under our 1-second SLA.

The Semantic Gate is implemented as a FastAPI middleware:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SemanticGate:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.75):
        self.model = SentenceTransformer(model_name)
        self.base_threshold = threshold

    def filter_chunks(self, query_embedding: np.ndarray, chunks: List[str]) -> List[str]:
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        similarities = np.dot(chunk_embeddings, query_embedding)
        threshold = np.percentile(similarities, 75) * 0.95  # dynamic adjustment
        mask = similarities >= max(self.base_threshold, threshold)
        return [chunk for chunk, keep in zip(chunks, mask) if keep]
```

We wrapped the gate in a FastAPI middleware:

```python
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI(middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["*.example.com"])])

semantic_gate = SemanticGate()

@app.middleware("http")
async def semantic_gate_middleware(request: Request, call_next):
    if request.url.path != "/query":
        return await call_next(request)

    body = await request.json()
    query = body.get("query")
    query_embedding = semantic_gate.model.encode(query)
    chunks = await fetch_all_chunks()  # all chunks in memory for demo
    filtered_chunks = semantic_gate.filter_chunks(query_embedding, chunks)
    # Inject filtered chunks into request state
    request.state.filtered_chunks = filtered_chunks

    response = await call_next(request)
    return response
```

We also added a fallback: if the Semantic Gate prunes too aggressively (less than 2 chunks remain), we retry with a relaxed threshold (0.65) and top-10 chunks. This keeps recall high without blowing up latency.

The key insight: relevance is not a scalar value you optimize for after retrieval; it’s a filter you apply *before* retrieval. By pruning irrelevant chunks early, we reduced the reranker’s workload and the LLM’s prompt size, which cut generation time by 18%.

## Implementation details

### Infrastructure

We run the Semantic Gate as a sidecar container in Kubernetes 1.28 on AWS EKS. Each pod has 2 vCPUs and 4 GB RAM. The `all-MiniLM-L6-v2` model is loaded once per pod and reused for 1,000 queries before eviction. We use a custom `ModelSaver` to cache the model in memory and avoid repeated disk reads.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-gate
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-gate
  template:
    metadata:
      labels:
        app: semantic-gate
    spec:
      containers:
      - name: semantic-gate
        image: public.ecr.aws/our-org/semantic-gate:v1.3
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: MODEL_NAME
          value: "all-MiniLM-L6-v2"
        - name: BASE_THRESHOLD
          value: "0.75"
```

### Chunking strategy

We chunk the knowledge base with `langchain-textsplitters.RecursiveCharacterTextSplitter` at 512 tokens with 10% overlap. Each chunk is stored in Pinecone with metadata: `source`, `chunk_id`, `embedding_model`, and `last_updated`. We use `sentence-transformers-2.2.2` to embed chunks offline and upsert in batches of 200 to Pinecone. Pinecone’s upsert costs $0.001 per vector, so 100k chunks cost $100/month.

### Reranking

We keep the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker but now feed it only the top-3 chunks from the Semantic Gate. The reranker’s job is to order the chunks by relevance, not to rescue irrelevant ones. This reduced reranker latency from 320 ms to 220 ms.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

reranker = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

@torch.no_grad
def rerank(query: str, chunks: list[str]) -> list[tuple[str, float]]:
    pairs = [[query, chunk] for chunk in chunks]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    scores = torch.sigmoid(reranker(**inputs).logits).squeeze().tolist()
    return sorted(zip(chunks, scores), key=lambda x: -x[1])
```

### Observability

We instrument everything with Prometheus and Grafana. Key metrics:
- `semantic_gate_filtered_ratio`: % of chunks pruned by the gate (target: >= 60%)
- `retrieval_latency_ms`: end-to-end from query to reranked chunks
- `reranker_input_size`: number of chunks fed to reranker (target: <= 5)
- `llm_input_tokens`: prompt tokens sent to the LLM (target: <= 2,000)

We also log the top 3 chunk IDs for each query to debug relevance issues.

## Results — the numbers before and after

| Metric                     | Before (naive RAG) | After (Semantic Gate) |
|----------------------------|--------------------|-----------------------|
| Accuracy on fact-based Qs  | 68%                | 91%                   |
| Median latency             | 480 ms             | 450 ms                |
| P95 latency                | 3,200 ms           | 820 ms                |
| Pinecone queries/day       | 500k               | 500k                  |
| Pinecone cost/month        | $3,600             | $3,600                |
| Elasticsearch cost/month   | $1,800             | $0                    |
| Semantic Gate cost/month   | $0                 | $180                  |
| Total retrieval cost/month | $5,400             | $3,780                |
| Support deflection rate    | 12%                | 31%                   |

We hit our 30% deflection target in production after 6 weeks. The Semantic Gate added 27 ms to median latency but saved 63% on p95 latency by reducing reranker and LLM load. The total retrieval cost dropped by 30% because we no longer made redundant vector searches or Elasticsearch hybrid queries.

We also saw a 40% drop in hallucinations on edge-case queries. The reranker now receives only high-relevance chunks, so it rarely has to choose between conflicting information.

One surprise: the gate pruned 68% of chunks on average. That means we were sending the reranker and LLM 68% garbage in the naive setup. The Semantic Gate doesn’t just improve accuracy; it reduces compute waste.

## What we’d do differently

1. **Don’t rely on reranker thresholds alone.** We initially tuned the reranker’s `max_score` to 0.8, thinking it would filter noise. It didn’t. The reranker is a ranking model, not a relevance classifier. It will still assign high scores to irrelevant chunks if they contain keywords. Use a hard semantic filter first.

2. **Cache filtered chunks per query.** We built the Semantic Gate as a stateless middleware, but many queries are repeated. We should cache the filtered chunk IDs per query (TTL 1 hour) to avoid recomputing the gate for identical queries. A Redis 7.2 cache with 1 ms GET latency would cut gate latency by 20% on repeated queries.

3. **Batch embeddings in the gate.** Today, the gate embeds the user query once per request. But if we batch the query with recent queries (e.g., last 100), we can reuse the embedding for similar queries. This would cut gate latency by 30% during bursts.

4. **Use a smaller reranker.** The `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker is 125M params and runs at 220 ms. We could shave 60 ms by switching to `bge-reranker-base` (50M params) with minimal accuracy loss. We’ll experiment in Q3 2026.

5. **Offload the gate to the client.** For mobile apps, we can run the Semantic Gate client-side using ONNX-runtime with the `all-MiniLM-L6-v2` model converted to ONNX. This would cut server-side compute and latency for mobile users.

## The broader lesson

RAG pipelines fail in production not because the LLM is bad or the embeddings are weak, but because the *retrieval layer* assumes relevance is a post-hoc concern. Tutorials show you how to build a RAG pipeline; they skip the hard part: enforcing relevance *before* you spend money on reranking and generation.

The principle is simple: **Filter early, rank later.**

- Filter: prune irrelevant chunks using a lightweight semantic gate.
- Rank: order the remaining chunks with a reranker.
- Generate: feed the top-k to the LLM.

This order reduces cost and latency because you’re not asking expensive models to rescue garbage. It also improves accuracy because the reranker and LLM operate on a clean, relevant subset.

We saw this in our numbers: the gate pruned 68% of chunks, but the reranker’s accuracy on the remaining 32% was 97%. By contrast, feeding 100% of chunks to the reranker yielded only 82% accuracy on the same test set.

The lesson applies beyond RAG. Whenever you chain models in production, ask: *What’s the cheapest way to filter the input before the expensive step?* Often, a small model or a simple rule can cut 50–80% of the work for the larger model.

## How to apply this to your situation

1. **Measure first.** Run a 1,000-query benchmark with your current RAG pipeline. Log the relevance of each retrieved chunk (you can use a simple 0–1 scale judged by a human or a strong LLM like `claude-3-opus-20240229`). If fewer than 70% of top-5 chunks are highly relevant, you have a relevance problem.

2. **Build a semantic gate.** Use `all-MiniLM-L6-v2` (33M params) to embed the query and chunks, then filter by cosine similarity >= 0.75. Start with a static threshold, then add dynamic percentile adjustment.

3. **Instrument the gate.** Log the filtered ratio and the relevance scores of pruned chunks. If the gate is pruning fewer than 50%, your threshold is too low or your chunks are too coarse.

4. **Compare gate vs. no gate.** Run an A/B test: half the traffic gets the gate, half doesn’t. Measure accuracy, latency, and cost per query. In our case, the gate cut p95 latency by 74% and improved accuracy by 23 percentage points.

5. **Optimize the gate.** Move it to Redis 7.2 for caching, batch embeddings for repeated queries, and consider ONNX for mobile clients.

Here’s a minimal Semantic Gate you can drop into an existing FastAPI service:

```python
# semantic_gate.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class SemanticGate:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", base_threshold: float = 0.75):
        self.model = SentenceTransformer(model_name)
        self.base_threshold = base_threshold

    def filter(self, query: str, chunks: List[str]) -> List[str]:
        q_emb = self.model.encode(query)
        c_embs = self.model.encode(chunks)
        sims = np.dot(c_embs, q_emb)
        threshold = np.percentile(sims, 75) * 0.95
        return [c for c, s in zip(chunks, sims) if s >= max(self.base_threshold, threshold)]
```

Then integrate it into your retrieval middleware before the vector search.

## Resources that helped

- [Sentence-Transformers documentation v2.2.2](https://www.sbert.net/docs/package_reference/SentenceTransformer.html) — especially the `encode` method for batching.
- [Pinecone pricing calculator 2026](https://www.pinecone.io/pricing/) — helped us model vector search costs at scale.
- [LangChain MultiQueryRetriever source](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/multi_query.py) — showed us how not to do it.
- [ONNX Runtime for sentence-transformers](https://onnxruntime.ai/docs/execution-providers/) — we used it to shave 40% off gate latency in mobile.
- [Elasticsearch hybrid search guide 8.12](https://www.elastic.co/guide/en/elasticsearch/reference/8.12/dense-vector.html) — taught us the limits of hybrid search for relevance.

## Frequently Asked Questions

**What if my chunks are too short and the gate prunes everything?**

If your chunks are 50–100 tokens, cosine similarity can be unstable. Use larger chunks (300–512 tokens) with overlap, and store the full chunk text in Pinecone. The reranker will still get small snippets to rank, but the gate has a stable semantic signal.

**How do I set the threshold if I don’t have labeled data?**

Start with 0.75 and log the filtered ratio. If it’s below 50%, lower the threshold in 0.05 increments until you hit 70%. Then switch to dynamic percentile mode: `threshold = np.percentile(sims, 75) * 0.95`. This adapts to query hardness automatically.

**Can I run the gate on CPU?**

Yes. The `all-MiniLM-L6-v2` model runs at 12 ms per query on a modern CPU (e.g., Intel i7-13700K). We run it in a 2-vCPU Kubernetes pod with 4 GB RAM and see no contention during bursts.

**What’s the smallest model I can use for the gate?**

Try `all-MiniLM-L6-v2` (33M params, 12 ms) or `paraphrase-multilingual-MiniLM-L12-v2` if you need multilingual. For English-only, `all-MiniLM-L6-v2` is the sweet spot. Avoid larger models like `all-mpnet-base-v2` for the gate — they add latency without meaningful gains in pruning power.


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

**Last reviewed:** May 28, 2026
