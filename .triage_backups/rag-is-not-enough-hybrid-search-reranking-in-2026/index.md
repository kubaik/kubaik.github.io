# RAG is not enough: hybrid search + reranking in 2026

Retrieval-augmented generation has a ranking problem, not a recall problem. Dense embedding search will usually put the right passage somewhere in the top hundred results. It will not reliably put it first. And an LLM reading a context window full of near-misses will answer confidently from the wrong chunk.

The standard remedy is a two-stage pipeline: cheap wide retrieval, then an expensive narrow reranker. This article covers why that split works, how to build it, and how to measure whether it is working in your own system.

## Why single-stage retrieval disappoints

A bi-encoder embeds the query and each document independently, then compares the two vectors. That independence is what makes it fast enough to search millions of documents, and it is also its weakness: the query never gets to "look at" the document. Paraphrase, negation, and multi-hop conditions all get compressed into a single vector before the comparison happens.

Sparse lexical retrieval has the mirror-image problem. BM25 is excellent at exact terms, rare identifiers, product codes, and names. It is indifferent to meaning. A query for "how do I cancel" and a document titled "Cancellation policy" share a stem and will match; a query for "how do I stop my subscription" and the same document may not.

Both failure modes are ranking failures. The correct document is often retrieved and then buried at position 40 behind documents with higher lexical overlap or closer vector proximity.

Context windows make this worse rather than better. A larger window lets you stuff more candidates into the prompt, but every additional irrelevant chunk dilutes the signal and increases the chance the model anchors on the wrong passage. Retrieval quality and generation quality are separate concerns, and improving one does not fix the other.

## The funnel model

Think of retrieval as a funnel with two stages.

Stage 1 is a coarse sieve. It is fast, wide, and forgiving. Run BM25, dense vector search, or both, and return somewhere between 50 and 200 candidates. Recall matters here; precision does not.

Stage 2 is a precision filter. A cross-encoder takes each query-document pair and scores them jointly, letting the query attend to the document text directly. It is far slower per pair, which is exactly why it only ever sees the shortlist.

The critical property: the reranker does not need to be smarter than your retriever in general. It only needs to be better at ordering a small candidate set. That is a much easier problem, and small models solve it well.

A useful intermediate stage is query rewriting or expansion before Stage 1 — correcting typos, expanding abbreviations, or generating paraphrases so the first stage has more to work with. This is optional and adds its own latency, so treat it as a tuning knob rather than a default.

## A worked example

Consider a support assistant over an internal policy corpus. The goal is to answer questions like "how do I dispute a charge?" with citations to current policy documents.

The pipeline:

1. Stage 1: BM25 over a `question` field, returning the top 100 candidates.
2. Stage 2: a cross-encoder scoring each (query, passage) pair, keeping the top 5.
3. Stage 3: an LLM generating an answer from those 5 passages only.

```python
from elasticsearch import Elasticsearch
from sentence_transformers import CrossEncoder

es = Elasticsearch("https://search.internal:9200")

query = "How do I dispute a charge?"

# Stage 1: wide lexical retrieval
response = es.search(
    index="policies_v1",
    query={"match": {"text": query}},
    size=100,
)
candidates = [hit["_source"]["text"] for hit in response["hits"]["hits"]]

# Stage 2: joint scoring of query-document pairs
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
scores = reranker.predict([(query, passage) for passage in candidates])

ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
top_passages = [passage for passage, _ in ranked[:5]]
```

Two details matter here. First, `predict` accepts a list of pairs and batches internally, so do not loop over candidates one at a time. Second, the reranker's `max_length` truncates long passages; if your chunks are much longer than 512 tokens, the tail of each chunk is invisible to the reranker and you should chunk smaller.

The same pattern works with dense first-stage retrieval. Swap the Elasticsearch call for a vector query and the rest is unchanged:

```python
# Stage 1 alternative: dense retrieval
query_vector = embedder.encode(query).tolist()
response = es.search(
    index="policies_v1",
    knn={"field": "embedding", "query_vector": query_vector, "k": 100},
)
candidates = [hit["_source"]["text"] for hit in response["hits"]["hits"]]
```

And the two can be combined before reranking, which is the actual "hybrid" step:

```python
def reciprocal_rank_fusion(rank_lists, k=60):
    """Combine multiple ranked lists by reciprocal rank."""
    fused = {}
    for ranks in rank_lists:
        for position, doc_id in enumerate(ranks, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + position)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)
```

Reciprocal rank fusion is a common way to merge a BM25 list and a vector list without having to normalize two incomparable score scales. The constant `k=60` is the value used in the original RRF paper and is a reasonable default; it damps the influence of top ranks so that a document appearing high in both lists beats a document appearing first in only one.

## Measuring whether it helps

Published leaderboard numbers do not transfer to your corpus. The only number that matters is the delta on your own data. Here is how to get it.

**Build a labeled set.** Take 100 to 200 real queries from your logs. For each, have a human mark which passages are relevant. This is a few hours of work and it is the single highest-leverage investment in the whole pipeline. Without it, every tuning decision is guesswork.

**Instrument the right things.** Log, per query:

- The rank of the first relevant passage before reranking
- The rank of the first relevant passage after reranking
- The reranker's score for the top result and for the second result
- The number of tokens sent to the LLM
- End-to-end latency at p50 and p95

**Compare configurations.** Run the same query set through BM25-only, dense-only, hybrid without reranking, and hybrid with reranking. For each, compute MRR@10 (mean reciprocal rank of the first relevant result) and recall@5. The interesting comparison is not "does reranking help" but "how much does it help per millisecond of added latency."

**Watch the score gap.** The difference between the top and second reranker scores is a cheap proxy for confidence. If that gap is near zero on many queries, the reranker is not separating relevant from irrelevant and you likely have a chunking or labeling problem rather than a model problem.

**Measure latency honestly.** Reranking latency scales linearly with the number of candidates. Benchmark at your actual k, on your actual hardware, with your actual batch size. A single-query benchmark will overstate throughput because it ignores batching.

## Failure modes to expect

**Stale documents with high lexical overlap.** A superseded policy that uses the same vocabulary as the current one will score well on both BM25 and a reranker trained on relevance rather than recency. Mitigation: filter by date or validity flag before reranking, or add a recency feature to the ranking. Do not expect the reranker to learn your document lifecycle.

**Short or ambiguous queries.** A one-word query gives the reranker almost nothing to condition on. The scores will be tightly clustered and the ordering will be close to arbitrary. Mitigation: detect short queries and either ask a clarifying question or fall back to a retrieval strategy that uses session context.

**Chunking that splits semantics.** If a policy's exception clause lives in a different chunk from its main rule, no reranker can recover it. Reranking operates on whatever units you indexed. Mitigation: chunk on document structure rather than fixed token counts, and consider overlapping chunks.

**Truncation at the reranker's max length.** Cross-encoders have a hard input limit. Passages longer than that limit are silently truncated. If your chunks are 1000 tokens and your reranker's limit is 512, you are ranking on the first half of each chunk. Mitigation: align chunk size to the reranker's limit.

**Score scale drift.** Reranker scores are not calibrated probabilities. A threshold that works on one corpus will not transfer to another. Mitigation: tune thresholds on your labeled set, and re-tune when the corpus or query distribution changes.

**Latency spikes under load.** The reranker is usually the last thing to get a GPU, and the first thing to get preempted. Mitigation: cache reranker scores keyed by (query hash, document id) with a short TTL, and make the reranker optional — if it times out, fall back to first-stage ordering rather than failing the request.

## Where the pieces live in practice

The two-stage pattern is available in several forms, and the right choice depends on how much control you need.

| Approach | First stage | Reranker | Control | Operational cost |
|---|---|---|---|---|
| Self-hosted | Elasticsearch or Postgres + pgvector | Cross-encoder on your own GPU/CPU | Full | You own scaling and uptime |
| Managed vector database | Vendor's hybrid query | Vendor's reranker or your own | Partial | Vendor pricing and limits |
| Managed retrieval API | Vendor's hybrid query | Vendor's built-in reranker | Least | Simplest, least tunable |

A self-hosted stack typically looks like a search engine for Stage 1 and a small cross-encoder served behind an HTTP endpoint for Stage 2. The reranker is stateless, which makes it easy to scale horizontally and easy to cache in front of.

A managed vector database with hybrid query support lets you express the first stage in one call, but check whether the reranker is applied before or after your `top_k` cutoff — applying it after is a common source of confusion.

A managed retrieval API that bundles hybrid search and reranking is the fastest path to a working system and the hardest to tune. It is a reasonable starting point if you have not yet built a labeled evaluation set, because you cannot tune what you cannot measure.

## A decision checklist

Before adding a reranker, confirm:

- You have a labeled query set of at least 100 examples.
- You know your current MRR@10 and recall@5 without reranking.
- Your chunks fit within the reranker's maximum input length.
- You have a latency budget and know how much of it reranking may consume.
- You have a fallback path if the reranker is unavailable.
- You are logging the reranker's score distribution, not just the final answer.

If the answer to the first two is no, build the evaluation set first. Everything else is premature.

## Common misconceptions

**"Reranking is just another embedding model."** A cross-encoder scores a query-document pair jointly and produces a relevance score, not an embedding. It cannot be indexed or precomputed, which is why it only runs on a shortlist.

**"You need a large GPU."** Small cross-encoders in the tens of millions of parameters run acceptably on CPU with an optimized runtime, and comfortably on a modest GPU. Benchmark your own candidate counts rather than trusting published throughput figures.

**"Hybrid search means combining sparse and dense vectors."** Hybrid search describes the first stage. Reranking describes the second. You can rerank a dense-only shortlist, a BM25-only shortlist, or a fused list. The stages are independent choices.

**"More candidates always means better results."** Recall improves with k, but latency grows linearly and the reranker's job gets harder as the shortlist fills with marginal candidates. Tune k on your labeled set; do not simply maximize it.

## FAQ

**Should the reranker be an LLM?**
An LLM can score relevance, but it costs far more per query and adds substantial latency compared to a small cross-encoder. Reserve the LLM for generation and use a purpose-built reranker for ranking.

**Does reranking work for non-English queries?**
It depends entirely on the reranker's training data. Multilingual cross-encoders exist, but their quality on any specific language should be verified on your own labeled set before you rely on it. Fine-tuning on in-domain pairs is usually necessary for low-resource languages.

**What is the smallest reranker worth using?**
Small cross-encoders in the 20-100M parameter range are the common starting point. The right size is the one that fits your latency budget while producing a measurable MRR improvement on your evaluation set. Measure, do not assume.

**How do I know the reranker is actually doing work?**
Log the rank change between first-stage and post-rerank ordering, and the score gap between the top two results. If the ordering rarely changes, the reranker is not adding value and you are paying latency for nothing.

**Can I skip Stage 1 and just rerank everything?**
No. Cross-encoder cost scales with the number of pairs scored. Reranking a million documents per query is not viable; the two-stage split exists precisely to make the expensive scorer affordable.

## What to do in the next 30 minutes

Pick one production query from your logs, run it through your current retrieval, and print the rank of the first genuinely relevant passage. Then add a cross-encoder reranker over the top 100 candidates and print the new rank. That single before-and-after comparison tells you more about whether reranking will help your system than any published benchmark.
