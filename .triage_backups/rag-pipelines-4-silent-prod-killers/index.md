# RAG pipelines: 4 silent prod killers

Retrieval-augmented generation (RAG) demos work on a laptop. Production traffic exposes a different class of problem: failures that return HTTP 200 with plausible but wrong answers, or that degrade gradually until a pager fires at 3 a.m. This article covers the failure modes that appear once a pipeline is serving real concurrency, and how to instrument each one so you can confirm or rule it out in your own stack.

## Why RAG pipelines fail quietly

A RAG pipeline has more moving parts than a typical CRUD service: an embedding model, a vector index, a metadata filter layer, a reranker, a cache, and an LLM. Each has its own tokenizer, its own memory model, and its own notion of what "the same input" means. When two components disagree about input normalization, nothing throws. The pipeline returns a result. It is simply the wrong result.

The other reason failures are quiet is that most RAG metrics measure the wrong layer. Request latency, HTTP status, and QPS all look healthy while retrieval precision collapses. You need retrieval-specific signals: recall against a labelled set, reranker score distributions, cache hit ratio by key class, and index-level latency percentiles broken out by filter predicate.

## Failure mode 1: tokenizer and normalization drift

Every model in the pipeline tokenizes text its own way. An embedding model may normalize Unicode punctuation, strip diacritics, or lowercase differently than the reranker or the LLM that consumes the retrieved chunks. When the index was built with one normalization and the query path uses another, exact-match signals (BM25, reranker cross-attention over rare tokens) silently degrade.

A common concrete case: text containing curly apostrophes (`’`, U+2019) is indexed after one model normalizes them to straight quotes (`'`, U+0027), while a reranker trained on the original distribution expects the curly form. Product names, proper nouns, and code identifiers are the tokens most likely to be affected.

**How to detect it.** Build a small labelled query set (a few hundred pairs is enough). For each query, log the raw string, the normalized string produced by each stage, and the retrieved chunk IDs. Diff the normalized forms across stages. Any stage that produces a different normalized string than the indexer is a candidate.

**How to fix it.** Normalize once, at ingestion, and store the normalized form. Pass the same normalized text to every downstream stage. In Python, `ftfy` is a reasonable choice for Unicode repair; the important property is that the same function runs on both the index and query paths, not which library you pick.

```python
import ftfy
import unicodedata

def normalize(text: str) -> str:
    # Repair mojibake and normalize punctuation to a single canonical form.
    text = ftfy.fix_text(text)
    # NFC keeps accented characters composed; NFKC would fold them.
    text = unicodedata.normalize("NFC", text)
    return text.strip()

# Apply at ingestion AND at query time. Never apply at only one.
indexed_text = normalize(raw_document)
query_text = normalize(user_query)
```

**How to measure the impact.** Compare recall@k on the labelled set before and after unifying normalization. If the pipeline was relying on exact token matches, you will see the delta immediately. Do not guess at a percentage; measure it on your own data.

## Failure mode 2: GPU memory fragmentation under concurrent reranking

Rerankers are usually the second-largest memory consumer after the LLM. A reranker served through an inference runtime such as ONNX Runtime or TensorRT allocates GPU memory per request. Under bursty concurrency, short-lived tensors of varying sizes fragment the allocator. The result is `CUDA out of memory` errors even though reported memory usage is well below the card's capacity.

This shows up as intermittent 500s under load, not as a steady degradation. It is easy to misdiagnose as a leak.

**How to detect it.** Log `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` (or the equivalent for your runtime) at request start and end. A leak shows rising allocated memory that never returns to baseline. Fragmentation shows stable allocated memory but rising reserved memory, or allocation failures at a total well below capacity.

**How to fix it.** Three levers, in order of impact:

1. Serve the reranker from a persistent session rather than creating one per request. Session creation is expensive and allocates a fresh memory pool each time.
2. Pre-allocate input and output tensors at the maximum batch size you will serve, and reuse them across requests.
3. Cap the batch size explicitly. A reranker that accepts unbounded batches will fragment fastest.

```python
import onnxruntime as ort
import numpy as np

opts = ort.SessionOptions()
opts.enable_mem_pattern = True
opts.enable_cpu_mem_arena = False  # GPU-only deployment

session = ort.InferenceSession(
    "reranker.onnx",
    sess_options=opts,
    providers=[("CUDAExecutionProvider", {"device_id": 0})],
)

# Pre-allocate buffers at the maximum batch size you will serve.
MAX_BATCH = 32
MAX_SEQ = 512
input_ids = np.zeros((MAX_BATCH, MAX_SEQ), dtype=np.int64)
attention_mask = np.zeros((MAX_BATCH, MAX_SEQ), dtype=np.int64)

def rerank(batch_tokens: np.ndarray) -> np.ndarray:
    n = batch_tokens.shape[0]
    assert n <= MAX_BATCH
    input_ids[:n] = batch_tokens
    attention_mask[:n] = 1
    return session.run(
        ["scores"],
        {"input_ids": input_ids[:n], "attention_mask": attention_mask[:n]},
    )[0]
```

**Trade-off.** Pre-allocation raises baseline GPU memory by the size of the buffers, but eliminates the fragmentation-driven OOMs. Measure both numbers on your hardware before committing.

## Failure mode 3: metadata index skew

Vector databases that support scalar filtering (Milvus, Qdrant, Weaviate) build scalar indexes to accelerate `WHERE`-style predicates. When the distribution of values in a filtered field is heavily skewed, the index for the minority value becomes inefficient, and queries filtering on it slow down disproportionately.

A concrete pattern: a `locale` field where 95% of rows are `en_US` and 5% are `vi_VN`. A query filtering on `vi_VN` may scan nearly the whole index. Under high concurrency this manifests as a small set of queries with P99 latency an order of magnitude worse than the median.

**How to detect it.** Break out latency percentiles by filter predicate. If `filter=locale:vi_VN` has a P99 that is 10x the P99 of `filter=locale:en_US`, you have skew. Most vector databases expose per-query timing in their logs or metrics endpoint.

**How to fix it.**

- Add a compound index over the frequently co-filtered fields (for example, `(category, locale)`) so the query planner can use a more selective prefix.
- Ensure the query planner is actually using the compound index. In Milvus, `EXPLAIN`-style output or the query plan in the logs will tell you.
- If a single value dominates and queries frequently target it, consider partitioning the collection by that field so the dominant value lives in its own partition.

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

collection = Collection("faq_vectors")

# Compound index over the fields that are filtered together.
collection.create_index(
    field_name="category",
    index_params={
        "index_type": "INVERTED",
        "params": {},
    },
)
collection.create_index(
    field_name="locale",
    index_params={
        "index_type": "INVERTED",
        "params": {},
    },
)
```

**Trade-off.** Compound indexes cost memory and slow ingestion slightly. The benefit only materialises if the query planner can use them, so verify with the plan output rather than assuming.

## Failure mode 4: cache stampedes and cold-start latency

A local cache (in-process, or a sidecar) is the standard way to shave latency off hot queries. The failure mode is the cold start: when the cache is empty, every request for a hot key hits the vector database simultaneously. This is a cache stampede. The database sees a burst of identical queries, latency spikes, and the cache fills slowly because each request is doing the same expensive work.

A second variant: TTL expiry. If many keys were written at the same time (for example, after a batch ingestion), they expire together and produce a synchronised stampede.

**How to detect it.** Log cache hit ratio as a time series. A stampede shows as a sharp drop in hit ratio followed by a spike in vector-database QPS. Correlate the two. If the spike is periodic, it is TTL synchronisation; if it is at deploy time, it is cold start.

**How to fix it.**

- Pre-warm the cache on startup by issuing the top-N queries from a recorded query log. This is a batch job, not a per-request one.
- Add jitter to TTLs so keys do not expire together.
- Use a single-flight or request-coalescing layer so that concurrent requests for the same key wait on one database call rather than issuing N.

```python
import asyncio
from collections import defaultdict

_inflight: dict[str, asyncio.Future] = defaultdict(asyncio.Future)

async def get_or_fetch(key: str, fetch):
    """Coalesce concurrent requests for the same key into one fetch."""
    if key in _inflight:
        return await _inflight[key]
    fut = _inflight[key]
    try:
        value = await fetch(key)
        fut.set_result(value)
        return value
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        _inflight.pop(key, None)
```

**Trade-off.** Request coalescing adds a small amount of coordination overhead and a failure mode of its own: if the single fetch hangs, all waiters hang. Bound the fetch with a timeout.

## Failure mode 5: DNS and connection-pool contention

At high QPS, the network layer becomes a source of latency that is invisible in application traces. Two patterns recur:

**DNS resolution.** A service that resolves its vector-database hostname on every request pays a resolver round-trip each time. Under load, the resolver (often CoreDNS in Kubernetes) can throttle or drop packets, causing multi-second stalls. Use a long-lived client with connection pooling and a cached resolved address, and prefer a stable service name over one that churns.

**Connection-pool exhaustion.** If the vector-database client and the LLM client share a connection pool, a slow LLM call can starve vector-search traffic. Isolate pools per downstream dependency and set `max_connections` based on the concurrency you actually serve, not the concurrency you hope to serve.

```python
import httpx

# Separate pools per downstream dependency.
vector_client = httpx.AsyncClient(
    base_url="http://milvus:19530",
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=httpx.Timeout(2.0, connect=0.5),
)

llm_client = httpx.AsyncClient(
    base_url="http://llm-gateway:8080",
    limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
    timeout=httpx.Timeout(30.0, connect=0.5),
)
```

**How to detect it.** Instrument DNS resolution time separately from connection time and from request time. In Kubernetes, CoreDNS exposes `coredns_dns_request_duration_seconds` and `coredns_dns_responses_total`; a rise in `SERVFAIL` or `NXDOMAIN` responses under load is the signal. For pool exhaustion, log pool wait time (httpx exposes this via instrumentation hooks; most clients do).

## Failure mode 6: metrics cardinality and scrape timeouts

Prometheus scrapes on a fixed interval. As the number of time series grows, the scrape payload grows, and eventually the scrape takes longer than the interval. The symptom is gaps in your dashboards and `scrape_timeout` errors in the Prometheus log.

The cause is almost always unbounded label cardinality: a label whose value is a user ID, a query string, or a raw error message. Each distinct value creates a new time series.

**How to detect it.** `prometheus_tsdb_head_series` shows the current series count. If it grows linearly with traffic, you have a cardinality leak. `scrape_duration_seconds` and `scrape_samples_scraped` show the cost per target.

**How to fix it.**

- Never put unbounded values in labels. Use a bounded set: status class (`2xx`, `4xx`, `5xx`), endpoint template (`/retrieve/:collection`), not the raw path.
- Aggregate high-cardinality dimensions before they reach Prometheus. If you need per-query detail, use traces or logs, not metrics.
- Increase `scrape_interval` only as a last resort; it reduces resolution for every metric.

## Choosing a vector store: a decision checklist

The right choice depends on your access pattern, not on a benchmark table. Use these questions:

| Question | If yes | If no |
|---|---|---|
| Do you need hybrid (vector + keyword) search in one query? | Prefer a store with native BM25 or sparse-vector support | A pure vector store is simpler |
| Is your filtered field heavily skewed? | Verify compound-index support before committing | Standard scalar indexes are fine |
| Do you need to run air-gapped? | Prefer a store with a single-binary or embedded mode | Managed or distributed stores are viable |
| Is your team already operating a relational database at scale? | Consider the vector extension for that database before adding a new system | A dedicated vector store is worth the operational cost |
| Is your expected QPS under a few hundred? | Almost any option works; optimise for operational simplicity | Measure index build time and filtered-query latency under load |
| Do you need per-tenant isolation? | Prefer a store with collection or partition-level isolation | Shared collections with metadata filters are fine |

The honest answer is that no benchmark table transfers between workloads. Vector search latency depends on dimensionality, index type, filter selectivity, and hardware. The only useful benchmark is one you run against your own data with your own query distribution.

## How to measure any of this

The pattern is the same for every failure mode above:

1. **Define a labelled set.** A few hundred query-document pairs with known relevant documents. This is the ground truth for recall.
2. **Instrument the pipeline per stage.** Log input hash, normalized input, retrieved IDs, reranker scores, and final answer. Correlate by request ID.
3. **Run a load test at your target concurrency.** Not at 1.1x, at the concurrency you actually expect at peak. Use a tool that can hold steady state for at least 30 minutes; transient bursts hide steady-state problems.
4. **Break out metrics by dimension.** P50 and P99 overall are not enough. Break out by filter predicate, by cache hit/miss, by collection, by tenant.
5. **Inject failures.** Kill the vector database mid-run. Throttle the LLM. Fill the cache. Observe recovery time and error rate.

The commands are mundane:

```bash
# Steady-state load test with a fixed concurrency for 30 minutes.
hey -z 30m -c 200 -m POST \
  -H "Content-Type: application/json" \
  -d '{"query":"..."}' \
  http://localhost:8000/retrieve

# Watch GPU memory during the run.
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Watch Prometheus scrape health.
curl -s localhost:9090/api/v1/query?query=scrape_duration_seconds | jq
```

## A note on cost

Cost claims about RAG stacks are almost always wrong because they depend on instance type, region, commitment level, and utilisation. The one useful exercise is to build your own cost model from first principles:

- List each component (vector store, cache, reranker GPU, LLM calls).
- For each, note the instance type and its on-demand hourly rate from your cloud provider's pricing page.
- Multiply by the number of instances you actually run at peak.
- Add egress, storage, and any managed-service markup.

Do this in a spreadsheet, not in your head. The result will be specific to your workload and will change as your traffic grows. Revisit it quarterly.

## The 30-minute action

Pick the failure mode above that is most plausible for your stack, and add one instrument to it in the next 30 minutes. If you are unsure where to start, add per-filter-predicate latency logging to your vector-search call and run a 10-minute load test at your expected peak concurrency. Compare the P99 for your most selective filter against the P99 for your least selective one. A gap larger than 3x is your next debugging session.
