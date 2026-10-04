# AI on slow networks: Nairobi teams' 2026 playbook

Users experience AI features as slow or fast based on a few measurable moments, not on the raw round-trip time of the model call. A feature that calls a model with 400–700 ms of average latency can still feel instant, and a feature that calls a model in 120–180 ms can still feel sluggish. The difference is usually architecture: what gets cached, where the request is orchestrated, and what the client renders while it waits.

This article covers a three-layer approach — pre-computed caches, client-side latency compensation, and a fallback policy that only calls the cloud on a cache miss — and the failure mode that breaks it most often: the cache stampede.

## The one-paragraph version

A typical low-latency AI stack has three layers. First, a cache that serves a large share of requests in tens of milliseconds. Second, client-side rendering that shows partial or progressive results so the user is not staring at a blank state. Third, a fallback policy that only reaches the model provider on a cache miss. The layer that trips teams up is the cache stampede: when a missing key triggers many identical requests at once, compute usage and latency spike together. The rest of this article is about building that stack and defending it.

## Why "shorter latency equals better UX" misleads teams

The intuitive model — measure the API, see 400 ms, blame the network — leaves three factors out.

1. **Perceived latency vs. actual latency.** Users wait for the first visible response and the final UI update, not for the full round trip. A 400 ms API can feel instant if the client renders progressive results.
2. **Cacheable vs. non-cacheable traffic.** In many AI applications a large fraction of requests are identical or near-identical. Caching turns those into single-digit-millisecond reads. The exact share depends on the product; measure it rather than assuming.
3. **Edge topology.** A request from one region to a model endpoint in another crosses undersea cables and regional IXPs, adding network time before the model even starts. An edge worker close to the user removes much of that pre-processing delay.

Teams that focus only on raw latency either over-provision expensive model capacity or abandon the feature. The more interesting failure mode is the opposite: caching indiscriminately and ending up with a high hit ratio on stale or incorrect results.

## A mental model: three stages, three budgets

Think of an AI feature as a pipeline:

1. **Input** — the user types a prompt or uploads a file.
2. **Compute** — tokenization, embedding, model inference, retrieval.
3. **Output** — return tokens, generate UI, update state.

Each stage has a latency budget you can shrink independently.

- Input and output are mostly UI and network. Shrink them with edge deployment and client-side rendering.
- Compute is where most teams focus, and it is the hardest to optimize without more GPUs or better models.
- The hidden lever is the *gap* between compute calls. Many AI features repeat the same or similar compute. Cache the result of that compute and perceived latency collapses to a cache read plus freshness validation.

A useful analogy is a coffee shop pre-brewing popular drinks during peak hours. Regulars get their order in seconds; the barista still makes fresh batches for rare orders. The cache is the warmer, not the barista.

## A worked example with stated assumptions

The numbers below are illustrative, chosen to make the arithmetic visible. Substitute your own measurements before making decisions.

**Feature:** an autocomplete endpoint that embeds a partial query, searches a vector store, and returns the top 5 suggestions.

**Scenario A — direct call.**

- Client POSTs to a function in a distant region.
- Function calls a hosted embedding model: assume 150 ms.
- Function calls a vector search: assume 60 ms.
- Function returns JSON: assume 10 ms.
- Total round trip: 150 + 60 + 10 = **220 ms**.
- Cost: dominated by embedding plus function compute.

**Scenario B — cached plus edge.**

- An edge worker near the user intercepts the request.
- Worker checks the cache: assume 5 ms read.
- Cache hit (assume 85 % of queries): worker returns cached suggestions in roughly 10 ms.
- Cache miss (the remaining 15 %): worker runs the same 220 ms pipeline.
- Total round trip on hit: ~15 ms.
- Total round trip on miss: ~220 ms, plus whatever the warming step costs asynchronously.

The model latency is unchanged. The *user* latency for the hit share drops by an order of magnitude. The miss share still waits, which is where client-side compensation matters.

To replace these assumptions with real numbers, instrument four things: the cache read time, the cache hit ratio, the full miss-path time, and the client time-to-first-render. Log them per request and compare percentiles, not averages.

## How this connects to things you already know

If you have used a CDN to cache static assets, this is the same idea applied to AI compute. The difference is that the cache key is a normalized prompt hash plus a similarity threshold, and the value is a list of suggestions rather than an image or HTML.

If you have used incremental static regeneration, you have already warmed caches in the background. This is the same pattern for AI features.

If you have used Redis for rate limiting or sessions, you already know how to shard, evict, and monitor it. The new twist is the staleness budget: for some AI features, results that are a few minutes old are acceptable, so TTLs can be longer than typical web caches.

The hard part is the thundering herd: when a popular prompt misses the cache, many users trigger the same compute at once. That is the failure mode most teams hit first.

## Common misconceptions, corrected

### "Caching AI results will give wrong answers."

For features like autocomplete, related articles, and Q&A suggestions, slightly stale results are usually acceptable. The work is in choosing a TTL and invalidating aggressively when the underlying data changes — new documents, updated catalogs, edited content.

### "The cache stampede is rare."

In any product with meaningful traffic, a single trending prompt can trigger many concurrent cache misses. Without a guard, you pay for the same compute many times over and latency spikes. A naive per-key lock is also a trap: if hundreds of requests contend for the same lock, they serialize and all wait on the first one.

### "Edge workers are only for static files."

Edge runtimes can orchestrate full AI pipelines. A worker near the user can call a model endpoint in another region; the network handshake still costs time, but the pre-processing delay shrinks so compute latency dominates instead of network latency.

### "Redis is too slow for AI."

A cache read is typically an order of magnitude faster than model inference. The bottleneck is usually the client or the network path, not the cache itself. Measure your own GET and SET percentiles before assuming otherwise.

## The advanced version

### Multi-tier caching

Add a second tier: an in-memory LRU cache inside the worker itself. If the remote cache misses, check the local cache. If it is there, return it immediately. This shaves time off the hit path and absorbs stampede spikes without reaching the remote cache at all.

```javascript
// wrangler.toml: compatibility_date = "2024-01-01"
// kv_namespaces = [
//   { binding = "AI_CACHE", id = "...", preview_id = "..." }
// ]

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const key = url.pathname + "?" + url.searchParams.toString();
    const normalized = key.toLowerCase().trim();

    // Tier 1: in-worker cache (Cache API, keyed by full URL)
    const cacheKey = new Request(`https://cache.internal/${encodeURIComponent(normalized)}`);
    const local = await caches.default.match(cacheKey);
    if (local) return local;

    // Tier 2: remote cache
    const cached = await env.AI_CACHE.get(normalized);
    if (cached) {
      const response = new Response(cached, { headers: { 'X-Cache': 'HIT' } });
      // Warm the local tier without blocking the response
      ctx.waitUntil(caches.default.put(cacheKey, response.clone()));
      return response;
    }

    // Tier 3: compute + store
    const start = Date.now();
    const result = await computeResult(normalized);
    const elapsed = Date.now() - start;

    await Promise.all([
      env.AI_CACHE.put(normalized, result, { expirationTtl: 300 }), // 5 min
      caches.default.put(cacheKey, new Response(result, { headers: { 'X-Cache': 'MISS' } }))
    ]);

    return new Response(result, {
      headers: { 'X-Cache': 'MISS', 'X-Latency': String(elapsed) }
    });
  }
};
```

Note the two fixes over a naive sketch: the Cache API is keyed by a synthetic `Request` rather than a bare string, and the background write uses `ctx.waitUntil` instead of an undefined `event`.

### Stampede guard with a single-flight lock

The goal is to let exactly one request compute a missing key while the others wait briefly and then read the result.

```python
import asyncio
import random
import redis.asyncio as redis

r = redis.Redis(host="cache.internal", port=6379, db=0)

LOCK_TTL_MS = 5000
MAX_WAIT_MS = 2000

async def get_or_compute(key: str):
    cached = await r.get(key)
    if cached is not None:
        return cached

    lock_key = f"lock:{key}"
    got_lock = await r.set(lock_key, "1", nx=True, px=LOCK_TTL_MS)

    if got_lock:
        try:
            result = await compute_result(key)
            await r.set(key, result, ex=300)
            return result
        finally:
            await r.delete(lock_key)

    # Someone else is computing. Wait with jittered backoff, then re-check.
    waited = 0
    while waited < MAX_WAIT_MS:
        delay = random.uniform(0.01, 0.1)
        await asyncio.sleep(delay)
        waited += delay * 1000
        cached = await r.get(key)
        if cached is not None:
            return cached

    # Fallback: compute ourselves rather than fail the request.
    return await compute_result(key)
```

Two details matter. The lock TTL must exceed the expected compute time, or a second request will start computing while the first is still running. And the waiter must have a bounded wait with a fallback, or a crashed lock owner leaves requests hanging.

### Adaptive cache warming

Warming every miss wastes compute. Warm only queries that cross a popularity threshold, and avoid warming near-duplicates of the same prompt.

```python
import asyncio
from collections import defaultdict

POPULARITY_THRESHOLD = 10  # queries seen at least this many times in the window
counts = defaultdict(int)

async def handle_query(query: str):
    key = normalize(query)

    cached = await r.get(key)
    if cached is not None:
        return cached

    counts[key] += 1
    if counts[key] >= POPULARITY_THRESHOLD:
        # Fire-and-forget warm; do not block the response.
        asyncio.create_task(warm_cache(key))

    return await get_or_compute(key)
```

In production, replace the in-process counter with a shared structure so every instance sees the same popularity signal, and deduplicate similar prompts before warming.

### Eviction policy

For an AI cache, least-frequently-used eviction often beats least-recently-used, because a popular key inserted long ago should not be evicted ahead of a key that was touched once recently. In `redis.conf`:

```
maxmemory 4gb
maxmemory-policy allkeys-lfu
```

Confirm the policy your deployment actually supports before relying on it; the available policies vary by version and distribution.

### Observability: the three numbers that matter

1. **Cache hit ratio.** Below roughly 60 % usually means the cache keys are too specific or the TTLs are wrong. The right target depends on your feature; measure it before setting one.
2. **P95 latency of cache misses.** If this is much higher than your compute budget, the pipeline is the bottleneck, not the cache.
3. **Stampede events per day.** Count how often many requests compute the same key concurrently. If it happens regularly, add a single-flight lock or probabilistic early refresh.

## Comparison of the layers

| Layer | Category | Typical role | When to use | Pitfall |
|-------|----------|--------------|-------------|---------|
| Edge worker | Edge runtime | Orchestration, request routing | All AI features | CPU time limits; long prompts can time out |
| Remote cache | Key-value store | Shared result storage | The cacheable share of requests | Stampede on miss; needs a guard |
| In-worker cache | In-memory LRU | Absorb hot keys | High-frequency, short TTL | Memory limited; evict aggressively |
| Vector search | Retrieval service | Similarity lookup | Hybrid or semantic features | Index freshness; update on document change |
| Model API | Hosted inference | Embeddings, generation | Rare or high-value prompts | Cold-start latency spikes |
| Warmer | Background job | Pre-fetch popular keys | Queries above a popularity threshold | Duplicate warming; deduplicate first |

## FAQ

**How should I build the cache key for AI autocomplete?**

Normalize the prompt: lowercase, trim whitespace, strip punctuation, and optionally truncate. Use that normalized string as the key. For semantic search, include a similarity threshold in the key so you do not return results that are only vaguely similar. A common failure mode is using raw user input as the key, which misses on trivial casing or spacing differences.

**How do I avoid a cache stampede?**

Use a single-flight lock per key with a bounded, jittered wait and a fallback. A short-TTL `SET NX` works well: the winner computes, the losers wait briefly and re-read. Do not use a global lock; it serializes all misses and kills throughput.

**What TTL should I pick?**

Start with 300 seconds for most features. For static data such as a product catalog, extend it. For fast-moving data such as news or social feeds, shorten it. The mistake to avoid is a long TTL that surfaces outdated results after a major content update. Watch the hit ratio and the staleness complaints together.

**Can I run Python at the edge?**

Not natively in most edge runtimes; JavaScript and WebAssembly are the usual targets. Python can be compiled to WebAssembly, but a large model running that way is typically slower than calling a hosted endpoint. The edge is best used for orchestration, caching, and warming, not heavy inference.

## What to do in the next 30 minutes

Pick your slowest AI endpoint and measure the cacheable share of its traffic. In your server logs or analytics, group requests by normalized prompt and count how many distinct prompts account for most of the volume. If a small set of prompts dominates, that is your cache key space and your first win.

Then check the cache read path directly:

```bash
redis-cli --latency-history -h cache.internal -p 6379
redis-cli GET "ai:autocomplete:how do i deploy redis on flyio"
```

If the key exists and is recent, the cache is already doing its job; the next step is a stampede guard. If it does not exist, add a 300-second TTL on the hot keys and watch the hit ratio over the next hour.
