# Designing AI Micro-SaaS Infra on a $150/Month Budget

Most cost-planning guides for AI products assume a clean environment, patient timelines, and traffic that behaves. The design holds in the simple case and breaks in specific ways once the cache is cold, the upstream API rate-limits, or a single customer's workload grows ten-fold overnight. The notes below describe the tradeoffs that actually matter when the monthly infra ceiling is $150.

## The one-paragraph version

AI products leak money on undifferentiated infrastructure. Treat the prompt layer like an ORM and the API gateway like a database connection pool: cache aggressively, batch aggressively, and never let a cold start touch the LLM budget. Build the rest of the stack (auth, billing, queues) on serverless that scales to zero when traffic dies. The mental trap is believing every request requires a GPU in production; once that assumption is dropped, the arithmetic becomes tractable.

## Why this concept confuses people

Teams get stuck because they treat "AI infra" as one monolithic problem. They picture a single GPU-heavy endpoint that must be always-on, with every request billed at a high per-call rate and every prompt hitting the model in real time. That mental model leads to bills in the thousands before a single feature ships. The confusion is that the cost driver is rarely the inference itself — it is the orchestration around it: auth checks, prompt templating, rate limiting, billing meters, and the retry storms that follow an upstream hiccup. Separate those concerns and the budget becomes tractable.

A second confusion is the belief that launching an AI product requires running custom embeddings or a self-hosted vector database. That is only true for a bespoke retrieval pipeline. For classification, summarization, or question answering, managed model APIs generally win on price/performance once prompt-engineering overhead is counted, and they ship with retry policies, rate limits, and regional failover that would otherwise take weeks to replicate.

A third confusion is conflating "AI" with "real-time." Most micro-SaaS products do not need sub-50 ms responses; they need sub-500 ms p95 with occasional spikes. That tolerance is what makes batching, caching, and queue-offloaded background work viable, and those three techniques are usually the largest levers on the LLM bill.

## The mental model that makes it click

Think of the product as a restaurant kitchen:

- The chef (the LLM) is expensive and slow; use them only when required.
- The line cooks (prompt templating, tokenization, safety checks) prepare everything so the chef only sees the final dish.
- The host (API gateway) buffers orders, batches similar requests, and never lets the chef take a single order when a full tray is available.
- The dishwasher (cache) cleans plates so the chef is not re-run for an identical dish.
- The maître d' (auth and billing) checks identity and runs the card before the host hands the order to the kitchen.

If a station idles, shut it down to zero. If a station is overloaded, scale only that station. Every dollar should map to a visible station, not a hidden tax on the chef's time.

In concrete terms:

- Every API request goes through an edge runtime (auth, routing, caching) before touching the backend.
- The backend is a single serverless function that talks only to managed services: a cache, a queue, and the model API for the final step.
- No GPU runs in production; inference is rented by the token.
- Customers are charged based on tokens consumed, not on internal infra.

## A worked example with the arithmetic shown

The scenario below is illustrative. Every figure is derived from stated assumptions so the reader can substitute their own.

Assume: 1,000 monthly active users, 100 API calls per user per month, 200-token prompt per call, 500-byte cached response. That is 100,000 calls/month.

**Step 1 — Cache before you call.**

Key the cache on a content hash (document hash, prompt-template version, model version). On a hit, return the stored JSON; on a miss, call the model and store the result. If the hit rate is 60%, only 40,000 calls reach the model.

*How to measure hit rate:* instrument every handler with a counter for `cache_hit` and `cache_miss`, emit them as structured logs, and compute the ratio over a rolling window. A dashboard that shows only total request count cannot tell you whether caching is working.

**Step 2 — Batch the misses.**

Instead of one document per prompt, batch up to 10 documents with a delimiter and a strict output schema. This cuts the model-call count by roughly 90% at the same total token volume, because batching reduces per-call overhead and amortizes the system prompt.

```python
BATCH_PROMPT = """
Extract key fields from these documents.
Each document is delimited by <doc id="N"> ... </doc>.
Return a JSON array with one object per document, in order.

{documents}
"""
```

*How to measure batching efficiency:* log `tokens_in`, `tokens_out`, and `documents_per_call` per invocation. Compare cost-per-document before and after; if it does not drop, the batch is not amortizing the system prompt.

**Step 3 — Hide cold starts.**

Edge runtimes have a small cold-start penalty (single-digit milliseconds to low tens). Pre-warming a handful of instances per region after deploy keeps the first request after a quiet period fast. The cost of a warm-up is a few requests against a health endpoint; the benefit is that a cold start never lands on a user's first request.

*How to measure cold starts:* emit a `cold_start: true` flag on the first invocation of each instance and track its p95 separately from warm requests.

**Step 4 — Absorb bursts with a queue.**

Put the model call behind a queue with a visibility timeout and a dead-letter queue. On a 429 from the upstream, retry with exponential backoff and jitter, capped at a small number of attempts. After the cap, return a cached failure or a `503` with `Retry-After`, and log the incident.

*How to measure retry storms:* count `429` responses per minute and the average number of retries per request. A rising retry count with flat success rate means the backoff is too aggressive or the concurrency limit is too high.

**Step 5 — Auth and metering at the edge.**

Validate the token at the edge and pass verified claims downstream so the backend never re-parses JWTs. Meter usage by writing a small record per request and flushing in batches to an analytics store. The backend then only has three jobs: verify the passed claims, check the cache, and call the model on a miss.

**Step 6 — Size the cache honestly.**

A cache that holds 500-byte responses for 100,000 distinct keys needs roughly 50 MB of working set, plus overhead. Provision for the tail, not the average: the top 1% of keys may be 10× larger. A 1 GB managed Redis instance is generous for this workload; a smaller tier is usually enough. The point is to size from measured key size and cardinality, not from a guess.

*How to measure:* log `key_size_bytes` and `distinct_keys_per_day`. Multiply the p99 key size by the daily distinct key count to get a working-set estimate.

**Step 7 — Monitoring that costs nothing and says something.**

Free tiers of edge logs and a cloud metrics service are sufficient for a product at this scale. Alert on: cache hit rate below a threshold, function duration p95 above the latency budget, and upstream error rate above a few percent. These three signals catch the majority of cost and reliability regressions.

**Putting it together.**

The dominant cost at this scale is the cache tier, followed by the edge plan, then inference. Compute, queue, and analytics are rounding errors. The exact dollar figures depend on the current published prices of the specific services chosen, which change; the structural point is that caching and batching move inference from the largest line item to one of the smallest.

## How this connects to things you already know

If you have run a web service behind a reverse proxy with a Redis cache, this stack is that pattern stretched across multiple providers:

- Edge runtime = reverse proxy plus cache in front of the app server
- Serverless function = the app server
- Managed Redis = the in-memory cache layer
- Queue = the background worker queue
- Columnar analytics store = the warehouse

The only novelty is treating the model API as an external dependency rather than part of the stack. That means the same reliability patterns apply: retries, circuit breakers, caching, rate limiting. The cost of inference is externalized to the provider rather than internalized into a GPU budget.

The second familiar pattern is pay-per-use serverless. The edge runtime, the function platform, and the queue all scale to zero when traffic dies. The difference is applying that model to the prompt layer, not just the backend — which most teams miss because they think of the model as part of the backend rather than as a service they consume.

## Common misconceptions, corrected

**Misconception 1: "We need a GPU in production to keep latency low."**

Managed model APIs with regional endpoints typically return a thousand tokens well inside a 500 ms p95 budget. If the latency budget is 500 ms, batching and caching can be applied aggressively without ever touching a GPU. A local GPU becomes necessary only for custom fine-tuned models with strict data-residency requirements — a narrow case for most micro-SaaS.

**Misconception 2: "Caching prompts is unsafe; every user deserves a fresh response."**

For non-personalized tasks such as document extraction or summarization, a content hash is a valid cache key. If the user edits the document, the hash changes and the cache invalidates. The real risk is prompt drift: when the template or the model version changes, cached outputs may no longer match the expected format. Mitigate by including a template hash and model version in the cache key, so updating the prompt invalidates old entries automatically.

**Misconception 3: "Batching increases latency for the first request in the batch."**

Batching increases latency only if the handler waits for the full batch before responding. Use a fast lane: return a cached result immediately when available, and return an estimated completion time when the batch is still assembling. That keeps p95 latency low while still reducing cost.

```javascript
// Fast-lane batcher (illustrative)
const batch = new Map();
const MAX_BATCH_SIZE = 10;
const MAX_WAIT_MS = 200;

async function handle(request) {
  const docId = new URL(request.url).searchParams.get('docId');
  const cached = await CACHE.get(docId);
  if (cached) return new Response(cached);

  if (!batch.has(docId)) batch.set(docId, { promises: [] });
  const entry = batch.get(docId);

  if (entry.promises.length >= MAX_BATCH_SIZE) {
    return new Response(
      JSON.stringify({ status: 'queued', etaMs: MAX_WAIT_MS }),
      { status: 202 }
    );
  }

  const result = await Promise.race([
    batchPromises(entry.promises),
    new Promise(resolve =>
      setTimeout(() => resolve({ status: 'queued', etaMs: MAX_WAIT_MS }), MAX_WAIT_MS)
    )
  ]);

  return new Response(JSON.stringify(result));
}
```

**Misconception 4: "Serverless can't handle high request rates."**

Edge runtimes are designed for very high per-instance throughput, and function platforms offer provisioned concurrency when needed. The bottleneck is almost never the serverless platform; it is the model API or the cache layer. At 100,000 requests per month, the average rate is roughly 0.038 requests per second — orders of magnitude below any serverless limit.

## The advanced version, once the basics are solid

Apply these only after the cache hit rate is stable and the inference bill is a small fraction of the total.

**1. Prompt compression.** Use a smaller, cheaper model to condense long inputs before sending them to the larger model. Measure token reduction and latency change separately; compression that saves tokens but adds 300 ms of latency may not be worth it.

**2. Dynamic batch sizing.** Instead of a fixed batch size, stop adding documents when the cumulative token count approaches the context limit. This maximizes throughput without risking truncated outputs.

```python
def can_add(doc_text: str, current_tokens: int, tokenizer, limit: int = 8000) -> bool:
    new_tokens = len(tokenizer.encode(doc_text))
    return (current_tokens + new_tokens) <= limit
```

**3. Regional failover with latency-aware routing.** Route requests to the nearest model endpoint and fail over when the primary region degrades. Measure the latency difference between regions before committing; if the gap is small, the extra routing complexity may not pay for itself.

**4. Cache stampede protection.** When a popular key expires, many concurrent requests can hit the model at once. Serialize the recomputation with a short-lived lock:

```javascript
const lockKey = `lock:${docId}`;
const lock = await KV.get(lockKey);
if (!lock) {
  await KV.put(lockKey, 'locked', { expirationTtl: 30 });
  try {
    const value = await fetchLLM(docId);
    await CACHE.put(docId, value);
    return value;
  } finally {
    await KV.delete(lockKey);
  }
}
// Another request holds the lock; wait briefly and re-read the cache.
```

**5. Cost attribution per customer.** Write usage records keyed by customer ID so you can bill on actual token consumption. Shard the table only if a single table becomes a bottleneck; premature sharding adds complexity for no benefit at small scale.

**6. Warm-up on deploy.** Run a short script after each deploy that hits the health endpoint in each region. The cost is trivial; the benefit is zero cold starts during the first minutes after a release.

**7. Canary deployments.** Route a small percentage of traffic to a new version and roll back automatically if error rate or latency degrades. This is standard practice and does not require special infrastructure beyond what the edge runtime or function platform already provides.

## Decision checklist

Before adding any component, answer these questions:

1. Does this component scale to zero when traffic is zero? If not, can it be replaced by one that does?
2. Is there a cache key that would let this request avoid the model entirely? If yes, implement the cache first.
3. Can similar requests be batched without violating the latency budget? If yes, batch.
4. What is the measured p95 latency and cost per request today? If you cannot answer, instrument before optimizing.
5. Does this component have a documented failure mode and a retry policy? If not, it will become the incident.
6. Is the cost of this component proportional to usage, or is it a fixed monthly fee? Fixed fees are the enemy of a $150 ceiling.

## Quick reference

| Concern | Component category | Typical configuration | Notes |
|---|---|---|---|
| Edge routing & auth | Edge runtime | Pay-per-request plan | Token validation, rate limiting, caching |
| Cache | Managed Redis | Small instance, sized to measured working set | Key on content + template + model version |
| Prompt templating | Edge KV store | Small storage tier | Version the template; include hash in cache key |
| Background queue | Managed queue | Standard, with dead-letter queue | Cap retries; alert on 429 rate |
| Compute | Serverless function | ARM64, modest memory | 200 ms average duration is a reasonable target |
| Analytics | Columnar warehouse | On-demand queries | Flush usage records in batches |
| Model API | Managed model endpoint | Batch up to N documents | Price depends on provider and model |
| Monitoring | Edge logs + cloud metrics | Free tier | Alert on hit rate, p95 latency, error rate |

## Frequently asked questions

**How should GDPR and data residency be handled for EU users?**

Use regional services in the EU for the edge runtime, cache, and analytics store, and confirm that the model provider offers an EU endpoint. Store customer data in EU-only buckets and encrypt at rest with region-scoped keys. The extra region adds a small fixed cost, but it remains within a modest budget.

**What happens if the model API rate-limits the application?**

First, apply exponential backoff with jitter in the handler. If retries are exhausted, return a cached result when available, otherwise a `503` with `Retry-After`. Log the incident and alert. Most providers apply a short cooldown; waiting and retrying usually resolves it without user-visible impact.

**When do open-weight models become cheaper than managed APIs?**

The break-even depends on the GPU rental price, the model's throughput, and the managed API's per-token price — all of which change. The correct approach is to compute cost-per-thousand-tokens for both options using current published prices and measured throughput, then compare. Below the break-even, the managed API wins on price and operational overhead; above it, self-hosting may win, but only if the team has the capacity to operate it.

**How should customers be billed based on tokens used?**

Meter tokens per request in the handler, write usage records keyed by customer ID to the analytics store, and run a daily job that sums tokens per customer. Generate invoices from that table. At small scale, the query cost is negligible.

## One thing to do in the next 30 minutes

Open the cost dashboard for the current infrastructure (cloud billing console, edge analytics, or warehouse query history). Filter to the last seven days and list the top five line items. For each, write down whether it scales to zero and whether a cache or batch could eliminate it. Pick the single largest line item that does not scale to zero, and add one instrumentation counter — `cache_hit`, `cache_miss`, or `model_call` — to the code path that touches it. You cannot optimize a cost you are not measuring.
