# Which AI wrappers made money in 2026

Most AI wrapper products do not fail because the underlying model is weak. They fail because the wrapper cannot bound latency spikes, cannot predict its own bill, and cannot explain what it adds beyond the model API. This article covers how to evaluate a wrapper business, the failure modes that show up under load, and the concrete engineering patterns that keep a wrapper alive.

## What "wrapper" actually means here

A wrapper is any product layer between a raw model API and an end user. That includes:

- A chat UI or copilot embedded in an existing SaaS product.
- A retrieval pipeline that grounds answers in a customer's documents.
- A safety or policy layer that filters inputs and outputs.
- A routing layer that picks a provider based on cost or latency.
- A testing harness for prompts and evaluation.

These are different businesses with different failure modes. The mistake that kills most of them is combining several into one SDK before any single one is solid.

## A rubric for evaluating a wrapper business

The rubric below is a decision aid, not a benchmark. The thresholds are illustrative starting points; teams should set their own based on their market and stage.

| Axis | What it measures | Illustrative healthy range | Red flag |
|---|---|---|---|
| Revenue per employee | Annual recurring revenue divided by headcount | Above roughly $120k | Below $80k |
| Warm-up time | Days from first paid user to stable p95 latency under expected load | Under 21 days | Over 30 days |
| Infra cost ratio | Monthly infrastructure spend divided by monthly revenue | Under 35% | Over 45% |
| Churn delta | Month-6 gross revenue churn minus month-1 | Under 15 points | Over 25 points |
| Feature gulf | Number of wrapper features the base model does not provide | 1–3 | 4 or more |

The arithmetic is straightforward. If monthly infrastructure spend is $30k and monthly revenue is $100k, the infra cost ratio is 0.30, or 30%. If month-1 gross churn is 4% and month-6 gross churn is 19%, the churn delta is 15 points.

None of these numbers should be copied from an article. They should be measured on your own product with your own instrumentation.

## How to measure each axis without fooling yourself

**Revenue per employee.** Pull ARR from your billing system and headcount from your HR system on the same date. Do not annualize a single strong month. Use trailing twelve months or a stable run rate.

**Warm-up time.** Instrument p50, p95, and p99 latency per request path. Define "stable" as p95 staying within a target band (for example, under 800 ms) for seven consecutive days at your expected peak QPS. Record the date the first paying customer signed and the date the band was first held. The difference is your warm-up time.

**Infra cost ratio.** Sum model API spend, vector database, cache, compute, and egress for a calendar month. Divide by that month's recognized revenue. Track it weekly, not monthly, so a spike does not surprise you at close.

**Churn delta.** Define gross revenue churn as revenue lost from cancellations and downgrades in a month, divided by revenue at the start of the month. Compare month 1 to month 6 for the same cohort. A rising delta usually means the product is not sticky beyond the initial novelty.

**Feature gulf.** List every feature your wrapper exposes that the base model API does not. If the list is longer than three, you are probably building a platform before you have a product.

## The failure mode that shows up most often

The most common failure is not model quality. It is unbounded latency during retrieval or tool-call storms. A typical pattern:

1. A user pastes a long document or asks a question that triggers many retrieval calls.
2. The vector store or cache starts evicting entries under memory pressure.
3. Cache hit rate collapses, so every request hits the model and the vector store.
4. p95 latency climbs well past the target band.
5. Users notice the slowdown and churn before the team can ship a fix.

The specific cause is often a cache eviction policy that does not match the access pattern. For example, a Redis cluster using `allkeys-lru` will evict frequently used entries when memory fills, which is exactly the wrong behavior for a prompt cache that expects a stable hot set. A `volatile-lru` or `noeviction` policy with a separate eviction path for cold data is usually a better fit, but the right choice depends on your workload.

The point is not the specific policy. The point is that a single misconfigured component can produce a latency spike that looks like a model problem and is actually an infrastructure problem.

## When to add a cache, and how to size it

A prompt cache is worth adding when the same or similar prompts recur. It is not worth adding when every request is unique.

A simple sizing exercise:

- Suppose you serve 10 requests per second at peak.
- Suppose 30% of those requests share a cached prefix.
- That is 3 requests per second served from cache.
- If each request costs $0.002 in model tokens, you save $0.006 per second, or about $15,500 per month at that rate.

This is arithmetic on stated assumptions, not a measured result. Substitute your own numbers before making a decision.

A minimal cache wrapper in Node.js:

```javascript
import { createClient } from 'redis';

const client = createClient({ url: process.env.REDIS_URL });
await client.connect();

function cacheKey(prompt, model) {
  return `prompt:${model}:${Buffer.from(prompt).toString('base64')}`;
}

async function cachedCompletion(prompt, model, callModel) {
  const key = cacheKey(prompt, model);
  const hit = await client.get(key);
  if (hit) {
    return JSON.parse(hit);
  }
  const result = await callModel(prompt, model);
  // TTL in seconds; tune to your data freshness needs
  await client.set(key, JSON.stringify(result), { EX: 300 });
  return result;
}
```

Two things matter in production: the TTL should match how often your source data changes, and the eviction policy should protect the hot set. Measure hit rate before and after any change.

## Bounding latency with a circuit breaker

A cache reduces average cost. A circuit breaker bounds worst-case latency. The pattern is simple: if a downstream call exceeds a threshold, stop calling it and return a fallback.

```javascript
class CircuitBreaker {
  constructor({ thresholdMs, cooldownMs }) {
    this.thresholdMs = thresholdMs;
    this.cooldownMs = cooldownMs;
    this.openUntil = 0;
  }

  async call(fn) {
    const now = Date.now();
    if (now < this.openUntil) {
      throw new Error('circuit_open');
    }
    const start = now;
    try {
      const result = await fn();
      return result;
    } finally {
      const elapsed = Date.now() - start;
      if (elapsed > this.thresholdMs) {
        this.openUntil = Date.now() + this.cooldownMs;
      }
    }
  }
}
```

This is deliberately small. In production you would add metrics, a half-open state, and per-dependency breakers. The key idea is that a slow dependency should degrade the experience, not take down the product.

## Retrieval: chunking and the cost of getting it wrong

Retrieval quality drives churn more than latency in enterprise products. A chunking policy that is too large wastes tokens and dilutes relevance. One that is too small loses context.

A common starting point is a sliding window of roughly 128 tokens with 25 tokens of overlap, stored in a vector database such as pgvector. The exact numbers depend on your documents and your embedding model. What matters is that you measure retrieval quality, not just latency.

A minimal retrieval call:

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("dbname=app")
register_vector(conn)

def search(query_embedding, k=5):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, embedding <=> %s AS distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        return cur.fetchall()
```

Measure recall on a labeled set of questions before and after changing chunk size, overlap, or embedding model. A change that improves latency but drops recall is usually a net loss for enterprise buyers.

## Safety layers: what they cost and what they buy

A safety layer sits between the model output and the user. It can redact PII, block toxic content, or enforce policy. The cost is added latency and added complexity. The benefit is that regulated buyers will not sign without it.

A minimal policy file:

```yaml
policies:
  - id: pii-redaction
    type: filter
    path: "$.text"
    pattern: "\b(?:\d[ -]*?){13,16}\b"
    replacement: "[REDACTED]"
  - id: toxicity-score
    type: score
    threshold: 0.75
    action: block
```

Two practical notes. First, regex-based PII detection is a first pass, not a guarantee; it will miss obfuscated values and it will occasionally redact legitimate numbers. Second, a toxicity threshold is a product decision, not an engineering one. Set it with your legal and support teams, and log every block so you can review false positives.

## Routing across providers: the trap

Routing between model providers based on cost or latency sounds attractive. In practice it introduces a quality variance problem: users notice when the same question gets a different-quality answer depending on which provider was cheaper that minute.

If you route, do it with a fixed quality floor. Route only between providers that pass the same evaluation set, and log which provider served each request so you can correlate quality complaints with routing decisions.

```javascript
async function routeWithFloor(prompt, providers, evaluate) {
  for (const provider of providers) {
    const result = await provider.complete(prompt);
    if (evaluate(result) >= provider.qualityFloor) {
      return { result, provider: provider.name };
    }
  }
  throw new Error('no_provider_met_quality_floor');
}
```

This is slower than picking the cheapest provider. It is also the only version that does not silently degrade quality.

## A decision checklist

Before shipping a wrapper, answer these questions in writing:

1. What single problem does this wrapper solve that the model API does not?
2. What is the p95 latency target, and what happens when a dependency exceeds it?
3. What is the infra cost ratio today, and what is the threshold that triggers a redesign?
4. What is the cache hit rate, and how is the hot set protected from eviction?
5. What is the retrieval recall on a labeled set, and how often is it re-measured?
6. What does the safety layer block, and who reviews false positives?
7. If routing is used, what is the quality floor and how is it evaluated?
8. What is the churn delta between month 1 and month 6 for the same cohort?

If any answer is "we will figure that out later," that is the item most likely to cause the next incident.

## FAQ

**Why do so many AI wrapper products fail?**
Most fail because they try to abstract models, prompts, retrieval, safety, and UI into one SDK before any single layer is solid. The survivors usually pick one layer and do it well.

**What is the biggest hidden cost?**
Warm-up time. The gap between first paid user and stable p95 latency is where most early churn happens, because users experience the product at its worst.

**Which layer is safest for a regulated industry?**
A dedicated safety or policy layer, because regulated buyers require demonstrable controls. It adds latency, so it should sit behind a circuit breaker and be measured like any other dependency.

**How do I know if my wrapper will survive?**
Track the five axes in the rubric. If two or more are in the red-flag range, stop adding features and fix them before shipping anything new.

## Take action in the next 30 minutes

Instrument p95 latency on your primary request path and log it to a time-series store. Run a single load test at your expected peak QPS and record the p95. If it exceeds your target band, you have found your first real problem, and you found it before your users did.
