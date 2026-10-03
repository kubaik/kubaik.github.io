# Buy vs build AI: 3 hard rules for 2026

## The real question behind "build vs buy"

Most teams frame the decision as a binary, then discover the cost sits in the 30–40% of work nobody scoped: token budgeting, retry storms, cache eviction, and the behaviour of mobile networks that drop to a few kilobytes per second mid-request. The algorithmic part of an AI feature is usually the cheap part. The operational envelope around it is where budgets and on-call rotations go.

Three failure patterns recur often enough to treat as archetypes:

- A service passes local tests but dies on first deploy because retry logic fires every 50 ms for 30 seconds against a throttled endpoint, burning a large token budget on a single failed call.
- A self-hosted embedding model shows acceptable p99 latency until the cache layer evicts keys under memory pressure, at which point every miss triggers a cold rebuild of a 512- or 768-dimension vector.
- A messaging bot that looks trivial accumulates a webhook queue backlog whenever the payment provider's rate limit is lower than documented, and each retry re-sends the full payload.

None of these are model-quality problems. They are infrastructure and edge-case costs that only appear under real load and real network conditions.

## Four axes for scoring each component

Score every candidate component along the same four axes. The point is not to produce a single number but to expose which axis dominates, because that determines build-versus-buy.

- **Latency floor.** The 95th-percentile end-to-end time from user action to first useful token, measured on the network conditions your users actually have, not on a datacentre link.
- **Token economics.** Cost per 1,000 tokens at current provider prices, plus amortised infrastructure cost for self-hosted options, computed over a stated load assumption.
- **Operational blast radius.** How many pages a single outage generates per quarter, and whether recovery needs a human.
- **Localisation and integration debt.** Extra code required for payment rails, message formats, low-bandwidth retries, and locale handling before the feature even reaches the model.

Two filters are worth applying early. Anything requiring a specific GPU class should be deprioritised unless the team already runs that hardware continuously, because idle GPU cost dominates the arithmetic. Anything requiring a cluster orchestrator should be deprioritised unless a managed option hides the cluster entirely.

### How to measure instead of guess

Vendor benchmarks and blog-post latency numbers rarely transfer. Instrument your own path:

- **Latency:** record timestamps at request receipt, first byte from the provider, and final token. Emit a histogram, not an average. Compare p50 and p95 separately, because cold starts and retries show up almost entirely in the tail.
- **Token spend:** log prompt tokens and completion tokens per request with a request ID and a feature tag. Aggregate daily. This is the only way to catch retry storms, since they inflate token counts without inflating successful-request counts.
- **Cache effectiveness:** instrument hit rate, miss rate, and rebuild time per miss. A cache with a 90% hit rate and a 1.2-second rebuild still produces bad p99 if misses cluster.
- **Network conditions:** replay recorded mobile traces through your stack, or at minimum throttle a staging environment to a realistic profile. Testing on office wifi hides the entire class of failures this article is about.

## Component-by-component analysis

### 1. Embedding cache and vector search

**What it does:** stores embeddings of repeated queries so semantic search does not call the model for every request.

**Why it usually wins:** once warm, cache hits replace a full model round trip with a local lookup. The latency improvement is large and the cost improvement is proportional to your repeat-query rate. There is no GPU budget and no rate-limit race.

**Failure mode:** the cache is cold for a period after every deploy and after every eviction event. A cold rebuild of a single 768-dimension vector can take seconds on constrained hardware. Teams commonly set a one-hour expiry and forget that the first hour after deploy is the worst hour.

**Build if:** your product has heavy repeat queries (marketplaces, tutoring, support tooling) and your token spend is dominated by search or retrieval rather than generation.

### 2. Prompt templating and dynamic few-shot retrieval

**What it does:** assembles context at runtime from a fragment library keyed by user segment and locale.

**Why it usually wins:** a single template layer plus a small module can serve many locales without maintaining a separate model per language.

**Failure mode:** the fragment library grows past a few thousand entries and lookup, serialisation, and placeholder filling start contributing measurable latency and token overhead. A naive dictionary lookup is fine at small scale and not fine at large scale.

**Build if:** prompts change per tenant or locale but the underlying model is fixed. Otherwise buy a templating layer or use the provider's built-in prompt management.

### 3. Webhook parsing and validation

**What it does:** converts raw business-messaging and payment callbacks into structured events.

**Why it usually wins:** a focused parser replaces a large body of hand-written regular expressions, and the correctness gain is usually larger than the latency gain.

**Failure mode:** the upstream API returns a 503 with a `Retry-After` header during load spikes, and naive exponential backoff re-sends the full payload on every attempt. Retry cost scales with payload size, so a large payload plus an aggressive backoff schedule is the worst combination.

**Build if:** you operate in a market where the payment or messaging rail is central to the product and no library handles it well.

### 4. LLM fallbacks and retry orchestration

**What it does:** routes to a secondary model when the primary throttles or errors, under a bounded retry budget.

**Why it usually wins:** a small queue worker keeps retry behaviour inside a predictable envelope and prevents unbounded fan-out.

**Failure mode:** the queue's default concurrency limiter still permits enough concurrent retries to trigger a second round of throttling on the fallback provider, producing a loop that consumes tokens without producing responses.

**Build if:** you must guarantee a response within a fixed time budget and the provider's own retry semantics are insufficient.

### 5. Localisation prompts and tone adaptation

**What it does:** rewrites output into a preferred tone or register without fine-tuning.

**Why it usually wins:** a template plus a short system prompt replaces per-market fine-tunes and keeps the model size fixed.

**Failure mode:** tone rules grow past a few hundred tokens and start crowding the context window, pushing first-token latency up for subsequent requests.

**Build if:** brand voice varies meaningfully across markets and a single system prompt cannot cover it.

### 6. Self-hosted embedding model

**What it does:** runs an open-weight embedding model on your own hardware.

**Why it can win:** at high, steady utilisation, self-hosting undercuts hosted embedding APIs.

**Failure mode:** cold-start latency on ARM instances can be several seconds unless the model is kept warm with synthetic requests. On a slow mobile link, a cold start can exceed client-side timeouts.

**Build if:** you already run the hardware continuously and your query pattern is predictable. Otherwise buy.

### 7. Fine-tuned adapter for domain jargon

**What it does:** trains a small adapter over a base model to handle domain-specific vocabulary.

**Why it can win:** inference cost grows only marginally while exact-match accuracy on in-domain queries improves.

**Failure mode:** adapter artifacts are large relative to mobile app bundles, which complicates over-the-air updates on low-storage devices. You need delta updates and compression, which is real engineering work.

**Build if:** your domain vocabulary is not represented in general models and you have enough in-domain data to train on.

### 8. Real-time transcription

**What it does:** converts voice messages into text for downstream processing.

**Why it can win:** self-hosted transcription can be cheaper than cloud services once egress is included, at sufficient volume.

**Failure mode:** transcription latency on a slow link can be several times real time. Buffering 30 seconds of audio adds 30 seconds to the response, which violates any interactive SLA.

**Build if:** the use case tolerates asynchronous responses (voice notes, field reports) rather than requiring live conversation.

### 9. Self-hosted reranker

**What it does:** reranks top-k candidates to improve ordering without another generation call.

**Why it can win:** a CPU-hosted reranker is cheap and reduces downstream token consumption.

**Failure mode:** reranker quality degrades on mixed-script documents, requiring a pre-filter step that adds latency.

**Build if:** your documents are long, your corpus is stable, and you have measured a real ranking problem that a reranker solves.

### 10. Fully fine-tuned conversational model

**What it does:** trains a bespoke chat model to embody a specific voice.

**Why it can win:** at large scale, a small fine-tuned model can match a larger general model on your narrow task at lower per-token cost.

**Failure mode:** model weights are too large for mobile delivery, so you need CDN distribution and delta updates, and cache misses under poor network conditions produce visible failures.

**Build if:** you have a large training corpus, a strong voice constraint, and the distribution infrastructure to ship weights.

## Choosing between them: a decision table

| Component | Default choice | Dominant risk | Choose build when |
|---|---|---|---|
| Embedding cache + vector search | Build | Cold-start and eviction | Repeat-query rate is high |
| Prompt templating | Buy | Fragment library growth | Prompts vary per tenant |
| Webhook parsing | Build | Upstream 503 storms | Payment rail is core to product |
| Retry orchestration | Build | Retry amplification | Hard response-time SLA |
| Tone adaptation | Buy | Context-window bloat | Brand voice varies by market |
| Self-hosted embeddings | Buy | Cold-start latency | GPUs run continuously |
| Fine-tuned adapter | Buy | Artifact size vs OTA | Strong domain vocabulary gap |
| Transcription | Buy | Latency vs real time | Async use case only |
| Reranker | Build | Mixed-script degradation | Measured ranking problem |
| Full fine-tune | Buy | Weight distribution | Large corpus + CDN |

The one-line rule: if the component's integration and localisation debt exceeds the cost of running it locally, build it; otherwise buy the API and wrap it in a thin local cache or queue.

## Worked example: sizing an embedding cache

Take a hypothetical product with 100,000 daily active users, each issuing 5 search queries per day, for 500,000 queries daily. Assume 40% of queries are exact repeats of a query already seen that day, and that a hosted embedding call costs $0.02 per 1,000 tokens with an average of 200 tokens per query.

- Daily embedding tokens without cache: 500,000 × 200 = 100,000,000 tokens.
- Daily cost without cache: 100,000,000 / 1,000 × $0.02 = $2,000.
- Queries served from cache at 40%: 200,000.
- Daily cost with cache: 300,000 × 200 / 1,000 × $0.02 = $1,200.
- Daily saving: $800. Monthly saving: roughly $24,000.

Now the infrastructure side. A single mid-size ARM instance with an in-memory vector index can plausibly serve this query volume, at a cost that is a small fraction of the monthly saving. The arithmetic is illustrative and depends on your repeat rate, token counts, and provider pricing, but the structure is what matters: the decision turns on repeat-query rate and token price, both of which you can measure in a day.

If the repeat rate is 5% instead of 40%, the same calculation produces a monthly saving of roughly $3,000, and the build case weakens considerably. Measure before you build.

## Failure-mode analysis: cache eviction under pressure

The most common way an embedding cache stops helping is eviction. A cache configured with a least-recently-used policy will, under memory pressure, evict entries that are frequently accessed but not recently accessed. In a workload with a small hot set and a large cold tail, this produces a cache that appears healthy in hit-rate dashboards but degrades sharply during traffic spikes.

Symptoms to watch for:

- Hit rate looks acceptable on average but p99 latency spikes correlate with memory pressure.
- Rebuild time per miss is high enough that a burst of misses saturates the embedding service.
- The eviction policy is the default, and nobody has tuned reserved memory.

Mitigations:

- Switch to a frequency-based eviction policy if your store supports it, so frequently accessed entries survive.
- Reserve headroom so the process does not hit the system OOM killer, which produces a full cache loss rather than a partial one.
- Pre-warm the cache after deploy with a synthetic workload drawn from your real query distribution.
- Alert on miss rate, not just hit rate, and alert on rebuild time separately.

## Failure-mode analysis: retry amplification

Retry logic that is not token-aware turns a transient provider error into a budget event. The mechanism is simple: each retry re-sends the full prompt, so cost scales with (number of retries) × (prompt size) × (concurrent failures).

A bounded approach uses a per-user token bucket. When the bucket is empty, fail fast with a clear error rather than retrying. This converts an unbounded cost into a bounded one and makes the failure visible to the user instead of invisible in the billing dashboard.

```javascript
import { Queue, Worker } from 'bullmq';

const queue = new Queue('llm-fallback', { connection: redis });

const worker = new Worker('llm-fallback', async job => {
  const tokensUsed = job.data.tokens;
  if (!tokenBucket.tryConsume(tokensUsed)) {
    throw new Error('token_limit_exceeded');
  }
  return callFallbackModel(job.data);
}, { connection: redis });
```

The important property is not the specific library but that the retry decision consults a budget. Any queue implementation that supports a custom admission check will do.

For webhook handlers, where the upstream may send a `Retry-After` header, a fixed delay with a hard attempt cap is usually better than exponential backoff, because backoff schedules can outlast the upstream outage and re-send payloads long after they are useful.

```python
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(30))
async def confirm_payment(payload: dict) -> bool:
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(mpesa_webhook, json=payload)
        r.raise_for_status()
    return True
```

## What to avoid building

Some components look attractive to build and rarely pay off:

- **A general-purpose LLM abstraction layer.** The maintenance cost grows with every provider API change, and the abstraction rarely hides the differences that matter (streaming semantics, tool-call formats, error taxonomies).
- **An in-memory document store for a large corpus.** Keeping every embedding in RAM works on a laptop and fails on a small instance. If your corpus is large, use a store designed for it.
- **A synchronous web framework serving blocking model calls.** Blocking calls under a synchronous ORM queue behind each other and produce latency cliffs under concurrency. Offload to a queue or use an async stack end to end.
- **Fine-tuning an embedding model on a small corpus.** A small in-domain gain often comes with a regression on the general case, and the net effect for bilingual or mixed-domain users is negative.

## A decision checklist

Before committing to build, answer all of these:

1. Can you state the current p50 and p95 latency for this path, measured on realistic network conditions?
2. Can you state the current daily token spend for this path, broken down by feature?
3. What is the repeat rate, cache hit rate, or other metric that determines whether a local implementation helps?
4. What is the monthly infrastructure cost of the build option, including idle time?
5. What is the on-call cost of the build option, expressed as expected pages per quarter?
6. Who maintains the build option when the upstream API changes?
7. What is the rollback plan if the build option underperforms?

If you cannot answer questions 1 through 3, you are not ready to decide. Measure first.

## FAQ

**Does the choice change with model provider pricing?**
Yes. The build case strengthens when hosted token prices rise or when your repeat rate is high, and weakens when prices fall. Re-run the arithmetic quarterly rather than treating the decision as permanent.

**Is a cache always cheaper than calling the model?**
No. A cache adds an operational component, a cold-start period, and an eviction failure mode. It pays off when the repeat rate is high enough that the avoided calls exceed the infrastructure and maintenance cost. Compute the break-even repeat rate for your own numbers.

**How do I decide between a vector database and an in-process index?**
In-process indexes are simpler and faster for corpora that fit in memory. External vector stores are necessary when the corpus exceeds available memory or when you need independent scaling and persistence guarantees. Start in-process and move when you hit a limit you can measure.

**Should retries use exponential backoff?**
For interactive requests, a bounded fixed delay with a hard attempt cap is often better, because exponential schedules can outlast the outage and re-send payloads after they are stale. For background jobs where latency does not matter, exponential backoff with jitter is reasonable.

**When is fine-tuning worth it over prompt engineering?**
When you have enough in-domain data that the accuracy gain exceeds the regression on general inputs, and when you have the distribution infrastructure to ship the artifact. If your corpus is small, prompt templates usually capture most of the gain at a fraction of the cost.

## Do this in the next 30 minutes

Add per-request token logging to your primary AI endpoint. Log prompt tokens, completion tokens, feature tag, and request ID to a table you can query. Without this, every build-versus-buy decision is a guess, and retry storms remain invisible until the invoice arrives. Once the data exists, you can compute your repeat rate, your cache break-even point, and your real cost per feature — which is the only sound basis for deciding what to build.
