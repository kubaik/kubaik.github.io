# AI unit costs: the metric both teams fight over

Most AI cost discussions start with a token price and end with an argument. The token price is knowable; the argument is about what a "unit" of AI work actually is. Product teams count calls, finance teams count compute-seconds, and neither number survives contact with the other's spreadsheet.

## Where per-call cost estimates break down

Documentation for AI frameworks covers tokens, embeddings and vector search. It rarely covers the economics that make finance teams reach for a calculator: who pays when an embedding call goes from 120 ms to 470 ms because a shared GPU queue is saturated?

The issue is not the AI feature. It is the gap between the unit economics visible in a notebook and the unit economics defensible in a budget meeting.

A typical failure mode: a prototype costs a fraction of a cent per call, and a product manager assumes that ratio holds at 10x traffic. It often does not, for three structural reasons.

**Pricing granularity differs between components.** Most SaaS pricing is request-based. AI pricing is frequently compute-based. A single user action such as "summarize" can trigger three sequential calls: embed the input, retrieve context, generate output. Each has a different billing clock. A retrieval step on CPU and a generation step on GPU are metered separately, so a per-call average hides the split.

**Latency and cost are coupled.** A timeout at 500 ms instead of 120 ms does not just fail a request; it consumes billed time. On a serverless platform, that extra time is billed at the configured memory size, not the memory actually used.

**Cold starts are billed.** A cold start is not only slow, it is metered. The cost is small per invocation but scales linearly with invocation count.

None of these are visible in a notebook, because a notebook runs one request at a time against a warm process.

## Define the unit as a useful outcome, not a technical event

The fix is to stop arguing about "cost per call" and tie cost to a measurable outcome. Instead of cost per token or cost per API call, define **cost per useful response**, where "useful" is a product signal, not an AI signal.

In practice this means a telemetry layer that records the downstream user action alongside the compute cost: did the user copy the output, open the next screen, or abandon?

Worked example with stated assumptions (all figures illustrative):

- A summarization call consumes 1,000 embedding tokens and 120 generation tokens.
- Assume an embedding rate of $0.00072 per 1,000 tokens and a generation rate of $0.0018 per 1,000 tokens.
- Compute cost per call = (1,000 / 1,000 × $0.00072) + (120 / 1,000 × $0.0018) = $0.00072 + $0.000216 = $0.000936.
- Add a vector search step of 85 ms at $0.00008 per CPU-second: 0.085 × $0.00008 = $0.0000068.
- Total per call ≈ $0.000943.

Now add the outcome. Suppose 32% of summaries lead to a save. Cost per saved summary = $0.000943 / 0.32 ≈ $0.00295. If traffic grows 8x and the save rate stays flat, cost per saved summary stays flat too — but total spend grows 8x. That is the number finance can plan against, and the number product can move by improving the save rate rather than by shaving tokens.

The disagreement usually dissolves once both teams look at the same denominator.

## Instrumentation: two layers, one join

Implementing cost per outcome requires two layers and a reliable join between them.

**Request-level telemetry.** Attach the total compute cost to the request context so each response carries its own cost tag. OpenTelemetry spans are a reasonable carrier: emit a numeric attribute such as `ai.response.cost.total` as a double, plus component attributes for embedding, retrieval, generation and cache.

**Outcome-level mapping.** Join the cost tag to the downstream user event. In a web app this is an event POST carrying an event type and the matching request ID. In a mobile app it is a batched analytics event sent to whatever product analytics backend the team already runs.

The hard part is **latency alignment**. If the user saves a summary 23 seconds after the response, but the cost tag is emitted 0.4 seconds after the response, the join can miss if the event store is queried too early. Two common mitigations:

- Emit the cost tag twice: once synchronously with the response, once asynchronously via a sidecar that re-attaches the tag if the outcome arrives late.
- Widen the join window and reconcile nightly, accepting a small undercount in the real-time view.

The first approach costs extra write volume; the second costs accuracy in the live dashboard. Pick based on whether finance is reading the dashboard daily or monthly.

**Cache behaviour belongs in the formula.** If embeddings are cached, the hit ratio directly moves cost per outcome. A high hit ratio lowers compute cost; an eviction storm raises it sharply for the duration of the storm. Any cost model that omits hit ratio will produce a number that swings without explanation.

## A composable cost model

Rather than one monolithic cost per call, break the unit into components:

- `cost_per_token_embedding`
- `cost_per_token_generation`
- `cost_per_vector_search`
- `cost_per_cache_hit`
- `cost_per_cache_miss`
- `cost_per_user_outcome`

Product and finance then agree on which combination matters for a given feature. For a chatbot, the natural unit may be cost per conversation turn: one embedding, one retrieval, one generation. For document Q&A, it may be cost per page processed: embeddings and vector search, no generation.

Two components deserve special attention.

**Latency as a probability discount.** Finance may want a latency penalty; product may resist because latency spikes correlate with drop-off, not with spend. A cleaner framing: if p99 latency exceeds a threshold, the probability of a positive outcome falls, so the *effective* cost per outcome rises even though the compute cost did not. Expressing latency as an outcome multiplier keeps the argument on measurable ground.

**Currency.** If infrastructure is billed in USD but users are in other markets, cost per outcome must be stored in more than one currency. Store `cost_usd`, `cost_local`, and `exchange_rate_timestamp`. Finance gets a single reporting currency; product gets local pricing; the timestamp makes reconciliation possible when a rate moves.

## Reference implementation

A minimal stack: Python 3.12, FastAPI, OpenTelemetry, and Redis for caching embeddings.

### 1. Define the cost model

```python
# cost_model.py
from dataclasses import dataclass


@dataclass
class CostPerCall:
    embedding_tokens: int
    generation_tokens: int
    vector_search_ms: int
    cache_hit: bool
    user_outcome: str | None = None

    # Rates are parameters, not constants: pass them in from config so the
    # model can be re-priced without a code change.
    embedding_rate_per_1k: float = 0.00072
    generation_rate_per_1k: float = 0.0018
    cpu_second_rate: float = 0.00008
    cache_hit_rate: float = 0.000012
    cache_miss_rate: float = 0.00019

    def compute(self) -> float:
        embedding_cost = (self.embedding_tokens / 1000) * self.embedding_rate_per_1k
        generation_cost = (self.generation_tokens / 1000) * self.generation_rate_per_1k
        vector_cost = (self.vector_search_ms / 1000) * self.cpu_second_rate
        cache_cost = self.cache_hit_rate if self.cache_hit else self.cache_miss_rate

        total = embedding_cost + generation_cost + vector_cost + cache_cost
        return round(total, 6)
```

Note the correction from the naive version: the cache cost is a flat per-call charge, not a percentage of the embedding cost, and the rates are constructor parameters so they can be changed without editing the class.

### 2. Add OpenTelemetry instrumentation

```python
# main.py
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

import cost_model

app = FastAPI()

provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces", timeout=3)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

tracer = trace.get_tracer(__name__)


@app.post("/summarize")
async def summarize(text: str, request: Request):
    with tracer.start_as_current_span("summarize") as span:
        embedding_tokens = len(text) // 5
        vector_search_ms = 85
        generation_tokens = 120
        cache_hit = len(text) > 100

        cost = cost_model.CostPerCall(
            embedding_tokens=embedding_tokens,
            generation_tokens=generation_tokens,
            vector_search_ms=vector_search_ms,
            cache_hit=cache_hit,
        )
        computed_cost = cost.compute()

        span.set_attribute("ai.response.cost.total", computed_cost)
        span.set_attribute("ai.response.cost.per_token_embedding", cost.embedding_rate_per_1k)
        span.set_attribute("ai.response.cost.per_token_generation", cost.generation_rate_per_1k)
        span.set_attribute("ai.response.outcome", "none")
        span.set_attribute("ai.request.id", request.headers.get("x-request-id", "unknown"))

        return {"summary": f"Summary of {text[:50]}...", "cost_usd": computed_cost}
```

The `x-request-id` header is what makes the later join possible. Without a stable request identifier on both the span and the outcome event, no amount of instrumentation will reconcile.

### 3. Async outcome attachment

```python
# outcome_sidecar.py
import asyncio
import time

import aiohttp


async def attach_outcome(request_id: str, delay_seconds: float = 15.0) -> None:
    """Wait for a possible user outcome, then emit it with the request id."""
    await asyncio.sleep(delay_seconds)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://analytics.local/events",
            json={
                "event_type": "summary_saved",
                "request_id": request_id,
                "timestamp": time.time(),
            },
        ) as resp:
            resp.raise_for_status()
```

The sidecar does not re-open the span. It emits a separate event keyed by `request_id`; the warehouse performs the join. Attempting to mutate an already-exported span is a common mistake and does not work with batch exporters.

### 4. Redis cache wrapper

```python
# cache.py
import hashlib
import json

import redis.asyncio as redis

r = redis.Redis(host="redis.local", port=6379, db=0, decode_responses=True)
TTL_SECONDS = 300


async def get_embedding(text: str) -> tuple[dict, bool]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_key = f"emb:{digest}"

    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached), True

    embedding = {"vector": [0.1] * 384, "tokens": len(text) // 5}
    await r.setex(cache_key, TTL_SECONDS, json.dumps(embedding))
    return embedding, False
```

Two corrections from the naive version: the cache key uses SHA-256 rather than Python's built-in `hash`, which is salted per process and therefore not stable across workers; and the return type is annotated as a tuple of `(value, hit)`.

### 5. Cost aggregation view

```sql
-- cost_view.sql
SELECT
    DATE(ts) AS day,
    COUNT(*) AS calls,
    SUM(cost_total) AS total_cost_usd,
    AVG(cost_total) AS avg_cost_per_call,
    SUM(CASE WHEN outcome = 'summary_saved' THEN 1 ELSE 0 END) AS saved_count,
    SUM(cost_total)
        / NULLIF(SUM(CASE WHEN outcome = 'summary_saved' THEN 1 ELSE 0 END), 0)
        AS cost_per_saved_summary
FROM ai_request_costs
WHERE service = 'summarizer'
GROUP BY DATE(ts);
```

Note that the columns are ordinary SQL identifiers. Span attributes are not columns until an ETL step flattens them; the view above assumes that step exists.

## How to measure this in your own system

Do not copy benchmark numbers from an article, including this one. Measure your own. The procedure:

1. **Instrument one endpoint.** Add the cost attribute and a request ID to a single high-traffic route.
2. **Log the raw components**, not just the total: embedding tokens, generation tokens, retrieval milliseconds, cache hit or miss.
3. **Emit one outcome event** tied to the same request ID.
4. **Run for a week** before drawing conclusions. Day-of-week effects on cache hit ratio are large.
5. **Compute the ratio**: total cost divided by count of positive outcomes, grouped by day.
6. **Compare against a control**: run the same feature with caching disabled for a subset of traffic and diff the two cost-per-outcome series. This is the only way to attribute savings to a specific change rather than to traffic mix.

What to instrument specifically: the four cost components above, plus cache hit ratio as a first-class metric. Cache hit ratio is the single most volatile input to cost per outcome, and it is invisible unless you chart it next to cost.

## Failure modes to plan for

### Double counting across services

A summarization system split into an embedding service and a generation service will emit two cost tags per logical request. If the generation step re-embeds the summary for a follow-up, the same tokens are counted twice.

**Symptom:** product and finance agree on the per-call total, but a bottom-up cost model from cloud billing does not match it.

**Fix:** add an explicit `cost_overlap` field with an `overlap_type` tag, and subtract it in the aggregation view. Make the overlap auditable rather than silently netting it out.

### Latency tax on outcomes

A vector search with p95 latency of 850 ms may look acceptable in isolation. If a measurable share of users abandon when the first response exceeds one second, the effective cost per outcome rises even though compute cost is unchanged.

**Fix:** treat latency as a multiplier on the outcome rate, not as an additive cost. Recompute cost per outcome using the observed outcome rate at each latency band.

### Cache invalidation storms

A short TTL on a popular key plus a burst of new queries can collapse the hit ratio for minutes at a time. During that window, cost per call can rise several-fold.

**Fix:** serve stale values for a short grace period while recomputing in the background (stale-while-revalidate). The stale hit is cheap; the miss is expensive. The blended cost during a storm is far below the full-miss cost.

A related trap is scheduled invalidation. A cron job that invalidates a large key set on a fixed interval creates a predictable stampede. Prefer event-driven invalidation with jitter, or a queue that spreads recomputation over time.

### Async outcome race conditions

If outcomes arrive after the trace has been exported, the join fails and cost per outcome is understated. The understatement is silent, which is worse than a visible error.

**Fix:** emit the cost record to a durable store keyed by request ID, and perform the join in the warehouse rather than in the tracing backend. Duplicate emission is acceptable; missing emission is not.

### Billing granularity mismatch

Serverless platforms bill at the configured memory size, not actual usage. A function configured for far more memory than it uses pays for the difference on every invocation.

**Fix:** right-size memory using observed peak usage, and re-check after dependency upgrades. Model weights and runtime versions change the memory profile.

## When this approach is the wrong choice

The unit cost layer adds real complexity. It is the wrong choice in several situations.

**Low-value prototypes.** Below roughly $50 per month in compute, the instrumentation cost — storage, bandwidth, and developer time — outweighs the benefit. Revisit when the feature has a monetization path.

**Deterministic batch pipelines.** A nightly job with a fixed cost and fixed duration does not need per-item cost tags. The only variable is job duration, which cloud billing already reports.

**Regulated environments with audit requirements.** If the cost model must be externally audited, OpenTelemetry attributes are not sufficient evidence. Such environments need an append-only ledger with signed entries, which is a different system with a different cost profile.

**Unmeasurable outcomes.** If there is no product event that reliably indicates a useful response, cost per outcome cannot be computed. In that case, fall back to cost per call and be explicit that it is a proxy.

## A decision checklist

Before building the unit cost layer, answer these questions:

- Is there a product event that reliably indicates a useful response? If not, stop here.
- Is monthly compute spend above the threshold where instrumentation pays for itself?
- Does the feature have more than one billing clock (CPU plus GPU, or multiple services)?
- Is the cache hit ratio variable enough to move the number materially?
- Do outcomes arrive within the join window, or is a sidecar needed?
- Is the reporting currency different from the billing currency?
- Who owns the definition of "useful outcome," and has finance agreed to it in writing?

If the answer to the first question is no, the rest do not matter.

## The one thing to do in the next 30 minutes

Pick your highest-traffic AI endpoint. Add a single numeric span attribute for its total compute cost and a stable request ID header, then emit one outcome event keyed by that same request ID. Deploy it. You will not have a cost-per-outcome number today, but in a week you will have the only input that ends the argument: a shared denominator both teams can see.
