# 7 ways to measure prompt platform ROI in 2026

## The problem: prompts are a second control plane

When an application moves from a handful of model calls to thousands per day, the cost model changes shape. Lambda duration, request counts and CPU time no longer explain the bill, because the dominant cost is now tokens, and the dominant source of value is now developer time that is saved or lost depending on how prompts behave.

A common failure mode is a dashboard that tracks infrastructure health while the token spend grows unobserved. The old stack answers "is the service up?" The new layer has to answer three different questions:

1. Which prompts actually save developer time?
2. Which agent graphs leak money instead of saving it?
3. Is the new layer worth its complexity tax?

The tooling landscape splits into three categories: pure prompt logging, full agent graph tracing, and hybrid approaches that attempt both. The rest of this article covers how to evaluate each category, what to instrument, and how to decide.

## Evaluation criteria that survive contact with production

Before comparing tools, fix the criteria. Vague criteria produce vague vendor comparisons. Five that hold up:

- **Token-level cost attribution** — can a given prompt token be mapped to a specific line item on the cloud bill?
- **Latency impact** — what does the instrumentation layer add, measured on both warm and cold paths?
- **Developer time saved** — does the layer reduce the time to diagnose a prompt regression?
- **Hard-reversal cost** — how long does it take to remove the layer if it backfires?
- **Operational overhead** — does setup and maintenance fit the team's actual capacity?

Two of these deserve elaboration because they are usually measured badly.

**Latency impact.** Vendor claims of "negligible overhead" are meaningless without a stated measurement window and a stated baseline. A span processor that adds 3 ms on a warm invocation can add 100 ms or more on a cold start, because cold starts pay for module initialization and connection setup. Measure both:

- Warm: run the agent runner in a loop for a few minutes and compare p50 and p99 end-to-end latency with the instrumentation enabled and disabled.
- Cold: force cold starts (for example, by redeploying or by using a runtime that recreates the execution environment) and compare the first invocation's latency across both configurations.

If a tool cannot be toggled with a feature flag or environment variable, cold-start comparison becomes expensive, which is itself a signal.

**Hard-reversal cost.** A layer that takes a weekend to unwind will be kept long past its usefulness. Prefer instrumentation that can be disabled at runtime. A span processor can typically be gated by an environment variable; a forked SDK cannot.

## The approaches, ranked by how much they actually tell you

### 1. OpenTelemetry spans with prompt token attributes

**What it does.** A custom span processor reads the model response object emitted by the LLM SDK and writes token counts onto the span as attributes, for example `llm.prompt_tokens` and `llm.completion_tokens`. The spans flow through an OpenTelemetry Collector to whatever backend is already in use.

**Strength.** Cost attribution becomes trace-scoped. Because each span carries the token counts and the trace carries the request identity, the infrastructure bill can be split into prompt and non-prompt costs per request. The processor runs in-process, so there is no extra network hop.

**Weakness.** The processor depends on the SDK emitting a span with the expected attribute names. If the SDK is used raw, without instrumentation, the attributes are absent and the processor silently produces nothing. This is the most common way this approach fails: the spans exist, the dashboard looks healthy, and every token count is zero.

**Best for.** Teams that control the agent runtime and can patch or wrap the SDK once.

### 2. Self-hosted prompt logging backed by Redis

**What it does.** A logging runner writes every prompt, completion and metadata record to a Redis instance that the team operates, rather than to a vendor endpoint.

**Strength.** The schema is typically flat and queryable, and the storage cost is predictable. As an illustrative calculation: at roughly 150 bytes per prompt record, a workload of 10,000 prompts per day produces about 1.5 MB per day, or roughly 45 MB over a 30-day retention window. Adjust the per-record estimate to the actual payload size before relying on it.

**Weakness.** Self-hosted runners often expect specific Redis modules (for example, a search module) that managed offerings do not always enable in every region. Verify module availability before committing.

**Best for.** Teams that want a searchable prompt log without per-seat or per-trace SaaS pricing.

### 3. Agent graph tracing

**What it does.** Each agent invocation emits a span, and the whole graph emits a parent span. The graph can be visualized in a tracing backend, and the cost of a graph can be computed as the sum of its child spans.

**Strength.** This is the only approach that directly answers "which agent in the graph leaks money?" Without per-hop spans, a five-agent graph is a single opaque cost.

**Weakness.** Tracing every hop adds overhead, and the overhead is worst on cold starts, where each hop may pay initialization cost. A graph with several agents can see first-request latency well above the warm-path figure. Measure the cold path before adopting.

**Best for.** Graphs with more than a few agents, where the cost of tracing is smaller than the cost of not knowing which hop is expensive.

### 4. Prompt caching keyed by normalized prompt text

**What it does.** The runner hashes the prompt text and stores the response with a TTL. Before invoking the model, it checks the cache.

**Strength.** Cache hit ratio maps directly to token savings, and the mapping is arithmetic rather than estimated. If the cache serves a fraction *h* of requests, token spend on those requests drops to zero, so the saving is proportional to *h* times the cached-request token cost.

**Weakness.** Prompt drift breaks the cache. If an upstream service renames a field, or a timestamp is embedded in the prompt, the hash changes and the cache misses on every request. A cache that silently stops hitting looks identical to a cache that was never effective.

**Best for.** Prompts that are genuinely static, with normalization applied deliberately.

### 5. Cloud billing tags

**What it does.** Tag the compute resources that run agents (for example, with `llm:prompt_type` and `llm:agent_id`) and group the bill by tag.

**Strength.** No code changes. The tagging is declarative infrastructure configuration.

**Weakness.** Tags do not always propagate to detailed line items when one function invokes another, so attribution is coarse. It answers "which service costs more" but not "which prompt costs more."

**Best for.** Teams that want zero instrumentation and accept coarse attribution.

### 6. Human-scored prompt evaluation

**What it does.** Each run is logged and a reviewer attaches a score. The score is then correlated with downstream outcomes such as regression tickets.

**Strength.** A human score is the only signal that captures "was this output actually good," which no token metric captures.

**Weakness.** The scoring is manual. Without sustained labeling capacity the metric decays into noise, and a stale score is worse than no score because it looks authoritative.

**Best for.** Teams with a reviewer who can label on a regular cadence.

### 7. Metrics exporter for token counters

**What it does.** An exporter scrapes an endpoint on the agent runner and exposes counters such as prompt tokens, completion tokens and estimated cost.

**Strength.** Token growth rate becomes alertable, which catches runaway loops before the bill does.

**Weakness.** The exporter parses the SDK response object, so any change to that object's shape breaks it. Pin the SDK version and add a health check that alerts on parse errors rather than on absence of data.

**Best for.** Teams already running a metrics stack that want alerting on token growth.

## How to measure each criterion

This section replaces any benchmark table, because a benchmark from one environment does not transfer to another. Measure on the target system.

**Token-level cost attribution.** Instrument the runner to emit per-request token counts, then join those records to the billing export on a request identifier. The join key must be present in both datasets; if the billing export does not carry the identifier, attribution is impossible and the tool category is wrong for the use case. Compare the sum of attributed token cost against the total model spend for the same period. A large gap means unattributed traffic exists.

**Latency impact.** Run the same workload with instrumentation enabled and disabled. Compare p50 and p99 for warm runs, then force cold starts and compare first-invocation latency. Record the measurement window and the workload shape alongside the numbers, because both change the result.

**Developer time saved.** Time how long it takes to diagnose a deliberately introduced prompt regression — for example, a prompt that omits a required field — with and without the instrumentation. The difference is the per-incident saving. Multiply by the observed incident rate to get a monthly figure.

**Hard-reversal cost.** Disable the layer in a staging environment and note how long the system takes to return to a known-good state. If disabling requires a code change rather than a configuration change, treat the reversal cost as high.

**Operational overhead.** Track the hours spent maintaining the layer over a month. Include upgrade work when the SDK or backend changes.

## Joining traces to the cloud bill

The join is the part most teams underestimate. The procedure:

1. Export traces to a tracing backend that retains trace identifiers for at least as long as the billing period.
2. Query the backend for the trace identifiers in the period of interest, along with their token attributes.
3. Load the billing export and locate the field that carries the request identifier.
4. Join on that identifier and aggregate cost by the dimensions you care about, such as prompt type or agent.

If the billing export does not carry a request identifier, the join cannot be done from billing data alone. In that case, allocate cost by token share: compute each prompt type's share of total tokens, then apply that share to the total model spend for the period. This is an approximation, and it should be labelled as one.

## A worked example

Suppose an agent runner handles 10,000 prompts per day. Assume, for illustration only, that the average prompt is 1,000 tokens and the average completion is 300 tokens, and that input tokens cost $3 per million and output tokens cost $15 per million. These are illustrative unit prices, not current published rates.

- Input tokens per day: 10,000 × 1,000 = 10,000,000 tokens = 10 million tokens.
- Input cost per day: 10 × $3 = $30.
- Output tokens per day: 10,000 × 300 = 3,000,000 tokens = 3 million tokens.
- Output cost per day: 3 × $15 = $45.
- Total per day: $75. Per 30-day month: $2,250.

Now suppose a cache is added and, measured over a week, it serves 40% of requests. If cached requests are evenly distributed across prompt types, the model spend on those requests falls to zero, so the monthly model spend becomes $2,250 × 0.6 = $1,350. The saving is $900 per month, before accounting for cache storage and the engineering time to maintain normalization.

That last clause matters. If normalization requires two days of engineering per quarter and a regression every quarter, the saving may be smaller than it appears. The arithmetic above is the easy part; the maintenance cost is the part that is usually omitted.

## Failure modes to plan for

**Silent zero attribution.** The span processor runs, spans are emitted, but the attribute names do not match what the SDK produces, so every token count is zero. Detect it with an assertion: if the sum of attributed tokens for a period is zero while model spend is non-zero, alert.

**Cache that never hits.** Prompt drift changes the hash. Detect it by tracking hit ratio as a first-class metric and alerting when it falls below a threshold rather than when it reaches zero.

**Exporter breakage on SDK upgrade.** The response object shape changes and the exporter raises an error. Pin the SDK version and add a health check that alerts on parse failures.

**Sampling that destroys attribution.** If the tracing backend samples spans, token totals computed from traces will undercount. Either disable sampling for the token spans or scale the totals by the known sampling rate and label the result as an estimate.

**Tag propagation gaps.** Compute resources invoked by other compute resources may not inherit tags on detailed billing lines. Verify with a small test before relying on tag-based attribution.

## Decision checklist

Work through these in order.

1. Do you control the agent runtime and can you wrap or patch the SDK? If yes, start with OpenTelemetry spans carrying token attributes. If no, go to step 3.
2. Can you gate the instrumentation with a configuration flag? If no, reduce the blast radius before proceeding, because reversal will be expensive.
3. Do you already operate a Redis instance with the required modules? If yes, self-hosted prompt logging is the lowest marginal-cost option.
4. Does the agent graph have more than a few hops? If yes, graph tracing is likely worth its overhead; measure the cold path first.
5. Are the prompts genuinely static? If yes, caching is the cheapest token reduction available. If no, fix normalization before adding a cache.
6. Do you have sustained labeling capacity? If yes, human scoring adds a signal no metric provides. If no, skip it.
7. Do you need alerting on token growth rate? If yes, add a metrics exporter and pin the SDK version.
8. Is coarse attribution acceptable? If yes, billing tags require no code changes.

## Verifying an OpenTelemetry token attribute end to end

A minimal processor that copies token counts from a model span onto a normalized attribute:

```python
from opentelemetry.sdk.trace import SpanProcessor


class PromptTokenSpanProcessor(SpanProcessor):
    def on_end(self, span):
        if span.name == "generation":
            prompt_tokens = span.attributes.get("gen_ai.prompt_tokens")
            if prompt_tokens is not None:
                span.set_attribute("llm.prompt_tokens", prompt_tokens)
```

Register it on the tracer provider, then confirm the attribute appears on the exported span. The verification step is the important part: a processor that runs but writes nothing is indistinguishable from a processor that works, until someone looks at the data.

To verify, emit a single request, export the trace, and inspect the `generation` span for the `llm.prompt_tokens` attribute. If the attribute is missing, check the SDK's actual attribute names against the ones the processor reads.

## Normalizing prompts before hashing

Normalization is where most cache implementations quietly fail. A Lua script that lowercases, collapses whitespace and trims the ends:

```lua
local prompt = ARGV[1]
local normalized = string.lower(string.gsub(string.gsub(prompt, "%s+", " "), "^%s*(.-)%s*$", "%1"))
local key = "prompt:" .. normalized
return redis.call("HSET", key, "value", ARGV[2], "ttl", 300)
```

Two cautions. First, lowercasing changes semantics for prompts where case matters, so apply it only when it is safe. Second, the script should be unit tested against representative prompts, including ones with embedded identifiers, because those will never hit the cache and should be excluded from it rather than hashed.

## The short version

If the agent runtime is under direct control, start with OpenTelemetry spans carrying token attributes, gated by a configuration flag so the layer can be removed cheaply. If it is not, self-hosted prompt logging backed by an existing Redis instance is the next lowest-friction option. Add graph tracing when the graph is large enough that per-hop cost is unknown. Add caching when prompts are static and normalization is deliberate. Add human scoring only when labeling capacity exists. Add a metrics exporter when alerting on token growth matters more than attribution precision.

Whatever is chosen, measure the four things that matter — attribution coverage, warm and cold latency, time to diagnose a regression, and reversal cost — on the real system, and record the measurement window alongside the number.

## Do this in the next 30 minutes

Pick one prompt in a staging environment, log its token counts to a span attribute, export the trace, and confirm the attribute is present and non-zero. If it is zero or missing, the attribute name mismatch is the most likely cause, and finding it now costs minutes rather than a billing cycle.
