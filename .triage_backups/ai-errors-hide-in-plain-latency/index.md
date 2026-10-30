# AI errors hide in plain latency

Model-centric metrics describe the model. They do not describe what a user experiences. A request can complete with a perfectly reasonable token count and an unchanged perplexity score while the page the user is waiting on takes four times longer to render. The gap between "the model is healthy" and "the product is healthy" is where most AI latency incidents live, and it is a gap you close with instrumentation, not with better prompts.

## The short version

Token counts, perplexity, and F1 scores describe the model's internals. They do not tell you whether users see slower responses, worse answers, or silent failures. The signals that track user-visible degradation are **end-to-end latency at the 95th and 99th percentiles** and **failure rate**, both measured at the edge where requests enter and leave your system.

The failure mode this article addresses is treating internal telemetry as a proxy for user pain. Internal telemetry is necessary for diagnosis. It is not sufficient for detection.

## Why model-centric metrics mislead

Most AI teams begin with model-centric evaluation: tokens per request, perplexity, or F1 on a held-out set. These metrics are cheap to compute, reproducible, and feel objective. What they miss is the **interface between the model and the rest of the system** — tokenization, embedding lookup, retry loops, cache behavior, serialization, and downstream services that time out.

Two patterns recur.

**The cache hit illusion.** A team adds a cache in front of embedding generation, sees a high hit rate, and watches tokens per request fall. Then a deploy changes the prompt slightly, the cache key no longer matches, and every request rebuilds an embedding. The added work is invisible in the hit-rate dashboard until the rate itself drops, but it shows up immediately as a latency step change. Users do not report cache misses. They report that the page got slower.

**The busy GPU.** GPU utilization saturates, so the model is assumed to be the bottleneck. In practice the GPU may be idle-waiting on a slow vector database query, a contended connection pool, or a synchronous serialization step. High utilization means the GPU is doing something, not that it is doing the thing on the critical path.

Both patterns share a cause: the team optimizes the layer it can see and ignores the contract it made with users — "this page responds in under two seconds."

## A latency budget, not a token budget

The mental model that resolves this is a **latency budget**. Decompose the request path into stages, assign each stage a target, and instrument each stage so you can see which one bleeds.

Stage | Illustrative target (p99) | What usually drives it
--- | --- | ---
Tokenization | 20 ms | Input length, tokenizer implementation
Embedding fetch | 50 ms | Vector DB query time, connection pool contention
Model inference | 300 ms | Context length, batch size, KV cache behavior
Post-processing | 30 ms | Parsing, formatting, guardrail checks
API gateway | 10 ms | Serialization, auth, routing
Total | 410 ms | Sum of stage targets

The numbers above are illustrative, not measured. The point is the structure: a per-stage target that sums to a user-facing promise. If any stage exceeds its target, total p99 can jump by hundreds of milliseconds even when the model is byte-for-byte unchanged.

The second half of the model is **failure-rate coupling**. Small increases in error rate often produce user-visible effects — retries, timeouts, manual workarounds — before average latency rises enough to trip a p95 alert. Latency and error rate belong on the same dashboard because they fail together.

## A worked example: the prompt change that broke post-processing

This is a composite scenario assembled from common failure patterns, not a specific incident.

### The change

A prompt template gains a trailing instruction:

```python
# Before
prompt = f"Answer the user's question: {user_input}"

# After
prompt = f"Answer the user's question: {user_input}\n\nDisclaimer: This is AI-generated."
```

Input grows from roughly 85 to 105 tokens. Perplexity on the internal eval set is unchanged. No alarm fires.

### The symptom

End-to-end p99 latency steps up sharply within minutes, and error rate rises alongside it. The first hypothesis is usually "more tokens means slower inference," which is true but rarely accounts for a step change of this size.

### The trace

Distributed tracing (for example, OpenTelemetry spans around each stage) replays the same traffic and attributes time per stage:

Stage | Before | After | Delta
--- | --- | --- | ---
Tokenization | 12 ms | 18 ms | +6 ms
Embedding fetch | 45 ms | 45 ms | 0 ms
Model inference | 290 ms | 310 ms | +20 ms
Post-processing | 18 ms | 200 ms | +182 ms
API gateway | 6 ms | 6 ms | 0 ms

The added tokens explain the inference delta. They do not explain the post-processing delta. The culprit is a new extraction step — a regex applied to the model output — that backtracks catastrophically on the longer string.

### The fix

Replace the regex with a linear scan:

```python
# Before: catastrophic backtracking on long inputs
import re
disclaimer = re.search(r'Disclaimer:.*', prompt, re.DOTALL)

# After: O(n) scan
def extract_disclaimer(text):
    idx = text.find('Disclaimer:')
    return text[idx:] if idx >= 0 else ''
```

The regex in the "before" block is not inherently catastrophic for this specific pattern, but patterns that combine nested quantifiers or ambiguous alternation with `.*` over long inputs frequently are. The general lesson holds: a linear scan has predictable cost, and predictable cost is what a latency budget requires.

### The lesson

Token count moved by about 24 percent. Perplexity did not move. User-visible latency roughly doubled because one stage blew its budget. The team's instinct — tune the model — would have been wasted effort.

## How to measure this in your own system

You cannot borrow someone else's benchmark for this. The numbers depend on your prompt mix, your hardware, your network, and your traffic shape. What you can borrow is the method.

**Instrument the edge first.** At your API gateway or ingress, record a histogram of end-to-end request duration with buckets that span your SLO. A reasonable starting set is `[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0]` seconds. This gives you p95 and p99 without sampling every request.

**Instrument each stage.** Wrap tokenization, embedding fetch, inference, and post-processing in spans. The goal is not a pretty trace viewer; it is the ability to attribute a p99 regression to a stage within minutes.

**Record error rate at the same boundary.** Count requests that returned an error to the caller, and count retries separately. A retry that succeeds still cost the user latency.

**Compare distributions, not averages.** Averages hide tail behavior. When you change anything — prompt, model version, cache policy — compare the p95 and p99 of the before and after windows, not the mean.

**Set the alert on the user contract.** Alert on p99 exceeding your SLO for a sustained window, and on error rate crossing a threshold. Do not alert on token count.

## Common misconceptions

**"If perplexity is flat, the user experience is fine."** Perplexity measures model likelihood on a held-out set. It says nothing about tokenization overhead, network hops, serialization, or downstream timeouts. A model can be perfectly calibrated and the product can still be slow.

**"We track GPU utilization, so we know if the model is the bottleneck."** Utilization tells you the GPU is busy. It does not tell you whether the GPU is on the critical path. A saturated GPU waiting on a vector DB query looks identical to a saturated GPU doing useful work.

**"We have a rate limiter, so failures are impossible."** Rate limiters protect upstream services from overload. They do not prevent downstream timeouts. If a vector DB returns 504s under load, the rate limiter happily admits requests that will all fail.

**"Latency is tokens times milliseconds-per-token."** The multiplier is not constant. Short prompts and long prompts have different per-token costs, and at high concurrency, KV cache pressure changes the curve again. The only reliable measurement is end-to-end p99 under real traffic.

## Failure injection and adaptive batching

Once every stage is instrumented, the next step is to verify that your budget reflects reality.

**Correlated failure injection.** Inject a small, controlled delay into one stage — say, an added 50 ms on a fraction of embedding fetches — and observe how p99 propagates. If a 50 ms stage delay becomes a 400 ms end-to-end jump, you have found a serialization point: a single-threaded connection, a synchronous queue, or a lock. Managed fault-injection services and open-source chaos tooling can both do this; the tool matters less than the discipline of injecting one variable at a time.

**Adaptive batching.** Static batch sizes are a compromise that is wrong at both ends of the load curve. Dynamic batching that targets the current latency budget — small batches under light load, larger batches under heavy load — trades a small amount of throughput for tail-latency control. Whether it helps depends on your inference server and your traffic; measure before and after with the same histogram you built above.

**SLO-based autoscaling.** Define an SLO in user terms — p99 under a threshold, error rate under a threshold — and scale on SLO violation rather than on CPU or GPU utilization. Utilization-based scaling reacts to the wrong signal; it will happily add replicas while the real bottleneck is a downstream database.

## Quick reference

Concept | What to measure | Where it hides
--- | --- | ---
Token budget | Tokens per request | Cost dashboards only
Latency budget | End-to-end p95/p99 | On-call pages
Failure budget | Error rate at the edge | User complaints
Stage latency | Per-stage p99 | Internal profiling
Cache behavior | Miss latency, not just hit rate | GPU queue
Scaling behavior | Non-linear token-to-latency curve | KV cache pressure

## FAQ

**Why do teams keep optimizing token counts?**

Token counts are cheap to log and correlate with billing, so they get built into dashboards first. Moving to latency budgets requires adding spans, adjusting alert thresholds, and accepting that some regressions will be attributed to code outside the model. That work feels like overhead until the first p99 step change, at which point it becomes the only thing that matters.

**How do you justify tracing work to a manager?**

Point at a recent debugging session and count the engineer-hours spent. Then propose instrumenting one endpoint in staging and running a load test. Once the stage-level p99 breakdown is visible, the trade-off between tracing effort and debugging effort becomes concrete rather than theoretical.

**What is the smallest useful change?**

Add an end-to-end request duration histogram at your API gateway with buckets spanning your SLO, and alert when p99 exceeds the SLO for a sustained window. That single change surfaces the class of regression described in this article.

**Should token counts be tracked at all?**

Yes, as a cost and prompt-drift signal. They should not gate deployments or trigger rollbacks. Quality gates belong on latency and error-rate SLOs.

**Do you need a full tracing stack to start?**

No. A histogram at the edge plus a coarse timer around each major stage gets you most of the diagnostic value. Full distributed tracing helps when you have many services and need to correlate across them, but it is not a prerequisite for finding a stage that blew its budget.

## Next step

Open your API gateway or ingress configuration and add a request-duration histogram with buckets `[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0]` seconds, plus an alert that fires when the 0.95 quantile exceeds your SLO for five consecutive minutes. Deploy it to staging, generate load, and confirm the histogram populates. That single change gives you the end-to-end signal that token-count dashboards cannot provide.
