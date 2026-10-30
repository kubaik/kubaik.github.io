# AI-native apps break when you copy old rules

## The conventional advice and where it stops working

The standard guidance for adding an AI feature is familiar: wrap the model in a REST endpoint, cache the response, monitor latency, put it behind a load balancer with autoscaling. That advice was written for APIs whose behavior resembles a function call — deterministic, cheap, and fast.

A model call is none of those things. It is a distributed workload with variable latency, variable cost, and no strong guarantee that two identical requests produce identical output. Copying REST patterns onto it produces predictable failure modes, and those failures usually appear under load rather than in development.

The gap shows up in three places:

1. **Requests are not idempotent.** A cache miss triggers a model call whose output can vary with temperature, top_p, seed, and the model version behind an alias.
2. **Cost is not linear in request count.** Cost scales with tokens consumed, and tokens consumed can grow when prompts get longer, when retries fire, or when a model version changes its output length.
3. **Failure modes are new.** A modest latency increase can cascade if retry budgets, queue depths, and connection pools were sized for a fixed-latency dependency.

Teams typically discover these gaps in production, while debugging retries, cache invalidation, and budget alerts simultaneously.

## What the standard pattern actually does under load

Consider a feature built exactly the way the old advice suggests: an HTTP endpoint that calls a model synchronously, caches the response, and returns JSON. It behaves correctly at low traffic. Then traffic doubles, and four things tend to happen in roughly this order.

**Connection pool exhaustion.** If the handler holds a database connection while it awaits the model, the pool becomes a scarce resource whose occupancy time is set by model latency rather than database latency. With a pool of size `N` and an average model latency of `L` seconds, the sustainable request rate is approximately `N / L`. A pool of 20 with a 2-second model call sustains about 10 requests per second before requests start queueing for a connection. The fix is not a bigger pool; it is not holding the connection across the model call. Fetch what you need, release the connection, then call the model.

**Cold starts.** If the model runs on infrastructure with scale-to-zero or slow instance initialization, new capacity arrives seconds after it is needed. Provisioned concurrency or a warm pool addresses this, but the underlying issue is that the request path includes an operation whose latency is not bounded by your own code.

**Cache stampede.** A short TTL plus a burst of identical requests produces simultaneous misses. If 200 requests arrive for the same key in the same second and the cache entry has just expired, all 200 may call the model. The standard mitigations are a per-key lock so only one caller populates the entry, and a short randomized jitter added to TTLs so entries do not expire in lockstep.

**Budget drift.** A token limit enforced in application code goes stale when the model or prompt changes. If cost monitoring tracks wall-clock compute rather than tokens, the drift is invisible until the invoice arrives.

A common cycle follows: add caching, hit cold starts, enlarge the pool, watch the bill, rewrite the caching layer, repeat. The root cause is the assumption that the model call behaves like a fast, deterministic dependency.

## A different mental model: the model call is a task

The reframing that resolves most of these problems is to stop treating the model call as a function invocation and start treating it as a task submitted to a queue.

| Traditional API call | Model call as distributed task |
|---|---|
| Synchronous function call | Asynchronous job with an ID |
| Latency bounded by your code | Latency varies with model, queue, retries |
| Idempotent for a given input | Output varies with sampling parameters and model version |
| Cache key is the URL | Cache key must include prompt and sampling parameters |
| Cost scales with CPU time | Cost scales with tokens, concurrency, and retries |

Four consequences follow from this shift.

**Requests become tasks.** The endpoint enqueues a job and returns a job ID immediately. A separate worker pool consumes jobs and calls the model. The caller's latency is now bounded by your own queue, not by the model. A status endpoint lets clients poll, or you can push results over Server-Sent Events or WebSockets.

**Cache keys must include sampling parameters.** Two requests with identical prompts but different temperature values are different requests. A cache key derived only from the prompt text will return results that do not match what the caller asked for.

**Retries must be selective.** Retrying on every 5xx multiplies token spend without improving the success rate for deterministic failures. Retry on rate limits and timeouts with exponential backoff and a jittered delay; do not retry on validation errors or content policy rejections.

**Cost becomes a first-class metric.** Track tokens consumed per request, per user, and per time window. A 500 ms call and a 3-second call may cost the same or differ by an order of magnitude depending on token counts, so latency is a poor proxy for spend.

## A worked example: sizing a queue and a token budget

The following numbers are illustrative, chosen to show the arithmetic rather than to describe a measured system.

Suppose a feature averages 1,500 input tokens and 500 output tokens per request, for 2,000 tokens total. Suppose the model's published price is $3 per million input tokens and $15 per million output tokens. Then the per-request cost is:

- Input: 1,500 tokens × $3 / 1,000,000 = $0.0045
- Output: 500 tokens × $15 / 1,000,000 = $0.0075
- Total: $0.012 per request

At 100,000 requests per month that is $1,200. Now suppose a bug causes a retry on 20% of requests. Effective requests become 120,000, and cost rises to $1,440 — a 20% increase with no change in user-visible behavior. If instead a prompt change doubles input tokens, input cost becomes $0.009 and total per-request cost becomes $0.0165, a 37.5% increase. Neither change is visible in a latency dashboard.

For queue sizing, suppose the worker pool can process 20 jobs per second per worker, and the model call takes 2 seconds. Each worker can therefore hold 2 jobs in flight, so 10 workers sustain 20 jobs per second. If the arrival rate is 50 jobs per second, the queue grows by 30 jobs per second, and a 500-job backlog forms in under 17 seconds. The relevant alerts are queue depth and queue age, not request latency.

To measure the real figures rather than these illustrative ones:

- Instrument token counts at the point where you build the request and where you receive the response, and emit them as a metric labeled by model ID and endpoint.
- Log cache hits and misses with the cache key, then compute the hit rate over a rolling window.
- Emit queue depth and the age of the oldest job as gauges.
- Compare p50, p95, and p99 latency separately, because a task queue usually improves the tail far more than the median.
- Run a load test that ramps to peak concurrency and holds it, then watch queue depth and token spend rather than only latency.

## Designing the cache key

A cache key for a model call must be a hash of everything that can change the output. At minimum: the prompt, the system message, the model ID, the temperature, top_p, top_k, the seed if set, and any tool or function definitions. If the feature is conversational, the prior turns are part of the input and belong in the key.

```python
import hashlib
import json

def make_cache_key(
    prompt: str,
    system: str,
    model_id: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
) -> str:
    params = {
        "prompt": prompt,
        "system": system,
        "model_id": model_id,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }
    # sort_keys makes the serialization stable regardless of dict insertion order
    encoded = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Two details matter. First, `sort_keys=True` prevents two logically identical parameter sets from producing different keys. Second, the model ID should be the exact version identifier, not an alias that can be repointed, because a repointed alias changes the output for the same key.

For TTL, choose a window that matches how long the underlying data is valid. Prices, inventory, and availability change on their own schedules; a TTL longer than that schedule serves stale answers. Add a small random jitter to each TTL so that a batch of entries written at the same time does not expire simultaneously.

To prevent stampedes, take a short-lived lock on the cache key before calling the model. If the lock is held, the caller either waits briefly or returns a cached value if one exists. The lock should have a timeout so a crashed worker does not block the key indefinitely.

## Budgeting tokens per user

Token budgets are the equivalent of rate limits for a metered dependency. Track consumption per user over a rolling window, and reject or downgrade requests that would exceed the limit.

```javascript
// tokenBudget.js
export class TokenBudget {
  constructor({ perUserLimit, windowSeconds }) {
    this.perUserLimit = perUserLimit;
    this.windowSeconds = windowSeconds;
  }

  // Returns true if the request is allowed.
  async check(userId, estimatedTokens) {
    const used = await this.getUsage(userId);
    return used + estimatedTokens <= this.perUserLimit;
  }

  async record(userId, tokens) {
    // Increment the counter and set the window expiry on first write.
    // Implementation depends on your store; Redis INCR with EXPIRE is typical.
    throw new Error("implement against your store");
  }

  async getUsage(userId) {
    throw new Error("implement against your store");
  }
}
```

The estimate passed to `check` should be an upper bound, computed from the prompt length plus the maximum output tokens you allow. Recording actual usage after the call lets you reconcile estimates against reality and tune the estimate over time.

A budget limit is only useful if exceeding it produces a defined behavior. Decide in advance whether an over-budget request is rejected with an error, downgraded to a cheaper model, or queued until the window resets. Each has different user-visible consequences.

## Retry policy

Retries are where cost and latency interact most sharply. The rules that tend to work:

- Retry only on transient conditions: rate limits, timeouts, and connection errors. Do not retry on validation failures or content policy rejections.
- Use exponential backoff with jitter. A fixed delay synchronizes retries across workers and produces repeated bursts.
- Cap the total number of attempts, and count retries against the token budget.
- Respect the retry-after header when the provider sends one.
- Make retried operations idempotent from the caller's perspective by keying the job, so a retry does not produce duplicate side effects.

A retry budget that was sized for a 3-second SLA will fire far more often against a model whose p99 latency is higher than that, and each firing costs tokens.

## When the simpler pattern is the right choice

Not every AI feature needs a task queue. The synchronous endpoint remains appropriate when:

- The model is fast enough that the caller's latency budget accommodates it, and requests are infrequent enough that pool occupancy is not a constraint.
- The output is advisory rather than decision-making, so occasional latency spikes and retries are acceptable.
- The cost per call is low enough that budget drift is not a meaningful risk.

The decision is about the risk profile, not about sophistication. If the output influences a financial decision, a safety-relevant action, or a user-visible commitment, the cost of a latency spike or a duplicated call is high, and the task-queue pattern pays for itself. If the output is a suggestion a user can ignore, the simpler architecture is usually correct.

## Decision checklist

Use this list to decide which pattern fits:

- Does the caller's latency budget accommodate the model's p99 latency, not its median?
- Is the model call holding a scarce resource such as a database connection while it runs?
- Can two callers submit the same request concurrently, and would that produce duplicate cost?
- Does the output depend on sampling parameters that could differ between callers?
- Is there a hard cost ceiling per user or per time window, and is it enforced in code?
- Are retries bounded, jittered, and counted against the budget?
- Are queue depth and queue age monitored as first-class metrics?
- Is the model version pinned, or is it an alias that can change underneath you?

If any answer is unfavorable, the corresponding part of the task-based design applies. The pattern is composable: you can adopt prompt hashing and token budgets without adopting a queue, and vice versa.

## Common objections

**A task queue complicates the code.** It adds infrastructure, not business logic. The model call itself is unchanged; what changes is where it runs and how the caller learns the result. The trade is a small amount of queue plumbing against the failure modes described above.

**Users will not poll for results.** Polling is a UX decision, not an architectural one. A progress indicator during a short wait is often preferable to a blank screen during a longer one. If you need push semantics, Server-Sent Events or WebSockets can sit on top of the same queue.

**The queue backend is expensive.** A queue backend is typically a small fraction of model spend, and the token savings from deduplicated work often exceed it. If cost is a constraint, an in-process queue or an existing message broker may be sufficient; the pattern does not depend on a specific product.

**The model is stateless, so why cache?** Caching is about avoiding repeated computation, not about model state. Identical prompts with identical parameters are common in practice, and the cache key described above ensures that only genuinely identical requests share an entry.

**Serverless functions already scale.** Autoscaling addresses capacity, not latency. A serverless function that calls a model synchronously still holds the caller for the duration of the call. The queue decouples the two.

## Action to take in the next 30 minutes

Open the handler for your highest-traffic AI endpoint and find the line where the model call begins. Check whether a database connection is held across that line. If it is, restructure the handler so the connection is released before the call, then measure p95 latency before and after under a load test that holds peak concurrency for at least 60 seconds.
