# When your agent goes silent: degrade gracefully

Most resilience advice assumes the upstream dependency is *down*. An agent or model is rarely down. It is *up and slow*, or *up and confidently wrong*. That difference invalidates the mental model most engineers carry over from HTTP services, and it means the failure shows up somewhere other than where it was caused.

## The one-paragraph version

When an upstream agent or model is slow or wrong, the default behavior of many systems is to hang, crash, or return garbage. Graceful degradation means the system keeps returning a useful, reduced answer instead of a failure. The core tools are deadlines, fallbacks, circuit breakers, and output validation. The hard part is not implementing these — it is deciding *what* degraded behavior is acceptable at each call site, because "degrade" is a product decision disguised as an infrastructure problem. A request that takes 30 seconds and returns a wrong answer is worse than one that takes 200ms and returns a slightly stale cached answer. The rest of this article is about making that trade explicit and testable.

## Why this concept confuses people

Infrastructure resilience patterns — retries, timeouts, circuit breakers — assume the upstream is unavailable. A model endpoint is almost never unavailable. It accepts the connection, returns HTTP 200, and takes 12 seconds. Or it returns valid JSON containing a hallucinated account number.

Consider a common timeout story. A service calls a model endpoint, sets a 30-second timeout, and moves on. In production, the endpoint's p99 latency is 12 seconds, the p95 is 4 seconds, and the median is 800ms. The timeout never fires. Instead, the calling service's thread pool or connection pool fills with requests waiting 12 seconds, its own p99 climbs toward 11 seconds, and the failure surfaces somewhere else — typically as a downstream timeout in a service that has nothing to do with the model. Teams then spend days investigating the wrong service.

The wrongness case is worse because there is no error to catch. A model returns a JSON object that parses fine, has every required field, and contains a fabricated value. The schema validator passes it. The integration test passes because the fixture was written by the same person who wrote the prompt. The failure appears only when a human reads the output, often much later, in a support ticket.

"Slow" and "wrong" feel like two problems. They are one problem: **the upstream is not a reliable function, and the calling code is written as if it is.**

## The mental model that makes it click

Think of a restaurant kitchen during a rush. The grill station is the model. When it is fast, everything is fine. When it is slow, the kitchen does not stand at the pass waiting — it sends out the salad that is already plated, tells the server to apologize for the delay on the steak, and keeps the dining room moving. The kitchen has a *degraded menu*, not a *failed menu*.

Graceful degradation is having that degraded menu ready before the rush starts.

Concretely, every call to an agent or model should answer three questions at design time:

1. **What is the deadline?** Not the timeout — the deadline for the *user-visible* operation. If a user is waiting on a page load, the deadline is measured in hundreds of milliseconds, not tens of seconds.
2. **What is the fallback?** A cached answer, a cheaper model, a deterministic rule, or an honest "we couldn't do this right now."
3. **What is the validation?** What does "wrong" look like for this call, and can it be detected in under 10ms?

If all three cannot be answered, the system is not resilient. It is a system that works in the demo.

## A worked example: 50 summaries on one page

A typical failure mode: an internal "summarize this ticket" feature calls a model to produce a one-line summary for a support dashboard. The dashboard renders 50 tickets per page.

A naive implementation:

```python
# naive.py — do not ship this
import httpx

async def summarize(ticket_text: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://model.internal/v1/summarize",
            json={"text": ticket_text},
        )
        r.raise_for_status()
        return r.json()["summary"]

async def render_dashboard(tickets):
    # 50 concurrent calls, each allowed 30s
    return await asyncio.gather(*(summarize(t.body) for t in tickets))
```

What happens in production: the model endpoint has a p99 of 12 seconds under load. Fifty concurrent calls saturate the model's own queue. Latency climbs. `asyncio.gather` waits for the slowest call. The dashboard takes 28 seconds to render. Users refresh. Load doubles. The endpoint falls over entirely.

A degraded version:

```python
# degraded.py
import asyncio
import httpx
from circuitbreaker import circuit  # circuitbreaker 2.0.0

DEADLINE_S = 1.5
CACHE_TTL_S = 300

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_model(text: str) -> str:
    async with httpx.AsyncClient(timeout=DEADLINE_S) as client:
        r = await client.post(
            "https://model.internal/v1/summarize",
            json={"text": text},
        )
        r.raise_for_status()
        summary = r.json()["summary"]
        if not is_plausible(summary, text):
            raise ValueError("implausible summary")
        return summary

async def summarize(ticket) -> tuple[str, str]:
    """Return (summary, source). Source is 'model', 'cache', or 'fallback'."""
    cached = cache.get(ticket.id)
    if cached and cache.age(ticket.id) < CACHE_TTL_S:
        return cached, "cache"
    try:
        summary = await asyncio.wait_for(call_model(ticket.body), timeout=DEADLINE_S)
        cache.set(ticket.id, summary)
        return summary, "model"
    except (asyncio.TimeoutError, httpx.HTTPError, ValueError):
        # Deterministic fallback: first 80 chars of the subject line
        return ticket.subject[:80], "fallback"
```

Three things changed, and each one matters:

- **The deadline is 1.5s, not 30s.** The user is looking at a dashboard. 1.5s is the budget for the whole page, and the model gets a fraction of it.
- **There is a fallback that is always available.** The subject line is already in the database. It is a worse summary, but it is a summary.
- **There is a circuit breaker.** After 5 consecutive failures, calls short-circuit for 30 seconds. The dashboard stays fast even when the model is fully down.

### Measuring the improvement

Do not trust a claim about "p99 goes from 28s to 1.8s" unless you produce it yourself. To measure this change, instrument the following and compare before and after on the same traffic profile:

- **Per-call latency histogram** for the model call, labeled with the outcome (`model`, `cache`, `fallback`, `timeout`, `circuit_open`). In Python, `prometheus_client.Histogram` or a simple `time.perf_counter()` around the call works.
- **End-to-end dashboard render time**, ideally as a server-side histogram plus a browser-side `PerformanceObserver` for `largest-contentful-paint`.
- **Pool saturation**: `httpx` does not expose this directly, so instrument your own semaphore or connection limit and record the number of waiters.
- **Upstream queue depth** if the model server exposes it (most inference servers expose a metrics endpoint).
- **Fallback rate**: the fraction of calls returning `source=fallback`. This is your degradation-quality signal.

Then run a controlled comparison: replay a fixed set of tickets through both implementations under the same concurrency, or use a load generator (`k6`, `locust`, `vegeta`) pointed at the two builds. The number you care about is the ratio of end-to-end p99 to your SLO, not the absolute latency.

## Connecting to primitives you already know

If you have ever written a database query with a `LIMIT`, you have already done graceful degradation. You decided that the first 100 rows are more useful than waiting for all 10 million.

If you have ever used a CDN, you have done it too. A stale asset served from cache is a degraded response. It is also almost always better than a 504.

If you have used `Promise.race` in JavaScript to implement a timeout, you have the primitive. The missing piece is deciding what the losing branch returns.

```javascript
// Node 20 LTS — deadline with a real fallback
async function withDeadline(promise, ms, fallback) {
  let timer;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve(fallback), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

const summary = await withDeadline(
  callModel(ticket.body),
  1500,
  { summary: ticket.subject.slice(0, 80), source: 'fallback' }
);
```

The difference between this and a plain `AbortController` timeout is that `AbortController` gives you a rejection. This gives you a *value*. That value is what the UI renders.

## Common misconceptions, corrected

**"Retries handle this."** Retries handle transient failures. They make latency worse during overload, because they add load to a system that is already slow. A retry with exponential backoff and jitter is correct for a network blip; it is actively harmful when the model itself is the bottleneck. Cap retries at 2, and only for idempotent calls.

**"A bigger timeout is safer."** A bigger timeout moves the failure from "error" to "slow," which is usually worse. A 500 error is visible and alertable. A 25-second success is invisible until your own p99 alert fires at 3am.

**"We validate the output with a JSON schema."** A schema catches malformed output. It does not catch a well-formed hallucination. Validation for model output needs a second layer: does the summary contain tokens from the input? Is the extracted amount within a stated tolerance of the source? Is the classification one of the labels the system actually handles? These checks are cheap and catch many confident-wrong outputs.

**"Degradation is an infrastructure concern."** It is a product concern. Someone has to decide whether a stale summary is acceptable. Engineering can implement it, but the decision is not engineering's to make silently.

**"Circuit breakers are for microservices."** They are for any dependency with a failure mode. A model endpoint is a dependency. So is a feature flag service, a vector database, and an OAuth provider.

## Advanced patterns, once the basics are solid

Once deadlines and fallbacks are in place, the interesting work is *adaptive* degradation.

The idea: the system should degrade *more* as load increases, not less. A static 1.5s deadline is fine at low traffic and too generous at high traffic. A common pattern is to shrink the deadline as the model's queue depth grows. The table below is a design sketch, not measured data — the thresholds must be derived from your own latency distributions.

| Signal | Low load | High load | Action |
|---|---|---|---|
| Model p95 latency | < 1s | > 5s | Shrink deadline 1.5s → 800ms |
| Circuit state | closed | open | Serve cache only |
| Cache hit rate | 70% | < 30% | Extend TTL 300s → 900s |
| Error rate | < 0.5% | > 5% | Disable model path entirely |
| Queue depth | < 10 | > 100 | Reject new model calls, fallback only |

The second advanced idea is *tiered models*. If a small fast model and a large slow model are both available, route by deadline: try the small one first, and escalate to the large one only if the deadline allows and the small one's output fails validation. This is sometimes called a cascade, and it is one of the few places where quality and latency can both improve, because most requests never need the big model.

The third is *shadow evaluation*. Run the fallback path on a sample of traffic even when the model is healthy, and log both outputs. This gives a real measurement of how much quality is lost during degradation, which is the number needed to make the product decision. Without it, "degraded" is a guess.

## A decision checklist

Before shipping any model call:

- [ ] Deadline is derived from the user-visible budget, not the model's p99
- [ ] Fallback returns a value, not an exception
- [ ] Circuit breaker wraps the call
- [ ] Output validation catches confident-wrong output, not just malformed output
- [ ] Degraded path is exercised in CI, not just written
- [ ] Metrics distinguish `source=model` from `source=fallback`
- [ ] The product owner has signed off on what "degraded" returns

## Frequently asked questions

**How do I choose a deadline for an LLM call?**
Start from the user-visible budget and work backwards. If a page must render in 2 seconds and the model is one of five calls, it gets at most 400ms. If that is unrealistic, the model call belongs behind a loading state or a background job, not in the request path. A deadline chosen from the model's own latency distribution is almost always too generous.

**Why does my p99 spike after I add a model call?**
Because the model's latency distribution has a long tail, and the request path now inherits it. If the model's p99 is 12s and it is called once per request, the request p99 becomes at least 12s. The fix is a deadline shorter than the SLO, plus a fallback, so the tail is truncated before it reaches users.

**What is the difference between a retry and a fallback?**
A retry asks the same dependency again, hoping for a different result. A fallback asks a different source, or returns a degraded value. Retries are for transient network errors. Fallbacks are for slow or wrong responses. Using a retry where a fallback belongs is a common cause of cascading overload.

**How do I test graceful degradation?**
Inject latency and errors at the client boundary, not the network. In Python, wrap the HTTP client with a fixture that sleeps or raises. In CI, run the degraded path as a first-class test case and assert on the fallback value. If the test suite never exercises the fallback, it is untested code, and it will be the code that runs during the incident.

**Does a circuit breaker need a half-open state?**
A useful implementation has three states: closed, open, and half-open. After the recovery timeout, the breaker allows a limited number of trial requests. If they succeed, it closes; if they fail, it reopens. Without the half-open state, the breaker either never recovers or floods the recovering dependency with full traffic.

## Your next 30 minutes

Open the module that makes your service's model calls. For every call, check two things: whether its deadline is shorter than the user-visible budget for that operation, and whether the call can return a value instead of raising. Then add a `source` field to the return value (`model`, `cache`, or `fallback`) and emit it as a metric label. That single field is what makes degradation measurable, and measurement is what turns a vague resilience goal into a decision you can defend.
