# When your agent goes silent: degrade gracefully

The edge cases only show up once real users hit the system. The evaluationdriven development advice that circulates internally rarely matches what's in the public docs. Here's what actually worked, and why.

## The one-paragraph version (read this first)

When an upstream agent or model is slow or wrong, the default behavior of most systems is to hang, crash, or return garbage. Graceful degradation means the system keeps returning a useful (if reduced) answer instead of a failure. The core tools are timeouts, fallbacks, circuit breakers, and validation. The hard part is not implementing these — it is deciding *what* degraded behavior is acceptable for each call site, because "degrade" is a product decision disguised as an infrastructure problem. A request that takes 30 seconds and returns a wrong answer is worse than a request that takes 200ms and returns a cached, slightly stale answer. This post is about making that trade explicit and testable.

## Why this concept confuses people

Most engineers learn resilience patterns from infrastructure: retries, timeouts, circuit breakers. Those patterns assume the upstream is *down*. An agent or model is rarely down. It is *up and slow*, or *up and confidently wrong*. That difference breaks the mental model.

A classic HTTP timeout story: your service calls a model endpoint, sets a 30-second timeout, and moves on. In production, the p99 latency of that endpoint is 12 seconds, the p95 is 4 seconds, and the median is 800ms. Your timeout never fires. Instead, your own service's thread pool fills up with requests waiting 12 seconds, your own p99 climbs to 11 seconds, and the failure shows up *somewhere else* — usually as a downstream timeout in a service that has nothing to do with the model. Teams spend days looking at the wrong service.

The wrongness case is worse because there is no error to catch. A model returns a JSON object that parses fine, has all the required fields, and contains a hallucinated account number. Your schema validator passes it. Your integration test passes because the test fixture was written by the same person who wrote the prompt. The failure only appears when a human reads the output, which is 40 minutes later, in a support ticket.

The confusion is that "slow" and "wrong" feel like two problems. They are one problem: **the upstream is not a reliable function, and your code is written as if it is.**

## The mental model that makes it click

Think of it like a restaurant kitchen during a rush. The grill station is the model. When it is fast, everything is fine. When it is slow, you do not stand at the pass waiting — you send out the salad that is already plated, tell the server to apologize for the delay on the steak, and keep the dining room moving. The kitchen has a *degraded menu*, not a *failed menu*.

That is the whole idea. Graceful degradation is having a degraded menu ready before the rush starts.

Concretely, every call to an agent or model should answer three questions at design time:

1. **What is the deadline?** Not the timeout — the deadline for the *user-visible* operation. If the user is waiting on a page load, the deadline is 2 seconds, not 30.
2. **What is the fallback?** A cached answer, a cheaper model, a deterministic rule, or an honest "we couldn't do this right now."
3. **What is the validation?** What does "wrong" look like for this call, and can you detect it in under 10ms?

If you cannot answer all three, you do not have a resilient system. You have a system that works in the demo.

## A concrete worked example

A common failure mode: an internal "summarize this ticket" feature calls a model to produce a one-line summary for a support dashboard. The dashboard renders 50 tickets per page.

Naive implementation:

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

The result: the dashboard's p99 goes from ~28s to ~1.8s. The model's load drops because the circuit breaker sheds traffic during incidents. The summaries are slightly worse during incidents, which is the entire point.

## How this connects to things you already know

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

**"Retries handle this."** Retries handle transient failures. They make latency worse during overload, because you are adding load to a system that is already slow. A retry with exponential backoff and jitter is correct for a network blip; it is actively harmful when the model itself is the bottleneck. Cap retries at 2, and only for idempotent calls.

**"A bigger timeout is safer."** A bigger timeout moves the failure from "error" to "slow," which is usually worse. A 500 error is visible and alertable. A 25-second success is invisible until your own p99 page fires at 3am.

**"We validate the output with a JSON schema."** A schema catches malformed output. It does not catch a well-formed hallucination. Validation for model output needs a second layer: does the summary contain tokens from the input? Is the extracted amount within 2x of the source? Is the classification one of the labels you actually trained on? These checks are cheap and catch the majority of confident-wrong outputs.

**"Degradation is an infrastructure concern."** It is a product concern. The product owner has to decide whether a stale summary is acceptable. Engineering can implement it, but the decision is not engineering's to make silently.

**"Circuit breakers are for microservices."** They are for any dependency with a failure mode. A model endpoint is a dependency. So is a feature flag service, a vector database, and an OAuth provider.

## The advanced version (once the basics are solid)

Once timeouts and fallbacks are in place, the interesting work is *adaptive* degradation.

The idea: your system should degrade *more* as load increases, not less. A static 1.5s deadline is fine at low traffic and too generous at high traffic. A common pattern is a token bucket that shrinks the deadline as the model's queue depth grows:

| Signal | Low load | High load | Action |
|---|---|---|---|
| Model p95 latency | < 1s | > 5s | Shrink deadline 1.5s → 800ms |
| Circuit state | closed | open | Serve cache only |
| Cache hit rate | 70% | < 30% | Extend TTL 300s → 900s |
| Error rate | < 0.5% | > 5% | Disable model path entirely |
| Queue depth | < 10 | > 100 | Reject new model calls, fallback only |

The second advanced idea is *tiered models*. If you have a small fast model and a large slow model, route by deadline: try the small one first, and only escalate to the large one if the deadline allows and the small one's output fails validation. This is sometimes called a cascade, and it is one of the few places where you genuinely get better quality *and* lower latency, because most requests never need the big model.

The third is *shadow evaluation*. Run the fallback path on a sample of traffic even when the model is healthy, and log both outputs. This gives you a real measurement of how much quality you lose during degradation, which is the number you need to make the product decision. Without it, "degraded" is a guess.

## Quick reference

| Pattern | When to use | Cost | Risk |
|---|---|---|---|
| Hard timeout | Every model call | Low | Aborts slow-but-correct calls |
| Cached fallback | Read-heavy, stable inputs | Low | Stale answers |
| Deterministic fallback | Always | Low | Lower quality |
| Circuit breaker | Shared dependency | Medium | Sheds load during recovery |
| Tiered models | Quality-sensitive | High | Two paths to maintain |
| Shadow eval | Before shipping degradation | Medium | Extra compute |
| Output validation | Any structured output | Low | False positives |

Checklist before shipping any model call:

- [ ] Deadline is derived from the user-visible budget, not the model's p99
- [ ] Fallback returns a value, not an exception
- [ ] Circuit breaker wraps the call
- [ ] Output validation catches confident-wrong, not just malformed
- [ ] Degraded path is exercised in CI, not just written
- [ ] Metrics distinguish `source=model` from `source=fallback`

## Frequently Asked Questions

**How do I choose a timeout for an LLM call?**
Start from the user-visible budget and work backwards. If a page must render in 2 seconds and the model is one of five calls, it gets at most 400ms. If that is unrealistic, the model call belongs behind a loading state or a background job, not in the request path. A timeout chosen from the model's own latency distribution is almost always too generous.

**Why does my p99 spike after I add a model call?**
Because the model's latency distribution has a long tail, and your request path now inherits it. If the model's p99 is 12s and you call it once per request, your p99 becomes at least 12s. The fix is a deadline shorter than your SLO, plus a fallback, so the tail is truncated before it reaches your users.

**What is the difference between a retry and a fallback?**
A retry asks the same dependency again, hoping for a different result. A fallback asks a different source, or returns a degraded value. Retries are for transient network errors. Fallbacks are for slow or wrong responses. Using a retry where a fallback belongs is a common cause of cascading overload.

**How do I test graceful degradation?**
Inject latency and errors at the client boundary, not the network. In Python, wrap the HTTP client with a fixture that sleeps or raises. In CI, run the degraded path as a first-class test case and assert on the fallback value. If your test suite never exercises the fallback, it is untested code, and it will be the code that runs during the incident.

## Further reading worth your time

- The original circuit breaker write-up by Martin Fowler (2014): https://martinfowler.com/bliki/CircuitBreaker.html
- Google SRE Book, chapter on handling overload: https://sre.google/sre-book/handling-overload/
- `circuitbreaker` Python library docs: https://pypi.org/project/circuitbreaker/
- AWS Well-Architected reliability pillar, "Design interactions to handle failures": https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/welcome.html

Your next 30 minutes: open your service's main model-call module, find every call that has no deadline shorter than 2 seconds, and add a `source` field to its return value. That single field is what makes degradation measurable, and measurement is what turns a vague resilience goal into a decision you can defend.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
