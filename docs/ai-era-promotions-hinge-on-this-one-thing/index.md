# AI-era promotions hinge on this one thing

Most AI feature postmortems do not end with "the model was wrong." They end with a cache key that was too coarse, a schema that drifted, or a rollback path that never existed. The model did what it was asked; the platform around it did not hold up.

This article is about the unglamorous engineering that keeps an LLM-backed feature running six months after launch: input and output validation, a fallback cache, and the three metrics that tell you a prompt is drifting before your users do.

## Why AI features rot in production

A conventional feature degrades when its inputs change in ways you did not anticipate. An LLM-backed feature degrades when its inputs change, when the model version changes, when the prompt template is edited, when the schema of the expected output is edited, and when any upstream cache or queue misbehaves. That is a larger surface, and it fails faster because the failure is often silent: the endpoint returns HTTP 200 with plausible-looking garbage.

A typical failure mode looks like this:

1. A summary endpoint calls an LLM with a 4-second upstream timeout.
2. The vector lookup that feeds the prompt occasionally takes 5 seconds under load.
3. The gateway's timeout fires first, so callers see 504s while the LLM call is still running and billing.
4. Someone adds a cache. The cache key is `user:{user_id}:summary` with a single TTL.
5. A user edits a transaction. The cached summary is now stale but still served for the rest of the TTL.
6. Support tickets arrive, and the team's response is to shorten the TTL, which raises LLM spend and latency.

None of those steps involve model quality. All of them are ordinary distributed-systems problems that microservice teams have hit before. The AI era did not invent them; it made them more expensive because the slow dependency is now a paid, nondeterministic API rather than a database index.

## The three gates

The pattern that holds up is to treat the LLM as an untrusted data source and put three checkpoints around it.

**Gate 1 — input validation.** Reject malformed or oversized prompts before they reach the model. This bounds cost, bounds latency, and prevents a single caller from consuming your rate limit.

**Gate 2 — output validation.** Parse the model's response against a schema before anything downstream sees it. If the response does not conform, treat it as a failed call, not as data.

**Gate 3 — rollback.** Keep the last known-good output for each logical key so a bad generation cannot poison the cache or the user experience.

The important property is that gates 2 and 3 are independent. Schema validation catches structural drift (the model started returning a string where you expected a number). Rollback catches semantic drift (the structure is valid but the content is wrong). You need both, because a well-formed wrong answer passes any schema check.

## A worked example: the summary endpoint

Consider a service that returns a short natural-language summary of a user's recent transactions. The endpoint is `POST /summaries`. The request carries a user ID, a hash of the prompt template in use, and a token budget.

Start with the request model. Pinning the prompt hash in the request is what lets you correlate a spike in validation failures with a specific template change later.

```python
from pydantic import BaseModel, Field

class SummaryRequest(BaseModel):
    user_id: str = Field(..., min_length=10, max_length=36)
    prompt_hash: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    max_tokens: int = Field(..., ge=50, le=2000)
    override_cache: bool = False
```

The `max_tokens` ceiling is a cost control, not a correctness control. Pick it from your own latency budget: if your gateway timeout is 4 seconds and your model produces roughly 60 tokens per second, a 2000-token completion will not finish in time. Measure your actual throughput and set the ceiling below the point where the timeout becomes the binding constraint.

Now the endpoint. The logic below assumes a Redis-compatible cache and an async Redis client.

```python
import redis.asyncio as redis
from datetime import datetime
from fastapi import HTTPException
from fastapi.responses import JSONResponse

r = redis.Redis(host="redis-cache", port=6379, decode_responses=True)

@app.post("/summaries")
async def get_summary(request: SummaryRequest):
    current_key = f"user:{request.user_id}:summary:current"
    last_good_key = f"user:{request.user_id}:summary:last_good"

    # Gate 1: input already validated by the request model.
    cached = await r.get(current_key)
    if cached and not request.override_cache:
        return JSONResponse(content={"summary": cached})

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        response_format={"type": "json_schema", "schema": summary_schema},
        messages=[{"role": "user", "content": request.model_dump_json()}],
    )

    # Gate 2: output validation.
    try:
        parsed = SummaryResponse.model_validate_json(
            response.choices[0].message.content
        )
    except ValidationError:
        # Gate 3: rollback to last known-good.
        last_good = await r.get(last_good_key)
        if last_good:
            return JSONResponse(
                content={"summary": last_good, "fallback": True},
                headers={"X-Cache-Fallback": "true"},
            )
        raise HTTPException(503, detail="No valid summary available")

    # Promote the validated output to both keys.
    await r.set(current_key, parsed.summary, ex=3600)
    await r.set(last_good_key, parsed.summary, ex=86400)

    return JSONResponse(content={"summary": parsed.summary})
```

Three details in that snippet matter more than they look.

**The fallback key has a longer TTL than the current key.** If both expire together, rollback has nothing to fall back to. A common choice is a current TTL sized to your tolerance for staleness (an hour, in the example) and a fallback TTL sized to your tolerance for serving old data during an incident (a day).

**The fallback is written only on success.** Writing the fallback on every request, including fallback responses, means a bad generation eventually overwrites your last good copy. Write it once, from the validated path.

**The fallback response is marked in a header.** Without `X-Cache-Fallback`, you cannot distinguish "the feature is healthy" from "the feature has been silently degraded for three days." That header is the cheapest observability you will ever add.

### What the earlier attempt got wrong

The instructive version of this story is the one where the team skips straight to caching. The first cache key is usually `user:{user_id}:summary` with one TTL and one value. That design has three defects that only appear under real traffic:

- **Staleness.** The summary is derived from mutable data. Any TTL is a bet that the underlying data will not change within it. For transaction data, that bet loses.
- **No recovery path.** When the LLM call fails, the only options are an error or a stale value. There is no third option, because no previous good value was retained separately.
- **Unbounded key growth with no versioning.** If the prompt template changes, every cached value is invalid but indistinguishable from a valid one.

Adding a version component to the key (`...:summary:{version}`) fixes the third defect and creates a new problem: old versions linger until they expire, and nothing tells you which version is being served. The two-key design — `current` and `last_good` — is simpler and makes the rollback semantics explicit.

## What to instrument

Four counters and one histogram cover most of what you need:

- `ai_summary_cache_hit_total` and `ai_summary_cache_miss_total` — the hit rate tells you whether the cache is doing anything.
- `ai_summary_validation_failure_total` — labelled by `prompt_hash`. A rise concentrated on one hash means that template changed or the model's behavior shifted for that input distribution.
- `ai_summary_fallback_total` — labelled by `user_id` bucket. A spike for one user means their data now produces unparseable output; a spike across all users means something systemic.
- `ai_summary_latency_seconds` — a histogram, not an average. You care about p95 and p99 against your gateway timeout, and averages hide exactly the tail that causes 504s.

The alert that earns its keep is a ratio, not a raw count: fallback responses divided by total responses, over a short window. The threshold depends on your tolerance, but the shape of the alert is the same everywhere — if the fallback rate exceeds your chosen bound for ten consecutive minutes, page someone. A raw count of fallbacks is useless because it scales with traffic.

To establish a baseline before you have production traffic, replay a fixed set of recorded prompts through the endpoint in staging and record the validation failure rate. That number is your reference. If production drifts above it, you have a real signal rather than a guess.

## Choosing where validation lives

There is a real trade-off between validating in your application and relying on the model provider's structured-output mode. Both are legitimate.

| Approach | Strength | Cost |
|---|---|---|
| Provider-enforced JSON schema | Fewer malformed responses reach your code | Ties you to one provider's feature set; still requires a semantic check |
| Application-side schema validation | Provider-agnostic; testable without network calls | Every response pays the parse cost; you own the schema |
| Both | Structural failures caught at the edge, semantic failures caught locally | Two schemas to keep in sync |

The third row is the common production choice, and the sync problem is real: if the provider's schema and your local model drift apart, you get failures that only reproduce in one environment. Generate both from a single source of truth if your tooling allows it.

## Failure modes to design against

**Cache stampede.** When a popular key expires, every concurrent request misses and calls the LLM simultaneously. The standard mitigations are a short lock around the regeneration, or serving the stale value while one request refreshes in the background. The `last_good` key makes the second option easy: serve it, mark the response with the fallback header, and refresh asynchronously.

**Retry storms.** A retry on a timeout multiplies load on a dependency that is already slow. Cap retries at one, add jitter, and never retry a request that failed schema validation — that failure is deterministic and will fail again.

**Silent schema drift.** The provider updates a model or you change a prompt, and the output shape changes in a way that still parses. This is why the `prompt_hash` label matters: it lets you compare failure rates across template versions instead of staring at an aggregate.

**Unbounded fallback storage.** The `last_good` key roughly doubles the number of keys per user. Size it before you ship: keys times average value size, plus headroom. If the fallback TTL is a day and your active user count is large, that is a real memory line item, not a rounding error.

## A decision checklist

Before shipping an LLM-backed endpoint, confirm:

1. Every request is validated against a schema with explicit bounds on size and cost.
2. Every response is parsed against a schema, and a parse failure is treated as a failed call.
3. A last-known-good value is retained separately from the current value, with a longer TTL.
4. Fallback responses are marked so they can be counted.
5. Cache hit rate, validation failure rate (by prompt hash), fallback rate, and latency percentiles are all exported.
6. The model identifier and prompt template version are configuration, not literals in the request path.
7. A retry policy exists, and it does not retry deterministic failures.
8. The fallback storage has a sizing estimate.

If any item is missing, that is the next thing to build — not a better prompt.

## FAQ

**Does this require a vector database?**
No. Vector stores solve retrieval, not resilience. The failures described here come from validation, caching and rollback, and they appear in features that use no embeddings at all.

**Can schema validation be skipped if the provider guarantees JSON output?**
No. Provider-enforced structure prevents malformed JSON; it does not prevent a well-formed response with the wrong content, a missing field your downstream code assumes, or a value outside the range you expected. The semantic check is yours to own.

**How should the prompt hash be computed?**
Hash the template text plus the model identifier, and store the mapping from hash to template in version control. The hash then identifies an exact configuration, which is what makes it useful as a metric label.

**What if a user has no last-good value yet?**
Return an explicit error rather than an empty summary. A 503 with a clear body is better than a plausible-looking blank, because the blank will be cached by clients and reported as a content bug rather than an availability bug.

**Is a 24-hour fallback TTL always right?**
No. It is a starting point. The correct value is the longest staleness your users will tolerate without filing tickets, which is a product question, not an infrastructure one.

## Next step

Open your LLM-backed endpoint and make one request with `curl -i`. Look at the response headers. If there is no header that tells you whether the response came from the model, from the cache, or from a fallback, you cannot measure the health of the feature at all. Add that header today, then add the counter that increments when it is set. That single change turns an invisible failure mode into a number you can alert on.
