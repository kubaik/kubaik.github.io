# Most LLM cost dashboards lie…

## Why provider pricing pages understate real spend

A provider's pricing page shows a clean per-token rate: one number for input, another for output. It is tempting to model the monthly bill as `total_tokens × rate`. That model isn't wrong so much as incomplete, and the gap between it and the invoice is where most LLM cost work happens.

The recurring sources of that gap are predictable:

- **Unlogged retries.** SDKs retry on 429 and 5xx by default. Each retry is a full billed request, and most dashboards count one logical call.
- **Cache misses you assumed were hits.** A cache breakpoint only fires when the prefix is byte-identical. A timestamp or interpolated user name near the top of a system prompt silently turns every request into a cache write.
- **Embeddings recomputed on every deploy.** If the indexing pipeline runs in CI without content hashing, unchanged documents are re-embedded on every push.
- **Agent loops.** A tool that returns an error the model can't interpret produces repeated calls with slightly different malformed arguments. Without an iteration cap, this continues until the context window fills.

A common first implementation reads the provider's usage API once a day and plots a single line. That line looks reasonable and is structurally incapable of answering the questions that matter: which feature got more expensive, which tenant is responsible, whether retries are inflating the count. It also misses shared credentials — a staging environment or an overnight eval harness calling the same production key.

LLM FinOps differs from general cloud cost management in one important way: the unit of cost is a token you generate yourself, at runtime, in code you control. That makes it more tractable than cloud spend, not less — but only if the telemetry exists.

## The four fields that make cost control possible

To control LLM cost you need to answer four questions per request: which model, how many input tokens, how many output tokens, and how many retries. Everything else is downstream. If your telemetry doesn't emit those four on every call, you cannot do FinOps — you can only do vibes.

The mechanism that makes this tractable is per-request attribution. Attach a `tenant_id`, a `feature` tag, and a `request_id` to every completion, and emit a structured log line or span containing the token counts the provider returns in the response body.

The documented usage fields differ by provider:

| Provider | Input tokens | Output tokens | Cached input |
|---|---|---|---|
| OpenAI Chat Completions | `usage.prompt_tokens` | `usage.completion_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| Anthropic Messages | `usage.input_tokens` | `usage.output_tokens` | `usage.cache_read_input_tokens` |
| Bedrock (Converse) | `usage.inputTokens` | `usage.outputTokens` | `usage.cacheReadInputTokens` |

Read those fields from the response. Do not estimate from a local tokenizer for billing purposes — a local count is useful for pre-flight budgeting, but it drifts from the billed count, and the drift is invisible.

The second mechanism is a gateway or proxy. Not because a proxy is magic, but because it is the only place you can enforce a budget, downgrade a model, and cache a response without touching every call site. A thin proxy between your app and the provider centralizes model routing rules, per-tenant rate limits, prompt caching, and cost telemetry emission. Teams that skip this layer usually end up with cost logic duplicated across services, and the copies drift apart.

The third mechanism, and the one most teams underinvest in, is prompt caching. The economics as documented by the major providers: Anthropic charges a premium to write a cache entry and a large discount to read it; OpenAI applies a discount to cached input tokens with no write surcharge. The exact multipliers change over time, so check the current pricing page rather than trusting a number from an article. The structural point is stable: a long, reused prefix becomes dramatically cheaper per request, and a prefix that changes per request saves nothing.

## A worked cost model

Because published rates move, treat the following as an illustrative model and substitute your own current rates. The arithmetic is what matters.

Assume a document-Q&A feature with these characteristics:

- 180,000 requests per day
- 6,200 input tokens per request on average
- 340 output tokens per request on average
- A 4,500-token static system prompt that is identical across requests

Daily input tokens: 180,000 × 6,200 = 1,116,000,000 (about 1.12 billion).
Daily output tokens: 180,000 × 340 = 61,200,000 (about 61 million).

Now apply illustrative rates of $2.50 per million input tokens and $10.00 per million output tokens:

- Input: 1,116 × $2.50 = $2,790
- Output: 61.2 × $10.00 = $612
- Total: roughly $3,400 per day, or about $102,000 per month

Output tokens are 5% of the token count but 18% of the bill in this model. That ratio is the reason `max_tokens` settings deserve attention: they bound the tail without affecting typical generation length.

Now apply two changes.

**Prompt caching.** Of the 6,200 input tokens, 4,500 are the static prefix. If cached reads are billed at half the input rate, the effective input cost becomes:

- Cached portion: 4,500 tokens × 180,000 requests = 810,000,000 tokens at half rate = $1,012.50
- Uncached portion: 1,700 tokens × 180,000 requests = 306,000,000 tokens at full rate = $765
- Total input: about $1,778 instead of $2,790

That is a 36% reduction in input spend with no change to output or quality. The saving scales with the ratio of static prefix to total input — if the prefix were 90% of input, the reduction would be proportionally larger.

**Model routing.** Suppose a cheap classifier identifies 70% of requests as answerable by a smaller model priced at one-fifth of the large model's input rate. Routing those requests moves 0.7 × $1,778 ≈ $1,245 of input cost down to roughly $249, and the output cost for that traffic similarly. The exact saving depends entirely on whether the small model's answers pass your evals — which is the real constraint, not the routing code.

**Retry capping.** Retries are small in dollars and large in tail latency. Setting `max_retries` to 2 with jittered backoff prevents the 429 storms where concurrent requests all retry at the same instant, amplifying the rate limit that caused the failure.

To measure your own version of this, instrument the wrapper to emit the four fields, then run two queries: total cost grouped by feature over 7 days, and `cached_tokens / prompt_tokens` grouped by feature. A stable-prefix workload with a cache hit ratio near zero means something is invalidating the prefix.

## Implementation: attribution, budget enforcement, caching

Start with attribution. Wrap every provider call in a function that reads usage from the response and emits a structured event.

```python
import structlog
from openai import OpenAI

log = structlog.get_logger()
client = OpenAI()

# Illustrative rates in USD per token. Replace with current published prices.
PRICES = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000, "cached_in": 0.075 / 1_000_000},
    "gpt-4o":      {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000, "cached_in": 1.25 / 1_000_000},
}

def complete(model: str, messages: list, tenant_id: str, feature: str, **kw):
    resp = client.chat.completions.create(model=model, messages=messages, **kw)
    u = resp.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0
    p = PRICES[model]
    cost = (
        (u.prompt_tokens - cached_tokens) * p["in"]
        + cached_tokens * p["cached_in"]
        + u.completion_tokens * p["out"]
    )
    log.info(
        "llm.call",
        tenant_id=tenant_id,
        feature=feature,
        model=model,
        prompt_tokens=u.prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=u.completion_tokens,
        cost_usd=round(cost, 6),
    )
    return resp
```

That function gives per-tenant, per-feature cost. Ship the events to a log store that supports aggregation — CloudWatch Logs Insights, Loki with LogQL, or ClickHouse — and "which tenant costs the most" becomes a one-line query.

Next, enforce a budget. A minimal middleware that rejects requests once a tenant crosses a daily dollar threshold bounds the worst case:

```python
from fastapi import FastAPI, Request, HTTPException
import redis.asyncio as redis
from datetime import datetime, timezone

app = FastAPI()
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
DAILY_CAP_USD = 25.0

@app.middleware("http")
async def budget_guard(request: Request, call_next):
    tenant = request.headers.get("x-tenant-id")
    if tenant:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"spend:{tenant}:{day}"
        spent = float(await r.get(key) or 0)
        if spent >= DAILY_CAP_USD:
            raise HTTPException(status_code=429, detail="daily budget exceeded")
    return await call_next(request)
```

Set an expiry on the counter key so it does not accumulate indefinitely. This middleware does not reduce cost by itself — it bounds the blast radius. The difference between a bad day costing tens of dollars and thousands is whether this exists.

Finally, prompt caching. For Anthropic's API, mark a cache breakpoint with `cache_control` after the static prefix and before any user-specific content:

```python
system = [
    {"type": "text", "text": LONG_STATIC_PROMPT, "cache_control": {"type": "ephemeral"}},
]
messages = [{"role": "user", "content": user_question}]
```

If a `datetime.now()` or a user name is interpolated into `LONG_STATIC_PROMPT`, every request is a cache miss and you pay the write premium indefinitely. This is the most common caching mistake and it is invisible unless you monitor the hit ratio.

## Failure modes and how to detect them

**Shared credentials across environments.** A nightly eval job runs thousands of completions against the production key. Detection: alert when a key's daily spend exceeds a multiple of its trailing 7-day median. Prevention: separate keys per environment, with separate budgets.

**The tool-schema loop.** In agent frameworks, a tool returning an error the model can't interpret causes repeated calls with slightly different malformed arguments, often dozens of times before any framework limit applies. If `max_iterations` is unset, it runs until the context window fills. Fix: set `max_iterations` explicitly, and return human-readable tool errors the model can act on rather than raw stack traces. Detection: count tool calls per logical request in your traces; a spike is visible immediately.

**Cache invalidation.** Covered above, but worth restating because it fails silently. You see cache fields in the response and assume caching works, while the hit rate is zero. Monitor `cached_tokens / prompt_tokens` per feature. A stable-prefix workload below roughly 0.5 warrants investigation.

**The 429 retry storm.** The provider returns 429, the SDK retries with exponential backoff, and without jitter, concurrent requests retry in lockstep. You pay for the retries that succeed and amplify the rate-limit problem. Fix: jitter plus a low `max_retries`.

**Embedding recompute.** Re-indexing the full corpus on every deploy multiplies embedding cost by deploy frequency. Hash document content and skip unchanged documents. This is a cheap change with a large effect on high-deploy-frequency teams.

## Choosing a stack

| Category | Examples | Choose when |
|---|---|---|
| Managed observability | Hosted request-logging and cost-tracking services | You want per-request visibility within a day and don't want to run infrastructure |
| OpenTelemetry instrumentation | OTel-based LLM instrumentation libraries | You already run OTel and want cost data in your existing backend |
| Self-hosted gateway | A proxy you run yourself, or an open-source gateway | You have multiple providers and want routing, budgets, and caching in one place |
| Local token counting | Provider tokenizer libraries | Pre-flight budget checks, not billing |

The build-versus-buy decision turns on how many places your code calls a provider. A single service calling one provider needs a wrapper function and nothing else. Multiple services calling multiple providers needs a gateway, because otherwise the cost logic is duplicated and the copies diverge.

Be skeptical of any tool that promises automatic traffic routing via a learned classifier. The routing is the easy part. The hard part is knowing which routes are safe to downgrade, and that requires eval data you already have. A small heuristic plus an eval set is usually enough.

## When not to do any of this

If monthly LLM spend is in the low hundreds of dollars, the engineering time to build a gateway, budget counter, and caching layer costs more than a year of savings. Set a hard `max_tokens`, use a cheaper model where quality allows, and revisit when spend crosses a threshold where an engineer-week is clearly cheaper than the bill.

If every request has a unique long input with no shared prefix, prompt caching won't help. Semantic caches — returning a stored answer for a similar question — are risky wherever a wrong answer carries a cost, and in regulated environments where every prompt and response must be auditable, a shared cache is a compliance problem rather than a cost win.

If the team is two engineers shipping a prototype, use the provider SDK directly, log usage to a file, and move on. FinOps is a scaling problem, and premature FinOps is as wasteful as premature optimization.

## FAQ

**How do I track LLM token costs per customer?**
Tag every request with a tenant identifier at the gateway or in the SDK wrapper, and emit a structured log line containing the token counts from the response's usage field. Send those logs to a queryable store and aggregate by tenant. The counts must come from the provider's response, not a local tokenizer, because the estimate drifts from the billed count.

**Why is the bill higher than the token count suggests?**
Three usual causes: retries that aren't logged, prompt cache misses assumed to be hits, and output tokens priced higher than input. Check the cached-token field in the usage object — if it is zero on a workload with a stable system prompt, the cache is not firing. Also check the SDK's retry setting, since each retry is a full billed request.

**When does prompt caching actually save money?**
When you have a long, stable prefix — a system prompt, few-shot examples, or retrieved context — reused across many requests. It saves nothing if the prefix changes per request. Monitor the ratio of cached to total input tokens to confirm it is working.

**How do I stop an agent from looping and burning tokens?**
Set an explicit iteration cap, return tool errors as messages the model can act on rather than raw exceptions, and add a per-request token budget that aborts when cumulative usage crosses a threshold. Log every tool call; a loop is invisible until you see the same call repeated dozens of times in a trace.

**Is a gateway required?**
No. A single service calling one provider needs only a wrapper function. A gateway becomes worthwhile when multiple services call multiple providers, because it prevents cost logic from being duplicated and drifting.

## What to do in the next 30 minutes

Open the function that all your provider calls go through and add three fields to whatever it logs: input tokens, output tokens, and cached input tokens, read directly from the response's usage object. If no single wrapper exists, creating one is the first fix and is a small change. Once those three fields are flowing into a queryable log store, run one query grouping cost by feature over the last 24 hours. The leak will be visible, and every other technique in this article follows from having that number.
