# AI agents eat your budget 3 ways

## Why happy-path docs stop being useful in production

Most AI agent tutorials end at a prompt and a success case. Provider documentation describes a single clean request: the request goes out, a response comes back, tokens are billed. That model is accurate and almost useless once an agent runs continuously, because production agents are not single API calls. They are systems in which retries, timeouts, queues, cold starts and concurrency interact. Each interaction has a cost, and most of those costs are invisible until they are instrumented.

The documented behavior of an LLM API is usually per-request: you are billed for input and output tokens on requests that return a response. What the docs typically do not describe is the system-level behavior around that request. A retry is a new request. A timeout is a request that may still be in flight. A queued job is a request that has not started yet. When these overlap, the bill and the latency both grow in ways that are not obvious from reading the API reference.

This article covers three cost vectors that dominate production agent budgets, then shows how to measure each one, how to cap it, and where the common failure modes hide.

## The three cost vectors

### 1. Token inflation from retries

Every retry that reaches the provider is a new request with its own input tokens. If a 500-token prompt is retried three times and the fourth attempt succeeds, the provider has processed roughly 2,000 input tokens across four requests, but only one of those requests produced a useful answer. The intermediate requests are not free in the sense that matters: they consume rate-limit budget, they consume concurrency slots, and depending on the provider they may be billed.

The arithmetic is simple and worth writing down:

- Prompt size: 500 input tokens
- Retries that reach the provider: 3
- Total input tokens processed: 500 × (1 + 3) = 2,000 tokens

Whether you are billed for all 2,000 or only the successful 500 depends on the provider's policy for failed or errored requests, and that policy is worth verifying directly rather than assuming. Even when failed requests are not billed, they still count against rate limits at many providers, which means a retry storm can push you into throttling even when your token spend looks flat.

### 2. Wall-clock inflation from timeouts and backoff

A timeout is not a single number. It is at least three numbers that multiply:

- Per-attempt timeout
- Backoff delay between attempts
- Maximum number of attempts

If the per-attempt timeout is 5 seconds, the backoff doubles from 1 second, and the maximum attempts is 3, the worst-case wall-clock time is not 15 seconds. It is the sum of each attempt's timeout plus each backoff: 5 + 1 + 5 + 2 + 5 = 18 seconds. Add jitter and the number moves again. The client-facing timeout must be larger than this worst case, or users see failures while the system is still retrying.

### 3. Resource drift in long-running processes

Agents that hold connections, streaming buffers or in-memory context for the duration of a request accumulate resources. When requests time out, those resources are not always released promptly. A process that runs for hours can accumulate memory, file descriptors and open HTTP connections faster than a short-lived process. This is not usually a token cost, but it is a real cost in instance size, restart frequency and operational attention.

## How to measure each vector

The core discipline is to log per-attempt data, not per-request data. A single log line at the end of a successful request hides everything that happened before it.

Instrument these fields on every attempt:

- `trace_id` — correlates attempts belonging to one logical request
- `attempt_number` — 1-based
- `input_tokens` and `output_tokens` from the provider response, when available
- `duration_ms` for the attempt
- `error_type` when the attempt fails
- `backoff_ms` before the next attempt
- `cold_start` boolean, if the runtime can report it

Then compute, per trace:

- Total attempts
- Total input tokens across attempts
- Wall-clock time from first attempt start to final result
- Whether the final result was a success or a give-up

With this data you can answer the questions that matter. What fraction of your token spend goes to attempts that did not produce the final answer? What is the p95 wall-clock time for traces that retried versus traces that did not? How often does the client timeout fire while the orchestrator is still retrying?

A useful comparison to run before and after any change:

| Metric | How to compute | Why it matters |
|---|---|---|
| Retry amplification | total attempts ÷ successful traces | Shows how much work is wasted |
| Token amplification | total input tokens ÷ tokens of final attempt | Shows how much token budget retries consume |
| Timeout mismatch rate | traces where client timeout fired but orchestrator later succeeded | Shows user-visible failures that were not real failures |
| Cold-start share | attempts with cold_start=true ÷ total attempts | Shows whether provisioning is worth it |

Run these against a representative hour of production traffic, not a synthetic benchmark. The numbers will differ from any published figure because they depend on your prompt sizes, your provider's throttling behavior and your traffic shape.

## A worked example

Assume an agent with these stated parameters, all illustrative:

- Prompt: 500 input tokens
- Expected output: 200 tokens
- Per-attempt timeout: 5 seconds
- Backoff: 1 second, doubling
- Maximum attempts: 3
- Provider input price: $3 per million tokens
- Provider output price: $15 per million tokens

Happy path, one attempt, success:

- Input tokens: 500
- Output tokens: 200
- Input cost: 500 ÷ 1,000,000 × $3 = $0.0015
- Output cost: 200 ÷ 1,000,000 × $15 = $0.0030
- Total: $0.0045

Retry path, three attempts, third succeeds:

- Input tokens processed: 500 × 3 = 1,500
- Output tokens: 200 (only the successful attempt produces output)
- Input cost: 1,500 ÷ 1,000,000 × $3 = $0.0045
- Output cost: $0.0030
- Total: $0.0075

The retry path costs 67% more than the happy path in this example, and that is before counting the wall-clock time. If the first two attempts each consumed the full 5-second timeout and the backoffs were 1 and 2 seconds, the user waited 5 + 1 + 5 + 2 + (successful attempt time) seconds. If the successful attempt took 2 seconds, the total is 15 seconds.

Now scale: 10,000 requests per day, of which 5% take the retry path.

- Happy-path requests: 9,500 × $0.0045 = $42.75
- Retry-path requests: 500 × $0.0075 = $3.75
- Daily total: $46.50

If the retry rate rises to 20% under load, the same 10,000 requests cost:

- Happy-path requests: 8,000 × $0.0045 = $36.00
- Retry-path requests: 2,000 × $0.0075 = $15.00
- Daily total: $51.00

The token cost rose about 10%, but the wall-clock cost and the rate-limit pressure rose much more. The token math understates the operational impact.

## Code: an instrumented agent

The following example uses Python 3.11, `httpx` for async HTTP, `backoff` for retry with jitter, `structlog` for structured logs, and `redis` for rate limiting and caching. Version pins are given so the example is reproducible; check the current release before deploying.

```bash
pip install httpx backoff structlog redis
```

```python
import asyncio
import hashlib
import time
import uuid

import backoff
import httpx
import structlog
from redis import Redis
from redis.exceptions import RedisError

# Configuration
API_URL = "https://api.example-llm-provider.com/v1/messages"
API_KEY = "REPLACE_WITH_SECRET"
MODEL = "REPLACE_WITH_MODEL_ID"

# Illustrative prices, expressed per token
INPUT_PRICE_PER_TOKEN = 3.0 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000

MAX_ATTEMPTS = 3
BASE_TIMEOUT_SECONDS = 5.0
BACKOFF_BASE_SECONDS = 1.0

redis = Redis(host="localhost", port=6379, db=0, decode_responses=True)
logger = structlog.get_logger()


class AgentError(Exception):
    pass


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_PRICE_PER_TOKEN
        + output_tokens * OUTPUT_PRICE_PER_TOKEN
    )


@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPStatusError, httpx.TimeoutException),
    max_tries=MAX_ATTEMPTS,
    base=BACKOFF_BASE_SECONDS,
    jitter=backoff.full_jitter,
)
async def call_llm(prompt: str) -> dict:
    headers = {
        "authorization": f"Bearer {API_KEY}",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=BASE_TIMEOUT_SECONDS) as client:
        start = time.time()
        response = await client.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        duration = time.time() - start
        body = response.json()
        usage = body.get("usage", {})
        return {
            "body": body,
            "duration_seconds": duration,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
```

The retry decorator handles backoff, but it does not give you per-attempt logging. Wrap the call so each attempt is recorded:

```python
async def call_llm_logged(prompt: str, trace_id: str, attempt: int) -> dict:
    try:
        result = await call_llm(prompt)
        logger.info(
            "attempt_success",
            trace_id=trace_id,
            attempt=attempt,
            duration_ms=int(result["duration_seconds"] * 1000),
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=estimate_cost(
                result["input_tokens"], result["output_tokens"]
            ),
        )
        return result
    except Exception as exc:
        logger.warning(
            "attempt_failure",
            trace_id=trace_id,
            attempt=attempt,
            error_type=type(exc).__name__,
        )
        raise
```

Now the workflow, which computes totals across attempts:

```python
async def agent_workflow(prompt: str) -> dict:
    trace_id = str(uuid.uuid4())
    started = time.time()
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await call_llm_logged(prompt, trace_id, attempt)
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            wall_clock = time.time() - started
            logger.info(
                "trace_complete",
                trace_id=trace_id,
                attempts=attempt,
                wall_clock_ms=int(wall_clock * 1000),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=estimate_cost(
                    total_input_tokens, total_output_tokens
                ),
            )
            return {
                "body": result["body"],
                "attempts": attempt,
                "wall_clock_seconds": wall_clock,
                "total_cost_usd": estimate_cost(
                    total_input_tokens, total_output_tokens
                ),
            }
        except Exception:
            if attempt == MAX_ATTEMPTS:
                logger.error(
                    "trace_failed",
                    trace_id=trace_id,
                    attempts=attempt,
                    wall_clock_ms=int((time.time() - started) * 1000),
                )
                raise AgentError("max attempts exceeded")
    raise AgentError("unreachable")
```

Note that the retry decorator and the outer loop both count attempts. In production, pick one place to own retries. Having both means the decorator can retry inside a single loop iteration, and your attempt accounting will not match reality. The example above is deliberately structured so the loop is the only retry owner; remove the decorator or move its logic into the loop before deploying.

## Rate limiting with an atomic check

A naive check-then-set rate limiter has a race: two workers can both read "under limit" and both proceed. The fix is a single atomic operation. Redis supports this with a Lua script:

```lua
-- KEYS[1]: rate limit key
-- ARGV[1]: limit
-- ARGV[2]: window in milliseconds
local current = redis.call("INCR", KEYS[1])
if current == 1 then
    redis.call("PEXPIRE", KEYS[1], ARGV[2])
end
return current
```

Call it from Python and compare the returned count to the limit:

```python
RATE_LIMIT_SCRIPT = """
local current = redis.call("INCR", KEYS[1])
if current == 1 then
    redis.call("PEXPIRE", KEYS[1], ARGV[2])
end
return current
"""


async def check_rate_limit(user_id: str, limit: int, window_ms: int) -> bool:
    key = f"ratelimit:{user_id}"
    try:
        count = redis.eval(RATE_LIMIT_SCRIPT, 1, key, limit, window_ms)
    except RedisError:
        # Fail open or closed depending on your risk tolerance.
        return True
    return int(count) <= limit
```

Decide explicitly whether a Redis outage should fail open (allow traffic) or fail closed (reject traffic). Failing open protects availability but can produce a retry storm. Failing closed protects the provider but produces user-visible errors. There is no default that is right for every system.

## Caching

Caching identical prompts is the cheapest way to reduce token spend, but it introduces staleness. A short TTL bounds the staleness window:

```python
CACHE_TTL_SECONDS = 300


async def cached_agent(prompt: str) -> dict:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_key = f"agent:cache:{digest}"
    cached = redis.get(cache_key)
    if cached is not None:
        logger.info("cache_hit", cache_key=cache_key)
        return {"body": cached, "attempts": 0, "wall_clock_seconds": 0.0,
                "total_cost_usd": 0.0}
    result = await agent_workflow(prompt)
    redis.setex(cache_key, CACHE_TTL_SECONDS, str(result["body"]))
    return result
```

Two cautions. First, only cache prompts whose answers are genuinely deterministic for your use case; an agent that incorporates current data will serve stale answers. Second, cache keys derived from raw prompt text will miss on trivial whitespace differences. Normalize the prompt before hashing if that matters.

## Failure modes to watch for

**Retry budget that ignores backoff.** Setting `max_retries=3` and `timeout=5s` does not bound wall-clock time to 15 seconds. The backoff delays add to it, and jitter moves it again. Compute the bound explicitly and set the client timeout above it.

**Client timeout below orchestrator budget.** If the client gives up at 5 seconds while the orchestrator retries for 18, users see failures that the system would have recovered from. Either raise the client timeout or lower the orchestrator budget so the two agree.

**Double retry ownership.** A retry decorator inside a retry loop multiplies attempts: three outer attempts times three inner attempts is nine provider calls. Audit for this pattern.

**Cold starts overlapping the first retry.** A serverless function that takes time to initialize may have its first attempt time out before the function is warm. The retry then races the cold start. Provisioned concurrency or a longer first-attempt timeout addresses this, at a cost.

**Unbounded concurrency.** When many clients retry at once, the retries arrive together and can trigger provider throttling, which produces more retries. A concurrency gate — a rate limiter or a queue with a fixed number of workers — breaks the feedback loop.

**Verbose error payloads.** If error messages are generated by the model, they consume tokens. Cap the length of error text you request or store.

**Resource drift in long-running processes.** Track memory and open connections over time. A maximum request duration plus periodic process recycling bounds the damage.

## A decision checklist

Before deploying an agent, answer these:

- What is the per-attempt timeout, the backoff schedule, and the maximum attempts? What is the resulting worst-case wall-clock time?
- Is the client timeout strictly greater than that worst-case time?
- Who owns retries — the decorator, the loop, or both?
- Is every attempt logged with tokens, duration and error type, correlated by trace ID?
- Is there a concurrency gate in front of the provider?
- Is the rate limiter atomic under concurrent access?
- Is caching enabled, and what is the maximum staleness it can produce?
- What happens to in-flight requests when a worker is recycled?
- What is the alerting threshold on retry amplification and token amplification?

## Next 30 minutes

Pick one production agent endpoint and add three fields to its existing logs: `trace_id`, `attempt_number`, and `input_tokens`. Run it for an hour, then compute token amplification as total input tokens divided by the input tokens of the final successful attempt. That single number tells you how much of your token budget is going to retries, and it is the first step toward controlling it.
