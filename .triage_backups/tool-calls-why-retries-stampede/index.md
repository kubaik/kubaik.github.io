# Tool calls: why retries stampede

## The gap between documented retry patterns and fan-out workloads

Most tool-use frameworks ship a retry helper. The documentation shows a clean loop: call the tool, catch the error, back off, try again. What the documentation does not address is what happens when hundreds of agent sessions hit the same tool at the same moment, all receive a 429, all read the same `Retry-After` header, and all sleep for the same duration. That is a thundering herd, and it is a common failure mode in production tool-use systems.

The gap is not that retry logic is wrong. It is that retry logic is *per-client* while the failure is *collective*. A rate limit is a shared resource constraint. A retry policy is a per-client decision. Those two things do not compose unless something sits between them.

This matters more as tool-use patterns shift. Sequential tool calls inside a single conversation loop are being replaced by fan-out: one orchestration call spawns N sub-agents, each with its own tool access. A single user request can generate dozens to hundreds of tool invocations in under two seconds. When one of those tools is a third-party API with a documented per-minute limit, per-client backoff stops working.

The confusing part is that the failure looks like a tool problem, not an architecture problem. Operators see 429s in the logs, add more retries, and the problem gets worse. The retries *are* the herd.

## How thundering herds form in tool-use systems

A thundering herd happens when a large number of processes simultaneously wait for the same event and then all wake up and contend for the same resource. In tool-use systems, the "event" is usually one of three things: a rate-limit reset, a circuit-breaker close, or a cache expiry.

The mechanism is straightforward. A typical tool wrapper uses exponential backoff with jitter — the standard recommendation:

```python
import random, time

def retry_with_backoff(fn, max_retries=5):
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            delay = min(2 ** attempt, 30)
            delay = delay * (0.5 + random.random() * 0.5)  # jitter
            time.sleep(delay)
    raise
```

This is fine for a single client. The jitter spreads retries across a window. But when many clients hit the limit at the same time, they all compute delays from the same distribution. The first retry wave arrives at roughly the same moment. If the limit is per-second, a spike appears at t+1s, another at t+2s, and so on. Jitter reduces the correlation; it does not eliminate it, because the clients still share a clock and a trigger.

The structural fix is coordination: a shared token bucket that all clients draw from, rather than independent retry timers. Redis with a Lua script is a common implementation because the token check and decrement happen atomically.

```lua
-- token_bucket.lua
-- KEYS[1] = bucket key
-- ARGV[1] = capacity, ARGV[2] = refill rate (tokens/sec), ARGV[3] = now (ms)
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local bucket = redis.call('HMGET', KEYS[1], 'tokens', 'last')
local tokens = tonumber(bucket[1]) or capacity
local last = tonumber(bucket[2]) or now
local delta = math.max(0, now - last) / 1000.0
local refill = delta * rate
tokens = math.min(capacity, tokens + refill)
if tokens < 1 then
  redis.call('HMSET', KEYS[1], 'tokens', tokens, 'last', now)
  return 0
end
tokens = tokens - 1
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'last', now)
redis.call('EXPIRE', KEYS[1], 60)
return 1
```

The difference is structural. With independent backoff, each client decides when to retry. With a shared bucket, the system decides. That shift separates tool-use patterns that scale from ones that collapse under load.

## A working tool wrapper

The pattern below routes every tool call through a shared limiter, and retries are scheduled by the limiter rather than by the caller.

```python
import asyncio
import time
import redis.asyncio as redis

class ToolClient:
    def __init__(self, redis_url, tool_name, capacity=100, rate=50):
        self.redis = redis.from_url(redis_url)
        self.tool_name = tool_name
        self.capacity = capacity
        self.rate = rate
        self._script = None

    async def _load_script(self):
        if self._script is None:
            with open('token_bucket.lua') as f:
                self._script = self.redis.register_script(f.read())

    async def acquire(self, timeout=10.0):
        await self._load_script()
        key = f"ratelimit:{self.tool_name}"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            now_ms = int(time.time() * 1000)
            ok = await self._script(
                keys=[key],
                args=[self.capacity, self.rate, now_ms]
            )
            if ok == 1:
                return True
            await asyncio.sleep(0.05)  # 50ms poll
        raise TimeoutError(f"rate limit wait exceeded for {self.tool_name}")

    async def call(self, fn, *args, **kwargs):
        await self.acquire()
        try:
            return await fn(*args, **kwargs)
        except RateLimitError:
            # server-side limit hit despite our bucket — back off and retry once
            await asyncio.sleep(1.0)
            await self.acquire()
            return await fn(*args, **kwargs)
```

The 50ms poll interval is a deliberate tradeoff. A shorter interval increases Redis load; a longer one adds latency to every call that has to wait. The right value depends on the tool's budget. If a tool is allowed 50 requests per second, a 50ms poll means at most 20 waiters per second per instance. If a tool is allowed 5 requests per second, a 200ms poll is more appropriate. Work the number out from the tool's documented limit rather than copying a constant.

The second piece is making sure fan-out does not spawn hundreds of sub-agents that all try to acquire at once. Use a semaphore at the orchestration layer:

```python
async def run_subagents(tasks, max_concurrent=20):
    sem = asyncio.Semaphore(max_concurrent)
    async def bounded(task):
        async with sem:
            return await task()
    return await asyncio.gather(*[bounded(t) for t in tasks])
```

Without the semaphore, hundreds of coroutines all wait on the same Redis key, which is itself a load problem. With it, at most `max_concurrent` are waiting, and the rest queue in memory where they belong.

## Measuring the effect instead of trusting a table

Published latency and error-rate tables for this kind of change are usually not reproducible, because they depend on the downstream API, the client count, and the burst shape. Measure on your own system instead. Four numbers tell the story:

- **Retry ratio**: retries divided by total tool calls, per tool. Instrument this in the tool wrapper, not the HTTP client, because the HTTP client does not know which tool a request belongs to.
- **429 rate**: rate-limit responses divided by total calls. If this is above roughly 1%, the retry policy is likely contributing to the problem rather than absorbing it.
- **p50 and p99 latency**: herds show up as a tail that grows faster than the median as load increases. A system without a herd has p99 that scales roughly linearly with load.
- **Tool success rate**: successful calls divided by attempted calls, after retries.

To compare configurations, run the same workload shape against each: fixed request count, fixed fan-out width, fixed tool budget. Record the four numbers above for (a) independent backoff without jitter, (b) independent backoff with jitter, (c) shared token bucket, and (d) shared token bucket plus a concurrency semaphore. The interesting result is usually that the shared bucket slightly *increases* p50, because every call now takes a Redis round trip, while dropping p99 sharply because the tail is no longer dominated by retry storms. Whether that trade is correct depends on whether your users feel the tail.

## Failure modes that are easy to miss

**Retry amplification across layers.** The tool wrapper retries. The HTTP client retries. The load balancer retries. The orchestrator retries. Each layer believes it is being resilient; the request count multiplies. Three retries at each of four layers is 81 requests for every original request. When the downstream service is already struggling, that is how a blip becomes an outage.

The fix is to retry at exactly one layer — usually the outermost one that has context about the overall operation — and disable retries everywhere else. In `httpx`, that means `transport=httpx.AsyncHTTPTransport(retries=0)`. In `requests`, it is `session.mount('https://', HTTPAdapter(max_retries=0))`. Check your load balancer's retry settings too; many default to retrying idempotent methods.

**Cache stampede on tool results.** Caching tool outputs is worthwhile for idempotent tools, but a cold cache means every concurrent request misses and calls the tool. The fix is single-flight: only one request calls the tool, the rest await the same result.

```python
import asyncio

class SingleFlight:
    def __init__(self):
        self._inflight = {}

    async def do(self, key, coro_factory):
        if key in self._inflight:
            return await self._inflight[key]
        fut = asyncio.create_task(coro_factory())
        self._inflight[key] = fut
        try:
            return await fut
        finally:
            self._inflight.pop(key, None)
```

Note the limitation: this implementation shares the result among callers within one process. Across processes, single-flight needs a distributed lock or a short-lived cache entry written by the winner.

**Circuit-breaker flapping.** When a tool is degraded, a circuit breaker opens and waiting requests fail fast. Then the breaker half-opens, sends one request, and if that request happens to hit a slow path, it closes again. The tool oscillates between open and closed, and every close sends a burst of traffic. Use a minimum open duration (commonly 30 to 60 seconds) and require several consecutive successes before closing, not one.

**Retrying non-idempotent tools.** If a tool sends an email, creates a record, or charges a card, a retry can duplicate the effect. The fix is an idempotency key passed to the tool plus a server-side dedup window. If the tool does not support idempotency keys, do not retry it automatically — surface the failure to the caller.

**Counting retries as separate requests in metrics.** If a tool dashboard shows far more requests per minute than user-facing operations, the difference is retries. That is a cost multiplier on any per-request pricing and it is invisible unless retries are instrumented separately.

**Ignoring `Retry-After`.** Many APIs send it and many clients ignore it in favour of their own backoff. If the server says wait 30 seconds, wait 30 seconds. A jittered 2-second retry only adds load.

**No timeout discipline.** A tool that takes 30 seconds to time out is worse than one that fails fast. Every waiting caller holds a connection, a coroutine, and memory. Set aggressive timeouts — 2 to 5 seconds for most tools — and fail fast.

**Sharing one Redis instance for rate limiting and caching.** Under memory pressure, Redis may evict rate-limit keys, and clients will suddenly believe they have full capacity. Use a separate instance or at least a separate logical database with a `noeviction` policy for rate-limit keys.

## Choosing between token bucket and leaky bucket

| Property | Token bucket | Leaky bucket |
|---|---|---|
| Burst behaviour | Allows bursts up to bucket capacity | Emits at a constant rate |
| Typical fit | Third-party APIs with per-minute quotas | Downstream with no burst tolerance |
| Implementation | Refill on read, decrement on acquire | Queue with fixed drain rate |
| Failure shape under overload | Short bursts, then rejection | Growing queue depth and latency |

For most third-party APIs with documented per-minute limits, a token bucket with capacity equal to the per-minute limit and a refill rate equal to the limit divided by 60 is the usual starting point. Leaky bucket is the better choice when the downstream service degrades under any burst at all.

## When this approach is the wrong choice

If tool calls are low-volume — sustained rates well under ten per second — none of this matters. Independent backoff with jitter is fine, and the Redis dependency is pure overhead. The herd needs enough simultaneous callers to form; below roughly 20 to 30 concurrent callers on the same tool, it usually does not.

If the tools are internal and you control the rate limits, fix the rate limits instead. A shared bucket is a workaround for an external constraint you cannot change.

If the workload is genuinely bursty with long idle periods — a batch job that runs once an hour — a token bucket wastes capacity during idle periods and throttles during bursts. A queue that processes work at a controlled rate is a better fit; the batch simply takes longer.

Finally, if you are using a managed agent platform that already handles rate limiting per tool, check its documentation before reimplementing this. Several platforms expose per-tool rate-limit configuration that does what the code above does without the Redis dependency.

## Frequently asked questions

**How do I know whether my system has a thundering herd?**

Look at the retry ratio over time. If it spikes in sync with traffic spikes, a herd is likely. If it is flat, it is not. The second signal is p99 latency: herds show up as a long tail that grows faster than p50 as load increases.

**What is the difference between a thundering herd and a retry storm?**

A retry storm is when retries themselves generate enough load to keep a service down. A thundering herd is when many clients wake up simultaneously and contend for the same resource. They often co-occur — a herd causes a storm — but the fixes differ. Retry storms are mitigated by capping retry counts and adding jitter. Herds are mitigated by coordination, usually a shared token bucket or a queue.

**How many retries is too many?**

Three is a practical maximum for most tools. Beyond that, a persistent failure is more likely than a transient one, and the retries only add load. If a tool fails three times in a row with the same error, fail the operation and let the caller decide. Transient network errors are the exception, where five retries with a cap of around ten seconds is reasonable.

**Does the shared bucket add latency?**

Yes, by the cost of one Redis round trip per call — typically a small single-digit number of milliseconds on a local network. The benefit is a much shorter tail. Whether that trade is correct depends on whether your users experience the tail.

## What to do next

Add a retry counter to your tool wrapper — one increment per retry, exported as a metric labelled by tool name — and let it run for a day. Then look at the ratio of retries to total calls for each tool. Start with the tool that has the highest ratio: that is where the leverage is. If the ratio is under 2%, the current policy is probably fine. If it is above 5%, the shared token bucket described above is the next step, applied to that one tool first.
