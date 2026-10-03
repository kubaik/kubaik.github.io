# AI agents: circuit breakers are not optional

## Why AI agents need a different failure model

Most teams wire an AI agent into production the same way they wire a stateless microservice: an HTTP endpoint, a queue, a timeout, a retry policy. That mental model breaks down because agents fail in ways that stateless services do not.

A stateless service that cannot reach its database usually returns a 5xx. An agent that cannot reach its retrieval layer often returns a 200 with an empty or nonsensical body, because the LLM still generates text from whatever context it has. A load balancer counts that 200 as success and keeps routing traffic. Meanwhile the downstream dependency is degraded and nobody notices until a human downstream of the agent reports missing work.

The documented behavior of most HTTP clients, retry libraries and load balancers is to treat transport success as application success. Circuit breakers exist precisely to break that assumption, but they are usually skipped for agents because an agent looks like a simple HTTP client. It is not. An agent is a stateful pipeline: it accumulates conversation context, calls tools, and can fabricate a successful-looking answer while the real system is down.

## The state machine, and the extra state agents need

A conventional circuit breaker is a three-state machine: closed (traffic flows), open (fail fast), half-open (probe recovery). For agents, a fourth concern is worth modelling separately: semantic failure, sometimes called hallucinated success. This is not a breaker state in the classic sense, but a validation gate that feeds the breaker's failure counter.

| State | Trigger | Action | Typical latency |
|-------|---------|--------|-----------------|
| Closed | Failure rate below threshold | Route traffic normally | baseline + validation cost |
| Open | Failure rate at or above threshold within the window | Return cached result or a fast error | sub-millisecond to a few ms |
| Half-open | Cool-off elapsed | Send a limited number of probe requests | baseline + probe cost |
| Semantic failure | Response fails validation | Count as a failure, do not return to caller | validation cost only |

The semantic failure row is the one that distinguishes an agent breaker from a service breaker. A 200 response with a 12-character body where a structured decision was expected is a failure, even though no exception was raised. The breaker cannot see that unless you validate before you count a success.

### What to validate first

Start with cheap, deterministic checks and only escalate to expensive ones if they prove insufficient:

- **Length and shape.** A response shorter than a domain-defined minimum, or one that fails a schema check, is suspicious. This is the cheapest gate.
- **Required fields.** If the agent is supposed to return a decision, an identifier, or a structured object, validate presence and type.
- **Cross-checks against inputs.** A routing decision that references an underwriter ID not present in the input set is wrong regardless of how fluent it reads.
- **Latency.** A response returned far faster than the dependency's normal floor usually means the dependency was not actually consulted.

Embedding-based or LLM-judge validation is a later step. It is slower, costs money, and introduces its own failure modes. Teams commonly reach for it first and regret it.

## A worked example: sizing the window and threshold

Assume an agent that handles 20 requests per second. You want the breaker to react within roughly 15 seconds of a sustained failure, and you do not want a single bad request to trip it.

Step 1: choose the observation window. At 20 rps, 15 seconds is 300 requests. A sliding window of 300 calls is a reasonable starting point.

Step 2: choose the failure threshold. If you set the threshold at 2%, that is 6 failures within the window. Six failures is low enough to trip on a real outage and high enough that six unrelated one-off errors will not normally cluster inside 15 seconds. If your error budget is tighter, lower the percentage; if your traffic is bursty, raise the window size rather than the percentage so you keep statistical power.

Step 3: choose the cool-off. The cool-off should be at least as long as the downstream dependency's documented recovery time, and shorter than your tolerance for degraded service. If the dependency is a managed vector store with a multi-minute failover, a 60-second cool-off will reopen the breaker repeatedly during the failover. Prefer a cool-off that exceeds the dependency's known recovery window.

Step 4: choose the probe count. A single probe is fragile: one unlucky slow call reopens the breaker. Three sequential probes, each required to succeed, is a common compromise. This is the fix for the warm-up spike described below.

These numbers are illustrative. The point is that each one should be derived from a stated traffic rate and a stated recovery time, not copied from a blog post.

## A minimal implementation

The example below uses Python 3.11, FastAPI, and Redis. It is deliberately small. Note that it fixes a common bug in naive implementations: counters must be incremented on every call, including successes, and the failure-rate check must not divide by zero.

```python
import time
from functools import wraps

import redis.asyncio as redis
from fastapi import HTTPException


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold_pct: float = 2.0,
        window_size: int = 300,
        recovery_timeout: int = 60,
        min_response_chars: int = 50,
        probes_required: int = 3,
    ):
        self.name = name
        self.failure_threshold_pct = failure_threshold_pct
        self.window_size = window_size
        self.recovery_timeout = recovery_timeout
        self.min_response_chars = min_response_chars
        self.probes_required = probes_required
        self.redis = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
        self.prefix = f"cb:{name}"

    async def _incr(self, field: str, ttl: int) -> int:
        key = f"{self.prefix}:{field}"
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        value, _ = await pipe.execute()
        return int(value)

    async def _get_int(self, field: str) -> int:
        value = await self.redis.get(f"{self.prefix}:{field}")
        return int(value) if value is not None else 0

    async def _get_state(self) -> str:
        return await self.redis.get(f"{self.prefix}:state") or "closed"

    async def _set_state(self, state: str) -> None:
        await self.redis.setex(f"{self.prefix}:state", self.recovery_timeout, state)

    async def _record_success(self) -> None:
        await self._incr("calls", self.recovery_timeout)
        await self.redis.delete(f"{self.prefix}:probes")

    async def _record_failure(self) -> None:
        calls = await self._incr("calls", self.recovery_timeout)
        failures = await self._incr("failures", self.recovery_timeout)
        if calls >= self.window_size:
            rate = failures / calls * 100.0
            if rate >= self.failure_threshold_pct:
                await self._set_state("open")

    async def _probe_ok(self) -> bool:
        probes = await self._incr("probes", self.recovery_timeout)
        return probes >= self.probes_required

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            state = await self._get_state()

            if state == "open":
                cached = await self.redis.get(f"{self.prefix}:cache")
                if cached is not None:
                    return cached
                raise HTTPException(status_code=503, detail="circuit open")

            try:
                result = await func(*args, **kwargs)
            except Exception:
                await self._record_failure()
                raise

            body = str(result)
            if len(body) < self.min_response_chars:
                await self._record_failure()
                raise HTTPException(status_code=502, detail="response failed validation")

            if state == "half-open":
                if await self._probe_ok():
                    await self._set_state("closed")
                else:
                    await self._set_state("half-open")
            else:
                await self._record_success()

            await self.redis.setex(f"{self.prefix}:cache", 300, body)
            return result

        return wrapper
```

A few things worth calling out. The counters share a TTL with the window, so old failures age out without a separate sliding-window data structure. The half-open state is entered by writing `half-open` with the same recovery timeout, so a half-open breaker that never finishes probing will eventually revert. The cache is written only after validation passes, so a cached value is always one that satisfied the gate.

Wiring it into FastAPI:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
breaker = CircuitBreaker(name="loan-router-agent")


@breaker
async def route_loan_application(application: dict) -> dict:
    # agent logic here
    return {"underwriter_id": "u123", "ticket_id": "t456"}


@app.post("/route")
async def route_application(request: Request):
    data = await request.json()
    try:
        return JSONResponse(content=await route_loan_application(data))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
```

If you use a framework that already ships a breaker, prefer it over hand-rolling. In the Python ecosystem, a small decorator-based breaker library is usually enough. In the JVM ecosystem, Resilience4j is the mature choice. In Node, Opossum is widely used. The category matters more than the specific package; check that the library supports async calls and exposes state as a metric before adopting it.

## How to measure whether it is working

Do not trust any latency or error-rate numbers you did not measure yourself. Instrument these four things before and after adding the breaker:

1. **Per-call latency histogram**, labelled by breaker state. Compare p50 and p95 in closed state against the pre-breaker baseline. The delta is the validation and state-store cost.
2. **Failure-rate counter**, labelled by failure cause (transport error, validation failure, timeout). This is what tells you whether your length gate is doing real work or just flagging short valid answers.
3. **State-transition counter**, labelled by from-state and to-state. A breaker that flips more than a few times per hour is either mis-tuned or masking a dependency that is genuinely flapping.
4. **Cache hit rate in open state.** If the breaker opens and the cache is empty, every caller gets a 503. That may be correct, but you should know it before it happens.

To measure the cost of the state store specifically, run the agent function with the breaker decorator replaced by a no-op that performs the same Redis round trip. The difference between the two histograms is the breaker's overhead, separate from the agent's own cost.

For load testing, drive the agent with a synthetic failure injection: make the downstream dependency return empty bodies for 30 seconds and confirm the breaker opens within your target window, then confirm it closes after the dependency recovers. This is the only way to know your thresholds are right.

## Failure modes to plan for

**Warm-up spike after reopening.** When a breaker moves from open to half-open, the first probe can be slow if the dependency is still recovering. A single-probe design will reopen the breaker on that one slow call. Requiring several sequential probes and caching the first successful response for a short TTL smooths this out.

**Cache stampede.** If the breaker caches successful responses and the cache expires during a traffic spike, many callers may hit the dependency simultaneously. Randomising the cache TTL by a fraction of its nominal value spreads evictions out. A lock or single-flight mechanism around the cache fill also helps.

**State-store outage.** If Redis is unavailable, the breaker cannot track state. A local in-process fallback counter that periodically syncs to Redis keeps the system running with degraded detection. Decide explicitly whether the fallback state is closed (traffic flows, no detection) or open (traffic stops). For most agents, closed is the safer default, because the alternative is a total outage caused by a monitoring dependency.

**Lost conversation state.** If the breaker opens mid-conversation, the agent may lose context. Serialising the agent's state before the call and restoring it after is one option, but it adds latency and complexity. A simpler approach is to make the caller responsible for retrying with the same context once the breaker closes.

**False-positive validation.** A length gate will flag legitimately short answers. Keep a whitelist of short valid responses, or make the minimum length domain-specific rather than global.

**Rate-limit errors counted as failures.** If the downstream dependency rate-limits you, the breaker will open even though the dependency is healthy. Classify rate-limit responses separately and exclude them from the failure counter, or handle them with a separate backoff path.

## When a circuit breaker is the wrong tool

- **Stateless read-only lookups.** If the agent is a thin wrapper around a vector lookup with no side effects, a timeout and a retry budget are usually enough. A breaker adds state you do not need.
- **Agents that call arbitrary external tools.** If the agent can invoke a browser or a shell, a breaker around the agent call does not protect you from a hang inside a tool. Use a hard timeout and process isolation instead.
- **Sub-10 ms agents.** If the agent's own latency is below 10 ms, a Redis-backed breaker's round trip is a significant fraction of total latency. Consider an in-process breaker with a shared counter, accepting that state is per-replica.
- **Idempotent, stateless agents.** Retry with exponential backoff and jitter is simpler and sufficient when there is no state to corrupt.

## Monitoring and alerting

Expose breaker state as a gauge and transitions as a counter. Alert on two conditions: state has been open for longer than your tolerance, and transition rate exceeds a threshold you would expect from normal dependency flakiness. The first catches real outages; the second catches mis-tuning and flapping dependencies.

```python
from prometheus_client import Counter, Gauge, Histogram

CB_STATE = Gauge(
    "ai_agent_circuitbreaker_state",
    "Current breaker state (0=closed, 1=half-open, 2=open)",
    ["agent_name"],
)
CB_TRANSITIONS = Counter(
    "ai_agent_circuitbreaker_transitions_total",
    "Breaker state transitions",
    ["agent_name", "from_state", "to_state"],
)
CB_VALIDATION_FAILURES = Counter(
    "ai_agent_circuitbreaker_validation_failures_total",
    "Responses rejected by semantic validation",
    ["agent_name"],
)
CB_LATENCY = Histogram(
    "ai_agent_circuitbreaker_latency_seconds",
    "Latency per breaker call",
    ["agent_name", "state"],
)
```

A validation-failure counter that stays near zero is a sign your gate is too loose or your agent is healthy. A counter that spikes during every downstream incident is a sign it is doing its job.

## A decision checklist

Before adding a breaker to an agent, answer these:

1. Does the agent have side effects or maintain state? If no, a retry budget may suffice.
2. What is the downstream dependency's documented recovery time? Your cool-off must exceed it.
3. What does a semantically wrong response look like, and can you detect it with a cheap deterministic check?
4. What is your traffic rate, and what window size gives you enough calls to make a percentage threshold meaningful?
5. What should happen when the state store is unavailable: fail open or fail closed?
6. How will you measure the breaker's own overhead, separately from the agent's?
7. Who gets paged when the breaker has been open for longer than your tolerance?

## What to do in the next 30 minutes

Pick one agent in your codebase that calls an external dependency, and write down three things: its current p95 latency, the dependency's documented recovery time, and one deterministic check that a valid response must pass. If you cannot state all three, you do not yet have the information needed to tune a breaker, and gathering it is the actual first step.

## FAQ

**Should each agent have its own breaker?**

Yes. A shared breaker masks failures for one agent while another is healthy, and it makes state transitions impossible to attribute. One breaker per agent-dependency pair is the usual rule.

**How do I distinguish a hallucination from a legitimate short answer?**

You usually cannot, in general. That is why validation should be domain-specific: check required fields and cross-references against the input rather than relying on length alone.

**What if the dependency is healthy but rate-limiting me?**

Classify rate-limit responses separately and exclude them from the failure counter, or route them to a dedicated backoff path. Otherwise the breaker will open against a healthy service.

**Do I still need timeouts if I have a breaker?**

Yes. A breaker only reacts after failures are observed. Without a timeout, a hanging call never becomes a failure and the breaker never trips.
</body>
</invoke>
