# Designing APIs for High-Latency Satellite Links

A service that behaves under a 20 ms round trip can fail in a specific, repeatable way when the round trip becomes 230 ms and the link flaps on a schedule. This article covers the failure modes that appear when a terrestrial-first stack is moved onto a satellite-primary link, and the concrete changes that address them: a shared circuit breaker, a retry budget, a failover gateway, and instrumentation that distinguishes "the link is bad" from "the service is bad."

The numbers used below are illustrative unless labelled as documented defaults. Where a figure appears, the arithmetic or the measurement method is shown so it can be reproduced against a real deployment.

## Why high-latency links break terrestrial-first stacks

Satellite links have three properties that matter to application code, and they interact:

1. **Elevated median RTT.** A geostationary link is roughly 600 ms round trip; low-earth-orbit constellations are typically far lower, but still well above a terrestrial path in the same region. The exact figure depends on the constellation, the ground station, and the time of day, and should be measured rather than assumed.
2. **Bursty loss and bufferbloat.** Satellite links tend to absorb bursts into a large queue, which converts packet loss into latency spikes. A link can be "up" and still be unusable for seconds at a time.
3. **Scheduled maintenance.** Some operators publish maintenance windows. A window is not a link failure; the link is up, but the path may flap repeatedly for the duration.

The failure mode that follows is not "requests are slow." It is **retry amplification**. A client that times out after 5 seconds and retries three times turns one slow request into four. If the retry is triggered by a cron job or a queue worker, the amplification is unbounded. The database connection pool, the upstream service, and the Redis instance all see traffic proportional to the retry count, not the request count.

The rest of this article builds a minimal async API gateway that survives this pattern, then adds instrumentation to prove it.

## Prerequisites and what you will build

- Python 3.12
- FastAPI (any recent release; pin an exact version in your lockfile)
- Redis (any release with `SET ... EX` and streams support; 7.x is a reasonable floor)
- PostgreSQL 16
- `httpx` for the outbound client

You do not need a satellite dish to reproduce the failure mode. A `tc netem` delay on a loopback interface, or a proxy that injects latency, is sufficient to exercise every code path below.

By the end you will have:

- An API that degrades gracefully when an upstream stalls
- A retry budget that is aware of the maintenance window
- A failover gateway that hides the link switch from the application
- Metrics that separate link health from service health
- A CI job that exercises both paths before a deploy

## Step 1 — environment and hardware choices

### Router and edge node

The router's job is to keep the application unaware that the link changed. Two features matter:

- **DSCP marking** so that bulk API traffic and interactive traffic do not share a queue. Without it, bufferbloat on the satellite link adds queuing delay to everything.
- **FQ-CoDel or an equivalent queue discipline** on the WAN interface, so a single bulk transfer cannot starve interactive requests.

A small always-on edge node (a single-board computer is sufficient) can act as a failover gateway, bridging the satellite link and a local terrestrial link so that the application sees one stable next hop. A cellular modem on a separate APN is the usual backup path; it is sized for control-plane traffic and failover, not for bulk transfer.

### Network layout

```
Client → FastAPI (container) → Redis → PostgreSQL
                     ↓
                Edge gateway (DSCP marking, FQ-CoDel)
                     ↓
        ┌────────────┴────────────┐
   Satellite link            Cellular backup
   (high RTT)                (lower RTT, metered)
```

The application talks to the edge gateway only. The gateway decides which WAN path is active.

### Install the stack

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv redis-server postgresql-16
python3.12 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn[standard] redis httpx
```

Pin exact versions in `requirements.txt` before deploying. Distribution packages often lag the upstream release; check `redis-server --version` after install and confirm it matches what your metrics code expects.

## Step 2 — core implementation

### Minimal FastAPI app with a bounded upstream call

The first change is to stop treating a timeout as an error to retry blindly. A timeout on a high-latency link is information: the path is slow, and retrying immediately will make it slower.

```python
from fastapi import FastAPI, Request
from redis.asyncio import Redis
from contextlib import asynccontextmanager
import httpx, logging

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = Redis(host="localhost", port=6379, decode_responses=True)
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, connect=2.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=25),
        transport=httpx.AsyncHTTPTransport(retries=0),
    )
    yield
    await app.state.client.aclose()
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)

@app.post("/immunize")
async def immunize(request: Request):
    data = await request.json()
    client: httpx.AsyncClient = request.app.state.client
    try:
        resp = await client.post("http://10.0.0.20/api/record", json=data)
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        logging.warning("primary path timeout")
        resp = await client.post("http://192.168.1.100/api/record", json=data)
        resp.raise_for_status()
        return resp.json()
```

Note `retries=0` on the transport. Retries are handled explicitly, once, and only to the fallback path. Letting the transport retry silently multiplies the load on a link that is already struggling.

### A shared circuit breaker in Redis

In-process breaker state is a trap: when the container restarts (which is exactly what happens under memory pressure during a flap), the breaker resets and the retry storm restarts. Keep the state in Redis so every replica shares it.

```python
from redis.asyncio import Redis

class CircuitBreaker:
    def __init__(self, redis: Redis, key: str = "cb:upstream_primary"):
        self.redis = redis
        self.key = key
        self.fail_key = f"{key}:failures"

    async def is_open(self) -> bool:
        return await self.redis.get(self.key) == "open"

    async def record_failure(self, window_s: int = 60, threshold: int = 5, open_s: int = 300):
        count = await self.redis.incr(self.fail_key)
        if count == 1:
            await self.redis.expire(self.fail_key, window_s)
        if count >= threshold:
            await self.redis.set(self.key, "open", ex=open_s)
            await self.redis.delete(self.fail_key)

    async def record_success(self):
        await self.redis.delete(self.fail_key)
```

Two details matter. First, the failure counter has its own TTL, so a slow trickle of failures does not accumulate forever. Second, the open state has an explicit TTL, so the breaker closes on its own rather than requiring an operator.

### Connection pool sizing for high RTT

A pool sized for a 20 ms round trip is too large for a 230 ms round trip. The relevant quantity is concurrency, not pool size: with RTT `R` and per-request service time `S`, the number of in-flight requests a single connection can sustain is roughly `1 / (R / S)`. Raising `R` lowers throughput per connection, so a fixed pool drains more slowly and queueing time grows.

Rather than guess, measure. Instrument the pool and the request duration, then compare:

```python
import time

@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    await request.app.state.redis.xadd(
        "stream:api_latency", {"ms": str(int(elapsed * 1000))}
    )
    return response
```

Run the service against a link with injected latency and against a low-latency link, and compare the p95 of `api_latency` at a fixed request rate. If the p95 grows faster than the RTT increase, the pool is too large and requests are queueing behind each other.

## Step 3 — handle the maintenance window

### Time-bounded breaker

If the operator publishes a maintenance window, encode it. During the window, the primary path is expected to be unreliable, so the breaker should be more willing to open and the fallback path should be preferred.

```python
from datetime import datetime, timezone

def is_maintenance_window(start_hour: int = 2, end_hour: int = 8) -> bool:
    now = datetime.now(timezone.utc)
    return start_hour <= now.hour < end_hour
```

Used in the endpoint:

```python
if is_maintenance_window() and await cb.is_open():
    resp = await client.post("http://192.168.1.100/api/record", json=data)
else:
    resp = await client.post("http://10.0.0.20/api/record", json=data)
```

The window is a hint, not a guarantee. A link can fail outside the window, and the window can pass without incident. The breaker still governs; the window only changes the preference order.

### Burst detection with a capped Redis stream

A stream of latency samples lets you detect a burst without a separate time-series database. Cap the stream length or it will grow until Redis runs out of memory.

```python
pipeline = app.state.redis.pipeline()
pipeline.xadd("stream:api_latency", {"ms": str(latency_ms)})
pipeline.xtrim("stream:api_latency", maxlen=1000, approximate=True)
pipeline.xlen("stream:api_latency")
_, _, length = await pipeline.execute()
```

`approximate=True` makes trimming cheaper. The trade-off is that the stream may hold slightly more than `maxlen` entries, which is fine for burst detection.

### Failover gateway

The gateway's job is to switch WAN paths without changing the application's next hop. On a Linux edge node, this is a routing and health-check problem, not an application problem:

```bash
# Health probe: mark the primary path down after N consecutive failures
#!/bin/sh
PRIMARY=10.0.0.1
FAILS=0
while true; do
  if ping -c 1 -W 2 "$PRIMARY" >/dev/null 2>&1; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 3 ]; then
      ip route replace default via 192.168.1.1
    fi
  fi
  sleep 5
done
```

Three consecutive failures at a 5-second interval means a 15-second detection time. Shorter intervals detect faster but generate more probe traffic; the right value depends on how much downtime the application can tolerate.

If the gateway is a managed router rather than a Linux box, the equivalent is a health-check rule that changes the default gateway. The mechanism differs; the requirement does not.

## Step 4 — observability and tests

### Metrics that separate link health from service health

Three signals are enough to tell the two apart:

- **Request duration**, bucketed. A p95 that tracks the link RTT is a link problem. A p95 that grows while the RTT is flat is a service problem.
- **Upstream error count**, by path (primary vs. fallback). This tells you which path is failing.
- **Active connections**, by backend. This catches pool exhaustion before it becomes an outage.

With Prometheus and the FastAPI instrumentator:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

The default histogram buckets are tuned for sub-second latency. On a link with a 230 ms median, most requests land in the first few buckets, which wastes resolution. Override the buckets to match the expected distribution:

```python
Instrumentator(
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
).instrument(app).expose(app)
```

### A synthetic test that exercises both paths

The test below forces the breaker open and confirms the fallback path is used. It does not require a real satellite link; the breaker is the unit under test.

```python
# tests/test_paths.py
import pytest
import httpx

@pytest.mark.asyncio
async def test_fallback_used_when_breaker_open():
    async with httpx.AsyncClient(timeout=3.0) as client:
        r1 = await client.post("http://127.0.0.1:8000/immunize", json={"pid": "123"})
        assert r1.status_code == 200

        await client.post("http://127.0.0.1:8000/admin/force-breaker?open=1")

        r2 = await client.post("http://127.0.0.1:8000/immunize", json={"pid": "456"})
        assert r2.status_code == 200
        assert r2.headers.get("x-path") == "fallback"
```

The `x-path` header is set by the endpoint so the test can assert which path was taken. Asserting only on the status code would pass even if the fallback were never reached.

For the network path itself, a separate test injects latency with `tc netem` and asserts that the p95 stays within a budget. That test is slower and belongs in a nightly job, not on every push.

### A dashboard that answers one question

The dashboard should answer: "is the link bad, or is the service bad?" Three panels:

| Panel | Query | Interpretation |
|---|---|---|
| Request duration p95 | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` | Tracks link RTT → link problem |
| Upstream errors by path | `sum(rate(upstream_errors_total[5m])) by (path)` | Primary rising, fallback flat → link problem |
| Redis connected clients | `redis_connected_clients` | Rising toward `maxclients` → pool exhaustion |

If the first two move together, the link is the cause. If only the first moves, look at the service. If the third moves, the breaker is not shedding load fast enough.

## Failure modes and how to detect them

**Retry amplification.** A client retries on timeout, and each retry adds load to a link that is already saturated. Detect it by comparing request count to upstream call count. If the ratio exceeds the configured retry budget, the budget is not being enforced.

**Breaker thrashing.** The breaker opens and closes repeatedly as the link flaps. Each cycle admits a burst of traffic. Detect it by counting breaker transitions per hour; a healthy breaker transitions rarely. Fix it by lengthening the open duration and adding jitter to the close.

**Pool exhaustion under fallback.** When the primary path fails, all traffic moves to the fallback, which may have a smaller pool. Detect it by tracking active connections per backend. Fix it by sizing the fallback pool for the full load, or by shedding load when the breaker is open.

**Silent fallback.** The fallback path is used for weeks and nobody notices, so the primary path is never exercised and its failure is discovered during an incident. Detect it by alerting on the fallback path being used continuously for more than a threshold. Fix it by periodically forcing a failover in a canary environment.

**Clock skew across replicas.** The maintenance window is computed from the local clock. If replicas disagree, some enter the window early and some late. Detect it by comparing `datetime.now(timezone.utc)` across replicas. Fix it by running an NTP client and alerting on drift.

## Decision checklist

Before moving a service to a satellite-primary link:

- [ ] Is the retry budget bounded, and is it enforced at the client rather than the transport?
- [ ] Is the circuit breaker state shared across replicas?
- [ ] Does the breaker have both a failure threshold and an open duration with a TTL?
- [ ] Is the connection pool sized by measurement rather than by default?
- [ ] Does the fallback path have enough capacity for the full load?
- [ ] Is the maintenance window encoded, and does it change preference rather than override the breaker?
- [ ] Are request duration, upstream errors by path, and active connections all instrumented?
- [ ] Does a test assert which path was used, not just that the request succeeded?
- [ ] Is there an alert for the fallback path being used continuously?
- [ ] Is clock skew across replicas monitored?

## FAQ

**Does this require a specific satellite provider?**
No. The pattern applies to any high-latency, bursty link with a published or predictable maintenance window. The provider-specific parts are the maintenance window and the failover mechanism.

**Can this be done without Redis?**
Yes, but the breaker state must live somewhere shared. A database table with a row-level lock works. The important property is that every replica sees the same state, so a container restart does not reset the breaker.

**What if the fallback path is also down?**
The breaker should have a terminal state: after N consecutive failures on both paths, the service returns a fast error rather than retrying. A fast error is better than a slow timeout, because it frees the connection pool.

**How do I size the retry budget?**
Start with the request rate and the acceptable amplification factor. If the service handles 1000 requests per second and the budget allows 1.1× amplification, the upstream can see at most 1100 calls per second. Instrument both counts and alert when the ratio exceeds the budget.

**Does DSCP marking help if the provider ignores it?**
It helps on the local egress queue, which is where bufferbloat usually originates. If the provider ignores DSCP, the marking still prevents your own bulk traffic from starving your own interactive traffic.

## One action for the next 30 minutes

Open the module that makes outbound HTTP calls and find every place a timeout triggers a retry. For each one, write down the maximum number of calls that a single client request can produce. If that number is greater than one, add a counter that increments on each retry and a log line that records the request ID. Deploy it, then look at the counter after the next slow period. The number you see is your amplification factor, and it is the first thing to fix.
