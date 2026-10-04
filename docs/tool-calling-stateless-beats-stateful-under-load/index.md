# Tool calling: stateless beats stateful under load

## The failure mode this comparison addresses

Most tool-calling bugs that surface in production are not logic bugs. They are resource bugs. A handler that opens a connection per call, or one that shares a long-lived connection across requests, can both pass unit tests and then fail at scale for opposite reasons: the first exhausts ephemeral ports and file descriptors, the second exhausts pool slots and blocks every caller behind one slow upstream.

The pattern that breaks fastest in incident reports is usually the one that keeps a long-lived connection or context per request instead of creating a fresh one, or the one that opens a fresh connection per request with no bound on concurrency. Both mistakes look reasonable in isolation. The point of this article is to describe how each pattern behaves, how to measure the difference on your own workload, and how to decide.

The comparison is not "stateless is always better." It is that stateless and pooled tool calls fail in different ways, and you should know which failure mode your system can tolerate before you pick.

## Option A — stateless, request-scoped tool calls

A stateless tool call opens a new connection, performs the work, closes it, and keeps nothing. It is the classic open-transport, call, close cycle used by Python's `requests` or `httpx`, Node's `fetch`, and Go's `net/http` client.

```python
# Python 3.11 + httpx
import httpx

def call_tool(query: str) -> str:
    with httpx.Client(timeout=5.0) as client:
        r = client.post(
            "https://api.example.com/v1/search",
            json={"q": query},
            headers={"X-API-Key": "prod-key"}
        )
        r.raise_for_status()
        return r.json()["result"]
```

Under the hood this uses a fresh TCP connection per request. The kernel tears the socket down when the call returns, so the process does not accumulate file descriptors. Reusing a single client instance across many requests is a common micro-optimisation, but it changes the pattern: a client held open keeps the socket alive and can accumulate sockets in `TIME_WAIT`. A misconfigured session object that is never closed can balloon a container's open file count under sustained load.

Where stateless calls are strong:

- Simple mental model: one call, one socket, one result.
- Predictable latency: no connection reuse means no head-of-line blocking across calls.
- Failure isolation: a crashed upstream affects only the requests that hit it, not every caller sharing a pool.
- Fits the twelve-factor model: each request starts clean, so no leftover state can poison the next call.

Where stateless calls are weak:

- Every call pays handshake cost (TCP, and TLS if applicable).
- Under high request rates with non-trivial upstream latency, sockets pile up in `TIME_WAIT` and can exhaust the ephemeral port range.
- If the upstream bills per connection, cost scales linearly with request volume.

## Option B — pooled, session-reusing tool calls

A pooled tool call keeps a persistent session across invocations: a Redis client, a Postgres connection pool, a message-bus channel, or a WebSocket held open between calls. This is the pattern to reach for when you need low latency or when the upstream charges per connection.

```javascript
// Node 20 LTS + redis@4.6
import { createClient } from 'redis';

const client = createClient({
  url: 'redis://prod-redis:6379',
  socket: { reconnectStrategy: (retries) => Math.min(retries * 100, 5000) }
});

await client.connect();

async function batchLookup(keys: string[]) {
  return await client.mGet(keys);
}
```

The same idea applies to Postgres: a connection pool inside the process, or an external pooler in transaction pooling mode, keeps a bounded set of connections open and reuses them. The important property is that the pool is bounded. The size is decided at start-up, so the kernel's file descriptor limit becomes an explicit dial rather than an invisible ceiling.

Where pooled calls are strong:

- Lower latency for high-frequency operations such as rate-limit lookups or feature-flag checks.
- Reduced upstream load: one connection serves many calls instead of one per call.
- Lower cost on services that bill per connection or per concurrent request.

Where pooled calls are weak:

- Every shared resource needs a timeout, a backoff, and a circuit breaker. Without them, one slow upstream or one misbehaving query can pin the pool and starve every other request.
- Pool sizing is a tuning problem. Too small and latency spikes under load; too large and memory and upstream connection limits become the constraint.
- Leaks are silent. A pool that never releases a slot looks healthy in logs until callers start timing out.

## How to measure the difference on your own workload

Rather than trusting a table of numbers from someone else's environment, instrument both patterns and compare. The measurement is straightforward.

What to instrument:

- Latency distribution, not just the mean: record p50, p95, and p99.
- Error rate, split by cause (upstream 5xx, timeout, connection refused, pool exhausted).
- Open file descriptors per process, sampled every second.
- Socket state counts: `ss -tan state time-wait | wc -l` and `ss -s`.
- Pool saturation, if pooled: in-use slots divided by total slots.
- Upstream connection count, if the upstream exposes it.

How to run the comparison:

1. Fix the workload: a constant request rate, a fixed upstream latency, and a fixed upstream failure rate. If you cannot control upstream latency, note it and repeat the test at different times.
2. Run each pattern for at least ten minutes at twice your expected peak rate. Short runs miss `TIME_WAIT` accumulation and pool warm-up effects.
3. Record the metrics above for the whole run, not just the end.
4. Change one variable at a time: pool size, timeout values, concurrency limit. A two-variable change tells you nothing about which knob mattered.

The result you are looking for is not "which is faster." It is "which one fails first, and how." A stateless pattern that stays under your p99 target and never approaches the file descriptor limit is a safe default. A pooled pattern that holds p99 steady but saturates its pool at 80 percent of peak is a capacity problem waiting for a traffic spike.

## Worked example: sizing a pool from first principles

Suppose a service makes tool calls that take 20 ms of upstream time each, and the service must sustain 500 calls per second. The concurrency needed to sustain that rate is the arrival rate multiplied by the service time:

```
concurrency = 500 calls/s × 0.020 s = 10 concurrent calls
```

So a pool of 10 connections is the theoretical minimum, assuming zero variance. Real upstream latency varies, so size for the tail. If the p99 upstream latency is 100 ms, the same formula gives:

```
concurrency = 500 calls/s × 0.100 s = 50 concurrent calls
```

That is the number to start from. Add headroom for retries and for the fact that a pool slot is held for the entire call, including any client-side processing. A pool of 50 to 75 connections is a reasonable starting range for this workload. Below 50, callers queue during tail-latency events. Above 75, you are paying for connections that sit idle most of the time and consuming upstream connection budget.

This arithmetic is illustrative. Substitute your own measured service time and p99 latency.

The same arithmetic applies to the stateless pattern in reverse: if each call opens a socket and the upstream takes 100 ms at p99, then at 500 calls per second you have roughly 50 sockets open at any instant, plus every socket that has closed but is still in `TIME_WAIT`. `TIME_WAIT` duration is typically twice the maximum segment lifetime, which on Linux defaults to 60 seconds. That means the number of sockets in `TIME_WAIT` can be much larger than the number of in-flight sockets. This is the mechanism behind ephemeral port exhaustion, and it is why stateless patterns can fail even though they "hold no state."

## Failure modes to watch for

### Ephemeral port exhaustion

A stateless service running at high request rates with non-trivial upstream latency can accumulate sockets in `TIME_WAIT` faster than the kernel reclaims them. Symptoms: connection refused or cannot assign requested address errors, rising error rate under load, and `ss -s` showing a large `TIME_WAIT` count. Diagnostic:

```bash
watch -n 1 "ss -tan state time-wait | wc -l"
```

Mitigations include reducing upstream latency (which reduces in-flight sockets), capping client concurrency so you do not open more sockets than you can retire, and on the host side widening the ephemeral port range or enabling `tcp_tw_reuse`. Host-level tuning is a mitigation, not a fix; if you need it, your concurrency is probably higher than the workload justifies.

### Pool starvation behind a slow upstream

A pooled service with a fixed pool size will queue callers once every slot is in use. If one upstream call hangs, it holds its slot for the full timeout. With a pool of 50 and a 30-second timeout, 50 hung calls take the entire service down for 30 seconds. Symptoms: latency climbing in lockstep across all endpoints, pool saturation at 100 percent, and a flat upstream request rate. Mitigations: set a per-call timeout well below the pool's tolerance, add a circuit breaker so repeated failures stop consuming slots, and monitor pool saturation as a first-class metric.

A circuit breaker wraps pooled calls and trips after a configured failure rate, then stops issuing calls for a cooldown period. The shape of the configuration matters less than having one:

```javascript
const breaker = {
  failureThreshold: 0.5,      // trip at 50% failures
  minimumCalls: 20,           // do not trip on a tiny sample
  cooldownMs: 5000,           // wait before probing
  halfOpenCalls: 3            // probe with a few calls first
};
```

### Silent connection leaks

A pool that never releases a slot looks healthy until callers start timing out. Common causes: a code path that acquires a connection and returns early without releasing it, an exception thrown between acquire and release, or an idle timeout set to zero so connections are never reclaimed. Symptoms: pool saturation that does not recover after traffic drops, and a slow climb in open file descriptors. Mitigation: acquire and release in a `try/finally` block, set an explicit idle timeout, and alert on pool saturation above a threshold for more than a few minutes.

### HTTP/2 session accumulation

Clients that multiplex over HTTP/2 can keep idle sessions open longer than expected. If the session layer's idle timeout exceeds the kernel's TCP keepalive interval, idle sessions accumulate. Symptoms: open file descriptors climbing while request rate is flat. Mitigation: set explicit caps on the number of sessions and empty sessions in the client configuration, and verify with `ss -tan` that session counts track request rate.

## Decision checklist

Work through these in order. The first question that has a clear answer usually settles it.

1. **What is the upstream p99 latency?** If it is under roughly 10 ms and the service is in the same region or network, the handshake cost of stateless calls is a small fraction of total call time. Stateless is usually fine.
2. **What is the expected peak request rate?** At low rates (under a few hundred per second), neither pattern is likely to hit resource limits. At high rates, the arithmetic above determines whether you need a pool.
3. **Does the upstream bill per connection or per concurrent request?** If yes, pooling can reduce cost directly. If no, cost is not a differentiator.
4. **Can you bound concurrency in the stateless case?** If you can cap in-flight calls with a semaphore or a worker pool, stateless becomes safe at high rates because you control socket creation. If you cannot, stateless is a liability at high rates.
5. **Does the team have production experience with connection pools?** Pools require tuning, monitoring, and leak discipline. If the team has not operated one, the first incident will be expensive. Start stateless and add a pool when the latency or cost case is proven.
6. **What is the blast radius of a slow upstream?** With stateless calls, a slow upstream slows the requests that hit it. With a pool, a slow upstream can stall every caller. If the upstream is unreliable, stateless plus a circuit breaker is the safer combination.

## What to do in the next 30 minutes

Pick one service that makes tool calls to an external dependency. Run this against its staging endpoint under your current load test:

```bash
watch -n 1 "ss -tan state time-wait | wc -l"
```

Note the number. Then run the same load test with a client-side concurrency cap equal to your expected peak rate multiplied by the upstream p99 latency in seconds. Note the number again. If the second number is materially lower and your latency did not regress, you have found a cheap improvement: bound the concurrency rather than changing the pattern. If the number is unchanged and your p99 is already acceptable, leave the pattern alone and spend the time on something else.
