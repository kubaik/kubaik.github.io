# Tool calling patterns: patterns that survive load vs

## Why tool call patterns decide whether an outage stays small

A tool call is a network call to something you do not control. The downstream service can garbage-collect, restart, throttle, or change its latency profile without warning. The pattern wrapped around that call determines whether a brief downstream hiccup produces a few slow requests or a self-inflicted denial-of-service.

The failure mode is well documented and easy to reproduce. A client retries on 5xx without a budget. The downstream is already saturated, so each retry adds load instead of relieving it. Latency rises, more requests cross the client timeout, more retries fire. The retry loop becomes the outage. CPU on the downstream climbs, connection pools drain, and the incident outlives the original trigger by an order of magnitude.

Two broad families of patterns exist:

- **Naive synchronous calls**: one call per request, a fixed timeout, retries on failure with no ceiling, no circuit breaker, no concurrency limit, no batching. This is the default shape of hand-rolled HTTP clients and simple RPC stubs.
- **Shaped calls**: retry budgets, backoff with jitter, circuit breakers, bulkheads, rate limits, and batching, all applied at the call site or in a shared client.

Naive calls look clean in review because the happy path is short. Shaped calls look noisy because the failure path is explicit. The trade-off only becomes visible under tail latency, downstream degradation, or a traffic spike. That is the comparison this article works through: what each pattern does, how to measure the difference, and how to choose.

## Pattern A: naive synchronous calls

The anatomy is minimal:

1. One thread, goroutine, or event-loop tick issues the call.
2. The call blocks or yields until a response arrives or a timeout fires.
3. On a non-2xx response, the client retries on a fixed interval or a simple exponential backoff with no ceiling.
4. No circuit breaker, no bulkhead, no batching, no per-caller rate limit.

A minimal Python example using `httpx`:

```python
import httpx

def call_payment_gateway(order_id: str) -> dict:
    url = f"https://payments.internal/v2/charge/{order_id}"
    response = httpx.post(url, json={"amount": 999}, timeout=5.0)
    response.raise_for_status()
    return response.json()
```

This is readable and matches the happy path exactly. It also has three properties that cause damage under load:

- **No retry budget.** The caller issues a new attempt on every 5xx, even when the downstream is saturated. Retries add load precisely when the downstream can least absorb it.
- **No backpressure.** The calling thread blocks for the full timeout. Under a 5xx storm the handler pool drains and the service stops accepting new work.
- **No failure-path observability.** Logs show `timeout` or `retrying`, but the aggregate retry rate is rarely exported as a metric, so the storm is invisible until it is an incident.

Pattern A is defensible in a narrow set of cases:

1. **In-memory downstreams with sub-10 ms p99.** A cache lookup against a local or same-AZ in-memory store typically returns fast enough that a blocking call never accumulates. The failure mode is a miss, not a stall.
2. **Low-volume internal calls.** A cron job or an admin endpoint running at a few requests per minute has no meaningful concurrency to exhaust.
3. **Prototypes.** Code that will be replaced before it takes production traffic.

The failure mode to watch for is a downstream that occasionally becomes slow rather than unavailable. A cron job that calls an endpoint which normally returns in 50 ms but sometimes takes 8 seconds will hold its connection for the full 8 seconds. If the scheduler allows overlapping runs, several instances pile up, each holding a connection. The connection pool empties and the next invocation fails for reasons unrelated to the original slowdown. This is the classic pattern: a latency change in one dependency propagates into a resource-exhaustion failure in the caller.

## Pattern B: shaped calls

Shaped calls treat the tool call as a distributed-systems problem rather than a function invocation. The components:

- **Retry budget.** A ceiling on total attempts per logical call, for example three attempts including the first. A budget is not the same as "retry three times after failure," which is four attempts.
- **Backoff curve with jitter.** Exponential growth with a cap and randomized jitter, for example 100 ms, 200 ms, 400 ms, capped at 5 s, with 20 percent jitter. Jitter prevents synchronized retries from many clients arriving at the same instant.
- **Circuit breaker.** After N failures within a window, the breaker opens and rejects calls immediately for a cooldown period. This converts a slow downstream into a fast failure, which is usually the better outcome for the caller.
- **Bulkhead.** A cap on concurrent in-flight calls to a given downstream. Excess calls fail fast instead of queueing and consuming threads.
- **Rate limit.** A per-caller or per-endpoint ceiling that prevents one tenant or one code path from consuming the whole capacity of a shared dependency.
- **Batching.** Grouping calls that target the same downstream to reduce connection churn and per-call overhead.

A Go example combining a retryable client, a circuit breaker, and a weighted semaphore as a bulkhead:

```go
import (
    "bytes"
    "fmt"
    "io"
    "net/http"
    "time"

    "github.com/hashicorp/go-retryablehttp"
    "github.com/sony/gobreaker"
    "golang.org/x/sync/semaphore"
)

var (
    sem = semaphore.NewWeighted(50) // bulkhead: max 50 concurrent calls

    cb = gobreaker.NewCircuitBreaker(gobreaker.Settings{
        Name:        "payment-gateway",
        MaxRequests: 1,
        Interval:    30 * time.Second,
        Timeout:     10 * time.Second,
        ReadyToTrip: func(counts gobreaker.Counts) bool {
            return counts.ConsecutiveFailures >= 5
        },
    })

    client = retryablehttp.NewClient()
)

func init() {
    client.RetryMax = 2 // 3 attempts total
    client.RetryWaitMin = 100 * time.Millisecond
    client.RetryWaitMax = 5 * time.Second
    client.Backoff = retryablehttp.LinearJitterBackoff
}

func callWithSafety(orderID string) ([]byte, error) {
    if !sem.TryAcquire(1) {
        return nil, fmt.Errorf("bulkhead full")
    }
    defer sem.Release(1)

    req, err := retryablehttp.NewRequest(
        "POST",
        fmt.Sprintf("https://payments.internal/v2/charge/%s", orderID),
        bytes.NewReader([]byte(`{"amount":999}`)),
    )
    if err != nil {
        return nil, err
    }
    req.Header.Set("Content-Type", "application/json")

    respAny, err := cb.Execute(func() (any, error) {
        return client.Do(req)
    })
    if err != nil {
        return nil, err
    }
    resp := respAny.(*http.Response)
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}
```

The behavior under stress is what matters:

- **Downstream GC pause.** The backoff curve spaces retries so the downstream has a chance to recover before the next attempt arrives.
- **5xx storm.** The circuit breaker opens after the configured consecutive failures, cutting traffic to zero until the cooldown expires and a probe succeeds.
- **Traffic spike.** The bulkhead rejects excess calls immediately, so the caller's thread pool does not drain.

Shaped calls are the right default for public APIs, payment flows, multi-tenant systems, and any downstream whose latency exceeds roughly 50 ms or is subject to garbage collection pauses.

## How to measure which pattern you actually have

Benchmark tables are easy to fabricate and hard to trust. What is useful is knowing exactly what to instrument and what to compare, so the numbers come from your own system.

Instrument these four signals on the client side:

1. `call_duration_seconds` as a histogram, labelled by downstream and by attempt number. The attempt label is what separates "the downstream is slow" from "our retries are slow."
2. `retry_attempts_total` as a counter, labelled by downstream and by reason (timeout, 5xx, connection error).
3. `circuit_breaker_state` as a gauge: 0 closed, 1 half-open, 2 open.
4. `inflight_calls` as a gauge, plus a counter for bulkhead rejections.

On the downstream side, capture p50, p95, and p99 latency, error rate, and connection pool saturation.

To compare patterns, run the same load profile against both implementations. A workable profile:

- Ramp from 1x to 10x expected peak concurrency over five minutes.
- Inject a latency fault: add a fixed delay to a percentage of downstream responses, or run the downstream with a constrained heap so it pauses under pressure.
- Inject an error fault: return 5xx for a short window, then recover.
- Record p50, p95, p99, error rate, retry rate, bulkhead rejections, and client and server CPU.

The comparison that matters is not "which is faster at steady state." It is "what happens to p99 and error rate during the fault window, and how long after the fault clears does the system return to baseline." A naive client typically shows a p99 spike that outlasts the fault because its own retries keep the downstream busy. A shaped client typically shows a bounded p99 and a faster return to baseline, at the cost of a small number of fast failures while the breaker is open.

If you want a single number to track, use the ratio of retry attempts to successful calls during the fault window. A ratio above roughly 1.5 means the client is amplifying load rather than absorbing it.

## Failure modes of shaped calls

Shaped calls are not free. The failure modes are specific and worth designing around.

**Cold-start breaker trips.** A newly started instance has no history. If the downstream is slow during startup, the breaker can open on the first few requests and reject traffic that would have succeeded. Mitigations: require a minimum request count before the breaker can trip, or use a half-open probe with a generous timeout.

**Backoff on the happy path.** Some implementations add a small delay before the first attempt, or retry on errors that are not worth retrying. Keep the first attempt immediate and only retry on genuinely transient conditions.

**Retry budget too generous.** A budget of five attempts with a 50 ms base interval and no jitter turns a 200 ms downstream pause into a multi-second tail because attempts stack up. Cap the interval, add jitter, and keep the total attempt count low.

**Correlated retries.** Without jitter, many clients retry at the same instant after a shared failure, producing a thundering herd. Jitter of 20 to 30 percent is usually enough to spread the load.

**Observability gaps.** A breaker that opens silently looks like a sudden drop in traffic. Export the breaker state and alert on it, otherwise the first sign of trouble is a support ticket.

**Non-idempotent retries.** Retrying a charge, a send, or a state mutation without an idempotency key can duplicate the effect. The retry pattern must be paired with an idempotency mechanism on the downstream, or the retries must be restricted to read-only operations.

That last point deserves emphasis. A retry budget is only safe when the operation is idempotent or the downstream deduplicates. Teams that add retry budgets without idempotency keys often trade a latency incident for a correctness incident, which is worse.

## Decision checklist

Work through these questions before choosing a pattern.

- **Is the operation idempotent?** If not, either add an idempotency key or restrict retries to safe methods. Without this, no retry pattern is safe.
- **What is the downstream's p99, not its p50?** If p99 is more than roughly three times p50, the downstream has tail behavior that a naive client will amplify.
- **Is the downstream subject to garbage collection or periodic pauses?** Runtimes with stop-the-world pauses produce exactly the kind of latency spike that triggers retry storms.
- **Can the caller absorb a traffic surge during a downstream outage?** If the answer is no, a bulkhead and circuit breaker are mandatory, not optional.
- **Is the retry configuration changeable without a deploy?** If every tuning change requires a code release, the configuration will drift out of date.
- **Is the caller multi-tenant?** If one tenant can consume the shared capacity of a downstream, per-tenant rate limits are needed.
- **What is the cost of a fast failure versus a slow success?** For user-facing checkout, a fast failure with a clear error is usually better than a 30-second hang.

A rough mapping from service characteristics to pattern:

| Service type | Typical downstream latency | Pause-prone | Outage blast radius | Suggested pattern |
|---|---|---|---|---|
| Internal admin job | < 10 ms | No | Low | Naive |
| Feature flag lookup | < 50 ms | No | Medium | Naive with a timeout |
| Cache-aside read | < 5 ms | No | Low | Naive with a timeout |
| Payment or charge call | 100-300 ms | Often | High | Shaped, with idempotency keys |
| Real-time analytics write | 50-200 ms | Often | Medium | Shaped |
| Image or media processing | 200-500 ms | Often | Medium | Shaped |
| Multi-tenant SaaS dependency | Variable | Often | High | Shaped, with per-tenant limits |

## A worked example: sizing a retry budget

Suppose a downstream has a p50 of 80 ms and a p99 of 400 ms, and a stop-the-world pause of roughly 300 ms every 30 seconds. A client timeout of 1 second will mostly succeed, but during a pause requests will time out.

With no retry budget, every timed-out request retries immediately. If the client sends 200 requests per second and the pause causes a 1-second window of timeouts, that is 200 retries arriving at the moment the downstream is trying to recover. The retries themselves take time, so the effective load during recovery is higher than the original load.

With a retry budget of three attempts, a 100 ms base backoff, a 5-second cap, and 20 percent jitter, the first retry for a request that timed out at t=1s arrives at roughly t=1.1s, the second at roughly t=1.3s, and the third at roughly t=1.7s. The retries are spread across 700 ms instead of arriving in a single burst. If the pause was 300 ms, the downstream is already recovered before the second retry, and most requests succeed on that attempt.

The arithmetic is illustrative, but the reasoning generalizes: the retry budget and backoff curve determine the shape of the load the downstream sees during recovery, and that shape is what decides whether the incident is brief or sustained.

## FAQ

**What is the minimum viable set of safeguards?**

Four things: a retry budget with a low ceiling, a backoff curve with jitter and a cap, a circuit breaker with a minimum request count before tripping, and a bulkhead that caps concurrent calls. Add an idempotency key on the downstream for any mutating operation. Anything less than this leaves one of the amplification paths open.

**How do I know if my current pattern is already unsafe?**

Export `retry_attempts_total` and `call_duration_seconds` by attempt number. If the retry rate rises during downstream latency spikes rather than falling, the pattern is amplifying load. If the p99 of the total call duration is more than a few multiples of the downstream p99, retries are contributing to the tail.

**Does a circuit breaker help if the downstream is only slow, not failing?**

Yes, if the breaker is configured to trip on slow calls as well as errors. Many implementations only count errors, which means a downstream that returns 200 responses slowly will never trip the breaker. Configure a slow-call threshold or use a timeout that converts slow responses into errors the breaker can see.

**Can shaped calls work in serverless environments?**

Yes, with adjustments. Serverless platforms cap concurrency at the platform level, which acts as a bulkhead. The retry budget, backoff, and idempotency requirements still apply. Keep timeouts short so a slow downstream does not consume the function's entire execution budget.

**What is the most common configuration mistake?**

A retry budget that is too large combined with a backoff base that is too small. Five attempts with a 50 ms base interval and no jitter produces a burst of retries within a few hundred milliseconds, which is exactly when the downstream is least able to handle them.

**Do shaped calls add latency on the happy path?**

A well-implemented client adds essentially nothing on the first attempt. Latency only appears when a retry or a breaker rejection occurs. If your implementation adds a fixed delay before the first attempt, remove it.

## Do this in the next 30 minutes

Open the file in your codebase that issues calls to your most critical downstream dependency. Find the retry logic, if any. Answer three questions in writing: is the operation idempotent, what is the total attempt ceiling, and what happens to concurrent calls when the downstream slows down. If you cannot answer all three from the code, that is the gap to close first. Add a counter for retry attempts labelled by downstream and reason, deploy it, and watch it for one traffic cycle. A retry rate that rises during latency spikes is the signal that the pattern needs to change.
