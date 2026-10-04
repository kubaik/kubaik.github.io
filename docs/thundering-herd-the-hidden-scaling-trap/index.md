# Thundering Herd: The Hidden Scaling Trap

## The one-paragraph version

A thundering herd occurs when many clients or processes respond to the same stimulus at the same moment and converge on a shared resource: a cache key expiring, a service restarting, a scheduled job firing, a lock being released. Each individual action is reasonable. The collective result is a spike of identical work against a backend that was sized for steady-state traffic. The failure is not peak load in the ordinary sense; it is correlated load. It is insidious because the instinctive remedy, retrying on failure, usually makes it worse by adding synchronized pressure to a resource that is already saturated. Effective mitigations all share one property: they break the correlation, either by spreading retries across time, by letting one caller do the work while others wait, or by admitting only a bounded number of requests to the backend.

## Why the concept confuses people

Most scaling intuition is capacity-based: add replicas, raise provisioned throughput, cache more aggressively. Thundering herd problems violate that model because adding capacity can raise the blast radius rather than reduce it. More application instances means more concurrent processes that can all miss the same cache key at the same instant.

Three specific sources of confusion recur.

First, the symptom and the cause look unrelated. Error rates and latency spike while overall request volume may be unremarkable. A single cache key expiry can trigger thousands of identical backend reads. Distributed tracing shows many concurrent requests, but unless spans are grouped by the key or resource they touch, the common origin is invisible. The useful diagnostic move is to aggregate traces by cache key, table partition, or downstream endpoint and look for a single resource attracting disproportionate concurrent traffic.

Second, retries are usually treated as unconditionally good. Retries improve resilience against transient, independent failures. They are harmful against correlated failures, because every client observes the same failure at the same time and retries at the same time. A retry loop without backoff and jitter converts a brief backend hiccup into a sustained overload. The timing and randomization of retries matter as much as the retry count.

Third, the behavior is emergent rather than localized. No single component is buggy. The system as a whole produces the failure. This makes it resistant to unit testing and to component-level capacity planning, and it means the fix is usually a coordination protocol rather than a code fix in one place.

## The mental model

Picture a single-lane bridge at rush hour. The bridge is the shared resource: a database, a cache layer, an upstream API. Traffic flows normally until an incident blocks the bridge. Every driver then looks for an alternate route, and if they all choose the same moment and the same alternate road, the alternate road jams just as badly.

The lesson is not "widen the bridge." It is traffic management. Three families of control map directly onto distributed systems:

1. Drivers wait a random interval before attempting the alternate route. This is exponential backoff with jitter.
2. One designated driver checks the alternate route and reports back. This is request coalescing, also called single-flight.
3. A dispatcher admits a bounded number of cars at a time. This is rate limiting, queuing, or a semaphore.

The defining property of a herd is independent actors responding to a shared stimulus. The goal of every mitigation is to decorrelate those responses.

## Worked example: cache miss on a serverless API

Consider an API built from a fleet of serverless functions behind an HTTP gateway. Each invocation reads a key from an in-memory cache; on a miss it reads from a managed key-value store and writes the result back with a TTL.

Assume the following illustrative figures, chosen to make the arithmetic visible rather than to describe any particular deployment:

- 1,000 concurrent invocations are in flight when a hot cache key expires.
- A backend read takes 150 ms.
- The backend sustains 2,000 reads per second.

With no mitigation, all 1,000 invocations miss simultaneously and issue 1,000 backend reads within roughly the same 150 ms window. That is a burst of about 6,700 reads per second against a backend rated for 2,000. The backend throttles. The invocations receive throttling errors or 5xx responses.

Now add naive retry-on-error with no backoff. Each of the 1,000 callers retries immediately. The retry burst is again about 6,700 reads per second, arriving on top of whatever traffic is still being served. The backend never sees a quiet interval long enough to drain, and the failure sustains itself. This is the self-inflicted denial of service: the clients are the attacker and the target is the same system.

The arithmetic also shows why partial mitigation is weak. If only half the callers back off, the remaining burst is still around 3,350 reads per second, above capacity. The mitigation has to reduce the count of concurrent backend reads, not merely the average rate.

### What to instrument

- Cache miss rate per key, sampled at high resolution (one-second buckets are usually sufficient to see the burst).
- Concurrent in-flight backend reads per key or per partition, not just total request rate.
- Retry attempts per request, and the distribution of retry delays actually observed.
- Backend throttling or 5xx counts, correlated with the above.

### What to compare

Run a load test that forces a hot key to expire while the system is under steady load. Record the peak concurrent backend reads per key with mitigation disabled, then with each mitigation enabled individually. The metric that matters is the peak, not the mean.

## Mitigation 1: exponential backoff with jitter

Backoff spaces retries out over time. Jitter randomizes the spacing so that clients that failed together do not retry together. The standard formulation is exponential backoff with full jitter: the delay is drawn uniformly from the interval between zero and an exponentially growing ceiling.

```javascript
// backoff.js
function fullJitterDelay(attempt, baseMs = 100, capMs = 20000) {
  const ceiling = Math.min(capMs, baseMs * 2 ** attempt);
  return Math.random() * ceiling;
}

async function withBackoff(fn, { maxAttempts = 5, baseMs = 100, capMs = 20000 } = {}) {
  let lastError;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const delay = fullJitterDelay(attempt, baseMs, capMs);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}

module.exports = { withBackoff, fullJitterDelay };
```

Two properties matter. The ceiling grows exponentially, so repeated failures push retries further apart. The actual delay is randomized across the whole interval, so a population of clients that failed at the same instant spreads out. A fixed delay, or exponential backoff without jitter, still leaves clients synchronized.

Backoff alone does not fix the initial burst. It prevents the burst from repeating indefinitely and gives the backend room to recover. Pair it with a mechanism that prevents the initial stampede.

## Mitigation 2: request coalescing (single-flight)

Coalescing ensures that for a given key, only one caller performs the expensive backend read while the others wait for its result. Within a single process this is straightforward. Across processes it requires a shared coordination point, typically a lock in the cache layer.

The following example uses a lock acquired with a conditional set (the `SET key value NX PX ttl` pattern supported by Redis and compatible stores) plus a short randomized wait for callers that lose the race. It is a conceptual implementation: it omits lock ownership tokens, which are needed to avoid a slow holder releasing a lock it no longer owns.

```javascript
// coalesced_fetch.js
const { withBackoff } = require('./backoff');

const LOCK_TTL_MS = 10000;

async function fetchWithCoalescing(key, redisClient, fetchFromBackend) {
  return withBackoff(async () => {
    const lockKey = `lock:${key}`;

    const acquired = await redisClient.set(lockKey, '1', {
      NX: true,
      PX: LOCK_TTL_MS,
    });

    if (!acquired) {
      // Another caller is already fetching. Wait a jittered interval,
      // then re-read the cache. If it is still empty, retry the whole
      // sequence with backoff.
      const waitMs = 50 + Math.random() * 100;
      await new Promise((resolve) => setTimeout(resolve, waitMs));

      const cached = await redisClient.get(key);
      if (cached !== null) {
        return cached;
      }
      throw new Error(`cache still cold for ${key}`);
    }

    try {
      const fresh = await fetchFromBackend(key);
      if (fresh !== null && fresh !== undefined) {
        await redisClient.set(key, fresh, { EX: 60 });
      }
      return fresh;
    } finally {
      await redisClient.del(lockKey);
    }
  });
}

module.exports = { fetchWithCoalescing };
```

Two failure modes deserve attention.

The first is the lock holder dying before it populates the cache. The TTL bounds the damage: after `LOCK_TTL_MS`, another caller can acquire the lock. If the TTL is too long, callers wait unnecessarily; if too short, the herd can re-form while the first fetch is still running. A reasonable starting point is a small multiple of the p99 backend read latency, then tune from observed data.

The second is the thundering herd moving to the lock itself. If every caller polls the lock aggressively, the coordination point becomes the bottleneck. The jittered wait above is deliberately short and randomized; an alternative is to subscribe to a change notification and be woken when the cache is populated.

## Mitigation 3: bound concurrency and pre-warm

Coalescing handles the case where many callers want the same value. A different class of herd occurs when many callers want different values from the same constrained resource. Common triggers include a scheduled job starting on every replica at the same wall-clock time, or a deploy causing every instance to rebuild an in-process cache simultaneously.

Two controls apply. Bound the number of concurrent operations against the resource with a semaphore or a queue, so that excess work waits rather than piling on. Then stagger the triggers: give each replica a deterministic but distinct start offset derived from its identity, and add jitter to scheduled jobs. A cron expression that fires at `0 * * * *` on every replica is a herd generator; the same job with a per-replica offset is not.

Pre-warming is the complementary move. If a key is known to be hot, refresh it before expiry rather than after, or refresh it from a single designated worker. This converts a synchronized miss into a single background write.

## A decision checklist

Use this to choose controls for a specific resource.

- Is the load correlated? If all callers act on the same event, assume a herd is possible. If failures are independent and scattered, ordinary retries are fine.
- Is the work idempotent and cheap to duplicate? If so, coalescing may be unnecessary and simple concurrency bounds suffice.
- Can callers wait? If a caller can tolerate a short delay, coalescing and queuing are available. If every caller must be served immediately, the only options are pre-warming, capacity headroom, or shedding load.
- Is there a coordination point already? A cache layer with conditional writes can host a lock. If not, adding one introduces a new dependency and a new failure mode.
- What is the recovery time of the backend? This sets the minimum useful lock TTL and the backoff ceiling.
- What happens when the mitigation fails? A lock that cannot be acquired, or a queue that is full, needs a defined fallback. Failing fast is usually better than unbounded waiting.

## How this connects to things you already know

A thundering herd is a distributed race condition. The same reasoning that leads to locks and atomic operations inside a single process leads to distributed locks and coordination primitives across processes.

It relates to load balancing but is not solved by it. A load balancer spreads requests across replicas; it does not stop those replicas from all hitting the same downstream key. Rate limiting is a direct mitigation, but it usually operates at the edge, while a herd can form between internal services that the edge never sees.

It relates to circuit breakers as cause relates to response. A circuit breaker stops upstream callers from hammering a failing service. A thundering herd is frequently the event that trips the breaker in the first place, which is why prevention is cheaper than reaction.

It relates to queues. A queue decouples producers from consumers and lets consumers pull at their own pace, which removes the direct contention that produces a herd. Herds still form when consumers race for the same item after pulling, which is why visibility timeouts and partitioning matter.

Finally, it is a case study in emergent behavior. Each component behaves correctly in isolation. The failure exists only in their interaction at scale, which is why it is found by load testing and production observation rather than by reading any single component's code.

## Do this in the next 30 minutes

Pick your single hottest cache key and add one metric: the count of concurrent in-flight backend reads for that key, bucketed per second. Emit it from the code path that performs the backend read, tagged with the key. Then force that key to expire during a load test and look at the peak. If the peak is more than a small multiple of one, you have a herd, and you now have a baseline to measure coalescing against.
