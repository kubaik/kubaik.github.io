# Run APIs in 50 spots: what breaks first

Running an API across many edge locations is not a deployment problem. It is a data-model problem. The network layer is the easy part: pushing a worker to fifty points of presence is a config change. What breaks afterward is everything that assumed a single writer, a single clock, and a single connection pool. This article walks through the failure modes in the order they usually appear, and gives you the instrumentation to confirm each one on your own system.

## The mental model: an edge tier is a cache that can compute

A CDN caches bytes. An edge compute tier caches *decisions* and sometimes *state*. That distinction matters because state at the edge implies you now own reconciliation between many copies of the truth.

A useful framing: treat the origin as the root of a tree and each edge location as a leaf. Leaves serve reads locally. Writes travel toward the root. When a leaf is partitioned, it must have a defined degradation policy — serve stale reads, queue writes, or return a cached error — and that policy must be a deliberate product decision, not an accident of which exception your code throws first.

The analogy of ants returning food to a nest is common but slightly misleading. Ants converge on a single nest; edge writes converge on an origin *and* must be idempotent, because retries after a timeout are indistinguishable from new writes unless you design for it.

## Failure mode 1: the database round trip dominates everything

The first symptom teams notice is that p99 latency at the edge is far worse than p50, and the gap does not correlate with CPU or network throughput. It correlates with write commit latency to the origin.

To confirm this on your own system, instrument three separate timings rather than one end-to-end number:

1. Time spent in the edge worker before the first outbound call.
2. Round-trip time from the edge location to the origin, measured per location.
3. Time from "write accepted by origin" to "write visible to a subsequent read from the same edge location."

If (3) is the large number, you have a commit-lag problem, not a compute problem. A useful illustrative example of the shape of this: if origin p95 is 35 ms and edge p95 without writes is 65 ms, the extra 30 ms is one round trip plus local overhead. If edge p99 with writes is 420 ms, the additional ~355 ms is queue depth and commit lag, not CPU. Those figures are illustrative only — substitute your own measurements.

The lesson is that user-perceived latency and write-durability latency are different metrics and must be tracked separately.

## Failure mode 2: connection pooling multiplies

At the origin, a pool of N connections serves all traffic. At fifty edge locations, each running its own pool, the origin sees up to 50 × N connections. A pool of 10 per location is 500 connections against the origin — often more than a small database will accept.

Mitigations, in order of preference:

- Route writes through a small number of regional aggregators rather than directly from every location.
- Use a connection proxy that multiplexes many client connections onto few server connections.
- Cap pool size per location explicitly and set a short idle timeout so that idle locations release capacity.

The failure mode when you get this wrong is not a clean error. It is connection starvation: requests queue at the pool, timeouts cascade, and the origin looks healthy on CPU while the edge looks broken.

## Failure mode 3: cache invalidation races with writes

A naive invalidation — delete the key after a write, then let the next read repopulate — has a well-known race. Between the delete and the repopulate, a concurrent read can fetch stale data from a replica and write it back into the cache, where it persists until the next invalidation.

At a single origin this race is rare. Across fifty locations it is routine, because the window is longer and the number of concurrent readers is larger.

A write-through pattern with a monotonically increasing version is one fix:

```typescript
// On the origin, after a successful write:
await env.ORIGIN_DB.prepare(
  'UPDATE orders SET status = ?, version = version + 1, updated_at = now() WHERE id = ?'
).bind(newStatus, orderId).run();

// Read path at the edge:
const cached = await env.KV.get(`order:${orderId}`);
const parsed = cached ? JSON.parse(cached) : null;
if (parsed && parsed.version >= expectedVersion) {
  return parsed;
}
// Otherwise fall through to origin, then write back with the version included.
```

The version field lets a stale read be rejected rather than overwritten. If the origin write fails, the edge keeps serving the older version — stale, but internally consistent.

## Failure mode 4: clock drift breaks anything with a lease

NTP-synchronized clocks across many locations can diverge by tens to hundreds of milliseconds depending on network conditions and host load. Any algorithm that assumes two nodes agree on "now" — leases, lock expiry, last-write-wins by timestamp — is unsafe under that drift.

Two practical responses:

- Use hybrid logical clocks (HLCs), which combine a physical timestamp with a logical counter so that causality is preserved even when physical clocks disagree.
- Avoid last-write-wins on wall-clock timestamps entirely. Prefer per-field version counters or CRDTs whose merge function does not depend on synchronized time.

If you must use leases, make the lease duration much larger than the expected drift, and treat a lease as advisory rather than authoritative.

## Failure mode 5: the observability firehose

Fifty data planes emitting traces, logs, and metrics is not a linear cost increase if you emit one trace per request per location. It is roughly fifty times the span volume of a single region, and most APM vendors price per span.

Before scaling out, decide what you actually need:

- Aggregate metrics (counts, histograms) at the edge and ship only aggregates.
- Sample traces at the edge with a consistent sampling key so that a single user's journey is either sampled everywhere or nowhere.
- Emit structured logs with a location tag so that a regional outage is visible as a gap in a series rather than as an absence you have to notice.

A concrete way to check whether your pipeline is viable: take your current span volume per second, multiply by the number of locations, and compare against your vendor's ingestion limit and per-span price. If the product exceeds your budget, reduce span count before you reduce locations.

## A worked example: a "create order" endpoint

Consider an endpoint that accepts an order and must be reachable from many locations with a single origin database.

The edge table stores only what the fast path needs:

```sql
-- origin schema
CREATE TABLE orders (
  id bigserial PRIMARY KEY,
  user_id bigint NOT NULL,
  product_id bigint NOT NULL,
  status text NOT NULL,
  version bigint NOT NULL DEFAULT 1,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- edge-local staging table
CREATE TABLE local_orders (
  id bigserial PRIMARY KEY,
  user_id bigint NOT NULL,
  product_id bigint NOT NULL,
  status text NOT NULL,
  edge_created_at timestamptz NOT NULL DEFAULT now(),
  origin_id bigint,
  sync_status text NOT NULL DEFAULT 'pending'
);
```

The staging row is smaller than the origin row, which reduces local storage and serialization cost. Writes are batched to the origin on a short interval to keep staleness bounded.

The reasoning behind the design, step by step:

1. Accept the write locally and assign a local id. This keeps the user-facing latency independent of origin round-trip time.
2. Mark the row `pending` and record the local creation time.
3. A background flush sends pending rows to the origin in a batch, using an idempotency key derived from the local id so that a retry after a timeout does not create a duplicate.
4. On success, record the origin id and mark the row `synced`.
5. On repeated failure, apply backoff and eventually surface a `failed` status rather than retrying forever.

The backoff schedule should be bounded and jittered. A common shape is exponential growth from a small base with a maximum, plus random jitter to avoid synchronized retries across locations:

```javascript
class Backoff {
  constructor(base = 100, max = 5000) {
    this.base = base;
    this.max = max;
    this.attempts = 0;
  }

  next() {
    const delay = Math.min(this.base * 2 ** this.attempts, this.max);
    this.attempts++;
    return delay + Math.floor(Math.random() * 100);
  }
}
```

Two breakers are needed, not one: a remote breaker for the origin and a local breaker for the cache. A local cache miss should not open the remote breaker, or you will shed traffic during a purely local event.

## Where CRDTs help, and where they do not

For data that converges naturally — a catalog price, a counter, a set of tags — a CRDT avoids coordination entirely. The merge function must be commutative, associative, and idempotent, so that messages arriving out of order or twice produce the same result.

A simplified last-writer-wins register keyed by node:

```typescript
interface VersionedValue<T> {
  value: T;
  timestamp: number;
  nodeId: string;
}

function mergeLWW<T>(a: VersionedValue<T>, b: VersionedValue<T>): VersionedValue<T> {
  if (a.timestamp !== b.timestamp) {
    return a.timestamp > b.timestamp ? a : b;
  }
  // Tie-break deterministically on node id.
  return a.nodeId > b.nodeId ? a : b;
}
```

This works for prices and similar values where the latest write is genuinely the one you want. It does not work for anything requiring an invariant across fields — an inventory count that must never go negative, for example. For those, either route writes to a single leader region or use a CRDT that encodes the invariant (a counter that supports decrement with a floor, for instance).

The trade-off is explicit: CRDTs buy availability and latency at the cost of merge complexity and the inability to express arbitrary constraints.

## Decision checklist before you scale out

Answer these before adding locations. If any answer is "unknown," that item is your next measurement, not your next deployment.

- What is the measured round-trip commit latency from each candidate location to the origin?
- What is your origin's maximum connection count, and what does 50 × pool size equal?
- For each write path, is the operation idempotent under retry? What is the idempotency key?
- What is the defined behavior when a location is partitioned: stale reads, queued writes, or errors?
- Which fields use wall-clock timestamps for conflict resolution, and can those be replaced with version counters or HLCs?
- What is your current span volume per second, and what does it become at N locations?
- What is the cache hit rate on the endpoints you intend to move, and what is the cost per origin miss?

## Comparison: single-region vs. edge tier

| Dimension | Single region | Edge tier |
|---|---|---|
| Read latency | Depends on user distance | Low for cache hits, origin-bound for misses |
| Write latency | One round trip to region | One round trip plus queue and commit lag |
| Consistency | Easy to reason about | Requires bounded staleness and versioning |
| Connection load on origin | One pool | One pool per location unless aggregated |
| Observability cost | Linear | Multiplies with location count |
| Failure surface | Origin outage | Origin outage plus per-location partition |

Neither column is universally better. The choice depends on whether your traffic is read-heavy and cacheable or write-heavy and consistency-sensitive.

## FAQ

**How do I know if my endpoint is a good candidate for the edge?**

Measure the cache hit rate for that endpoint over a representative period. If a large majority of requests can be served from a cached response with bounded staleness, it is a candidate. If most requests mutate state or require cross-record invariants, keep them in a region.

**What is the smallest useful first step?**

Pick one read-only endpoint. Put a cache in front of it at the edge. Instrument hit rate and origin request rate. Compare p95 latency from a few geographic vantage points before and after. If origin request rate does not drop, the cache is not doing useful work and the endpoint is not a good candidate.

**How do I handle a location that loses connectivity to the origin?**

Decide in advance: serve stale reads with a clear staleness indicator, or fail the write with a retryable error. Queueing writes is viable only if you have an idempotency key and a bounded queue with a defined overflow behavior.

**Do I need a globally distributed SQL database?**

Only if your write patterns genuinely require multi-region write availability and your application can tolerate the cross-region commit latency. Many systems are better served by a single-leader write path with edge read replicas and bounded staleness.

## Action for the next 30 minutes

Pick one endpoint you are considering moving to the edge. Add three timers to it: time before the first outbound call, round-trip time to the origin, and time from write-acknowledged to read-visible. Run it from a single non-local vantage point and record the three numbers. Those three numbers will tell you which of the failure modes above you will hit first — and whether the edge is the right move at all.
