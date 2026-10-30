# Bank APIs broke under African 2G

## The problem with cache-first advice on unreliable links

Most API design guidance assumes a stable last mile: reliable DNS, a CDN close by, and a client that stays connected long enough to complete a request. Cache aggressively, rate-limit defensively, validate inputs thoroughly. That advice is sound when the network is a minor variable.

In many markets it is the dominant variable. Mobile data is frequently the primary access path, and connections can drop from 4G to 3G to 2G within a single session, sometimes several times a minute. A typical failure mode looks like this:

1. A client sends a request. The connection stalls before the response arrives.
2. The client times out and retries.
3. The retry reaches an edge cache that is still within its TTL, so it returns a response computed before the first request ever completed.
4. The user sees stale state (for example, a balance that has already changed), acts on it, and a dispute follows.

None of the individual steps is exotic. The combination is what breaks the standard playbook, because the playbook assumes that cache freshness and client retries are independent concerns. On a flaky link they are tightly coupled.

This article works through why the conventional setup fails, what a network-aware design looks like, how to measure whether you actually have the problem, and how to decide which parts apply to your system. It deliberately avoids quoting results from other companies' systems; instead it shows what to instrument so you can produce your own numbers.

## A worked failure: five-minute TTL, remote idempotency store

Consider a conventional payments API with these components:

- A cache (any key-value store) in front of the balance endpoint, TTL 5 minutes.
- A CDN in front of the API, also caching the balance response.
- Per-user rate limiting implemented as a lookup in a shared remote store.
- Idempotency keys stored in a managed remote database with a 24-hour expiry.

Now walk through a single user session on a slow link, with reasoning shown at each step:

- **t=0s.** The client requests its balance. The origin computes it and returns 500 GHS. The cache stores this with a 5-minute TTL.
- **t=180s.** The user completes a transfer elsewhere. The balance is now 300 GHS, but the webhook that would notify your system is batched by the bank and has not arrived.
- **t=185s.** The client requests its balance again. The connection stalls; the client times out after 5 seconds and retries.
- **t=190s.** The retry hits the cache. The entry is 190 seconds old but well within its 5-minute TTL, so the cache returns 500 GHS.
- **t=200s.** The user sees 500 GHS, believes they have funds, and initiates a second transfer that will fail or overdraw. A dispute follows.

The webhook may arrive minutes later and invalidate the cache, but by then the user has already acted on stale data. Invalidation that is correct but late is indistinguishable from no invalidation, from the user's point of view.

Two structural faults produce this:

- **TTL is being used as a proxy for freshness.** A TTL bounds how long data *may* be served; it says nothing about whether the data is still true. When the underlying value changes on an external system's schedule, a time-based bound is the wrong tool.
- **The idempotency store is remote.** If the connection drops between sending a request and receiving the response, the client retries. If the retry reaches a different edge node, or the remote store lookup itself times out, the system may treat the retry as a new request. That is how duplicate transactions happen: not because idempotency keys were missing, but because the store holding them was unreachable at exactly the moment it mattered.

A third, quieter fault: rate limiting that performs a remote lookup on every request adds the lookup's round-trip time to every call. On a link where round-trip time is already hundreds of milliseconds, this is a meaningful tax on latency that buys little, since the limiter's state does not need global consistency to be useful.

## The mental model: design for the connection, not the server

Reframe the goal. The cache is not primarily there to reduce origin load; it is there to let a client get a usable answer when the network is briefly unusable. Under that framing, the design assumptions change:

- A session will include at least one multi-second drop.
- External notifications (webhooks, settlement callbacks) will arrive late and out of order.
- The client's network type will change mid-session.
- Clients will retry the same request several times in quick succession.

From those assumptions, four design rules follow.

**1. Separate freshness from caching.** The TTL should bound how long a value may be *served without revalidation*, not how long it may be *trusted*. Pair every cached value with an event that invalidates it, and treat the TTL as a backstop for when the event never arrives.

**2. Make invalidation event-driven, with TTL as fallback.** When the source of truth signals a change, invalidate immediately. When it does not, fall back to TTL expiry. The important part is that the fallback is explicitly a degraded mode, not the primary mechanism.

**3. Keep rate limiting local.** A per-process token bucket (for example, Go's `golang.org/x/time/rate`) avoids a network round trip on every request and degrades gracefully: if the process restarts, the bucket resets, which is acceptable for abuse prevention and unacceptable only if you need exact global quotas. Choose accordingly.

**4. Make idempotency state durable and local to the request path.** If the store that answers "have I seen this key?" is unreachable, the retry cannot be safely deduplicated. A local, durable store on the node that receives the retry removes that dependency.

## A layered design

A workable shape for a balance endpoint under these constraints:

- **Tier 1 — in-process cache.** A small LRU (order of a few thousand entries) with a very short TTL, on the order of seconds. Its job is to absorb retry storms from a single client, not to serve as the system's cache.
- **Tier 2 — shared cache.** A key-value store with a short TTL (tens of seconds) for balance data, invalidated by events from the source of truth. This is the tier that carries most of the load reduction.
- **Tier 3 — CDN.** A short TTL (low minutes at most) for geographic distribution. Be explicit about which responses are cacheable; balance responses usually should not be cached at the CDN at all unless the response is scoped to a single authenticated user and the CDN is configured to respect that.

For idempotency keys, a local SQLite database in WAL mode on each node that can receive retries gives durable, low-latency deduplication. The node checks its local store first; on a hit it returns the previously stored response without contacting the origin. On a miss it forwards the request, then stores the key and response locally.

Sketch of the deduplication path:

```sql
-- Run once per node.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;

CREATE TABLE IF NOT EXISTS idempotency (
  key         TEXT PRIMARY KEY,
  response    BLOB NOT NULL,
  status      INTEGER NOT NULL,
  created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_idempotency_created
  ON idempotency (created_at);
```

```go
// Pseudocode: check-then-store on the request path.
func handleWithIdempotency(w http.ResponseWriter, r *http.Request) {
    key := r.Header.Get("Idempotency-Key")
    if key == "" {
        http.Error(w, "missing Idempotency-Key", http.StatusBadRequest)
        return
    }

    if rec, ok := lookupLocal(key); ok {
        // Replay the stored response. Do not re-execute the operation.
        w.Header().Set("Idempotent-Replay", "true")
        w.WriteHeader(rec.Status)
        w.Write(rec.Response)
        return
    }

    rec := executeOperation(r)
    // Store before responding so a crash after the write but before the
    // response still leaves a replayable record.
    storeLocal(key, rec)
    w.WriteHeader(rec.Status)
    w.Write(rec.Response)
}
```

Two things to get right: store the record *before* sending the response, so a crash mid-response still leaves a deduplication record; and expire records on a schedule that matches the contract you advertise to clients (the retention window you promise in your API docs).

## How to measure whether you have this problem

The failure mode above is invisible in aggregate latency dashboards. Measure it directly.

**Instrument these on the server:**

- Age of the data served, in seconds, for every response that comes from a cache. Emit it as a histogram. The p99 of this metric is the number that matters.
- Cache hit ratio, split by tier. A high hit ratio is not good news if the hits are stale.
- Rate limiter lookup latency, as a separate span from the endpoint's own work.
- Idempotency key outcomes: new, replayed, and rejected, as counters.

**Instrument these on the client or in synthetic probes:**

- Time to first byte and time to last byte, per network type if the client can report it.
- Retry counts per logical operation. A distribution with a long tail here is the signature of the drop-and-retry pattern.
- Whether the client ever observed a value that later turned out to be wrong. This is the only metric that directly measures user-visible staleness; it requires the client to compare a cached value against a later authoritative one.

**Compare before and after.** Run the same synthetic probe against the old and new configurations and compare the *age-of-served-data histogram*, not just latency. A redesign that keeps latency flat while collapsing the p99 age from minutes to seconds is a win even if no other number moves.

**Simulate the network.** On Linux, `tc` (traffic control) with `netem` can emulate delay, jitter, and loss on a loopback or test interface:

```bash
# Emulate a slow, lossy link on loopback for local testing.
tc qdisc add dev lo root handle 1: htb default 11
tc class add dev lo parent 1: classid 1:1 htb rate 1mbit
tc class add dev lo parent 1:1 classid 1:11 htb rate 1mbit
tc qdisc add dev lo parent 1:11 handle 10: netem delay 600ms loss 2%
```

Adjust `delay` and `loss` to match the conditions you care about, and remove the qdisc with `tc qdisc del dev lo root` when finished. Run your integration tests under these conditions in CI. The specific parameters are a starting point, not a recommendation; derive yours from measurements of your actual user base.

## Decision checklist

Work through this before changing anything.

- **Does the endpoint return data whose truth is controlled by an external system?** If yes, TTL alone is not sufficient; you need an invalidation signal.
- **Can the client safely act on data that is N seconds old?** If no, the TTL must be short and the response must carry enough information for the client to know its age.
- **Does the operation mutate state?** If yes, it needs idempotency, and the idempotency store must be reachable on the retry path.
- **Is the idempotency store on the same node that receives retries?** If no, a retry that lands on a different node can duplicate the operation.
- **Does rate limiting require a network round trip?** If yes, measure how much latency it adds and whether exact global limits are actually required.
- **Have you tested under emulated loss and delay?** If not, you have not tested the conditions that produce these failures.

## When the conventional advice is correct

The standard playbook is not wrong; it is conditional. It holds when:

- **The data is not authoritative for money movement.** Product catalogs, marketing content, and reference data tolerate long TTLs because staleness has no financial consequence.
- **The client can tolerate staleness by contract.** If the API documents that a value may be up to an hour old, a long TTL is honest rather than a bug.
- **The data changes faster than any invalidation could propagate.** Market data feeds are the canonical example; there, short TTLs and client-side interpolation are the norm, and event-driven invalidation is not feasible.
- **The endpoint is internal.** Admin tools and internal dashboards can accept longer TTLs and coarser invalidation.
- **Strong consistency is required.** If no staleness is acceptable, caching is the wrong tool and the endpoint should read from the source.

The distinguishing question is not "fintech or not" but "what happens to the user if this value is wrong, and for how long." Answer that and the caching policy follows.

## Comparison of the two shapes

| Concern | Time-driven only | Event-driven with local state |
|---|---|---|
| Freshness bound | TTL, set by guesswork | Event plus TTL backstop |
| Retry safety | Depends on remote store reachability | Local durable store on the retry path |
| Rate limiting cost | One remote lookup per request | Local, no round trip |
| Behaviour during a partition | Serves stale data until TTL expires | Serves stale data only until the event or backstop |
| Operational complexity | Lower | Higher: needs an event pipeline and local storage |
| Failure mode when the event is lost | Silent staleness until TTL | Silent staleness until TTL (same backstop) |

The last row matters: event-driven invalidation does not remove the need for a TTL. It changes the TTL from the primary mechanism to a safety net.

## Common objections

**"Shorter TTLs will destroy the cache hit ratio and overload the origin."**

Test it rather than assuming. The hit ratio depends on request arrival patterns, not only on TTL. If the same user retries several times within seconds, a short-TTL in-process cache still absorbs those retries. Measure origin request rate before and after; if it rises, that is the cost of freshness, and you can decide whether to pay it.

**"Local state on edge nodes is fragile."**

It is, which is why the local store must be durable (WAL mode, `synchronous = FULL`) and why the design should tolerate a node losing its local state. If a node restarts and loses its idempotency records, the worst case is that a retry is treated as new. You can bound that risk by also writing keys to a shared store asynchronously, accepting that the shared store is a backstop rather than the primary path.

**"Why not just poll the source of truth?"**

Polling is simpler to build and easier to reason about, but it has a fixed latency floor equal to the poll interval and it scales with the number of entities you poll. Event-driven invalidation has lower steady-state latency but requires handling out-of-order and duplicate events. Choose based on whether your source of truth offers events at all; if it does not, polling with a short interval and a short TTL is a reasonable compromise.

**"Isn't this just accepting eventual consistency?"**

Yes. The point is to make the consistency window explicit and bounded, rather than an accident of a TTL chosen for load reasons. Document the window in the API contract, expose the data's age in the response, and let clients decide what to do with it.

## Summary

The conventional caching playbook assumes a stable last mile and a reachable control plane. On intermittent mobile links, both assumptions fail. The fixes are structural: separate freshness from caching, make invalidation event-driven with a TTL backstop, keep rate limiting local, and put idempotency state on the retry path. None of these is exotic, and none requires abandoning caching. They require being explicit about what the cache is for.

The measurement discipline matters as much as the design. Track the age of served data, not just latency. Simulate loss and delay in CI. Compare configurations on the staleness histogram, not on aggregate throughput.

## Do this in the next 30 minutes

Open your balance or account-state endpoint and add a response header or log field recording the age in seconds of the data being served, measured from the moment the value was produced by the source of truth. Deploy it to one environment. If you cannot populate that field because nothing in your stack tracks when the value was produced, that is the finding: you are serving data whose freshness you cannot measure, and every other decision in this article depends on fixing that first.
