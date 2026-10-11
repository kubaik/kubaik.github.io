# Rate limiters: burst vs abuse, two designs

The default configuration is fine right up until it isn't. Code review was never the hard part; knowing when a limiter is about to fail under real traffic is. This article skips the happy-path tutorial and goes straight to the design question that matters.

## The problem with one bucket

Most rate limiter designs collapse two very different things into one number: a user's legitimate burst and an attacker's abuse. A mobile client reconnecting after a subway ride fires 40 requests in two seconds. A scraper fires 40 requests in two seconds. If the limiter is a single fixed window of 100 requests per minute, both get the same treatment — one gets throttled unfairly, the other gets a free pass until the window rolls over. The interesting engineering question is not "what limit should I set" but "what signal distinguishes a burst from an attack, and what do I do with it."

This article compares two common approaches: a **token bucket with a separate concurrency cap** (the layered limiter) and a **sliding-window counter with a penalty box** (the reputation limiter). They are not mutually exclusive, and most production systems end up with pieces of both. But the design philosophies are different enough that picking one first changes everything downstream — the Redis schema, the error responses, the observability, and how a 429 gets explained to a support ticket.

The layered limiter is usually the better default for a new API, because its guarantees are stronger. But it has a real weakness around distributed state that the reputation limiter handles more gracefully, and pretending otherwise would make this comparison useless.

## Option A: token bucket + concurrency cap

The token bucket is one of the most widely deployed rate limiting algorithms, and for good reason: it has two parameters that map cleanly onto two different concepts. The **refill rate** (tokens per second) governs sustained throughput. The **bucket size** (maximum tokens) governs how much burst is tolerated. A bucket of 100 tokens refilling at 10/s means a client can burst to 100, then is held to 10/s afterward.

That already separates burst from sustained abuse. But it does not separate burst from *parallelism*, which is where the second layer comes in. A concurrency cap limits how many requests a single client can have in flight simultaneously. This matters because a burst of 40 requests fired in parallel is a different signal than 40 requests fired over two seconds. The first can exhaust a database connection pool; the second usually cannot.

A minimal implementation using a Redis-backed token bucket looks like this. The Lua script runs atomically on the server, which avoids the read-modify-write race that plagues naive implementations:

```lua
-- token_bucket.lua
-- KEYS[1] = bucket key
-- ARGV[1] = capacity, ARGV[2] = refill_per_sec, ARGV[3] = now_ms, ARGV[4] = cost
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

local delta = math.max(0, now - ts) / 1000.0
tokens = math.min(capacity, tokens + delta * refill)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / refill * 1000) + 1000)

return {allowed, math.floor(tokens)}
```

Calling it from a request handler (Python, using a Redis client that supports `EVALSHA`):

```python
import time
import redis

r = redis.Redis()
SCRIPT = r.register_script(open('token_bucket.lua').read())

def allow(user_id: str, cost: int = 1) -> tuple[bool, int]:
    now_ms = int(time.time() * 1000)
    allowed, remaining = SCRIPT(
        keys=[f'rl:{user_id}'],
        args=[100, 10, now_ms, cost],
    )
    return bool(allowed), int(remaining)
```

The concurrency cap is separate and usually lives in-process or in a lightweight shared store. An asyncio semaphore per user works if there is a single process; for multiple processes a shared counter with a TTL is needed, which is where the distributed-state weakness shows up.

Where the layered limiter shines: it is simple to reason about, the parameters have physical meaning, and a well-tuned bucket produces few false positives on legitimate mobile traffic. Where it struggles: when a client is legitimately distributed (a CI system with many runners behind one API key), the concurrency cap becomes a blunt instrument, and per-key overrides multiply the config surface.

### A worked tuning example

Suppose load testing shows a downstream service handles 50 requests per second before p99 latency degrades, and a typical interactive client bursts to 30 requests after an idle period. Those two observations give the parameters directly:

- Refill rate = 50 tokens/s (matches the sustained capacity of the dependency).
- Bucket capacity = 50 tokens (one second of headroom, comfortably above the observed 30-request burst).

Check the bound: over any interval of length T, allowed requests are at most `capacity + refill × T`. Over one second that is 100; over ten seconds, 550. If the downstream service can only absorb 50/s sustained, the bucket size is what allows a momentary overshoot, so keep it proportional to the largest burst the dependency can survive, not to the largest burst a client might send.

For the concurrency cap, instrument in-flight request counts per client and take a p99 over a week. Set the cap above that p99 — for example, if p99 in-flight is 12, start at 20 and tighten only if the dependency shows contention. Setting the cap from intuition is how false positives get introduced.

## Option B: sliding-window counter + penalty box

The reputation limiter takes a different stance. Instead of trying to model the shape of traffic, it counts requests in a sliding window and escalates consequences for clients that repeatedly hit the limit. The window is precise (no fixed-window boundary burst), and the penalty box is the innovation: a client that trips the limit multiple times in a short period gets progressively longer cooldowns, rather than a simple 429 that resets the moment the window rolls.

A sliding-window counter is typically implemented with two Redis keys — the current window and the previous window — and a weighted sum. The math, assuming a 60-second window and a request arriving 15 seconds into the current window:

- Previous window count: 80
- Current window count: 20
- Weight on previous = (60 - 15) / 60 = 0.75
- Estimated count = 20 + 80 × 0.75 = 80

That estimate is compared against the limit. The approximation is well-documented and is what many "sliding window" implementations actually do; the exact sliding log is more accurate but stores one entry per request, which is expensive at scale.

The penalty box is where the reputation limiter earns its name. A typical policy:

| Trips in last hour | Cooldown |
| --- | --- |
| 1 | 0s (normal 429) |
| 2 | 60s |
| 3 | 300s |
| 4+ | 3600s |

Those numbers are illustrative — the point is the shape, not the specific values. The escalation is stored alongside the client identity, often with a longer TTL than the rate limit key itself.

Where the reputation limiter shines: it handles distributed clients gracefully (each request is just a count), it produces a clear audit trail for abuse investigations, and the penalty escalation is a genuine deterrent against scrapers that would otherwise just retry after every 429. Where it struggles: the sliding window is an approximation, and the penalty box can misfire on legitimate clients that happen to have a bad deploy — a retry storm from your own mobile app can look exactly like an attack.

## Head-to-head: what the algorithms actually guarantee

The token bucket's guarantee is stated in the algorithm itself: over any interval of length T, the number of requests allowed is at most `capacity + refill × T`. That is a hard bound, and it holds regardless of how requests are distributed. This is why token buckets are the standard choice when a downstream resource with a known capacity needs protection.

The sliding-window counter's guarantee is weaker: it bounds requests per window, but the weighted approximation can allow slightly more than the limit at window boundaries. The error is bounded by the weight on the previous window, so the worst case is roughly `limit × (1 + (W-1)/W)` where W is the window length in seconds — for a 60-second window that is under 2× the nominal limit at the boundary, though in practice the approximation is much tighter. Teams that need an exact bound use a sliding log or a token bucket instead.

What neither algorithm does on its own is distinguish a burst from an attack. That distinction comes from the *second* layer in each design: the concurrency cap in Option A, the penalty box in Option B. This is the part most tutorials skip, and it is the part that actually matters.

## Head-to-head: developer experience and operational cost

The layered limiter is easier to tune because its parameters are physical: "this endpoint can sustain 50 rps and tolerate a 200-request burst." Those can be derived from load testing the downstream service. The concurrency cap is harder — it requires knowing the connection pool size, typical request latency, and how much headroom to leave. But once set, the behavior is predictable.

The reputation limiter is easier to *deploy* because there is only one counter to maintain per client, and the penalty box is just another key. But it is harder to *reason about* in an incident: when a customer complains about being blocked for an hour, the trip history has to be reconstructed, which means the penalty state has to be durable and queryable. If it lives only in Redis with a short TTL, the audit trail is lost.

Both designs share the same distributed-state problem: the limiter state must be consistent across all instances handling a given client. Sticky routing by client ID solves this but breaks when instances scale or restart. A centralized store (Redis, or a purpose-built service) is the usual answer, and it introduces a new failure mode — if the store is unavailable, do you fail open (allow everything) or fail closed (block everything)? Failing open for availability and accepting the risk is the common choice, and it is worth writing down explicitly as a decision.

### Failure modes to plan for

Three failure modes recur across both designs:

1. **Store unavailability.** If the shared counter store is down, the limiter has no state. Fail-open means an outage in the store becomes an outage in the protected dependency; fail-closed means the store is now a single point of failure for all traffic. Either way, log the condition loudly and alert on it, because a silent fail-open is indistinguishable from a working limiter until the dependency falls over.
2. **Key collision or key exhaustion.** If limiter keys are derived from user input without normalization, two clients can share a bucket, or one client can create unbounded keys. Namespace keys with a prefix, normalize the identity, and set a TTL on every key.
3. **Clock skew.** Both algorithms depend on timestamps. If the Lua script uses the Redis server's clock rather than the caller's, skew between application hosts does not matter. Passing `now_ms` from the application, as above, is simpler but reintroduces skew — prefer server-side time where the client clock is not trustworthy.

### Key derivation and security

Rate limit keys are usually derived from client identity — an API key, a user ID, an IP address. If those keys are guessable or spoofable, an attacker can either exhaust another user's bucket (denial of service against a specific account) or evade their own limit by rotating keys. IP-based limiting is especially fragile behind shared NATs and corporate proxies, where many legitimate users share one address. API-key-based limiting is better but only if the key itself is treated as a secret — which means the same discipline applied to database credentials applies here: no keys in client-side code, no keys in public repositories, rotation on suspected exposure. Secrets committed to public repositories is a well-documented and common way rate limit keys leak, and once a key is public, the limiter is protecting the wrong identity.

## A decision framework

Use the layered limiter (token bucket + concurrency cap) when:

- Clients are mostly interactive — mobile apps, browsers, CLI tools — where bursts are natural and expected.
- A downstream resource has a known capacity that needs protection (a database, a third-party API with its own limits).
- Per-endpoint tuning work is affordable and predictable, provable bounds are wanted.

Use the reputation limiter (sliding window + penalty box) when:

- Clients are mostly machine-to-machine — integrations, batch jobs, partner systems — where sustained rate matters more than burst shape.
- Abuse is a real, ongoing concern and escalation should be automatic and auditable.
- Approximation at window boundaries is acceptable in exchange for simpler state.

Use both when: the API is public with mixed client types, which describes most public APIs. The layered limiter handles the interactive clients; the penalty box handles the repeat offenders. The two can share the same Redis instance but should use separate key namespaces so a bug in one does not corrupt the other.

## Recommendation and caveats

Starting with the layered limiter for a new API is usually the right call, because its guarantees are stronger and its parameters are easier to justify in a design review. Add a penalty box later, once there is data on which clients actually abuse the system — building the escalation policy before that data exists means guessing at thresholds, and guessed thresholds are how a legitimate customer gets blocked during their biggest traffic day.

The weakness of the layered limiter is real: the concurrency cap is hard to tune correctly, and getting it wrong produces either false positives (throttling users who are not actually a problem) or false negatives (letting a parallel flood through). The mitigation is to make the cap configurable per API key and to start it high, then tighten based on observed p99 in-flight counts. Do not set it from a guess.

The weakness of the reputation limiter is that the penalty box is a policy decision disguised as a technical one. How long is the punishment? Does a trip during a known outage count? Can a customer appeal? These are product questions, and they need product answers, not just code.

## Frequently Asked Questions

**How do I choose between token bucket and sliding window?**

If a provable upper bound on request rate over any interval is required, use a token bucket — the bound is `capacity + refill × T` and it holds exactly. If a simple counter that is cheap to store is enough and a small approximation at window boundaries is tolerable, a sliding-window counter is fine. The choice is about which guarantee is actually needed, not which algorithm is more modern.

**Why does a rate limiter block legitimate users after a deploy?**

A common cause is a retry storm: when a client deploy goes wrong, the client retries failed requests aggressively, and the retry traffic looks identical to an attack. The fix is not to loosen the limiter but to add a retry budget on the client side (exponential backoff with jitter, capped total retries) and to distinguish retried requests from new ones, for example with a header. Without that, any limiter will eventually misfire on your own traffic.

**Should rate limit state live in Redis or in-process?**

In-process is faster and has no network failure mode, but it only works if a given client always hits the same instance, which is rarely true behind a load balancer. Redis (or another shared store) makes the state consistent across instances but introduces a dependency: if Redis is down, the choice is fail-open or fail-closed. Failing open and logging loudly is the common pattern; the alternative is an outage caused by your own protection mechanism.

**How do I rate limit without breaking shared-IP users?**

Do not rate limit on IP alone for authenticated endpoints. Use the authenticated identity (API key, user ID) as the primary key and treat IP as a secondary signal for unauthenticated traffic. If limiting by IP is unavoidable, set the limit high enough that a large corporate NAT does not trip it, and monitor the distribution of requests per IP so anomalous behavior from a single address is visible.

**How do I measure whether a limiter is tuned correctly?**

Instrument four things: the allowed/blocked ratio per client, the p99 in-flight request count per client, the number of distinct clients that trip the limit in a rolling hour, and the retry rate of blocked requests. A limiter that is too tight shows a rising blocked ratio among clients that never retry; one that is too loose shows the protected dependency's p99 latency climbing without any corresponding limit trips. Compare before and after a limit change rather than trusting the absolute numbers.

## The one thing to check first

Open the rate limiter configuration right now and find the single number that governs burst tolerance — the bucket capacity or the window limit. Then ask: is that number derived from a measurement of downstream capacity, or was it copied from an example? If it was copied, the next step is a tuning task, not a design task, and it is the highest-leverage 30 minutes available on this today. Write down the number, the reason, and the date next to it in the config file.
