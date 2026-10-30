# Cache stampedes: the patterns that scale and the ones…

Most caching write-ups stop exactly where the interesting part starts: the moment a popular key expires and every client races to rebuild it at once. This article covers the root cause, the patterns that hold up, and the failure modes that show up only after deployment.

## The gap between what the docs say and what production needs

Caching clients typically expose a simple `get(key)` and `set(key, value)` pair, maybe with a TTL, and that is where the documentation ends. In practice, caching is a distributed systems problem wrapped in a key-value interface. The examples work fine in isolation, but break when real traffic, real databases, and real concurrency arrive.

A typical production failure looks like this: cache keys are derived from a session identifier plus a time window. When the window rolls over, every client holding that key misses simultaneously and races to rebuild it. If 400 clients share the key, the database receives 400 identical queries in the same instant. Database CPU saturates, p99 latency climbs by an order of magnitude, and the on-call rotation gets paged.

This is not a cache-server problem. It is a tool-use pattern problem. Teams that treat the cache as a local variable — set it once, forget it — are optimizing for the happy path and ignoring the failure modes. That works until it doesn't, and when it fails, it fails at exactly the moment the system is under the most load.

The documentation gap is wider than most engineers expect. Client libraries assume you will handle concurrency, retries, and invalidation yourself. They do not warn that a single misconfigured TTL can turn a 100 ms API call into a multi-second database query when the key expires.

The practical conclusion: design your cache usage patterns before you pick a cache layer. Otherwise the cache scales the happy path and collapses the moment reality hits.

## How stampedes actually work under the hood

Caching is a coordination problem. Every client wants the same value, and every cache miss triggers a read from the source. When that scales to hundreds of clients, a small event — a TTL expiry — can cascade into a full stampede.

A thundering herd happens when many processes race to refresh the same stale key at the same time. The first process to miss triggers a database query; if every process misses simultaneously, the database sees load proportional to the client count. This is feedback amplification: one cache miss becomes many database reads.

The key insight is that the problem is not the cache. It is the coordination mechanism around the cache. If you do not control how clients refresh stale keys, you are relying on luck to avoid a stampede.

There are three common patterns for refreshing stale keys:

1. **Lazy invalidation** — the key expires naturally, and the first client to request it refreshes it. Simple, but dangerous under load.
2. **Proactive refresh** — a background job refreshes keys before they expire, so clients rarely see stale data. Reduces misses but adds complexity and can waste resources if data rarely changes.
3. **Coordinated refresh** — clients coordinate so only one refreshes the key while others wait or use a stale value temporarily. The most robust, but requires a coordination mechanism.

Most teams default to lazy invalidation because it is the simplest. In low-traffic systems it works fine. In systems with traffic spikes or shared sessions, it fails spectacularly: the moment a popular key expires, every client that needs it races to refresh it, the database becomes the bottleneck, and latency spikes.

The trigger can be surprisingly small. A session token that expires after 30 minutes of inactivity is a classic case: users log out, but browser tabs still hold the token. When it expires, every open tab races to refresh at once. The trigger is not a traffic spike — it is a silent session expiry.

Cache eviction policy is another hidden factor. Redis supports several eviction policies, and `allkeys-lru` evicts keys aggressively under memory pressure. If keys are large or the memory limit is tight, Redis can evict keys before their TTL expires, producing extra misses on top of the scheduled ones. That extra miss pressure compounds the stampede.

The patterns that scale reduce coordination overhead and distribute refresh load. The patterns that fail assume clients will refresh keys independently and luckily avoid a stampede.

A single-node cache is also subject to single-instance bottlenecks regardless of client version. Cluster mode helps with horizontal scaling and memory distribution, but it does not solve the stampede problem by itself, because coordination for a given key still lands on one shard.

## Step-by-step implementation with real code

The walkthrough below builds a cache layer in Python using a Redis client with a coordinated refresh pattern: only one client refreshes a stale key at a time, while others wait or use a stale value.

### Step 1: Define the cache miss handler

The first step is to handle cache misses gracefully. Instead of letting every client race to refresh the key, use a distributed lock so only one client refreshes it. Others wait or use a stale value temporarily.

```python
import time
import logging
from redis import Redis
from redis.lock import Lock

logger = logging.getLogger(__name__)

class StampedeSafeCache:
    def __init__(self, redis_client: Redis, lock_timeout=5.0, stale_ttl=10.0):
        self.redis = redis_client
        self.lock_timeout = lock_timeout  # Max time to hold lock
        self.stale_ttl = stale_ttl       # How long to serve stale data

    def get(self, key: str, fetch_func, ttl: int):
        # Try to get the value from cache
        value = self.redis.get(key)
        if value is not None:
            return value

        # Cache miss: try to acquire a lock for this key
        lock = Lock(self.redis, key + ':lock', timeout=self.lock_timeout, blocking_timeout=2.0)
        acquired = lock.acquire(blocking=True)
        if not acquired:
            # Failed to get lock: serve stale data if available
            stale = self.redis.getdel(key + ':stale')
            if stale is not None:
                logger.warning(f"Using stale data for key {key}")
                return stale
            # If no stale data, just wait a bit and retry
            time.sleep(0.1)
            return self.get(key, fetch_func, ttl)

        try:
            # Re-check cache in case another client refreshed it while we waited
            value = self.redis.get(key)
            if value is not None:
                return value

            # Fetch fresh data
            value = fetch_func()
            if value is None:
                return None

            # Set the new value with TTL
            self.redis.setex(key, ttl, value)

            # Publish the new value as stale for others to use temporarily if needed
            self.redis.setex(key + ':stale', self.stale_ttl, value)
            return value
        finally:
            lock.release()
```

What this code does:

- Tries to read the key from cache first.
- On a miss, acquires a lock for the key, ensuring only one client refreshes it.
- If the lock cannot be acquired, tries stale data or waits briefly and retries.
- Once the lock is held, re-checks the cache in case another client refreshed it while waiting.
- Fetches fresh data, sets the new value, and also sets a short-lived stale copy as a fallback.

The lock timeout bounds how long a crashed holder can block others. The stale TTL bounds how long fallback data can be served. Both values should be derived from the observed cost of `fetch_func()`: if a refresh normally takes 200 ms, a 5-second lock timeout is generous; if it can take 3 seconds under load, a 5-second timeout is tight and should be raised.

### Step 2: Add a background refresher

Coordinated refresh works well for interactive requests, but it does not help background jobs or cron-like tasks. A complementary approach is to refresh keys proactively before they expire, so clients rarely see stale data.

```python
import asyncio
import time
import logging
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

async def background_refresher(redis: Redis, key_pattern: str, fetch_func, ttl: int):
    while True:
        # Scan for keys matching the pattern
        keys = []
        async for key in redis.scan_iter(match=key_pattern):
            keys.append(key.decode())

        # For each key, refresh if it's within a "refresh window"
        for key in keys:
            ttl_remaining = await redis.ttl(key)
            if ttl_remaining <= ttl // 2:  # Refresh when half the TTL is left
                try:
                    value = await fetch_func(key)
                    if value is not None:
                        await redis.setex(key, ttl, value)
                        await redis.setex(key + ':stale', ttl // 3, value)
                except Exception as e:
                    logger.error(f"Failed to refresh key {key}: {e}")

        await asyncio.sleep(5)  # Run every 5 seconds
```

This refresher scans for keys matching a pattern and refreshes them when their TTL drops below half. It sets a new TTL and updates the stale copy with a shorter TTL so it does not linger.

Two operational notes: `scan_iter` is a cursor-based scan, not a blocking `KEYS` call, so it is safe to run against a live instance, but the loop cost grows with key count. And the 5-second sleep is a starting point, not a recommendation — the correct interval is short enough that the refresher visits every hot key at least twice per refresh window.

### Step 3: Integrate with your API

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from redis import Redis

app = FastAPI()
redis = Redis(host='localhost', port=6379, db=0)
cache = StampedeSafeCache(redis, lock_timeout=5.0, stale_ttl=10.0)

def fetch_user_session(session_id: str):
    # Simulate a database query
    return {"session_id": session_id, "user_id": "user_123", "valid": True}

@app.get("/session/{session_id}")
async def get_session(session_id: str, request: Request):
    # Use the cache with a 30-minute TTL
    session_data = cache.get(
        f"session:{session_id}",
        lambda: fetch_user_session(session_id),
        ttl=1800
    )
    if session_data is None:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return session_data
```

The endpoint uses the cache to fetch session data. On a miss, it calls `fetch_user_session`, and the cache layer handles refresh coordination. Note that the sync Redis client is called from an async endpoint here; in a real deployment, use the async client (`redis.asyncio`) for request paths so the lock wait does not block the event loop.

### Step 4: Monitor and tune

The last step is to monitor cache behavior and tune TTLs, lock timeouts, and stale TTLs against real traffic. Instrument at minimum:

- `cache_hits_total` and `cache_misses_total` — the ratio is your hit rate.
- `cache_lock_wait_seconds` — a histogram, not an average; the tail is what matters.
- `cache_stale_usage_total` — how often clients fall back to stale data.
- `cache_refresh_duration_seconds` — how long `fetch_func` actually takes.
- Database query rate during cache expiry windows.

Alerts should be tied to these signals rather than to a fixed number copied from another system. A reasonable starting rule is to alert when the hit rate drops materially below its trailing baseline, or when the p99 lock wait approaches the lock timeout, because that means clients are starting to time out and fall back.

## How to measure whether your fix worked

Because every workload differs, the useful guidance is a measurement method, not a benchmark table. To evaluate a stampede fix:

1. **Establish a baseline.** Record p50, p95, and p99 request latency, database queries per second, and cache hit rate under a representative load. A load-testing tool that can simulate many concurrent clients hitting the same key is sufficient; the test script matters more than the tool.
2. **Reproduce the stampede.** Expire a hot key deliberately while the load test is running, and watch database QPS and p99 latency for the next few seconds. If nothing happens, the key is not hot enough or the load is not concurrent enough.
3. **Apply the pattern.** Enable coordinated refresh and repeat the identical test.
4. **Compare the same metrics.** The meaningful comparison is database QPS at the moment of expiry, and the shape of the p99 latency curve across the expiry window, not a single peak number.
5. **Watch the tail, not the average.** Lock waits and stale fallbacks are tail events. Averages will hide them.

The expected shape of the improvement is a flat database QPS line through the expiry event, at the cost of a small latency increase for the clients that wait on the lock. Whether that trade is worth it depends on how much headroom the database has.

## Failure modes nobody warns you about

Even with a well-designed cache layer, several failure modes appear only after deployment.

### 1. Lock contention under extreme load

The coordinated refresh pattern uses a lock per key. Under extreme load, if thousands of clients miss the same key at once, they all contend for the same lock, lock acquisition time spikes, and some clients time out.

**Fix:** shard the lock. Instead of one lock per key, derive a lock name from a hash of the key so contention spreads across a pool.

```python
import hashlib
from redis.lock import Lock

def get_lock_name(key: str, shard_count=100):
    return f"lock:{hashlib.md5(key.encode()).hexdigest()[:8]}"

lock = Lock(redis, get_lock_name(key), timeout=lock_timeout)
```

With this scheme, clients waiting on a shard that happens to be occupied will still wait, so the shard count should be chosen so that collisions are rare for your hot-key distribution. The benefit is bounded by how many distinct hot keys exist.

### 2. Stale data poisoning

The stale-data mechanism is useful, but it can poison the cache if the refresher fails or fetches bad data. If the refresher writes an incorrect stale copy, clients may serve it for the entire stale TTL.

**Fix:** validate before writing the stale copy, and keep the stale TTL short.

```python
if value is not None and is_valid(value):
    await redis.setex(key + ':stale', self.stale_ttl, value)
```

The deeper fix is to make the stale copy a last resort, not a default: log every stale serve, and alert if the rate rises, because that is an early signal that the refresher is failing.

### 3. Memory bloat from stale copies

Each key has a stale copy with a shorter TTL. At millions of keys, stale copies can consume a meaningful fraction of cache memory, which in turn triggers evictions and more misses.

**Fix:** account for stale copies in capacity planning, or avoid storing them entirely and let clients fall back to the source under a bounded timeout. If you keep them, consider a separate eviction policy or a separate logical namespace so stale copies cannot evict primary entries.

### 4. Clock skew across clients

If clients have skewed clocks, TTL calculations can differ, and expiries scatter in ways that are hard to predict. In systems spanning multiple regions or observing daylight-saving transitions, some clients may see a TTL expire at one local time and others at another.

**Fix:** compute TTLs from a consistent time source, store timestamps in UTC, and add jitter so expiries spread out.

```python
import random

def get_ttl_with_jitter(base_ttl: int, jitter_pct=0.1):
    jitter = int(base_ttl * jitter_pct)
    return base_ttl + random.randint(-jitter, jitter)

ttl = get_ttl_with_jitter(1800)  # 30 minutes ± 180 seconds
```

Jitter is the cheapest stampede mitigation available and should be applied even when coordinated refresh is in place, because it reduces the number of keys that expire in the same second.

### 5. Cache failover during a stampede

If the cache fails over to a replica during a stampede, the new primary may not have the latest data, and clients that miss will fetch from the database while the failover is still settling. That combination can produce timeouts and 5xx responses.

**Fix:** size replicas so failover does not degrade throughput, and add client-side retries with exponential backoff for cache misses during failover. Note that retries add load to the source, so bound them and pair them with a circuit breaker.

### The hidden cost of complexity

The biggest failure mode is operational. Coordinated refresh adds moving parts: lock contention, stale data usage, memory bloat, TTL tuning. Each one needs a metric and an alert, or the system becomes harder to debug than the stampede it replaced.

The rule that follows: do not add this complexity unless the measurements say you need it. If your hit rate is high and your tail latency is flat through expiry windows, you do not have a stampede problem. If database QPS spikes every time a hot key expires, you do.

## Choosing tooling by capability, not by name

Tool choice matters less than capability. When evaluating a cache and its client for stampede resistance, check for:

| Capability | Why it matters | What to check |
|---|---|---|
| Atomic set-if-absent | Needed for lock acquisition without races | Documented behavior of the set operation with a conditional flag |
| TTL introspection | The refresher needs to know remaining TTL | A command that returns remaining TTL per key |
| Non-blocking key iteration | Refreshers must not stall the server | Cursor-based scan, not a blocking keys command |
| Scripted atomic operations | Re-check-and-set must be atomic | Server-side scripting support |
| Cluster-aware locking | Locks must land on the right shard | How the client routes lock keys |
| Async client | Request paths must not block the event loop | An async API in the client library |

Background job runners and metrics systems should be selected on the same basis: retry semantics, task prioritization, and histogram support respectively. A tool that lacks the capability you need will cost more to work around than it saves.

## When this approach is the wrong choice

Coordinated refresh is not a silver bullet. It adds complexity, latency, and operational overhead. It is worth it only when:

1. **Cache misses are expensive.** If the source can absorb the miss load, the pattern adds latency for no benefit.
2. **Keys are hot and shared.** If each client has its own keys, stampedes are unlikely.
3. **Traffic is bursty or unpredictable.** Steady, low load rarely produces simultaneous misses.
4. **You have operational capacity.** Without monitoring for lock contention and stale usage, the pattern can hide problems rather than fix them.
5. **The cache is genuinely distributed.** A single-node cache can still stampede, but the coordination mechanics and failure modes differ.

A decision checklist for a specific system:

- Does a single key expire while more than a handful of clients are waiting on it?
- Does database QPS spike in a narrow window after expiry?
- Is the refresh operation idempotent and safe to run once per key?
- Can the source tolerate the retry load if the lock holder fails?
- Do you have a metric for lock wait and stale fallback?

If the first two answers are yes and the last three are yes, coordinated refresh is likely worth the complexity. If not, start with jitter and a shorter, staggered TTL, measure again, and only then add locking.

## Next 30 minutes

Pick your single hottest cache key, add jitter to its TTL, and instrument two counters — cache misses and source queries — around its expiry window. Run a load test that expires the key deliberately, and record the source query rate for the ten seconds after expiry. That number tells you whether you have a stampede problem, and it gives you the baseline you need before changing anything else.
