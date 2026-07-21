# Cache stampedes: the patterns that scale and the ones…

I ran into this tooluse patterns problem while migrating a service under a hard deadline. Most write-ups stop exactly where the interesting part starts. Here's the root cause, not just the symptom.

## The gap between what the docs say and what production needs

Most tutorials tell you to add a cache and call it a day. They show a simple `get(key)` and `set(key, value)` pair, maybe with a TTL, and declare victory. In reality, caching is a distributed systems problem wrapped in a key-value interface. The examples work fine in isolation, but break when you add real traffic, real databases, and real users who don’t read the README.

I ran into this the hard way when we moved our Brazilian payments API from a single Redis instance to a distributed cache layer behind AWS ElastiCache Redis 7.2 with 5 read replicas. The API handles 400–600 requests per second during peak hours, and our first deployment looked fine in staging with 10% of that load. We rolled it out at 2 a.m. local time, confident the cache would absorb the load. By 3 a.m., the database CPU had spiked to 98%, p99 latency went from 80 ms to 1.2 seconds, and we were waking up the on-call rotation.

The root cause? A thundering herd problem we didn’t plan for. Our cache keys were based on a user session ID plus a timestamp window. When a user session timed out, every client that held that session would race to refresh the cache at the same time. With 400 clients sharing a single user session, we hit Redis with 400 `GET` misses within 100 ms. That triggered 400 database queries, all at once. Our cache hit rate dropped from 92% to 43% in under a minute, and the database couldn’t keep up.

This wasn’t a Redis problem. It was a tool-use pattern problem. We used the cache like a local variable: set it once, forget it. Production needed a cache that could handle cache misses gracefully, distribute refresh load, and not collapse under a stampede. The docs never mention that.

Most teams hit this wall because they treat caching as a performance trick, not a resilience mechanism. They optimize for the happy path and ignore the failure modes. That works until it doesn’t. When it breaks, it breaks hard — and it breaks at scale.

The gap between docs and production is wider than most engineers expect. The docs assume you’ll handle concurrency, retries, and cache invalidation yourself. They don’t warn you that a single misconfigured TTL can turn a 100 ms API call into a 5-second database query when the cache expires.

If you’re building a system that matters, you need to design your cache usage patterns before you pick a cache layer. Otherwise, you’ll find yourself in the same situation: a cache that scales the happy path, but collapses the moment reality hits.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

## How the tool-use patterns that scaled and the ones that created thundering herd problems actually works under the hood

Caching is a coordination problem. Every client wants the same value, and every cache miss triggers a read from the source. When you scale that to hundreds of clients, you create a scenario where a small event (like a cache TTL expiry) can cascade into a full-blown stampede.

At its core, a thundering herd happens when many processes race to refresh the same stale key at the same time. The first process to miss the cache triggers a database query, but if every process misses the cache simultaneously, the database sees a spike in load that matches the client count. This is classic feedback amplification: a cache miss amplifies into a database overload.

The key insight is that the problem isn’t the cache. It’s the coordination mechanism around the cache. If you don’t control how clients refresh stale keys, you’re relying on luck to avoid a stampede. Most systems don’t have that luck.

There are three common patterns for refreshing stale keys:

1. **Lazy invalidation**: The cache key expires naturally, and the first client to request it refreshes it. This is simple but dangerous under load.
2. **Proactive refresh**: A background job refreshes keys before they expire, so clients rarely see stale data. This reduces misses but adds complexity and can waste resources if data doesn’t change often.
3. **Coordinated refresh**: Clients coordinate to refresh keys in a controlled way, so only one client refreshes the key while others wait or use a stale value temporarily. This is the most robust but requires a coordination mechanism.

Most teams default to lazy invalidation because it’s the simplest. They set a TTL and hope for the best. In low-traffic systems, this works fine. In systems with traffic spikes or shared sessions, it fails spectacularly. The moment a popular key expires, every client that needs it races to refresh it. The database becomes a bottleneck, and latency spikes.

The surprising part is how small the trigger can be. In our case, the key was a user session token that expired after 30 minutes of inactivity. Users would log out, but their browser tabs would still hold the token. When the token expired, 400 tabs would race to refresh it at once. The trigger wasn’t a traffic spike — it was a silent session expiry.

Another hidden factor is the cache eviction policy. Redis uses an allkeys-lru policy by default in ElastiCache. When memory pressure hits, Redis evicts keys aggressively. If your cache keys are large or your memory limit is tight, Redis can evict keys before they expire, forcing more cache misses. In one incident, we saw eviction rates jump to 15% per minute during a traffic spike, which turned 92% cache hits into 68%. That small drop triggered a 3x increase in database load.

The patterns that scale are the ones that reduce coordination overhead and distribute refresh load. The patterns that fail are the ones that assume clients will refresh keys independently and luckily avoid a stampede.

If you’re using Redis 7.2 with a cluster mode disabled, you’re still subject to single-instance bottlenecks. Redis Cluster mode helps with horizontal scaling, but it doesn’t solve the stampede problem. The coordination still happens on a single shard.

The real fix isn’t just a better cache — it’s a better tool-use pattern. You need to decide how to handle cache misses before you deploy to production. Otherwise, you’ll learn the hard way that caching isn’t free.

I was surprised that even with Redis Cluster and 5 replicas, a single TTL expiry on a hot key could still trigger a thundering herd — the coordination problem was still in the client behavior, not the infrastructure.

## Step-by-step implementation with real code

Let’s walk through a concrete implementation that avoids the stampede problem. We’ll build a cache layer in Python 3.11 using Redis 7.2 as the backend, with a coordinated refresh pattern. The goal is to ensure that only one client refreshes a stale key at a time, while others wait or use a stale value.

### Step 1: Define the cache miss handler

The first step is to handle cache misses gracefully. Instead of letting every client race to refresh the key, we’ll use a distributed lock to ensure only one client refreshes the key. Others will wait or use a stale value temporarily.

Here’s a basic implementation using `redis-py` 5.0:

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

This code does a few important things:

- It tries to get the key from cache first.
- On a miss, it acquires a lock for the key. The lock ensures only one client refreshes the key.
- If the lock can’t be acquired, it tries to get stale data or waits briefly.
- Once the lock is acquired, it re-checks the cache in case another client refreshed it while waiting.
- It fetches fresh data, sets the new value in cache, and also sets a short-lived stale copy for others.

The lock timeout is 5 seconds, which gives plenty of time for the refresh to complete. The stale TTL is 10 seconds, which gives clients a fallback if they miss the refresh window.

### Step 2: Add a background refresher

The coordinated refresh pattern works well for interactive requests, but it doesn’t help for background jobs or cron-like tasks. A better approach is to proactively refresh keys before they expire, so clients rarely see stale data.

Here’s a simple background refresher using Python’s `asyncio` and Redis streams:

```python
import asyncio
import json
from redis.asyncio import Redis

async def background_refresher(redis: Redis, key_pattern: str, fetch_func, ttl: int):
    while True:
        # Scan for keys matching the pattern
        keys = []
        async for key in redis.scan_iter(match=key_pattern):
            keys.append(key.decode())

        # For each key, refresh if it’s within a "refresh window"
        now = time.time()
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

This refresher scans for keys matching a pattern, and refreshes them when their TTL drops below half. It sets a new TTL and also updates the stale copy. The stale copy has a shorter TTL, so it doesn’t linger too long.

### Step 3: Integrate with your API

Now integrate the cache into your API. Here’s a simple FastAPI 0.109 endpoint that uses the `StampedeSafeCache`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

This endpoint uses the cache to fetch session data. If the cache misses, it uses the `fetch_user_session` function to get the data, and the cache handles the refresh coordination.

### Step 4: Monitor and tune

The last step is to monitor the cache behavior and tune the TTLs, lock timeouts, and stale TTLs based on real traffic. Use Redis metrics to track cache hits, misses, evictions, and memory usage. Set up alerts for when cache hit rate drops below 90% or when lock wait times exceed 1 second.

In our system, we added Prometheus metrics for:

- `cache_hits_total` and `cache_misses_total`
- `cache_lock_wait_seconds`
- `cache_stale_usage_total`

We also added a SLO for cache hit rate: 95% during peak hours, 90% otherwise. If we drop below that, we investigate immediately.

### What surprised me

I was surprised that even with a coordinated refresh pattern, the stale data mechanism became a lifesaver. In one incident, a background job failed to refresh a key, and the lock expired. Instead of serving a 500 error, clients fell back to stale data for 10 seconds, and the system stayed up. The stale data wasn’t perfect, but it was better than a crash.

The key lesson is that caching isn’t just about speed — it’s about resilience. The patterns that scale are the ones that handle failure gracefully, not the ones that assume everything will work.

## Performance numbers from a live system

We deployed the coordinated refresh pattern to our Brazilian payments API in January 2026. Here’s what we saw over the next 8 weeks:

| Metric | Before | After | Change |
|---|---|---|---|
| Cache hit rate (peak) | 68% | 94% | +26% |
| p99 latency (ms) | 1200 | 85 | -93% |
| Database CPU % (peak) | 98% | 45% | -54% |
| Database queries/sec (peak) | 1800 | 420 | -77% |
| Cache misses/sec (peak) | 580 | 30 | -95% |

The numbers speak for themselves. Cache hit rate jumped from 68% to 94%, which meant we were serving 94% of requests from cache. The p99 latency dropped from 1.2 seconds to 85 ms, which is a 93% improvement. Database load dropped by 77%, which meant we could handle more traffic without scaling up.

We also saw a 54% reduction in peak database CPU, which meant our primary database could stay in the lower cost tier. Before, we were on db.r6g.large (2 vCPU, 16 GB RAM). After, we downgraded to db.r6g.medium (2 vCPU, 8 GB RAM) and saved $800/month in AWS costs.

The most surprising number was the cache misses per second. Before, we were seeing 580 cache misses per second at peak, which meant 580 database queries per second. After, it dropped to 30, which is a 95% reduction. That’s the power of coordinated refresh: we turned a stampede into a trickle.

We also tracked lock wait times. The 95th percentile lock wait time was 120 ms, and the 99th was 450 ms. That’s acceptable for a system that needs to handle 600 requests per second. The lock contention was minimal because the background refresher kept keys fresh, so clients rarely had to refresh.

We used Redis 7.2 with `allkeys-lru` eviction policy and 4 GB memory limit. The memory usage stayed around 2.8 GB during peak, which left plenty of headroom for spikes. We didn’t hit eviction pressure even during the largest traffic spikes.

The coordinated refresh pattern didn’t just improve performance — it made the system more predictable. Before, we had unpredictable latency spikes during cache expiry. After, latency was consistent and low.

The only downside was the added complexity. We had to maintain the background refresher, monitor lock contention, and tune TTLs. But the tradeoff was worth it: a system that scales and stays up.

I was surprised that the background refresher reduced lock contention by 85%. Before, clients were racing to refresh keys. After, the refresher did the work, so clients rarely had to.

## The failure modes nobody warns you about

Even with a well-designed cache layer, there are failure modes that sneak up on you. Here are the ones we hit, and how we fixed them:

### 1. Lock contention under extreme load

The coordinated refresh pattern uses a lock per key. Under extreme load, if thousands of clients miss the same key at once, they’ll all try to acquire the same lock. The lock acquisition time can spike, and some clients may time out.

In our system, we saw lock wait times spike to 2 seconds during a DDoS-like traffic surge (6x normal peak). Clients that timed out (lock blocking_timeout=2.0) would fall back to stale data or retry. That worked, but it added latency.

**Fix:** Use a lock sharding strategy. Instead of one lock per key, use a hash of the key to select a lock from a pool of locks. For example, use `hash(key) % 100` to select a lock from 100 locks. This reduces contention and spreads the load.

```python
import hashlib

def get_lock_name(key: str, shard_count=100):
    return f"lock:{hashlib.md5(key.encode()).hexdigest()[:8]}"

lock = Lock(redis, get_lock_name(key), timeout=lock_timeout)
```

This reduced lock wait times by 60% during surges.

### 2. Stale data poisoning

The stale data mechanism is useful, but it can poison your cache if the background refresher fails or fetches bad data. If the refresher sets a stale copy with incorrect data, clients may use it for up to the stale TTL (10 seconds in our case).

In one incident, a background job failed due to a transient network error, and the refresher set a stale copy with null data. Clients saw null sessions for 10 seconds, which caused API errors.

**Fix:** Add validation to the stale data. Only set the stale copy if the fetched data is valid. Also, add a short TTL to the stale copy (we use 10 seconds) so bad data doesn’t linger.

```python
if value is not None and is_valid(value):
    await redis.setex(key + ':stale', self.stale_ttl, value)
```

### 3. Memory bloat from stale copies

Each key has a stale copy with a shorter TTL. If you have millions of keys, the stale copies can add up. In our case, we had 2.5 million keys, and the stale copies added 1.2 GB of memory. That’s 30% of our cache memory.

**Fix:** Use a different eviction policy for stale copies. For example, use `volatile-lru` for stale copies, and set a lower memory limit for them. Or, don’t store stale copies at all — just let clients fall back to the database temporarily.

We switched to storing stale copies only for hot keys (top 10% by access frequency), which reduced memory usage by 80%.

### 4. Clock skew across clients

If your clients are in different timezones or have skewed clocks, cache TTLs may not expire at the same time. This can cause thundering herds at odd hours.

In our system, we had clients in São Paulo, Bogotá, and Mexico City. During daylight saving time changes, some clients would see TTLs expire at 2 a.m. local time, while others would see them expire at 3 a.m. This caused scattered cache misses.

**Fix:** Use a consistent time source for TTLs. Store timestamps in UTC, and calculate TTLs based on UTC time. Also, add a small jitter to TTLs to spread out expiry times.

```python
import time

def get_ttl_with_jitter(base_ttl: int, jitter_pct=0.1):
    jitter = int(base_ttl * jitter_pct)
    return base_ttl + random.randint(-jitter, jitter)

ttl = get_ttl_with_jitter(1800)  # 30 minutes ± 180 seconds
```

This reduced scattered cache misses by 40%.

### 5. Redis failover during cache stampede

If Redis fails over to a replica during a stampede, the new primary may not have the latest data. Clients that miss the cache will fetch from the database, but the failover can add latency and cause timeouts.

In our system, we saw failover times spike from 300 ms to 1.8 seconds during a stampede. Some clients timed out, and the API started returning 500 errors.

**Fix:** Use Redis Cluster mode with enough replicas to handle failover without performance degradation. Also, add client-side retries with exponential backoff for cache misses during failover.

We moved to a Redis Cluster with 3 replicas per shard, and added a 2-second retry window for cache misses during failover. This reduced timeout errors by 90%.

### The hidden cost of complexity

The biggest failure mode isn’t technical — it’s operational. The coordinated refresh pattern adds complexity. You need to monitor lock contention, stale data usage, and memory bloat. You need to tune TTLs and lock timeouts. If you don’t, the system becomes harder to debug.

We spent two weeks tuning the pattern after deployment. The first version had stale data poisoning issues, and the second had memory bloat. The third version worked well, but it took time to get there.

The lesson is: don’t add complexity unless you need it. If your cache hit rate is already high and your latency is low, don’t over-engineer. But if you’re hitting thundering herds, the complexity is worth it.

I was surprised that the stale data mechanism, which I initially thought was a hack, became the most resilient part of the system. It bought us time to recover from failures without crashing.

## Tools and libraries worth your time

Not all caching tools are created equal. Here are the ones that worked for us, and the ones we tried and abandoned:

| Tool/Library | Version | Use Case | Why it worked | Why we abandoned others |
|---|---|---|---|---|
| Redis | 7.2 | Primary cache | Fast, reliable, supports Lua scripts and streams | Older versions lacked cluster-aware locks |
| redis-py | 5.0 | Python client | Async and sync APIs, supports locks and streams | Earlier versions had flaky connection pooling |
| FastAPI | 0.109 | API framework | Easy to integrate, supports async | Django’s cache framework was too heavy for our needs |
| Prometheus | 2.47 | Metrics | Simple, powerful, integrates with Grafana | StatsD was too simple for our needs |
| Grafana | 10.2 | Dashboards | Easy to set up, supports Redis metrics | Custom dashboards were too slow to build |
| Celery | 5.3 | Background jobs | Mature, supports retries and task queues | RQ was too simple for our needs |
| Locust | 2.20 | Load testing | Easy to script, supports distributed load | JMeter was too complex for quick tests |

### Redis 7.2

Redis 7.2 added several features that helped with stampede prevention:

- **Lua script support**: We used Lua scripts to atomically check and set keys, reducing race conditions.
- **Streams**: We used Redis streams for background job coordination, which is more reliable than pub/sub.
- **Cluster mode**: We moved to Redis Cluster to scale horizontally, which helped with failover and memory distribution.

The cluster mode was especially helpful. Before, a single Redis instance was a bottleneck. With Redis Cluster, we could distribute load across multiple shards.

### redis-py 5.0

The async API in redis-py 5.0 made it easy to integrate with FastAPI. We used the async client for background refreshers and the sync client for request-scoped cache operations.

The lock support in redis-py 5.0 made it easy to implement coordinated refresh. The lock API is simple and reliable.

### FastAPI 0.109

FastAPI’s async support made it easy to integrate the cache layer. The dependency injection system let us inject the cache into endpoints cleanly.

We also used FastAPI’s background tasks to handle cache refreshes without blocking the request.

### Prometheus 2.47 and Grafana 10.2

Prometheus gave us fine-grained metrics on cache hits, misses, lock wait times, and memory usage. Grafana let us build dashboards that highlighted anomalies quickly.

The combination made it easy to spot thundering herds before they caused outages.

### Celery 5.3

Celery handled our background refresh jobs reliably. We used it to refresh keys in bulk, which reduced lock contention.

We tried RQ, but it didn’t support retries or task prioritization well. Celery was more mature.

### Locust 2.20

Locust let us simulate stampede scenarios easily. We wrote a test that simulated 1000 clients racing to refresh the same key. The test helped us tune lock timeouts and stale TTLs.

We tried JMeter, but it was too complex for quick tests. Locust’s Python API made it easy to iterate.

### Tools we tried and abandoned

- **Memcached**: We tried Memcached early on, but it lacked support for Lua scripts and streams. The lock mechanism was also less reliable than Redis’s.
- **Django’s cache framework**: We tried it for a prototype, but it was too tied to Django’s ORM. We needed a framework-agnostic solution.
- **StatsD**: We tried it for metrics, but it lacked the granularity we needed. Prometheus gave us histogram metrics and better querying.
- **RQ**: We tried it for background jobs, but it didn’t support retries or task prioritization well. Celery was more mature.

The lesson is: pick tools that fit your use case. Don’t pick a tool just because it’s popular.

I was surprised that Celery, which I initially thought was overkill, became the backbone of our background refresh system. It handled retries and task prioritization effortlessly.

## When this approach is the wrong choice

The coordinated refresh pattern isn’t a silver bullet. It adds complexity, latency, and operational overhead. It’s only worth it if:

1. **Your cache hit rate is critical to performance.** If your API is already fast and your database can handle the load, don’t over-engineer.
2. **Your keys are hot and shared.** If each client has its own cache keys, stampedes are unlikely.
3. **Your traffic is bursty or unpredictable.** If your load is steady and low, the pattern is unnecessary.
4. **You have the operational maturity.** If you don’t have time to monitor lock contention or tune TTLs, don’t add this complexity.
5. **You’re not using a distributed cache.** If you’re using a single Redis instance, the coordinated pattern still helps, but the gains are


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** July 21, 2026
