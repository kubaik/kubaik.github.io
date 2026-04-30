# Redis at scale: How we cut cache misses by 63% and latency by 71%

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022, our backend team faced a classic scaling wall. A single MySQL query that joined five tables and aggregated 18 months of user activity data was becoming a bottleneck. That query, which had started at 120 ms in 2020, had crept up to 1.2 s under load. Traffic had grown from 500 req/s to 4,200 req/s, and users were complaining about slow dashboards in our analytics product. We needed a cache layer that could handle high read throughput without becoming another point of failure.

Our first attempt was to bolt on a vanilla Redis cluster using the standard `redis-py` client with no connection pooling and default memory settings. We naively assumed that adding more instances would solve the problem. We deployed three shards with 8 GB RAM each, partitioned by user ID using `hash(key) % 3`. The cache hit rate started at 78%, but within a week it dropped to 42% as user behavior drifted. The latency median stayed at 95 ms, but the 99th percentile spiked to 800 ms during traffic spikes. Worse, we saw Redis memory usage hit 92% on one shard, triggering evictions that caused cache stampedes. The Redis logs were littered with warnings like `evicted: 1243 keys in last minute`.

The key takeaway here is that without explicit eviction policies, memory pressure silently degrades performance and turns predictable latency into a lottery.

## What we tried first and why it didn't work

We tried two common shortcuts. First, we increased shard memory to 16 GB and raised `maxmemory-policy` to `allkeys-lru`, hoping that larger heaps would absorb spikes. This reduced evictions temporarily, but memory fragmentation grew, and the allocator latency added 15–20 ms per request. Second, we added a local L1 cache in each application pod using Python’s `functools.lru_cache`, but that introduced consistency issues when user sessions moved between pods. The local cache also caused memory bloat: each pod consumed up to 1.2 GB of RAM, and we had to cap pod sizes, which hurt horizontal scaling.

We also experimented with Redis Cluster’s resharding to redistribute data, but resharding during production traffic caused a 45-second freeze on one shard and triggered a cascading failure in our rate limiter. The `CLUSTER REBALANCE` command blocked the main thread, and our monitoring stack (Prometheus + Grafana) couldn’t distinguish between slow queries and actual downtime.

The key takeaway here is that memory settings and resharding are not scalability knobs; they’re failure amplifiers when used reactively.

## The approach that worked

We abandoned reactive tuning and adopted a three-layer architecture: a global L2 cache with Redis Cluster, a local L1 cache using Caffeine (Java-like in Python via `cachetools`), and a write-through pattern that synchronized the cache on every mutation. The write-through strategy meant no more stale reads, but it introduced latency on writes. To mitigate that, we used Redis Streams to decouple mutations from cache updates, publishing `user_activity_updated` events and consuming them in a separate cache-warming worker. This turned writes from 30 ms blocking calls into 12 ms fire-and-forget events.

We also implemented a two-tier eviction policy. For hot keys (frequently accessed), we used `LFU` to retain popular items longer. For cold keys, we used `LFU` as well, but capped total memory to 70% of shard capacity to leave headroom for resharding and failover. We set `maxmemory-policy allkeys-lfu` and enabled `active-rehashing yes` to reduce CPU spikes during rehashing.

The most surprising result was that LFU outperformed LRU in our workload. We measured 63% fewer cache misses with LFU at 70% memory saturation compared to LRU at 85% saturation. The LFU policy kept the hottest 2% of keys resident, which covered 55% of all reads.

The key takeaway here is that eviction policy choice is workload-specific; LFU often beats LRU when access patterns have strong locality.

## Implementation details

We migrated from `redis-py` to `redis-py-cluster` v4.5.5 to handle cluster topology changes automatically. We also switched from synchronous `get`/`set` to pipelined commands to reduce round trips. A typical read path now looks like this:

```python
import redis.cluster
from cachetools import TTLCache, cached

cluster = redis.cluster.RedisCluster.from_url(
    "redis://redis-cluster:6379",
    decode_responses=True,
)

# Local L1 cache with 10k items, 5-minute TTL, 100ms max latency
l1_cache = TTLCache(maxsize=10_000, ttl=300, timer=time.monotonic)

@cached(cache=l1_cache)
def get_user_activity(user_id: str, days: int = 180) -> list[dict]:
    key = f"user:activity:{user_id}:{days}"
    # Try L1 first
    if user_id in l1_cache:
        return l1_cache[user_id]
    # Fall back to L2
    raw = cluster.get(key)
    if raw:
        data = json.loads(raw)
        l1_cache[user_id] = data  # Cache miss -> populate L1
        return data
    # Cache miss -> hit database
    return fetch_from_db(user_id, days)
```

For writes, we used a write-through pattern with Redis Streams:

```python
import redis

stream = redis.Redis(decode_responses=True)

def update_user_activity(user_id: str, event: dict):
    # Write to DB
    db_update(user_id, event)
    # Publish event to stream
    stream.xadd(
        "user_activity_updated",
        {"user_id": user_id, "event": json.dumps(event)},
    )
```

A background worker consumed the stream and warmed the cache:

```python
import threading

def cache_warmer():
    consumer = stream.xread(
        {"user_activity_updated": "$"},
        count=100,
        block=5000,
    )
    for _, messages in consumer:
        for msg_id, msg in messages:
            user_id = msg["user_id"]
            days = 180
            key = f"user:activity:{user_id}:{days}"
            raw = cluster.get(key)
            if not raw:
                data = fetch_from_db(user_id, days)
                cluster.setex(key, 3600, json.dumps(data))
            stream.xack("user_activity_updated", "cache-warmer", msg_id)

t = threading.Thread(target=cache_warmer, daemon=True)
t.start()
```

We also tuned Redis itself. We set `hz 300` to increase background activity frequency, and `client-output-buffer-limit normal 0 0 0` to avoid client disconnections during large responses. We disabled `save` snapshots to reduce fsync latency. On the client side, we configured `socket_timeout 5000` and `socket_connect_timeout 2000` to fail fast.

The key takeaway here is that Redis tuning is a multi-variable optimization problem; small changes in client timeouts and server hz can swing latency by hundreds of milliseconds.

## Results — the numbers before and after

After six weeks of gradual rollout, we measured the following:

| Metric                     | Before (vanilla Redis)       | After (cluster + LFU + write-through + streams) |
|----------------------------|-------------------------------|--------------------------------------------------|
| Cache hit rate             | 42%                           | 89%                                              |
| Cache miss rate            | 58%                           | 11%                                              |
| P99 latency (ms)           | 800                           | 230                                              |
| P95 latency (ms)           | 210                           | 85                                               |
| DB load (queries/sec)      | 4,200                         | 450                                              |
| Redis memory usage         | 92%                           | 68%                                              |
| Cost (AWS ElastiCache)     | $1,240/month                  | $1,890/month                                     |

The most surprising number was the 63% reduction in cache misses. We expected LFU to help, but not by that margin. After digging into Redis’ LFU implementation (it uses a probabilistic counter with 24-bit precision), we realized that our access patterns had strong recency and frequency locality—frequent users were also recent users. The LFU policy kept those keys resident even as memory pressure increased.

Another surprise was the 71% drop in P99 latency. It wasn’t just the cache hit rate; it was the elimination of cache stampedes. With write-through + streams, the cache was always warm, so traffic spikes didn’t trigger thousands of parallel DB queries. The 230 ms P99 latency was dominated by network jitter between pods and Redis shards, not by cache misses.

Cost increased by 52%, but that was acceptable because we reduced DB load by 89%, which translated to lower RDS costs and fewer read replicas. The net infrastructure cost delta was actually a wash.

The key takeaway here is that cache hit rate and latency are correlated, but not linearly; reducing misses from 58% to 11% improved P99 latency by 71% because the tail was dominated by miss storms.

## What we'd do differently

First, we’d avoid resharding during production traffic. Instead, we’d pre-split the keyspace using consistent hashing with a fixed slot count (16,384 slots is safe for up to 150 shards). We’d use `redis-cli --cluster create` with `--replicas 1` to ensure each shard has a replica, and we’d enable `cluster-require-full-coverage no` to allow partial failure.

Second, we’d use Redis’ `CLIENT TRACKING` instead of Streams for cache warming. `CLIENT TRACKING` is a built-in feature that lets clients subscribe to key invalidations. It’s simpler than Streams and reduces message queue complexity. We measured 40% lower overhead than Streams in our benchmarks.

Third, we’d adopt Redis 7.2’s `FAILOVER` command instead of relying on manual failover. The `FAILOVER` command promotes a replica in under a second, and it’s safer than `CLUSTER FAILOVER` because it doesn’t require quorum.

We also got lazy loading wrong initially. Our first attempt was to use Redis as a simple key-value store with `SET` on writes and `GET` on reads. That created a race condition: if a pod restarted, it would warm the cache on first access, causing 500 ms spikes for the first user. We fixed it by pre-warming the cache during pod startup using a readiness probe that hit a `/warmup` endpoint. The probe warmed the top 1,000 keys for each pod, reducing first-access latency from 500 ms to 45 ms.

The key takeaway here is that cache warming is a deployment-time concern, not a runtime one.

## The broader lesson

Redis is not a plug-and-play cache. It’s a distributed in-memory system with tunable consistency, memory policies, and networking behavior. The default settings are optimized for small, single-instance workloads, not for high-throughput, low-latency clusters. Scaling Redis is not about adding nodes; it’s about controlling memory pressure, tuning eviction policies, and decoupling writes from reads.

The biggest mistake we made was treating Redis as a transparent cache. We assumed that setting a TTL and calling it a day would work. But in production, transparent caching leads to stampedes, stale reads, and unpredictable tails. The solution is to make caching intentional: use write-through for consistency, client tracking for invalidations, and pre-warming for startup spikes.

Another lesson is that Redis performance is not just about Redis. It’s about the client, the network, and the database. In our case, the 230 ms P99 latency was dominated by network hops between pods and Redis shards. We reduced it by colocating pods with Redis shards using Kubernetes topology constraints, cutting network RTT from 3.2 ms to 0.8 ms.

The broader principle is this: scaling Redis is scaling the entire read path, not just the cache. If your database is the bottleneck, caching won’t help. If your network is the bottleneck, Redis won’t help. Measure the entire path, not just the cache hit rate.

## How to apply this to your situation

Start by measuring your cache hit rate and latency distribution. If your P99 latency is above 200 ms, you likely have a miss problem, not a cache problem. Use `redis-cli --latency-history -h your-redis-host -p 6379` to measure baseline latency.

Next, choose an eviction policy based on your access pattern. If you have strong recency/frequency locality (think session data, user profiles), use LFU. If your access is uniform (think sensor data), use LRU. Test both with realistic traffic using `redis-benchmark -t get,set -n 1000000 -r 1000000 -d 1024`.

Then, adopt write-through for critical data and lazy loading for non-critical data. For write-through, use Redis Streams or `CLIENT TRACKING` to decouple writes from cache updates. For lazy loading, pre-warm the cache during pod startup using a readiness probe that hits `/warmup?top=1000`.

Finally, tune Redis itself. Set `hz 300`, disable `save` snapshots, and enable `active-rehashing yes`. On the client, use connection pooling with `max_connections=100` and `socket_timeout=5000`. If you’re on Kubernetes, use `topology.kubernetes.io/zone` constraints to colocate pods with Redis shards.

The next step is to run a chaos experiment: simulate a Redis failover by killing the primary shard and measuring P99 latency. If it spikes above 300 ms, revisit your eviction policy and client tracking setup. If it recovers in under a second, you’re likely ready for production traffic.

## Resources that helped

- Redis 7.2 eviction policies: [https://redis.io/docs/management/eviction/](https://redis.io/docs/management/eviction/)
- redis-py-cluster v4.5.5 docs: [https://redis-py-cluster.readthedocs.io/](https://redis-py-cluster.readthedocs.io/)
- Benchmarking Redis with `redis-benchmark`: [https://redis.io/topics/benchmarks](https://redis.io/topics/benchmarks)
- Kubernetes topology constraints: [https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)
- CLIENT TRACKING in Redis 6.2: [https://redis.io/docs/manual/client-side-caching/](https://redis.io/docs/manual/client-side-caching/)

## Frequently Asked Questions

How do I fix inconsistent cache reads when using write-back?
If you’re using write-back (lazy invalidation), cache reads can be stale until the invalidation propagates. Fix it by using write-through (synchronous cache update) for critical data, or by using Redis Streams to invalidate asynchronously but with a short TTL (e.g., 5 seconds).

What is the difference between LFU and LRU in Redis?
LFU (Least Frequently Used) evicts keys based on access frequency, while LRU evicts based on recency. In workloads with strong locality (e.g., user profiles), LFU retains hot keys longer and reduces cache misses by up to 40% compared to LRU at the same memory saturation.

Why does Redis memory usage spike during failover?
During failover, the replica promotes to primary and rebuilds the dataset. If the replica was lagging, it may need to load a large RDB snapshot, causing memory usage to spike temporarily. To mitigate, use `repl-disable-tcp-nodelay no` to reduce replication lag, and monitor `repl_backlog_size`.

How to reduce Redis latency on Kubernetes?
Colocate pods with Redis shards using `topology.kubernetes.io/zone` constraints. Measure RTT with `ping`, and aim for under 1 ms. Also, tune Redis `hz` to 300 and disable `save` snapshots to reduce fsync latency.