# Memory Systems for Production Agents

Production gives you neither a clean environment nor a patient timeline. Memory systems for agents fail in ways that unit tests rarely surface: state that outlives its usefulness, caches that serve stale reasoning, and sessions that accumulate until a node falls over. The gap between what documentation promises and what a live environment requires is usually found in configuration defaults, eviction behavior, and the boundary between what the agent remembers and what it should forget.

## The gap between documented behavior and production behavior

Documentation describes a system in isolation. Production runs that system next to a database, a network that drops packets, a scheduler that evicts pods, and a workload that changes shape at 3 a.m. The documented behavior of a cache is that it returns the value most recently written. The production behavior is that it returns the most recently written value *that survived eviction, serialization, and replication*. Those are different guarantees.

A common trap is assuming default configurations suffice for production workloads. Defaults are chosen to be safe for a single-node demo, not for a multi-tenant agent that holds conversation state across hours. The subtle part is not that things break loudly — it is that context leaks quietly. A session that should have expired at 30 minutes lives for six hours and slowly biases every downstream decision the agent makes.

Before adopting any memory layer, answer three questions: what is the maximum acceptable staleness, what happens when the store is full, and who is responsible for deleting state. If any answer is "the default," that is the leak.

## Three approaches and where each leaks context

The three common shapes for agent memory are in-memory stores, caching layers, and stateful services. Each leaks context differently, and the leak usually traces back to a lifecycle decision nobody made explicitly.

### In-memory stores

In-memory stores such as Redis or Memcached keep data in RAM for low-latency reads and writes. They suit session management, short-term agent scratchpads, and rate-limit counters. Context leaks appear in three places:

- **Eviction policy.** When the store reaches `maxmemory`, it must evict. Redis supports policies including `noeviction`, `allkeys-lru`, `volatile-lru`, and `allkeys-lfu`. With `noeviction`, writes fail once memory is full — which is honest but can cascade into application errors. With `allkeys-lru`, the store will happily evict a session key that a long-running agent still needs, because recency is a proxy for importance and it is a poor one. A leak here looks like an agent that "forgets" mid-task under load.
- **Network round trips.** Every read is a network hop. Code that assumes the hop is instantaneous will produce timeouts and retries under contention, and retries against a store that is already saturated make things worse.
- **Serialization.** Data must be encoded on write and decoded on read. Large JSON blobs inflate memory and CPU, and a schema change on one side of the boundary can silently produce garbage on the other.

### Caching layers

A caching layer sits in front of a slower backend to absorb read load. It is the right tool for read-heavy agent workloads: tool results, retrieved documents, embedding lookups.

Context leaks in caches come from:

- **Invalidation.** If the cache is not invalidated when the underlying data changes, the agent reasons over stale facts. This is the single most common agent-memory bug: a document is updated, the cache is not, and the agent confidently cites the old version.
- **Stampedes.** When a popular key expires, every concurrent request misses simultaneously and hits the backend. For an agent that fans out to an LLM provider, a stampede can mean a burst of duplicate, billable calls.
- **Size limits.** Caches are finite. If the working set exceeds capacity, hit rate collapses and the backend absorbs the load the cache was meant to remove.

### Stateful services

Stateful services keep state across requests — session-affine workers, actor-style runtimes, or Kubernetes StatefulSets with persistent volumes. They suit agents that must maintain a long conversation or a multi-step plan.

Leaks here are structural:

- **Session lifetime.** Sessions that are never explicitly terminated hold memory and connections. A failure mode is a service that grows steadily until it is restarted, then repeats the cycle.
- **Replication.** State replicated across nodes must converge. During a network partition, two nodes can both believe they own a session, and reconciliation can silently drop one branch of the conversation.
- **Resource footprint.** Stateful services are harder to scale and to reschedule than stateless ones. Over-provisioning to avoid eviction wastes capacity; under-provisioning turns a node failure into data loss.

## A worked example: versioned cache keys

The following example uses Redis with the `redis-py` client. It demonstrates the invalidation pattern that avoids the stale-read leak: instead of deleting cache entries on write, bump a version key and let the old entries expire naturally. This is sometimes called a versioned or generational cache.

```sh
docker run -d --name redis-cache -p 6379:6379 redis:7.2
```

```python
import redis

redis_client = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)

VERSION_SUFFIX = ":version"

def current_version(key):
    version = redis_client.get(f"{key}{VERSION_SUFFIX}")
    return int(version) if version is not None else 0

def get_data_from_backend(key):
    # Stand-in for an expensive lookup: database, API, or model call.
    return f"Data for {key}"

def read_through(key, ttl=60):
    version = current_version(key)
    cached = redis_client.get(f"{key}:{version}")
    if cached is not None:
        return cached, True

    # Stampede guard: only one caller fetches; others wait briefly.
    lock_key = f"{key}:lock"
    got_lock = redis_client.set(lock_key, "1", nx=True, ex=5)
    if not got_lock:
        # Another worker is fetching. Re-check the cache once.
        cached = redis_client.get(f"{key}:{current_version(key)}")
        if cached is not None:
            return cached, True
        return get_data_from_backend(key), False

    try:
        data = get_data_from_backend(key)
        redis_client.set(f"{key}:{version}", data, ex=ttl)
        return data, False
    finally:
        redis_client.delete(lock_key)

def invalidate(key):
    # New generation. Old keys age out via TTL.
    redis_client.incr(f"{key}{VERSION_SUFFIX}")

def write_and_invalidate(key, new_value, ttl=60):
    # Persist to the source of truth first, then invalidate.
    persist_to_backend(key, new_value)
    invalidate(key)
    redis_client.set(f"{key}:{current_version(key)}", new_value, ex=ttl)

def persist_to_backend(key, value):
    # Placeholder for the actual write path.
    pass
```

Two design points are worth stating explicitly, because both are common sources of bugs.

First, **write the source of truth before invalidating the cache**. If the order is reversed, a concurrent reader can miss the cache, read the old value from the backend, and repopulate the cache with stale data — and that entry will survive until its TTL expires.

Second, **increment the version rather than deleting keys**. Deletion races: a reader that fetched the old value just before the delete can write it back afterward. A version bump makes the old key unreachable immediately, and TTL cleans it up. The cost is that old generations occupy memory until they expire, so the TTL must be short relative to the write rate.

The stampede guard above uses a single lock key with a 5-second expiry. This is a coarse mechanism: if the backend call takes longer than the lock TTL, a second caller will also fetch. That is an acceptable trade for most workloads, but it means the lock TTL must be tuned above the p99 backend latency, not the average.

## How to measure instead of guessing

Every number that matters here is measurable on your own infrastructure. The following is what to instrument and how to compare, rather than a borrowed benchmark.

**Latency distribution.** Record the full histogram, not the mean. In Python, wrap the Redis call and record elapsed time; export to Prometheus as a histogram and read p50, p95, and p99 from Grafana. The mean hides the tail, and the tail is where agent timeouts live.

**Hit rate.** Redis exposes `keyspace_hits` and `keyspace_misses` via `INFO stats`. Hit rate is `hits / (hits + misses)`. Track it over time; a falling hit rate is the earliest signal that the working set has outgrown the cache or that a key scheme is wrong.

**Eviction pressure.** `INFO stats` also reports `evicted_keys`. If this is climbing, the store is discarding data you asked it to keep. Cross-reference with `used_memory` against `maxmemory`.

**Memory per item.** Measure directly: load a representative sample of your real payloads, read `used_memory` before and after, and divide. Serialization format, key length, and data structure overhead all matter, and the only honest number is the one from your own data.

**Stampede behavior.** Under a controlled load test, expire a hot key and count backend calls in the following second. If the count equals the request count, the guard is not working.

A simple instrumentation wrapper for the read path:

```python
import time
from prometheus_client import Histogram, Counter

CACHE_LATENCY = Histogram("cache_read_seconds", "Cache read latency")
CACHE_RESULT = Counter("cache_reads_total", "Cache reads", ["result"])

def timed_read(key):
    start = time.perf_counter()
    data, hit = read_through(key)
    CACHE_LATENCY.observe(time.perf_counter() - start)
    CACHE_RESULT.labels(result="hit" if hit else "miss").inc()
    return data
```

Run this for a week before changing any configuration. The data will tell you whether the problem is capacity, key design, TTL, or backend latency — and those have different fixes.

## Failure modes that rarely appear in documentation

**Cache stampede.** Many concurrent misses on one key produce a burst of backend load. Mitigations: a distributed lock (shown above), request coalescing, or probabilistic early expiration, where a fraction of readers refresh the key slightly before it expires.

**Stale reads after writes.** The cache and the source of truth diverge because invalidation happened before or instead of the write. Mitigation: write-then-invalidate, versioned keys, and a short TTL as a backstop.

**Unbounded session growth.** Sessions are created but never closed. Mitigation: an explicit session TTL enforced by the store, not by application code that might skip a cleanup path.

**Eviction of live state.** The store evicts keys the agent still needs because the policy treats all keys as equal. Mitigation: separate the stores. Put ephemeral cache data in one instance with `allkeys-lru` and durable session state in another with `noeviction` or `volatile-*` policies, so a cache miss never destroys a session.

**Replication divergence.** Two nodes disagree about session state during a partition. Mitigation: a single writer per session, or a consensus-backed store if the workload genuinely requires it.

**Memory fragmentation.** Long-running stores can hold more RSS than `used_memory` reports. Mitigation: monitor the ratio and restart or reshard on a schedule if it drifts.

## Decision checklist

Use this to pick an approach before writing code.

| Requirement | In-memory store | Caching layer | Stateful service |
|---|---|---|---|
| Sub-millisecond reads | Yes | Yes | Depends on implementation |
| Survives node restart | Only with persistence enabled | No by default | Yes, with persistent volumes |
| Strong consistency | Not by default | No | Achievable with consensus |
| Handles long conversations | Possible, with explicit TTLs | No | Yes |
| Scales horizontally | Yes | Yes | Harder; needs partitioning |
| Operational complexity | Low | Low | High |

Choose an in-memory store when state is short-lived and loss is tolerable. Choose a caching layer when the goal is absorbing read load in front of a slower backend. Choose a stateful service when state must survive restarts and requests must be routed to the node holding it. If two rows conflict, the harder requirement wins.

## When this approach is the wrong choice

- **Low read/write volume.** If the backend handles the load comfortably, a cache adds a consistency problem in exchange for nothing. Measure before adding a layer.
- **Strong consistency required.** Caches and eventually consistent replicas cannot provide it. Use a database with the consistency model you actually need.
- **Limited operational capacity.** A stateful service is a commitment: backups, failover, upgrades, and partition management. If that is not staffed, a managed database is the better answer.
- **Regulated data.** Cached personal data inherits every retention and deletion obligation of the source. If deletion requests must be honored promptly, a cache with opaque TTLs is a compliance risk.

## What to do in the next 30 minutes

Run these two commands against your production Redis instance and read the output:

```sh
redis-cli config get maxmemory
redis-cli config get maxmemory-policy
redis-cli info stats | grep -E "evicted_keys|keyspace_hits|keyspace_misses"
```

If `maxmemory` is `0`, the store has no memory ceiling and will consume the host's RAM until the OS intervenes. If `maxmemory-policy` is `noeviction` and the store holds session state, writes will begin failing under pressure. If `evicted_keys` is nonzero and climbing, the store is discarding data you asked it to keep. Note the three values, then decide whether the current policy matches what the data in that instance actually is.
