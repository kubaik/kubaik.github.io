# Store data once, serve everywhere: cheaper

## What the multi-region docs leave out

Provider documentation covers the control plane well: latency-based DNS routing, global accelerators, CDN edge functions. What it covers poorly is the cost structure that appears once real traffic flows.

A common failure mode looks like this: a team replicates compute into three regions, points latency-based DNS at them, and ships. Each region's application still reads from the original database. Every request from a distant region becomes a cross-region round trip. Latency rises by the speed-of-light floor plus TLS and query time, and the egress line item grows with read volume, not write volume. Compute was cheap to duplicate; data access was not.

The reason is data gravity. Replicating a database into N regions means paying for replication traffic continuously, whether or not anyone reads those replicas. Caching reads at the edge inverts that: you pay for data movement only on cache misses, and the miss rate is a tunable you control.

The rest of this article works through a read-heavy API design that keeps one authoritative database and serves most reads from regional caches, including the invalidation logic that makes it safe.

## The two cost drivers nobody puts on the slide

**Replication transfer.** Managed cross-region replication is billed per GB transferred. Exact prices vary by provider, region pair, and direction, and they change. The relevant point is structural: replication cost scales with *dataset size × number of regions × time*, independent of traffic. A 1 TB dataset replicated to three regions moves roughly 1 TB per region on initial sync, then incremental changes thereafter. If you want a real number, pull the current per-GB cross-region rate from your provider's pricing page and multiply by your dataset size and region count. Treat any figure you see quoted in a blog post (including this one) as stale until you verify it.

**Read egress.** When application servers in region B query a database in region A, the response bytes cross a region boundary and are billed. This cost scales with *read volume × response size*. It is the one that surprises teams, because it looks like normal database traffic in application code.

A useful measurement exercise: instrument your database client to log response bytes per query, tagged with the region of the caller and the region of the database. Aggregate by day. If cross-region bytes dominate, caching is the lever. If write volume dominates, it is not.

## How regional caching actually works

The design has four moving parts.

**One authoritative database.** All writes go to a single primary. This keeps consistency reasoning simple and avoids multi-master conflict resolution, which is a much harder problem than most teams want to take on.

**A cache in each serving region.** The cache holds serialized read results keyed by resource ID and any dimension that affects the response (locale, region preference, API version). TTLs are short — seconds to minutes — so staleness is bounded by design rather than by invalidation correctness alone.

**Explicit invalidation on write.** After a successful write, the application deletes the affected cache keys in every region. Deletion, not update, so the next read repopulates from the source of truth.

**A stale-serving fallback.** If the cache misses and the origin is slow or unavailable, serve a stale copy with a short revalidation window. This converts an outage into degraded freshness.

The correctness argument: a read is either a cache hit (bounded staleness, at most the TTL) or a miss (fresh from the primary). A write invalidates all cached copies before returning. The window where a client can read stale data after a write is the gap between the write committing and the invalidation completing — which is why invalidation should be synchronous in the write path, not queued.

## Worked example: when caching beats replication

State the assumptions explicitly, because the answer flips if they change.

Assume a read-heavy API with:
- 1 TB of relational data
- Three serving regions
- 100 reads per second per region, average response 4 KB
- 1 write per second per region

**Option A: replicate the database to all three regions.**
Replication transfer is roughly dataset size per region on initial sync, plus incremental change traffic. Steady-state cost scales with write volume and dataset size. Read egress is zero, because reads are local. But you now operate three database clusters, three backup schedules, three upgrade paths, and a replication topology.

**Option B: one primary, regional caches.**
Cross-region traffic occurs only on cache misses. At a 90% hit rate, cross-region read volume is 10 reads/sec/region × 4 KB = 40 KB/sec/region, or about 3.4 GB/day/region. Writes still cross regions for invalidation, but invalidation messages are tiny — a key name, not a payload.

The arithmetic that matters here is the ratio, not the dollar figure: Option B moves roughly (1 − hit rate) × read volume across region boundaries, while Option A moves a function of dataset size and write volume. For read-heavy workloads with a high hit rate, B moves far less data. Plug in your provider's current per-GB rate to get a number; do not trust a number from an article.

The break-even point is a write-heavy workload with a low cache hit rate. If most requests are writes, or if cache keys are so diverse that hit rates stay low, Option B pays invalidation overhead for little benefit.

## Implementation: origin, cache, and invalidation

A minimal FastAPI service with a single PostgreSQL primary and a per-region Redis cache.

```python
# main.py
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import redis
import os
import time

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

DB_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

r = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)


def get_db():
    return psycopg2.connect(DB_URL, connect_timeout=2)


@app.get("/user/{user_id}")
async def get_user(user_id: str, x_region: str = Header("us-east-1", alias="X-Region")):
    cache_key = f"user:{user_id}:{x_region}"

    cached = r.get(cache_key)
    if cached is not None:
        return {"user_id": user_id, "data": cached, "source": "cache"}

    start = time.time()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT data FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="not_found")

    latency_ms = (time.time() - start) * 1000
    r.set(cache_key, row[0], ex=30)
    return {
        "user_id": user_id,
        "data": row[0],
        "latency_ms": round(latency_ms, 2),
        "source": "origin",
    }
```

Two things to note. The `Header` default is the actual value, with `alias` mapping the HTTP header name — the earlier form where the default was a header name was wrong. And the cache stores the column value, not the whole row tuple, so what you read back is what you wrote.

Invalidation on write, using `SCAN` rather than `KEYS`:

```python
@app.post("/user/{user_id}")
async def update_user(user_id: str, payload: dict):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET data = %s WHERE id = %s",
                (payload["data"], user_id),
            )
        conn.commit()
    finally:
        conn.close()

    pattern = f"user:{user_id}:*"
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor=cursor, match=pattern, count=100)
        if batch:
            r.delete(*batch)
        if cursor == 0:
            break

    return {"updated": user_id}
```

`SCAN` is safe to run against a live cache; `KEYS` blocks the server on large keyspaces. The loop terminates when the returned cursor is zero. Deleting in batches of 100 keeps each round trip bounded.

## Regional fan-out for warm caches

Invalidation deletes; it does not warm. After a write, the next read in each region is a miss. For frequently read records, that means a burst of cross-region reads right after every write.

A durable queue per region solves this. The write path publishes a small event (user ID plus the new value) to each region's queue. A regional consumer writes the value into that region's cache. The cache is warm before the next read arrives.

Redis Streams with consumer groups work for this, as do managed queues. The important properties are durability (the event survives a consumer restart) and at-least-once delivery (duplicate writes to a cache are idempotent, so this is fine).

```python
# consumer.py
import json
import os
import redis

r = redis.Redis.from_url(os.getenv("REDIS_URL"))

STREAM = "user_updates"
GROUP = "region_group"
CONSUMER = os.getenv("AWS_REGION", "local")

try:
    r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
except redis.exceptions.ResponseError:
    pass  # group already exists

while True:
    response = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=10, block=5000)
    if not response:
        continue
    for _stream_name, messages in response:
        for msg_id, fields in messages:
            user_id = fields[b"user_id"].decode()
            data = fields[b"data"].decode()
            r.set(f"user:{user_id}:{CONSUMER}", data, ex=30)
            r.xack(STREAM, GROUP, msg_id)
```

Unlike pub/sub, a consumer group tracks which messages each consumer has acknowledged, so a restarted consumer resumes where it left off. The stream is trimmed separately by a scheduled job; deleting each message immediately after acknowledgment removes the durability benefit for any consumer that is temporarily offline.

## Measuring whether it worked

Do not trust a hit-rate number from anywhere, including this article. Measure your own.

Instrument three counters at the cache client: `hits`, `misses`, and `bytes_returned_from_origin`. Emit them per region per minute. The hit rate is `hits / (hits + misses)`; the cross-region byte volume is the sum of `bytes_returned_from_origin` for regions that are not the primary's region.

For latency, measure p50 and p99 at the edge, not in the application. A cache hit that still traverses a slow proxy is not a fast request. Compare the p99 of a region against the p99 of a single-region deployment serving the same endpoints; the difference is the cost of the topology, and it should be small.

For cost, pull the actual line items from your provider's billing export and separate cross-region transfer from intra-region. If cross-region transfer is not the dominant variable cost, caching is not your lever — look at compute or storage instead.

## Failure modes and how to handle each

**Cache stampede.** When a hot key expires, many concurrent requests miss simultaneously and hit the primary. Mitigate with a per-key lock: `SET lock:{key} 1 NX PX 5000`. The first request acquires the lock and refreshes; others serve the stale value or wait briefly. Alternatively, add jitter to TTLs so keys do not expire in lockstep.

**Invalidation gaps.** If the write commits but the invalidation call fails, the cache serves stale data until the TTL expires. Keep TTLs short enough that this window is acceptable, and log every failed invalidation. Do not retry invalidation in a background worker unless the TTL is longer than the retry window.

**Clock skew.** TTLs are enforced by the cache server's clock, not the application's. If cache nodes drift, effective TTLs vary. Run NTP on cache hosts and monitor drift. This is a real concern but a small one; the fix is operational, not architectural.

**Unbounded key spaces.** If every request produces a unique key — search queries with arbitrary parameters, paginated results — the cache fills with single-use entries and the hit rate collapses. Either normalize keys (drop irrelevant parameters, round pagination) or do not cache those endpoints.

**Cold start after a region comes online.** A new region's empty cache sends every request to the primary. Pre-warm by replaying the top-N most-read keys before routing traffic to the region, or ramp traffic gradually so the cache fills at a rate the primary can absorb.

**Memory growth.** Caches hold more than the sum of their values: per-key overhead, fragmentation, and connection buffers all count. Set a `maxmemory` policy and monitor eviction rate. If evictions climb while hit rate falls, your working set does not fit and you need a larger cache or a smaller TTL.

## When not to use this

**Strong cross-region consistency is required.** Financial ledgers, inventory with hard reservation semantics, and anything requiring serializable transactions across regions are poor fits. Use a database designed for it and accept the latency and cost.

**Writes dominate reads.** Caching helps reads. If your workload is 80% writes, the cache is overhead.

**Data residency rules forbid it.** Some jurisdictions require that personal data never leave the country. A shared primary in another region violates that regardless of caching. You need regional databases and a replication strategy that respects the boundary — caching does not solve a legal constraint.

**Your traffic is already regional.** If 95% of users are within one region, the operational cost of multi-region caching exceeds the benefit. One region plus a disaster-recovery replica is simpler.

## A decision checklist

Before committing to regional caching, answer these:

- What is the read-to-write ratio on the endpoints you plan to cache?
- What is the realistic cache hit rate given your key space? Estimate from request logs, not intuition.
- What is the maximum staleness your product tolerates? That sets your TTL ceiling.
- What is the current cross-region egress cost per month, from the billing export?
- Can your write path tolerate a synchronous invalidation call? If not, what is the failure behavior?
- What is the p99 latency from each target region today?
- Do any compliance rules restrict where the data can be stored or processed?

If the read-to-write ratio is high, the hit rate is estimable above roughly 80%, staleness of seconds is acceptable, and egress dominates your variable cost, caching is the right lever. Otherwise, replicate and pay for it deliberately.

## What to do in the next 30 minutes

Pick your slowest endpoint and measure it from a region far from your database. Run a loop of 100 requests from a machine in that region, recording the response time of each, and compute the p99. Then check your provider's billing export for cross-region transfer on that same endpoint's traffic. If the p99 exceeds your target and cross-region transfer is a visible line item, you have the two numbers that justify a cache — and you can size the TTL from the staleness your product already tolerates.
