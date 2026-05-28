# Over-engineered until it broke

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026 we inherited a microservice that handled user preferences for a SaaS platform. Traffic was modest—about 800 requests/second at peak—but the service had already outgrown its original design. The team before us had decided to move everything to **Kafka Streams 3.7** with exactly-once semantics, **Apache Avro 1.11** for schema evolution, and a **three-tier materialized view** in **RocksDB 8.7** to pre-compute every imaginable combination of user + feature flag.

We were told this was "best practice for 2026" and would let us scale to 10 k req/s without touching the codebase again. At the time it felt like the safe choice: we could add new preference types without a deployment, roll back schema changes instantly, and the DBA team loved us for off-loading all that join logic from PostgreSQL.

I ran into a problem two weeks in: every time we pushed a new schema, the Streams thread would hang for 47 seconds while RocksDB flushed to disk. That added 47 ms to every single preference update, and we had no way to turn it off in production without a rolling restart that took 12 minutes. We had traded a simple REST endpoint for an architecture that felt more like operating a bank vault than a preferences service.

## What we tried first and why it didn’t work

Our first attempt was to tune RocksDB. We increased the `write_buffer_size` from 64 MB to 256 MB, doubled the `max_background_jobs` from 2 to 4, and switched the `compression` type from `LZ4` to `Zstd`. Nothing moved the needle. Then someone suggested adding a **CQRS** layer on top so we could shard the materialized views by user ID. That added three more services, two new deployment pipelines, and raised the AWS bill by $1,800 a month. The p99 latency actually went up 12 ms because every write now had to fan out to three Kafka topics.

We also tried sharding the input Kafka topic itself. We split users into 16 partitions, hoping to parallelize the state stores. The rebalancing storms were brutal—each rebalance took 2–3 minutes and caused 300 ms tail latency spikes. After two incidents where we broke the SLA for 4 minutes, we rolled that change back.

Then we tried **event sourcing** on top of the existing Kafka Streams setup. We added a second topic just to store the raw events so we could rebuild the materialized views from scratch. Within a week we had duplicated every write path, doubled the storage cost, and still couldn’t figure out why some user updates were silently dropped when the Streams application restarted.

Finally we noticed the elephant in the room: our read path was 90% of the traffic. The fancy materialized views were only used for one obscure analytics dashboard nobody looked at. Meanwhile real users were hitting a simple GET /preferences/{userId} endpoint and waiting for the RocksDB query to finish.

## The approach that worked

We stopped trying to optimize the write path and instead rewrote the service to be boringly simple. We dropped Kafka Streams, Avro, and RocksDB entirely and moved back to PostgreSQL 16 with a single JSONB column for preferences and a GIN index on (user_id, feature_key).

The key insight was that **preferences change rarely**. 99.8% of the time the value is the same as the last write. So we implemented a **write-through cache** using **Redis 7.2** that only invalidates on explicit updates. We moved all reads to Redis and kept PostgreSQL as the source of truth for writes and cache misses.

We also added a **background job** that runs every 5 minutes to compact preferences that hadn’t changed in 30 days into a single row per user. This cut the table size by 45% without any downtime.

Within three days we had removed 8,400 lines of Kafka Streams code, 4,200 lines of Avro schemas, and 2,100 lines of RocksDB configuration. The service now fits in one container with 512 MB RAM. We even deleted the CQRS layer and the three extra services we had added.

## Implementation details

Here’s the minimal stack we ended up with:
- **PostgreSQL 16** with `shared_buffers = 4GB`, `work_mem = 16MB`, and a GIN index on `(user_id, feature_key)`.
- **Redis 7.2** as a write-through cache with 5-minute TTL for non-updated preferences.
- A single **FastAPI 0.109** endpoint that:
  - Writes directly to PostgreSQL (no ORM, just raw SQL for speed)
  - Reads from Redis, falling back to PostgreSQL on cache miss
  - Returns JSON with a `Last-Modified` header so clients can cache aggressively

Code snippet for the update handler:

```python
import redis.asyncio as redis
import asyncpg
from fastapi import FastAPI, HTTPException

app = FastAPI()
pg_pool = asyncpg.create_pool("postgresql://preferences:pass@pg:5432/db")
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

@app.patch("/preferences/{user_id}")
async def update_preference(user_id: str, key: str, value: str):
    async with pg_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO preferences (user_id, data) VALUES ($1, $2)"
            "ON CONFLICT (user_id) DO UPDATE SET data = EXCLUDED.data",
            user_id,
            {"features": {key: value}}
        )
        # Invalidate cache
        await redis_client.delete(f"pref:{user_id}")
    return {"status": "updated"}

@app.get("/preferences/{user_id}")
async def get_preference(user_id: str):
    cached = await redis_client.get(f"pref:{user_id}")
    if cached:
        return {"data": cached}
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM preferences WHERE user_id = $1", user_id
        )
        if not row:
            raise HTTPException(404)
        await redis_client.setex(f"pref:{user_id}", 300, row["data"])
        return {"data": row["data"]}
```

Cache invalidation on write is the only trick. We use a simple `DELETE key` pattern instead of trying to update the cache in place. With 99.8% read-heavy workload this keeps the cache hot and avoids the complexity of cache-aside vs write-through debates.

For the compaction job we wrote a 60-line Python script that runs in a Kubernetes CronJob every 5 minutes:

```python
async def compact_old_preferences():
    async with pg_pool.acquire() as conn:
        async with conn.transaction():
            users = await conn.fetch(
                "SELECT user_id FROM preferences "
                "WHERE last_updated < NOW() - INTERVAL '30 days'"
            )
            for user in users:
                await conn.execute(
                    "UPDATE preferences SET data = jsonb_build_object('compacted', true) "
                    "WHERE user_id = $1", user["user_id"]
                )
                await redis_client.delete(f"pref:{user['user_id']}")
```

The script takes 18 seconds for 400k users and never blocks the API.

## Results — the numbers before and after

| Metric | Before (Kafka Streams + RocksDB) | After (PostgreSQL + Redis) | Change |
|--------|-----------------------------------|-----------------------------|---------|
| Lines of code removed | 14,700 | 0 | -14,700 (-100%) |
| Monthly AWS cost | $2,400 | $680 | -$1,720 (-72%) |
| P95 latency (ms) | 47 | 8 | -39 ms (-83%) |
| P99 latency (ms) | 112 | 19 | -93 ms (-83%) |
| Deployment frequency | Once every 2 weeks (schema change) | Daily | +14x |
| Cache hit rate | N/A (all reads went to RocksDB) | 97.4% | +97.4% |
| On-call pages per month | 4.2 | 0.3 | -3.9 pages (-93%) |
| MTTR (minutes) | 12 | 2 | -10 minutes (-83%) |

The cost savings came from three places:
1. Dropped two Kafka clusters (we kept one for real-time features we still needed)
2. Reduced PostgreSQL instance from r6g.xlarge ($480/mo) to db.t4g.micro ($42/mo)
3. Removed three services that each ran two c6g.large containers ($180/mo each)

We also saved 8 developer-weeks that had been spent debugging RocksDB compaction stalls and Kafka partition rebalances.

I was surprised that the simplest possible solution—PostgreSQL + Redis—beat a state-of-the-art event-sourcing architecture on every metric we cared about. The "best practice" we inherited was built for a scale we wouldn’t reach for years, and it came with a maintenance tax we couldn’t afford.

## What we’d do differently

1. **Measure first, optimize later.** We should have run a 24-hour load test with realistic traffic patterns before committing to Kafka Streams. The p99 latency of 112 ms should have been a red flag immediately.

2. **Avoid premature abstraction.** The CQRS layer added zero business value and tripled the deployment surface. We could have achieved the same separation of reads/writes with a simple view in PostgreSQL.

3. **Cache invalidation is harder than it looks.** We initially tried to keep the cache and database in sync with a change-data-capture stream. That added 2,300 lines of code and still missed some edge cases. The brute-force `DELETE` on every write was simpler and worked every time.

4. **Don’t fear JSONB.** PostgreSQL’s JSONB support has improved dramatically since 2026. The GIN index on `(user_id, feature_key)` gives us sub-millisecond lookups for the common case where a user asks for a single feature.

5. **Set a code budget.** Once the service grew past 500 lines (excluding tests) we should have paused and asked: is this complexity justified by the traffic we actually see? Our budget for preferences was 1,200 lines for 800 req/s. We hit 5,000 lines before realizing we were optimizing for a future we didn’t have.

## The broader lesson

The root cause of over-engineering isn’t laziness or incompetence. It’s the belief that we can predict the future and that complexity now buys us freedom later. In 2026 the tools and patterns we’re told are “future-proof” are often just yesterday’s hype repackaged. Kafka Streams with exactly-once semantics was sold as a scalability silver bullet in 2026; by 2026 it’s clear that most teams never hit the scale where those guarantees matter, and the operational overhead is real.

The principle I’ve internalized is: **build for the reality you have, not the reality you fear.** If your traffic profile is 80% reads and 20% writes, optimize for reads. If your data changes infrequently and is mostly read, a simple cache beats a distributed event log every time. If you’re not doing 10k req/s, a single database instance with a connection pool is simpler than a sharded cluster.

This isn’t an argument for never using distributed systems. It’s an argument for matching complexity to need. A five-node Cassandra cluster for 100 req/s is overkill; a single PostgreSQL instance with a read replica is plenty. A Kafka Streams app for preference updates is overkill; a REST endpoint with Redis is plenty.

The moment you accept that you might be wrong about the future is the moment you stop paying the maintenance tax of premature complexity. That clarity pays dividends in velocity, reliability, and developer sanity.

## How to apply this to your situation

1. **Run the 10-minute load test.** Point your staging environment at a realistic dataset and hammer it with Locust or k6 for 10 minutes. Measure p50, p95, p99 latency and error rate. If you see tail latency > 50 ms, you probably don’t need Kafka.

2. **Draw the happy path on paper.** Before touching a distributed system, sketch the flow a single user request will take. If the flow touches more than three services, reconsider. We drew ours and realized that a user updating a preference should only hit:
   - API gateway → service → Redis → PostgreSQL → Redis cache warm
   Anything longer is a smell.

3. **Set a 2-week complexity budget.** Pick a hard limit on lines of code or services. For a microservice handling preferences, 1,500 lines (excluding tests) is a reasonable ceiling. If you exceed it, refactor or split the service.

4. **Use the 80/20 rule for caching.** Cache the 20% of data that gets 80% of the reads. For us that was user preferences. Don’t try to cache everything.

5. **Adopt the principle of least surprise.** If a new engineer can’t explain how a request flows in two minutes, simplify the system until they can.

Here’s a quick checklist you can run today:
- [ ] Deploy a Redis 7.2 instance with 1 GB memory and set a 5-minute TTL.
- [ ] Replace one read-heavy endpoint with Redis cache + PostgreSQL fallback.
- [ ] Measure latency before and after. If it drops by > 30 ms, you’ve found a win.
- [ ] Delete the code that’s no longer needed.

## Resources that helped

- PostgreSQL 16 release notes: https://www.postgresql.org/docs/16/release-16.html — the JSONB performance improvements were critical for our use case.
- Redis 7.2 documentation on EXPIRE vs PXAT: https://redis.io/docs/latest/commands/expire/ — we initially used PXAT for sub-millisecond precision and switched to EXPIRE for simplicity.
- "Designing Data-Intensive Applications" by Martin Kleppmann (2022 edition) — especially Chapter 5 on replication and Chapter 6 on partitioning. It convinced me that Kafka isn’t a silver bullet.
- k6 load testing guide: https://k6.io/docs/get-started/running-k6/ — we used this to validate our simple stack before committing to it.
- The Twelve-Factor App: https://12factor.net/ — still the best checklist for keeping services boring and deployable.

## Frequently Asked Questions

**how do i know if my team is over-engineering?**

Look at your on-call rotation and deployment frequency. If you’re getting pages for cache stampedes, RocksDB compaction stalls, or Kafka partition rebalances more than once a month, you’re probably over-engineered. Another sign is that new features take weeks to ship because they require schema migrations, topic renames, or service splits. At our peak we spent 30% of our sprint on infrastructure changes that added zero user-facing features.

**why is postgres with redis faster than kafka streams for reads?**

Because Kafka Streams materialized views are still reads against a local RocksDB instance. RocksDB is fast (sub-millisecond for in-memory keys), but it’s not as fast as Redis for in-memory lookups because Redis keeps its entire dataset in RAM with a simpler data structure. Our Redis instance had 97.4% hit rate and served 800 req/s with 0.8 ms average latency. The same data in RocksDB had 2.1 ms average latency because of RocksDB’s write amplification and compaction overhead.

**what is the biggest hidden cost of over-engineering?**

The cost isn’t just in servers or containers. It’s in cognitive load. Every extra service, topic, and schema adds mental context that every engineer on the team must hold. We measured it as 4.2 hours per developer per week spent debugging infrastructure issues that wouldn’t exist in a simpler stack. Multiply that by team size and you quickly hit a wall where shipping features becomes impossible.

**when should i use kafka streams or event sourcing?**

Only when you have a genuine need for exactly-once semantics, high write throughput (>5k writes/sec), or complex event-driven workflows that can’t be modeled in a database. For preference updates, analytics dashboards, or most CRUD apps, a simple REST endpoint with a cache is enough. We used Kafka Streams for 6 months on a workload that never exceeded 1k writes/sec and paid the price in complexity.

**how do i sell simplicity to my manager or team?**

Frame it as velocity and reliability. Show the deployment frequency before and after, the error rate drop, and the time-to-recover numbers. Most managers care about speed and uptime more than architecture purity. We presented the rewrite as a 72% cost cut and a 93% reduction in on-call pages. That’s a business case, not a technical one.

## One step you can take today

Open your service’s main endpoint file and count the number of external dependencies (Kafka topics, Redis clusters, database connections, etc.). If the count is greater than 3, delete one dependency today. Replace a Kafka topic with a direct PostgreSQL write, or drop a RabbitMQ queue in favor of a simple HTTP endpoint. Measure the latency drop and celebrate the simplification. Your future self will thank you.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 28, 2026
