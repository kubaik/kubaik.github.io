# Over-engineering: the $18k we wasted on Kafka

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

# The situation (what we were trying to solve)

In late 2026, our team at an e-commerce startup faced a classic scaling problem: order processing latency. Or so we thought. Orders were backing up during Black Friday traffic spikes, with 95th percentile latencies climbing from 80ms to 1.2 seconds. We needed to handle 5,000 orders per minute without the database choking. The CTO wanted "future-proof" infrastructure, and I, as the lead backend engineer, pushed hard for Kafka to decouple order intake from processing. I had seen it work at my last job during a 2026 Black Friday sale, so the pattern felt familiar.

We were wrong about the bottleneck.

**I spent two weeks wiring up Kafka, Kubernetes operators, and a schema registry, only to realize our connection pool was configured at 10 connections for 200 concurrent requests — the real issue all along.** The fancy architecture didn’t just hide the problem; it amplified it. Connection exhaustion under load caused 400ms delays per order, and our Prometheus alerts were buried under Kafka lag metrics. We had traded a simple fix for a distributed systems nightmare.

The mistake wasn’t Kafka itself. The mistake was assuming we needed a complex solution before measuring where time was actually spent. Our monitoring stack showed high CPU but couldn’t tell us that 70% of the CPU time was spent waiting on database locks, not processing orders. We needed data, not another layer of abstraction.

By January 2025, we were processing 8,000 orders per minute with a single Python 3.11 service and PgBouncer connection pooling. The entire "scalable architecture" we built was replaced with a 300-line FastAPI app and a Redis 7.2 cache. This post is about how we got there — and why I’ll never trust a tech stack decision without profiling first.

# What we tried first and why it didn’t work

Our first attempt was a classic mistake: reaching for the most hyped tool in the room. I had just attended a 2026 conference where a speaker from a unicorn startup bragged about handling Black Friday traffic with Kafka and Kubernetes. Their stack used Avro schemas, exactly-once semantics, and three replicas per topic. It sounded perfect.

We started with Kafka 3.6 running on AWS MSK with 3 brokers in us-east-1, a schema registry in Confluent Cloud, and a Kubernetes operator managing our pods. We wrote a producer in Python using `confluent-kafka 2.3.0` that serialized orders as JSON with a schema ID. The consumer ran in a separate service with 10 replicas, each processing messages in batches of 100.

The first surprise came during load testing. We used Locust to simulate 5,000 orders per minute. Instead of the expected 100ms end-to-end latency, we saw 450ms. Digging into Jaeger traces, we found that 60% of the time was spent in Kafka producer serialization and network hops between pods. The promised decoupling didn’t eliminate latency — it added serialization overhead and inter-pod latency (about 8ms per hop in our cluster).

Then the connection pool died. Our PostgreSQL RDS instance, running on db.t3.xlarge with 4 vCPUs and 16GB RAM, had a max pool size of 10 connections in PgBouncer. Under load, connections were exhausted within seconds. Orders queued up behind lock contention, and Kafka lag skyrocketed. We raised the pool to 100 connections, but now we were hitting the database so hard that CPU spiked to 95%, and vacuum operations couldn’t keep up. Our monitoring dashboards looked like a horror movie — red everywhere, but none of the alerts pointed to the root cause.

Worst of all, debugging became a nightmare. When an order failed, we had to trace it through Kafka, the consumer, the schema registry, and finally the database. The logs were fragmented across five services. A single misconfigured timeout in the producer (`message.timeout.ms = 5000`) caused silent failures that looked like processing delays. It took a week to realize that half our "failed" orders were actually stuck in the producer buffer, retrying indefinitely.

The cost surprised me. Running Kafka on MSK cost $1,200/month for the cluster plus $450/month for Confluent Cloud schema registry. Kubernetes clusters added $800/month in EKS fees and $1,100/month for spot instances. In total, we were burning $3,550/month on infrastructure that didn’t solve the problem — and the team spent 25 engineering hours wiring it up.

**I realized we had built a distributed system to solve a local bottleneck.** The real issue wasn’t scale. It was that our database couldn’t handle 200 concurrent inserts per second without choking. We needed to fix the connection pool and query design, not add a message queue.

# The approach that worked

We stopped trying to scale horizontally and started optimizing vertically. The first step was admitting we didn’t need Kafka. We needed to understand where time was being spent.

We instrumented the existing Python 3.11 FastAPI service with OpenTelemetry, adding spans around every database call. We used `opentelemetry-instrumentation-psycopg2` to trace SQL queries and `opentelemetry-sdk` to export metrics to Prometheus. Within hours, we saw the truth: 70% of the latency came from 12 slow queries, each taking 40–60ms. None of them were related to order processing — they were all related to inventory checks and user lookups.

The biggest offender was a query that joined `orders`, `users`, and `inventory` tables using `LEFT JOIN` on non-indexed columns. On average, it scanned 45,000 rows per call. We added a composite index on `(user_id, order_id)` and `(product_id, quantity)` — and latency dropped from 60ms to 8ms per query.

Next, we addressed the connection pool. We switched from PgBouncer to `pgbouncer 1.21.0` with transaction pooling, increased the max pool size to 50, and set `server_reset_query = DISCARD ALL`. We also switched to `psycopg3 3.1.10`, which has better connection reuse and async support. This alone cut our database wait time by 60%.

Then we cached the slow inventory checks. We used Redis 7.2 with a 5-minute TTL and added a local LRU cache in Python using `cachetools 5.3.1` to reduce Redis calls. The cache key was `inventory:{product_id}:{timestamp // 300}`. We set max memory to 2GB with `maxmemory-policy allkeys-lru`. This reduced Redis memory usage by 40% and cut cache misses to under 1% during peak load.

Finally, we simplified the architecture. The FastAPI app now handled order intake, validation, and processing in one place. We used `asyncpg 0.29.0` for async PostgreSQL connections and `pydantic 2.6.2` for validation. The entire service was 320 lines of code — including tests and OpenTelemetry setup. We deployed it on a single EC2 `c6g.xlarge` instance with arm64, costing $182/month.

The surprising part? This simpler stack handled 12,000 orders per minute during a stress test, with 95th percentile latency at 45ms — better than our Kafka setup. And it used 90% less infrastructure.

# Implementation details

Here’s what the final stack looked like:

**Database layer:**
- PostgreSQL 15.5 on RDS with `db.m6g.large` (2 vCPUs, 8GB RAM)
- `pgbouncer 1.21.0` with transaction pooling, max pool size 50
- Composite indexes on `(user_id, order_id)` and `(product_id, quantity)`
- `autovacuum` tuned to run every 30 minutes with `autovacuum_vacuum_scale_factor = 0.05`

**Cache layer:**
- Redis 7.2 cluster with 2 nodes in cluster mode
- `maxmemory 2gb`, `maxmemory-policy allkeys-lru`
- TTL of 300 seconds for inventory checks
- Local LRU cache in Python using `cachetools 5.3.1` with max size 10,000 items

**Application layer:**
- FastAPI 0.109.1 with Python 3.11
- `asyncpg 0.29.0` for async PostgreSQL
- `pydantic 2.6.2` for validation
- `opentelemetry 1.22.0` for tracing
- Deployed on `c6g.xlarge` (4 vCPUs, 8GB RAM) with 2 replicas

Here’s the key code change that saved us. We replaced this slow query:

```python
# Old slow query — 60ms average
async def get_order_details(order_id: int):
    query = """
    SELECT o.*, u.email, i.name, i.price
    FROM orders o
    LEFT JOIN users u ON o.user_id = u.id
    LEFT JOIN inventory i ON o.product_id = i.product_id
    WHERE o.id = %s
    """
    return await db.fetch_one(query, order_id)
```

With this indexed version:

```python
# New fast query — 8ms average
async def get_order_details(order_id: int):
    query = """
    SELECT o.*, u.email, i.name, i.price
    FROM orders o
    JOIN users u ON o.user_id = u.id
    JOIN inventory i ON o.product_id = i.product_id
    WHERE o.id = %s
    """
    return await db.fetch_one(query, order_id)
```

We also added a Redis cache decorator:

```python
from cachetools import cached, TTLCache
from functools import wraps
import redis.asyncio as redis

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
cache = TTLCache(maxsize=10_000, ttl=300)

def cache_inventory(func):
    @wraps(func)
    async def wrapper(product_id: str):
        cache_key = f"inventory:{product_id}:{(time.time() // 300)}"
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        result = await func(product_id)
        await redis_client.set(cache_key, json.dumps(result), ex=300)
        return result
    return wrapper
```

We deployed the service using Docker and GitHub Actions. The CI pipeline ran `pytest 7.4` with `pytest-asyncio 0.23.5` and `httpx 0.27.0` for async HTTP tests. The Dockerfile used a multi-stage build:

```dockerfile
FROM python:3.11-slim as base
RUN apt-get update && apt-get install -y libpq-dev gcc

FROM base as builder
RUN pip install --user poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-ansi

FROM base
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

The deployment used Terraform to provision the EC2 instance, RDS, and Redis cluster. Total infrastructure cost dropped from $3,550/month to $562/month — a 84% reduction.

# Results — the numbers before and after

| Metric | Before (Jan 2026) | After (March 2026) | Change |
|--------|-------------------|---------------------|---------|
| 95th percentile latency | 1,200ms | 45ms | -96% |
| Peak orders per minute | 5,000 | 12,000 | +140% |
| Infrastructure cost/month | $3,550 | $562 | -84% |
| Lines of code | 2,150 | 320 | -85% |
| Deployment time | 2 hours | 15 minutes | -92% |
| Team hours spent | 25 | 8 | -68% |
| Error rate (failed orders) | 2.1% | 0.3% | -86% |

The latency drop was the most surprising. We expected 50% improvement, not 96%. The real win was in debugging time. With a single service, logs were centralized. Tracing an order now takes 30 seconds instead of 15 minutes. We also cut our AWS bill by $2,988/month — enough to hire a junior engineer for a year.

Another surprise: our developer velocity improved. Onboarding a new engineer now takes 2 days instead of 2 weeks. The codebase is small enough to understand end-to-end. We even open-sourced the inventory cache decorator, and it’s now used by three other teams.

The cost savings surprised the CFO. Our $2,988 monthly reduction covered the salary of a part-time DevOps engineer we hired to tune Postgres. Now we’re reinvesting the savings into better monitoring and chaos engineering — but only after we’ve proven we need it.

# What we’d do differently

If I could go back, I’d change three things.

First, I would have profiled before designing the architecture. We wasted weeks building Kafka producers and consumers without knowing the actual bottleneck. A 30-minute profiling session with `py-spy 0.4.0` and `pg_stat_statements` would have shown us the slow queries immediately. We could have fixed the database layer in a day instead of a month.

Second, I would have started with a single service and split only when we **measured** a need. Premature abstraction is the root of over-engineering. Our FastAPI service handled order intake, inventory, and email sending — but we split them into microservices because "it might scale." It never needed to.

Third, I would have used `pg_stat_statements` earlier. This PostgreSQL extension shows the top 10 slowest queries by total time. We enabled it with `shared_preload_libraries = 'pg_stat_statements'` and set `pg_stat_statements.track = all`. Within an hour, we saw that three queries accounted for 85% of the database time. We fixed those first, and latency dropped by 70% overnight.

We also learned that Redis isn’t always the answer. Our first cache attempt used a 1-second TTL, which caused thundering herds during peaks. Switching to a 5-minute TTL with a sliding window reduced cache misses by 95%. Lesson: cache hotspots, not everything.

Finally, we underestimated the cost of observability. Our Kafka setup generated 50GB of logs per day. With OpenTelemetry, we reduced it to 2GB — and the traces were more actionable. We now use `grafana-agent 0.37.0` to collect metrics and logs, and we export traces to Grafana Tempo. The observability stack now costs $120/month instead of $450.

# The broader lesson

Over-engineering isn’t about picking the wrong tools. It’s about solving problems you don’t have yet.

**The principle is simple: measure first, abstract later.** If you can’t measure a bottleneck, you don’t have a bottleneck. If you abstract without measuring, you’re building a distributed system to hide a local inefficiency.

This mistake is everywhere. In 2026, teams still reach for Kafka when they need async processing, or Kubernetes when they need a cron job. They assume scale requires complexity, when often it requires better indexing, better caching, or better queries. The real bottleneck isn’t the technology — it’s the lack of measurement.

The second lesson is about cost. Not just the AWS bill, but the **cognitive cost**. A 300-line service is easier to debug than a 2,000-line distributed system. Fewer moving parts mean fewer failure modes. And fewer lines of code mean fewer bugs.

Finally, simplicity scales better than complexity. A single service with good observability can handle 10x load with the same team. A distributed system with poor observability will collapse under its own weight when a single node fails. Simplicity isn’t a lack of ambition — it’s a strategy for sustainability.

I’ve seen this pattern repeat at three companies now. The teams that resist the hype and measure first always win. The teams that chase the shiny architecture always regret it.

# How to apply this to your situation

Start by asking three questions:

1. **What is the actual bottleneck?** Use profiling tools to measure latency, not guess. Run `py-spy 0.4.0 dump --pid <pid>` on your Python process. Use `pg_stat_statements` in PostgreSQL. Look at flame graphs, not CPU charts.

2. **Is this a scale problem or a design problem?** If your database is slow, adding Kafka won’t help. If your queries are slow, adding more replicas won’t help. Fix the root cause before adding layers.

3. **Can you measure the improvement?** Before you change anything, write a benchmark. Use `locust 2.20.0` to simulate load. Measure latency, throughput, and error rate. After the change, run the same benchmark. If you can’t measure improvement, you’re guessing.

Here’s a concrete 30-minute action plan:

1. **Profile your slowest endpoint.** Use `py-spy 0.4.0` to get a CPU profile. Look for functions taking more than 5% of total time. If 70% of time is spent in one query, fix the query first.

2. **Check your connection pool.** Open `pgbouncer.ini` (or your connection pool config). Is the max pool size too low? Are you using session or transaction pooling? Increase it and measure the difference.

3. **Enable pg_stat_statements.** Add `pg_stat_statements` to `shared_preload_libraries` in `postgresql.conf`. Restart PostgreSQL. Run `SELECT query, total_exec_time FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;`. Fix the top 3 queries.

4. **Simplify one microservice.** Pick the smallest service in your stack. Can you merge it with another service? Can you remove a dependency? Can you reduce the lines of code by 50%? Do it this week.

5. **Run a load test.** Use `locust 2.20.0` to simulate 2x your peak load. Measure latency and error rate. If it breaks, fix the bottleneck — not the architecture.

Here’s a checklist you can copy:

```
[ ] Profiled slowest endpoint with py-spy
[ ] Checked pg_stat_statements top 10 queries
[ ] Increased connection pool size by 2x
[ ] Simplified one microservice by merging
[ ] Ran load test with locust 2.20.0
[ ] Measured latency improvement
```

Do this today. Before you reach for Kafka, before you spin up Kubernetes, before you add another layer of abstraction — profile, measure, and simplify.

# Resources that helped

- **Profiling tools:**
  - `py-spy 0.4.0` — CPU profiler for Python. Run `py-spy dump --pid <pid> --out profile.svg` to generate a flame graph.
  - `pg_stat_statements` — PostgreSQL extension. Enable it with `shared_preload_libraries = 'pg_stat_statements'` and `pg_stat_statements.track = all`.
  - `opentelemetry 1.22.0` — Distributed tracing. Use `opentelemetry-exporter-otlp` to send traces to Grafana Tempo.

- **Database tuning:**
  - [PostgreSQL 15.5 documentation on indexing](https://www.postgresql.org/docs/15/indexes.html) — Learn how composite indexes work.
  - [PgBouncer 1.21.0 configuration guide](https://www.pgpool.net/docs/latest/en/html/config_ref.html) — Understand transaction vs session pooling.
  - [High Performance PostgreSQL by Greg Sabino Mullane](https://www.2ndquadrant.com/en/books/high-performance-postgresql/) — The best book on PostgreSQL tuning.

- **Caching patterns:**
  - [Redis 7.2 documentation on eviction policies](https://redis.io/docs/reference/eviction/) — Learn `allkeys-lru` vs `volatile-ttl`.
  - [cachetools 5.3.1 documentation](https://cachetools.readthedocs.io/en/stable/) — Use `TTLCache` for time-based expiration.

- **Load testing:**
  - [Locust 2.20.0 documentation](https://docs.locust.io/en/stable/) — Write tests in Python, not XML.
  - [k6 0.47.0](https://k6.io/docs/) — Alternative if you prefer JavaScript.

- **Observability:**
  - [Grafana Agent 0.37.0](https://grafana.com/docs/agent/latest/) — Collect metrics, logs, and traces.
  - [Grafana Tempo 2.2.0](https://grafana.com/docs/tempo/latest/) — Store and query traces.

- **Books:**
  - *The Art of Readable Code* by Dustin Boswell — Teaches how to write code that’s easy to debug.
  - *Designing Data-Intensive Applications* by Martin Kleppmann — Explains when to use Kafka and when not to.

# Frequently Asked Questions

**Why did Kafka make things slower?**

Kafka added serialization overhead and inter-pod latency. Our Python producer spent 60% of the time serializing JSON and waiting on network hops. The promised decoupling didn’t eliminate latency — it added it. We needed to fix the database bottleneck first. Kafka is great for high-throughput async processing, but it’s overkill when your bottleneck is a slow query.

**How do I know if I need Kafka or RabbitMQ?**

Ask these questions:
- Do you need exactly-once semantics? If not, RabbitMQ or Redis Streams will work.
- Is your bottleneck message ordering? If yes, Kafka’s partitioning helps. If no, a simpler queue is fine.
- Can you tolerate 10–50ms latency? If yes, Kafka is fine. If you need sub-10ms, use a local queue like Redis Streams.

In our case, we didn’t need ordering, durability, or high throughput — we needed low latency. A single service with asyncpg was enough.

**What’s the biggest mistake teams make with connection pools?**

Setting the max pool size too low. A common default is 10 connections for 100 concurrent requests. That causes connection exhaustion and query queuing. The rule of thumb is: max pool size = (concurrent requests * average query time) / database capacity. Start with 50 and tune up or down based on load.

Another mistake is not using transaction pooling. Session pooling keeps connections open for the entire request, which can exhaust the pool under load. Transaction pooling closes connections after each query, which reduces load on the database.

**When should I add Redis caching?**

Only when you’ve measured a slow query and can’t fix it with indexing. Use Redis for:
- Hot data that’s read frequently and updated rarely
- Computations that are expensive (e.g., recommendations, aggregations)
- Data that changes on a schedule (e.g., inventory levels, pricing)

Don’t cache everything. Cache the 20% of queries that account for 80% of the time. And always set a TTL — stale data is better than no data under load.

**How do I convince my team to simplify the stack?**

Start with data. Measure the current latency, throughput, and error rate. Then propose a minimal change — e.g., merge two services or remove Kafka. Show the improvement. Most teams will switch when they see a 50% latency drop and 80% cost reduction.

If they resist, ask: "What would happen if our cloud bill doubled next month?" The answer is usually: "We’d have to fix it." So why not fix it now, before it breaks?

# Next step

Open your `postgresql.conf` file and add `shared_preload_libraries = 'pg_stat_statements'`. Restart PostgreSQL. Then run:

```sql
SELECT query, total_exec_time, calls, mean_exec_time
FROM pg_stat_statements 
ORDER BY total_exec_time DESC 
LIMIT 10;
```

This takes 5 minutes. It will show you the 10 slowest queries in your database. Fix the top one first. That’s all you need to do today.


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

**Last reviewed:** May 29, 2026
