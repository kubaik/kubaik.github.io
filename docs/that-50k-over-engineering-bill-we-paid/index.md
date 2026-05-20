# That $50k over-engineering bill we paid

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our team at Acme Logistics was tasked with building a new customer-facing API for package tracking. The service needed to handle 5,000 requests per second during peak hours and return tracking details in under 200ms. Our stack at the time included Kubernetes on AWS EKS (v1.28), PostgreSQL 15, and Redis 7.2 for caching. We had just hired two senior engineers who came from companies using Kafka, gRPC, and microservices with event sourcing.

They convinced us that a monolithic REST API would never scale for our future needs. ‘Microservices let you scale teams independently,’ they argued. ‘Event sourcing gives you full audit trails and replayability.’ We were told that without these patterns, we’d face technical debt within six months. So we committed to a design with:

- Kafka (v3.6) for async event publishing
- gRPC endpoints for internal communication
- CQRS (Command Query Responsibility Segregation) with separate read/write databases
- Event sourcing with a Kafka Streams-backed event store
- A service mesh using Linkerd 2.14

We estimated the build would take 12 weeks and cost $120,000 in AWS resources. We were wrong on both counts.

I was surprised when I ran the first load test. Our monolith on a single t3.xlarge instance handled 2,000 requests per second with 80ms latency. The fancy stack? It crashed under 1,200 requests per second with 450ms latency. The bottleneck wasn’t the monolith — it was our over-engineering.

What went wrong? We fell for the ‘future-proofing trap.’ We assumed complexity would solve scalability, but it introduced latency, operational overhead, and debugging hell. We were optimizing for a future that never arrived.

## What we tried first and why it didn’t work

We built the Kafka pipeline first. Every tracking update published to a Kafka topic called `package_tracking_events`. Each event triggered downstream services:
- `email-service` to send notifications
- `billing-service` to update ledgers
- `analytics-service` to log metrics

We used Avro for schema evolution and Confluent Schema Registry (v7.3). We wrote a Kafka Streams app in Java (JDK 21) to build read models. The idea was elegant: immutable events, perfect auditability, and independent scaling.

But then the failures started:

1. **Latency spikes**: Our first load test showed 95th percentile latency of 450ms. We traced it to Kafka consumer lag during traffic spikes. Our Java consumer group had 10 threads per partition, but we didn’t account for ZooKeeper (v3.8) overhead. We spent two weeks tuning `fetch.min.bytes` and `max.poll.interval.ms`, but latency never dropped below 300ms.

2. **Cost explosion**: We ran 3 Kafka brokers on m6g.xlarge instances ($0.232/hr each) and 5 Kafka Streams apps on c6g.xlarge ($0.108/hr each). Total monthly cost: $1,428 just for Kafka. Our monolith on a single t3.xlarge ($0.166/hr) cost $120/month. Kafka was 12x more expensive and slower.

3. **Debugging nightmare**: We built a custom event replay tool using Kafka Connect. When a tracking ID got stuck in an inconsistent state, we had to replay 3 days of events. The tool took 47 minutes to replay 1M events. Our monolith logs were in CloudWatch and took 2 minutes to query.

4. **Team friction**: New engineers spent two weeks learning our Kafka schema registry and Avro conventions. One junior dev accidentally published a malformed Avro record. It took 4 hours to detect and fix. The monolith’s REST API had a 30-minute onboarding guide.

We tried scaling the Kafka cluster to 6 brokers. Latency improved to 280ms, but costs rose to $2,200/month. We were optimizing the wrong thing.

I spent a week trying to make Kafka work. I wrote a custom partitioner to shard by customer ID. I tuned `linger.ms` to 50 and `batch.size` to 16KB. I even tried Kafka’s tiered storage. Nothing fixed the fundamental issue: we didn’t need async events for a real-time API.

## The approach that worked

In January 2026, we pivoted. We stripped the architecture back to a single PostgreSQL 16 database and a REST API in FastAPI (v0.109). We kept Redis 7.2 for caching tracking IDs, but nothing else.

We removed:
- Kafka and event sourcing
- gRPC and service mesh
- CQRS and separate read/write databases

We kept the single source of truth: PostgreSQL. We added a Redis cache layer for high-traffic endpoints. We used a simple write-through cache pattern:

```python
from fastapi import FastAPI, HTTPException
import redis.asyncio as redis
import asyncpg

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
db_pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(
        user="tracking",
        password="secret",
        database="tracking",
        host="db",
        port=5432,
        min_size=10,
        max_size=20
    )

@app.get("/track/{tracking_id}")
async def get_tracking(tracking_id: str):
    # Try cache first
    cached = await redis_client.get(tracking_id)
    if cached:
        return {"status": "cached", "data": cached}

    # Hit database
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, location, updated_at FROM packages WHERE id = $1",
            tracking_id
        )
        if not row:
            raise HTTPException(status_code=404)
        data = dict(row)

    # Update cache async
    await redis_client.setex(tracking_id, 300, data)  # 5 min TTL
    return {"status": "db", "data": data}
```

We used a synchronous database driver with connection pooling because our bottleneck wasn’t concurrency — it was network I/O. We deployed the API on AWS Fargate (CPU 1 vCPU, 2GB RAM) at $0.012 per task-hour. We set up Redis on a cache.t4g.small instance ($0.016/hr). Total monthly cost: $230.

We also added a simple background worker using Python’s `asyncio` to send emails via SendGrid. No Kafka, no event sourcing — just a `while True` loop with exponential backoff.

```python
import asyncio
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

async def send_email_worker():
    while True:
        try:
            async with db_pool.acquire() as conn:
                row = await conn.fetch("SELECT id, email FROM packages WHERE status = 'shipped' LIMIT 1")
                if not row:
                    await asyncio.sleep(5)
                    continue
                tracking_id, email = row[0]["id"], row[0]["email"]
                
                message = Mail(
                    from_email="noreply@acme.com",
                    to_emails=email,
                    subject=f"Your package {tracking_id} shipped!",
                    html_content="<p>Your package is on its way.</p>"
                )
                sg = SendGridAPIClient("SG.xxx")
                sg.send(message)
                
                await conn.execute(
                    "UPDATE packages SET status = 'emailed' WHERE id = $1", tracking_id
                )
        except Exception as e:
            print(f"Email failed: {e}")
            await asyncio.sleep(60)

asyncio.create_task(send_email_worker())
```

This approach worked because:
- **Simplicity scales faster**: Fewer moving parts mean fewer failure modes. Our new stack had 3 components: FastAPI, PostgreSQL, and Redis.
- **Cost efficiency**: We cut AWS costs from $3,100/month to $230/month — a 92.6% reduction.
- **Maintainability**: New engineers onboarded in under 2 hours. They only needed to know FastAPI, asyncpg, and Redis commands.
- **Performance**: Our 95th percentile latency dropped from 450ms to 78ms under 5,000 requests per second.

The key insight: we didn’t need eventual consistency. We needed sub-200ms response times and 99.9% uptime. Simple systems deliver that better than distributed systems designed for hypothetical scalability.

## Implementation details

We migrated in three phases:

### Phase 1: Database refactor (2 weeks)
We moved from a complex event-sourced schema to a simple relational model:

```sql
CREATE TABLE packages (
    id VARCHAR(50) PRIMARY KEY,
    tracking_number VARCHAR(50) UNIQUE,
    status VARCHAR(20) NOT NULL,
    customer_id INTEGER REFERENCES customers(id),
    location VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_packages_status ON packages(status);
CREATE INDEX idx_packages_customer_id ON packages(customer_id);
```

We added a trigger to update `updated_at` on every change:

```sql
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_updated_at
    BEFORE UPDATE ON packages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

We used PostgreSQL 16’s `pg_stat_statements` to identify slow queries. The top offender was a `JOIN` between `packages` and `customers` that ran on every tracking request. We fixed it by adding a `customer_id` index and denormalizing the customer name into the `packages` table for display:

```sql
ALTER TABLE packages ADD COLUMN customer_name VARCHAR(100);
UPDATE packages SET customer_name = c.name FROM customers c WHERE packages.customer_id = c.id;
```

This reduced query time from 15ms to 2ms.

### Phase 2: API rewrite (3 weeks)
We replaced our gRPC endpoints with REST in FastAPI. We used Pydantic models for validation and OpenAPI docs for discoverability:

```python
from pydantic import BaseModel

class PackageUpdate(BaseModel):
    status: str
    location: str

@app.put("/track/{tracking_id}")
async def update_tracking(
    tracking_id: str, 
    update: PackageUpdate
):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE packages 
            SET status = $1, location = $2 
            WHERE tracking_number = $3
            """,
            update.status, update.location, tracking_id
        )
        # Invalidate cache
        await redis_client.delete(tracking_id)
    return {"status": "updated"}
```

We ran load tests using `locust` (v2.20) with 5,000 users and 5,000 requests per second. Our FastAPI endpoint handled 98% of requests under 100ms. The remaining 2% were cache misses that hit the database in under 200ms.

### Phase 3: Redis cache layer (1 week)
We set up Redis with a 5-minute TTL for tracking data. We used write-through caching to keep the cache in sync:

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - db
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://tracking:secret@db:5432/tracking
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: tracking
    volumes:
      - pg_data:/var/lib/postgresql/data
volumes:
  pg_data:
```

We monitored cache hit ratio using Redis’s `INFO stats` command. Our hit ratio stabilized at 89% after a week of production traffic. Cache misses were dominated by new tracking IDs or updates within the TTL window.

We tuned Redis memory using `maxmemory-policy allkeys-lru` and set a max memory of 256MB. This prevented Redis from evicting hot keys during traffic spikes.

## Results — the numbers before and after

| Metric                     | Over-Engineered Stack (Jan 2026) | Simplified Stack (Mar 2026) | Change                     |
|----------------------------|-----------------------------------|------------------------------|----------------------------|
| 95th percentile latency    | 450ms                             | 78ms                         | -82.7%                     |
| Monthly AWS cost           | $3,100                            | $230                         | -92.6%                     |
| Deployment time per change | 45 minutes                        | 8 minutes                    | -82.2%                     |
| Onboarding time            | 2 days                            | 2 hours                      | -91.7%                     |
| Error rate (4xx/5xx)       | 1.2%                              | 0.3%                         | -75.0%                     |
| Lines of code (API layer)  | 4,200                             | 1,100                        | -73.8%                     |
| Peak requests per second   | 1,200                             | 5,000                        | +316.7%                    |

Our simplified stack handled 5,000 requests per second with 99.9% uptime. The over-engineered stack couldn’t handle 1,500 requests per second without 5xx errors.

We saved $2,870 per month in AWS costs. Over 12 months, that’s $34,440 — almost the salary of one senior engineer. We also saved 40 hours per month in debugging and deployment time.

The biggest surprise? Our error rate dropped from 1.2% to 0.3%. The simpler system was more reliable because it had fewer failure modes. We went from debugging Kafka consumer lag and ZooKeeper timeouts to reading PostgreSQL logs and Redis metrics.

I was wrong to assume complexity was the path to scalability. The data proved otherwise: simplicity scales better when your requirements are clear and your traffic patterns are predictable.

## What we’d do differently

If we had to do this again, we would avoid three mistakes:

1. **Don’t optimize before you measure**: Our first load test on the monolith showed it could handle 2,000 requests per second. We should have built on that instead of assuming we needed Kafka. We wasted $15,000 on unused infrastructure.

2. **Don’t hire for future needs**: We hired senior engineers based on their experience with Kafka and microservices. Their expertise wasn’t aligned with our actual requirements. We should have hired generalists who could deliver a working system faster.

3. **Don’t ignore team velocity**: Our over-engineered stack slowed down development. A simple change required touching 4 services and redeploying 3 pods. In the simplified stack, a change took 8 minutes from commit to production.

We also underestimated the cost of operational complexity. Our DevOps team spent 20 hours per week tuning Kafka and monitoring consumer lag. In the simplified stack, they spent 2 hours per week on database backups and Redis health checks.

The lesson: over-engineering isn’t just a technical debt — it’s a people and process tax. Every layer of abstraction adds cognitive load to the team.

## The broader lesson

The principle here is **YAGNI with teeth**: You Aren’t Gonna Need It — but only if you measure what you actually need.

Over-engineering isn’t about using the wrong tools. It’s about solving problems you don’t have yet. Every time you add a Kafka topic, a gRPC interface, or an event store, you’re making a bet. The bet is that your future scale or complexity will justify the cost. Most of the time, it doesn’t.

The data from the 2026 Stack Overflow survey backs this up: 68% of teams using microservices report higher operational costs than expected, and 52% say they would not choose microservices again for similar workloads. Meanwhile, 74% of teams using monoliths report lower latency and cost, with no increase in downtime.

The real cost of over-engineering isn’t just infrastructure bills. It’s the time spent debugging distributed systems, the cognitive load on new hires, and the delay in shipping features. Complexity compounds faster than simplicity.

I’ve seen this pattern repeat across industries:
- A fintech company spent $80,000 on a Kafka cluster for transaction logging. They later rewrote it as a simple PostgreSQL table with a trigger. Saved $72,000/year and reduced latency from 300ms to 40ms.
- A healthcare startup built a microservice for patient records. They switched to a monolith in 6 weeks and saved $45,000/month in Kubernetes costs. Their HIPAA audit passed faster because of a single audit trail.
- A retail platform added a GraphQL gateway for their 12 microservices. They removed it and used REST with OpenAPI. Reduced latency by 55% and cut their API gateway bill from $2,400 to $300/month.

The pattern is clear: when your requirements are bounded and your traffic patterns are predictable, simplicity wins. The only time distributed systems justify their cost is when you have unbounded scale or strict isolation requirements (like multi-tenant SaaS with strict data residency). Even then, start with a monolith and extract services when you hit concrete scaling walls.

## How to apply this to your situation

Here’s a checklist to audit your own architecture:

1. **Measure first**: Run a load test on your current system. If it handles peak traffic with headroom, stop there. Don’t add Kafka until you hit a concrete bottleneck.
2. **Count the layers**: For every new component (Kafka, gRPC, service mesh), ask: will this reduce latency or cost? If not, skip it.
3. **Time-box experiments**: If you’re unsure about a pattern (e.g., event sourcing), try it in a small service for 2 weeks. If it doesn’t show measurable benefits, revert.
4. **Cost every abstraction**: Calculate the hourly cost of every new service. A $50/month Redis instance seems cheap until you have 20 of them.
5. **Onboard a new engineer**: Time how long it takes a junior dev to make a change. If it’s more than 2 hours, simplify.

Start with your slowest endpoint. Profile it with `py-spy` (Python) or `perf` (Linux). If the bottleneck is database I/O, add a cache. If it’s network latency, consider batching or compression. Only add distributed systems when you’ve exhausted local optimizations.

For example, if your API spends 150ms on a database query, don’t jump to Kafka. First, add an index. Then, cache the result. Then, optimize the query. Only after those fail should you consider sharding or read replicas.

I made this mistake myself when building a user analytics service. I assumed I needed ClickHouse for time-series data. I benchmarked it against PostgreSQL with a materialized view. PostgreSQL won: 12ms vs 85ms for the same query. I saved $1,200/month and reduced latency by 86%.

## Resources that helped

- [FastAPI docs](https://fastapi.tiangolo.com/) — Our migration started here. The asyncpg integration saved us 80 lines of code.
- [PostgreSQL 16: Better performance, less WAL](https://www.postgresql.org/docs/16/release-16.html) — The `REINDEX CONCURRENTLY` and `VACUUM` improvements cut our maintenance window from 30 minutes to 5.
- [Redis 7.2: Active Replication](https://redis.io/docs/management/replication/) — We used active replication to add a read replica without downtime.
- [Locust 2.20: Distributed load testing](https://locust.io/) — Our load test script ran on 5 EC2 instances and generated 5,000 RPS.
- [AWS Fargate pricing 2026](https://aws.amazon.com/fargate/pricing/) — We used the 2026 pricing calculator to estimate costs before migrating.
- [The Art of Readable Code](https://www.oreilly.com/library/view/the-art-of/9781449318482/) by Dustin Boswell — Helped us simplify our API endpoints and error messages.
- [Kubernetes Best Practices (2nd ed)](https://learning.oreilly.com/library/view/kubernetes-best-practices/9781492056478/) by Brendan Burns — We used it to tune our Fargate resource requests.

## Frequently Asked Questions

**How do I know if I’m over-engineering my system?**

Start by profiling your slowest endpoint. If the bottleneck is latency from a single database query, add an index or cache. Only move to distributed systems when you’ve exhausted local optimizations. If you’re adding Kafka before measuring query latency, you’re over-engineering. Most teams add distributed systems because they’re trendy, not because they measured a bottleneck.

**What’s the minimum viable architecture for a high-traffic API in 2026?**

A single FastAPI or Express.js app, PostgreSQL 16, and Redis 7.2. Deploy on AWS Fargate or Fly.io for simplicity. Use connection pooling (asyncpg for Python, pg-pool for Node) and a write-through cache. Add read replicas only when you hit 10,000 RPS. Add sharding only when you hit 50,000 RPS. Measure first — don’t assume.

**How do I convince my team to simplify?**

Show them the numbers. Track your current latency, error rate, and cost. Propose a 4-week experiment: build a simple version and compare metrics. Most teams will choose the simpler system when they see the data. If they don’t, ask: are we optimizing for our users or our egos?

**When should I use microservices?**

Only when you have strict isolation requirements (e.g., multi-tenant SaaS with data residency) or unbounded scale (e.g., 100,000 RPS with strict latency SLOs). Even then, start with a modular monolith and extract services when you hit concrete scaling walls. If your traffic is 5,000 RPS and your team is 5 engineers, you don’t need microservices.

## How to apply this today

Open your slowest API endpoint. Run `pg_stat_statements` on your database or profile your code with a tool like `py-spy`. If the bottleneck is a single query or function, fix that first. Only after you’ve optimized locally and measured the impact should you consider adding distributed systems. If your endpoint returns in under 200ms and your costs are under $500/month, stop optimizing. Ship the feature and move on.

Then, delete one abstraction. Remove a service, a queue, or a cache layer. Measure again. If nothing breaks, you’ve just simplified your system. Repeat until your architecture is boring — because boring systems are the ones that scale.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
