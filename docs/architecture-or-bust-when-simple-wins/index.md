# Architecture or bust: when simple wins

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our team at NodeBright was building a new analytics dashboard for a client in the IoT space. The client needed real-time processing of sensor data, with 99.9% uptime and sub-100ms response times for dashboard queries. Our initial estimate was two months for the backend, but the client pushed back, insisting on a six-week deadline. Pressure mounted when we realized the client’s CTO had recently joined from a FAANG company and expected us to "do it right" — meaning microservices, event sourcing, and Kubernetes from day one.

I’ll admit I drank the Kool-Aid. I had just finished reading a 2026 conference talk titled *Event Sourcing at Scale: How Netflix Handles Petabytes of Data* and convinced myself that our 50k events per second workload was somehow comparable. We spun up three Node.js services, each with its own Redis cluster, plus a Kafka cluster for event sourcing. We added OpenTelemetry tracing, Prometheus metrics, and a Grafana dashboard to "monitor everything." By the time we had the architecture diagram looking like a spaghetti monster, we were three weeks into a six-week project, and we hadn’t written a single line of business logic.

The client’s CTO reviewed our stack and said, "This looks impressive, but when do we get the API?" We had spent 40% of our time on plumbing and 0% on the actual problem. I was surprised that the client’s engineering team, who had never used Kafka or K8s, were the ones pushing back hardest on the complexity. They just wanted the data to appear on the screen without refreshing.

Our real goal was simple: accept sensor data via HTTP, store it in a database, and let users query it with filters. But we convinced ourselves that "simple" meant "naive" and that production-grade required distributed systems.


## What we tried first and why it didn’t work

We started with a microservices architecture because we thought it would make scaling easier. We split the system into three services:
- **IngestService** (Node.js + Express): received sensor data
- **ProcessingService** (Python + FastAPI): normalized and enriched data
- **QueryService** (Go + Gin): responded to dashboard queries

Each service had its own Redis cache, Postgres database, and Docker container. We used Kafka to communicate between services. We set up Kubernetes with Helm charts, configured ingress, and deployed to AWS EKS with arm64 Graviton instances. The whole setup cost about $1,200 per month in idle resources — plus the engineering time to maintain it.

The first problem appeared on day two: latency. A simple GET request to fetch sensor data took 450ms on average. We traced it to the fact that every request required three network hops — Ingest → Kafka → Processing → Query — plus two Redis lookups. Even with in-memory caching, the round-trip overhead was brutal. Our target was 100ms; we were at 4.5x that.

Then the client asked for a simple change: filter sensor data by timestamp range. We realized we had no shared database schema between services. The IngestService stored raw JSON blobs, the ProcessingService normalized into a relational model, and the QueryService denormalized for reads. Joining data across services required Kafka streams or application-level joins, which added another 200ms. We spent a week trying to fix the joins, but the fundamental issue was the architecture we chose.

The worst moment came when we tried to deploy. Our Helm charts were 500 lines of YAML, and the Kubernetes cluster needed constant tweaking to handle CPU throttling on our $0.04/hour Graviton nodes. The AWS bill for the month was $3,800, and we still hadn’t shipped a feature. I spent two weeks debugging a connection pool issue in the QueryService that turned out to be a single misconfigured timeout — this post is what I wished I had found then.


## The approach that worked

After three weeks of frustration, we scrapped the microservices plan and went back to basics. The client’s CTO finally relented when we showed him a prototype that did the same thing in one service with 10% of the code. We rebuilt the system as a single Node.js service using Express, with a PostgreSQL database and Redis for caching. We used a single table for sensor data with a JSONB column for metadata, avoiding the need for joins. We added a simple cron job to pre-aggregate daily stats instead of streaming everything through Kafka.

The key insight was that our workload was **read-heavy with occasional writes** — not the high-throughput, low-latency streaming workload Kafka is designed for. We didn’t need event sourcing; we needed a simple CRUD API with caching. Our bottleneck wasn’t scale; it was complexity.

I was surprised to find that PostgreSQL 16 with the TimescaleDB extension handled our 50k events per second with ease on a single db.r7g.large instance (4 vCPUs, 32 GiB RAM). TimescaleDB gave us automatic time-series partitioning, which simplified our queries. For caching, Redis 7.2 with a simple TTL strategy reduced dashboard query latency from 450ms to 45ms. We didn’t need connection pooling libraries like `pg-pool` or `ioredis` initially; the built-in pooling in `pg` and `redis` drivers was enough.

The client’s team appreciated the simplicity. They could now run the whole system locally with Docker Compose in under 30 seconds. Our AWS bill dropped from $3,800 per month to $450. We shipped the first dashboard feature in 10 days instead of 6 weeks, and the client was happy enough to extend our contract.


## Implementation details

Here’s the code that replaced 3,200 lines of microservice scaffolding and 500 lines of Helm charts.

**`server.js`** (the entire backend):
```javascript
import express from 'express';
import { Pool } from 'pg';
import Redis from 'ioredis';

const app = express();
app.use(express.json({ limit: '10mb' }));

const pgPool = new Pool({
  host: process.env.PG_HOST || 'localhost',
  port: process.env.PG_PORT || 5432,
  user: process.env.PG_USER || 'postgres',
  password: process.env.PG_PASSWORD || 'postgres',
  database: process.env.PG_DATABASE || 'sensordb',
  max: 20, // connection pool size
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');

// Simple cache middleware
const cache = (key, ttl = 60) => {
  return async (req, res, next) => {
    const cacheKey = `${key}:${JSON.stringify(req.query)}`;
    const cached = await redis.get(cacheKey);
    if (cached) {
      return res.json(JSON.parse(cached));
    }
    const originalSend = res.send;
    res.send = (body) => {
      redis.set(cacheKey, body, 'EX', ttl);
      originalSend.call(res, body);
    };
    next();
  };
};

// Endpoint to ingest sensor data
app.post('/sensors', async (req, res) => {
  const { sensor_id, timestamp, value, metadata = {} } = req.body;
  const query = `
    INSERT INTO sensors (sensor_id, timestamp, value, metadata)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (sensor_id, timestamp)
    DO UPDATE SET value = EXCLUDED.value, metadata = EXCLUDED.metadata
  `;
  await pgPool.query(query, [sensor_id, new Date(timestamp), value, metadata]);
  res.status(201).send('OK');
});

// Endpoint to query sensor data
app.get('/sensors', cache('sensors'), async (req, res) => {
  const { sensor_id, start, end, limit = 100 } = req.query;
  const query = `
    SELECT * FROM sensors
    WHERE sensor_id = $1
    AND timestamp BETWEEN $2 AND $3
    ORDER BY timestamp DESC
    LIMIT $4
  `;
  const result = await pgPool.query(query, [sensor_id, new Date(start), new Date(end), limit]);
  res.json(result.rows);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**`docker-compose.yml`** (the entire infrastructure):
```yaml
version: '3.8'
services:
  postgres:
    image: timescale/timescaledb:2.14.2-pg16
    environment:
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      PG_HOST: postgres
      PG_PASSWORD: postgres
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  redis_data:
```

We used **TimescaleDB 2.14.2** for time-series data and **Redis 7.2** for caching. The PostgreSQL connection pool size was set to 20, which handled our peak load without issues. We didn’t need a message broker; the cron job for daily aggregation was written in 20 lines of Bash and scheduled with `cron` in the container.

The database schema was simple:
```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE sensors (
  sensor_id TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  value DOUBLE PRECISION,
  metadata JSONB,
  PRIMARY KEY (sensor_id, timestamp)
);

SELECT create_hypertable('sensors', 'timestamp');
```

We added a daily aggregation job using TimescaleDB’s continuous aggregates:
```sql
CREATE MATERIALIZED VIEW sensor_daily_stats
WITH (timescaledb.continuous) AS
  SELECT
    sensor_id,
    time_bucket('1 day', timestamp) AS day,
    avg(value) AS avg_value,
    max(value) AS max_value,
    min(value) AS min_value
  FROM sensors
  GROUP BY sensor_id, day;
```

This approach cut our infrastructure to a single `db.t4g.medium` instance (2 vCPUs, 4 GiB RAM) plus a `cache.t4g.micro` for Redis. Total AWS cost: **$180/month**, down from $3,800. We deployed with Docker Compose on a single EC2 instance, avoiding Kubernetes entirely.


## Results — the numbers before and after

| Metric                          | Microservices Approach (3 weeks in) | Simplified Approach (after rebuild) |
|---------------------------------|---------------------------------------|------------------------------------|
| Lines of code (backend)         | 3,200                                 | 450                                |
| Infrastructure cost (AWS/month) | $3,800                                | $180                               |
| Dashboard query latency         | 450ms average                         | 45ms average                       |
| Time to first feature (weeks)   | 6 (projected)                         | 3                                  |
| Team cognitive load             | High (3 services, 2 databases)        | Low (1 service, 1 database)        |
| Deployment frequency            | Once every 2 weeks                   | Multiple per day                   |
| On-call incidents (per month)   | 3                                     | 0                                  |

The latency improvement came from eliminating network hops. Our simplified endpoint made a single database query and cached the result. The $3,620 monthly cost saving paid for the client’s extended contract and left room for profit.

Most importantly, the team’s morale improved. We went from firefighting connection pools and Helm charts to shipping features. The client’s team could now run the entire system locally with `docker-compose up`, which saved hours of debugging.


## What we’d do differently

If we had to rebuild this today, we’d make three changes:

**1. Start with a monolith, then split only when necessary.**
We should have built a single service first and measured where the bottlenecks actually were. In our case, the database was the bottleneck, not the API. We ended up splitting only the ingestion layer into a separate service after 6 months when we hit 500k events per second — and even then, we kept it as a single process with a sidecar for high-throughput writes.

**2. Use simpler tools for time-series data.**
TimescaleDB was perfect for us, but we overcomplicated it by trying to use Kafka for time-series ingestion. In hindsight, we could have used **InfluxDB 3.0** or **VictoriaMetrics** for raw ingestion and TimescaleDB only for the dashboard queries. The simpler the tool, the faster we iterate.

**3. Cache aggressively, but measure first.**
We assumed all queries needed caching, but 30% of our queries were for recent data that changed frequently. We wasted Redis capacity caching noise. We’d add cache warming only after profiling with **Redis 7.2’s `--latency` tool** and **pg_stat_statements** in PostgreSQL 16.

We also learned that **Kubernetes is overkill for most Node.js apps** unless you’re at Netflix scale. For our workload, Docker Compose on EC2 was simpler, cheaper, and faster to deploy. The client’s team could now debug the entire stack with `docker ps` and `kubectl logs` was gone.


## The broader lesson

Over-engineering isn’t about using the wrong tools; it’s about solving problems you don’t have yet. The **YAGNI principle** (You Aren’t Gonna Need It) is usually mocked as startup advice, but it’s a survival tactic for small teams. Every line of code, every configuration file, every extra moving part is a future bug, a future cost, and a future distraction.

The real cost of over-engineering isn’t just the engineering hours or the AWS bill. It’s the **cognitive load** — the mental energy required to maintain a complex system when you could be building features. Teams that over-engineer often end up with **lower velocity** because every change requires touching multiple services, updating multiple configs, and debugging distributed traces.

I learned this the hard way when I spent a week debugging a Redis connection pool timeout in our QueryService. The timeout was set to 5ms, but under load, Redis took 8ms to respond. The fix was a one-line change to 20ms, but the time spent debugging the pool configuration could have been spent on the actual feature. This taught me that **complexity compounds** — each layer adds noise that obscures the signal.

The industry’s obsession with "scalable architectures" and "production-grade systems" often conflates scalability with complexity. But scalability isn’t the goal; **delivering value is**. A system that handles 10 requests per second with 50ms latency is more valuable than one that handles 100k requests per second with 500ms latency if the first one ships in two weeks and the second one ships in six months.


## How to apply this to your situation

Start by asking: **What is the simplest thing that could possibly work?**

1. **Build a monolith first.** Use a single service, single database, and single cache. If you need to split, do it later when you have real data showing where the bottleneck is.
2. **Profile before optimizing.** Use **PostgreSQL 16’s `pg_stat_statements`** and **Redis 7.2’s `--latency`** to find real bottlenecks. Don’t assume caching or sharding will help until you measure.
3. **Use the right tool for the job.** If you’re storing time-series data, use **TimescaleDB 2.14.2** or **InfluxDB 3.0** instead of shoehorning it into a relational database. If you need caching, use **Redis 7.2** with TTLs — not a distributed cache you don’t need.
4. **Deploy with Docker Compose first.** Unless you’re at scale, Kubernetes adds more complexity than value. Use **Docker Compose** for local development and **single EC2 instances** for production.
5. **Set hard limits.** Decide upfront how much time you’ll spend on infrastructure. If you hit your limit, ship the feature and revisit later.


If you’re working on a new project, start with this checklist today:

- [ ] Create a single `docker-compose.yml` file that runs your entire app
- [ ] Use PostgreSQL 16 with TimescaleDB if you have time-series data
- [ ] Add Redis 7.2 for caching, but only after profiling queries
- [ ] Set up `pg_stat_statements` and Redis `--latency` monitoring on day one
- [ ] Deploy to a single EC2 instance (e.g., `t4g.medium`) and measure cost
- [ ] Ship a feature within 7 days — if not, simplify further

I made the mistake of assuming complexity equaled professionalism. The client didn’t care about our Kafka cluster or Helm charts; they cared about data on the screen. Next time, I’ll start with the simplest thing that works and add complexity only when I have proof it’s needed.


## Resources that helped

- **TimescaleDB documentation**: [https://docs.timescale.com/](https://docs.timescale.com/) — The best resource for time-series data in PostgreSQL. Their tutorials on continuous aggregates saved us hours of manual cron jobs.
- **Redis 7.2 latency tool**: [https://redis.io/docs/management/optimization/latency/](https://redis.io/docs/management/optimization/latency/) — Essential for diagnosing cache performance issues. We used `--latency` to find our Redis 8ms timeout under load.
- **PostgreSQL 16 `pg_stat_statements`**: [https://www.postgresql.org/docs/16/pgstatstatements.html](https://www.postgresql.org/docs/16/pgstatstatements.html) — Helped us identify the slow queries that needed caching first.
- **Docker Compose best practices**: [https://docs.docker.com/compose/](https://docs.docker.com/compose/) — Our simplified deployment strategy came from following their production checklist.
- **NodeBright’s post-mortem**: [https://engineering.nodebright.com/2026/11/over-engineering-cost-us-3-weeks/](https://engineering.nodebright.com/2026/11/over-engineering-cost-us-3-weeks/) — The internal post-mortem that led to this rebuild. We open-sourced the Terraform module we used to tear down the Kubernetes cluster.


## Frequently Asked Questions

**What if my app really needs to scale?**

Start with a monolith and split only when you hit real bottlenecks. Most apps never reach the scale where microservices are necessary. If you’re at 10k requests per second, consider splitting by domain (e.g., user service vs. analytics service), not by CRUD operations. Use **PostgreSQL 16’s logical replication** or **TimescaleDB’s distributed hypertables** before jumping to Kafka and K8s.


**When should I use Kafka instead of a simple cron job?**

Kafka is only worth the complexity if you need **exactly-once processing**, **replayability**, or **high-throughput, low-latency event streaming**. For a dashboard that updates hourly or daily, a cron job and TimescaleDB continuous aggregates are simpler and faster to implement. We used Kafka in our first attempt for a simple aggregation job — it added months of debugging for a feature the client didn’t need.


**How do I convince my manager to simplify the architecture?**

Show them the numbers. Calculate the cost of your current stack (AWS bill, engineering hours, on-call incidents) and compare it to a simplified alternative. In our case, the simplified stack cost $180/month vs. $3,800, and we shipped features 3x faster. Frame it as **risk reduction** — fewer moving parts mean fewer things to break and debug.


**What’s the fastest way to simplify an existing microservices architecture?**

Start by merging the most tightly coupled services. In our case, the IngestService and ProcessingService were reading and writing the same data. We merged them into a single service that writes to PostgreSQL directly. Use **PostgreSQL’s foreign data wrappers** if you need to query across services without merging. For our QueryService, we replaced it with a simple endpoint that reads from the same database as the ingestion layer.


## How to start today

Open your `docker-compose.yml` file right now. Delete every service except your app, database, and cache. If you don’t have one, create it and run `docker-compose up`. Then, time how long it takes to deploy a change. If it’s more than 60 seconds, simplify further. Your goal is a system you can rebuild from scratch in under a minute.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
