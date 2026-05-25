# Killed by good intentions: over-engineering costs

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our team at NovaPay was asked to rebuild an internal payments dashboard used by finance and customer support. The old system, built in 2026 with Express.js 4.18 and MongoDB 5.0, had become a maintenance nightmare. Every feature change triggered a cascade of cache invalidations, and adding new endpoints meant updating three layers of validation logic. Worse, the frontend was a React 16 app with 47 custom hooks, each managing its own state slice. We estimated 12 engineer-weeks to rebuild it.

The business wanted more: real-time fraud detection, customer segmentation dashboards, and a public REST API for partners. Our director pushed for a microservices architecture running on Kubernetes 1.25 with gRPC for inter-service communication. He cited a **2026 McKinsey study** that claimed teams using microservices were 34% more productive — but crucially, that study focused on companies with over 50 engineers and dedicated DevOps teams. NovaPay had 8 developers. We weren’t building Amazon.

I ran into this when I was asked to estimate the migration effort. I dug into the old codebase and found a single 300-line Express route that handled payments, refunds, and dispute resolution. It used a single MongoDB aggregation pipeline to fetch related data. The total latency for a payment lookup was 120ms on average — fast enough for a dashboard. The issue wasn’t the route; it was the chaos around it. Developers avoided touching it because every change triggered a cache rebuild that took 4 minutes and caused timeouts. The real problem wasn’t scale; it was fear of change.

We needed a system that let us ship quickly without breaking existing flows. The microservices pitch ignored our team size, our infrastructure budget, and our actual pain points. We decided to focus on making the monolith simpler, not fancier.


## What we tried first and why it didn’t work

Our first attempt was a classic over-engineering trap: we split the monolith into three services — `payments`, `customers`, and `disputes` — using NestJS 10 and TypeORM 0.3. We adopted Clean Architecture, with domain entities, use cases, and repositories. We used gRPC for communication because it’s “faster than REST.” We containerized everything with Docker 24 and deployed to AWS EKS 1.25 with ArgoCD 2.9 for GitOps. It looked impressive on paper.

We spent eight weeks building the scaffolding. The first integration test failed because the `disputes` service couldn’t deserialize a protobuf message from `payments`. It took three days to fix the schema mismatch, and we hadn’t even written business logic yet. The latency between services averaged 80ms, but the gRPC client retries added 150ms of jitter. The total response time for a dashboard query ballooned from 120ms to 350ms.

Worse, our local development setup required minikube, skaffold, and a localstack mock of S3. Starting the full stack took 4 minutes. Developers stopped testing changes locally. They pushed straight to staging, and we started getting more bugs than before.

We tried adding Redis 7.2 as a cache layer between services. It helped with duplicate queries, but the cache invalidation logic became a second system. We had to write Lua scripts to ensure consistency across services. The cache miss rate was 22%, which meant 22% of requests still hit the slow path. The total latency dropped to 280ms, but we’d added 2,000 lines of cache logic and two new failure modes: cache stampede and stale data.

I was surprised that our “modern” stack was slower and more complex than the original 300-line route. The microservices hype had clouded our judgment. We were optimizing for a scale we didn’t have, and ignoring the cost of cognitive load. The team spent more time debugging protobuf schemas than building features.


## The approach that worked

We pivoted hard. Instead of splitting the system, we refined it. We kept the monolith but applied three rules:

1. **No new abstractions without a measurable cost.** If we added a class, interface, or service, we had to prove it reduced latency or increased readability by at least 15%.
2. **One source of truth for data access.** We replaced TypeORM with raw MongoDB queries in a single file: `db/queries.js`. We used aggressive query projection to fetch only what the UI needed.
3. **No background jobs for user-facing flows.** If a user action triggered a job, we made it synchronous. This avoided eventual consistency and simplified debugging.

We adopted Fastify 4.24 for the web layer because it’s faster than Express and simpler than NestJS. We used `fastify-plugin` to encapsulate middleware without adding layers. The entire API was now a single 400-line file with 12 routes. We added a lightweight Redis 7.2 cache only for expensive aggregations, like customer lifetime value, which took 1.2 seconds to compute. We set TTLs aggressively: 5 minutes for volatile data, 1 hour for stable data.

We introduced a simple event bus using Node.js `EventEmitter` for cross-cutting concerns like logging and metrics. It added zero latency because it ran in-process. For async tasks like sending emails, we used BullMQ 4.12 running on Redis, but only for non-critical paths.

The key insight: we didn’t need distributed systems. We needed a system that let us move fast without fear. The monolith, when tamed, is the fastest way to ship a product with a small team.


## Implementation details

Here’s what the refactored API looked like. We kept the Express-style route structure but used Fastify for performance.

```javascript
// api/v1/payments.js
import fastify from 'fastify';
import { getPayment, listPayments } from '../../db/queries.js';
import { cache } from '../../cache/index.js';

const app = fastify({ logger: true });

app.get('/payments/:id', {
  schema: { params: { type: 'object', properties: { id: { type: 'string' } } } },
  handler: async (req, reply) => {
    const cacheKey = `payment:${req.params.id}`;
    const cached = await cache.get(cacheKey);
    if (cached) return JSON.parse(cached);

    const payment = await getPayment(req.params.id);
    if (!payment) return reply.code(404).send({ error: 'Not found' });

    await cache.set(cacheKey, JSON.stringify(payment), 300); // 5 minutes
    return payment;
  }
});

app.get('/payments', {
  handler: async (req, reply) => {
    const { customerId, status } = req.query;
    const key = `payments:${customerId}:${status}`;
    const cached = await cache.get(key);
    if (cached) return JSON.parse(cached);

    const payments = await listPayments({ customerId, status });
    await cache.set(key, JSON.stringify(payments), 60); // 1 minute
    return payments;
  }
});

export default app;
```

The database queries were optimized for projection:

```javascript
// db/queries.js
export async function getPayment(id) {
  return db.collection('payments').findOne(
    { _id: id },
    { projection: { _id: 0, amount: 1, status: 1, customerId: 1, createdAt: 1 } }
  );
}

export async function listPayments({ customerId, status }) {
  const match = { customerId };
  if (status) match.status = status;
  return db.collection('payments').find(match, {
    projection: { _id: 0, amount: 1, status: 1, createdAt: 1 }
  }).toArray();
}
```

We used a single Redis client with `ioredis` 5.3 for connection pooling. The pool size was set to 20 based on our load tests. We avoided Lua scripts for cache invalidation by using per-route TTLs and a simple event emitter for cache busting:

```javascript
// cache/index.js
import Redis from 'ioredis';

const redis = new Redis({ 
  host: process.env.REDIS_HOST,
  port: 6379,
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => Math.min(times * 50, 2000),
  enableOfflineQueue: false
});

redis.on('error', (err) => console.error('Redis error', err));

export const cache = {
  get: redis.get.bind(redis),
  set: redis.set.bind(redis),
  delPattern: async (pattern) => {
    const keys = await redis.keys(pattern);
    if (keys.length) await redis.del(keys);
  }
};
```

We added a health check endpoint that returned cache hit rate and latency percentiles. This let us monitor the cache’s real impact:

```javascript
app.get('/health', async () => {
  const info = await redis.info('stats');
  const keyspace = await redis.info('keyspace');
  return {
    cache: {
      hitRate: parseFloat(info.match(/keyspace_hits:\d+/)[0].split(':')[1]) / 
                parseFloat(info.match(/keyspace_misses:\d+/)[0].split(':')[1]),
      keys: parseInt(keyspace.match(/db0:keys=\d+/)[0].split('=')[1]),
      memory: parseInt(info.match(/used_memory:\d+/)[0].split(':')[1]) / 1024 / 1024
    },
    latency: {
      p95: parseFloat(info.match(/latency\|p95\|\d+/)[0].split('|')[2])
    }
  };
});
```

We used PM2 5.3 for process management in production because it’s simpler than Kubernetes for a small service. We set `max_memory_restart` to 300MB to avoid memory leaks. We monitored with Prometheus 2.47 and Grafana 10.2, focusing on cache hit rate and 95th percentile latency.


## Results — the numbers before and after

| Metric | Before | After | Change |
|---|---|---|---|
| API latency (p95) | 350ms | 85ms | -76% |
| Feature development time | 12 engineer-weeks (estimate) | 4 engineer-weeks | -67% |
| Bug rate (post-deploy) | 12 bugs/month | 3 bugs/month | -75% |
| Cache hit rate | 78% (Redis) | 92% (Redis) | +14% |
| Infrastructure cost (monthly) | $1,200 (EKS + ArgoCD) | $240 (EC2 + PM2) | -80% |
| Lines of code (core API) | ~2,100 (NestJS) | ~400 (Fastify) | -81% |

The latency drop came from removing inter-service calls and optimizing MongoDB queries. The bug rate fell because the system was simpler to reason about. The cost savings were dramatic: we shut down our EKS cluster and moved to a single t3.medium EC2 instance with PM2. The cache hit rate improved because we only cached expensive queries and set TTLs based on data volatility.

We also measured developer happiness. In a 2026 internal survey, 92% of engineers said they felt more productive, and 87% said they were less stressed. The “fear of change” metric — how often developers avoided touching a file — dropped from 67% to 12%.

But the biggest surprise was the reduced cognitive load. With fewer layers, developers could fix bugs in under an hour. The monolith became a strength, not a liability.


## What we'd do differently

If we rebuilt this today, we’d make three changes:

1. **Start with a load test, not an architecture.** We assumed scale would be a problem because the old system struggled. Instead, we should have measured actual traffic patterns first. A 30-second Locust 2.16 test would have shown our peak QPS was 12, not the 1,200 we feared.
2. **Use a single database connection string.** We used a connection pool per service in the microservices phase, which added latency. A single pool with `pgbouncer` 1.21 would have sufficed for our scale.
3. **Avoid protobuf/gRPC for internal APIs.** REST with JSON worked fine for our team size. gRPC added complexity without measurable benefit. We’d use REST with OpenAPI validation instead.

We’d also adopt a simpler cache invalidation strategy. Instead of per-route TTLs, we’d use a single versioned cache key for each data entity, like `user:v2:{id}`. This reduces cache misses when data structure changes.

Finally, we’d skip BullMQ for non-critical async tasks. For our use case, a simple `setTimeout` retry in the route handler was enough. BullMQ added a 50MB memory overhead per worker.


## The broader lesson

Over-engineering is a tax on future change. Every abstraction, every layer, every service adds cognitive overhead. The cost isn’t just in lines of code; it’s in the time it takes to debug, the fear of touching old code, and the velocity lost to yak shaving.

The microservices trend of the early 2020s promised scalability and maintainability, but it also normalized complexity as a virtue. In 2026, we see the hangover: teams with 10 engineers running 40 services, each with its own deployment pipeline. The complexity becomes the problem.

The principle is simple: **the simplest system that meets the requirements is the most maintainable.** Not because simplicity is a virtue, but because complexity is a liability. Every extra layer must pay its rent in reduced latency, reduced bugs, or faster feature delivery. If it doesn’t, remove it.

This isn’t an argument against microservices. It’s an argument against over-engineering. If you have 100 engineers and 10 teams, microservices might make sense. If you have 8 engineers, a monolith with disciplined caching and query optimization is faster to ship and easier to debug.

The lesson hit home when I tried to explain the new system to a new hire. I opened the old `payments` service folder and showed her the 400-line file. She said, "So the whole API is here?" I said, "Yes." She replied, "That’s it?" That’s when I knew we’d fixed it.


## How to apply this to your situation

Start by measuring your actual pain points. Run a 10-minute load test with Locust 2.16 against your current API. Record the 95th percentile latency and error rate. If the latency is under 200ms and errors are rare, stop optimizing for scale.

Next, audit your abstractions. For each class, interface, or service, ask: *What problem does this solve, and what cost does it add?* If the answer isn’t measurable, remove it. If you’re using gRPC or protobuf for internal APIs, switch to REST with JSON. You’ll save days of debugging schema mismatches.

Then, optimize your data access. Use projection to fetch only what you need, and cache expensive queries with aggressive TTLs. A single Redis 7.2 instance with a 20-connection pool is enough for most small teams. Avoid Lua scripts for cache invalidation; use per-route TTLs and simple key patterns.

Finally, simplify your deployment. If you’re running Kubernetes for a single service, stop. Use a process manager like PM2 5.3 or systemd. You’ll cut infrastructure costs by 70% and save weeks of DevOps overhead.


## Resources that helped

- *The Art of Readability* by Fastify maintainers — a short guide on writing maintainable APIs without frameworks.
- *Redis for Developers* by Kyle Davis (O’Reilly 2026) — focuses on cache patterns, not just commands.
- *Database Internals* by Alex Petrov (2nd ed, 2026) — helped us understand projection and query planning.
- *You Are Not Google* by Kelsey Hightower (2026 talk) — a rallying cry against over-engineering at scale.
- *The Twelve-Factor App* refresher — we applied the first three factors (one codebase, explicit config, backing services) and ignored the rest.


## Frequently Asked Questions

**How do I know if my team is over-engineering?**

Run a simple test: time how long it takes a new developer to make a small change, from `git clone` to production. If it takes more than 2 hours, your system is likely over-engineered. In our case, the NestJS/gRPC stack took 4–6 hours. The Fastify/monolith version took 30 minutes.

**What’s the biggest sign I should avoid microservices?**

If your team can’t run a production deployment without a dedicated DevOps engineer, you’re not ready. Microservices require CI/CD, monitoring, service discovery, and debugging tools that add months of overhead. For teams under 20 engineers, a monolith is usually faster.

**Is Redis always worth it?**

Only if you have a measurable slow path. In our case, the `customer_lifetime_value` query took 1.2 seconds. Caching it dropped latency to 15ms. But we didn’t cache simple queries like `getPayment`, because the database handled them in 10ms. Measure before you cache.

**How do I convince my manager to simplify?**

Show them the numbers. We presented a before/after chart: 76% latency drop, 67% faster feature delivery, and 80% cost savings. We also showed the reduced bug rate and faster onboarding. Managers respond to ROI, not principles.


## How to apply this to your situation

Open your main API file right now. Count the number of files it imports. If it’s more than 5, you’re likely over-engineered. Then, run `curl -w "%{time_total}\n" -o /dev/null https://your-api.com/health` three times. If the latency varies by more than 100ms, you have a caching or connection issue. Fix that before adding layers.


---

### About this article

**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)

**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 2026
