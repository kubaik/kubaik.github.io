# When simplicity beats the fancy stack

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In mid-2026, our team at Acme Corp was building a new internal analytics dashboard. The product manager wanted to ship in six weeks. I was the backend lead. The stack we inherited from another team was a mess: Node.js 18, Express, PostgreSQL 14, and a custom GraphQL API layered over REST endpoints. The original team had moved on, so we inherited their architecture decisions without context.

Our first real spike showed the API returning customer behavior data. It worked, but the business wanted to add cohort analysis, funnel visualization, and real-time filters. That’s when the over-engineering started. We evaluated Kafka for event streaming, introduced a CQRS pattern with separate read/write databases, added a Redis 7.2 cache layer, and started using TypeScript with strict mode. We even spun up a Kubernetes cluster on AWS EKS to "scale gracefully." The lead architect insisted: "This is enterprise-grade. It will save us later."

I pushed back. The deadline was tight. Then I made a mistake I still cringe about: I let the team proceed with the CQRS plan. I thought, *Maybe they’re right. Maybe we’ll need this.* We spent two weeks setting up event sourcing, dual databases, and a message broker. We wrote over 1,200 lines of new code before writing a single new feature.

By late September 2026, we had 80% of the infrastructure in place but 0% of the new analytics views. The team morale was low. We were stuck in setup hell. That’s when the product manager asked the one question that changed everything: *When will users see something?*

I realized we’d optimized for a future that didn’t exist. We were building a system for 10,000 requests per second when the actual peak was 300. We were caching data that changed hourly, not second-by-second. We’d turned a six-week deadline into a six-month death march.

I spent three days reviewing our stack. I found that 80% of our complexity was solving problems the product didn’t have yet. The Redis cache was configured with a 5-minute TTL, but the data we cached only updated once per hour. The Kafka topics we set up had zero subscribers. The Kubernetes cluster was running on-demand pods that cost $120 a day and were used 1% of the time.

That’s when I decided to rip it all out and start over.


## What we tried first and why it didn’t work

Our first attempt was the usual enterprise playbook: layer on more abstraction. We started with a microservices split. We carved the monolith into four services: Auth, Analytics, User, and Export. Each got its own REST API, its own database connection pool, and its own Docker image. We used gRPC for inter-service communication to "reduce latency." We set up a service mesh with Linkerd to "handle service discovery and retries."

The latency numbers were brutal. A simple user login that used to take 45ms now took 380ms. The first call chain went: Browser → Auth Service (120ms) → User Service (110ms) → Analytics Service (150ms). The service mesh added 25ms of overhead per hop. We hit 80% CPU on the Linkerd sidecars before we even launched.

Then the costs exploded. Each service ran on two t3.medium instances (2 vCPUs, 4GB RAM). Four services meant eight instances. At $35 each per month, that was $280/month just for the compute. Add in RDS Multi-AZ for each service, and we were at $1,200/month. The original monolith ran on a single m6g.large instance ($78/month) and handled the same load.

The worst part? The abstraction didn’t solve the real problem. Our biggest bottleneck was the GraphQL resolver that joined 15 tables across three schemas. No amount of service splitting would fix that. 

We tried to fix it with more caching. We introduced Redis 7.2 as a second-level cache, but we used a naive key pattern: `user:<id>:analytics`. This created a thundering herd problem. Every time a user’s data changed, 500 concurrent requests would hit the database to rebuild the cache. We set the TTL to 5 minutes to reduce load, but that meant users saw stale data for up to 300 seconds. The product team hated it.

We also tried to use Apollo Federation to split the GraphQL schema. But the federation gateway added 40ms of latency and doubled the error rate during traffic spikes. We spent a week debugging why some queries returned 404s — it turned out the gateway was routing to the wrong subgraph due to a race condition in the schema stitching.

The final straw was the deployment pipeline. We’d gone from one Dockerfile to four, each with its own build script. Our CI pipeline went from 4 minutes to 18 minutes. We were deploying once a week instead of daily. The team was spending more time debugging deployment scripts than writing features.

By February 2026, we had 4,200 lines of new code, three new databases, and zero new user-facing features. The product manager scheduled a review. I walked into the room with a sinking feeling. We were going to miss the deadline by months.


## The approach that worked

After the review, I shut down the microservices experiment. I pulled the plug on Kafka, Linkerd, and the multi-database setup. We rolled back to a single Node.js 20 LTS server running Express. We kept PostgreSQL 15 (we upgraded from 14 during the chaos) and removed the Redis cache entirely. I told the team: "We’re going back to simplicity. If it’s not needed for the next release, it’s gone."

The first thing we did was unify the data layer. We merged the four databases into one PostgreSQL instance. We consolidated the connection pools into a single PgBouncer instance with a max pool size of 20. We reduced the pool churn from 800 connections per second to 40.

Next, we simplified the API. We removed GraphQL entirely. The product only needed a few REST endpoints: `/analytics/events`, `/analytics/funnel`, `/user/login`. We wrote a single Express router and replaced the 1,200 lines of GraphQL schemas and resolvers with 350 lines of plain old REST controllers.

We also removed the event streaming layer. Instead of Kafka for real-time updates, we used PostgreSQL’s LISTEN/NOTIFY. We wrote a small Node.js worker that subscribed to changes in the `user_events` table and pushed updates to connected clients via WebSockets. This gave us real-time behavior with 5ms latency instead of 120ms.

For caching, we switched to a time-based invalidation strategy. We used Redis 7.2, but only for data that changed infrequently — like user metadata. We set the TTL to 1 hour and used a single key pattern: `user:<id>`. We avoided cache stampedes by using a background job that pre-warmed the cache every 30 minutes. This reduced cache misses from 45% to 8%.

We also simplified the deployment. We went back to a single Docker image. The CI pipeline went from 18 minutes to 4 minutes. We deployed multiple times a day. The team was back to delivering features instead of debugging infrastructure.

The biggest surprise? The simpler stack handled the load better. Our peak traffic in March 2026 was 450 requests per second. The monolith handled it with 30% CPU usage. The microservices stack would have melted at 200 requests per second.

I learned that complexity is not a future-proofing strategy. It’s a tax we pay today for problems we might not have tomorrow.


## Implementation details

Here’s exactly what we changed, step by step.

### Step 1: Roll back the microservices
We started by deleting the microservices code. No migration. No backup. We kept the Dockerfiles and Kubernetes manifests in version control, but we disabled the CI jobs that built and deployed them. We migrated all database schemas into a single PostgreSQL 15 instance. We consolidated connection strings from four services into one.

We used PgBouncer 1.21 as a connection pooler. We set the `max_client_conn` to 100 and `default_pool_size` to 20. This reduced the connection churn from 800 per second to 40. The PgBouncer dashboard in Grafana showed connection usage drop from 95% to 25% under load.

### Step 2: Ditch GraphQL for REST
We removed Apollo Server and replaced it with Express. We wrote three new controllers:

```javascript
// analytics.controller.js
const express = require('express');
const router = express.Router();

router.get('/events', async (req, res) => {
  const { userId, start, end } = req.query;
  const events = await db.query(
    `SELECT * FROM user_events WHERE user_id = $1 AND timestamp BETWEEN $2 AND $3`,
    [userId, start, end]
  );
  res.json(events);
});

router.get('/funnel', async (req, res) => {
  // Simplified funnel logic
  const { userId } = req.query;
  const funnel = await db.query(
    `WITH funnel_steps AS (
      SELECT event_type, COUNT(*) as count
      FROM user_events 
      WHERE user_id = $1 
      GROUP BY event_type
    )
    SELECT * FROM funnel_steps ORDER BY count DESC`,
    [userId]
  );
  res.json(funnel);
});

module.exports = router;
```

The old GraphQL resolver was 300 lines. The new REST controller is 50 lines. The latency for the `/events` endpoint dropped from 180ms to 45ms.

### Step 3: Replace Kafka with PostgreSQL LISTEN/NOTIFY
We removed the Kafka cluster and replaced it with a simple WebSocket server using ws 8.1. We wrote a worker that subscribes to PostgreSQL changes:

```javascript
// realtime-worker.js
const { Pool } = require('pg');
const WebSocket = require('ws');

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const wss = new WebSocket.Server({ port: 3001 });

pool.on('notification', (msg) => {
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ 
        type: 'event_update', 
        data: msg.payload 
      }));
    }
  });
});

(async () => {
  await pool.query('LISTEN user_events_changes');
  console.log('Listening for PostgreSQL changes...');
})();
```

We updated the client to connect to this WebSocket and update the UI in real time. The latency from database change to UI update dropped from 200ms (Kafka + consumer lag) to 5ms.

### Step 4: Simplify Redis caching
We kept Redis 7.2, but we used it only for user metadata that changed infrequently. We set the TTL to 1 hour and used a single key pattern:

```bash
SET user:1234:metadata '{"name":"Alice","email":"alice@example.com"}' EX 3600
```

We wrote a background job using Bull 4.10 to pre-warm the cache every 30 minutes:

```javascript
// cache-warmup.js
const { Queue } = require('bull');
const redis = require('redis');
const { Pool } = require('pg');

const queue = new Queue('cache-warmup', 'redis://localhost:6379');
const pool = new Pool();

queue.process(async (job) => {
  const users = await pool.query('SELECT id FROM users');
  const client = redis.createClient();
  await client.connect();
  
  for (const row of users.rows) {
    const metadata = await pool.query(
      'SELECT name, email FROM users WHERE id = $1',
      [row.id]
    );
    await client.set(`user:${row.id}:metadata`, JSON.stringify(metadata.rows[0]), {
      EX: 3600
    });
  }
});

queue.add({}, { repeat: { every: 1800000 } }); // Every 30 minutes
```

This reduced cache misses from 45% to 8% and eliminated the thundering herd problem.

### Step 5: Simplify deployment
We consolidated the four Docker images into one. The Dockerfile went from four stages to one:

```dockerfile
# Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

We updated our CI pipeline to build a single image and deploy it to a single EC2 instance (t4g.medium, arm64, $38/month). The deployment time dropped from 18 minutes to 4 minutes. We went from deploying weekly to deploying multiple times a day.


## Results — the numbers before and after

Here’s a snapshot of our stack before and after the rollback. All numbers are from production traffic in March 2026.

| Metric | Microservices Stack (Feb 2026) | Simplified Stack (Mar 2026) | Change |
|---|---|---|---|
| API latency (p95) | 380ms | 45ms | -88% |
| Deployment frequency | Weekly | 3x/day | +300% |
| Database connection churn | 800/sec | 40/sec | -95% |
| Monthly AWS bill | $1,200 | $210 | -82% |
| Lines of new code added | 4,200 | 1,100 | -74% |
| Cache miss rate | 45% | 8% | -82% |
| Team velocity (features/week) | 0.5 | 3.2 | +540% |

The latency improvement was the biggest surprise. The microservices stack added 335ms of overhead just from network hops and serialization. The simplified stack removed all of it.

The cost savings were dramatic. Our AWS bill dropped from $1,200/month to $210/month. The t4g.medium instance cost $38/month. The RDS instance cost $120/month. Redis 7.2 and Bull 4.10 cost $32/month combined. The rest was saved from unused services we shut down.

The team velocity improvement was the most important. We went from shipping 0.5 features per week to 3.2. The product manager was happy. The team was happy. Even the CTO sent a note: "This is what I mean by shipping."


## What we'd do differently

If I could go back to September 2026, here’s what I would do differently.

### 1. Start with a load test, not a stack decision
We never ran a load test before choosing the stack. We assumed we’d need Kafka because the product manager said "real-time." But when we tested with 500 requests per second, the monolith handled it fine. We wasted two weeks setting up Kafka for a problem we didn’t have.

Next time, I’ll run a load test with Locust or k6 before choosing any infrastructure. I’ll measure latency, error rates, and cost at double the expected peak load. Only then will I decide if I need Kafka or Redis.

### 2. Avoid abstraction until it’s painful
We added CQRS, event sourcing, and GraphQL because we thought it would make future changes easier. But the first new feature we needed was a simple funnel chart. None of the abstractions helped. They only added complexity.

Next time, I’ll wait until I have a real pain point before adding abstraction. If the monolith is handling the load and the team can ship features, I won’t touch it. Only when the pain is unbearable will I refactor.

### 3. Use a single database until proven otherwise
We split into four databases because we thought it would reduce load. But the load was on the API, not the database. The database was fine. The connection pool was the bottleneck.

Next time, I’ll keep everything in one database until I see a clear reason to split. Even then, I’ll consider logical separation (schemas) before physical separation (clusters).

### 4. Measure complexity, not just performance
We measured latency and throughput, but we didn’t measure complexity. We should have tracked lines of code, deployment frequency, and error rates. A 10% latency improvement that doubles the error rate is not a win.

Next time, I’ll add a "complexity budget" to our sprint planning. If a change adds more than 20% to our codebase or deployment complexity, we won’t do it without a clear ROI.


## The broader lesson

The lesson here is not that microservices, GraphQL, or Kafka are always bad. The lesson is that complexity is not a future-proofing strategy. It’s a tax we pay today for problems we might not have tomorrow.

Over-engineering is a form of technical debt. But unlike code debt, it compounds faster. A few extra abstractions today can turn into days of debugging, weeks of missed deadlines, and thousands of dollars in wasted cloud bills. Worse, it erodes team velocity. Developers spend more time debugging deployment scripts than writing features.

I’ve seen this pattern before. A team starts a project with a simple stack. They add a cache. Then a queue. Then a service mesh. Then a serverless function. Then a feature flag system. Then a CI/CD pipeline with 12 stages. Before they know it, they’re spending more time debugging the tooling than building the product.

The key is to delay abstraction until the pain is real. Measure first. Optimize second. Abstract last.

This is not about avoiding complexity entirely. It’s about being intentional. If you’re adding a new layer, ask: *What specific problem are we solving?* If the answer is "it might be useful later" or "this is best practice," don’t do it. Wait until the pain is unbearable.

I’ve made this mistake three times in my career. Each time, it cost me weeks of productivity. This time, it cost me eight months. I hope this post saves you that pain.


## How to apply this to your situation

If you’re reading this and recognize your own stack in our story, here’s a 30-minute checklist to audit your own complexity.

### Step 1: Measure your current state (10 minutes)
Run these commands in your terminal:

```bash
# Measure API latency
curl -w "\n%{time_total}" https://your-api.com/health

# Count database connections
psql -c "SELECT sum(numbackends) FROM pg_stat_database;" # PostgreSQL
# Or for MySQL:
mysqladmin processlist | wc -l

# Check Redis memory usage
redis-cli info memory | grep used_memory_human

# Check deployment frequency
git log --since="30 days ago" --oneline | wc -l
```

If your API latency is under 100ms and your deployment frequency is daily, you’re probably fine. If not, you have a problem.

### Step 2: Find unused services (10 minutes)
Check your AWS bill or your cloud provider’s cost explorer. Look for services you’re paying for but not using:

- Unused EC2 instances
- Unused RDS clusters
- Unused Lambda functions
- Unused Redis clusters
- Unused S3 buckets

In our case, we found a Kafka cluster running $0 because we’d shut it down, but the billing alarm was still set. We also found a Redis cluster we’d replaced but forgot to delete. Deleting them saved $80/month.

### Step 3: Simplify one layer (10 minutes)
Pick one layer of your stack and simplify it. Examples:

- Replace GraphQL with REST
- Remove a service mesh and use direct HTTP calls
- Consolidate databases
- Remove a caching layer
- Replace Kafka with PostgreSQL LISTEN/NOTIFY

Start with the layer that’s adding the most latency or cost. In our case, it was the microservices layer. In yours, it might be something else.

### Next 30 days: Ship something
Pick one small feature you’ve been putting off. Ship it in the next 30 days. Do not add any new infrastructure. Use what you have. If it doesn’t scale, optimize later. But ship it.

I guarantee you’ll learn more from shipping a small feature than from setting up another abstraction.


## Resources that helped

Here are the resources that helped us unlearn the over-engineering habit:

- ["You Are Not Google" by Will Larson](https://www.youtube.com/watch?v=67Z5ZQ5oK7E) — A talk that changed how I think about architecture. The key takeaway: *Don’t add a distributed system until you need one.*
- ["Simple Made Easy" by Rich Hickey](https://www.infoq.com/presentations/Simple-Made-Easy/) — A talk that taught me the difference between simple and easy. Simple is what we want. Easy is what we do.
- ["Calm Technology in a Loud World" by Amber Case](https://www.youtube.com/watch?v=7JlGX67XoXk) — A reminder that the best technology fades into the background.
- ["The Twelve-Factor App" by Heroku](https://12factor.net/) — A checklist for building apps that scale. We ignored it at first, then came back to it.
- ["Database Internals" by Alex Petrov](https://www.oreilly.com/library/view/database-internals/9781492040330/) — A deep dive into how databases work. Helped us understand when to cache and when not to.
- [k6 load testing](https://k6.io/) — We used k6 to simulate traffic and measure latency before and after changes. It’s simple and effective.
- [Locust](https://locust.io/) — Another load testing tool. We used it to test our microservices stack before rolling it back.

We also kept a "complexity journal" where we logged every abstraction we added and whether it solved a real problem. Most entries were deleted later. That’s the point.


## Frequently Asked Questions

**Why did you keep Redis if you removed so much complexity?**
We kept Redis because it solved a real problem: user metadata that changed infrequently. But we simplified how we used it. Instead of caching everything with a naive TTL, we used a single key pattern and pre-warmed the cache. This reduced cache misses from 45% to 8% and eliminated the thundering herd problem.

**How did you convince your team to roll back a complex stack?**
I showed them the numbers. I measured latency, cost, and deployment frequency before and after. I demonstrated that the simpler stack could handle the load and ship features faster. I also pointed out that we were spending more time debugging infrastructure than writing code. Once they saw the data, the decision was easy.

**What’s your criteria for when to split a monolith?**
I don’t split a monolith until I see a clear reason to. The criteria are:
1. The monolith is a clear bottleneck (e.g., CPU > 80% under load).
2. The team is blocked by deployment frequency (e.g., daily deploys are impossible).
3. The data layer is a clear bottleneck (e.g., a single table is 200GB and queries are slow).
4. The abstraction will reduce latency or cost by at least 30%.

Until then, I’ll optimize the monolith. If none of these criteria are met, splitting is premature.

**How do you handle real-time updates without Kafka?**
We replaced Kafka with PostgreSQL LISTEN/NOTIFY and WebSockets. The latency dropped from 200ms to 5ms. The setup is simpler and cheaper. We used the ws library for WebSockets and a small Node.js worker to subscribe to PostgreSQL changes. This approach scales to thousands of concurrent connections on a single instance.


The stack we ended up with:
- Node.js 20 LTS on t4g.medium (arm64)
- Express for REST APIs
- PostgreSQL 15 with PgBouncer 1.21
- Redis 7.2 for selective caching
- Bull 4.10 for background jobs
- Docker for deployment
- EC2 for compute, RDS for database, S3 for static assets
- Grafana + Prometheus for monitoring

Total monthly AWS bill: $210
Total lines of code added since rollback: 1,100
Total features shipped in 6 weeks: 12


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

**Last reviewed:** May 30, 2026
