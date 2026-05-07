# We wasted 6 months over-engineering

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

# We wasted 6 months over-engineering — the code still runs faster now

## The situation (what we were trying to solve)

In early 2023, our team was asked to deliver an internal analytics dashboard. The goal was simple: visualize sales data from 3 regional databases and calculate KPIs like monthly revenue, customer acquisition cost, and churn rate. We estimated the project would take 4–6 weeks. Six months later, we finally shipped something that worked.

We didn’t set out to over-engineer. Like many teams, we fell into the trap of "future-proofing" based on what we thought we’d need. We built an event-driven architecture with Kafka for data ingestion, a CQRS pattern with separate read/write models, a microservice for each KPI, and a GraphQL gateway to unify them all. We used Kubernetes to orchestrate everything, Terraform to provision cloud resources, and Prometheus/Grafana for observability. Our deployment pipeline had 8 stages with canary releases and feature flags.

Our initial benchmark was 500ms response time for dashboard queries. We measured 80ms on the first prototype — before any of the "scalable" layers were added. Yet we still built the whole stack. Why? Because we feared that if we didn’t plan for 10x traffic, we’d regret it later. We followed the same patterns we’d seen in tech blogs about Netflix and Airbnb, assuming that scale would come — and that simplicity was a risk.

I remember a late-night standup where we debated whether to use gRPC or REST for internal service communication. Someone argued: "What if we need real-time updates later? gRPC streams are the future." We chose gRPC. Cost of that decision? Zero users saw it. 

**Summary:** We started with a simple problem but built a distributed system for a problem that didn’t need one. We optimized for hypothetical scale instead of solving the immediate need.


## What we tried first and why it didn’t work

We began with a monolithic Django app that read directly from the production databases. It worked fine — 150ms average response time, 500 lines of code, one developer running it. Then, we added one feature: caching. We used Django’s built-in cache framework with Redis. Response time dropped to 40ms. We were hooked on the speed.

But then came the “requirement” from leadership: “We might have 10,000 concurrent users next quarter. We need this to scale.” So we started splitting things up. We extracted the KPI calculations into separate services, each with its own database. We added Kafka to stream changes from the regional databases into a central data warehouse. We built a GraphQL API gateway that joined data from these services. We even added a message queue for background processing.

We used Docker Compose for local development, then Kubernetes in staging. We spent two months writing Helm charts and tweaking resource limits. We configured autoscaling based on CPU and memory — even though our peak traffic was 300 users. Our codebase grew from 500 to 12,000 lines. Our deployment pipeline went from one Dockerfile to eight services, each with its own CI/CD workflow.

The result? Nothing broke — but nothing worked either. Dashboard queries now took 1.2 seconds on average. Why? Because each request triggered 5–7 internal API calls. Latency compounded. Our caching layer, once a single Redis instance, now required a Redis cluster with a 5-node Sentinel setup. We had to maintain connection pooling, retry logic, and circuit breakers. We spent more time debugging inter-service timeouts than adding features.

I remember trying to explain to a frustrated product manager why the simple dashboard now took 1.2 seconds. She said, "It used to load in half a second. Now it feels slower." We had traded simplicity for scalability — and the users didn’t notice the difference.

**Summary:** We replaced a working monolith with a distributed system that was slower, harder to debug, and cost more to run — all to prepare for scale we never reached.


## The approach that worked

After six months of struggle, we hit a breaking point. The dashboard was still broken, and the team was burned out. We decided to rip it all out and start over — but this time, with a single constraint: **whatever we build must be faster and cheaper to run than the original monolith.**

We stripped everything back to a single Python service using FastAPI. It read directly from the regional databases (we added read replicas to avoid slowing down production). We used a single Redis instance for caching, but we added a 5-minute TTL to prevent stale data. We built a simple cron job that pre-computed daily KPIs and stored them in a materialized view. The dashboard now read from this view — no joins, no real-time updates.

We removed Kafka, Kubernetes, Helm, Terraform, Prometheus, Grafana, and all the inter-service communication. We replaced them with a 200-line Bash script that deployed the service to a single EC2 instance using systemd. Our CI/CD pipeline went from 8 stages to 2: lint and deploy.

We measured everything. We ran a load test with Locust simulating 10,000 users — 3x our peak traffic. Response time stayed under 150ms. Database CPU usage was 12%. Our cloud bill dropped from $1,800/month to $350/month.

The real surprise? **Our users didn’t notice the architecture change — they noticed the speed.** Average session time increased by 30%. Support tickets about slow dashboards dropped to zero.

**Summary:** We discovered that simplicity isn’t the enemy of scalability — it’s often the enabler. By removing layers, we made the system faster, cheaper, and easier to maintain.


## Implementation details

Here’s exactly what we did — no abstractions, just code and configs.

### The new stack: one service, one cache, one database

We used FastAPI (Python 3.11) because it’s lightweight and async-ready. We chose Uvicorn with Gunicorn workers for production. No Kubernetes, no Docker Swarm — just a single EC2 instance (t3.medium, $34/month).

```python
# main.py
from fastapi import FastAPI
import redis
import psycopg
from psycopg_pool import ConnectionPool
from datetime import datetime, timedelta
import os

app = FastAPI()

# Single Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Connection pool for PostgreSQL read replicas
pool = ConnectionPool(
    conninfo=os.getenv('DATABASE_URL'),
    min_size=2,
    max_size=5,
    max_waiting=10,
    timeout=30
)

@app.get("/revenue")
async def get_revenue(month: str = None):
    cache_key = f"revenue:{month or 'latest'}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"data": float(cached)}
    
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if month:
                cur.execute("SELECT SUM(amount) FROM sales WHERE DATE_TRUNC('month', sale_date) = %s", (month,))
            else:
                cur.execute("SELECT SUM(amount) FROM sales WHERE sale_date >= %s", (datetime.now() - timedelta(days=30),))
            result = cur.fetchone()[0] or 0.0
    
    redis_client.setex(cache_key, 300, str(result))  # 5-minute TTL
    return {"data": result}
```

### Caching strategy: time-based, not event-based

We used Redis with a fixed TTL (300 seconds) instead of trying to invalidate on every write. This meant our dashboard could serve stale data for up to 5 minutes — but in practice, KPIs like monthly revenue don’t change that often. 

```bash
# Crontab entry to refresh daily aggregates
0 2 * * * /app/refresh_daily_views.sh >> /var/log/refresh.log 2>&1
```

The `refresh_daily_views.sh` script runs a SQL query to update a materialized view:

```sql
-- refresh_daily_views.sql
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_kpis;
```

### Deployment: no Kubernetes, just systemd

We used a simple systemd service to run the FastAPI app. No Docker, no containers. Just Python and systemd.

```ini
# /etc/systemd/system/dashboard.service
[Unit]
Description=Analytics Dashboard
After=network.target

[Service]
User=appuser
WorkingDirectory=/app
ExecStart=/usr/bin/gunicorn -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000 main:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then we wrote a minimal deployment script:

```bash
#!/bin/bash
# deploy.sh
git pull origin main
docker build -t dashboard:latest .  # Wait — we said no Docker? Oops. We actually used Docker for a week before realizing it was overkill.
# Corrected version:
pip install --upgrade pip
pip install -r requirements.txt
systemctl restart dashboard
```

Yes, we initially tried Docker. We spent a week tweaking the Dockerfile, optimizing layers, and debugging permission issues. Then we realized: **we don’t need Docker to run a Python app.** We removed it. The systemd file above is all we needed.

### Monitoring: what actually matters

We didn’t need Prometheus or Grafana. We used a simple health check endpoint and CloudWatch alarms.

```python
@app.get("/health")
async def health():
    try:
        pool.check()
        redis_client.ping()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
```

We set up a CloudWatch alarm on the `dashboard_health_check_status` metric. If it returns anything other than 200 for 5 minutes, we get an alert. That’s it.

**Summary:** We built a system that did exactly what we needed — nothing more, nothing less. We removed everything that didn’t add immediate value.


## Results — the numbers before and after

| Metric | Before (over-engineered) | After (simple) |
|--------|--------------------------|----------------|
| Avg response time | 1,200ms | 85ms |
| P99 latency | 3,200ms | 250ms |
| Cloud cost (monthly) | $1,800 | $350 |
| Lines of code | 12,000 | 1,200 |
| Deployment time | 45 minutes | 2 minutes |
| On-call incidents (monthly) | 8 | 0 |
| Feature development time | 3 weeks per KPI | 2 days per KPI |

We also ran a user survey after launch. 87% of users said the dashboard felt "faster" or "much faster" than before. 0% noticed the architecture change.

The biggest surprise? **Our database costs actually increased slightly** — from $200/month to $250/month — because we added read replicas. But our overall cloud bill dropped by 80% because we eliminated all the orchestration layers.

We measured CPU and memory usage under load. The simple service used 40% less CPU than the over-engineered one. Why? Because we removed the overhead of serialization, network hops, and inter-service retries.

**Summary:** Simplicity paid off in speed, cost, and reliability. The numbers don’t lie — we built a system that worked better by doing less.


## What we'd do differently

If we could go back, here’s what we’d skip:

1. **Kafka and event-driven architecture**: We added Kafka to stream data from regional databases. It took 6 weeks to configure and tune. The only consumer was our own service — and it only needed daily snapshots. We replaced it with cron-triggered SQL scripts.

2. **Microservices for KPIs**: We split revenue, acquisition cost, and churn into separate services. Each had its own database. The GraphQL gateway joined them. This added 200ms of latency per request. We consolidated everything into one service and pre-computed daily aggregates.

3. **gRPC for service communication**: We chose gRPC over REST because of “future real-time needs.” We never used streams. We replaced it with simple REST endpoints in the same service.

4. **Kubernetes and Helm**: We spent two months writing Helm charts. Our cluster had 12 pods, 3 of which were Redis, 2 were PostgreSQL, and the rest were sidecars. We replaced it with a single EC2 instance and systemd.

5. **Prometheus and Grafana**: We set up Prometheus to scrape 20 endpoints every 15 seconds. We created 15 dashboards. We never looked at them. We keep one CloudWatch dashboard now — it has 3 graphs: CPU, memory, and health check status.

6. **Feature flags and canary releases**: We added LaunchDarkly to manage feature flags across 8 services. We spent more time debugging flag inconsistencies than shipping features. We removed all flags. We deploy the entire service at once.

7. **Connection pooling everywhere**: We added PgBouncer, Redis connection pools, and HTTP connection pools. We spent weeks tuning pool sizes. In the end, our simple service needed one connection pool — and only because we used async SQL drivers.

**The biggest mistake?** We optimized for scale we didn’t have and complexity we didn’t need. We followed the “best practices” we’d read in blogs and conference talks — but we didn’t measure whether they were necessary. 

**Summary:** We learned to question every dependency. If a tool or pattern doesn’t solve a current problem, don’t use it. Assume simplicity until proven otherwise.


## The broader lesson

Over-engineering isn’t about using the wrong tools — it’s about using the right tools for the wrong problem. It’s the habit of solving tomorrow’s problems today.

The principle we should have followed is this: **Build for today’s constraints, not tomorrow’s fears.**

Here’s what that means in practice:

- **Measure first, optimize later.** We assumed we’d have 10,000 users. We never measured. In reality, our peak was 300. We wasted $1,450/month on unused capacity.
- **Complexity compounds.** Every abstraction adds latency, debugging time, and cognitive load. A system with 12,000 lines of code is harder to change than one with 1,200 — even if the smaller one does the same thing.
- **Simplicity scales better than architecture.** A well-tuned monolith can outperform a distributed system if the distributed system is poorly tuned. We saw this in our load tests: the simple service handled 10,000 users with 150ms response time. The over-engineered one couldn’t beat 1.2 seconds even under 500 users.
- **Observability is not dashboards.** We thought we needed Prometheus and Grafana to “see” what was happening. But the only thing that ever broke was our health check. A single endpoint told us everything we needed.

This isn’t an anti-patterns rant. It’s a plea for pragmatism. Every “best practice” has a cost — and that cost should be justified by a real problem, not a hypothetical one.

**Summary:** The best architecture is the one you don’t have to explain to your teammates — because it’s just a few lines of code and a clear purpose.


## How to apply this to your situation

Not every project needs microservices. Not every API needs GraphQL. Not every cache needs Redis Cluster. Here’s how to decide what’s worth building:

1. **Start with a working prototype in one file.** Can you solve the problem in 100 lines of code? Use FastAPI, Flask, or even plain Node.js. Deploy it to a $5/month VM. If it works, ship it.

2. **Measure before you optimize.** Run a load test. Simulate real traffic. Measure latency, memory, CPU, and cost. If your prototype handles 2x your peak traffic with 50% headroom, you’re done. Stop there.

3. **Ask: what breaks first?** For us, it was the database. We added read replicas. For others, it might be the cache. Measure it. Don’t assume.

4. **Use time-based caching, not event-based.** If your data changes hourly, cache for 5 minutes. If it changes daily, cache for 24 hours. Most dashboards don’t need real-time updates.

5. **Avoid abstractions until you need them.** Don’t add a message queue until you have 1,000 messages/second. Don’t use Kubernetes until you have 50 services. Don’t use GraphQL until you have 10 endpoints.

6. **Ship, then measure, then improve.** Not the other way around. We spent six months building a “scalable” system before we knew if anyone would use it. **Ship the simple version first. Improve only when you have data.**

Here’s a quick checklist for your next project:

| Check | Yes | No |
|-------|-----|----|
| Can I build this in one file? |  |  |
| Does it handle 2x peak traffic on a $5 VM? |  |  |
| Am I caching aggressively? |  |  |
| Do I have a single health check endpoint? |  |  |
| Can I deploy it with a 3-line shell script? |  |  |

If any answer is “No,” stop and simplify.

**Next step:** Pick one project this week. Remove one layer of abstraction — Docker, Kubernetes, Kafka, GraphQL, microservices, or feature flags. Replace it with a simpler alternative. Measure the result. If it’s faster and cheaper, keep it. If not, revert. You’ll learn more in one day than in six months of over-engineering.


## Resources that helped

- [FastAPI docs](https://fastapi.tiangolo.com/) — We switched from Django REST to FastAPI and cut our response time in half.
- [The Twelve-Factor App](https://12factor.net/) — We violated 8 of the 12 factors and paid the price. Re-reading it helped us simplify.
- [Calvin French-Owen’s “Scale Fast, Simplify Later” talk](https://www.youtube.com/watch?v=F9yG2z8bXQ4) — A brutal takedown of over-engineering.
- [Redis for caching: when to use TTL vs event-based invalidation](https://redis.io/docs/manual/persistence/) — We learned that time-based TTL is often enough.
- [Load testing with Locust](https://locust.io/) — We used it to prove our simple service could handle 10,000 users.
- [PostgreSQL materialized views](https://www.postgresql.org/docs/current/rules-materializedviews.html) — Pre-computing aggregates saved us from real-time joins.
- [systemd for simple deployments](https://www.freedesktop.org/software/systemd/man/systemd.service.html) — We replaced Docker Swarm with 15 lines of config.


## Frequently Asked Questions

### Why didn’t you use serverless to cut costs further?

Serverless (Lambda, Fargate) adds latency and cold starts. Our dashboard needs sub-200ms response time. With Lambda, we saw 80–150ms cold starts and 40–80ms warm starts — acceptable, but not better than our EC2 instance. Also, Lambda costs scale with requests. At 10,000 requests/day, our $350/month EC2 instance is cheaper than Lambda would be. We measured it: Lambda would cost $420/month for the same traffic.


### How do you handle high availability if you’re running on one EC2 instance?

We don’t. Our uptime requirement is 99% — and our EC2 instance has 99.95% uptime in us-east-1. If it fails, we restore from a snapshot in under 5 minutes. We’ve had one outage in 12 months — a 10-minute DNS misconfiguration. For our use case, a single instance is fine. If we ever need 99.9%, we’ll add a second instance in another AZ — but that’s still simpler than Kubernetes HA.


### What if you need to add a new feature? Won’t the monolith become messy?

We’ve added 8 new features in the last 8 months. Each took 2–3 days. The codebase is 1,200 lines — easy to understand. We follow a simple rule: if a feature requires more than 50 lines of code, it goes in its own file. We’ve never hit the point where the monolith became unmaintainable. That point is usually much further away than teams assume.


### How did you convince leadership to accept a simpler solution?

We didn’t argue about architecture — we argued about outcomes. We showed them the numbers: 80% cost savings, 14x faster response time, 0 on-call incidents. We also ran a live demo: we built a new KPI dashboard in 2 hours using the simple stack. Leadership cares about speed, cost, and reliability — not architecture diagrams. We led with the results.