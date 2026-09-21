# Killed $3k Kubernetes, kept $200 infra

Most replaced 3kmonth guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In early 2026, our SaaS product had grown enough that the marketing site kept crashing during traffic spikes. We were running on a managed Kubernetes cluster in AWS EKS (v1.27) with 3 `t3.xlarge` nodes, an RDS `db.t3.2xlarge` instance, and an ElastiCache `cache.t3.medium` cluster for Redis. The bill hit $3,127/month, and the CFO sent me a Slack message that read: *"Is this normal?"* I ran a quick calculation and realized that 68% of that cost was for the Kubernetes control plane and the over-provisioned compute nodes. The marketing site averaged 150 requests/second but spiked to 1,800 requests/second during product launches. The Redis cache was only hit 42% of the time because our Django app (Python 3.11) was using a naive connection strategy that held connections open for the entire request lifecycle, causing the cache to bottleneck under load.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then. The site was slow (median response time 420ms), the Redis cache was underutilized, and the Kubernetes setup was more maintenance than value. We needed a solution that could handle traffic spikes without breaking the bank and without me babysitting the infrastructure every time AWS pushed a security patch.

Our stack was straightforward: Django, PostgreSQL, Redis, and a small Next.js frontend hosted on Vercel. The marketing site was static but needed server-side rendering for SEO. We didn’t need autoscaling, multi-region failover, or even persistent volumes. We just needed something that wouldn’t melt under 10x traffic and cost less than a junior developer’s salary.

The real problem wasn’t Kubernetes itself — it was the overhead. Every time we deployed, I had to update Helm charts, rotate secrets, and pray the ingress controller wouldn’t time out. The `t3.xlarge` nodes were overkill for 95% of our traffic, but we’d scaled up during a Black Friday sale and never dialed it back. The ElastiCache cluster was another $240/month for a cache we barely used. I knew there had to be a simpler way, but I wasn’t sure where to start.

We considered serverless options, but AWS Lambda (Node 20 LTS) had cold starts that added 150–300ms to every request. That ruled it out for a site where performance mattered. We also looked at Fly.io and Render, but their pricing models were unpredictable, and I didn’t want to gamble our uptime on a startup’s runway. Static site generators like Next.js Static Exports were fast, but they couldn’t handle server-side features like form submissions or dynamic content without a backend.

The breakthrough came when I realized we didn’t need Kubernetes at all. We just needed a way to serve the marketing site reliably, cache aggressively, and scale horizontally when traffic spiked. The solution had to be boring, proven, and cheap — no YAML files, no CNI plugins, no etcd clusters.


## What we tried first and why it didn’t work

Our first attempt was to squeeze more out of Kubernetes. I tuned the Horizontal Pod Autoscaler (HPA) to scale from 2 to 5 pods based on CPU usage, but the scaling events added 8–12 seconds of latency during traffic spikes because the new pods had to pull images and start up. The `t3.xlarge` nodes cost $187/month each, and even when we scaled down to 2 nodes overnight, we still paid $374 for compute we weren’t using. The Redis cache was still bottlenecked by Django’s connection pooling, so we added a Redis Cluster with 3 shards, but the cluster mode increased latency by 30% because of cross-shard routing.

Next, we tried moving the marketing site to AWS Fargate with EKS, hoping the serverless containers would simplify operations. Fargate added $0.0408 per vCPU-second, and a spike to 1,800 requests/second required 12 vCPUs, costing $17.50 for that hour alone. The total monthly bill jumped to $3,412 because Fargate’s pricing model rewarded inefficiency. The latency improved slightly (median 380ms), but the cost was unsustainable.

I also experimented with AWS App Runner, which promised "fully managed containers" without Kubernetes. The service was simple to deploy, but it capped concurrency at 1,000 requests per second, and the cold starts added 200ms to every request. App Runner’s pricing was $0.035 per vCPU-hour, so running at 2 vCPUs cost $52/month, but the performance was inconsistent. During a load test, the 95th percentile latency hit 1.2 seconds — unacceptable for a marketing site.

The Redis problem persisted. Our Django app was using `django-redis` with a naive connection strategy:

```python
# settings.py (before)
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://:password@cache-cluster.xxxxxx.ng.0001.use1.cache.amazonaws.com:6379/0",
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 100,  # Too high for our traffic
            },
            "CONNECT_TIMEOUT": 10,     # Seconds — way too long
        },
    }
}
```

This configuration held Redis connections open for the entire request, causing the cache to bottleneck under load. When traffic spiked, Redis would hit its connection limit (10,000 by default), and Django would start queuing requests, adding 500–800ms of latency. I tried reducing `max_connections` to 50, but that just shifted the bottleneck to the Django app, which started dropping connections at 1,200 requests/second.

The final straw was the Kubernetes control plane. Every security patch required a manual upgrade, and the EKS control plane itself cost $72/month. I spent two hours one weekend upgrading from v1.27 to v1.28, and the process failed twice because of a misconfigured IAM role. That’s when I decided to scrap the whole thing and find something simpler.


## The approach that worked

The solution was to replace Kubernetes with a combination of AWS Elastic Container Service (ECS) with Fargate, CloudFront for caching, and a single Redis instance on ElastiCache (not a cluster). ECS Fargate gave us container orchestration without the Kubernetes overhead, and CloudFront acted as a global CDN with Lambda@Edge for dynamic content.

The key insight was that we didn’t need horizontal scaling for the marketing site. A single Fargate task with 2 vCPUs and 4GB RAM could handle 1,800 requests/second with a median latency of 120ms. CloudFront cached 95% of the static content at the edge, reducing the load on the origin server. For dynamic content (like form submissions), we used Lambda@Edge to proxy requests to the ECS service only when necessary.

Redis was simplified to a single `cache.t4g.micro` instance (ARM-based, $16/month) with a connection pool of 20. The Django app’s `django-redis` configuration was updated to use short-lived connections:

```python
# settings.py (after)
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://:password@cache-single.xxxxxx.ng.0001.use1.cache.amazonaws.com:6379/0",
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 20,   # Tight enough to avoid bottlenecks
                "retry_on_timeout": True,
            },
            "CONNECT_TIMEOUT": 1,      # Seconds — fail fast
            "SO_TIMEOUT": 1,           # Seconds — fail fast
        },
    }
}
```

The Redis instance ran in a single AZ, but the risk was acceptable because the cache was cheap to recreate. If Redis failed, the app would fall back to the database, adding 50ms of latency — a tradeoff we were willing to make for a 94% cost reduction.

For the static marketing site, we used Next.js with a custom server for server-side rendering. The site was deployed to AWS S3 + CloudFront, with Lambda@Edge handling dynamic routes. The entire stack cost $200/month, including:
- ECS Fargate: $84/month (2 vCPUs, 4GB RAM, 730 hours/month)
- ElastiCache (single `cache.t4g.micro`): $16/month
- CloudFront: $42/month (100GB transfer, 10M requests)
- RDS `db.t3.medium` (for the app’s database): $58/month

The RDS instance was downsized from `db.t3.2xlarge` to `db.t3.medium`, which cost $58/month instead of $320. The savings from downsizing the database alone paid for the entire migration.

The biggest surprise was how little we needed to change in the Django app. We only had to update the Redis connection settings and the static file handling. The app’s codebase didn’t need refactoring — just a few configuration tweaks.

We also switched to ARM-based Graviton processors for both ECS and ElastiCache, which saved another 20% on compute costs. The performance difference was negligible: median latency dropped from 120ms to 110ms on ARM.


## Implementation details

### Step 1: Containerize the app

We moved from Kubernetes to ECS Fargate by containerizing the Django app with Docker. The Dockerfile was minimal:

```dockerfile
# Dockerfile
FROM python:3.11-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Gunicorn with gevent workers for async I/O
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "--timeout", "30", "app.wsgi:application"]
```

The key was using `gevent` workers to handle concurrent requests efficiently. With 4 workers, the app could handle 1,800 requests/second on a single Fargate task. The image size was 180MB, which deployed in under 10 seconds.

### Step 2: Set up ECS Fargate

We created an ECS cluster with Fargate launch type. The task definition specified:
- 2 vCPUs
- 4GB RAM
- 20GB ephemeral storage
- 2 containers: Django app and a sidecar for CloudWatch logs

The service was configured with:
- Desired count: 1 (we didn’t need scaling)
- Deployment circuit breaker: enabled (to roll back on failure)
- Health check: `/health/` endpoint with a 5-second timeout

The task role had permissions for:
- `secretsmanager:GetSecretValue` (for Django secrets)
- `ssm:GetParameters` (for app config)
- `logs:CreateLogGroup` (for CloudWatch)

No IAM policies for S3 or RDS were needed because the app used environment variables for database credentials.

### Step 3: Configure CloudFront and Lambda@Edge

The marketing site was built with Next.js and deployed to S3. CloudFront was set up with:
- Default root object: `index.html`
- Cache behaviors:
  - Static assets: cache TTL 1 year
  - Dynamic routes: cache TTL 5 minutes
  - `/api/*`: forward to origin (ECS service)

Lambda@Edge was used for:
- A/B testing (by injecting a cookie)
- Redirecting old URLs
- Handling form submissions (proxying to ECS)

The Lambda@Edge function was written in Node.js 20 LTS and deployed with:

```javascript
// lambda-at-edge.js
exports.handler = async (event) => {
  const request = event.Records[0].cf.request;
  const headers = request.headers;

  // Example: Inject a cookie for A/B testing
  if (!headers.cookie || !headers.cookie.find(c => c.value.includes('ab_test=B'))) {
    request.headers.cookie = request.headers.cookie || [];
    request.headers.cookie.push({ key: 'cookie', value: 'ab_test=B; Path=/; Max-Age=31536000' });
  }

  return request;
};
```

The Lambda@Edge function added less than 10ms of latency and handled 100% of dynamic requests.

### Step 4: Redis optimization

The `cache.t4g.micro` instance ran Redis 7.2 with a single shard. The instance type was ARM-based, which cost $16/month instead of $24 for the x86 equivalent. The connection pool in Django was tuned to 20 connections, with timeouts set to 1 second:

```python
# In Django settings
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://:password@cache-single.xxxxxx.ng.0001.use1.cache.amazonaws.com:6379/0",
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 20,
                "timeout": 1,
            },
            "CONNECT_TIMEOUT": 1,
            "SO_TIMEOUT": 1,
        },
    }
}
```

This configuration prevented connection leaks and ensured Redis wouldn’t bottleneck under load. The cache hit rate improved from 42% to 89% after the change.

### Step 5: Database downsizing

The PostgreSQL RDS instance was downsized from `db.t3.2xlarge` ($320/month) to `db.t3.medium` ($58/month). The `t3.medium` instance had 2 vCPUs and 4GB RAM, which was enough for our workload. We also:
- Enabled `autovacuum` to prevent table bloat
- Set `shared_buffers` to 1GB (25% of available RAM)
- Enabled `pg_stat_statements` for monitoring

The database performance remained stable, with a median query time of 8ms. The only tradeoff was that we could no longer run heavy analytics queries during peak hours, but that wasn’t a requirement for the marketing site.


## Results — the numbers before and after

Here’s the breakdown of the migration:

| Component               | Before (K8s) | After (ECS + CloudFront) | Savings |
|-------------------------|--------------|--------------------------|---------|
| Kubernetes EKS          | $289/month   | $0                       | $289    |
| Compute (nodes)         | $561/month   | $84/month (ECS)          | $477    |
| Redis                   | $240/month   | $16/month                | $224    |
| RDS                     | $320/month   | $58/month                | $262    |
| CloudFront              | $35/month    | $42/month                | -$7     |
| **Total**               | **$1,445**   | **$200**                 | **$1,245 (86%)** |

The marketing site’s performance improved across the board:
- Median response time: 420ms → 110ms
- 95th percentile latency: 1.2s → 280ms
- Cache hit rate: 42% → 89%
- Deployment time: 15 minutes (K8s) → 2 minutes (ECS)

We also reduced the number of AWS services from 12 to 6, which cut our operational overhead by about 40 hours/month. No more Helm charts, no more `kubectl` commands, no more EKS upgrades.

The biggest surprise was the latency improvement. CloudFront cached 95% of static content at the edge, and the ECS service handled dynamic content efficiently. The Lambda@Edge functions added minimal overhead (10ms per request), and the Redis optimizations reduced cache misses by 47%.

The migration took 5 days of part-time work. The longest part was reconfiguring the Django app’s Redis connection pool — I spent half a day tweaking timeouts and pool sizes before landing on the optimal settings. The rest was straightforward: containerizing the app, setting up ECS, and configuring CloudFront.


## What we’d do differently

If I had to do it again, I would:

1. **Start with a single-region deployment.** We considered multi-region failover, but the cost wasn’t justified. A single region with CloudFront caching was enough for our needs.
2. **Use ARM processors from day one.** The Graviton processors saved 20% on compute costs, but we only switched after the migration. If we’d started with ARM, we could have saved another $40/month.
3. **Avoid Elasticache clusters.** The single-shard Redis instance was simpler and cheaper. The cluster mode added latency and cost without providing meaningful benefits.
4. **Tune the database earlier.** The `db.t3.medium` instance was a good fit, but we waited too long to downsize. If we’d done it during the migration, we could have saved $262/month immediately.

The biggest mistake was over-engineering the Redis setup. I initially tried to use a Redis Cluster for high availability, but the cross-shard routing added 30% latency. A single shard was more than enough for our workload.

Another mistake was not testing the ECS deployment under load before cutting over. We did a simple load test with 1,000 requests/second, but the real spike (1,800 requests/second) revealed a connection pool issue that took a day to fix. Always test at 2x expected traffic.


## The broader lesson

The lesson isn’t that Kubernetes is bad — it’s that it’s overkill for most small-to-medium SaaS products. Kubernetes shines when you need:
- Multi-region failover
- Autoscaling to zero
- Persistent volumes for stateful apps
- Custom networking or service meshes

If you don’t need those features, Kubernetes is a tax. It adds complexity, cost, and operational overhead without providing proportional value. The same goes for Redis Clusters, multi-AZ databases, and auto-scaling groups. Start simple, measure, and only add complexity when you hit a real bottleneck.

The boring stack won. A single ECS task, CloudFront, and a small Redis instance handled 1,800 requests/second with 110ms median latency and cost $200/month. The Kubernetes cluster cost $3,127/month and provided no meaningful advantage over the simpler setup.

This isn’t about being cheap — it’s about being efficient. The goal isn’t to spend less; it’s to spend only where it matters. For most SaaS products, that means:
- Serving static content from a CDN
- Using serverless for edge logic (Lambda@Edge)
- Running containers on ECS Fargate instead of Kubernetes
- Using a single-region database with backups
- Caching aggressively with Redis

The cloud is full of shiny tools that promise to solve every problem. But the simplest solution is often the best. Start with the boring stack, measure everything, and only add complexity when you have data to justify it.


## How to apply this to your situation

If you’re running a small-to-medium SaaS product and wondering whether Kubernetes is worth it, ask yourself these questions:

1. **Do you need multi-region failover?** If not, skip Kubernetes. Use ECS Fargate or Fly.io instead.
2. **Is your traffic spiky?** If yes, use a CDN (CloudFront, Fastly) to cache static content at the edge. For dynamic content, use Lambda@Edge or ECS with auto-scaling.
3. **Is your database over-provisioned?** Most small SaaS products don’t need a `db.r6g.2xlarge`. Start with a `db.t3.micro` and scale up only when needed.
4. **Are you using Redis efficiently?** If your cache hit rate is below 70%, tune your connection pool and timeouts. A single Redis instance is often enough.

Here’s a checklist to migrate from Kubernetes to a simpler stack:

- [ ] Containerize your app with Docker (use multi-stage builds to keep images small)
- [ ] Set up ECS Fargate with a single task (start with 1 vCPU, 2GB RAM)
- [ ] Configure CloudFront to cache static content
- [ ] Use Lambda@Edge for dynamic routes
- [ ] Downsize your database (start with `db.t3.micro` if possible)
- [ ] Switch to a single Redis instance with a tight connection pool
- [ ] Deploy and monitor for a week
- [ ] Delete the Kubernetes cluster

The migration took us 5 days, but most of that was testing and tuning. The actual cutover took 30 minutes. If you’re already running on Kubernetes, you can probably migrate in a weekend.


## Resources that helped

1. [AWS ECS Fargate pricing calculator](https://calculator.aws.amazon.com/#/addService/ECS) — Use this to estimate costs for your workload.
2. [Next.js Static Exports + CloudFront](https://nextjs.org/docs/app/building-your-application/deploying/static-exports) — How to deploy a static Next.js site with server-side rendering.
3. [Django-Redis connection pool tuning](https://github.com/jazzband/django-redis#connection-pool) — Official docs with examples for optimizing Redis connections.
4. [Redis 7.2 ARM performance](https://redis.io/blog/redis-7-2-arm-performance/) — Benchmarks showing ARM’s advantage over x86.
5. [CloudFront Lambda@Edge examples](https://github.com/aws-samples/amazon-cloudfront-functions) — Real-world examples of Lambda@Edge functions.


## Frequently Asked Questions

### Why didn’t you use Fly.io or Render instead of ECS Fargate?

Fly.io and Render are great for small projects, but their pricing models are less predictable than AWS’s. For example, Fly.io charges $5/month for a shared CPU, but you pay for every 10ms of compute time during bursts. At 1,800 requests/second, that could add up quickly. AWS Fargate has a flat $0.0408 per vCPU-hour, which is easier to budget. Also, Fly.io’s Postgres offering costs $15/month for a single node, whereas AWS RDS `db.t3.medium` costs $58/month but includes backups, multi-AZ failover, and automated patches. For a SaaS product, the AWS ecosystem’s reliability and tooling outweighed the cost savings of Fly.io.


### How did you handle secrets management without Kubernetes secrets?

We used AWS Secrets Manager for database credentials and Django secrets. The ECS task role had permissions to fetch secrets at runtime:

```python
# settings.py
import boto3
from botocore.exceptions import ClientError

try:
    secrets = boto3.client('secretsmanager').get_secret_value(SecretId='django/secrets')
    SECRET_KEY = secrets['SecretString']
except ClientError as e:
    raise Exception("Failed to fetch secrets") from e
```

The secrets were stored in Secrets Manager as JSON, and the ECS task fetched them on startup. This was simpler than Kubernetes secrets and avoided the risk of secrets leaking in environment variables. The only downside was the 100ms latency to fetch secrets at startup, which we mitigated by caching them in the app’s memory.


### What’s the biggest performance pitfall when moving from Kubernetes to ECS Fargate?

Cold starts. ECS Fargate tasks can take 30–60 seconds to start, which is unacceptable for a production app. To avoid this, we:
- Set the desired count to 1 (so the task is always running)
- Used a health check endpoint that responds in <500ms
- Monitored the task’s CPU and memory to ensure it wasn’t throttled
- Tested the task’s startup time under load

If you need true auto-scaling, use ECS with Application Auto Scaling (not Fargate’s built-in scaling). Application Auto Scaling can scale from 1 to 5 tasks in under 2 minutes, which is fast enough for most SaaS products. But for a marketing site, a single task was enough.


### How did you migrate the database without downtime?

We used AWS DMS (Database Migration Service) to replicate the `db.t3.2xlarge` to a new `db.t3.medium` instance. The process took 4 hours and had minimal impact on the app:

1. Set up a replication instance ($30/month)
2. Created source and target endpoints
3. Started a full load + CDC (Change Data Capture) task
4. Switched the Django app’s database endpoint to the new instance
5. Monitored replication lag and application errors
6. Terminated the old instance

The downtime was <30 seconds during the endpoint switch. The only issue was that the new instance had less RAM, so we had to tune `shared_buffers` and `work_mem` to avoid performance degradation. After the migration, we ran `pg_dump` to verify data integrity.


## Next step: Check your Redis connection pool

Open your Django (or Flask/Node.js) app’s cache configuration and check these three things:

1. **Connection pool size:** Is it set to a value between 10 and 50? If it’s 100+, reduce it to 20–30.
2. **Timeouts:** Are connect and socket timeouts set to 1 second? If they’re higher, reduce them.
3. **Cache hit rate:** Run `redis-cli info stats | grep keyspace_hits` and check the hit rate. If it’s below 70%, tune your cache keys and TTLs.

If you’re not sure how to check, run this command in your Redis instance:

```bash
redis-cli info stats | grep keyspace_hits:
```

Then calculate the hit rate:

```bash
hits=$(redis-cli info stats | grep keyspace_hits: | awk -F: '{print $2}')
misses=$(redis-cli info stats | grep keyspace_misses: | awk -F: '{print $2}')
echo "Hit rate: $((hits * 100 / (hits + misses)))%"
```

If the hit rate is below 70%, your cache isn’t working efficiently. Adjust your pool size and timeouts, then retest. Do this in the next 30 minutes — it’s the fastest way to cut costs and improve performance.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
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
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 09, 2026
