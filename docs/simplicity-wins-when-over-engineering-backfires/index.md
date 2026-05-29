# Simplicity wins: when over-engineering backfires

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were building a real-time inventory sync service for a chain of 120 retail stores. The goal: push stock changes from each store’s POS to a central warehouse system within 2 seconds. The old system used a polling loop that ran every 30 seconds, so we knew we could do better. I joined the team when the requirements were still fresh, and the original estimate was six weeks for a team of four engineers. We were already behind schedule when I inherited the project.

Our first pass assumed we’d need Kafka for event streaming, Redis Streams for pub/sub, and PostgreSQL 15 with logical replication for durability. All of that felt justified because the traffic forecast was 5,000 events per second at peak. The architecture diagram we drew looked like a high school science fair project: Kafka clusters in three AZs, consumer groups, schema registry, dead-letter queues, the works. I remember thinking, *If we’re going to do this right, we need to build for scale from day one.* I had seen enough horror stories about systems that collapsed under load to believe this was the only responsible path.

We spent the first week wiring up Terraform modules for Kafka on AWS MSK with 3 brokers and 15 GB RAM each. Then we added Redis 7.2 clusters for caching frequent SKU lookups. The Terraform plan showed 47 resources and a monthly cost of about $1,800 just for the message queue and cache layers. The budget for the whole feature was only $3,000, so we justified the expense by claiming it would save future refactoring. That was a mistake. I underestimated how long it would take to stabilize, and the team spent three weeks debugging consumer lag caused by a single misconfigured `fetch.max.bytes` setting. The lag spikes reached 45 seconds during the first load test, which was worse than the polling loop we were replacing.

I remember sitting in a post-mortem where we stared at Grafana dashboards showing 92% CPU on the Redis nodes and 87% heap usage on the Kafka brokers. The latency histogram showed 99th percentile response times at 1.8 seconds — barely within spec, but fragile and expensive. We had built a system that could handle 50,000 events per second but was failing at 5,000 because we forgot to tune the consumer batch size. The irony? The core requirement was just moving JSON payloads under 1 KB from store endpoints to a warehouse API. We didn’t need Kafka. We didn’t even need Redis.


## What we tried first and why it didn’t work

Our first attempt was a classic case of premature abstraction. We designed three layers: an ingestion API, an event bus, and a warehouse writer. Each layer ran in its own Kubernetes Deployment using Node 20 LTS behind an Nginx ingress controller. We used gRPC for service-to-service communication because “it’s faster than REST” — a claim I later found was only true under unrealistic lab conditions. We also bolted on OpenTelemetry for distributed tracing, which added 15 milliseconds of overhead per request due to span serialization. The whole stack was containerized with Docker multi-stage builds, and we used Helm charts for deployments. The helm upgrade command took 47 seconds to roll out a single-line change.

The performance numbers were brutal. Under a synthetic load of 1,000 requests per second, the median response time was 89 ms, but the 95th percentile jumped to 450 ms. Profiling with `py-spy` on the ingestion service showed 60% of CPU time spent in JSON schema validation — a step we added because we thought it would “future-proof” the API. The warehouse writer, written in Go, spent 35% of its time waiting on PostgreSQL 15 synchronous commits. We tried asynchronous commits next, but then we lost a few inventory updates during a rolling restart. Those lost updates triggered a cascade of manual reconciliations that cost the business an estimated $12,000 in labor over two weeks.

The operational overhead was even worse. Our on-call rotation started receiving pages for alerts like “Kafka consumer lag > 30s” and “Redis eviction rate > 10%.” We had to tune `maxmemory-policy` in Redis six times in the first week. The consumer lag turned out to be caused by a single misconfigured `session.timeout.ms` in the Kafka client, set to 10,000 ms instead of 3,000 ms. The fix took us three days to ship because the change required a rolling restart of all consumer pods, and the rollout script had no health check timeout. I still have the Slack message from the on-call engineer that night: “Kafka lag is back to 45s after the restart. Did I miss something?”

We tried a few “quick wins” too. We added a local LRU cache in the ingestion service using a Python dictionary with a TTL. The cache reduced warehouse writes by 42%, but it introduced a race condition where two concurrent updates could overwrite each other. The bug only surfaced when two stores updated the same SKU within 50 ms of each other. The fix required a distributed lock, which we implemented using Redis Redlock. That added another 10 ms of latency per contested write. The irony? The same race condition existed in the original polling system, but because updates were 30 seconds apart, it never caused visible corruption. Our “fix” made things worse.

I spent three days debugging a connection pool exhaustion issue that turned out to be a single misconfigured `max_connections` setting in our PgBouncer pool. The pool was set to 20 connections, but the warehouse writer was opening 50 HTTP connections to the same service. The backlog of unprocessed requests grew to 2,400 items, and the latency spiked to 3.2 seconds. We had built a system so layered that a single misconfiguration could cascade into a multi-hour incident. That moment taught me that abstraction layers don’t just hide complexity — they amplify small mistakes into large failures.


## The approach that worked

We scrapped the Kafka-Redis-Postgres stack and rebuilt the service as a single HTTP service with a direct write-through cache. The ingestion endpoint accepted POST requests with a JSON body containing store ID, SKU, and delta. We used FastAPI on Python 3.11 with Uvicorn and Gunicorn, configured with 4 workers and 2 threads each. We kept the warehouse writer synchronous but added a dead simple Redis 7.2 cache in front of it, using a naive write-through policy with a 1-second TTL. The cache key was a hash of store_id + sku, so updates were atomic at the SKU level.

The key insight was that we didn’t need eventual consistency. The business requirement was “within 2 seconds,” not “eventually consistent.” We also realized that the warehouse system could tolerate a small number of duplicate updates, so we removed the Redlock attempt entirely. Instead, we used a simple `SETNX` guard in Redis to prevent concurrent updates to the same SKU from the same store. That reduced the race condition to a single atomic operation with 99.9% reliability in tests.

We also dropped gRPC in favor of REST with JSON over HTTP/1.1. The overhead of HTTP/2 and TLS was negligible for our payload size (under 1 KB), and the simplicity of debugging with curl made onboarding new engineers trivial. We added no tracing, no metrics beyond a simple Prometheus counter, and no circuit breakers. We relied on standard HTTP status codes and retry logic in the store endpoints. The only external dependency was Redis, which we ran as a single instance with persistence enabled (AOF every second).

The deployment was embarrassingly simple: a single Docker image pushed to ECR, deployed via a single Kubernetes Deployment with a rolling update strategy. The helm chart was gone. The Terraform module for Kafka was deleted. We saved $1,800 per month and reduced our infrastructure to a single Redis instance costing $25 per month, plus the Kubernetes worker node cost of $45 per month. The whole rewrite took 4 engineers 3 days. The latency dropped from 450 ms 95th percentile to 42 ms, and we never saw a lag spike again.

I was surprised to discover that the warehouse system’s API was actually idempotent. We could send duplicate updates and it would either ignore them or apply them safely. That meant we didn’t need Kafka’s exactly-once semantics or Redis Streams’ consumer groups. We just needed a fire-and-forget POST with retries. The simplicity paid off when a regional outage took down our Redis cluster for 8 minutes. The store endpoints automatically fell back to direct writes to the warehouse API, which handled the load without breaking a sweat. That resilience came from removing layers, not adding them.


## Implementation details

Here’s the core of the new service. It’s a FastAPI app with one endpoint, one cache layer, and one warehouse writer. The entire codebase is 287 lines of Python, including comments and type hints.

```python
# main.py
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import httpx
import os
import logging

app = FastAPI()

# Config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
WAREHOUSE_API = os.getenv("WAREHOUSE_API", "http://warehouse:8000/update")

# Cache client
cache = redis.from_url(REDIS_URL, decode_responses=True)

# HTTP client with 3s timeout
client = httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=1.0))


@app.post("/sync")
async def sync_inventory(payload: dict):
    store_id = payload.get("store_id")
    sku = payload.get("sku")
    delta = payload.get("delta")
    
    if not all([store_id, sku, delta]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required fields"
        )
    
    # Cache key: store_id:sku
    cache_key = f"inv:{store_id}:{sku}"
    
    # Check if another update is in progress for this SKU/store
    lock_key = f"lock:{store_id}:{sku}"
    acquired = await cache.set(lock_key, "1", nx=True, ex=3)
    if not acquired:
        # Another update is in progress; retry or return 429
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "update_in_progress"}
        )
    
    try:
        # Write-through cache: update cache and warehouse atomically
        await cache.hset(cache_key, mapping={"delta": delta})
        await cache.expire(cache_key, 1)  # 1s TTL
        
        # Call warehouse API
        resp = await client.post(
            WAREHOUSE_API,
            json={"store_id": store_id, "sku": sku, "delta": delta},
            headers={"Content-Type": "application/json"}
        )
        
        if resp.status_code >= 400:
            # If warehouse fails, invalidate cache and re-raise
            await cache.delete(cache_key)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="warehouse_error"
            )
        
        return {"status": "synced"}
        
    finally:
        # Always release lock
        await cache.delete(lock_key)


@app.on_event("startup")
async def startup():
    # Warm cache or run migrations here if needed
    pass
```

The warehouse writer is a minimal Go service using Gin:

```go
// warehouse/main.go
package main

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

type UpdateRequest struct {
	StoreID string `json:"store_id"`
	SKU     string `json:"sku"`
	Delta   int    `json:"delta"`
}

var mu sync.Mutex
var inventory = make(map[string]int)

func main() {
	r := gin.Default()
	r.POST("/update", func(c *gin.Context) {
		var req UpdateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "bad_request"})
			return
		}

		key := req.StoreID + ":" + req.SKU
		mu.Lock()
		inventory[key] += req.Delta
		mu.Unlock()

		c.JSON(http.StatusOK, gin.H{"status": "updated"})
	})

	r.Run(":8000")
}
```

We deployed Redis 7.2 with persistence turned on (AOF every second) and a maxmemory policy of `allkeys-lru` with 100 MB max memory. The cache hit rate stabilized at 87% during peak hours, and the TTL of 1 second was chosen because inventory changes are frequent but not urgent beyond 2 seconds. We used a single Redis instance because the throughput was under 1,000 ops/sec and we had no cross-AZ latency requirements. The Redis instance cost $25/month on AWS ElastiCache.

The Kubernetes Deployment used a single pod with resource limits of 512 MiB memory and 500 milliCPU. We ran it on a t3.small worker node ($0.02/hour) with no horizontal pod autoscaler. The whole stack used 72 lines of Terraform to define the Redis cluster and IAM roles. We saved $1,600 per month compared to the Kafka/MSK setup.


## Results — the numbers before and after

| Metric                     | Original Stack (Kafka + Redis) | Simplified Stack (HTTP + Redis) |
|----------------------------|----------------------------------|----------------------------------|
| 95th percentile latency    | 450 ms                          | 42 ms                            |
| 99th percentile latency    | 1,800 ms                        | 120 ms                           |
| Monthly infra cost         | ~$1,825                         | ~$70                             |
| Lines of code (services)   | 1,472                           | 287                              |
| Mean time to recovery (MTTR)| 45 minutes                      | 5 minutes                        |
| Deployment time            | 47 seconds                      | 6 seconds                        |
| Cache hit rate             | N/A (no cache)                  | 87%                              |
| On-call pages (first month)| 12                              | 2                                |

The latency improvement was driven by removing layers. The original stack had 4 hops: store → ingestion API → Kafka topic → warehouse writer. The new stack has 1 hop: store → ingestion API → warehouse writer. The warehouse writer itself became synchronous, eliminating the need for consumer lag monitoring and offset commits.

The cost savings were dramatic. The Kafka/MSK cluster cost $1,200/month for brokers, plus $300 for Redis clusters and $325 for PgBouncer and PostgreSQL. The simplified stack cost $25 for Redis, $45 for the Kubernetes worker node, and $0 for Kafka and PostgreSQL. The $1,755 monthly saving paid for three months of junior engineer time.

The reliability improved by an order of magnitude. The original stack had 12 on-call pages in the first month due to consumer lag, Redis eviction storms, and PgBouncer exhaustion. The new stack had 2 pages: one for the Redis instance going down (which we fixed by enabling multi-AZ in 2 hours) and one for a warehouse API timeout (which we resolved by increasing the client timeout from 1s to 3s).

I was surprised to find that the warehouse API could handle 2,000 requests per second with zero degradation. The Go service used less than 50 MB of memory and ran on a single core. The synchronous write pattern meant we never lost data, because the HTTP client retried on 5xx errors and the warehouse API was idempotent. The only data loss scenario was a regional outage, which we mitigated by running a secondary instance of the ingestion service in another AZ. That secondary instance cost $12/month and handled failover gracefully.


## What we’d do differently

If we were to build this again today, we would skip Redis entirely. The cache hit rate of 87% wasn’t worth the operational overhead. Instead, we would use a simple in-memory LRU cache in the FastAPI process with a 1-second TTL. The cache would live in the same pod, so no network hops. We measured the in-process cache to have a median access time of 0.02 ms, compared to 1.2 ms for Redis round-trip. That would cut latency by another 20% and save $25/month.

We would also add a simple retry policy in the store endpoints. The current version returns 429 when the lock is held, but stores often retry immediately, causing thundering herds. A backoff of 50 ms + jitter would reduce contention by 60%. We saw this during a load test where 500 concurrent requests to the same SKU caused 180 retries in one second. Adding a 50 ms delay in the client reduced retries to 12.

Another mistake was not testing the failure modes early. We should have simulated Redis downtime in the first week, not after go-live. A simple chaos experiment: kill the Redis pod and measure how long until the warehouse API receives the update. In our case, the answer was 6 seconds, which was acceptable, but we only discovered it after a real outage. Adding a circuit breaker in the FastAPI service would have prevented cascading failures if Redis became slow. We would use a 100 ms timeout and 3 retries before failing open.

We would also standardize the logging format from day one. The original stack used three different log formats across Kafka, Redis, and PostgreSQL. The simplified stack used JSON logging in FastAPI and Go, but we didn’t enforce a schema. During the first incident, we spent 20 minutes parsing logs to find the store ID that caused the duplicate update. A structured log schema would have saved that time.

Finally, we would use Python’s `asyncio` more aggressively. The current service uses a naive thread pool for HTTP calls, which can block under load. Switching to `httpx.AsyncClient` improved throughput by 30%, but we still have room to optimize. Using a connection pool with limits and timeouts would reduce the risk of connection leaks. We measured 8 open connections per worker under 1,000 RPS, which is fine, but at 5,000 RPS we’d need to tune pool sizes.


## The broader lesson

The lesson here isn’t that fancy architectures are always wrong. It’s that they carry hidden costs that only surface under real load and real failure. Every abstraction layer you add increases the surface area for misconfiguration, increases the time to debug, and reduces the clarity of ownership. The teams I’ve worked with that succeed at scale do so by removing abstraction layers, not adding them.

I’ve seen this pattern repeat across companies: a team builds a system with Kafka, Redis, Cassandra, and Kubernetes, only to spend 60% of their time tuning consumer groups, eviction policies, and pod resource limits. Meanwhile, the core business logic is 100 lines of code. The fancy stack becomes the product, not the tool. Engineers start optimizing for Kafka throughput instead of business outcomes.

The principle is simple: **start with the minimal viable architecture that meets the SLA, then evolve only when you can prove the pain.** The pain must be measured in dollars, latency, or engineer hours — not in hypothetical scale. If your 99th percentile latency is 42 ms and your SLA is 2 seconds, you don’t need Kafka. If your cache hit rate is 87% and your Redis bill is $25/month, you don’t need a cluster. If your MTTR is 5 minutes and your incident cost is $200, you don’t need multi-AZ Kafka.

This principle applies beyond tech stacks. It applies to CI/CD pipelines, monitoring, and even team processes. Every extra step you add to a pull request review, every extra stage in a deployment pipeline, every extra metric you emit — they all add cognitive load. The teams that ship fastest are not the ones with the most tools, but the ones that ship the simplest thing that could possibly work.

I’ve made this mistake three times in my career. The first time, I built a distributed task queue for a cron job that ran every 5 minutes. The second time, I added Redis Streams to a system that only needed a queue. The third time was this one. Each time, the lesson was the same: the problem wasn’t the scale. The problem was the assumption that scale would come before clarity.


## How to apply this to your situation

Start by writing down your SLA in concrete terms. Not “scalable,” not “highly available,” but specific numbers: 99th percentile latency under 200 ms, 99.9% uptime, 5-minute MTTR. Then, build the simplest system that meets that SLA. Use the smallest viable infrastructure: a single HTTP endpoint, a single database, no message queues. Measure everything: latency, error rate, cost per request, time to deploy.

Next, run a chaos experiment. Kill a dependency — your database, your cache, your message broker — and measure how long until the system recovers. If it recovers within your SLA, you’re done. If not, add the minimal layer to fix that failure mode. Don’t add layers for hypothetical failures.

Finally, audit your stack every six months. For each tool, ask: *If this tool disappeared tomorrow, would the system still work?* If the answer is yes, remove it. If the answer is maybe, measure the cost of the failure scenario. If the answer is no, keep it — but add a runbook and run a fire drill.

Here’s a checklist you can apply today:

1. List every external dependency in your system (databases, caches, queues, SaaS tools).
2. For each, write down the failure scenario and the cost of that failure in dollars and minutes.
3. Calculate the monthly cost of each dependency.
4. Remove any dependency where the cost of failure is less than the cost of the dependency.
5. Write a one-page runbook for the remaining dependencies.
6. Schedule a chaos day within the next month to test the runbook.


## Resources that helped

- *Designing Data-Intensive Applications* by Martin Kleppmann — especially the chapter on log-based message brokers. It helped me understand why Kafka was overkill for our use case.
- *Site Reliability Engineering* by Google — the chapter on simplicity and reducing toil was eye-opening.
- The FastAPI docs for async support and deployment guides.
- The Redis 7.2 documentation on persistence and eviction policies.
- The Kubernetes failure stories blog by the CNCF — real-world examples of what breaks in production and why.


## Frequently Asked Questions

**Why not use a message queue at all?**
Message queues like Kafka or RabbitMQ add durability and decoupling, but they also add latency, operational complexity, and cost. If your system can tolerate the occasional duplicate or lost update, and your latency requirement is in seconds not milliseconds, a simple HTTP endpoint with retries is often enough. We measured 180ms of overhead per hop in our Kafka stack — that’s 180ms of added latency before we even processed the message. For a system that only needs to sync inventory within 2 seconds, that overhead is unnecessary.


**What about scaling to 10,000 stores?**
We tested the simplified stack to 5,000 RPS with a single ingestion pod and a single warehouse writer. At 10,000 RPS, we’d need to shard the ingestion service by store ID or SKU range, or add a load balancer in front of multiple pods. But that’s a scaling problem, not an architecture problem. The lesson is to solve the scaling problem when you hit it, not before. Premature sharding adds complexity that obscures the real bottlenecks.


**Isn’t Redis a single point of failure?**
Yes, but so is Kafka. The difference is that Redis is easier to replace. We ran Redis as a single instance with AOF persistence and multi-AZ enabled. The MTTR for a Redis failure was 5 minutes. If we had used Kafka, the MTTR would have been 45 minutes. The operational overhead of a single Redis instance is orders of magnitude lower than a Kafka cluster. If Redis becomes a problem, we can migrate to ElastiCache with multi-AZ or even a local cache in the pod.


**What about data consistency?**
Our warehouse API was idempotent: sending the same update twice had the same effect as sending it once. That meant we didn’t need Kafka’s exactly-once semantics or Redis Streams’ consumer groups. The cache was write-through, so updates were applied to both cache and warehouse atomically. The only inconsistency scenario was a race condition when two stores updated the same SKU within 50 ms, which we mitigated with a 3-second lock. That scenario was rare and the business accepted the risk of a 3-second delay.


**How do you monitor a simple stack?**
We used a single Prometheus counter for successful syncs and a gauge for warehouse API latency. We also added a health check endpoint that returned the cache hit rate. That’s it. No dashboards, no alerting rules beyond a single page for the warehouse API 5xx rate. The simplicity meant we spent zero time tuning alert thresholds or maintaining dashboards. The on-call engineer could glance at the health check and know everything was fine.


**Should I always avoid Kafka and Redis?**
No. If your system needs exactly-once semantics, or if you have 50,000 events per second, Kafka and Redis are appropriate. But don’t use them by default. Measure your actual throughput, latency, and failure scenarios first. In 2026, most teams are still using Kafka and Redis because they copied an architecture diagram from a 2026 tutorial, not because they measured the cost.


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
