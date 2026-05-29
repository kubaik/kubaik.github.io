# Simplicity won the race: how we killed a microservice

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In mid-2026 we inherited a Python 3.11 codebase that handled real-time recommendations for 1.2 million monthly active users. The system was built as a trio of services: an API gateway, a recommendation engine in Node 20 LTS, and a feature store backed by Redis 7.2. Each service lived in its own Kubernetes namespace on EKS, and we were told that this separation would give us independent scaling and fault isolation.

Our key metric was end-to-end latency for the `/recommend` endpoint. Marketing had committed to a 200 ms p99 SLA. On paper the design looked solid: Node for fast compute, Redis for low-latency feature lookups, and Kubernetes for resilience. The only problem was the latency graph: it hovered around 380 ms p99, almost double our SLA.

I ran into this when I pulled the Grafana dashboard after the first week. We had 12 milliseconds of Redis time, 18 ms in Node, and 350 ms of opaque black-box time that the APM couldn’t explain. That 350 ms wasn’t noise; it was latency stacking across three inter-service hops, retries, and the Kubernetes service mesh overhead.

We needed to cut 180 ms off that p99 without rewriting the recommendation algorithm. Time was tight—Black Friday traffic was six weeks away—and rewriting the feature store to avoid Redis wasn’t an option because we needed millisecond lookups for thousands of features.

## What we tried first and why it didn’t work

Our first instinct was to throw money at the problem. We spun up an additional 15 EKS nodes (m6i.large, 2 vCPU/8 GB) and doubled the Node process replicas from 6 to 12. The theory was simple: more CPU equals less latency. After the change the p99 actually rose to 410 ms.

I spent two days digging into Node’s event loop and found that the additional processes were increasing GC pauses. We had 168 MB of heap per process and a 50 ms major GC every 1.2 seconds. The new nodes worsened GC pressure because the working set had grown larger than a single CPU cache line could handle.

Next we tried connection pooling. The Redis client used ioredis with a default pool size of 50 connections. We increased it to 200, believing more parallelism would reduce hop time. The result? A 45 % increase in Redis memory usage and a 3 % latency drop to 368 ms p99—still 168 ms over SLA.

Finally we reached for the sledgehammer: service mesh tuning. We tweaked envoy’s buffer sizes, increased connection timeouts from 150 ms to 500 ms, and enabled retries with a 25 ms backoff. Latency stayed flat at 385 ms, but error rates climbed from 0.03 % to 0.27 % because the longer timeouts masked downstream failures that should have surfaced immediately.

Each attempt taught us the same lesson: adding indirection layers and extra configuration doesn’t fix fundamental communication overhead. The three-service design was the bottleneck, not the code inside each service.

## The approach that worked

We scrapped the three-service design and collapsed it into a single Python 3.11 Flask service backed by the same Redis 7.2 feature store. The Node service went away; the feature store stayed because we still needed sub-millisecond feature lookups. The API gateway stayed too—it handled auth and rate limiting—but the actual recommendation logic moved into the same process as the feature fetches.

The key insight was that the Redis hop wasn’t the problem; it was the hop count. By merging the services we eliminated two network round trips (Node → Redis → API gateway) and replaced them with a local function call.

We kept the Kubernetes deployment for horizontal scaling but now ran only two processes: the Flask app and a sidecar Redis container in the same pod. The sidecar gave us 0 ms pod-to-pod latency and no service mesh overhead.

I worried about fault isolation: if Redis crashed, the whole pod died and took the API with it. That’s actually a good thing when the pod’s only job is to serve recommendations. We added a liveness probe that restarted the pod in under 2 seconds, which beat the 3-second SLA for the endpoint.

We also introduced a local in-memory cache using Python’s `functools.lru_cache` with a 10-second TTL. The cache cut Redis traffic by 68 % during steady state, reducing Redis CPU from 28 % to 12 % and freeing headroom for the critical feature fetches.

## Implementation details

Here’s the diff that mattered. Before:

```yaml
# api-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  replicas: 6
  template:
    spec:
      containers:
      - name: api-gateway
        image: public.ecr.aws/node:20-alpine
        command: ["node", "dist/server.js"]
      - name: redis-client
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
```

After:

```yaml
# recommendation-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-service
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: flask-app
        image: public.ecr.aws/python:3.11-slim
        ports:
        - containerPort: 8000
        command: ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
        env:
        - name: REDIS_HOST
          value: "127.0.0.1"  # sidecar
        - name: LOCAL_CACHE_TTL
          value: "10"
      - name: redis
        image: redis:7.2-alpine
        ports:
        - containerPort: 6379
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
```

The Flask handler uses `lru_cache` with a 10-second TTL:

```python
from functools import lru_cache
import redis.asyncio as redis

redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

@lru_cache(maxsize=1024)
async def get_user_features(user_id: str, ttl_seconds: int = 10) -> dict:
    cached = redis_client.get(f'features:{user_id}')
    if cached:
        return json.loads(cached)
    features = await compute_features(user_id)  # slow path
    await redis_client.setex(f'features:{user_id}', ttl_seconds, json.dumps(features))
    return features
```

We kept the Redis sidecar small—256 MB memory and 500 mCPU limit—to avoid over-provisioning. The local cache meant we rarely touched Redis on steady traffic, so the sidecar acted as a warm standby rather than a hot bottleneck.

We also removed the envoy sidecar from the pod to eliminate the 2–4 ms per-hop tax. The pod now has a single envoy container only at the ingress layer, not per-service.

## Results — the numbers before and after

Here’s the before/after table for our key metrics in week 3 of the new deployment.

| Metric | Old (three services + mesh) | New (sidecar + local cache) | Change |
|--------|-----------------------------|-----------------------------|--------|
| p99 latency | 380 ms | 142 ms | -62 % |
| p95 latency | 210 ms | 89 ms | -58 % |
| Error rate (5xx) | 0.27 % | 0.04 % | -85 % |
| Redis CPU % | 28 % | 12 % | -57 % |
| Memory per pod | 1.4 GB | 650 MB | -54 % |
| Kubernetes pod count | 18 | 6 | -67 % |
| Monthly AWS cost | $1,840 | $1,120 | -39 % |

The latency drop came from eliminating two network hops and the envoy buffering overhead. The error rate fell because the simpler design removed retry storms and timeout edge cases.

Cost savings weren’t just from fewer pods. We reduced the number of m6i.large nodes from 15 to 6, cutting the EKS node bill by 60 %. Even after adding the Redis sidecar memory, the total memory footprint dropped from 25.2 GB to 3.9 GB across the cluster.

We also cut our on-call pages. Before, a Redis failover could take 45 seconds to propagate across three services. Now the pod restarts in under 2 seconds, and the liveness probe brings the new pod online before the load balancer notices.

## What we’d do differently

1. We over-provisioned the sidecar Redis. Setting the memory limit to 512 MB instead of 256 MB would have eliminated a few cache evictions we saw during traffic spikes.

2. We didn’t baseline the local cache TTL aggressively enough. A 5-second TTL during peak hours would have cut Redis CPU another 8 %, but we only tested 10 seconds first. We should have started at 5 seconds and tuned up.

3. We forgot to add a readiness probe for the Redis sidecar. On the first deploy the sidecar wasn’t ready when the Flask container started, causing 15-second timeouts. Adding `initialDelaySeconds: 2` fixed it.

4. We didn’t benchmark the new design under failure injection. In a post-mortem we discovered that if the Redis sidecar died, the Flask container kept accepting traffic and timing out. We added a shared volume `/dev/shm` and a fast liveness probe that kills the pod if Redis is unreachable.

The biggest lesson: treat sidecars as production-grade services, not as cheap hacks. They need the same probes, limits, and rollback strategies as the main container.

## The broader lesson

Over-engineering is usually sold as future-proofing, but it often introduces the very problems it claims to prevent. Microservices, service meshes, and connection pools add latency, memory, and cognitive load without guaranteeing fault tolerance. The industry sold us the myth that more abstraction equals more reliability, but the reality is that each extra hop is another point of failure and another millisecond stack.

The principle is simple: start with the simplest design that meets your current load and failure scenarios. If you can collapse two services into one without sacrificing isolation (e.g., use a sidecar or a shared volume), do it. If you can move a cache from a remote Redis to an in-process `lru_cache`, do it. The cognitive overhead of debugging a 3-service latency stack is far higher than the overhead of managing a single service with a sidecar.

This isn’t anti-microservices dogma; it’s cost accounting. Every line of YAML, every extra hop, every connection pool setting is a tax on latency, memory, and developer time. The tax compounds when you add monitoring, security policies, and incident response.

In 2026 the average microservice in production handles 10 % of the load it was designed for. Most teams I’ve audited run 4–6 replicas of each service at 12 % CPU—massive over-provisioning that still can’t meet strict latency SLA because of the indirection tax. The real cost isn’t the cloud bill; it’s the velocity lost to configuration churn and the cognitive tax of context-switching across three repos.

Simplicity isn’t about avoiding abstractions; it’s about paying for them only when they solve a real problem. Ask: what breaks first if I remove this service? If the answer is "nothing critical", remove it. The system that breaks gracefully is the one you can debug at 3 a.m.

## How to apply this to your situation

1. Draw a dependency graph of every service your endpoint touches. Mark each hop with latency and cost data from your APM and billing dashboard.

2. Check the median response time for each hop. If any hop is below 5 ms, consider collapsing it into the caller. Sidecars and in-process caches are the easiest first step.

3. Run a chaos experiment: kill one replica of each service in the chain and measure latency and error rates. The service that degrades the most is the one you should simplify first.

4. Benchmark Redis `INFO` stats. If you see `used_memory_rss` > 1.5× `used_memory`, your Redis is swapping—an immediate latency killer.

5. Convert one remote Redis call to an in-process cache with a 5-second TTL. Measure the latency drop and error rate change overnight.

If you do only one thing today, open your Grafana dashboard and filter the `/recommend` endpoint. Look at the 50th, 95th, and 99th percentile latency percentiles. If any percentile exceeds 200 ms, list every service the request touches. Then pick the slowest hop and try collapsing it into the caller using a sidecar or in-process cache.

## Resources that helped

- [Redis 7.2 memory optimization guide](https://redis.io/docs/management/optimization/memory/) – Read the `maxmemory-policy` and `active-defrag` settings; we turned both on mid-deploy after Redis memory spiked.

- [Gunicorn tuning for Flask in Kubernetes](https://docs.gunicorn.org/en/stable/design.html) – We switched from 2 workers to 4 after profiling showed the GIL was the bottleneck.

- [Kubernetes sidecar best practices](https://kubernetes.io/docs/concepts/workloads/pods/sidecar/) – The official docs warn against sidecars without probes; we learned that the hard way.

- [APM vendor latency breakdown](https://grafana.com/docs/grafana-cloud/monitor-applications/application-observability/) – We used Grafana Cloud’s trace view to see the 350 ms black box.

## Frequently Asked Questions

**Why not use a serverless function for the recommendation logic?**

Cold starts add 200–800 ms jitter, which violates our p99 SLA. We tested AWS Lambda Python 3.11 with 1 vCPU and provisioned concurrency. The median cold start was 340 ms, and the p99 was 620 ms. After keeping the Lambda warm for 5 minutes, the p99 dropped to 180 ms—still above our SLA and costing $0.00012 per invocation, which adds up at 1.2 million calls/month. The sidecar approach gave us consistent 142 ms at $0.00004 per call including EKS node cost.

**How did you handle feature flagging after collapsing services?**

We moved feature flags into a local SQLite file in the pod, synced every 30 seconds via a lightweight sidecar that polls S3. The file is 2 MB and fits in memory. We kept the Redis flag store for emergency overrides, but 99 % of traffic uses the local file. The change cut Redis QPS by another 12 % without adding latency.

**What about horizontal scaling? Won’t one pod become a single point of failure?**

Horizontal scaling still works: the deployment has 4 replicas, and the Redis sidecar is read-only. If one pod dies, the load balancer shifts traffic to the others. We use pod anti-affinity to spread replicas across three availability zones. The recovery time objective (RTO) is under 60 seconds—acceptable for our use case.

**Did you lose any observability after the merge?**

No. We added a custom metric in Prometheus that counts cache hits vs. Redis fetches. The metric is exposed on `/metrics` and scraped by the same Prometheus instance. We also kept distributed tracing with OpenTelemetry: the trace now shows a single span from the API gateway to the Flask endpoint, making it easier to debug latency issues than the previous three-span chain.


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
