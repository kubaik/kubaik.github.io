# Only boring stacks survive AI services

Most platform-abstraction guides assume a clean environment and a patient timeline. Tutorials show the happy path. What follows is the reasoning behind a deliberately unglamorous architecture, the failure modes it avoids, and how to tell when it stops being enough.

## The conventional wisdom and where it breaks

Standard advice for building AI-powered services assumes a team of five or more engineers, each with a dedicated ops or platform role. The playbook: start with microservices, add Kubernetes, sprinkle in feature flags, finish with a monitoring stack. For a solo founder who is also the sole engineer, this advice is worse than useless.

Microservices and Kubernetes are complexity amplifiers. They make sense when a team is large enough that Conway's Law starts to bite, or when hundreds of services have genuinely different scaling needs. For a solo founder shipping an AI service, the overhead of managing even one Kubernetes cluster often outweighs the benefits. The problem is not that microservices are bad; it is that they are prematurely optimized for an org chart that does not exist yet.

A typical pattern: the first service ships quickly, the second takes twice as long, and by the third, deployments are unreliable and debugging is painful. The trap is not the architecture itself but the assumption that a sophisticated platform is needed early. Most solo founders do not need Kubernetes. They need something that works today, is easy to explain to a non-technical co-founder, and can scale without daily firefighting.

The mismatch is between platform complexity and team capacity. Every new abstraction introduces a new failure surface, and for a solo engineer, failure surfaces multiply faster than debugging ability.

## The "just one more abstraction" trap

A common failure mode is incremental accretion. A project starts as a simple web app behind a reverse proxy, then needs retries, so a task queue and a broker appear. Then feature flags. Then a CDN for latency spikes. A representative stack after a year of this:

- NGINX (reverse proxy)
- Flask (app)
- Celery (task queue)
- Redis (broker and cache)
- PostgreSQL (primary database)
- A feature-flag SDK
- A CDN for static assets
- Prometheus plus Grafana (monitoring)
- Docker Compose for local development
- Kubernetes "for scalability"

This stack is not rare. It is a typical progression for teams that start simple and layer on tools as pain points appear. The tools themselves are fine. The problem is hidden coupling. A retry storm in Celery can saturate Redis, which slows the feature-flag lookups, which time out and trigger 5xx responses in the web app. The stack becomes a Rube Goldberg machine of dependencies.

A representative failure: the sole engineer is unavailable and the system breaks. There is no on-call rotation because there is no team. The non-technical co-founder cannot SSH into a Kubernetes pod to restart a service. Error messages sit too deep in the stack. `celery.exceptions.TimeoutError: Queue full after 30s` does not say that Redis is out of memory and swapping, which is why the queue filled up.

The real cost is not the cloud bill. It is cognitive load. Every service adds a mental model the solo engineer must maintain. At three services, things are fine. At ten, half the week goes to debugging dependency conflicts instead of building product. The stack becomes a distraction from the core value: the model and the user experience around it.

## A survivability-first mental model

For a solo founder building AI services, the goal is survivability, not scalability. The stack should handle growth for a few months without requiring a platform team. That means favoring boring, proven tools with well-documented failure modes.

A representative boring stack:

- FastAPI (app framework)
- An ASGI server to run it
- PostgreSQL with a connection pooler such as pgBouncer
- Redis (cache and rate limiting)
- Celery (background tasks, usually one queue)
- A PaaS host with managed Postgres and Redis add-ons
- GitHub Actions for CI/CD
- Sentry for error tracking
- Cloudflare for DNS and static-asset CDN
- No Kubernetes, no feature-flag service, no custom dashboards

Most AI services do not need microservices. They need a monolith that scales vertically (a bigger VM) before it scales horizontally (more VMs). The monolith keeps the codebase small, deployment simple, and debugging straightforward. The model is usually the bottleneck, not the web server.

### A worked example, with arithmetic

A solo founder builds an audio transcription service using Whisper. Version one: a FastAPI app that accepts a file, runs inference, and stores the result in PostgreSQL. It runs on one VM with 8 GB RAM and 4 vCPUs. Traffic is 50 requests per day. The stack is simple enough to explain in 30 seconds.

Three months later, traffic is 2,000 requests per day. Assume each request costs roughly 2 seconds of CPU-bound inference. That is:

- 2,000 requests/day × 2 s = 4,000 CPU-seconds/day
- 4,000 / 86,400 s in a day ≈ 0.046 CPU-seconds per wall-clock second
- Spread over 4 vCPUs, average utilization ≈ 0.046 / 4 ≈ 1.2%

Average utilization is not the constraint. Peak is. If traffic arrives in a two-hour evening window, 2,000 requests × 2 s = 4,000 CPU-seconds compressed into 7,200 seconds, or about 0.56 CPU-seconds per second, roughly 14% of 4 vCPUs. Still fine. The VM reports 90% CPU only if inference is slower than assumed, requests arrive in bursts, or other work competes for the same cores.

The response is to scale vertically: 16 GB RAM, 8 vCPUs. Same stack, no new services, no new abstractions, no Kubernetes or Terraform. The system survives. The arithmetic above is illustrative; the point is that the decision is driven by measured peak utilization, not by a feeling that the stack "should" be bigger.

Deferring complexity is not ignoring scalability. It is waiting until real traffic data identifies what actually needs to scale. Most AI services never reach that point. Those that do usually need model optimization, not infrastructure.

## Measuring when vertical scaling stops working

Before adding any component, instrument four numbers and watch them over a week:

1. **Peak CPU and memory per instance.** Sample every 15 seconds. Averages hide the bursts that cause timeouts.
2. **p50, p90, p99 latency per endpoint.** Percentiles, not means. A rising p99 with a flat p50 usually means queueing, not slow code.
3. **Database connection pool saturation.** Track active versus idle connections and wait time for a connection. A pooler such as pgBouncer exposes these as stats.
4. **Queue depth and age.** For background work, the age of the oldest unprocessed item is more informative than the count.

The signals that justify a new component:

- Peak CPU above roughly 70% sustained for hours, with p99 latency climbing in the same window.
- Database write throughput saturating the primary, confirmed by disk I/O wait and replication lag.
- Background queue age growing without bound during normal traffic.
- A single endpoint consuming a disproportionate share of CPU, such that isolating it would meaningfully reduce blast radius.

If none of these hold, the correct action is usually a bigger instance, a query index, or a cache, not a new service.

## Cache stampedes and the fix that fits a monolith

A typical failure mode in these systems is a cache stampede. A new feature goes live, traffic spikes, and cache invalidation lags. PostgreSQL sees a multiple of normal load. The error pattern is clear: p99 latency jumps from around 120 ms to around 1.8 s, and error tracking lights up with timeout errors.

The fix is not more caching. It is to spread out invalidation using probabilistic early refresh:

```python
import random
from fastapi import FastAPI

app = FastAPI()

CACHE_TTL = 300  # 5 minutes
PROBABILITY = 0.2  # 20% chance to refresh early

@app.get("/items/{item_id}")
async def read_item(item_id: str, use_cache: bool = True):
    if not use_cache:
        # Bypass cache for testing
        return {"data": await expensive_query(item_id)}

    data = redis.get(f"item:{item_id}")
    if data is None or random.random() < PROBABILITY:
        # Refresh cache early with 20% probability
        data = await expensive_query(item_id)
        redis.setex(f"item:{item_id}", CACHE_TTL, data)

    return {"data": data}
```

This is trivial in a monolith. In a microservice architecture, coordinating cache invalidation across services usually means introducing a message bus or event sourcing, another layer with its own latency and failure modes.

## Background tasks without a second platform

Most AI services need to run inference on a schedule or process uploaded files. A task queue with Redis as the broker is the boring choice:

```python
from celery import Celery
import whisper

celery = Celery('tasks', broker='redis://redis:6379/0')

@celery.task(bind=True, max_retries=3)
def transcribe_audio(self, file_url: str) -> str:
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_url)
        return result["text"]
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
```

Note that loading the model inside the task is wasteful if the worker processes many tasks. Load it once at worker startup and reuse it:

```python
from celery import Celery
import whisper

celery = Celery('tasks', broker='redis://redis:6379/0')
model = whisper.load_model("base")

@celery.task(bind=True, max_retries=3)
def transcribe_audio(self, file_url: str) -> str:
    try:
        result = model.transcribe(file_url)
        return result["text"]
    except Exception as exc:
        self.retry(exc=exc, countdown=60)
```

The trade-off is that a queue adds a moving part, but it is a well-understood one. The alternative, running inference in the web process, blocks the API and causes timeouts. The boring stack accepts a small increase in complexity for a large gain in reliability.

## Where the conventional wisdom is right

Two scenarios justify more infrastructure for a solo founder.

**Regulated industries.** If the service handles health data, financial transactions, or government-regulated content, strict audit trails, separate environments, and compliance checks may be required. Isolating services reduces blast radius and simplifies compliance. Even here, start with a monolith and split only when compliance forces it, not by premature optimization.

**Planned team growth.** If engineers will be hired within 6 to 12 months, some platform investment now can prevent future pain. Keep it boring: one cluster, templated deployments, a shared database, a simple CI/CD pipeline. Avoid serverless, service meshes, and GitOps until at least three engineers can maintain them.

Outside these cases, the conventional wisdom is a trap. Microservices are a tool for managing team size, not product complexity. A solo founder needs a product that works today and can grow without daily firefighting.

## A decision checklist

Ask four questions before adding any component.

1. **What is the blast radius of a single service failing?**
   For a chatbot API, one VM failing may mean sub-second errors for a few users. For a payment processor, it may mean lost revenue. Higher blast radius argues for isolation, but isolation does not require microservices. Multiple VMs behind a load balancer achieve much of it within one service.

2. **How much traffic variability is expected?**
   A niche tool grows slowly; vertical scaling suffices. A viral consumer app can spike tenfold in a day, which argues for horizontal scaling. Start with one service and add a load balancer or read replicas before splitting.

3. **How many hours per week are available for platform maintenance?**
   If the answer is fewer than four, do not add the abstraction. Time spent debugging Kubernetes is time not spent on the model, the UX, or distribution.

4. **What measurement justifies this change?**
   Name the metric and the threshold. "p99 latency above 1 s for three consecutive days" is a reason. "It feels more scalable" is not.

A comparison of common shapes:

| Scenario | Recommended shape | Reversible? |
|----------|-------------------|-------------|
| Low blast radius, slow growth | Single app + Postgres + Redis on a PaaS | Yes |
| Low blast radius, fast growth | Same, plus a load balancer in front of multiple VMs | Yes |
| High blast radius, slow growth | Same, plus a read replica and error tracking | Yes |
| High blast radius, fast growth | Read/write split, Redis cluster, multiple regions | Harder |
| Compliance-bound, multi-team | Separated services with audit trails | Hard |

Only the last two rows warrant splitting services, and only after vertical and single-service horizontal scaling have been exhausted and measured.

## Common objections

**"What if the service needs to scale to a million users?"**

Most AI services never reach that scale. Those that do usually need a different kind of scalability: model optimization, CDN caching, or edge deployment, not microservices. If that scale arrives, traffic data will identify the bottleneck, and there will be resources to hire help for a refactor. Until then, the boring stack buys time to validate the product.

**"The boring stack feels old-fashioned. Modern apps use serverless."**

Serverless platforms add latency and cost for some AI workloads. Cold starts can add hundreds of milliseconds to seconds to a request, which is unacceptable for interactive chat. Workers are faster but impose CPU and memory limits. The boring stack gives predictable latency and cost. If serverless is needed, use it for specific functions such as image resizing, not the core API.

**"What about vendor lock-in?"**

Lock-in is a risk, but it is not unique to this stack. Kubernetes lock-in is often worse because it is harder to move between clouds. PostgreSQL, Redis, and most PaaS hosts have open-source or portable equivalents. Migration is usually a matter of changing connection strings and deployment targets, not rewriting the platform.

**"The cool kids are using agents and event-driven architectures."**

Event-driven architectures add complexity that is justified when multiple teams and services need to coordinate. For a solo founder, an event bus is another moving part that can fail, and its failure modes are harder to reason about than a function call. Stick to the monolith until events solve a measured problem.

## If starting over today

Five changes worth making on a new project:

1. **Start with FastAPI rather than Flask.** Async support and automatic OpenAPI docs fit AI services well. The performance difference is negligible at low traffic, but the developer experience is better as the app grows.

2. **Use a PaaS with managed Postgres and Redis.** Managed add-ons save configuration time. Pricing is more predictable than assembling RDS and ElastiCache by hand, and the developer experience is better for a solo founder.

3. **Prefer a simpler queue than Celery for new projects.** Redis Queue (RQ) is simpler, needs no extra services, and covers retries and scheduling. Celery is more feature-rich; for solo founders, simplicity wins.

4. **Do not build a custom feature-flag service.** A database flag or a YAML file in the repository is enough. For dynamic flags, use Redis with a TTL. A dedicated flag service is not worth the complexity until multiple teams need it.

5. **Measure latency from day one.** Log p50, p90, and p99 per endpoint. Store the data in a time-series database or a plain file. A spike will be visible immediately, without waiting for a monitoring stack.

A minimal FastAPI middleware to start:

```python
from fastapi import FastAPI, Request
import time
import statistics

app = FastAPI()
latencies = []

@app.middleware("http")
async def log_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time
    latencies.append(latency)
    if len(latencies) > 100:
        latencies.pop(0)
    return response

@app.get("/stats")
async def get_stats():
    return {
        "p50": statistics.median(latencies),
        "p90": sorted(latencies)[int(len(latencies) * 0.9)] if latencies else 0,
        "p99": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
    }
```

This gives real-time visibility without a heavy monitoring stack. Note that the in-process list is per worker, so with multiple workers the stats are per worker, not global. For a single-VM deployment that is usually acceptable; for multiple VMs, aggregate the logs instead.

## Summary

The boring stack is reliable rather than glamorous. It survives early growth because it is built for survivability, not scalability. Microservices and Kubernetes are tools for managing team size, not product complexity. For solo founders, their overhead outweighs the benefits until there is real traffic and a team to manage it.

Systems that work well start simple and grow vertically before horizontally. They use proven tools with a minimal operational surface area. They measure latency from day one and fix problems while they are small. They avoid premature abstraction.

The assumption that a sophisticated platform is needed early is the part that trips people up. Most AI services never need more than a monolith with a few well-understood dependencies. The boring stack buys time to validate the product, iterate on the model, and focus on users rather than infrastructure.

## FAQ

**How do I know when to split the monolith into microservices?**

Split only when a single service becomes a bottleneck that vertical scaling cannot solve. Signs: database writes saturating the primary, background tasks backing up without bound, or API latency consistently high under normal load. Even then, consider splitting into two services (API and worker) before going full microservices. Most solo founders never reach this point.

**Is PostgreSQL enough for an AI service?**

For most AI services, yes. PostgreSQL with a connection pooler handles thousands of requests per second on a single VM. If more is needed, add a read replica or partition the data. The bottleneck is almost always the model or the API layer, not the database. Do not over-engineer the database until measurements show it is the problem.

**What is the simplest way to add horizontal scaling without Kubernetes?**

Put a load balancer in front of multiple VMs running the same service. Most PaaS hosts support this out of the box. Each VM runs the same app and the load balancer distributes traffic. This is horizontal scaling without service discovery or orchestration.

**How should secrets and environment variables be handled?**

Use environment variables with a `.env` file for local development and the platform's secrets manager for production. Avoid hardcoding secrets in code or Docker images. One file for local, one command to set production secrets. Add more sophistication only when there is a concrete need.

## Take action in the next 30 minutes

Add the latency middleware above to one endpoint, deploy it, and record p50, p90, and p99 for the next 24 hours. That single measurement will tell you more about whether your stack needs a new component than any architecture diagram.
