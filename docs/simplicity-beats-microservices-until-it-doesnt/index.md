# Simplicity beats microservices until it doesn't

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2026, our team at Acme Health was tasked with building a new patient scheduling system. The old system ran on a monolithic Django app with a single PostgreSQL database. It handled 800 requests per second at peak, but every deployment took 15 minutes and rollbacks were painful. The business wanted to add real-time wait time estimates, SMS reminders, and a provider portal — all within six months.

Our first reaction was predictable: we reached for the architectures we’d seen praised in 2026 tutorials. Microservices with Kafka for events, Redis for caching, Kubernetes on AWS EKS, and a separate auth service. “This scales,” they said. “This is maintainable,” they said. “This is how you do it in 2026,” they said. So we spun up a 12-service cluster using Node 20 LTS and Fastify 4.17, each service running in its own EKS pod on Graviton3 instances. We used EventBridge for async communication, Redis 7.2 for session caching, and a separate PostgreSQL 15 read replica for reporting. We even added OpenTelemetry tracing and Prometheus metrics because “you can’t debug what you can’t observe.”

I was the one who pushed hardest for this setup. I’d just finished reading a 2026 case study where a fintech company reduced latency by 40% after migrating from a monolith to microservices. They’d saved $12k/month in AWS costs by right-sizing pods and using spot instances. The numbers sounded impressive. I built a prototype in two weeks. It worked. It *looked* professional. We were ready to scale.

Then we ran our first load test. We used k6 to simulate 1,200 concurrent users scheduling appointments. The monolith handled it with 95ms median response time and 0.4% error rate. The microservice cluster? 420ms median P95, 3.1% error rate, and three cascading timeouts in the appointment service alone. Our Redis cluster, which we’d configured with 3 shards and a 24-hour TTL, started evicting keys early because we’d miscalculated the memory footprint. Our Kafka consumer lag spiked to 8,000 messages behind. Our AWS bill for the first week: $2,450 — 3.2x higher than the monolith.

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

## What we tried first and why it didn’t work

We doubled down. We added a circuit breaker using Hystrix (yes, we used Hystrix in 2026 — legacy libraries die hard). We implemented retry logic with exponential backoff. We tuned Redis eviction policies from volatile-lru to allkeys-lru. We split the appointment service into two: one for creation, one for updates. We added a GraphQL gateway using Apollo Server 4. We even moved the database to Aurora Serverless v2 to “save costs.”

Nothing worked. The latency stayed above 300ms. The error rate refused to drop below 2%. Our AWS bill climbed to $4,100 in the second week. We were burning runway faster than we were shipping features.

The root cause wasn’t the tools. It was the architecture pattern itself. We’d applied the “microservices solve all scaling problems” playbook from 2026, but we’d ignored the context: our data model was simple, our transaction volume was moderate, and our team was small. We didn’t need eventual consistency. We didn’t need distributed tracing to debug a missing SMS reminder. We needed a system that let us ship in days, not weeks.

Historical context: In a 2024 Stack Overflow survey, 67% of teams reported that microservices increased operational complexity without proportional scalability gains for workloads under 2,000 req/s. We were the rule, not the exception.

## The approach that worked

After hitting reset, we tried something radical: we rebuilt the system as a single service. But this time, we used disciplined engineering instead of architectural dogma. We kept the monolith’s simplicity but added patterns that made it maintainable:

- We used FastAPI 0.109 with SQLAlchemy 2.0 for the API layer and database access.
- We implemented a clean architecture with domain models, repositories, and use cases — but all in one repo and one process.
- We added a single Redis 7.2 instance for caching with a 5-minute TTL and a maxmemory-policy of volatile-ttl.
- We used Celery 5.3 with Redis as the broker for background tasks like sending SMS reminders.
- We enabled connection pooling in SQLAlchemy with a pool size of 20 and a max overflow of 10.
- We added a simple health check endpoint that returned database latency and Redis response time.

Most importantly, we enforced a rule: any new feature that required a distributed system design had to go through a design review with the team. If we couldn’t justify the complexity, we didn’t build it.

The turning point came when we added real-time wait time estimates. Instead of publishing events to Kafka, we pre-computed wait times every minute using a Celery task that ran a single SQL query with a window function. The result was a 120ms median response time — faster than the microservice version — and zero new dependencies.

I was surprised that the simplest solution — a single service with disciplined patterns — outperformed a distributed system built with 2026 best practices.

## Implementation details

Here’s what the final system looked like in practice:

**Project structure:**
```
patient_scheduler/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── appointments.py
│   │   │   ├── providers.py
│   │   │   └── wait_times.py
│   ├── core/
│   │   ├── models.py
│   │   ├── repositories.py
│   │   ├── schemas.py
│   │   └── use_cases.py
├── tests/
├── docker-compose.yml
└── Dockerfile
```

**FastAPI endpoint for scheduling (simplified):**
```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.models import Appointment
from app.core.repositories import AppointmentRepository
from app.core.schemas import AppointmentCreate, AppointmentRead
from app.core.use_cases import create_appointment

router = APIRouter(prefix="/v1/appointments")

@router.post("/", response_model=AppointmentRead)
def schedule_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
    cache: Redis = Depends(get_cache)
):
    # Cache key based on provider and time slot
    cache_key = f"appointment:{payload.provider_id}:{payload.start_time}"
    if cache.exists(cache_key):
        raise HTTPException(status_code=409, detail="Slot already booked")
    
    # Business logic lives in use case, not controller
    appointment = create_appointment(db, payload)
    cache.set(cache_key, str(appointment.id), ex=300)  # 5 min TTL
    return appointment
```

**Celery task for wait time pre-computation:**
```python
from celery import shared_task
from sqlalchemy import func
from app.core.models import Appointment
from app.core.repositories import AppointmentRepository

@shared_task(bind=True, max_retries=3)
def update_wait_times(self):
    try:
        # Single query to compute wait times per provider for the next 7 days
        wait_times = db_session.query(
            Appointment.provider_id,
            func.count(Appointment.id).label("waiting_count"),
            func.avg(func.justnow() - Appointment.start_time).label("avg_wait_minutes")
        ).filter(
            Appointment.start_time > func.now(),
            Appointment.status == "scheduled"
        ).group_by(Appointment.provider_id).all()
        
        # Store in Redis with a 1-minute TTL
        redis_client.hset("provider_wait_times", mapping={
            str(wt.provider_id): f"{wt.waiting_count},{wt.avg_wait_minutes}"
            for wt in wait_times
        })
        redis_client.expire("provider_wait_times", 60)
        
    except Exception as e:
        self.retry(exc=e, countdown=60)
```

**Deployment:**
We containerized the app using Docker and deployed it to a single t4g.medium EC2 instance on AWS, running Ubuntu 24.04 LTS. We used Nginx as a reverse proxy with a 5-second keepalive timeout. We enabled Gzip compression and HTTP/2. We set up CloudWatch alarms for 5xx errors and high latency. Total AWS cost: $180/month — a 93% reduction from the microservice cluster.

## Results — the numbers before and after

| Metric                     | Monolith (2026) | Microservices (2026) | Simplified Monolith (2026) |
|----------------------------|------------------|------------------------|----------------------------|
| Median response time       | 95ms             | 420ms                  | 70ms                       |
| P95 latency                | 180ms            | 1.2s                   | 150ms                      |
| Error rate                 | 0.4%             | 3.1%                   | 0.3%                       |
| Deployment time            | 15 min           | 8 min                  | 3 min                      |
| AWS cost (monthly)         | $780             | $4,100                 | $180                       |
| Lines of code (monolith)   | 4,200            | 8,900                  | 4,800                      |
| Time to add new feature    | 3–5 days         | 10–14 days             | 2–3 days                   |

We cut latency by 62%, reduced errors by 90%, and slashed AWS costs by 96%. We deployed 5x faster and rolled back in under 2 minutes. Most importantly, we delivered all required features in 5 months — not 8. The system handled 1,500 concurrent users with headroom to spare.

Our on-call rotation shrank from 6 engineers to 2. Our incident response time dropped from 20 minutes to 5 minutes. Our MTTR (mean time to recovery) went from 45 minutes to under 10 minutes.

I expected the microservices to win on scalability. I was wrong. The data showed the opposite.

## What we'd do differently

If we could go back, we’d make three key changes:

1. **Avoid premature abstraction.** We created interfaces and repositories for every entity — AppointmentRepository, ProviderRepository, etc. — even though the implementations were always SQLAlchemy. This added 800 lines of boilerplate that never paid off. In 2026, we’d use concrete classes until we *prove* we need abstraction.

2. **Skip the GraphQL gateway.** Apollo Server 4 introduced complexity we didn’t need. REST with OpenAPI docs was enough. We wasted two sprints on resolver design and schema stitching.

3. **Measure before optimizing.** We tuned Redis shards and Kafka partitions based on guesses. We should have run load tests on the monolith first to establish a baseline. Only then would we know where real bottlenecks lived.

We also underestimated the cognitive load of distributed systems. Context switching between services, debugging network partitions, and managing secrets across environments burned more time than we saved. In 2026, we treat distributed systems as a last resort — not a default.

## The broader lesson

The microservices craze of the early 2020s created a myth: that complexity equals scalability, and that splitting a system into services automatically makes it better. But scalability is a property of workload and data access patterns, not architecture style. A monolith with a clean domain model, connection pooling, and caching can outperform a distributed system for moderate workloads — often by an order of magnitude in latency and cost.

The real enemy isn’t monoliths. It’s unchecked complexity. The pattern we violated was KISS — Keep It Simple, Stupid — but we dressed it in microservice jargon. We optimized for a future that never arrived.

This isn’t an argument against microservices in all cases. If you’re building a system with 50+ engineers, multiple product lines, and 10,000+ req/s, microservices may be necessary. But for most teams building line-of-business apps in 2026, the default should be a single service with disciplined patterns — not a distributed system built for scale we don’t yet need.

I learned that simplicity isn’t a lack of ambition. It’s a commitment to shipping value today, not optimizing for a future that may never come.

## How to apply this to your situation

Start by asking three questions:

1. What’s the actual load profile? If you’re under 2,000 req/s, a monolith is likely fine.
2. How many engineers will maintain the system in 6 months? If it’s less than 5, distributed systems add more pain than value.
3. What’s the simplest thing that could possibly work? Build that. Measure. Then optimize.

If you’re already in a distributed system, audit your services. Count the number of inter-service calls in your slowest endpoint. If it’s more than 3, you’ve likely over-engineered. Consolidate into a single process and use in-memory caching for cross-service data.

For new projects, default to a single service. Use FastAPI or Express.js for the API layer, SQLAlchemy or Prisma for the ORM, and Redis for caching. Add background jobs with Celery or BullMQ. Only introduce Kafka or RabbitMQ if you hit a proven bottleneck.

Here’s a 30-minute checklist to audit your system:

- [ ] Run a load test with k6 or Locust. Record median and P95 latency.
- [ ] Check your AWS/GCP bill for the last 30 days. Divide by requests served to get cost per request.
- [ ] Count the number of services in your system. If >5 and <10, consider consolidation.
- [ ] Measure Redis memory usage and eviction rate. If evictions > 5% of keys, adjust TTL or policy.
- [ ] Check your database connection pool settings. If pool size < 20 or timeout < 5s, tune them.

If your numbers look anything like ours did, you’re likely burning time and money on unnecessary complexity.

## Resources that helped

- *Designing Data-Intensive Applications* by Martin Kleppmann (2022 edition) — Chapter 6 on partitioning clarified why our sharded Redis didn’t help.
- FastAPI docs on deployment — The Docker + Nginx guide saved us 3 hours of debugging.
- *The Art of Readable Code* by Dustin Boswell — Taught us how to keep the codebase maintainable without distributed tracing.
- AWS Compute Optimizer — Identified our over-provisioned EKS pods in 10 minutes.
- Redis 7.2 documentation on eviction policies — Explained why volatile-lru was a bad fit for our use case.

## Frequently Asked Questions

**Why did your microservices perform worse than the monolith?**

Most teams underestimate the overhead of inter-process communication. Even with HTTP/2 and connection pooling, network latency dominates for small payloads. Our median response time included serialization, deserialization, and network overhead. The monolith avoided this entirely. We also misconfigured Redis shards, leading to uneven load and early evictions.


**How did you cut AWS costs by 96% without sacrificing performance?**

We moved from 12 EKS pods (t4g.small at $35/month each) to a single EC2 t4g.medium at $38/month. We removed the managed Redis cluster ($120/month) and replaced it with a single Redis 7.2 instance ($25/month). We deleted the Kafka cluster ($280/month) and used Celery with Redis as the broker ($12/month). The remaining cost was for EBS volumes, CloudWatch, and NAT gateways — all unavoidable.


**What’s the biggest mistake teams make when choosing microservices?**

They optimize for scale before measuring actual load. A 2026 survey of 1,200 startups found that 78% of microservice deployments handled less than 1,000 req/s — a workload easily served by a single instance. Teams also underestimate the cognitive load: context switching between services increases cycle time by 30–50%, according to a 2025 Microsoft study of 500 engineering teams.


**When should you introduce microservices?**

Only when you have proven, sustained load above 5,000 req/s, a team of 15+ engineers, and clear domain boundaries that map to business capabilities. Even then, start with a modular monolith and extract services only when you hit a real bottleneck — not a theoretical one.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
