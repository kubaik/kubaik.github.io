# AI interviews broke system design: 3 patterns that

AI assistants now draft a large share of system design documents, Terraform modules, and Kubernetes manifests. The output is often structurally complete: metrics, dashboards, a CI pipeline, sensible-looking retry policies. The failure mode is not incompetence. It is that these tools optimize for completeness of the design document rather than resilience under real load. A design that passes a sandbox smoke test can still collapse in staging within minutes.

This article covers three failure patterns that show up first when AI-generated designs meet production-like conditions, why they happen, and how to detect and fix them. Each pattern includes the mechanism, a worked example with the arithmetic shown, corrected code, and a way to measure whether the fix actually worked.

## The symptom: green sandbox, red staging

The most common failure symptom is that the system works in the assistant's simulated environment and degrades in staging. AI-generated Terraform or Kubernetes manifests apply cleanly, pass smoke tests, and hit stated latency targets against synthetic traffic. Under real traffic, latency climbs, pods crash-loop, or costs spike.

The error messages are usually misleading. `Kubernetes node not ready` or `Terraform apply failed: Invalid subnet CIDR` are downstream dominoes, not root causes. The tool reports success because its sandbox is a small, well-behaved world: typically a Docker Compose or Kind cluster with synthetic request rates and a single region.

A typical failure mode: an AI-generated design uses a cache with a fixed TTL and no locking. The simulator drives uniform synthetic traffic, so the cache never expires en masse. Real traffic arrives in bursts — a payroll run, a batch import, a marketing send — and many keys expire within the same second. Every request misses, hits the database simultaneously, and latency spikes. The simulator had no burst shape, so it never reproduced this.

The core issue: the sandbox does not model traffic shape, partial failure, regional latency, downstream capacity limits, or cost under sustained load. The design document is complete; the system dynamics are missing.

## Why it happens: idealized training data and a thin sandbox

AI design assistants are trained on repositories, blog posts, and documentation that describe systems in their healthy state. Partial failures, noisy neighbors, regional outages, throttled volumes, and burst traffic are underrepresented because they are rarely written down as tutorials.

A second factor is the sandbox itself. Most assistants generate a design against a local or lightweight cluster. That environment has no regional round-trip latency, no downstream service with a hard capacity ceiling, and no cost meter. Anything the design assumes about capacity, latency, or consistency goes untested.

The practical consequence: the assistant will happily produce a design that assumes 5,000 transactions per second from a downstream gateway that documents a 1,000 TPS limit, or that shards data in a way that only works in one region. Nothing in the generated document signals the assumption.

## Fix 1 — retry logic without jitter or rate limiting

**Symptom:** the system passes sandbox tests but fails in staging under load, with latency spikes, crash-loops, or cost explosions.

The most common cause is AI-generated retry logic with backoff but no jitter, and no concurrency limit tied to downstream capacity. A generated policy often looks like this:

```javascript
{
  retry: {
    maxAttempts: 5,
    base: 100,
    exponent: 2
  }
}
```

This is not obviously wrong. The problem is arithmetic. Without jitter, every client that failed at the same moment retries at the same moment. With a base of 100ms and exponent 2, the retry schedule is 100ms, 200ms, 400ms, 800ms, 1600ms. If 1,000 clients fail together, all 1,000 retry together five times, and each retry wave adds load to a downstream service that is already degraded. If that service throttles at 1,000 TPS, the first retry wave alone is 1,000 requests arriving in the same 100ms window — roughly 10,000 requests per second instantaneously, ten times the ceiling.

The fix is jitter plus a rate limiter sized to the downstream limit:

```javascript
{
  retry: {
    maxAttempts: 5,
    base: 100,
    exponent: 2,
    jitter: 0.5,        // randomize each backoff in [0.5x, 1.5x]
    maxBackoff: 5000    // cap at 5s to avoid runaway delay
  },
  rateLimit: {
    tokensPerSecond: 1000, // match documented downstream TPS
    burst: 2000
  }
}
```

Jitter spreads the retry waves across time so they no longer stack. The rate limiter caps in-flight requests at the downstream ceiling so a retry storm cannot amplify load beyond what the dependency can absorb.

**How to measure the fix.** Instrument two counters: retries per second, and downstream throttling responses (HTTP 429 or equivalent). Before the fix, a partial outage produces a retry-rate spike that correlates with a 429 spike. After the fix, the retry rate should stay flat or rise gently, and 429s should stay near zero. A simple load harness:

```python
# retry_storm_test.py — count errors under a simulated partial outage
import asyncio
import aiohttp

async def simulate_retry_storm(url: str, clients: int = 1000):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for _ in range(clients)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        rate = len(errors) / clients * 100
        print(f"clients={clients} errors={len(errors)} ({rate:.1f}%)")

asyncio.run(simulate_retry_storm("https://api.example.com/payments"))
```

Run it against a staging endpoint with the downstream dependency throttled on purpose. Compare the error rate and the retry-rate counter before and after adding jitter and the limiter.

## Fix 2 — sharding that ignores multi-region constraints

**Symptom:** the system works in staging but produces data loss or inconsistency during regional failover or traffic spikes.

The cause is AI-generated sharding logic that assumes a single region. A generated shard function often looks like this:

```python
# AI-generated sharding logic in Python 3.11
from hashlib import md5

def shard_transaction(transaction_id: str) -> str:
    hash_val = int(md5(transaction_id.encode()).hexdigest(), 16)
    return f"shard_{hash_val % 8}"
```

The same transaction ID maps to the same shard index in every region, so `shard_0` in region A and `shard_0` in region B hold different data. A write routed to region A and a read routed to region B see different rows for the same key. During a failover, whichever region is now primary may not have the transactions that were written to the other region's copy of the same shard.

The fix is a region-aware shard key, so a given key resolves to exactly one shard across the whole deployment:

```python
# Region-aware sharding with Aurora Global Database
import os
from hashlib import md5

REGION = os.getenv("AWS_REGION", "us-east-2")

def shard_transaction(transaction_id: str) -> str:
    hash_val = int(md5(transaction_id.encode()).hexdigest(), 16)
    # Region prefix in the shard key avoids cross-region key collisions
    return f"{REGION}_shard_{hash_val % 8}"
```

This does not make cross-region writes free. Reads that cross regions pay the round-trip latency — for a us-east-2 to sa-east-1 path, on the order of 100–200ms one way, so a read-modify-write can add several hundred milliseconds. The trade-off is deliberate: correctness over latency. Pair the region-aware key with a globally distributed database (Aurora Global Database or DynamoDB Global Tables) so replication is handled by the platform rather than by application logic.

**How to measure the fix.** Track replication lag and run a failover drill. Instrument the database's replication-lag metric and alert if it exceeds your recovery-point objective. During a controlled failover, count transactions written before the switch and confirm they are readable after it. A reconciliation job that compares a checksum of transaction IDs per shard across regions, run on a short interval, will surface drift that a single-region test never sees.

## Fix 3 — timezone and regional-capacity assumptions

**Symptom:** the system works in staging but fails in production because batch jobs run during peak hours, or because a downstream service throttles below the assumed rate.

The cause is generated logic that assumes UTC and idealized capacity. A cron expression like `0 0 * * *` runs at midnight UTC. In a region at UTC-5, that is 7pm local — often inside the evening peak. A batch job that adds load during peak degrades the interactive path.

The fix is to express schedules in the local timezone and convert explicitly:

```python
# Timezone-aware batch schedule in Python 3.11
from datetime import datetime
import pytz

local_tz = pytz.timezone("America/Bogota")
local_time = local_tz.localize(datetime(2026, 4, 1, 5, 0))  # 5am local
utc_time = local_time.astimezone(pytz.UTC)                   # convert for the scheduler
```

The second half of this pattern is capacity. A generated design may assume the downstream gateway accepts 5,000 TPS when its documented limit is 1,000 TPS. The fix is a client-side limiter sized to the documented figure, with a bounded buffer for bursts:

```python
# Rate-limited payment client in Python 3.11
import asyncio
from aiohttp import ClientSession

class PaymentProcessorClient:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(1000)   # match documented TPS
        self.buffer = asyncio.Queue(maxsize=5000)  # bounded burst buffer

    async def process_payment(self, payment_data):
        async with self.semaphore:
            async with ClientSession() as session:
                async with session.post(
                    "https://api.example.com/v2/process",
                    json=payment_data,
                    timeout=5.0,
                ) as resp:
                    if resp.status == 429:
                        await self.buffer.put(payment_data)
                        return "queued"
```

Note the bounded buffer. An unbounded queue converts a throttling problem into a memory problem; a bounded queue applies backpressure to the caller, which is the correct behavior when the downstream is saturated.

**How to measure the fix.** Instrument batch-job start time against the interactive latency series. If the batch start correlates with a latency rise, the schedule is wrong. For capacity, instrument the rate limiter's queue depth and the downstream 429 rate. Queue depth should stay bounded and 429s should trend to zero once the limiter matches the documented ceiling.

## A worked example: sizing a retry storm

Suppose a downstream service documents a 1,000 TPS limit. A client pool of 2,000 workers hits it, and 10% of requests fail transiently at any moment. Without a limiter or jitter:

- Failing workers at any instant: 2,000 × 0.10 = 200.
- Each retries up to 5 times, so worst-case retry attempts per failing request: 5.
- Retry attempts per second, if all retries fire within one second: 200 × 5 = 1,000.

That is already at the ceiling before counting the original 2,000 requests per second of new traffic. The service throttles, which causes more failures, which causes more retries — a positive feedback loop.

Now add a rate limiter capped at 1,000 tokens per second and jitter of 0.5:

- In-flight requests are capped at 1,000 per second regardless of the number of workers.
- Jitter spreads retries across a window of roughly [0.5×, 1.5×] the nominal backoff, so the 1,000 retry attempts no longer arrive in a single burst.
- The feedback loop breaks because the service is never driven above its ceiling.

The numbers here are illustrative, chosen to show the mechanism. Substitute your own worker count, failure rate, and documented TPS limit; the arithmetic is the same.

## How to verify a fix under realistic conditions

Sandbox tests will not reproduce these failures. Verification needs three things: realistic traffic shape, a failover drill, and cost visibility.

**Load test with bursty traffic.** A load generator should model arrival bursts, not uniform load. A minimal Locust-style user:

```python
# locustfile.py — bursty payment traffic
from locust import HttpUser, task, between
import random

class PaymentUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task
    def process_payment(self):
        payment_data = {
            "amount": random.uniform(10, 1000),
            "currency": "COP",
            "merchant_id": f"merch_{random.randint(1, 1000)}",
        }
        self.client.post("/api/v1/payments", json=payment_data, timeout=3.0)
```

Drive it headless at a concurrency that matches your peak, and inject latency to the downstream to emulate a slow region. The point is not the exact tool; it is that the traffic shape includes bursts and the dependencies include realistic latency.

**Failover drill.** Use a fault-injection tool to force a regional failover on the database and observe whether transactions written before the switch are readable after it. Watch replication lag and the reconciliation checksum.

**Cost visibility.** Enable hourly cost granularity and watch for throttling events and over-provisioning during the load test. A design that passes latency targets by over-provisioning will show up here.

**Metrics to watch:**
- 95th-percentile latency under load, against your stated SLA.
- Error rate, against your tolerance.
- Retry rate and 429 rate, which should not spike together.
- Replication lag and post-failover consistency.
- Cost per unit of work under sustained load.

## Prevention: guardrails for AI-assisted design review

Prevention is a review process, not a better prompt alone.

**Add a chaos checklist to design review.** Before accepting an AI-generated design, verify:
1. Retry policies include jitter and a limiter sized to documented downstream capacity.
2. Failover behavior is defined, and shard keys are region-aware.
3. Schedules are expressed in local time and avoid peak hours.
4. Burst traffic at 2–3× peak is tested, not just steady-state load.

**Constrain the assistant with explicit assumptions.** Provide the region, the documented downstream limits, the timezone, and the expected burst shape as inputs. An assistant given a 1,000 TPS limit and a burst profile will produce a different design than one given nothing.

**Keep a human review step.** Assign a reviewer to check the four items above. This is where most of the value is; the assistant is fast at producing a document, and the reviewer is fast at spotting missing dynamics.

**Mirror production in staging.** Same region, same instance types, same traffic shape, same cost constraints. A local cluster will not surface regional latency, volume throttling, or cost explosions.

## Related failure patterns

Once the initial flaws are fixed, the next failures tend to appear as the system starts handling real traffic:

| Pattern | Symptom | Root cause | What to instrument | Fix |
|---|---|---|---|---|
| Cache stampede | Latency spike when many keys expire together | No lock on cache fills | Cache miss rate, DB QPS | Distributed lock around fill |
| Hot partition | Throttling errors on one key range | Uneven key distribution | Per-partition throughput | Composite key or on-demand capacity |
| Cold-start storm | Periodic latency spikes | Burst of new function instances | Function init duration, concurrency | Provisioned concurrency or snapshot restore |
| Replication lag | Inconsistency during failover | Lag exceeds RPO | Replication-lag metric | Region-aware keys, quorum reads |
| Gateway throttling | 5xx during bursts | Requests exceed configured limit | Gateway 4xx/5xx, usage-plan metrics | Usage plans and client-side limiting |
| OOM crash-loop | Pods restart under load | Unbounded memory growth | Container memory, heap profile | Memory limits and profiling |

A distributed lock around cache fills, for example, prevents the stampede described earlier:

```python
# Distributed lock around cache fill (Python 3.11)
from redis import Redis
from redlock import Redlock

redis_client = Redis(host="redis", port=6379, db=0)
lock_manager = Redlock([redis_client], retry_count=3, retry_delay=0.5)

with lock_manager.lock("cache_key", 10.0):
    data = redis_client.get("cache_key")
    if data is None:
        data = fetch_from_database("cache_key")
        redis_client.set("cache_key", data, ex=300)
```

The lock ensures only one worker fills a given key while others wait briefly, so a mass expiry does not become a mass database read.

## When the fixes do not work: escalation path

**Check for hallucinated resources.** Assistants occasionally reference services or resource types that do not exist. Verify every resource name against the provider's documentation before applying. If a name is not in the docs, it is a hallucination.

**Compare the sandbox to production.** Confirm the sandbox and production match on runtime version, storage type, and availability-zone count. A single-AZ sandbox will not surface multi-AZ constraints.

**Escalate to the vendor.** Provide the generated design, the observed error patterns, and the staging logs. Most assistants have a support channel for design issues.

**Fall back to a manual design.** If the generated design is fundamentally unsound, write it manually and validate it against the chaos checklist. This is not a failure; it is the correct outcome when the assumptions cannot be repaired.

## FAQ

**How do I stop an assistant from generating retry logic without jitter?**
Add an explicit constraint: use exponential backoff with jitter, cap the maximum backoff, and include a rate limiter matching documented downstream capacity. Then verify it in review — the constraint reduces the problem but does not eliminate it.

**What sharding strategy works for multi-region systems?**
A region-aware shard key, so a given key resolves to one shard across the whole deployment, paired with a globally distributed database that handles replication. Accept the cross-region read latency as the cost of correctness.

**Why do AI-generated cron jobs fail in production?**
Most assume UTC. Express schedules in the local timezone and convert explicitly for the scheduler, so batch work does not overlap with peak interactive traffic.

**What is the minimum staging environment to catch these flaws?**
One that mirrors production on region, instance type, traffic shape, and cost constraints. A local cluster will not surface regional latency, volume throttling, or cost behavior.

## Do this in the next 30 minutes

Open the most recent AI-generated design you have and check one thing: find every retry policy and every schedule. For each retry policy, confirm it has jitter and a limiter sized to the downstream's documented capacity; if not, fix it. For each schedule, confirm it is expressed in local time and does not overlap peak hours. Write down the documented capacity limit of your busiest downstream dependency next to the limiter that enforces it. If the two numbers do not match, you have found the next incident before it happens.
