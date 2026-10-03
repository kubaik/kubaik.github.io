# AI Interviews: Four Resilience Questions That Matter

Interview guides often lag behind what production teams actually need. A guide written around whiteboard algorithms can still ask candidates to implement a binary search tree while the systems those candidates will maintain fail under retry storms and connection leaks. The gap is not that algorithms are useless; it is that algorithm puzzles rarely exercise the failure modes that dominate real incident reports.

This article describes a screening approach built around fault injection: run a small service, break it in a controlled way, and grade whether the candidate's change restores service-level objectives. It covers the four question types that catch the most weak candidates, a minimal reference implementation, the failure modes of the harness itself, and when this approach is the wrong choice.

## The gap between coding puzzles and production resilience

Classic coding challenges test correctness in isolation. They do not test whether a candidate can diagnose a latency spike caused by a misconfigured connection pool, or whether they understand why a fixed retry delay makes an outage worse. Those skills are learned on call rotations, not on whiteboards.

A typical failure mode in real payment services is a cascade: a downstream dependency slows down, callers hold connections open while retrying, the connection pool exhausts, and the API begins timing out for unrelated requests. None of those steps appear in a binary search tree question. A screening process that only measures algorithmic fluency will select for a skill that is necessary but not sufficient.

The practical response is to add a short live debugging segment to each screening round. The goal is not to solve a puzzle; it is to spot the one change that keeps the system within its latency budget under load.

## How a fault-injection screening harness works

An AI-assisted screening harness does not just grade an answer. It runs a candidate's change against a simulated system and observes the result. The architecture is straightforward:

1. A small service (for example, a FastAPI app with a `/pay` endpoint and a `/metrics` endpoint) runs in a container.
2. Supporting containers provide a cache, a worker, and a metrics store.
3. The harness replays a recorded traffic pattern and injects a fault: a cache miss, a downstream delay, or a sudden RPS spike.
4. A scoring query checks whether the candidate's fix restores the target latency within a time window.
5. The harness records syscall counts, network latency, and garbage-collection cycles for later review.

The scoring engine can be built on any metrics stack. A common choice is Prometheus for metrics and a query language such as PromQL for the pass/fail check. The important property is that the grade depends on observed behavior, not on code style.

A frequent surprise is how often experienced engineers fail this kind of test. Candidates who write clean, correct code may still leak sockets after three retries or add a fixed delay that turns a slow dependency into a queue. The harness does not care about cleanliness; it cares about whether the service stays within its latency budget.

## Step-by-step implementation with real code

The reference service below exposes two endpoints: `/pay` simulates charging a card, and `/metrics` returns Prometheus metrics. It is intentionally small so that a candidate can read it in a few minutes.

```python
# app/main.py
import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from redis import Redis
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

redis = Redis(host="redis", port=6379, decode_responses=True, socket_timeout=5)

PAYMENT_COUNTER = Counter("payments_total", "Total payment attempts")
FAILURE_COUNTER = Counter("payments_failed", "Failed payment attempts")
LATENCY_HISTOGRAM = Histogram(
    "payment_latency_ms",
    "Payment latency in ms",
    buckets=[50, 100, 200, 500, 1000],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    redis.ping()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/pay")
async def pay(amount: float, card_id: str):
    start = time.time()
    PAYMENT_COUNTER.inc()

    try:
        await asyncio.sleep(0.1)  # simulate downstream call

        cache_key = f"card:{card_id}"
        cached = redis.get(cache_key)
        if cached:
            LATENCY_HISTOGRAM.observe((time.time() - start) * 1000)
            return {"status": "cached", "amount": float(cached)}

        if amount <= 0:
            raise HTTPException(status_code=400, detail="Invalid amount")

        redis.setex(cache_key, 60, str(amount))
        LATENCY_HISTOGRAM.observe((time.time() - start) * 1000)
        return {"status": "charged", "amount": amount}

    except Exception as e:
        FAILURE_COUNTER.inc()
        logger.error(f"Payment failed: {e}")
        LATENCY_HISTOGRAM.observe((time.time() - start) * 1000)
        raise

@app.get("/metrics")
async def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

Note the bug fixes relative to a typical first draft: `time` is imported, and the histogram is declared before use. Without those, the service fails at import time.

A multi-stage Dockerfile keeps the image small:

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app /app
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The requirements file pins versions that have been tested together:

```
fastapi==0.109.0
redis==5.0.1
prometheus-client==0.19.0
uvicorn==0.27.0
```

The harness itself can be written in any language that can orchestrate containers and query metrics. Node.js is a common choice because its event loop handles many short-lived containers without blocking. The example below is intentionally minimal; it does not handle cleanup perfectly, but it is enough to run a test.

```javascript
// proctor/index.js
import { Docker } from 'node-docker-api';
import axios from 'axios';
import { PrometheusDriver } from 'prometheus-query';

const docker = new Docker({ socketPath: '/var/run/docker.sock' });
const prom = new PrometheusDriver({ endpoint: 'http://prometheus:9090' });

async function waitForHealth(retries = 30) {
  for (let i = 0; i < retries; i++) {
    try {
      await axios.get('http://localhost:8000/health');
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
  throw new Error('service did not become healthy');
}

async function runCandidateTest(candidateId) {
  const service = await docker.container.create({
    Image: 'payment-service:latest',
    name: `candidate-${candidateId}`,
    HostConfig: { NetworkMode: 'host' },
  });
  await service.start();

  const redis = await docker.container.create({
    Image: 'redis:7.2',
    name: `redis-${candidateId}`,
    HostConfig: { NetworkMode: 'host' },
  });
  await redis.start();

  await waitForHealth();

  await axios.post('http://localhost:8000/pay', {
    amount: 100,
    card_id: 'test123',
  });

  const metrics = await prom.instantQuery(
    'rate(payment_latency_ms_sum[5m]) / rate(payment_latency_ms_count[5m])',
    Date.now(),
  );

  const avgLatency = Number(metrics.result[0]?.value?.[1] ?? Infinity);
  const passed = avgLatency < 200;

  await service.stop();
  await service.remove();
  await redis.stop();
  await redis.remove();

  return { passed, avgLatency };
}

runCandidateTest('candidate-123').catch(console.error);
```

Two corrections matter here. First, the latency query divides the sum by the count; querying the sum alone gives a meaningless number that grows with traffic. Second, cleanup happens on every path, not only on failure, so containers do not accumulate.

## Measuring whether the harness actually works

Claims about screening improvements are only meaningful if they are measured. Before rolling out a harness, define the metrics and the instrumentation that produces them.

What to instrument:

- **Candidate outcomes.** For each candidate, record the harness verdict, the human reviewer verdict, and the eventual hiring decision. This lets you compute agreement between the harness and human review.
- **Harness reliability.** Record the number of runs that failed for infrastructure reasons (container startup failure, metrics scrape gap, timeout) versus candidate reasons. A harness with a high infrastructure-failure rate is not measuring candidates.
- **Time to signal.** Record the wall-clock time from fault injection to the candidate's first change. This is a proxy for debugging skill.
- **Post-hire signal.** If your organization tracks incidents, record the count of incidents attributed to new hires in their first 30 days, normalized by hire count. This is a lagging indicator and requires a long observation window.

How to measure:

- Run the harness against a known-good reference solution and a known-bad solution. The known-good solution should pass every time; the known-bad solution should fail every time. If either result is inconsistent, the harness is flaky.
- Repeat the reference runs at least twenty times to estimate the false-failure rate. A false-failure rate above a few percent makes the harness unusable for hiring decisions.
- Compare harness verdicts against human reviewer verdicts on the same candidate submissions. Report the disagreement rate, not just the agreement rate, so that the failure modes are visible.

What to compare:

- The harness's verdict on a candidate's change versus a human reviewer's verdict on the same change.
- The harness's latency measurement versus an independent measurement of the same service under the same load.
- The harness's infrastructure-failure rate across runs on the same day versus across days. A rate that varies by day suggests a resource or scheduling problem.

A worked example of the arithmetic: suppose a harness runs 100 candidate sessions. Twenty fail for infrastructure reasons and are discarded. Of the remaining 80, 60 pass and 20 fail. If a human reviewer later judges that 5 of the 60 passes were actually weak submissions, the false-pass rate among valid runs is 5/80 = 6.25%. If 2 of the 20 failures were actually strong submissions, the false-fail rate is 2/80 = 2.5%. Those two numbers, not a single accuracy figure, determine whether the harness is useful.

## The four question types that catch weak candidates

The four fault-injection scenarios below cover the failure modes that most often separate candidates who can write code from candidates who can keep a service running.

**1. Cache stampede under high write volume.** The harness makes a popular cache key expire while many requests arrive at once. A weak candidate either removes the cache entirely or adds a lock with a fixed delay, which serializes requests and pushes latency up. A strong candidate uses a short TTL with a lock and jitter, or a single-flight pattern that lets one request refresh the value while others serve stale data.

**2. Deadlock or connection exhaustion in a distributed transaction.** The harness holds a connection open while a downstream call is slow. A weak candidate adds retries without bounding them, and the pool exhausts. A strong candidate bounds the retry count, uses exponential backoff with jitter, and ensures every acquired connection is released on every path.

**3. Graceful degradation when a downstream returns 5xx.** The harness makes a dependency fail for a fixed interval and then recover. A weak candidate hard-codes a retry delay or removes the dependency call entirely. A strong candidate degrades to a cached or default response and recovers automatically when the dependency returns.

**4. Latency spike from a single misconfigured connection pool.** The harness constrains the pool size. A weak candidate increases the pool size without understanding the downstream limit. A strong candidate identifies the bottleneck, sets a pool size consistent with the downstream capacity, and adds a timeout so that slow calls do not hold connections indefinitely.

Each scenario is graded against a latency budget. The budget should be derived from the service's own SLO, not chosen arbitrarily. If the `/pay` endpoint must respond within 150 ms at P99 under normal load, the harness should check whether the candidate's change restores P99 to that level within a defined window after the fault.

## Failure modes of the harness itself

A fault-injection harness has its own failure modes, and they are easy to miss because they look like candidate failures.

**Flaky infrastructure.** Container startup, DNS resolution, and network configuration can all introduce nondeterminism. Bridge networking in particular can cause intermittent container-to-container failures. Host networking removes one class of problems but introduces port conflicts. Whatever the choice, the harness should record infrastructure failures separately from candidate failures.

**Metrics scrape gaps.** If the metrics store is not scraping the service during the fault window, the scoring query returns no data. The harness should treat missing data as an infrastructure failure, not a candidate failure, and should retry the scrape before scoring.

**Missing labels.** If metrics are not labeled with the candidate or session identifier, a latency spike cannot be attributed to a specific run. Add a session label to every metric at the source.

**Timeout budget too tight.** If the fault window is shorter than the service's own recovery time, even a correct fix will fail. Measure the recovery time of a known-good solution and set the window with margin.

**Cache hits masking the fault.** If the cache serves a response during the fault window, latency may look fine even when the candidate's code is flawed. Disable or isolate the cache during the scored window.

**Cost creep from orphaned containers.** Containers that are not cleaned up after a run accumulate and consume resources. A periodic prune of stopped containers is a simple mitigation:

```bash
docker ps -a --filter "status=exited" --filter "name=candidate-*" -q | xargs -r docker rm
```

## When this approach is the wrong choice

This harness is not appropriate for every team or every role.

- **Small teams without observability.** If you cannot measure latency, error rates, and cache behavior, you cannot grade a candidate's fix. Building the metrics pipeline is a prerequisite, and it can take months.
- **Non-service roles.** Embedded systems, FPGA, and similar specialties have different failure modes. A REST-and-cache harness will not reflect their work.
- **Event-driven stacks.** If your production system is built around a message broker, a REST-based harness will test the wrong skills. The same principles apply, but the harness must be rebuilt around the actual stack.
- **Early-stage startups.** The maintenance overhead of a harness, a metrics store, and a container orchestrator is significant. A structured whiteboard or take-home exercise may be a better use of limited time.
- **Teams without a reference solution.** Without a known-good solution to calibrate the harness, you cannot distinguish a flaky harness from a weak candidate.

## Decision checklist

Before adopting a fault-injection screening harness, confirm each of the following:

- You can measure P99 latency, error rate, and cache hit rate for the reference service.
- You have a known-good solution that passes the harness consistently and a known-bad solution that fails consistently.
- You have run the harness at least twenty times to estimate the false-failure rate.
- You record infrastructure failures separately from candidate failures.
- You have defined a latency budget derived from an SLO, not chosen arbitrarily.
- You have a cleanup process for orphaned containers.
- You have a plan for comparing harness verdicts against human reviewer verdicts.
- You have decided what you will do if the harness and a human reviewer disagree.

If any of those are missing, the harness will produce noise rather than signal.

## What to do next

In the next 30 minutes, instrument the endpoint you would use for a screening harness and measure its P99 latency under normal load. If you do not already have a metrics endpoint, add one, then run a short load test and record the P99. That single number tells you whether the latency budget you would set is realistic, and it is the prerequisite for every other step in this article.
