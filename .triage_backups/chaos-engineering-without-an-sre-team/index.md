# Chaos engineering without an SRE team

## The gap between chaos write-ups and small-team reality

Most chaos engineering content assumes a dedicated reliability team, a staging environment that mirrors production, and someone whose full-time job is running GameDays. That is not how most backend teams operate. A five-to-fifteen person team shipping a fintech product on AWS typically has a rotating on-call schedule and a chat channel where the same three people debug everything.

The gap is organizational more than technical. Vendor material on chaos engineering assumes you can afford to break things deliberately, which assumes you have the headcount to run experiments, analyze results, and fix what you find. Without that headcount, chaos engineering becomes something you read about and never do.

The counterintuitive part: teams without dedicated SREs often need this practice *more*, not less. When reliability depends on a few people's institutional memory, the failure modes nobody has seen are the ones that page at 3 AM. The practical approach is to scale the practice down to what a small team can sustain — not full GameDays, but a lightweight, continuous fault-injection habit wired into the existing CI/CD pipeline.

This article covers which experiments are worth running, how to automate them without a platform team, and where the approach breaks down. The recurring obstacle is that most chaos tooling assumes a level of observability and blast-radius control that small teams do not have.

## What chaos engineering actually is under the hood

At its core, chaos engineering is controlled fault injection with a hypothesis. You state what you expect to happen when a dependency fails, inject that failure, and observe whether reality matches. The *control* is the hard part without an SRE team: the experiment must not take down production for real customers.

The mechanism that makes this safe is blast-radius control — scoping experiments to a subset of traffic, a single availability zone, or a canary deployment. In practice, small teams achieve this with three patterns:

1. **Request-level injection** — a proxy or middleware fails a percentage of calls to a specific dependency, only for requests tagged as synthetic or coming from internal test accounts.
2. **Instance-level injection** — terminating or degrading a single compute instance or task behind a load balancer, relying on the LB to route around it.
3. **Dependency-level injection** — pointing a service at a mock or a degraded version of a downstream dependency (for example, a Redis instance with latency injected via `tc` or a sidecar proxy).

The key insight is that a chaos *platform* is not required. What is required is a way to toggle faults at runtime, scoped to a subset of traffic, with automatic rollback if error rates cross a threshold. That is achievable with feature flags, a service mesh, or a plain middleware layer.

A common mistake is treating chaos experiments as one-off events. The value comes from running them continuously — every deploy, or on a schedule — so that regressions in resilience are caught the same way regressions in functionality are caught by tests. That is the difference between a GameDay that happens once a quarter and a resilience test suite that runs on every PR.

## A minimal harness: middleware plus a measurement script

The example below builds a small chaos harness in Python using FastAPI and a feature-flag store. It assumes Python 3.11, FastAPI, and a Redis instance reachable at `localhost:6379`. The goal is to inject latency and errors into calls, scoped to requests that carry a specific header.

First, a middleware that checks a flag and applies fault injection:

```python
import random
import asyncio
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from redis.asyncio import Redis

redis = Redis(host="localhost", port=6379, decode_responses=True)

class ChaosMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Only inject faults for requests tagged as synthetic
        if request.headers.get("X-Chaos-Experiment") != "enabled":
            return await call_next(request)

        # Check if chaos is globally enabled for this service
        if not await redis.get("chaos:enabled"):
            return await call_next(request)

        # Inject latency: 30% of requests get 200-500ms added
        if random.random() < 0.3:
            delay = random.uniform(0.2, 0.5)
            await asyncio.sleep(delay)

        # Inject errors: 10% of requests fail with 503
        if random.random() < 0.1:
            raise HTTPException(status_code=503, detail="Chaos experiment: simulated failure")

        return await call_next(request)
```

This middleware is intentionally simple. It reads a Redis key to enable or disable chaos globally, and only affects requests carrying a specific header. Toggle it via an admin endpoint or a CLI command. Note that the header gate is what keeps the blast radius at zero for real users — the fault path is unreachable unless a caller opts in.

Next, a script that runs an experiment against an environment, measuring error rates and latency before, during, and after injection:

```python
import httpx
import asyncio
import statistics
from datetime import datetime

async def measure_endpoint(client, url, headers, n=100):
    latencies = []
    errors = 0
    for _ in range(n):
        start = datetime.now()
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 500:
                errors += 1
            else:
                latencies.append((datetime.now() - start).total_seconds() * 1000)
        except httpx.RequestError:
            errors += 1
    return {
        "error_rate": errors / n,
        "p50_ms": statistics.median(latencies) if latencies else 0,
        "p99_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies, default=0),
    }

async def run_experiment():
    async with httpx.AsyncClient(timeout=10.0) as client:
        headers = {"X-Chaos-Experiment": "enabled"}
        baseline = await measure_endpoint(client, "http://localhost:8000/api/health", headers)
        print(f"Baseline: {baseline}")

        # Enable chaos
        await client.post("http://localhost:8000/admin/chaos/enable")
        during = await measure_endpoint(client, "http://localhost:8000/api/health", headers)
        print(f"During chaos: {during}")

        # Disable chaos
        await client.post("http://localhost:8000/admin/chaos/disable")
        after = await measure_endpoint(client, "http://localhost:8000/api/health", headers)
        print(f"After: {after}")

        # Assert recovery
        assert after["error_rate"] < 0.01, "Service did not recover after chaos disabled"

if __name__ == "__main__":
    asyncio.run(run_experiment())
```

This script is the core of a lightweight practice. It is not a platform; it is a test. Run it in CI on every merge to main, against an environment that mirrors production. If the service does not recover, the pipeline fails.

The `n=100` sample size is a starting point, not a guarantee. With 100 requests, the standard error on an error rate near 0 is roughly 1 percentage point (sqrt(0.01 * 0.99 / 100) ≈ 0.0099), so a threshold like "error rate < 1%" is noisy at this sample size. For assertions you intend to enforce in CI, either raise `n` to a few hundred or assert on a wider margin. Similarly, `statistics.quantiles(latencies, n=100)[98]` approximates p99 only when you have well over 100 samples; below that, the fallback to `max` is what actually runs.

For infrastructure-level chaos, a managed fault-injection service (for example, AWS Fault Injection Simulator on AWS) can terminate a percentage of tasks in a service, with a stop condition that aborts the experiment if a CloudWatch alarm fires. This is more involved but still achievable without a dedicated team.

## How to measure the effect of an experiment

There is no universal benchmark for "what p99 should look like under chaos" — it depends on your service, your dependencies, and your traffic. What matters is that you measure the same things before, during, and after, and that you know how to read the numbers.

**What to instrument.** At minimum: request rate, error rate (5xx and client-visible failures), latency percentiles (p50, p95, p99), and saturation signals for the affected dependency (connection pool usage, queue depth, CPU/memory). If you have distributed tracing, capture trace IDs so you can follow a single request through the degraded path.

**What command to run.** The measurement script above is the harness. In CI, run it as a job that starts the service, applies the middleware, runs baseline → chaos → recovery, and fails the build if recovery does not happen within a stated window.

**What to compare.** Three comparisons matter:
- *During vs. baseline*: how much did latency and error rate move? If error rate stayed flat, either your fallback works or your retries are masking the fault (see failure modes below).
- *After vs. baseline*: did the service return to its prior state? A slow return usually means a pool that is not draining or a cache that is not warming.
- *During vs. your hypothesis*: did the system behave the way you predicted? A mismatch is the finding, whether it is better or worse than expected.

As an illustrative example, not a measured result: suppose a service handles 200 requests per second at peak with a p99 of 120 ms. If you inject 300 ms of latency into 30% of calls to a cache dependency, the p99 will move — but by how much depends on whether the service falls back to a database, how that fallback is pooled, and whether the extra load saturates it. The only way to know is to run the experiment and read the numbers. A fallback that was never exercised under load is a common source of surprise: the database connection pool may be sized for normal traffic, not for a 30% increase.

Recovery time is the metric most teams forget to track. After disabling chaos, how long until p99 returns to baseline? The answer tells you whether your connection pools drain and your caches warm correctly. Those are exactly the behaviors that determine whether a real incident lasts 30 seconds or three hours.

## Failure modes nobody warns you about

**Experiments that do not actually test anything.** If your service has retries with exponential backoff, injecting a 10% error rate may be completely masked by retries. You think you are testing resilience; you are testing your retry logic. Inject failures that exceed the retry budget — for example, fail 100% of calls to a dependency for 30 seconds — to force the fallback path to run.

**Observability gaps.** You cannot learn from an experiment you cannot see. Without distributed tracing, it is hard to tell which service degraded first. Before running any experiment, ensure traces, metrics, and logs are correlated by request ID.

**Blast-radius creep.** It is easy to start with a safe experiment (one instance, 1% of traffic) and gradually widen scope until production is affected. Always keep a kill switch: a flag or circuit breaker that automatically disables chaos if error rates exceed a threshold for a stated duration.

**Alerting noise.** Chaos experiments can page on-call engineers if they trigger real alerts. Either configure alerting to ignore synthetic traffic, or run experiments during business hours with a known point of contact.

**Retry amplification.** When you inject errors, clients retry. An aggressive retry policy (many retries, no backoff) can multiply load on the failing dependency, turning a small experiment into a cascading failure. Use exponential backoff with jitter and cap retries.

**Cultural drift.** Without an SRE team, chaos engineering can feel like extra work that does not ship features. It needs explicit backing from engineering leadership and a clear policy — for example, a chaos experiment is part of the definition of done for any new service. Otherwise it gets deprioritized and dies.

## Tooling categories and when to use them

| Category | Examples | Best for | Trade-offs |
|----------|----------|----------|------------|
| Managed cloud fault injection | AWS Fault Injection Simulator | Infrastructure-level chaos on cloud primitives (instances, tasks, managed databases) | Native integration, no agents to run; limited to one cloud, template setup has a learning curve |
| Kubernetes-native chaos | Chaos Mesh and similar CRD-based tools | Teams already running Kubernetes with someone to manage it | Rich experiment types; assumes Kubernetes and adds operational surface |
| Network-level proxies | Toxiproxy and similar | Local and staging fault injection without touching app code | Lightweight, easy to run locally; manual orchestration, no scheduling |
| Test-framework plugins | pytest plugins for fault injection | Python test suites that want faults inside existing tests | Integrates with existing test runs; language-specific and often early-stage |
| Commercial chaos platforms | Managed SaaS offerings | Teams that want a UI and support | Fast to start; recurring cost and vendor dependency |

For most small teams, network-level proxies plus a managed cloud fault-injection service covers the common cases. Kubernetes-native tools are powerful but assume you already run Kubernetes and have someone to manage the CRDs.

## When this approach is the wrong choice

Chaos engineering without an SRE team is not for everyone.

- **Pre-product-market-fit.** If you are still figuring out what to build, deliberate fault injection is a distraction. Talk to users instead.
- **No redundancy to exercise.** A single-instance application with a single database has no blast radius to control; any fault injection is just an outage.
- **No basic observability.** If you cannot measure error rates and latency, you cannot run experiments. Invest in metrics and tracing first.
- **Active on-call burnout.** If the team is already stretched thin, adding chaos experiments can make things worse. Reduce toil and improve runbooks first.

## Common production pitfalls and their costs

**Running chaos in production without a kill switch.** The classic mistake: a latency injection is enabled in production and does not stop when expected because the control plane is also affected. A dead man's switch — an external process that disables chaos if it does not receive a heartbeat — prevents this.

**Ignoring retry amplification.** Covered above, but worth repeating because it is the most common way a small experiment becomes a large incident.

**Never testing the fallback.** A fallback that has never been exercised is likely broken. A typical discovery: the fallback to a secondary database fails because the credentials expired months ago. Regular fault injection surfaces this before a real incident does.

**Letting experiments page on-call.** Synthetic traffic should not trigger production alerts. Tag it and filter it.

## Frequently asked questions

**How do I run chaos experiments without affecting real users?**
Scope request-level injection to synthetic traffic or internal test accounts, using a header like `X-Chaos-Experiment: enabled`. For infrastructure-level chaos, use a canary deployment or a separate environment that mirrors production. Do not inject faults into 100% of production traffic unless you have a kill switch and a tested rollback plan.

**What is the minimum observability needed?**
Three things: metrics (error rate, latency percentiles, saturation), logs with request IDs, and a way to correlate them. Distributed tracing is strongly recommended but not strictly required for simple experiments. On AWS, CloudWatch metrics plus X-Ray traces are a reasonable start; OpenTelemetry is the vendor-neutral option.

**How often should experiments run?**
For a small team, a lightweight experiment on every merge to main (in staging) is a reasonable target, with production experiments on a monthly cadence. The goal is to catch regressions early. Automate as much as possible so the overhead stays low.

**Can this be done with serverless (Lambda, Fargate)?**
Yes. Managed fault-injection services support Lambda and ECS/Fargate. For Lambda, you can inject errors by modifying environment variables or adding a layer that introduces latency. For Fargate, you can terminate tasks or inject network latency via a sidecar. The principles are the same, though the blast radius is often smaller because the platform handles more resilience for you.

## What to do in the next 30 minutes

Pick the one service that pages your team most often. Add the `ChaosMiddleware` from this article to its repository, gated behind the `X-Chaos-Experiment: enabled` header, and wire up a Redis key to toggle it. Then run the measurement script against your staging environment with a latency injection of 200 ms on 10% of calls, and record the baseline, during, and after numbers. If the service does not return to baseline within 30 seconds of disabling the injection, you have found a real weakness — and a concrete first fix.
