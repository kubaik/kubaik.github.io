# Chaos engineering without an SRE team

measure control is the kind of decision that looks reversible until it isn't. Most write-ups stop exactly where the interesting part starts. Here's what changed once we stopped guessing and started measuring.

## The gap between what the docs say and what production needs

Most chaos engineering content assumes you have a dedicated reliability team, a staging environment that mirrors production, and someone whose full-time job is running GameDays. That's a fantasy for the vast majority of engineering teams. If you're a five-to-fifteen person backend team shipping a fintech product on AWS, you don't have an SRE function — you have a rotating on-call schedule and a Slack channel where the same three people debug everything.

The gap isn't tooling. It's organizational. Chaos engineering as described by the big cloud vendors assumes you can afford to break things deliberately, which assumes you have the headcount to run experiments, analyze results, and fix what you find. Without that, chaos engineering becomes a thing you read about and never do.

But here's the counterintuitive part: teams without dedicated SREs need chaos engineering *more*, not less. When your reliability depends on three people's institutional memory, the failure modes you haven't seen are the ones that will page you at 3 AM. The trick is scaling the practice down to what a small team can actually sustain — not running full GameDays, but building a lightweight, continuous fault-injection habit into your existing CI/CD pipeline.

This post covers how to do that: which experiments are worth running, how to automate them without a platform team, and where the approach breaks down. The part that trips people up is that most chaos tooling assumes a level of observability and blast-radius control that small teams don't have — and that's what this post actually covers.

## How Chaos engineering practices that fit a team without a dedicated SRE function actually works under the hood

At its core, chaos engineering is controlled fault injection with a hypothesis. You state what you expect to happen when a dependency fails, inject that failure, and observe whether reality matches. The control is the hard part without an SRE team — you need to ensure the experiment doesn't take down production for real customers.

The mechanism that makes this safe is blast-radius control: scoping experiments to a subset of traffic, a single availability zone, or a canary deployment. In practice, small teams achieve this with three patterns:

1. **Request-level injection** — using a proxy or middleware to fail a percentage of calls to a specific dependency, only for requests tagged as synthetic or coming from internal test accounts.
2. **Instance-level injection** — terminating or degrading a single EC2 instance or ECS task behind a load balancer, relying on the LB to route around it.
3. **Dependency-level injection** — pointing a service at a mock or a degraded version of a downstream dependency (e.g., a Redis instance with latency injected via `tc` or a sidecar proxy).

The key insight is that you don't need a chaos platform. You need a way to toggle faults at runtime, scoped to a subset of traffic, with automatic rollback if error rates cross a threshold. That's achievable with feature flags, a service mesh, or even a simple middleware layer.

What most teams get wrong is treating chaos experiments as one-off events. The value comes from running them continuously — every deploy, or on a schedule — so that regressions in resilience are caught the same way regressions in functionality are caught by tests. This is the difference between a GameDay that happens once a quarter and a resilience test suite that runs on every PR.

## Step-by-step implementation with real code

Let's build a minimal chaos harness in Python using FastAPI and a feature-flag service. We'll assume you're running Python 3.11, FastAPI 0.110, and Redis 7.2. The goal is to inject latency and errors into calls to a downstream service, scoped to requests that carry a specific header.

First, a middleware that checks a feature flag and applies fault injection:

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

This middleware is intentionally simple. It reads a Redis key to enable/disable chaos globally, and only affects requests with a specific header. You can toggle it via a simple admin endpoint or a CLI command.

Next, a script that runs a chaos experiment against a staging environment, measuring error rates and latency before, during, and after injection:

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

This script is the core of your chaos practice. It's not a platform; it's a test. You can run it in CI on every merge to main, against a staging environment that mirrors production. If the service doesn't recover, the pipeline fails.

For infrastructure-level chaos, you can use AWS Fault Injection Simulator (FIS) with a simple experiment template that terminates a percentage of ECS tasks. A typical template might target 10% of tasks in a service, with a stop condition that aborts if CloudWatch alarms fire. This is more involved but still achievable without a dedicated team.

## Performance numbers from a live system

To make this concrete, consider a typical fintech API running on ECS Fargate with a PostgreSQL 15 backend and Redis 7.2 for caching. The service handles 200 requests per second at peak, with a p99 latency of 120ms and a 0.2% error rate under normal conditions.

When you inject 300ms of latency into 30% of calls to the Redis cache, you'd expect the p99 to rise. In a well-instrumented system, the p99 might jump to 450ms, and the error rate might stay flat if the service has a fallback to the database. If it doesn't, the error rate could spike to 5% as timeouts cascade.

A common failure mode here is that the fallback path is untested. Teams often add a circuit breaker (e.g., using `pybreaker` or `resilience4j`) but never verify that the fallback actually works under load. Injecting faults reveals that the fallback is slower than expected or that it introduces a new bottleneck — say, the database connection pool is sized for normal traffic, not for a 30% increase.

Another number worth tracking is recovery time. After disabling chaos, how long until p99 returns to baseline? In a healthy system, it's under 30 seconds. If it takes minutes, you have a connection pool that isn't draining, or a cache that isn't warming correctly. These are the kinds of issues that cause multi-hour outages during real incidents.

Cost-wise, running chaos experiments in staging adds minimal overhead — maybe 5-10% more compute during the experiment window. The real cost is engineering time: expect to spend 2-4 hours per month maintaining the harness and reviewing results. That's a fraction of what a dedicated SRE would cost, and it's within reach for most teams.

## The failure modes nobody warns you about

The first failure mode is **chaos experiments that don't actually test anything**. If your service has retries with exponential backoff, injecting a 10% error rate might be completely masked by retries. You think you're testing resilience, but you're just testing your retry logic. To avoid this, you need to inject failures that exceed your retry budget — e.g., fail 100% of calls to a dependency for 30 seconds.

The second is **observability gaps**. You can't learn from an experiment if you can't see what happened. Many teams run chaos without distributed tracing (e.g., AWS X-Ray or OpenTelemetry), so they can't tell which service degraded first. Before running any experiment, ensure you have traces, metrics, and logs correlated by request ID.

The third is **blast radius creep**. It's easy to start with a safe experiment (one instance, 1% of traffic) and gradually increase scope until you accidentally take down production. Always use a kill switch — a feature flag or a circuit breaker that automatically disables chaos if error rates exceed a threshold (e.g., 5% for 1 minute).

The fourth is **cultural**. Without an SRE team, chaos engineering can feel like extra work that doesn't directly ship features. You need buy-in from engineering leadership and a clear policy: chaos experiments are part of the definition of done for any new service. Otherwise, it gets deprioritized and dies.

## Tools and libraries worth your time

| Tool | Version | Use case | Pros | Cons |
|------|---------|----------|------|------|
| AWS Fault Injection Simulator | N/A (managed) | Infrastructure-level chaos (EC2, ECS, RDS) | Native AWS integration, no agents | Limited to AWS, can be complex to set up |
| Chaos Mesh | 2.6 | Kubernetes-native chaos | Rich experiment types, CRDs | Requires Kubernetes, overkill for small teams |
| Gremlin | SaaS | Full-platform chaos | Easy to use, good UI | Expensive, vendor lock-in |
| Toxiproxy | 2.7 | Network-level fault injection | Lightweight, easy to run locally | Manual setup, no orchestration |
| pytest-chaos | 0.3 | Python test-level chaos | Integrates with pytest | Limited to Python, early stage |

For most small teams, I'd start with Toxiproxy for local and staging experiments, and AWS FIS for production-like infrastructure tests. Chaos Mesh is powerful but assumes you're already running Kubernetes and have someone to manage it.

## When this approach is the wrong choice

Chaos engineering without an SRE team is not for everyone. If your service is pre-product-market-fit and you're still figuring out what to build, deliberate fault injection is a distraction. Your time is better spent talking to users. Similarly, if you're running a monolithic app with no redundancy — a single EC2 instance and a single database — there's no blast radius to control; any fault injection is just an outage.

It's also the wrong choice if you don't have basic observability. If you can't measure error rates and latency, you can't run experiments. Invest in metrics and tracing first.

Finally, if your team is already stretched thin and on-call burnout is a real risk, adding chaos experiments might make things worse. In that case, focus on reducing toil and improving runbooks before introducing deliberate failures.

## Common production pitfalls and what they cost

**Pitfall 1: Running chaos in production without a kill switch.** This is the classic mistake. A team enables a latency injection experiment in production, and the injection doesn't stop when expected because the control plane is also affected. Result: a 45-minute outage and a postmortem that could have been avoided with a simple timeout on the experiment. Always have a dead man's switch.

**Pitfall 2: Ignoring the cost of retries.** When you inject errors, your clients retry. If your retry policy is aggressive (e.g., 5 retries with no backoff), you can amplify load on the failing dependency by 5x, turning a small experiment into a cascading failure. Use exponential backoff and jitter, and cap retries.

**Pitfall 3: Not testing the fallback.** As mentioned, a fallback that's never exercised is likely broken. A typical cost: you discover during a real incident that your fallback to a secondary database doesn't work because the credentials expired six months ago. Chaos experiments force you to test these paths regularly.

**Pitfall 4: Overlooking the human element.** Chaos experiments can page on-call engineers if they trigger real alerts. Make sure your alerting is configured to ignore synthetic traffic, or run experiments during business hours with a known point of contact.

## Frequently Asked Questions

**How do I run chaos experiments without affecting real users?**
Use request-level injection scoped to synthetic traffic or internal test accounts. Tag requests with a header like `X-Chaos-Experiment: enabled` and have your middleware only apply faults to those requests. For infrastructure-level chaos, use a canary deployment or a separate staging environment that mirrors production. Never inject faults into 100% of production traffic unless you have a kill switch and a rollback plan.

**What's the minimum observability I need for chaos engineering?**
You need at least three things: metrics (error rate, latency percentiles, saturation), logs with request IDs, and a way to correlate them. Distributed tracing is highly recommended but not strictly required for simple experiments. If you're on AWS, CloudWatch metrics and X-Ray traces are a good start. OpenTelemetry is the vendor-neutral option.

**How often should I run chaos experiments?**
For a small team, running a lightweight experiment on every merge to main (in staging) is ideal. For production experiments, once a month is a reasonable cadence. The goal is to catch regressions early, so the more frequent the better — as long as you can handle the overhead. Automate as much as possible.

**Can I do chaos engineering with serverless (Lambda, Fargate)?**
Yes. AWS Fault Injection Simulator supports Lambda and ECS/Fargate. For Lambda, you can inject errors by modifying the function's environment variables or using a layer that adds latency. For Fargate, you can terminate tasks or inject network latency via a sidecar. The principles are the same, but the blast radius is often smaller because serverless platforms handle more of the resilience for you.

## What to do next

Pick one service — the one that pages you most often — and write a single chaos experiment for it. Start with a latency injection: add 200ms to 10% of calls to its primary dependency, scoped to requests with a `X-Chaos-Experiment: enabled` header. Run it in staging, measure the p99 and error rate, and verify that the service recovers within 30 seconds after you disable the injection. If it doesn't, you've found a real weakness worth fixing. Do this today: open your service's repository, add the middleware from the code example above, and run the experiment script. That's the first step.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
