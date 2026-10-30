# Always-On vs On-Demand Agents: Real Cost Trade-offs

## Why the always-on versus on-demand decision matters

Agent workloads are usually bursty. A request arrives, the agent reasons, calls a model or a tool, writes a result, and goes quiet. Between bursts, nothing happens. The infrastructure choice determines whether that idle time costs money, and whether the burst path is fast or slow.

The two common shapes are:

- **Always-on**: a persistent container or VM running the agent loop, listening on a queue or HTTP endpoint. Billing accrues per second of wall-clock time, whether or not work is flowing.
- **On-demand**: a stateless function invoked per event. Billing accrues per invocation and per millisecond of execution, rounded up.

The trade-off is not "serverless is cheaper." It is a set of coupled decisions about state, latency, retry semantics, and operational surface. This article walks through the mechanics, a worked cost model, a migration path with code, the failure modes that only appear in production, and a decision checklist for when each model is wrong.

## How each model actually works

### Always-on agents

A persistent agent is a long-running process that subscribes to a work source and processes tasks in a loop. Typical implementations use a container orchestrator (ECS, Kubernetes, Nomad) or a plain VM with a supervisor.

What you get:

- **Warm state.** In-memory caches, open database connections, loaded models, and connection pools survive across tasks.
- **Predictable latency.** No boot cost on the request path. Tail latency is dominated by the work itself, not by infrastructure.
- **Simple local reasoning.** You can hold a lock, keep a counter, or maintain a session in process memory.

What you pay for:

- **Idle billing.** The container is billed for every second it exists, including the seconds it spends waiting.
- **Long-uptime failure modes.** Memory leaks, file descriptor exhaustion, connection pool drift, and slow degradation over days or weeks of uptime.
- **Restart semantics you must design.** Crash recovery, graceful shutdown, and in-flight task handling are your responsibility.

### On-demand agents

A function-based agent is invoked per event, runs to completion, and exits. The runtime is torn down or frozen between invocations.

What you get:

- **Billing tied to work.** You pay for invocation count and execution duration. Idle time is free.
- **Elastic concurrency.** The platform scales out horizontally without you provisioning capacity.
- **Crash isolation.** A failed invocation does not poison a long-lived process; the next invocation starts clean.

What you pay for:

- **Cold starts.** The first invocation after a period of inactivity pays a boot cost that varies with runtime, package size, and initialization work.
- **Externalized state.** No in-memory state survives. Caches, sessions, and locks move to Redis, DynamoDB, or another store, adding a network hop.
- **Retry semantics you must design.** Automatic retries are helpful for transient errors and dangerous for non-idempotent work.

### Billing granularity in practice

The mechanical difference is granularity. Container billing is per second with a minimum chargeable duration of one second. Function billing is per invocation plus per millisecond of execution, rounded up to the nearest millisecond. A function that runs for 187ms is billed for 187ms. A container that idles for 59 minutes and works for one minute is billed for 60 minutes.

That asymmetry is the entire cost argument. Whether it wins depends on duty cycle: the fraction of wall-clock time the workload is actually doing work.

## A worked cost model you can reproduce

Do not trust a cost comparison that does not show its assumptions. Here is a model you can fill in with your own numbers.

### Step 1: Measure the duty cycle

Instrument the always-on service to record, per task, the wall-clock time spent processing. Divide total processing time by total uptime over a representative window (a week is a reasonable start).

```
duty_cycle = total_processing_seconds / total_uptime_seconds
```

If an agent processes tasks for 40 seconds out of every 600 seconds of uptime, the duty cycle is roughly 0.067, or 6.7%. That number, not the request count, drives the comparison.

### Step 2: Price the always-on container

Container pricing is quoted per vCPU-hour and per GB-hour. The arithmetic:

```
monthly_cost = (vcpu * vcpu_price_per_hour + memory_gb * memory_price_per_hour)
             * hours_per_month
```

Using a 0.25 vCPU / 0.5 GB container and the published Fargate rates of $0.04048 per vCPU-hour and $0.004445 per GB-hour, with 730 hours in a month (illustrative arithmetic):

```
vcpu_cost   = 0.25 * 0.04048 * 730 = 7.39
memory_cost = 0.5  * 0.004445 * 730 = 1.62
monthly     = 9.01 per container
```

Four such containers cost about $36.04 per month at those rates. The exact figure depends on region and current pricing; re-derive it with the rates on your bill rather than reusing these.

### Step 3: Price the on-demand function

Function pricing is quoted per invocation and per GB-second of execution. The arithmetic:

```
gb_seconds   = memory_gb * execution_seconds
monthly_cost = invocations * (per_invocation_price + gb_seconds * per_gb_second_price)
```

With 120,000 invocations per month, 0.5 GB memory, and 0.187 seconds average execution, the workload consumes:

```
gb_seconds = 120000 * 0.5 * 0.187 = 11,220 GB-seconds
```

Multiply by the published per-GB-second rate for your region and add the per-invocation charge. The point is not the final dollar figure; it is that the function cost scales with work performed, while the container cost scales with time elapsed.

### Step 4: Add the costs that are easy to forget

A comparison that stops at compute is wrong. Add:

- **Provisioned concurrency**, if you use it. It is billed as capacity held, not work done, so it reintroduces idle cost.
- **State store traffic.** Every invocation that reads or writes Redis or DynamoDB adds latency and cost.
- **Log ingestion.** Verbose structured logging is cheap per line and expensive at volume.
- **NAT and data transfer.** Functions in a private subnet that reach the internet pay for NAT gateway hours and per-GB processing.
- **Retry amplification.** Every retry is a billed invocation. A job with a 10% retry rate costs 10% more than its success count suggests.

### Step 5: Find the crossover

The crossover is the duty cycle at which container cost equals function cost for the same workload. Below it, on-demand wins on compute. Above it, always-on wins. Compute it once with your own rates and revisit it when either your traffic shape or your provider's pricing changes.

## Migrating agent logic to an on-demand model

The migration is mostly about removing assumptions that only hold in a persistent process.

### Step 1: Reduce the entry point to a pure handler

A persistent agent often carries an HTTP framework it does not need. Strip the handler down to the event-processing logic.

```python
# agent_lambda.py
import json
import os
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.utilities.data_classes import SQSEvent

logger = Logger()
tracer = Tracer()


@tracer.capture_lambda_handler
@logger.inject_lambda_context(log_event=True)
def lambda_handler(event: SQSEvent, context: LambdaContext) -> None:
    for record in event.records:
        task_id = None
        try:
            payload = json.loads(record.body)
            task_id = payload.get("task_id")
            logger.info("Processing task", extra={"task_id": task_id})

            process_task(payload)

            logger.info("Task completed", extra={"task_id": task_id})
        except Exception:
            logger.exception("Task failed", extra={"task_id": task_id})
            raise


def process_task(payload: dict) -> dict:
    # Replace with the actual agent logic.
    return {"status": "completed", "output": "success"}
```

Two things to note. First, `SQSEvent` from the Powertools data classes iterates records safely; do not assume a single message per invocation, because batch size is configurable. Second, re-raising the exception is deliberate: it tells the platform the batch failed so the retry policy applies. Swallowing the error silently is how messages disappear.

### Step 2: Externalize state

Any in-memory dictionary, cache, or lock must move to a shared store. Redis is a common choice for low-latency key-value state.

```python
import json
import os
import redis

redis_client = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=int(os.environ.get("REDIS_PORT", "6379")),
    socket_timeout=1.0,
    socket_connect_timeout=1.0,
)

def mark_processing(task_id: str) -> None:
    redis_client.set(f"task:{task_id}", json.dumps({"status": "processing"}))

def get_state(task_id: str) -> dict | None:
    raw = redis_client.get(f"task:{task_id}")
    return json.loads(raw) if raw else None
```

Set explicit socket timeouts. A function that blocks on a hung connection burns billed time and holds concurrency slots that other invocations need.

### Step 3: Make the work idempotent

Automatic retries are the norm in event-driven systems. If a retry can charge a card twice, send a duplicate email, or double-append to a ledger, the design is broken. Idempotency keys are the standard fix.

```python
import hashlib
import json

def idempotency_key(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def process_once(payload: dict, store, handler) -> dict:
    key = idempotency_key(payload)
    existing = store.get(f"idem:{key}")
    if existing is not None:
        return json.loads(existing)

    result = handler(payload)
    store.set(f"idem:{key}", json.dumps(result), ex=86400)
    return result
```

The expiry is a policy decision: long enough to cover the retry window, short enough to bound storage growth.

### Step 4: Decide whether to pay for warmth

If cold-start latency breaks your latency budget, provisioned concurrency keeps a fixed number of execution environments warm. It is billed as capacity held, so it converts a variable cost into a fixed one. Treat it as a latency purchase, not a cost optimization, and measure the latency it actually buys before committing.

### Step 5: Instrument before you tune

Emit metrics for invocation count, error count, duration, cold starts, and retry count. Alarm on error rate and on duration at the tail, not the average. Cost anomalies almost always show up first as a change in invocation or duration distribution.

## Failure modes that appear only in production

### Retry amplification on non-idempotent work

The most expensive failure mode is a retry loop over a side effect. A batch fails, the platform retries it, the side effect runs again, and the bill and the damage both grow. Idempotency keys plus a dead-letter queue are the standard mitigation.

### Poison-pill batches

When a queue delivers messages in batches, one unprocessable message can cause the whole batch to be redelivered repeatedly. Without a dead-letter queue and a bounded receive count, a single malformed payload can consume unbounded compute. Configure a maximum receive count and route exhausted messages to a DLQ.

### Cold-start cost hiding in the duration metric

A cold start is not just latency; it is billed execution time spent on initialization. If initialization dominates the handler duration, the cost model inverts. Reduce package size, lazy-load heavy dependencies, and move shared libraries into layers.

### Concurrency ceilings

Managed function platforms impose a default regional concurrency limit. When the limit is reached, invocations are throttled and events queue up. If the queue's visibility timeout is shorter than the time to drain, messages can be processed twice. Set the visibility timeout to at least the function timeout plus a margin, and reserve concurrency for critical functions so a noisy neighbor cannot starve them.

### Log retention as a silent cost

Default log retention is often indefinite or short depending on configuration. Unbounded retention accumulates storage cost; too-short retention destroys the evidence needed to debug an incident. Set an explicit retention period per log group and stream to a central store only what you will actually query.

### VPC and NAT charges

A function inside a VPC that needs outbound internet access routes through a NAT gateway, which is billed hourly plus per GB processed. This is a common surprise when a function that previously ran with public networking is moved behind a private subnet.

### Deployment package limits

Function deployment packages have size limits, and the unzipped limit is the one that bites when a data-processing dependency is bundled in. Layers, container-image packaging, or moving the heavy work to a separate service are the usual escapes.

## Decision checklist

Use this to pick a model before writing infrastructure code.

| Question | If yes | If no |
|---|---|---|
| Does a single task run longer than the platform's function timeout? | Always-on | Either |
| Does the agent need in-process state between tasks? | Always-on (or externalize state) | Either |
| Is the duty cycle high (the process is busy most of the time)? | Always-on | On-demand |
| Is p99 latency budget under the cold-start cost? | Always-on or provisioned concurrency | On-demand |
| Are all side effects idempotent or keyed? | On-demand | Fix idempotency first |
| Is the workload spiky with long idle periods? | On-demand | Either |
| Can the team operate a state store and DLQ? | On-demand | Always-on |

The honest summary: on-demand wins when the duty cycle is low and the work is idempotent. Always-on wins when the work is long-running, stateful, or latency-critical. Most real systems end up hybrid, with latency-critical paths on persistent infrastructure and bursty background work on functions.

## What to measure before you commit

Do not choose based on a blog post, including this one. Instrument the current system and let the data decide.

1. **Duty cycle.** Total processing seconds divided by total uptime over one week. This is the single most predictive number.
2. **Latency distribution.** p50, p95, and p99, not the mean. Cold starts live in the tail.
3. **Retry rate.** Failed invocations divided by total invocations, per job type. Multiply by cost to see the amplification.
4. **Idempotency coverage.** The fraction of side-effecting operations that are keyed. Anything below 100% is a migration blocker.
5. **State access pattern.** Reads and writes per task, and the latency of the backing store. This becomes the new floor on per-task latency.
6. **Cost per successful task.** Total infrastructure cost divided by successfully completed tasks. This is the only cost metric that accounts for retries and failures.

Run both models against a shadow copy of production traffic if the workload permits. Compare cost per successful task and p99 latency, not headline monthly cost.

## Take action in the next 30 minutes

Open your monitoring dashboard and compute the duty cycle of your busiest always-on agent: divide total processing seconds by total uptime over the last seven days. If that number is below 0.2 and every side effect in the agent is idempotent, you have a concrete candidate for migration. If it is above 0.5, stop considering the migration and spend the time on the failure modes above instead.
