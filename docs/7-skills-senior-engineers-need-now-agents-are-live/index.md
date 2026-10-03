# 7 skills senior engineers need now agents are live

## Why long-running agents break short-lived assumptions

A request-scoped service and a long-running agent look similar in code review and behave nothing alike in production. The service opens a connection, does work, closes the connection, and exits. The agent opens a connection, does work, and keeps the process alive for days or weeks while holding sockets, file descriptors, in-memory queues, and partial workflow state.

The classic failure mode is a socket or file descriptor leak. An agent opens one HTTP connection per user session and never closes it. Under steady traffic the process reaches the OS file descriptor limit, and every subsequent `connect()` fails with `EMFILE`. The confusing part is that CPU and memory look fine right up until the moment nothing works, because leaked descriptors are not visible in a typical CPU or heap dashboard.

This article covers seven skills that become table stakes for senior engineers once agents run continuously, how to tell whether a team actually has each skill, and where the common traps are. It is organized around failure modes rather than library recommendations, because the libraries change faster than the failure modes do.

## How to tell whether a skill is actually present

Skill checklists are easy to write and hard to verify. A more useful test is to look for observable behavior during incidents and in the codebase.

Two signals tend to separate teams that have internalized these skills from teams that have read about them:

- **Time to first meaningful action after an alert.** This is the interval between the alert firing and the on-call engineer taking an action that changes the system state, as opposed to an action that gathers more information. Instrument it by logging alert timestamps and the first state-changing action (a config change, a restart, a rollback) with a shared trace ID.
- **Whether the diagnosis references resources or only symptoms.** An engineer who says "the queue is backing up" is describing a symptom. An engineer who says "the prefetch count is 500 and the consumer is single-threaded, so the queue depth is bounded by consumer throughput" is describing a resource constraint.

To measure the first signal, add a timestamp to your alert payload, log the first mutating action in the incident channel with the same incident ID, and compute the delta. To measure the second, review incident write-ups and tag each root cause as either a resource constraint (descriptors, connections, memory, concurrency, queue depth) or an external cause (upstream outage, bad deploy, DNS). Over a few months the ratio tells you where to invest training.

## The seven skills

### 1. Async I/O patterns beyond async/await

**What it is.** Understanding backpressure, cancellation, and resource cleanup as first-class concerns rather than details the runtime handles automatically.

**Why it matters for agents.** An agent that holds thousands of concurrent connections needs to stop accepting new work when a downstream dependency slows down, and needs to release resources when a caller cancels. `async/await` gives you the syntax for concurrency but not the policy.

**Failure mode.** A coroutine awaits a downstream call with no timeout and no cancellation path. The caller times out and moves on, but the coroutine keeps running, holding its connection and its slot in the concurrency limit. Over hours, the process accumulates orphaned coroutines that consume descriptors without producing work.

**How to check.** Count in-flight coroutines or tasks and compare against your concurrency limit. If the count grows monotonically while throughput is flat, you have orphans. In Python, `asyncio.all_tasks()` gives you the live set; in Node, `process._getActiveHandles()` is a diagnostic, not a stable API, so prefer instrumenting your own task registry.

### 2. Connection pooling with precise timeouts

**What it is.** Reusing TCP connections to databases and upstream services instead of opening a new one per call, with explicit idle and lifetime bounds.

**Why it matters for agents.** Connection setup is expensive relative to many agent calls, and a long-lived process will accumulate connections unless something forces them to close.

**Failure mode.** Two opposite traps. If the pool's max size is too small, requests queue behind available connections and latency spikes under load. If idle timeout and max lifetime are unset, connections live forever and the upstream eventually closes them from its side, leaving the pool holding dead sockets that fail on first use.

**Documented behavior worth knowing.** PostgreSQL's default `idle_in_transaction_session_timeout` is 0, meaning disabled. Leaving a transaction open on a pooled connection will hold locks indefinitely unless you set that parameter or your pool enforces it. Check your server's `SHOW idle_in_transaction_session_timeout` before assuming the database will clean up after you.

**How to measure.** Instrument three numbers: pool wait time (time a request spends waiting for a connection), connection churn (connections opened per minute), and p99 query latency. Churn that scales with request rate means you are not reusing connections. Pool wait time that scales with request rate means the pool is too small.

### 3. State machine design for long-running processes

**What it is.** Modeling a workflow as explicit states and transitions, with each transition persisted, instead of as a long function with nested awaits.

**Why it matters for agents.** A function that runs for hours cannot survive a restart. A state machine can, because the current state is data, not a program counter.

**Failure mode.** The over-engineered version: a state machine framework for a workflow that has three linear steps, adding configuration and indirection without adding resumability that matters. The under-engineered version: a 500-line async function with implicit state encoded in local variables, which loses all progress on restart and cannot be tested in isolation.

**How to check.** Ask what happens if the process is killed between any two operations. If the answer is "it retries from the beginning," the workflow is not a state machine regardless of how the code is structured. If the answer is "it resumes from the last persisted transition," it is.

### 4. Distributed tracing with context propagation

**What it is.** Carrying a trace identifier and correlation metadata across service boundaries and asynchronous handoffs so a single logical request can be followed end to end.

**Why it matters for agents.** A single agent action can fan out into dozens of internal calls over minutes. Without propagation, you have logs from many services and no way to order them.

**Failure mode.** Trace context is propagated across HTTP but dropped at queue boundaries, so the trace ends at the producer and a new one starts at the consumer. The result is two disconnected traces that look unrelated in the UI. The fix is to serialize the context into message headers, which most tracing libraries support but few teams configure by default.

**Cost note.** High-cardinality traces (one trace per request, with many spans) generate significant storage and network volume. Sampling is the usual mitigation: keep all traces for errors and a percentage for successes. Decide the sampling policy before enabling tracing in production, not after the bill arrives.

### 5. Idempotency keys and deduplication

**What it is.** Attaching a unique, stable identifier to each logical operation so a retry can be recognized as a retry rather than a new operation.

**Why it matters for agents.** Agents retry. Networks fail, upstreams time out, processes restart mid-operation. Without idempotency, every retry risks a duplicate side effect.

**Failure mode.** The key is generated at the wrong layer. If the retry logic generates a fresh key on each attempt, the downstream sees two distinct operations and processes both. The key must be generated once, at the point the logical operation is created, and reused across every retry of that operation.

**How to check.** Find the code that generates the key and the code that retries. If they are in different layers and the retry layer does not receive the key from the caller, you likely have duplicate side effects under retry. A simple test: force a timeout on an operation that writes to an external system, let the retry run, and count the writes.

### 6. Async queue tuning under backpressure

**What it is.** Adjusting prefetch counts, batch sizes, visibility timeouts, and dead-letter routing so that consumers are not overwhelmed when producers burst.

**Why it matters for agents.** Agents reconnect after outages and tend to resume aggressively. A consumer that reconnects and immediately fetches its maximum prefetch can stampede a downstream service that is still recovering.

**Failure mode.** The prefetch count is set high to maximize throughput in steady state, and the same setting causes a thundering herd during recovery. The queue drains into a downstream that cannot handle the burst, the downstream errors, the messages are redelivered, and the cycle repeats.

**How to measure.** Track queue depth, consumer processing time, and downstream error rate together. If downstream errors rise as queue depth falls, you are draining too fast. Reduce prefetch and confirm that downstream error rate falls even though drain time increases.

### 7. Autoscaling hysteresis and graceful degradation

**What it is.** Deliberately asymmetric scale-out and scale-in behavior, plus a defined degraded mode for when scaling cannot keep up.

**Why it matters for agents.** Agents are often stateful and slow to start. Aggressive scale-in kills instances mid-work; aggressive scale-out multiplies load on shared dependencies like databases.

**Failure mode.** Symmetric scaling with short delays. Load rises, instances scale out, each new instance opens database connections, the database saturates, latency rises, the autoscaler interprets the latency as more load and scales out again. The system oscillates and never stabilizes.

**How to measure.** Log the timestamp and reason for every scale event alongside database connection count and p99 latency. If scale-out events correlate with database saturation rather than with request queue depth, your scaling signal is measuring the wrong thing.

## A worked example: diagnosing a descriptor leak

Consider an agent that has been running for six days and starts failing to connect to any upstream. CPU is at 15%, heap usage is flat, and the error is `EMFILE: too many open files`.

Step 1: confirm the diagnosis. On Linux, `ls /proc/<pid>/fd | wc -l` gives the current open descriptor count, and `cat /proc/<pid>/limits` shows the limit. If the count is at or near the limit, the diagnosis is confirmed.

Step 2: identify what is holding descriptors. `ls -l /proc/<pid>/fd | awk '{print $11}' | sort | uniq -c | sort -rn | head` groups open descriptors by target. If most of them point at a single upstream host and port, the leak is in the client for that upstream.

Step 3: find the code path. If the client is a pooled HTTP client, the leak is usually a response body that is never read or closed. Many HTTP clients hold the connection until the body is consumed or explicitly released, and a code path that returns early on a non-200 status without reading the body will leak one connection per call.

Step 4: fix and verify. Add explicit cleanup on every path, including error paths, and add a metric for open descriptors per upstream. Set an alert at 70% of the limit so the next leak is caught in hours rather than days.

The general lesson is that descriptor counts belong on the same dashboard as memory and CPU for any long-running process. They are cheap to collect and they fail slowly enough to be caught before an outage.

```python
import asyncio
import httpx

async def fetch_with_cleanup(client: httpx.AsyncClient, url: str) -> bytes:
    # Read the body on every path so the connection is released back to the pool.
    async with client.stream("GET", url) as response:
        if response.status_code != 200:
            # Still consume the body before returning.
            await response.aread()
            raise RuntimeError(f"unexpected status {response.status_code}")
        return await response.aread()

async def main() -> None:
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        data = await fetch_with_cleanup(client, "https://example.invalid/data")
        print(len(data))

if __name__ == "__main__":
    asyncio.run(main())
```

The two details that matter here are the explicit `limits` and `timeout` arguments and the fact that the error path still consumes the response body. Both are easy to omit and both cause the failure described above.

## Honorable mentions

**Retry budgets with jitter.** Exponential backoff without jitter synchronizes retries across clients, producing bursts at regular intervals. Adding random jitter spreads them out. A retry budget caps the total retry volume as a fraction of successful requests, which prevents retries from overwhelming a recovering service.

**Circuit breakers with persisted state.** A circuit breaker that resets on restart is useless for a process that restarts frequently. Persisting breaker state to shared storage lets a new instance inherit the open circuit instead of immediately hammering a failing dependency.

**Container resource limits matched to steady state.** Setting memory limits to peak usage means the limit never triggers and provides no protection. Setting them to steady-state usage means legitimate spikes get killed. The workable approach is to set the limit above steady state with headroom, and set a separate alert at steady state so growth is visible before it becomes fatal.

**Structured logging with correlation IDs.** The value is not the format but the correlation. Every log line emitted while handling a logical operation should carry the same identifier, so logs can be grouped without relying on timestamps.

## Approaches that look appealing and usually are not

**Isolating every coroutine in its own thread pool.** The intent is fault isolation. The effect is that a single blocking coroutine can exhaust the pool, and the context-switching overhead is paid on every call. Async runtimes already provide concurrency; adding threads on top usually adds failure modes rather than removing them.

**Retry middleware without idempotency.** Retrying on 5xx is correct only if the operation is safe to repeat. Applied to non-idempotent writes, it converts transient errors into duplicate side effects. The middleware is not wrong; deploying it without idempotency keys is.

**Distributed locks on every state transition.** Lock acquisition and renewal add latency to every transition, and the lock service becomes a dependency whose failure stalls the workflow. Locks are appropriate for genuinely contended resources, not as a default guard on all state changes.

**Dashboards with every available metric.** A dashboard that shows twenty metrics makes it harder to spot the one that changed. A small set of leading indicators, such as p99 latency, error rate, active instance count, and open descriptors, catches most incidents and is readable under pressure.

## Choosing where to start

| Situation | Skill to prioritize | First measurement to take |
|---|---|---|
| Agents calling external APIs | Async I/O and connection pooling | Open descriptors per upstream, tracked over time |
| Agents mutating external systems | Idempotency keys | Duplicate writes under forced retry |
| Agents on Kubernetes or serverless | Autoscaling hysteresis | Scale events correlated with database connection count |
| Agents consuming from queues | Queue tuning under backpressure | Downstream error rate as a function of queue depth |
| Agents with multi-step workflows | State machine design | Behavior when the process is killed mid-workflow |

If the agent has been running for more than a day and there is no descriptor metric, start there. It is the cheapest signal to collect and it catches the most common failure mode.

## FAQ

**How do I find a descriptor leak quickly?**

Compare `ls /proc/<pid>/fd | wc -l` against the limit in `/proc/<pid>/limits`. If the count is near the limit, group descriptors by target with `ls -l /proc/<pid>/fd | awk '{print $11}' | sort | uniq -c | sort -rn | head`. A single target dominating the list points at the client for that target.

**How do I size a connection pool?**

Start from the database's connection limit divided by the number of application instances, then reduce further if the database shows lock contention. The pool is not a throughput knob; it is a limit on concurrent work. Measure pool wait time and p99 query latency together, and increase the pool only if wait time is high and the database has headroom.

**What is the smallest useful idempotency implementation?**

Generate a unique key once per logical operation, send it as a header, and have the receiver store it with a TTL and reject duplicates. The key must be generated above the retry layer so every retry carries the same value.

**Why does a callback sometimes never fire under load?**

Usually because the event loop is blocked by synchronous work, or because the callback is waiting on a resource that is exhausted. Add a timeout to every callback and log when it fires. A timeout that triggers consistently under load points at a resource constraint rather than a logic bug.

## Your next 30 minutes

Pick the agent that has been running longest in your environment. Run `ls /proc/<pid>/fd | wc -l` and compare it to the limit in `/proc/<pid>/limits`. If the count is above 50% of the limit, group the descriptors by target and identify the client responsible. Add that count to your monitoring before you do anything else, so the next time it grows you will see it happening instead of discovering it during an outage.
