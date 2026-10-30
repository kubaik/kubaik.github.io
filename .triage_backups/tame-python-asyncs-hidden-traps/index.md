# Tame Python async’s hidden traps

## The mismatch that causes most async surprises

Python's async story rests on a genuine architectural mismatch that is easy to miss. `async def` and `await` are language-level constructs, but on CPython the interpreter still executes bytecode under the global interpreter lock (GIL). Async gives you concurrency — the ability to make progress on many I/O operations at once — but it does not give you parallel execution of Python bytecode. Two coroutines never run CPU-bound Python at the same instant.

That distinction explains a common failure mode. A service is rewritten from a threaded WSGI stack to an ASGI framework, the request-per-second number drops, and CPU sits pinned at 100% on every core. Nothing is "broken." The workload was CPU-heavy, so the event loop had nothing to yield to, and the async rewrite added scheduling overhead without removing the real bottleneck.

The decision that actually matters is not "async or sync." It is which of two concurrency models you use for the blocking parts of your request path:

- **Threaded async**: an async framework that offloads blocking calls to a thread pool.
- **Task-based async**: an async framework where every I/O dependency is async-native, so no threads are involved.

Both are legitimate. They fail in different ways, cost different amounts to operate, and suit different teams. The rest of this article covers how each works, how to measure the tradeoff on your own workload, and how to choose.

## Threaded async: how it works and where it shines

In threaded async, the framework still schedules coroutines on an event loop, but when a request handler needs to call synchronous code, that callable is dispatched to a thread from a pool. The event loop continues serving other coroutines while the thread blocks.

A minimal example using Starlette's `run_in_threadpool`:

```python
import time
import requests
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

app = FastAPI()

def fetch_sync(url: str) -> dict:
    # requests is blocking; this runs in a worker thread
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()

@app.get("/proxy")
async def proxy(url: str):
    data = await run_in_threadpool(fetch_sync, url)
    return {"ok": True, "keys": list(data)[:5]}
```

Where threaded async is a good fit:

- **Mixed workloads.** Synchronous libraries such as `pandas`, `numpy`, `boto3`, and most cloud SDKs can be called without rewriting them.
- **Incremental adoption.** You can move one endpoint at a time onto an ASGI framework while the rest of the codebase stays synchronous.
- **Ecosystem stability.** You are not blocked waiting for an async client to exist for a given service.

The hidden cost is that threads share memory and are still subject to the GIL. If the work you offload is CPU-bound, the threads serialize against each other and against the event loop's own Python execution. Threaded async is a good fit when the offloaded work is dominated by waiting — database round trips, HTTP calls, file reads — and a poor fit when it is dominated by computation.

A second, subtler cost is pool exhaustion. If a downstream dependency slows down, blocked threads accumulate. Once the pool is full, further offloads queue, and latency rises for every endpoint that shares the pool — including endpoints that have nothing to do with the slow dependency. This is a classic "mystery latency" incident: CPU looks normal, memory looks normal, and the service intermittently stalls.

## Task-based async: how it works and where it shines

In task-based async, every I/O operation is expressed as an awaitable. The event loop uses an OS-level I/O multiplexer (`epoll` on Linux, `kqueue` on macOS, `io_uring` on newer kernels) and cooperative multitasking. When a coroutine awaits I/O, control returns to the loop, which runs other ready coroutines. No threads, no thread stacks, no GIL contention between workers.

```python
import httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/proxy")
async def proxy(url: str):
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return {"ok": True, "keys": list(data)[:5]}
```

Where task-based async is a good fit:

- **High concurrency with low memory.** Coroutines are far cheaper than threads. A single process can hold many thousands of in-flight connections without a proportional memory cost for stacks.
- **Latency-sensitive I/O paths.** Because nothing blocks the loop, a slow dependency does not stall unrelated requests — provided timeouts are set.
- **Spiky traffic.** Fewer, smaller instances can absorb bursts, which usually reduces infrastructure cost.

The costs are real too:

- **Library coverage.** Many libraries still ship synchronous clients only. Wrapping them with `run_in_threadpool` reintroduces the threaded model, so a "task-based" service can quietly become a hybrid.
- **Silent bugs.** A missing `await` produces a coroutine object instead of a result. Python emits `RuntimeError: coroutine 'foo' was never awaited` at garbage-collection time, which is often far from the line that caused it. Similarly, an `await` on a non-awaitable raises `TypeError` immediately.
- **Blocking calls hide in plain sight.** A synchronous ORM call, a `time.sleep`, or a CPU-heavy loop inside a coroutine blocks the entire loop, and the symptom is *all* requests slowing down at once.

## How to measure the tradeoff on your own workload

Published benchmarks for this comparison are close to useless, because the answer depends entirely on the ratio of waiting to computing in *your* handlers. Measure it instead.

**What to instrument:**

1. **Wall time per handler**, broken into time spent awaiting I/O versus time spent executing Python.
2. **Event loop blocking.** A loop that is blocked cannot service timers. Schedule a coroutine that sleeps for a known interval and records the actual elapsed time; the difference is loop lag.
3. **Thread pool saturation**, if using threaded async: queued tasks versus pool size.
4. **RSS per process** and **CPU utilization** at steady state.

**What to run:**

- Generate load with a tool that reports latency percentiles, not just means. `wrk2` and `k6` both report p50/p90/p99; a mean alone will hide the tail behavior that matters.
- Run the same endpoint under both implementations with the same concurrency and the same artificial dependency latency.
- Vary the injected latency (for example, 1 ms, 50 ms, and 300 ms) to see where the curves diverge. The crossover point is the useful output of the exercise.

**What to compare:**

- p50 and p99 latency at fixed concurrency.
- Throughput at the concurrency level where p99 first exceeds your SLO.
- Peak RSS and CPU per unit of throughput.

**Illustrative arithmetic for the memory question.** Suppose a threaded server holds 100 threads per process and the platform reserves 8 MB of virtual address space per thread stack. That is nominally 800 MB of address space, though only touched pages count toward RSS. If the same concurrency is served by 100 coroutines at roughly a few kilobytes of Python object overhead each, the resident cost is orders of magnitude smaller. The point is not the exact number — it is that thread stacks scale linearly with concurrency while coroutine state does not. Substitute your platform's actual stack size and measure RSS to get the real figure.

**Illustrative arithmetic for the cost question.** If a service needs N instances to hold p99 under target, and task-based async halves N for the same traffic, the compute line item halves. Whether that is worth the migration depends on how large the compute line item is relative to engineering time. For a small service, the migration usually costs more than it saves. For a service running hundreds of instances, it usually does not.

## Failure modes to recognize

**Threaded async: pool exhaustion.** A downstream dependency degrades from 50 ms to 5 s. Threads block, the pool fills, and requests that do not touch that dependency begin queuing behind it. Symptom: broad latency increase with normal CPU. Mitigation: set aggressive timeouts on every offloaded call, size the pool deliberately, and monitor queue depth.

**Threaded async: accidental CPU offload.** A handler offloads a CPU-heavy transform to the thread pool. Under the GIL the threads serialize, so throughput does not improve and latency worsens. Symptom: high CPU, low throughput, no I/O wait. Mitigation: move genuinely CPU-bound work to a process pool or a separate worker service.

**Task-based async: a blocking call in a coroutine.** A synchronous client or an ORM call runs directly inside `async def`. The loop stalls for the duration. Symptom: every concurrent request slows down simultaneously. Mitigation: audit for synchronous I/O in async paths, and enable a debug mode that warns on slow callbacks where the framework supports it.

**Task-based async: missing timeout.** A coroutine awaits a dependency that never responds. The task hangs indefinitely, holding whatever resources it acquired. Mitigation: set explicit timeouts on every client, and wrap handler bodies in an overall deadline.

**Both: unhandled task exceptions.** A background task raises and the exception is only logged at garbage collection. Symptom: work silently does not happen. Mitigation: retain references to created tasks and attach done-callbacks that log or report failures.

## The decision checklist

Work through these in order. The first question that produces a clear answer usually decides the matter.

1. **Is the bottleneck waiting or computing?** If handlers spend most of their time waiting on network or disk, either model works and task-based is usually cheaper to run. If handlers spend significant time in Python-level computation, neither model fixes it — you need processes, native extensions, or a separate compute service.
2. **Can you adopt async-native clients for every dependency on the hot path?** If yes, task-based async is viable. If a critical dependency has no async client and no maintained alternative, threaded async is the honest choice.
3. **How much of the codebase must change?** A greenfield service is cheap to start task-based. A large existing codebase with synchronous data access is expensive, and a staged migration through threaded async may be the only realistic path.
4. **Does the team have async debugging habits?** Task-based async rewards teams that read loop-lag metrics and know how to trace a task. Threaded async rewards teams that understand pool sizing and timeouts. Neither is easier in the abstract; they are differently hard.
5. **What is the cost of being wrong?** If the service is small, pick whichever ships sooner and revisit later. If the service is large and latency-sensitive, invest in the measurement described above before committing.

## Practical configuration guidance

For threaded async:

- Never leave the pool size at an arbitrary default. Choose it from the expected concurrency and the timeout budget: if a call may take up to T seconds and you need to sustain C concurrent calls, you need roughly C threads, and you should verify that the resulting memory footprint is acceptable.
- Set a timeout on every offloaded call. A call with no timeout can hold a thread forever.
- Export pool queue depth and active thread count as metrics, and alert on sustained queue growth rather than on absolute values.

For task-based async:

- Set explicit timeouts on every client, at the connection and total-request level.
- Avoid synchronous I/O inside `async def`. If you must call it, offload it explicitly so the cost is visible in code review.
- Retain references to fire-and-forget tasks and attach error handling; otherwise failures disappear.
- Track event loop lag as a first-class metric. It is the single best leading indicator that something is blocking the loop.

For both:

- Load test with percentile reporting before and after any concurrency change.
- Watch p99, not the mean. Concurrency problems show up in the tail first.

## Take action in the next 30 minutes

Run this against a representative endpoint in your service to find out whether your handler is actually asynchronous or is delegating to a thread pool:

```bash
python - <<'PY'
import asyncio, inspect
from your_app import app  # adjust to your module

async def main():
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        kind = "async" if inspect.iscoroutinefunction(endpoint) else "sync (thread pool)"
        print(f"{getattr(route, 'path', '?'):40} {kind}")

asyncio.run(main())
PY
```

Any route reported as `sync (thread pool)` is running under the threaded model. Note which of those handlers touch a slow dependency, then check whether they have timeouts and whether the thread pool size is set deliberately. That list is your migration backlog, and it is usually shorter than expected.
