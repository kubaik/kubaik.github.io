# Tool calls under load: async batch vs fire-and-forget

## The two patterns, stated precisely

A "tool call" here means any outbound request made from a request handler or worker to another service: an internal microservice, a model inference endpoint, a third-party API. The pattern used to dispatch those calls determines throughput, tail latency, and failure behavior far more than most code review discussions acknowledge.

Two dispatch patterns dominate:

- **Async batch calling.** Items are buffered, grouped by destination, and sent as one request (or a small number of requests). Results are fanned back out to the original callers. Grouping happens in-process or in a local queue before the network hop.
- **Fire-and-forget with retries.** Each item triggers its own call. The caller either does not wait for the result or waits only for an enqueue acknowledgment. Failures are retried with backoff, usually by a task queue or worker.

Neither is universally correct. The choice depends on workload shape (calls per request), latency budget for the main flow, required failure semantics, and how many distinct destinations exist. The rest of this article works through each pattern, then gives a measurement procedure so the decision is based on your own numbers rather than a rule of thumb.

## Why the choice matters

The failure mode that motivates this comparison is not "the tool call is slow." It is "the tool call is slow, so the request handler holds a connection, a task, and memory while it waits." Under load, that turns a downstream latency increase into an upstream capacity collapse: the handler pool saturates with requests that are all blocked on I/O, and the service stops accepting new work even though its own CPU is idle.

Three quantities predict whether this happens:

1. **Tool calls per request.** If each request makes one call, per-call overhead dominates and batching has nothing to group. If each request makes dozens, the per-call overhead (connection setup, TLS, serialization, scheduling) is multiplied by that count.
2. **Latency budget for the main flow.** A user-facing endpoint with a 200 ms budget cannot afford to wait for a batch drain interval plus a queue round trip. A background job with a minutes-long budget can.
3. **Failure semantics.** Whether a single failed item must block the whole result, can be skipped, or must be retried indefinitely changes which pattern is even legal.

If these three numbers are unknown, any pattern choice is a guess. The instrumentation section below describes how to get them.

## Option A: async batch calling

### How it works

A batcher holds a queue of pending items. A scheduler drains the queue when either the batch size is reached or a drain interval elapses, whichever comes first. The drained batch is sent as one request per destination. Responses are matched back to items by an index or a client-supplied ID.

The essential components:

- A bounded queue (unbounded queues convert overload into out-of-memory kills).
- A grouping key (destination URL, tenant, schema version).
- A drain trigger (size threshold, time threshold, or both).
- A response demultiplexer that maps results back to callers.
- A partial-failure policy (see below).

### A minimal implementation

```python
import asyncio
import httpx
from typing import Dict, List, Tuple

class AsyncBatcher:
    def __init__(self, url: str, batch_size: int = 50,
                 drain_interval: float = 0.05, max_concurrent: int = 4):
        self.url = url
        self.batch_size = batch_size
        self.drain_interval = drain_interval
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._workers = []

    async def start(self):
        self._workers = [asyncio.create_task(self._run()) for _ in range(2)]

    async def stop(self):
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        await self.client.aclose()

    async def add(self, item_id: str, payload: Dict):
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put((item_id, payload, fut))
        return await fut

    async def _run(self):
        while True:
            batch: List[Tuple[str, Dict, asyncio.Future]] = []
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=self.drain_interval)
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            deadline = asyncio.get_running_loop().time() + self.drain_interval
            while len(batch) < self.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:
                    break
            await self._dispatch(batch)

    async def _dispatch(self, batch):
        ids = [item[0] for item in batch]
        payloads = [item[1] for item in batch]
        futures = [item[2] for item in batch]
        async with self.semaphore:
            try:
                resp = await self.client.post(self.url, json={"items": payloads})
                resp.raise_for_status()
                results = resp.json()["results"]
                for fut, result in zip(futures, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
```

```python
# usage
batcher = AsyncBatcher(url="http://internal-service/extract", batch_size=50)
await batcher.start()
result = await batcher.add("doc-1", {"text": "..."})
```

This version fixes several problems common in naive batchers: the queue is bounded so overload produces backpressure instead of memory growth; the drain loop waits for either a full batch or the interval, so a partially filled batch is not stranded; each caller gets its own future, so one slow item does not corrupt the others' results; and the HTTP client is reused with an explicit connection limit.

### Failure modes specific to batching

- **Straggler amplification.** Tail latency of the batch is the maximum of its items' latencies, not the average. A batch of 50 where one item takes 400 ms makes all 50 callers wait 400 ms. This is the single most common reason batching makes P99 worse while improving throughput.
- **Head-of-line blocking.** If the queue is FIFO and one destination is slow, items behind it wait even if their destination is healthy. Grouping by destination before dispatch mitigates this.
- **Unbounded queue growth.** If the drain rate is below the arrival rate, a bounded queue applies backpressure (correct) and an unbounded queue consumes memory until the process dies (incorrect). Choose bounded.
- **Partial failure ambiguity.** If a batch request returns 200 with per-item errors, the caller must parse the body. If it returns 500, it is unclear which items were processed. Idempotency keys on the downstream service are the only reliable fix.

### When batching fits

- Calls per request are naturally high (tens or more) and arrive close together in time.
- The main flow can tolerate the drain interval plus the batch's maximum item latency.
- Destinations are few and stable, so grouping keys do not fragment into batches of one.
- The downstream service accepts a bulk endpoint and defines per-item error semantics.

## Option B: fire-and-forget with retries

### How it works

Each item is enqueued to a broker (Redis, SQS, RabbitMQ, Kafka) and a worker pool consumes the queue. The request handler returns as soon as the enqueue succeeds. Retries are handled by the worker framework, typically with exponential backoff and jitter, and terminal failures go to a dead-letter queue.

### A minimal implementation

```python
# tasks.py
from celery import Celery
import requests

app = Celery('tasks', broker='redis://redis:6379/0')

@app.task(bind=True, max_retries=5, acks_late=True)
def compliance_check(self, user_id: str, email: str):
    try:
        response = requests.post(
            "https://compliance.example.com/check",
            json={"user_id": user_id, "email": email},
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # 2^retries seconds, capped; add jitter in production
        countdown = min(2 ** self.request.retries, 300)
        raise self.retry(exc=exc, countdown=countdown)
```

```python
# app.py
from tasks import compliance_check

@app.post("/signup")
def signup(email: str):
    user_id = create_user(email)
    compliance_check.delay(user_id, email)
    return {"status": "pending_compliance"}
```

Two details matter more than the retry count. `acks_late=True` means the message is acknowledged only after the task completes, so a worker crash does not silently drop work. The `countdown` cap prevents unbounded backoff growth. Jitter (adding a random offset to the countdown) is required in production to avoid synchronized retry storms when a downstream service recovers.

### Failure modes specific to fire-and-forget

- **Retry storms.** When a downstream service fails, every in-flight task retries on roughly the same schedule. Without jitter and a cap, the retry traffic can exceed the original traffic and prevent recovery. This is a well-documented pattern; the mitigation is jitter plus a circuit breaker that pauses dispatch while the downstream is unhealthy.
- **Queue growth outpacing workers.** If arrival rate exceeds drain rate, queue depth grows. With Redis or SQS this is a memory or cost problem; with Kafka it is a lag problem. Alert on queue depth and age of the oldest message, not just on worker count.
- **Lost correlation.** The task runs in a different process from the original request, so the trace context must be propagated explicitly (a header or a field in the task payload). Without it, a failed task cannot be traced back to the user action that caused it.
- **Duplicate execution.** At-least-once delivery means tasks can run more than once. The downstream call must be idempotent, or the task must check a completion flag before acting.

### When fire-and-forget fits

- Calls per request are low (typically one) or are event-driven and not tied to a request lifecycle.
- The main flow cannot wait for the tool call, or the result is not needed to produce the response.
- Destinations are numerous or vary per request, so grouping would not help.
- Failure isolation is required: one failing call must not block unrelated work.

## Choosing between them: a decision checklist

Work through these in order. The first question that has a clear answer usually determines the pattern.

1. **Does the response depend on the tool call result?** If no, fire-and-forget is the default; batching only helps if you also need throughput on the enqueue side.
2. **How many calls per request, at the median and at P95?** Compute both. A median of 1 with a P95 of 50 is a spiky workload; a median of 40 with a P95 of 60 is a batch workload. They call for different patterns.
3. **What is the main flow's latency budget, and how much of it is left after the tool call?** If the budget is under roughly 200 ms, a drain interval of 50 ms plus a batch maximum latency is a large fraction of it. Fire-and-forget is safer unless the call is off the critical path.
4. **Must every item succeed, or can some be skipped?** If every item must succeed, fire-and-forget with a DLQ and manual replay gives an auditable path. Batching requires the downstream to define per-item error semantics.
5. **How many distinct destinations?** One or two stable destinations favor batching. Hundreds of per-request destinations fragment batches and favor fire-and-forget.
6. **What is the downstream service's rate limit and bulk API support?** A bulk endpoint with a documented item limit sets your batch size ceiling. A rate limit per caller favors a queue that can smooth traffic.

## How to measure before you choose

The decision should be driven by four metrics collected in your own system. None of them require new infrastructure beyond what most services already run.

**1. `tool_calls_per_request` (histogram).** Record the count of outbound tool calls made while handling each request, labeled by endpoint. This is the single best predictor of whether batching will help. A median above roughly 10 makes batching worth evaluating; a median below 2 makes it unlikely to pay off.

**2. `tool_call_latency_ms` (histogram, labeled by destination and outcome).** Time each call from just before dispatch to just after the response is parsed. Label by destination so a slow downstream is visible separately from a slow caller.

**3. `tool_call_queue_depth` and `oldest_item_age_seconds` (gauges).** For batching, this is the in-process queue. For fire-and-forget, it is the broker queue. The age of the oldest item is more actionable than the depth: it directly bounds how stale a result can be.

**4. `tool_call_errors_total` (counter, labeled by destination and error class).** Distinguish timeouts, connection errors, 4xx, and 5xx. A retry policy that treats all of them identically will retry non-retryable errors and waste the retry budget.

A minimal instrumentation wrapper:

```python
import time
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def instrumented_call(destination: str, coro_fn, *args, **kwargs):
    with tracer.start_as_current_span(f"tool_call.{destination}") as span:
        start = time.perf_counter()
        try:
            result = await coro_fn(*args, **kwargs)
            span.set_attribute("outcome", "ok")
            return result
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("outcome", type(exc).__name__)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            span.set_attribute("latency_ms", elapsed_ms)
```

Run this for one to two weeks, then compute:

- The median and P95 of `tool_calls_per_request` per endpoint.
- The P99 of `tool_call_latency_ms` per destination.
- The correlation between queue depth and request latency.

If the P95 of calls per request is high and the P99 of tool latency is low relative to the budget, batching is likely to help. If calls per request are low and the P99 is dominated by a single slow destination, batching will not fix it; the destination is the problem.

## Sizing a batch: a worked example

Suppose the main flow's latency budget is 300 ms, and the downstream service's P99 per-item latency is 80 ms. With a drain interval of D and a batch size of B, the worst-case added latency for an item that arrives just after a drain is approximately D plus the batch's maximum item latency. If item latencies are independent, the maximum of B items grows with B even when the mean is stable.

Illustrative arithmetic with stated assumptions: assume per-item latency is roughly constant at 80 ms and the drain interval is 50 ms. Then:

- An item arriving just after a drain waits up to 50 ms, then the batch takes about 80 ms. Worst case ≈ 130 ms.
- If per-item latency has a long tail (P99 of 400 ms), the batch's maximum is 400 ms, so worst case ≈ 450 ms, which exceeds a 300 ms budget.

The conclusion is that batch size does not affect the mean much but strongly affects the tail, because the tail of a batch is the maximum of its items. A practical approach is to cap batch size by the latency budget rather than by throughput: choose the largest B such that the observed P99 of the batch maximum stays within budget. Measure the batch maximum directly by recording, per batch, the maximum per-item latency.

If the downstream service exposes a bulk endpoint with a documented item limit, that limit is a hard ceiling. Otherwise, start with a size that keeps the batch request payload under a few hundred kilobytes and adjust based on the batch-maximum metric.

## Partial failures: the policy question

Both patterns must answer the same question: when one item fails, what happens to the others?

Three policies are common:

- **All-or-nothing.** The whole batch is retried. Simple, but a single poison item blocks the batch indefinitely. Requires a retry cap and a DLQ for the batch.
- **Per-item retry.** Failed items are re-queued individually. More precise, but requires the downstream to identify which items failed, and it can multiply queue traffic.
- **Skip-and-log.** Failed items are recorded and dropped. Acceptable only when the work is genuinely optional.

For fire-and-forget, the equivalent choice is the retry budget and the DLQ policy. A retry budget of three to five attempts with jitter and a cap is a common starting point; the DLQ must be monitored, because an unmonitored DLQ is a silent data loss channel.

## Operational considerations

**Backpressure.** Both patterns need an explicit answer to "what happens when arrival exceeds drain rate." For batching, a bounded queue that blocks or rejects is the answer. For fire-and-forget, the broker's retention limit and the worker autoscaling policy are the answer. In both cases, the failure should be visible as a metric before it becomes an outage.

**Idempotency.** At-least-once delivery applies to both patterns: a batch request can be retried after a timeout even though the downstream processed it, and a task can run twice. The downstream operation should be idempotent, keyed on a client-supplied ID, or guarded by a completion check.

**Observability cost.** Batching requires per-item metrics inside the batch pipeline to attribute failures. Fire-and-forget requires trace context propagation into the worker. Both are real costs and should be budgeted when choosing.

**Schema evolution.** If the tool call payload changes frequently, a worker-based pattern lets the worker be redeployed independently of the main service. A batcher embedded in the main service couples the two deployment cycles.

## FAQ

**How do I know if my workload is batch-shaped?**

Log `tool_calls_per_request` per endpoint for a week and look at the median and P95. A high median with a low spread is batch-shaped. A low median with a high P95 is spiky; batching will help the spikes but not the common case, so evaluate whether the spikes are worth the added complexity.

**What batch size should I start with?**

Start with a size that keeps the batch request payload small and the batch-maximum latency within budget. Measure the maximum per-item latency within each batch; if the P99 of that maximum exceeds your budget, reduce the size. There is no universal number, because it depends on the downstream's latency distribution.

**How do I handle a poison item that always fails?**

Cap retries and route terminal failures to a DLQ. For batching, retry the batch once, then split it or fall back to per-item calls for the failed batch so the poison item can be isolated.

**What retry strategy works for fire-and-forget?**

Exponential backoff with jitter, a cap on the delay, and a maximum attempt count. Without jitter, retries synchronize and amplify load on a recovering downstream. The cap prevents a task from being delayed indefinitely.

**How do I measure tool call latency without adding heavy instrumentation?**

Wrap the call site in a span or a timer that records duration and outcome, labeled by destination. Export as a histogram. The wrapper above is a few lines and adds negligible overhead compared to the network call itself.

## One action to take in the next 30 minutes

Add a `tool_calls_per_request` histogram to your main request handler: increment a counter at each outbound tool call site and record the total per request at the end of the handler. Deploy it, and in a week you will have the median and P95 that determine whether batching is worth building. Everything else in this article follows from those two numbers.
