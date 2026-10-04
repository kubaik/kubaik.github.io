# Agent tracing: when logs lie and metrics mislead

Distributed tracing assumes a request enters a process, work happens on the calling thread, and the span ends when the handler returns. Background agents violate that assumption. A worker, a queue consumer, or a remote inference endpoint runs on a different thread, process, or machine, often minutes after the caller moved on. The trace still renders, but the timeline it draws can be misleading rather than merely incomplete.

## The failure mode in one paragraph

A trace shows a request entering a service and then nothing until the same trace exits, tagged with a suspiciously small duration. The gap is rarely zero latency; it is missing data. The tracing system modelled the agent's work as synchronous, the parent span ended at enqueue time, and no span was ever created for the time the message sat in the broker or the time the worker spent processing it. Tools such as Jaeger and OpenTelemetry cannot infer that an agent is running asynchronously unless instrumentation creates a span for that work. This article covers how to model agent work as first-class spans so the trace reflects busy time rather than the parent thread's blocked time.

## Why synchronous assumptions break

Tracing was designed around synchronous request/response protocols. A span starts when a handler is invoked and ends when it returns. The parent-child relationship encodes causality: the child runs inside the parent's execution window, so its duration is bounded by the parent's.

When an agent runs in the background, the parent typically enqueues a message and continues. The parent span finishes in milliseconds; the agent's work may take seconds or minutes. Auto-instrumentation attaches the agent span as a child of the finished parent, which distorts the timeline and inflates reported latency. Two symptoms are common:

1. The parent span's duration equals the child span's duration plus a small, oddly round offset, because the tracer inserted a synthetic queue-delay bucket that does not correspond to CPU time.
2. Child spans appear orphaned, with no parent context, because the tracer dropped the propagation context when the message was serialized to the broker.

The deeper trap is treating this as a configuration tweak. It is a modelling gap: the agent must be an explicit node in the trace graph, not a side effect of the parent's execution.

## The mental model: traces as a DAG

A trace is a directed acyclic graph where nodes are units of work and edges are causal links. In synchronous code the graph is a straight line. Across an async boundary the graph branches: an enqueue node, a transport node representing time in the queue, and a leaf node representing agent execution.

The queue itself must become a span. Without it, the latency between enqueue and dequeue is invisible, and the agent span's start time appears to be the enqueue time rather than the actual processing start. Context propagation only travels through synchronous boundaries by default; carrying it across a broker requires explicit injection into message headers and extraction on the consumer side.

A representative failure mode: instrumenting only the broker client. Client instrumentation creates spans for the client calls (for example, a Redis `LPUSH`), but not for the queue itself. That leaves a gap between the parent span ending and the client span starting, time that is neither accounted for nor visible. The fix is to wrap the enqueue operation with a span that covers the full round trip: parent span, enqueue span, broker call, dequeue span, agent span.

## A worked example: tracing a Celery task

The following example instruments a Celery task with OpenTelemetry and exports to Jaeger. The goal is a trace where worker processing time is visible rather than hidden behind enqueue latency. Version numbers are illustrative; pin whatever your environment supports and verify compatibility between the API, SDK, exporter, and instrumentation packages, since instrumentation packages track the API version loosely.

### Step 1: Install instrumentation packages

```bash
pip install opentelemetry-api \
            opentelemetry-sdk \
            opentelemetry-exporter-jaeger \
            opentelemetry-instrumentation-celery \
            opentelemetry-instrumentation-redis
```

### Step 2: Create a custom enqueue span

Celery's default instrumentation does not wrap the enqueue call. Wrap it with a context manager that starts a span and injects the propagation context into the message headers.

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import BaggagePropagator
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()
baggage_propagator = BaggagePropagator()

@app.task(bind=True)
def process_data(self, payload):
    return len(payload)

def enqueue_with_trace(task, *args, **kwargs):
    headers = dict(kwargs.pop('headers', {}))
    with tracer.start_as_current_span("enqueue") as span:
        propagator.inject(headers)
        baggage_propagator.inject(headers)
        result = task.apply_async(*args, headers=headers, **kwargs)
        return result

result = enqueue_with_trace(process_data, payload=b"...")
```

Two corrections matter here. First, `start_as_current_span` returns a context manager; calling `tracer.end_span()` afterwards is not a valid API and would either fail or end the wrong span. Second, `inject` writes into the carrier dictionary you pass it, so pass the headers dict directly rather than a fresh dict whose contents you then copy.

### Step 3: Extract context in the worker

On the consumer side, extract `traceparent` from the message headers and start the agent span as a child of that context.

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from celery.signals import task_prerun, task_postrun

tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()

@task_prerun.connect
def task_prerun_handler(task_id=None, task=None, **kwargs):
    request = kwargs.get('request') or getattr(task, 'request', None)
    headers = getattr(request, 'headers', None) or {}
    ctx = propagator.extract(headers)
    span = tracer.start_span(
        name=f"celery:{task.name}",
        context=ctx,
        kind=trace.SpanKind.CONSUMER,
    )
    token = trace.use_span(span, end_on_exit=False)
    trace.set_span_in_context(span)

@task_postrun.connect
def task_postrun_handler(task_id=None, task=None, **kwargs):
    span = trace.get_current_span()
    if span and span.is_recording():
        span.end()
```

Celery signals are synchronous; declaring the handlers `async` would cause them to return coroutines that Celery never awaits, so the spans would never start or end. The context token returned by `use_span` should be detached in `task_postrun` to avoid leaking context across tasks on a reused worker thread.

### Step 4: Observe the trace in Jaeger

Run a few tasks and open the Jaeger UI. A correctly instrumented trace shows:

- A root span for the API call that enqueued the task.
- An explicit `enqueue` span wrapping the broker call.
- A `celery:tasks.process_data` span starting when the worker dequeued the message.

Without the custom enqueue span, the trace shows the root span ending at the enqueue call, then a gap, then the agent span. With it, the timeline is continuous.

## How to measure whether your traces are honest

Do not trust a single trace. Instrument and compare:

1. Pick a workload with a known processing duration, for example a task that sleeps for a fixed interval.
2. Record the wall-clock time between enqueue and worker start using your broker's timestamps or a monotonic clock in the producer and consumer.
3. Compare that measured interval to the sum of span durations between the enqueue span and the agent span in the exported trace.
4. If the trace's accounted time is consistently shorter than the wall-clock interval, you have an unmodelled gap.

Repeat across a batch of messages and look at the distribution, not the mean. A gap that appears only at the tail is a queue-contention or consumer-lag problem; a gap that appears on every message is an instrumentation problem.

## Common misconceptions

**Auto-instrumentation will handle agents.** Auto-instrumentation instruments the client library, not the agent's lifecycle. A broker client span covers the client call, not the time the message spends waiting. Covering the queue requires an explicit span.

**Baggage and context propagate automatically across queues.** Propagation works across synchronous boundaries. Across a broker, you must inject into message headers and extract on the other side. Client instrumentation does not do this for you.

**The agent span should be a child of the enqueue span.** The agent span should be a sibling of the queue spans. The enqueue span ends when the message is on the queue; the agent span begins when the worker dequeues it. A parent-child relationship compresses the timeline and hides queue latency.

**The UI will show the gap anyway.** A UI renders gaps as whitespace, but it does not attribute them to a span. Without explicit spans, the gap cannot be filtered, alerted on, or analysed.

## Advanced cases

**Retries.** A retry can create a second agent span with the same trace ID but a different parent, which breaks the DAG. Use a deterministic retry identifier in the message headers, name the span with the retry count, and set a `retry_count` attribute. This keeps the DAG intact.

**Sub-agents.** When an agent spawns downstream work, use span links to connect related spans without creating a parent-child relationship. This preserves the timeline while keeping trace size bounded.

**Serverless handlers.** The same pattern applies to function-as-a-service runtimes: wrap the handler with an explicit span, propagate context via the event headers, and start the agent span when execution begins. The runtime sets up initial context; you extend it.

## Quick reference

| Concept | What it is | What it is not | Mechanism |
|---|---|---|---|
| Agent span | Explicit span covering agent work | A child of the enqueue span | `tracer.start_span` with extracted context |
| Queue span | Span covering enqueue and dequeue | The broker client span | Custom span in application code |
| Context propagation | Injecting `traceparent` into message headers | Automatic via auto-instrumentation | `TraceContextTextMapPropagator` |
| Retry handling | `retry_count` attribute and retry-aware span name | Assuming the backend merges spans | Custom attribute |
| Link | Connecting unrelated spans without hierarchy | Parent-child relationship | Span links |

## FAQ

**Why do traces show zero duration for async tasks?**
The parent span ends before the agent starts and no explicit queue span exists. Wrap enqueue and dequeue with spans and propagate context via message headers.

**Does broker client instrumentation cover queue latency?**
No. It covers the client call, not the time the message waits in the queue. Create a custom span around enqueue and dequeue.

**How do I propagate context across SQS without losing baggage?**
Use message attributes to carry `traceparent` and baggage. Extract them on the consumer side and set the context before starting the agent span.

**Can I use this pattern with Kafka?**
Yes. Use the Kafka client instrumentation for produce and consume spans, and propagate context via record headers. The consumer span should start when the poll returns a record, not when processing finishes.

## One thing to do in the next 30 minutes

Open your tracing UI, pick a trace containing an async agent, and check whether an explicit span exists between the parent's end and the agent's start. If it does not, add a span around the enqueue call, inject the propagation context into the message headers, extract it in the worker, and confirm the gap is gone.
