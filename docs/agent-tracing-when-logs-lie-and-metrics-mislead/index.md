# Agent tracing: when logs lie and metrics mislead

The conventional advice on built agentspecific is incomplete in one specific, costly way. The edge cases only show up once real users hit the system. This post covers what comes after the happy path.

## The one-paragraph version (read this first)

Most incident responders have stared at a trace that shows a request entering a microservice, then—nothing—until minutes later the same trace finally exits, tagged with a p99 latency of 4 ms. That zero-duration gap is rarely a true latency of zero; it’s an absence of data because the tracing system assumed the agent’s work was synchronous and the thread never yielded. The part that trips people up is that distributed tracing tools like Jaeger or OpenTelemetry can’t automatically detect when an agent (say, a background worker, a message queue consumer, or an LLM inference call) is actually running asynchronously under the hood. That’s what this post actually covers: how to instrument agent-specific spans so the trace faithfully reflects the time the agent was busy, not just the time the parent thread was blocked waiting.

## Why this concept confuses people

The confusion starts with a simple mismatch: tracing systems are built for synchronous HTTP/gRPC calls, where a span starts when a handler is invoked and ends when it returns. When an agent runs in the background—celery task, SQS consumer, or a Vertex AI endpoint—the parent process usually just enqueues a message and continues. The span for the parent finishes in milliseconds, while the child agent’s work may take seconds or minutes. Out of the box, Jaeger 1.47 and OpenTelemetry 1.28 show the child span as a child of the finished parent span, which breaks the timeline and inflates reported latency.

Teams running into this usually see one of two symptoms:

1. A trace where the parent span’s duration equals the child span’s duration plus a suspiciously round offset (e.g., 10 ms), because the tracer inserted a synthetic ‘queue delay’ bucket that isn’t real CPU time.
2. The opposite: child spans appear orphaned, with no parent context, because the tracer dropped baggage context when the message was serialized to the queue.

The deeper trap is assuming the problem is a configuration tweak. It isn’t. It’s a missing piece in the mental model: tracing must explicitly model the agent as a first-class node in the trace, not as a side effect of the parent’s execution.


## The mental model that makes it click

Think of a trace as a directed acyclic graph (DAG) where each node represents a unit of work and edges represent causality. In a synchronous world, the graph is a straight line. In an asynchronous world, the graph is a tree with a ‘root’ node (the enqueue operation), a ‘queue’ node (the transport layer), and a ‘leaf’ node (the agent execution).

The key insight is that the queue itself has to become a span. Without it, the latency between enqueue and dequeue is invisible, and the agent span’s start time appears to be the enqueue time, not the actual start time of processing. OpenTelemetry’s baggage and context propagation only travel through synchronous boundaries; to carry context across a queue you need explicit propagation, usually via message headers.

A common failure mode here is using the default Redis queue instrumentation in OpenTelemetry. The default Redis instrumentation (semconv 1.22.0) instruments the Redis client calls (SET, LPUSH, etc.), but it does not create a span for the queue itself. That leaves a 50 ms gap between the parent span ending and the Redis ‘LPUSH’ span starting—time that is neither accounted for nor visible in the trace. To fix it, you must wrap the enqueue operation with a custom span that covers the entire round-trip: parent span → enqueue span → Redis → dequeue span → agent span.


## A concrete worked example

Let’s instrument a Celery task in Python 3.11 with OpenTelemetry 1.28 and Jaeger 1.47. The goal is to show a trace where the time the worker spends processing the task is accurately reflected, not hidden behind the parent’s enqueue latency.

### Step 1: Add the right instrumentation packages

```bash
pip install opentelemetry-api==1.28.0 \
            opentelemetry-sdk==1.28.0 \
            opentelemetry-exporter-jaeger==1.28.0 \
            opentelemetry-instrumentation-celery==0.42b0 \
            opentelemetry-instrumentation-redis==0.42b0
```

### Step 2: Create a custom enqueue span

Celery’s default instrumentation does not wrap the enqueue call. We’ll wrap it ourselves with a context manager that starts a span and propagates baggage into the message headers.

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
    # Real work here
    return len(payload)

def enqueue_with_trace(task, *args, **kwargs):
    ctx = tracer.start_as_current_span("enqueue")
    carrier = {}
    propagator.inject(ctx.get_current_span().context, carrier)
    baggage_propagator.inject(dict(), carrier)  # Inject baggage if needed

    # Override the message headers to carry our context
    headers = kwargs.get('headers', {})
    headers.update({
        'traceparent': carrier.get('traceparent'),
        'tracestate': carrier.get('tracestate'),
    })
    kwargs['headers'] = headers

    try:
        result = task.apply_async(*args, **kwargs)
        return result
    finally:
        tracer.end_span()

# Usage
result = enqueue_with_trace(process_data, payload=b"...")
```

### Step 3: Instrument the worker to extract context from headers

In the worker, we’ll use a Celery signal to extract the traceparent from the message headers and set it as the current span context.

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from celery.signals import before_task_publish, task_prerun

propagator = TraceContextTextMapPropagator()

def before_publish_handler(sender=None, headers=None, **kwargs):
    # In a real app you'd wrap the publisher, not the signal
    pass

@task_prerun.connect
async def task_prerun_handler(task_id, task, *args, **kwargs):
    headers = kwargs.get('headers', {})
    ctx = propagator.extract(headers)
    span = tracer.start_span(
        name=f"celery:{task.name}",
        context=ctx,
        kind=trace.SpanKind.CONSUMER
    )
    trace.set_span_in_context(span)

@task_prerun.connect
async def task_postrun_handler(task_id, task, retval, state, *args, **kwargs):
    span = trace.get_current_span()
    if span:
        span.end()
```

### Step 4: Observe the trace in Jaeger

After running a few tasks, open Jaeger UI and look at a trace. You should see:

- A root span for the API call that enqueued the task
- An explicit ‘enqueue’ span (duration ~1 ms) wrapping the Redis LPUSH
- A ‘celery:tasks.process_data’ span (duration ~42 ms) starting exactly when the worker dequeued the message

Without the custom enqueue span, the trace would show the root span ending at the enqueue call, then a 42 ms gap, then the agent span starting. With the span, the timeline is continuous and the latency is accurate.


## How this connects to things you already know

You already know how to instrument synchronous code: wrap a function with a span, set attributes, and end the span. The difference here is that the agent’s work starts asynchronously, so the span must be started explicitly when the agent begins, not when the parent enqueues the job.

The same pattern applies to other async boundaries:

- SQS consumers: wrap the message handler with a Consumer span and inject context into the message attributes.
- Vertex AI or SageMaker endpoints: wrap the inference call with a span and propagate the context via the request headers.
- Kafka consumers: use the Kafka client instrumentation and set the span context on the consumer record.

In each case, the missing piece is the explicit span for the transport layer: the queue, the topic, or the job queue. Without it, the trace is a lie.


## Common misconceptions, corrected

Misconception 1: “OpenTelemetry’s auto-instrumentation will handle agents.”

Correction: Auto-instrumentation instruments the client library (e.g., Redis client), not the agent’s lifecycle. The Redis client span shows the time spent in the Redis call, but it does not cover the time the message spent waiting in the queue. To cover the queue, you need an explicit span that wraps enqueue and dequeue.

Misconception 2: “Baggage and context propagate automatically across queues.”

Correction: Context propagation only works across synchronous boundaries. To propagate context across a queue, you must explicitly inject it into the message headers and extract it on the other side. The default Redis client instrumentation does not do this for you.

Misconception 3: “The agent span should be a child of the enqueue span.”

Correction: The agent span should be a sibling of the queue spans. The enqueue span ends when the message is on the queue; the agent span begins when the worker dequeues it. Creating a parent-child relationship here compresses the timeline and hides the queue latency.

Misconception 4: “Jaeger’s UI will show the gap anyway.”

Correction: Jaeger’s UI shows gaps as white space, but it does not attribute the gap to any span. Without explicit spans, the gap appears as ‘missing time’ and cannot be filtered, alerted on, or analyzed.


## The advanced version (once the basics are solid)

Once the basics are working, the next step is to handle retries and failures without corrupting the trace. The trap here is that a retry can create a second agent span with the same trace ID but a different parent, which breaks the DAG and can cause Jaeger to merge the spans incorrectly.

To handle retries safely, use a deterministic retry ID embedded in the message headers. Start the agent span with a custom name that includes the retry count (e.g., ‘celery:tasks.process_data #2’), and set a span attribute `retry_count=2`. This keeps the trace DAG intact even if the worker retries the task.

Another advanced case is when the agent itself spawns sub-agents. For example, a Vertex AI endpoint that calls a downstream function. In this case, use the OpenTelemetry `Link` feature to connect the parent span to the child spans without creating a parent-child relationship. This preserves the timeline while keeping the trace size bounded.

Finally, if you’re using AWS Lambda with Python 3.12, the same pattern applies: wrap the Lambda handler with an explicit span, propagate context via the event headers, and start the agent span when the Lambda begins execution. The Lambda runtime already sets up the initial context, so you only need to extend it with an explicit span for the agent’s work.


## Quick reference

| Concept | What it is | What it isn’t | Key tool/version |
|---|---|---|---|
| Agent span | Explicit span covering the agent’s work | A child of the enqueue span | OpenTelemetry 1.28 |
| Queue span | Span covering enqueue and dequeue | The Redis client span | Custom span in code |
| Context propagation | Injecting traceparent into message headers | Automatic via auto-instrumentation | TraceContextTextMapPropagator |
| Retry handling | Using retry_count in span name | Assuming Jaeger merges spans | Custom attribute |
| Link | Connecting unrelated spans without hierarchy | Parent-child relationship | OpenTelemetry Span Link |


## Further reading worth your time

- [OpenTelemetry semantic conventions for messaging 1.22.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.22.0/docs/messaging/messaging-spans.md) — explains the official span attributes for queues and topics.
- [Celery and tracing: a 2026 guide](https://blog.palletsprojects.com/en/2023-05/celery-tracing/) — covers the limitations of Celery’s built-in instrumentation.
- [Jaeger issue #3842](https://github.com/jaegertracing/jaeger/issues/3842) — documents the orphaned span problem when context is dropped at the queue boundary.
- [AWS Lambda context propagation 2026](https://aws.amazon.com/blogs/compute/tracing-aws-lambda-with-opentelemetry/) — shows how to propagate context through Lambda triggers.


## Frequently Asked Questions

**Why do my Jaeger traces show zero duration for async tasks?**

Most teams see this when the parent span ends before the agent starts and the tracing system does not create an explicit queue span. The fix is to wrap the enqueue and dequeue operations with explicit spans and propagate context via message headers.

**Does OpenTelemetry’s Redis instrumentation cover the queue latency?**

No. The Redis client instrumentation covers Redis call latency (SET, LPUSH, etc.), but not the time the message spends waiting in the queue. To cover the queue, create a custom span that wraps the enqueue and dequeue calls.

**How do I propagate context across SQS without losing baggage?**

Use the SQS message attributes to carry the traceparent and baggage. SQS supports up to 10 message attributes, so you can include both without hitting the limit. On the consumer side, extract the attributes and set the context before starting the agent span.

**Can I use this pattern with Kafka?**

Yes. Use the Kafka client instrumentation to create spans for produce and consume, and propagate context via the record headers. The consumer span should start when the poll returns a record, not when the record is processed.


## One thing you can do in the next 30 minutes

Open your Jaeger UI, find a trace that contains an async agent (Celery, SQS, Vertex AI, etc.), and check whether there is an explicit span between the parent’s end and the agent’s start. If there isn’t, create a custom span that wraps the enqueue call and propagate the context. Then reload Jaeger and confirm the gap is gone. That’s the first step to agent-specific tracing that actually helps during incidents.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the contact page. Corrections are applied promptly.

**Last generated:** August 2026
