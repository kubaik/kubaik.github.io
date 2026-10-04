# Why your multi-agent system degraded slowly

## The one-paragraph version

Most teams assume a multi-agent system degrades because one agent failed or the orchestrator crashed. A quieter failure mode is a slow, cumulative change in how messages travel between agents. A small increase in a common header, or a small extra cost per serialization, can push p99 latency far above its original value over months without any obvious alert. The part that trips people up is that the agents themselves still succeed: queries return, tasks complete, the process does not crash. Every user request simply waits longer. That is the failure mode this article covers.

## Why this concept confuses people

Engineers expect degradation to announce itself: exceptions, timeouts, CPU spikes. Here the failure is invisible because the system keeps producing correct outputs.

A typical scenario is a Python orchestrator that fans out to several agents over a message broker, using JSON payloads. Each agent processes tasks quickly, but the orchestrator fans out to eight agents per request. Over time, teams add lightweight metadata: request IDs, trace context, agent tags, retry counters, agent version. Each addition is small and individually justified. After a few months, p99 latency has risen sharply while average latency stays nearly flat, because the slowest slice of requests now dominates. Alerts based on average latency never fire, and p99 is rarely monitored in staging. Teams spend days chasing agent CPU or database queries, missing the fact that the bottleneck is now serialization and round trips between agents.

This confusion is rooted in how observability tools are tuned. Most teams instrument agent-level metrics: CPU, memory, queue depth, error rates. Network and serialization costs between agents are invisible unless explicitly measured with tracing and payload-size tracking. The common mistake is to assume that because agents still respond within their usual budget, the entire system is healthy, ignoring that the orchestration layer now adds far more overhead per request than it used to.

## The mental model that makes it click

Think of the multi-agent system as a city transit network. The agents are buses, each completing its route in a predictable time. The orchestrator is the control system that schedules routes and dispatches riders. The problem is not the buses. It is the growing number of riders and their increasingly heavy luggage.

- **Payload size**: weight of luggage per rider
- **Serialization overhead**: time to load and unload luggage at each stop
- **Network latency**: time for the bus to travel between stops
- **Orchestration latency**: time the control system spends coordinating schedules

A small increase in payload size does not break the bus, but it adds time at every stop. With eight stops per trip, that per-stop cost is multiplied by eight on every request. Over months, teams add several stops' worth of metadata, and the overhead grows accordingly. Meanwhile, the buses still run on time. No alarms fire, but commuters wait longer. The degradation is gradual, cumulative, and invisible unless you measure the transit time between agents, not just the time agents spend working.

Another useful analogy: a restaurant kitchen with multiple stations. Each dish moves between stations to be prepared. The head chef assigns tasks and tracks progress. Adding a new label to each ticket slows the ticket's movement because the chef spends more time reading labels before assigning stations. The stations themselves work at the same speed, but the tickets spend more time in transit. The kitchen is not on fire, but service slows down.

The key insight: **the system's correctness is preserved, but its responsiveness degrades because coordination overhead grows silently.**

## A worked example

Consider a Python orchestrator that fans out to eight agents per request and collects responses. Suppose each agent processes tasks in roughly 15 ms, and the initial p99 latency is 80 ms. Over three months, several small changes accumulate. All figures below are illustrative, chosen to make the arithmetic visible; substitute your own measurements.

1. **Metadata creep.** Each request starts with a 120-byte JSON payload. Over time the team adds a request ID (16 bytes), trace context (32 bytes), agent tags (8 bytes), retry metadata (16 bytes), and agent version (4 bytes). Total added: 76 bytes. Payload grows from 120 to 196 bytes.

2. **Serialization cost.** Suppose serialization cost scales with payload size. If the measured cost is 0.02 ms per agent per 100 bytes, then 196 bytes costs about 0.039 ms per agent versus 0.024 ms at 120 bytes. Across eight agents that is about 0.31 ms versus 0.19 ms per request, a delta of roughly 0.12 ms.

3. **Network framing.** The broker adds a roughly constant per-message framing cost. If that is 0.5 ms per message, eight agents cost about 4 ms per request. This does not grow with payload size, but it is a fixed tax that is easy to forget.

4. **Broker queueing.** As payload size increases, broker memory usage grows. If the broker's memory limit is not tuned, occasional pauses and queueing delays appear. Suppose these add 2 ms to p99 intermittently.

5. **Orchestrator GC.** In a garbage-collected runtime, larger payloads mean more live objects and longer pauses. Suppose pauses grow from about 1 ms to about 3 ms per cycle under the larger payload.

Summing the deltas that scale with payload: serialization adds about 0.12 ms, and GC adds about 2 ms. Broker queueing adds about 1 ms. Payload growth itself adds a small amount of copy and allocation cost. That is on the order of a few milliseconds, not hundreds.

So where does a large p99 increase come from? The answer is usually **queueing and retries**, not the per-message arithmetic. As broker memory pressure grows, queueing becomes more volatile. The orchestrator starts seeing timeouts on a small percentage of requests. It retries those requests. Retries add load, which increases queueing, which causes more timeouts. The p99 latency balloons while the average stays low, because only a small fraction of requests are affected.

The lesson is not that metadata is expensive in absolute terms. It is that metadata pushes the system closer to a threshold where queueing and retry feedback take over.

## The orchestrator code, before and after

A first version of the orchestrator might look like this:

```python
import asyncio
import json
import nats
from fastapi import FastAPI

app = FastAPI()
nc = await nats.connect("nats://localhost:4222")

async def call_agent(payload: dict) -> dict:
    subject = f"agent.{payload['agent_id']}"
    response = await nc.request(subject, json.dumps(payload).encode(), timeout=0.03)
    return json.loads(response.data)

@app.post("/task")
async def process_task(task: dict):
    agents = ["agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7", "agent8"]
    tasks = [call_agent({"task": task, "agent_id": agent}) for agent in agents]
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

After some months, payload size grows and the orchestrator starts timing out on a small percentage of requests. A natural reaction is to add retries:

```python
async def call_agent_with_retry(payload: dict, max_retries=3) -> dict:
    for attempt in range(max_retries):
        try:
            subject = f"agent.{payload['agent_id']}"
            response = await nc.request(subject, json.dumps(payload).encode(), timeout=0.03)
            return json.loads(response.data)
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            payload["retry_count"] = attempt + 1
            await asyncio.sleep(0.01 * (attempt + 1))
```

The retry logic adds latency variability. Worse, if the timeout and backoff are not adjusted as payload size grows, retries increase queueing pressure, which increases latency, which triggers more retries. The p99 latency balloons while the average stays low because only a small fraction of requests are affected.

Note also that the first snippet uses a top-level `await` outside an async context, which will not run as written. In practice the connection is created during application startup, as shown later.

## How this connects to things you already know

This problem is a cousin of the N+1 query problem in databases, but it happens in distributed orchestration instead of SQL. In a monolith, an N+1 query adds database latency proportional to the number of rows fetched. In a multi-agent system, a fan-out to eight agents multiplies every per-message cost by eight, and retries multiply it again.

It is also similar to the thundering herd problem in caching: when many requests retry simultaneously, the broker becomes overwhelmed. The difference is that the thundering herd here is silent. No cache stampede alerts fire because the broker does not surface serialization or queueing latency by default.

Another familiar pattern is memory growth in long-running services. In this case, the growth is not a leak in the classic sense. It is payload size. Each request carries more metadata, and the broker's memory usage grows. GC pauses increase, but the team does not correlate GC pauses with payload size because they are monitoring CPU, not serialization or broker memory.

The key connection: **orchestration overhead is invisible unless you measure it explicitly.** Most teams measure agent-level metrics and assume the orchestration layer is negligible. When the fan-out factor is high and payload size grows, the orchestration layer becomes the bottleneck.

## Common misconceptions, corrected

### Misconception 1: "If the agents are fast, the system is fast."

**Correction:** Agent speed is necessary but not sufficient. The orchestration layer's overhead, including serialization, network framing, queueing, and GC pauses, can dwarf agent processing time. A system with eight agents each taking 15 ms can still have a high p99 latency if orchestration overhead and queueing dominate.

### Misconception 2: "Retries fix timeouts."

**Correction:** Retries can amplify the problem. Each retry increases queueing pressure, which increases latency, which triggers more retries. The retry loop becomes a positive feedback loop that degrades p99 latency without changing average latency. The fix is not more retries. It is reducing the need for retries by increasing timeouts, reducing payload size, or fixing the underlying bottleneck.

### Misconception 3: "Monitoring message rate is enough."

**Correction:** Message rate tells you throughput, not latency. A broker can handle the same message rate at very different latencies. The difference is serialization, queueing, and GC overhead. Monitor message latency, not just rate.

### Misconception 4: "Payload size growth is unavoidable."

**Correction:** It is avoidable with discipline. Prefer compact binary encodings over JSON for internal messages. Enforce size limits on payloads and reject or log requests that exceed them. Compression can help for large payloads, but measure the CPU cost, because compression can add milliseconds per message.

### Misconception 5: "GC pauses are a Python problem."

**Correction:** GC pauses happen in any language runtime with garbage collection. Node.js has similar issues with large JSON payloads. Python's GC is more visible because it is synchronous and can block the event loop. In both cases, the fix is to reduce payload size or move the orchestrator to a runtime with more predictable latency.

## The advanced version

Once payload size and serialization are under control, the next layer of degradation comes from broker queueing and connection handling.

### Tune broker memory and queueing

Brokers typically have a configurable memory limit. If the broker handles a high message rate with small payloads, memory usage grows slowly. If payload size grows, memory usage grows proportionally. Over time, the broker's memory usage approaches its limit, and queueing delays increase.

To address this:

1. Set an explicit memory limit in the broker configuration. The exact key depends on the broker; for a NATS-style server it looks like this:
   ```yaml
   max_memory: 4GB
   max_file_descriptors: 10000
   ```

2. Use persistent streams if you need durability across restarts. Persistence adds latency but prevents message loss.

3. Monitor disk and memory usage for the broker and set alerts well below the limit.

### Switch to a compact binary encoding

JSON is convenient but verbose. A schema-based binary encoding or a compact binary format such as MessagePack can cut payload size substantially and reduce serialization time. The exact numbers depend on the schema and the data, so measure them on your own payloads. A simple benchmark is to serialize a representative payload one thousand times in each format and compare total wall time and byte size.

### Reuse connections in the orchestrator

Creating a new broker connection per request adds a TCP handshake and protocol setup cost. For high request rates this is wasted work. Reuse a single connection created during application startup:

```python
import json
import nats
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nats = await nats.connect(
        "nats://localhost:4222",
        max_reconnect_attempts=-1,
        reconnect_time_wait=1,
        connection_name="orchestrator",
    )
    yield
    await app.state.nats.close()

app = FastAPI(lifespan=lifespan)

async def call_agent(payload: dict) -> dict:
    subject = f"agent.{payload['agent_id']}"
    response = await app.state.nats.request(subject, json.dumps(payload).encode(), timeout=0.03)
    return json.loads(response.data)
```

### Monitor orchestration overhead explicitly

Tracing lets you attribute latency to serialization, network, and broker queueing. A minimal instrumentation looks like this:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

async def call_agent(payload: dict) -> dict:
    with tracer.start_as_current_span("call_agent") as span:
        span.set_attribute("payload_size", len(json.dumps(payload).encode()))
        subject = f"agent.{payload['agent_id']}"
        response = await app.state.nats.request(subject, json.dumps(payload).encode(), timeout=0.03)
        return json.loads(response.data)
```

Then set alerts on the metrics that actually matter:

- broker message latency at the 99th percentile
- average and maximum payload size per subject
- broker queue depth
- orchestrator GC pause duration

### Consider a runtime with more predictable latency

For high-throughput systems, a runtime with less GC pressure can reduce latency variability. The trade-off is development cost and ecosystem maturity. Measure before committing. A useful experiment is to run the same workload against two orchestrator implementations and compare p99 latency under sustained load.

## Quick reference

| Problem | Symptom | Root cause | Fix |
|---|---|---|---|
| Payload size growth | p99 latency rises, agent CPU flat | Added metadata, verbose encoding | Compact encoding, enforce size limits |
| Broker queueing pressure | Intermittent timeouts, memory growth | Memory pressure from larger payloads | Tune broker memory, use persistence carefully |
| Orchestrator GC pauses | High latency variability | Large payloads in a GC runtime | Reduce payload size, change runtime |
| Retry amplification | p99 spikes, retry count rises | Timeouts too low, no backoff | Raise timeouts, add exponential backoff |
| Connection overhead | Fixed per-request cost | New connection per request | Reuse connections |

## FAQ

### How do I know if degradation is from orchestration overhead?

Check three things: p99 latency, payload size, and broker queue depth. If p99 latency is rising but agent CPU and memory are flat, the problem is likely orchestration overhead. If payload size has grown substantially over the same period, that is a strong signal. Trace message flow to see where latency is added. A common failure mode is finding that most of the latency sits in serialization and network layers, not in agent processing.

### Should I switch to gRPC instead of JSON-RPC?

Switching transport will not help unless you also address payload size growth and broker queueing. The real win is reducing payload size with a compact encoding and tuning the broker. gRPC is better for bidirectional streaming, but for request-reply orchestration a compact encoding over your existing broker is often simpler.

### My team uses Node.js for the orchestrator. Is this problem the same?

Yes. Node.js has similar issues with large JSON payloads and GC pauses. The fix is the same: reduce payload size, reuse connections, and monitor orchestration overhead with tracing. The event loop adds per-message overhead that compounds with fan-out.

### We have already tuned the broker and switched encodings. Why is p99 still high?

Check for retry loops and connection reuse. If the orchestrator retries a small percentage of requests without adjusting timeouts, the retry overhead can dominate p99. Verify that broker connections are reused rather than created per request. Finally, check for GC pauses in the orchestrator runtime, which can persist even after switching to a compact encoding.

### How much latency does a single retry add?

It depends entirely on the timeout and the queueing state. If the timeout is 30 ms, the retry itself costs at least 30 ms, plus whatever queueing delay the retry causes. Under pressure, that second cost can be much larger than the first. Retries are a positive feedback loop: the more you retry, the worse latency gets. The fix is to increase timeouts or remove the need for retries.

## Do this in the next 30 minutes

Open your orchestrator's request handler and add one line that records the serialized payload size as a metric, tagged by subject or route. Then graph the 99th percentile of that metric over the last 30 days alongside your p99 request latency. If payload size has grown while agent latency has not, you have found the beginning of the failure mode.
