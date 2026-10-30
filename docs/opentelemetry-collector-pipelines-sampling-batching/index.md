# OpenTelemetry collector pipelines: sampling, batching…

The OpenTelemetry Collector looks simple at the diagram level: receivers take data in, processors modify it, exporters send it out. That mental model holds until real traffic arrives. Then the interesting behavior appears — data drops, memory climbs, and dashboards go flat with no obvious cause.

The collector is not a black box. It is a set of pipelines, each with a memory footprint, a CPU cost, and a failure mode. When sampling, batching, and routing are combined in one pipeline without understanding how they interact, the result is a config that passes validation and behaves badly under load. This article covers the mechanics of those three concerns and how to design a pipeline that degrades predictably instead of failing silently.

## The scenario: a collector that works until it doesn't

A common pattern: a few services on Kubernetes, the collector deployed as a DaemonSet, SDKs pointed at the local agent. For a while everything is fine. Then traffic grows, someone adds a high-cardinality attribute to a span, and the collector starts dropping data. Restarting the collector restores service for a day, then the same thing recurs. The team blames the backend; the backend team blames the collector; nobody reads the pipeline config.

That pattern is a backpressure problem, not a mystery. The rest of this article explains why it happens and what to change.

## The mechanism: pipelines are queues with processors in the middle

A pipeline is defined under `service.pipelines`. Each entry has a name, a list of receivers, a list of processors, and a list of exporters. The collector fans out from each receiver into the pipeline and fans in from the processors to each exporter. Data moves through the pipeline as batches of telemetry items — spans, metric data points, or log records.

The detail that matters operationally: processors run synchronously in the pipeline's goroutine. If a processor blocks, the pipeline blocks. If the pipeline blocks, the receiver's queue fills. When the queue is full, the receiver drops data. Those drops are counted in the collector's own metrics — `otelcol_receiver_refused_spans`, `otelcol_receiver_refused_metric_points`, `otelcol_receiver_refused_log_records` — but those metrics are only visible if something is scraping the collector itself.

So the first failure mode is invisible backpressure. A slow exporter or an expensive processor causes drops at the receiver, and the only external symptom is missing data. The collector is behaving correctly; it is simply out of capacity somewhere downstream, and nothing upstream is told about it except through the drop counters.

### Batching: the memory multiplier

The `batch` processor groups items into batches before handing them to exporters. Its main knobs are `send_batch_size` (target number of items per batch), `timeout` (maximum wait before flushing a partial batch), and `send_batch_max_size` (a hard cap on batch size even when `send_batch_size` is larger).

Batching reduces the number of export requests, which matters for backends that bill per request or enforce rate limits. It also holds data in memory. A large `send_batch_size` combined with a slow exporter means the processor accumulates large batches and may hold several in flight. Spans carrying attributes are not small; a batch of thousands of spans can be tens of megabytes. Multiply by pipelines per collector and collector replicas, and the memory budget disappears.

A typical failure mode is setting a large batch size to reduce cost, then seeing the collector OOM during a traffic spike. The batch processor has no independent queue limit of its own; it relies on the exporter to apply backpressure. If the exporter has its own internal queue — many do — the batch processor can keep feeding it until memory runs out.

### Sampling: probabilistic vs. tail-based

The collector supports sampling at two distinct points.

Probabilistic sampling (the `probabilisticsampler` processor) keeps a fraction of spans based on a hash of the trace ID. It is cheap, stateless, and consistent across replicas: the same trace ID hashes the same way everywhere, so a trace is either kept or dropped as a whole rather than split. Its weakness is blindness — it drops the traces that matter (errors, slow requests) at the same rate as the ones that don't.

Tail-based sampling (the `tail_sampling` processor) buffers spans until a trace is complete, then decides based on policies such as latency, error status, or attribute values. It solves the blindness problem but introduces two costs:

- **Memory.** Spans are held until the decision is made. The processor has a `decision_wait` setting (default 30 seconds) and a `num_traces` setting controlling how many traces it tracks. Throughput times decision window gives a lower bound on tracked traces; tracked traces times spans per trace times span size gives the memory floor.
- **Topology.** All spans for a trace must arrive at the same collector instance. In a DaemonSet deployment, spans from one trace can land on different pods, so tail sampling requires a two-tier setup: local agents forward by trace ID to a second layer that owns the sampling decision.

### Routing: connectors and the fan-out problem

There are two broad ways to route data: filtering or routing processors that drop or tag items, and connectors, which are pipelines that act as both exporter and receiver. A routing connector sends data to different downstream pipelines based on attribute conditions.

A common pattern sends everything into one pipeline, then splits by service name or environment. The problem is that each downstream pipeline has its own processors and exporters, each with its own memory and CPU cost. Routing to three backends triples the export work. If one backend is slow, it can back up the shared source pipeline unless the slow path is isolated.

The collector does not isolate pipelines automatically. A slow exporter in one pipeline can cause the receiver to drop data for all pipelines that share that receiver. The practical fixes are separate receivers per destination, a `memory_limiter` at the head of each pipeline to shed load before it reaches expensive processors, and — where the collector build supports it — exporter-side queues with explicit sizing so a slow backend absorbs the delay instead of propagating it upstream.

## The fix: a pipeline that degrades gracefully

The goal is a pipeline that keeps the data you need under load and sheds the rest predictably. Four rules cover most of it:

1. Put a `memory_limiter` processor first in every pipeline.
2. Batch with a conservative size and a timeout matched to backend latency.
3. Sample early and cheaply where possible; use tail sampling only where the cost is justified.
4. Isolate routing so a slow backend cannot take down the whole collector.

### A worked example

Assume a service produces 20,000 spans per second at peak, and each span averages 1 KB in memory including attributes. The collector runs as a DaemonSet with 10 pods, so each pod handles 2,000 spans/s.

With `send_batch_size: 10000` and `timeout: 10s`, each pod could hold up to 10,000 spans per batch, which is about 10 MB per batch. With multiple batches in flight, batching alone can exceed 100 MB per pod. Add a tail-sampling buffer and the figure reaches the hundreds of megabytes — fine under a 1 GB limit, but with little headroom for a spike.

A more conservative config: `send_batch_size: 2000`, `send_batch_max_size: 3000`, `timeout: 5s`. That caps each batch at roughly 3 MB; with two batches in flight, batching memory stays under about 10 MB per pod. The trade-off is more export requests: at 2,000 spans/s and 2,000 spans per batch, that is one request per second per pod, which most backends handle comfortably.

Now add a two-tier sampling design. The DaemonSet agent samples probabilistically at 50%, halving volume cheaply, and forwards to a gateway collector that applies tail sampling: keep 100% of error traces and 10% of the rest. The gateway sees 10,000 spans/s. With `decision_wait: 10s` and an assumed 10 spans per trace, the gateway tracks roughly 10,000 traces at any moment (10,000 spans/s × 10 s ÷ 10 spans per trace). At 1 KB per span and 10 spans per trace, that is about 10 KB per tracked trace, or roughly 100 MB of buffered spans. A gateway with a 1 GB limit has room for that plus overhead — but the margin should be verified, not assumed.

The arithmetic is illustrative, not a benchmark. The point is that every knob has a memory cost that can be estimated before deployment. Substitute your own span size and spans-per-trace figures; both are measurable from existing trace data.

### Code: a minimal collector config with memory limiter and batching

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 400
    spike_limit_mib: 100
  batch:
    send_batch_size: 2000
    send_batch_max_size: 3000
    timeout: 5s

exporters:
  otlp:
    endpoint: backend.example.com:4317
    tls:
      insecure: false

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp]
```

The `memory_limiter` checks the collector's memory usage every `check_interval`. When usage approaches `limit_mib`, it begins refusing data; `spike_limit_mib` defines the headroom above the soft limit before refusal becomes aggressive. Placing it first ensures expensive processors never see data that would push memory past the limit. This is the single most effective protection against OOM kills, because it converts an uncontrolled crash into a controlled, observable drop.

### Routing with connectors

Suppose errors should go to one backend for alerting and everything else to a cheaper storage backend. A routing connector expresses that:

```yaml
connectors:
  routing:
    default_pipelines: [traces/cheap]
    table:
      - context: span
        condition: attributes["error"] == true
        pipelines: [traces/errors]

service:
  pipelines:
    traces/in:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [routing]
    traces/errors:
      receivers: [routing]
      processors: [batch]
      exporters: [otlp/errors]
    traces/cheap:
      receivers: [routing]
      processors: [batch]
      exporters: [otlp/cheap]
```

Spans with an `error` attribute go to the errors pipeline; everything else goes to the cheap pipeline. The `batch` processor appears in each downstream pipeline deliberately — batching after routing avoids sending large batches toward the wrong backend.

Note the isolation gap: the `memory_limiter` sits only in the input pipeline. The downstream pipelines inherit backpressure but have no limiter of their own. If the errors backend is slow, it backs up the routing connector and eventually the input pipeline. Full isolation requires separate receivers per destination or explicit exporter queue sizing, and that cost should be weighed against how much the two backends actually differ in reliability.

## What to monitor afterwards

The collector exposes its own metrics on a Prometheus endpoint (port 8888 by default). The metrics that matter most:

- `otelcol_receiver_refused_spans` and the metric/log equivalents. Any non-zero value means data is being dropped at the receiver. This is the canary for backpressure.
- `otelcol_processor_batch_batch_send_size`. The distribution of batch sizes. If it sits at the maximum constantly, the timeout may be too long or the batch size too small for the traffic.
- `otelcol_exporter_send_failed_spans`. Failed exports. A sudden rise means the backend is rejecting data or the network is failing.
- `otelcol_processor_memory_limiter_refused_spans`. Data refused by the memory limiter. Expected under load; a sustained high rate means the limit is too low or the pipeline too expensive.

Alert on `rate(otelcol_receiver_refused_spans[5m]) > 0` persisting for more than a few minutes. That gives time to react before data loss becomes significant. Also watch the collector's process memory against its limit: sustained proximity to the limit means either raise the limit or reduce pipeline cost, and the choice should be deliberate rather than a reaction to the next OOM.

## How to measure this yourself

None of the numbers above should be taken on faith. Each is measurable:

- **Span size.** Sample real spans from your backend and measure serialized size, or compute it from attribute counts and value lengths. A rough figure is enough for planning.
- **Spans per trace.** Query your trace store for the distribution of span counts per trace. The mean matters less than the tail, since tail sampling buffers by trace.
- **Pipeline throughput.** Run `telemetrygen` (part of the OpenTelemetry project) against a staging collector at your peak rate and watch `otelcol_receiver_refused_spans` and process memory. Raise load until refusals begin; that point is your measured ceiling.
- **Batch behavior.** Compare `otelcol_processor_batch_batch_send_size` distributions before and after changing `timeout` or `send_batch_size`. The difference tells you whether batches are being cut by size or by time.

A load test that reproduces peak traffic on a staging collector will surface backpressure and memory problems before production does, and it costs far less than an incident.

## The broader lesson: pipelines are capacity planning

A collector config is not just a wiring diagram; it is a capacity plan. Every processor adds latency and memory. Every exporter adds a network dependency. The collector scales with the resources given to it and the efficiency of the config, not automatically.

Teams often treat the collector as infrastructure that "just works" and only examine it when data goes missing. A better approach treats the pipeline as code: version it, review it, and load-test it against a synthetic workload matching peak traffic. The design principle is graceful degradation rather than peak capacity. Assume that at some point the backend will be slow or the traffic will spike. The pipeline should shed load in a controlled way — memory limiter, sampling, queue limits — rather than crash or silently drop everything.

## Frequently Asked Questions

### Why does the collector drop spans under load?

The usual cause is backpressure from a slow exporter or an expensive processor. When the pipeline cannot keep up, the receiver's queue fills and it refuses data. Confirm with `otelcol_receiver_refused_spans` and `otelcol_exporter_send_failed_spans`. A `memory_limiter` at the head of the pipeline and a smaller batch size are the first adjustments.

### Should probabilistic or tail-based sampling be used in the collector?

Probabilistic sampling is cheap and stateless but drops errors and slow traces at the same rate as everything else. Tail-based sampling keeps traces by policy but requires buffering and a two-tier setup so all spans for a trace reach the same instance. A common pattern is probabilistic sampling at the agent tier for volume reduction and tail sampling at a gateway tier for the traces that matter for debugging.

### How is telemetry routed to different backends?

Use a routing connector or a filter processor to split data by attribute, with separate pipelines per destination. The routing connector acts as an exporter in the source pipeline and a receiver in each destination pipeline. Each destination pipeline carries its own resource cost, and a slow backend can back up the shared source pipeline unless the paths are isolated.

### What is a safe batch size?

There is no universal value. Compute the memory cost: batch size times average item size times in-flight batches. For spans averaging 1 KB, a batch of 2,000 uses about 2 MB. Keeping total batching memory to a small fraction of the collector's memory limit, and setting `send_batch_max_size` to bound the worst case, is more reliable than copying a number from another deployment.

## Next step

Open the collector config and add a `memory_limiter` processor as the first processor in every pipeline. Set `limit_mib` to roughly 80% of the container's memory limit and `spike_limit_mib` to around 10%. Then scrape the collector's metrics endpoint and record `otelcol_receiver_refused_spans` and `otelcol_processor_memory_limiter_refused_spans` before and after the change. That single edit will not solve every pipeline problem, but it converts silent data loss into a visible, measurable signal.
