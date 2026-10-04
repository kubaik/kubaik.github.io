# Why APM Misses GPU-Bound LLM Incidents

## The blind spot in code-centric monitoring

Application Performance Monitoring (APM) is usually treated as the single source of truth for incidents. That works well when the workload is CPU-bound, the latency budget is tight, and the hot path runs entirely inside instrumented code. It stops working the moment a request leaves the process and enters a hardware accelerator or an external inference server.

The reason is structural, not a product defect. A tracer records the span between two points in your code: the call to `model.generate()` and its return. Everything the GPU does in between — kernel launch, memory allocation, batch scheduling, KV cache eviction — happens outside the tracer's visibility. The user's perceived latency, however, includes all of it plus the network round trip and browser rendering. The result is a dashboard that reports a healthy few hundred milliseconds while users experience seconds of waiting.

The standard advice — add tracing, define SLOs, monitor your endpoints — assumes the endpoint is a function you control. When the endpoint delegates to a model server that batches requests on a GPU, that assumption breaks. This article covers where the gap comes from, how to measure it, and how to decide how much telemetry you actually need.

## What the tracer can and cannot see

Consider a typical request path through a GPU-backed inference endpoint:

1. Your web process receives the HTTP request and writes the prompt to the model server.
2. The model server (for example, vLLM) places the request in a scheduling queue.
3. The scheduler forms a batch when enough requests are waiting or a timeout elapses.
4. The GPU executes prefill and decode kernels; KV cache memory is allocated and reused.
5. Tokens stream back through the server to your process and out to the client.

An auto-instrumented tracer covers step 1 and the tail of step 5. It does not cover steps 2 through 4, which is where most of the wall-clock time is spent under load. Typical contributors to end-to-end latency, in rough order of magnitude for a hosted model:

- Python and framework overhead: single-digit milliseconds.
- Queue wait in the scheduler: highly variable, from near zero to seconds when the GPU is saturated.
- GPU prefill and decode: hundreds of milliseconds to seconds, depending on prompt and output length.
- Network round trip to the user: tens to hundreds of milliseconds depending on geography.
- Browser rendering: tens to hundreds of milliseconds.

An APM span that captures only the first and last items will understate the total by however much time the middle items consume. Under light load that gap may be small. Under saturation it can be an order of magnitude.

### Failure mode: the green dashboard

A common pattern is an SLO defined on the APM span, for example p95 under 500 ms. The APM satisfies this because the span closes before the GPU work is accounted for. Meanwhile the GPU is at high memory utilization, the scheduler queue is growing, and users are timing out and retrying — which adds more load and makes the queue longer. The dashboard stays green through the entire degradation.

This is the central failure mode: the metric being alerted on is not causally connected to the user-visible symptom. Fixing it requires either measuring the true end-to-end path at the edge, or measuring the resource layer that causes the delay, or both.

## Instrumenting the layers APM cannot reach

The shift is from code-centric monitoring to resource-centric monitoring. That means collecting metrics from three additional places.

**GPU telemetry.** Utilization, memory usage, and memory bandwidth. The category of tool here is a GPU metrics exporter — a daemon that reads device counters and exposes them in a Prometheus-compatible format. Vendors ship such exporters; the exact metric names differ between them, so check your exporter's documentation rather than assuming a name.

**Inference server internals.** Most production model servers expose a metrics endpoint. For vLLM, setting the metrics option on the engine exposes Prometheus-format counters including queue depth and KV cache usage. Consult the version's documentation for the exact metric names, since these have changed across releases.

**Edge and client telemetry.** Real user monitoring, CDN logs, or a synthetic probe from representative regions. This is the only layer that captures the full path the user experiences.

A comparison of what each layer can answer:

| Layer | Question it answers | Typical source |
|---|---|---|
| Application APM | How long did my code wait on the model call? | Tracer spans |
| GPU exporter | Is the device saturated, and on what resource? | Device counters exposed as Prometheus metrics |
| Inference server | Is work queuing, and is the KV cache under pressure? | Server metrics endpoint |
| Edge / RUM | What did the user actually experience? | CDN logs, real user monitoring |

None of these layers alone is sufficient. The APM tells you your code is waiting; the GPU exporter tells you why; the edge telemetry tells you whether it mattered to users.

### How to measure the gap

Do this before adding any tooling, because it tells you whether you have a problem and how large it is.

1. Pick a representative endpoint and record the APM-reported p95 latency over a fixed window, for example one hour.
2. From the same window, record the client-observed p95 latency. If you have real user monitoring, use it. If not, run a synthetic probe from at least two regions and record the full request duration including token streaming.
3. Compare the two numbers. The difference is your instrumentation gap.
4. Correlate that gap with a GPU metric from the same window. If the gap widens when GPU memory or compute utilization rises, the GPU layer is the cause.

The instrumentation gap, not any single absolute latency figure, is the number that justifies the work. If the gap is small and stable, the existing stack is adequate.

## Worked example: reasoning about a latency spike

Suppose a synthetic probe reports a 2.0 s p95 end-to-end time while the APM span for the same endpoint reports 400 ms. Assume the probe and the APM cover the same requests.

- Unexplained time = 2.0 s − 0.4 s = 1.6 s.

Next, decompose the 1.6 s using the layers above. Suppose the GPU exporter shows memory utilization pinned near its ceiling and the inference server's queue-depth metric rising in the same window. A plausible causal chain:

1. Concurrent requests exceed the KV cache capacity configured for the model.
2. The scheduler cannot admit new requests without evicting cache entries for in-flight sequences.
3. Evicted sequences must be recomputed, which consumes GPU time that would otherwise serve new requests.
4. Queue depth grows, so each new request waits longer before its first token.
5. The APM span, which measures only the client-side call, does not include the queue wait.

The remediation levers, in order of least disruption:

- Reduce the maximum sequence length so each sequence consumes less KV cache. A smaller cache footprint per request means more concurrent requests fit.
- Lower the memory utilization target passed to the engine, leaving headroom so the scheduler is not forced to evict.
- Cap concurrency at the admission layer so the scheduler never sees more work than it can hold.
- Add capacity if the above do not restore headroom.

Note that the arithmetic above uses illustrative numbers. The method is what transfers: measure the gap, attribute it to a layer, then adjust the parameter that governs that layer's capacity.

### Verifying the fix

After changing a parameter, re-run the same measurement over a comparable window. The check is not "did the APM improve" — the APM may not move at all. The check is whether the instrumentation gap narrowed and whether the GPU metric that was saturated now has headroom. If the gap narrowed but the GPU metric is still saturated, you have moved the bottleneck rather than removed it.

## When the conventional approach is sufficient

Not every LLM deployment needs GPU telemetry. The existing APM stack is adequate when:

- The model is small enough and the hardware fast enough that generation completes in well under the APM's resolution, and there is no batching.
- Responses are precomputed and served from a cache, so the request path is ordinary HTTP.
- Inference runs on CPU and the framework's call stack is fully visible to the tracer.
- The endpoint is an embeddings call with no autoregressive generation loop.
- Traffic is low enough that the GPU is never saturated, so queue wait is negligible.

The deciding factor is not model size in parameters. It is whether the GPU is ever a contended resource and whether queue wait ever contributes meaningfully to user-visible latency. A large model on a lightly loaded dedicated device may need less telemetry than a small model serving heavy concurrent traffic.

A practical test: if the APM's p95 and the client-observed p95 track each other closely across your busiest periods, you do not have an instrumentation gap worth closing.

## A decision checklist

Work through these in order. Stop when the answer is "no."

1. Does the request path include autoregressive generation on an accelerator? If no, standard APM is likely sufficient.
2. Is the accelerator ever saturated during peak traffic? If no, the gap is probably small.
3. Does client-observed latency diverge from APM latency during peak? If no, you have no gap to close.
4. Is the divergence correlated with a GPU or queue metric? If yes, you have found the layer to instrument.
5. Can you act on that metric — is there a parameter, a concurrency limit, or a capacity change that would move it? If no, adding the metric produces alert fatigue without remediation.

## Adding the metrics: a minimal starting configuration

Start with the smallest set that can explain a latency gap, then expand only if it cannot.

The following Prometheus scrape configuration assumes a model server exposing metrics on port 8000 and a GPU exporter on port 9400. Adjust ports and paths to match your deployment.

```yaml
scrape_configs:
  - job_name: 'ai-endpoint'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
  - job_name: 'gpu-exporter'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:9400']
```

Alongside these, add one derived metric that bridges resource telemetry and user experience:

```
ai_endpoint_user_timeout_ratio = (client_reported_timeouts / total_requests) * 100
```

If this ratio rises while APM latency and GPU utilization are both flat, the instrumentation gap is real and neither of your existing views is capturing the cause.

For alerting, avoid thresholds copied from another system. Derive them: observe the metric during a known-good period, note its distribution, and set the alert above the normal range with enough duration to avoid firing on transient spikes. A threshold of 80% utilization is a starting hypothesis, not a universal rule.

## Common objections

**"This is too complex for our team."**
Start with one metric from one layer. A single GPU memory metric correlated against client-observed latency answers the first diagnostic question. Expand only when that metric fails to explain an incident.

**"Our APM supports OpenTelemetry, so it covers everything."**
OpenTelemetry propagates context across process boundaries, but it does not read device counters or scheduler internals. Instrumentation at the application layer cannot observe state that the application does not expose.

**"Our infrastructure team already watches GPU dashboards."**
Infrastructure dashboards answer "is the hardware healthy," not "is the user waiting." A device can be within its operating limits while the request queue in front of it is deep enough to cause timeouts. The two views must be correlated to be useful.

**"More exporters will bloat the stack."**
Each exporter is a process and a scrape target. The cost is real but bounded, and it is paid once. Compare it against the cost of an incident that is invisible to your current monitoring.

**"We already have an SLO on latency."**
Check what the SLO is measured against. If it is measured on an APM span that closes before the GPU work completes, the SLO is not measuring the thing users experience.

## Summary

APM tools are built around code paths they can instrument. GPU-accelerated inference spends most of its wall-clock time outside those paths, in scheduling queues, kernel execution, and memory management. The result is dashboards that stay green while users wait.

The remedy is not to replace APM but to add the layers it cannot see: GPU device metrics, inference server queue and cache metrics, and edge or real-user telemetry. The measurement that justifies the work is the gap between APM-reported latency and client-observed latency, correlated against a resource metric.

Start small, verify that each added metric can explain a real incident, and set thresholds from observed behavior rather than copied defaults.

## Do this in the next 30 minutes

Open your APM dashboard and note the p95 latency for one LLM-backed endpoint over the last 24 hours. Then run a synthetic request to that endpoint from a region far from your servers, timing the full response including token streaming. Subtract the APM figure from the measured figure. If the difference is more than a small fraction of the total, you have a quantified instrumentation gap — and a concrete number to bring to the next capacity or observability discussion.
===END===
