# How we cut production incidents by 70% with real-time APM

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2023, our SaaS platform handled 2.3 million requests per day across three regions. We had Prometheus, Grafana, and a custom metrics dashboard, but still averaged 12 production incidents per month. Each incident meant 45 minutes of mean time to detect (MTTD) and 2 hours of mean time to resolve (MTTR). The trigger was usually an upstream dependency timing out: our payment service would hang for 6 seconds, our cache would evict critical keys, or our database would spike to 100% CPU. We knew *something* was wrong, but the alerts arrived after customers had already complained.

I got this wrong at first by assuming more dashboards would help. I set up 37 new panels tracking everything from GC pauses to DNS latency. What I missed was that the dashboards only showed *what* had happened, not *why*. Worse, the noise from 120 alert rules meant we ignored 89% of them. Then I noticed something surprising: our logging pipeline, Fluentd 1.16, was dropping 14% of log lines under load because we hadn’t tuned the buffer size. Once I fixed that, the noise became louder, not smarter.

The real problem wasn’t lack of data—it was lack of context. We needed to know not just that response time spiked to 800ms, but *which* upstream call caused the spike and *which* code path triggered it. We also needed to catch anomalies *before* they became outages, not after.

The key takeaway here is that traditional monitoring shows symptoms, not root causes, and more dashboards increase noise without improving signal.

## What we tried first and why it didn’t work

Our first attempt was to wire every upstream HTTP call through OpenTelemetry 1.32 with automatic instrumentation. We shipped 150 lines of Python middleware that wrapped requests and emitted spans. Within a week we had 12 GB of trace data per day—enough to fill a small SSD. We stored traces in Jaeger 1.47 and queried them with Grafana. The latency of a typical checkout flow jumped from 420ms to 480ms because of the extra serialization overhead.

Then we tried to reduce the volume by sampling. We set head-based sampling at 10%, but we still got 1.2 GB/day. We tried tail-based sampling with a custom OTL collector running on Kubernetes, but the collector itself consumed 3 vCPUs and 2 GB RAM, and it introduced 120ms of extra latency on cold starts. The worst failure was during Black Friday week: our collector hit a memory leak, restarted 47 times in 2 hours, and dropped 34% of traces right when we needed them most.

We also tried anomaly detection with Prometheus’ built-in Holt-Winters forecasting. It flagged 42 anomalies in production, but only 3 were real incidents. The false positives were caused by innocent traffic spikes from our marketing campaigns. Tuning the sensitivity to reduce false alarms meant missing real latency spikes that lasted only 30 seconds. We ended up disabling the detector entirely.

The key takeaway here is that blindly instrumenting everything increases cost and latency, and naive sampling or anomaly detection fails under real-world traffic patterns.

## The approach that worked

We pivoted to a strategy we called *critical path tracing with SLO-based alerting*. Instead of tracing every request, we traced only the slowest 5% of requests per service and added distributed context only where it mattered. We used OpenTelemetry’s tail-based sampler configured with a custom latency filter: keep the slowest 1 in 20 requests, but drop anything under 100ms. This reduced trace volume by 94% without losing critical context.

Next, we shifted from metric-based alerting to error-budget-based alerting. We defined an SLO of 99.9% of requests completing in under 500ms for our checkout flow. We used the SLO burn rate to trigger alerts: if the error budget burned 5% in 5 minutes, we triggered a page. We implemented this in Prometheus using the SLI exporter from Google’s SRE workbook, version 0.5.0. The change cut pages by 60% because we only alerted when we were actually violating the user-visible contract.

Finally, we added *upstream dependency tracing*. When a service detected a slow downstream call, it injected an OpenTelemetry baggage item containing the caller’s trace context. This let us stitch traces across service boundaries without forcing every upstream to instrument itself. We called this pattern *reverse propagation* and it surprised me: even services we didn’t control began showing up in our traces because their clients were instrumented.

The key takeaway here is that focusing instrumentation on the critical path and using SLO burn rates reduces noise and surfaces only actionable signals.

## Implementation details

We instrumented our Python 3.11 backend using the OpenTelemetry Python SDK 1.22. We avoided auto-instrumentation for libraries like Django because it added 20ms of overhead per request. Instead, we manually wrapped only the critical paths: checkout, refund, and subscription billing.

Here’s the instrumentation code for a checkout endpoint:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Configure tracer provider once at startup
trace.set_tracer_provider(TracerProvider())
exporter = OTLPSpanExporter(
    endpoint="https://otlp.nr-data.net:4317",
    headers={"api-key": os.getenv("NEW_RELIC_LICENSE_KEY")},
)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

# Wrap critical checkout function
tracer = trace.get_tracer(__name__)

def checkout(user_id, cart, payment_method):
    with tracer.start_as_current_span("checkout") as span:
        span.set_attribute("user.id", user_id)
        # Manually add upstream context when calling payment service
        ctx = TraceContextTextMapPropagator().extract(carrier=payment_headers)
        tracer = trace.get_tracer(__name__, context=ctx)
        with tracer.start_as_current_span("payment_charge"):
            result = charge_payment(payment_method)
        if result.status == "failed":
            span.record_exception(ValueError("Payment failed"))
            span.set_status(trace.Status(trace.StatusCode.ERROR))
        return result
```

On the frontend, we used New Relic Browser 1272 to capture end-to-end traces from browser to backend. We configured it to only capture traces longer than 3 seconds to keep volume low. We also used its *session replay* feature to correlate slow traces with actual user sessions, which helped us see that 18% of slowdowns happened during ad-blocker initialization.

For alerting, we ran a custom Prometheus alert manager 0.26 with SLO-based rules:

```yaml
# alert.rules.yml
- alert: HighCheckoutLatency
  expr: |
    (sum(rate(http_request_duration_seconds_bucket{service="checkout", le="0.5"}[5m]))
     by (job)
     / sum(rate(http_request_duration_seconds_count{service="checkout"}[5m]))
     by (job))
    < 0.999
  for: 5m
  labels:
    severity: page
  annotations:
    summary: "Checkout SLO burned 5% in 5 minutes"
```

We deployed the collector as a sidecar in Kubernetes using the OpenTelemetry Operator 0.89. The sidecar shared a volume with the app for the batch span processor, reducing serialization latency by 15ms. We set the queue size to 2048 and the export timeout to 30s to handle spikes. Early on, we hit a memory leak in the collector when traces exceeded 1MB. We fixed it by upgrading to version 0.91 and setting `max_traces_per_span` to 100.

The key takeaway here is that targeted instrumentation, smart sampling, and SLO-based alerting require careful tuning to avoid turning observability into overhead.

## Results — the numbers before and after

After six weeks, we saw clear improvements. The trace volume dropped from 12 GB/day to 700 MB/day, a 94% reduction. End-to-end latency for the checkout flow fell from 480ms to 430ms, a 10% improvement. More importantly, the mean time to detect an incident dropped from 45 minutes to 3 minutes—an improvement of 93%. The mean time to resolve also fell from 2 hours to 45 minutes, a 62% improvement.

During a simulated Black Friday load test with 10,000 concurrent users, our system handled 8,200 requests per second with 99.95% success, and the p99 latency stayed under 450ms. Before the change, the same load caused 60% of requests to time out and the p99 latency hit 2.1 seconds.

We also cut our incident count by 70%. In the first month after rollout, we had only 3 incidents, down from 12. Each incident now involved clear traces showing the root cause: a cache stampede in the product catalog, a misconfigured database pool, or a thundering herd on a new product launch.

What surprised me most was that the biggest win wasn’t faster resolution—it was fewer pages. The team stopped dreading the alert channel because 92% of pages now pointed to real issues, not false alarms.

The key takeaway here is that real-time, context-rich APM reduces incident volume, detection time, and resolution time while improving overall latency and reliability.

## What we'd do differently

We would not have shipped the OpenTelemetry Python auto-instrumentation by default. It added 20ms per request and bloated our traces with noise from Django middleware and Jinja templates. Instead, we would have started with manual instrumentation only on the critical paths and added auto-instrumentation later, once we knew what we needed.

We also would have avoided storing full traces in Jaeger. Jaeger 1.47 scaled poorly beyond 50 million spans per day. We wasted 3 engineer-weeks tuning Cassandra compaction before switching to New Relic’s managed backend, which handled 200 million spans per day without tuning.

Finally, we would have set up SLO burn-rate alerting earlier. We spent too long tweaking PromQL queries to detect anomalies. Once we moved to error-budget-based alerts, we reduced false positives by 84% and cut our on-call rotation load in half.

The key takeaway here is that starting small, avoiding auto-instrumentation overhead, and prioritizing SLO-based alerting leads to faster, more reliable outcomes.

## The broader lesson

The core principle is this: **instrumentation should follow user impact, not architecture.** Every line of code you add to emit a span or log a metric has a cost. That cost isn’t just CPU and memory—it’s cognitive load, storage bills, and alert fatigue. The only way to justify that cost is to ensure each byte of telemetry directly improves your ability to detect and resolve incidents that affect users.

A corollary is that **context is more valuable than completeness.** A single trace showing a 5-second upstream call is worth 10,000 metrics charting CPU and memory. Context lets you see the *why*, not just the *what*. And context is what turns a page from "something is wrong" into "payment service is slow because the Redis cluster is evicting keys too aggressively."

Finally, **alerts should reflect user pain, not system noise.** If your alerts are based on CPU > 90% or memory > 80%, you’re alerting on symptoms, not outcomes. Instead, define SLOs based on user-facing metrics like latency or error rate, and alert when those SLOs are at risk. This flips the script from "system is broken" to "user experience is at risk."

The key takeaway here is that observability isn’t about collecting data—it’s about collecting the right data to make the right decisions, fast.

## How to apply this to your situation

Start by defining your *critical user journeys*. For an e-commerce site, these are checkout, login, and product search. For a SaaS app, it’s onboarding, billing, and data export. Pick one journey and instrument it end-to-end: frontend, API, database, and any third-party services.

Next, set a realistic SLO for that journey. If your users expect sub-second response, target p95 < 1s and p99 < 2s with 99.9% availability. Use a service like Nobl9 or Google’s SLO toolkit to calculate the error budget.

Then, choose one APM tool that supports distributed tracing and SLO-based alerting. We used New Relic because it handled high-volume traces with low overhead, but Datadog APM 1.72 or Honeycomb 1.89 are also solid choices. Avoid self-hosting Jaeger or Zipkin unless you have dedicated SREs—scaling them is a full-time job.

Finally, implement a *feedback loop*. After each incident, ask: did our tracing show the root cause? If not, add instrumentation to that path. Over four weeks, you’ll have a minimal but effective observability layer that prevents future incidents.

The key takeaway here is that you don’t need perfect instrumentation—you need *just enough* to catch incidents before users do.

To apply this immediately, run this command to check your current trace volume:

```bash
# Assuming OpenTelemetry collector is running
curl -s http://localhost:8888/metrics | grep otelcol_receiver_accepted_spans_total
```

If the metric is above 10,000 spans per minute, you’re likely collecting too much. Reduce the sample rate or filter by latency.

## Resources that helped

1. [OpenTelemetry Python SDK docs](https://opentelemetry.io/docs/instrumentation/python/) — the manual instrumentation guide saved us from auto-instrumentation overhead.
2. [SLOs in Practice by Nobl9](https://nobl9.com/resources/slo-in-practice) — the best practical guide to error-budget alerting we found.
3. [New Relic’s trace sampling guide](https://docs.newrelic.com/docs/apm/transactions/trace-sampling/best-practices-trace-sampling/) — taught us how to reduce trace volume without losing signal.
4. [Google’s SRE workbook, chapter 6](https://sre.google/workbook/alerting-on-slos/) — showed us how to turn SLOs into actionable alerts.
5. ["Observability Engineering" by Charity Majors et al.](https://www.oreilly.com/library/view/observability-engineering/9781492076438/) — the chapter on distributed tracing changed how we thought about context.


## Frequently Asked Questions

How do I fix high cardinality in traces causing high storage costs?

Reduce cardinality by filtering attributes before they’re exported. Use OpenTelemetry processors to drop or hash high-cardinality fields like user IDs or request IDs. For example, in your collector config, use the `attributes` processor to remove `http.request.header.x-api-key` before exporting to your backend. This cuts storage by 40–60% with no loss of signal for incident detection.

What is the difference between head-based and tail-based sampling, and which should I use?

Head-based sampling decides to keep or drop a trace when it starts, using fixed rules like sample 10% of traces. Tail-based sampling waits until the trace ends, then applies complex rules like "keep if any span duration > 500ms." Use head-based sampling for low-overhead filtering, but switch to tail-based when you need to catch rare, long-lived issues. We switched to tail-based after head-based missed a 2-second slowdown that happened only twice per hour.

Why does my APM tool slow down my application under load?

APM tools add serialization, network I/O, and GC pressure. Each span requires memory allocation and serialization to JSON or protobuf. Under load, these allocations trigger GC pauses and network buffers fill up, adding latency. We saw 20ms of overhead from OpenTelemetry Python SDK under 1000 RPS. To mitigate, batch exports, use async spans, and run the collector as a sidecar to share resources.

How do I correlate frontend and backend traces when users block third-party scripts?

Use server-side trace context injection. When the backend detects a slow downstream call, it injects an `X-Trace-Context` header into the response. The frontend reads this header and attaches it to its own traces. This works even if third-party scripts are blocked, because the header comes from your server. We used this to trace 87% of slow frontend experiences back to backend latency, even when ad blockers were active.

| Tool | Version | Use Case | Latency Overhead | Storage Cost (per 10M spans) |
|------|--------|----------|------------------|-------------------------------|
| Jaeger | 1.47 | Self-hosted tracing | 15ms | $450/month (Cassandra) |
| New Relic | 1272 | Managed APM + traces | 8ms | $180/month |
| Datadog APM | 1.72 | Managed traces + logs | 12ms | $220/month |
| Honeycomb | 1.89 | High-cardinality events | 5ms | $150/month |

The table shows that managed APM tools reduce operational overhead and latency compared to self-hosted Jaeger, especially at scale.