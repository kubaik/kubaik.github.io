# Observability 2026: ditch the three pillars

The conventional advice on observability is incomplete. Treating logs, metrics, and traces as three separate pillars works in the simple case and breaks in a specific way under load. Here is the fuller picture, with the parts that survive scrutiny.

## The one-paragraph version

The "three pillars of observability" (logs, metrics, traces) is being supplemented—and in some architectures replaced—by a single, unified data model that treats every signal as telemetry: events, spans, profiles, and stack traces all land in the same store and get the same treatment. Three realities drive this: sampling becomes a correctness problem at high request rates, columnar storage for high-cardinality data has become cheap enough to keep more of it, and the only thing worse than "not enough data" is "too much noise that still doesn't answer the question." The new model is not new tooling so much as a shift in what you ask for and how you store it. A useful starting point is exporting every span with its parent context, its resource labels, and any attached stack trace as a single event. Anything less tends to break the first time you try to correlate a 200 ms spike in p99 latency with a GC pause that happened 30 s earlier.

## Why this concept confuses people

Most teams still reach for a metrics server plus a dashboard tool plus a trace backend and call it "observable." That stack made sense when the fastest signal you cared about was a one-second scrape, but it strains when the median request is 8 ms and the p99 is 120 ms. A common version of this trap shows up during a runtime migration—say, moving a checkout service from Node to Go—when traces suddenly look empty because the OpenTelemetry SDK on one side is dropping a large fraction of spans once its buffer fills. The three-pillar model also encourages treating logs, metrics, and traces as separate products: "ship logs to the log store, metrics to the metrics store, traces to the trace store." That separation creates query walls you hit the moment you need to ask, "Show me all events from this user session where the database latency exceeded 500 ms but the upstream service didn't time out."

## The mental model that makes it click

Think of your system as a single, append-only ledger of **events**. Every HTTP request, every function call, every GC cycle, every cache miss, every queue message becomes an event with:

- identity: trace_id, span_id, parent_id
- time: precise timestamp (ns)
- type: request, log, metric, profile, exception
- payload: the actual data (body, stack traces, resource labels)

Storing everything in the same place removes the ETL tax. Queries no longer need to fan out across three systems and reassemble the timeline. Instead you write one query that filters on time range, trace_id, and whatever labels you care about. The storage layer is not a time-series DB for metrics and a log DB for logs—it is a columnar store optimized for point lookups and range scans, with a lightweight indexing layer for trace relationships. Examples of that category include ClickHouse, or a managed columnar warehouse that supports wide, sparse rows.

## A concrete worked example

Instrument a simple Go service to emit unified telemetry, and send everything to a ClickHouse table called `telemetry_events`.

First, define the schema:

```sql
CREATE TABLE telemetry_events (
  event_time DateTime64(9),
  trace_id    String,
  span_id     String,
  parent_id   String,
  event_type  LowCardinality(String),
  service     LowCardinality(String),
  host        LowCardinality(String),
  labels      Map(String, String),
  body        String
)
ENGINE = MergeTree
ORDER BY (event_time, trace_id, span_id);
```

Next, instrument the service. Note that the OpenTelemetry Go API does not expose a generic `otel.Record` function; to attach arbitrary payloads to a span you use span attributes, span events, or a dedicated log bridge. The example below uses span events, which is the supported mechanism:

```go
package main

import (
  "context"
  "os"
  "time"

  "go.opentelemetry.io/otel"
  "go.opentelemetry.io/otel/attribute"
  "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
  "go.opentelemetry.io/otel/propagation"
  "go.opentelemetry.io/otel/sdk/resource"
  sdktrace "go.opentelemetry.io/otel/sdk/trace"
  semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
  "go.opentelemetry.io/otel/trace"
)

func initTracer() (*sdktrace.TracerProvider, error) {
  exp, err := otlptracehttp.New(
    context.Background(),
    otlptracehttp.WithEndpoint(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
    otlptracehttp.WithURLPath("/v1/traces"),
  )
  if err != nil {
    return nil, err
  }

  tp := sdktrace.NewTracerProvider(
    sdktrace.WithBatcher(exp),
    sdktrace.WithResource(resource.NewWithAttributes(
      semconv.SchemaURL,
      semconv.ServiceNameKey.String("checkout"),
      semconv.ServiceVersionKey.String("1.21.0"),
    )),
  )
  otel.SetTracerProvider(tp)
  otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
    propagation.TraceContext{},
    propagation.Baggage{},
  ))
  return tp, nil
}

func handler(ctx context.Context) {
  ctx, span := otel.Tracer("").Start(ctx, "checkout",
    trace.WithAttributes(semconv.HTTPMethodKey.String("POST")))
  defer span.End()

  time.Sleep(50 * time.Millisecond)

  // Attach a payload as a span event with attributes.
  stack := "...stack trace..."
  span.AddEvent("exception", trace.WithAttributes(
    attribute.String("method", "POST"),
    attribute.String("stack", stack),
  ))
}
```

On the query side, you can now ask a single question that would have required three tools before:

```sql
SELECT
  event_time,
  labels['method'] AS method,
  body
FROM telemetry_events
WHERE event_time > now() - INTERVAL 5 MINUTE
  AND trace_id = 'abc123'
  AND event_type IN ('request', 'exception')
ORDER BY event_time;
```

### How to measure whether this is actually faster

Do not trust a single latency number. Instrument it. On the query side, wrap the query in your client and record wall-clock time for a fixed set of trace IDs. On the storage side, use ClickHouse's `system.query_log` table, which records `query_duration_ms`, `read_rows`, and `memory_usage` for every query:

```sql
SELECT
  query_duration_ms,
  read_rows,
  formatReadableSize(memory_usage) AS mem
FROM system.query_log
WHERE query LIKE '%telemetry_events%'
  AND type = 'QueryFinish'
ORDER BY event_time DESC
LIMIT 20;
```

Then compare against the same question answered across three systems. The comparison that matters is not raw latency but *time from question to answer*, including the human time spent switching tools and reconciling timestamps. Measure that by logging a start timestamp when an engineer opens an investigation and an end timestamp when they close it, and compare the distribution before and after.

## How this connects to things you already know

You already use a columnar store for analytics, a search engine for full-text search, or a time-series DB for metrics. The unified telemetry model extends that pattern to every kind of signal. The mental shift is from "I need a pipeline for logs, a pipeline for metrics, a pipeline for traces" to "I need a single pipeline that can route every event to the right storage engine based on its shape and retention policy."

The cost curve flips in favor of self-hosting once you stop sampling, but the exact break-even depends on your volume, your query patterns, and your labor cost. A worked estimate: assume 500 GB/day of raw telemetry, Zstd plus delta encoding achieving roughly 10:1 compression on typical structured events (this is an illustrative figure—measure your own ratio), giving about 50 GB/day on disk, or 1.5 TB/month. At an illustrative $0.023/GB/month for infrequent-access object storage, that is roughly $35/month for cold storage. Add compute for ingestion and query, and the variable cost is dominated by how many queries you run, not how much you store. The honest comparison is against your current managed bill for the same retention and query volume—pull the invoice and divide by events ingested.

## Common misconceptions, corrected

**Misconception 1: "Unified telemetry means I have to rewrite all my dashboards."**
You can keep existing dashboards because most dashboard tools read from a query API. The change is under the hood: a metrics endpoint can be backed by a view over the unified table rather than a separate scrape target. Migration is incremental—one dashboard at a time, verifying that the new data source returns the same series.

**Misconception 2: "Storing everything will kill my storage budget."**
Compression ratios for structured telemetry are typically high, but they are not a constant. Measure yours: ingest a representative day, then compare `sum(data_compressed_bytes)` against `sum(data_uncompressed_bytes)` in `system.parts`. If the ratio is poor, the usual cause is a high-cardinality column in the sort key or a `String` column holding JSON that should be decomposed into typed columns.

**Misconception 3: "I'll lose the ability to alert on metrics."**
You gain the ability to alert on any field in the event. Instead of alerting on a counter named `http_requests_total`, you alert on events filtered by `event_type='metric' AND labels['status']='5xx'`. Whether your alerting system can consume that depends on the system; many support a generic webhook or a query-backed rule, which is the practical integration point.

## The advanced version, once the basics are solid

Once you are comfortable with a single telemetry table, the next step is to add **profiling telemetry** and **eBPF events** to the same store.

Profiling telemetry in Go can emit pprof data as events. The `runtime.Stack` call below captures all goroutine stacks, which is a coarse but dependency-free starting point:

```go
ticker := time.NewTicker(30 * time.Second)
defer ticker.Stop()

for range ticker.C {
  buf := make([]byte, 1<<20)
  n := runtime.Stack(buf, true)
  profile := buf[:n]

  ctx, span := otel.Tracer("").Start(context.Background(), "profile")
  span.AddEvent("profile", trace.WithAttributes(
    attribute.String("body", string(profile)),
  ))
  span.End()
}
```

eBPF events (syscalls, network flows, GC pressure) can be streamed into the same pipeline. The exact tooling varies; the pattern is a userspace collector that reads a perf ring buffer and forwards OTLP:

```sh
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_* { printf("%s %d\n", probe, args->pid); }' \
  | your-otlp-forwarder --endpoint http://collector:4318
```

The combined dataset lets you correlate a latency spike with a GC pause, a syscall storm, and a burst of 5xx responses in one query. The value is not a specific detection-time number—it is that the correlation is expressible at all without moving data between systems.

## Failure modes to expect

**The Kafka lag mirage.** A payment service moves to a streaming framework with exactly-once semantics and the three-pillar view reports everything is fine: low consumer lag, no error logs, clean traces. Then a latency spike hits. The root cause is often that consumer group lag is sampled coarsely while the actual lag oscillates in short bursts. The unified model helps because you can instrument the client to emit an offset event on every partition reassignment, with lag as a label. A query joining offset events to request events reveals the pattern. Usual fixes: increase partition count and tune `max.poll.interval.ms`.

**The file descriptor leak.** A service in Kubernetes starts crashing every few hours. No error logs, no GC pressure, no latency spike. The culprit is often a dependency that opens a socket or file and never closes it; each leak consumes a descriptor until the container hits its limit. A unified store catches this if you export a gauge for open descriptors and alert on its slope, not its level. The fix is to patch the dependency and set an explicit descriptor limit.

**The merge storm.** A columnar store with many small partitions will spend CPU on background merges instead of queries. This typically appears after someone adds a high-cardinality label to the sort key or partition key. The symptom is query timeouts that correlate with merge activity, visible in `system.merges`. The fix is to reconsider the partition key—usually coarser, such as monthly rather than daily—and to avoid high-cardinality columns in the sort key.

## Quick reference

| Concept | Three-pillar model | Unified event model |
|---|---|---|
| Data shape | Separate schemas per signal | Single table: `event_time`, `trace_id`, `span_id`, `event_type`, `body` |
| Storage engine | Time-series DB, log DB, trace store | Columnar store |
| Sampling | Often on by default in SDKs | Configurable; 100% for low volume, tail-based for high |
| Cost model | Per-host or per-GB ingestion plus query | Storage plus compute, both measurable |
| Query latency | Cross-service fan-out | Single-table scan |
| Cardinality limit | Low for labeled metrics | High, bounded by memory and merge cost |

## Frequently Asked Questions

**How do I migrate without losing dashboards?**
Add an OTLP endpoint to your existing pipeline, point SDKs at a collector, and configure the collector to dual-write: metrics to the existing metrics store, traces to the new store. Dashboards keep working because the old store still serves them. Migrate one dashboard at a time, verifying parity before switching.

**What retention policy makes sense for a 500 GB/day stream?**
There is no universal answer, but a defensible starting point is: keep raw events hot for the window that covers most of your investigations (commonly 14–30 days), then tier to object storage for a longer window. Compute the cost from your own compression ratio and your provider's storage tiers rather than from a quoted figure. The decision rule is whether rehydrating from cold storage is fast enough for the incidents that actually need it—test that path before you rely on it.

**Isn't storing full stack traces too expensive?**
Only if you store them naively. Measure the compressed size of a representative trace rather than assuming. The real cost is usually CPU during compression, which you can offload to a dedicated ingestion tier that returns pre-compressed blocks. If CPU is the bottleneck, sample stack traces rather than dropping them entirely.

**Can I still use PromQL or a trace query language?**
Often yes. A metrics query API can be backed by an adapter over the unified table, and many trace backends support a pluggable storage layer. The practical approach is to materialize a view that exposes the shape the query language expects:

```sql
CREATE MATERIALIZED VIEW spans_view ENGINE = ReplacingMergeTree
ORDER BY (trace_id, span_id) AS
SELECT
  trace_id,
  span_id,
  parent_id,
  event_time,
  service,
  labels
FROM telemetry_events
WHERE event_type = 'span'
```

## Integration example: collector configuration

A collector configuration that routes traces, metrics, and logs to a single table while forwarding metrics to an existing metrics endpoint. The exporter name and options depend on your collector distribution; check that the ClickHouse exporter is present before relying on it.

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:
  transform/set_resource_attributes:
    log_statements:
      - context: resource
        statements:
          - set(attributes["service.version"], resource.attributes["service.version"])
          - set(attributes["deployment.environment"], resource.attributes["deployment.environment"])

exporters:
  clickhouse:
    endpoint: tcp://clickhouse:9000
    database: observability
    table: telemetry_events
    timeout: 5s
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, transform/set_resource_attributes]
      exporters: [clickhouse]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [clickhouse, prometheus]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [clickhouse]
```

Run it with the container image for your chosen distribution:

```sh
docker run --rm -it \
  -v $(pwd)/config.yaml:/etc/otel/config.yaml \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 8889:8889 \
  your-collector-image:latest \
  --config=/etc/otel/config.yaml
```

## A query that correlates GC pressure with latency

Once GC events and request events share a table, a self-join on `trace_id` expresses the correlation directly. This is the kind of question that is awkward across three systems and trivial in one:

```sql
SELECT
  te1.event_time,
  te1.labels['gc_count'] AS gc_count,
  te2.event_time,
  te2.labels['http.method'] AS method,
  (te2.event_time - te1.event_time) AS gc_to_request_ms
FROM telemetry_events te1
JOIN telemetry_events te2
  ON te1.trace_id = te2.trace_id
WHERE te1.event_type = 'gc_pressure'
  AND te2.event_type = 'request'
  AND te2.event_time > te1.event_time
  AND te2.event_time < te1.event_time + INTERVAL 5 SECOND
ORDER BY gc_to_request_ms DESC
LIMIT 10;
```

A caveat worth stating plainly: self-joins on a large event table are expensive. If you run this often, materialize a narrow projection containing only the fields the join needs, and keep the wide payload columns out of it.

## Decision checklist

Before committing to a unified store, answer these:

- What is your actual ingestion volume, in events per second and bytes per day, measured rather than estimated?
- What compression ratio do you observe on a representative day, measured from `system.parts`?
- Which investigations currently require switching between tools, and how long do they take end to end?
- What is the retention window that covers most investigations, and what is the cost of each tier beyond it?
- Can your alerting system query the new store directly, or does it need an adapter?
- Who operates the store during an incident, and is that person on call for the same rotation as the services it observes?

If the last question has no answer, the migration will fail for organizational reasons long before it fails for technical ones.

## Now do this

Create the `telemetry_events` table from the worked example in a scratch ClickHouse instance, insert a few hundred synthetic rows with a script, and run the single-table query above against them. Then run the same query with `EXPLAIN` and read the plan. That thirty-minute exercise tells you more about whether this model fits your workload than any comparison table, because it uses your data shape, your cardinality, and your query patterns.
