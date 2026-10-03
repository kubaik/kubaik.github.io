# Observability for Long-Running LLM Agent Workflows

Agentic workflows break the assumptions that most observability stacks are built on. Request/response dashboards report CPU, memory and 5xx rates, but they say nothing about agent state, tool-call latency, retry storms, or how deep the queue feeding the agents has grown. When an agent fleet stalls, the CPU graph often looks normal while every correlation query returns nonsense.

This article walks through a reference stack for instrumenting long-running LLM agents: a Rust agent that emits structured events, a log-shipping sidecar with a bounded buffer, a small metrics service that exposes queue depth and agent lifecycle counters, and Prometheus plus Grafana on top. It ends with the failure modes that bite in practice and a checklist for deciding what to sample.

## The three properties that matter

Any observability stack for agentic workloads needs three things before it needs anything else:

1. **Millisecond-resolution event timestamps that survive log buffering.** Agent steps are short and interleaved; second-resolution timestamps make causal ordering impossible to reconstruct.
2. **Real-time queue depth and agent lifecycle counters.** Agents that block on a queue look healthy at the CPU level. Queue depth is the signal that actually predicts timeouts.
3. **Sampling that does not perturb the agent runtime.** Instrumentation that adds blocking I/O to the hot path changes the behavior you are trying to measure.

Traces, logs and metrics are all built on top of those three. Get them wrong and no dashboard will save you.

## Reference architecture

The stack described here runs on a single `t3.large` instance (2 vCPU, 8 GB) with Ubuntu 24.04 LTS and Docker Compose. The only external dependency is a managed search service for log storage; the same shape works with a self-hosted OpenSearch cluster or any log backend that accepts JSON documents.

Components:

- A Rust agent container (Tokio async runtime) that simulates an LLM agent calling a tool and emitting structured events to stdout.
- A Node sidecar that tails the agent's stdout, enriches events with a trace ID, and pushes them to the search backend through a pipeline processor.
- Prometheus and Grafana on the same Compose network, scraping the metrics service.
- A small Python service exposing `/metrics` and `/event` so Prometheus can scrape agent state without touching the agents themselves.

Pin your versions explicitly. Floating tags are the most common cause of "it worked yesterday" in this kind of stack.

## Step 1 — environment setup

1. Provision a fresh Ubuntu 24.04 LTS VM and install Docker.
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker  # refresh group membership without logout
   ```

2. Create the project directory and the `.env` file with your search endpoint and credentials.
   ```env
   OPENSEARCH_ENDPOINT=https://my-domain.us-east-1.aoss.amazonaws.com
   AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXXXXX
   AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   REGION=us-east-1
   ```

3. Bring the stack up in detached mode.
   ```bash
   docker compose up -d --build
   ```

4. Tail the sidecar logs to confirm events are flowing.
   ```bash
   docker compose logs -f sidecar
   ```
   Expected output resembles:
   ```
   2026-05-14T12:34:56.789Z agent=agent-1 trace=7f3a… event=tool_called tool=search duration_ms=142
   ```

5. Open Grafana at `http://localhost:3000` and add the search backend as a datasource. A "trace ID correlation missing" panel is expected at this stage; the trace ID is wired up in Step 2.

### Decisions that are hard to reverse

Two choices in this setup are expensive to change later:

- **Storage class and index size in the search backend.** Once data lands, changing the storage class or shrinking an index typically requires re-indexing. Pick the class based on your retention requirement and expected daily ingest, and confirm the per-GB-month price for your region before you commit.
- **Sampling rate.** Dropping events at ingest loses them permanently. If you later decide you needed the high-latency tails you sampled away, there is no way to recover them. Start conservative: sample only what you have measured you can afford to lose.

## Step 2 — the agent side: structured events

The agent emits one structured event per tool call. The `tracing` crate is a good fit because it gives millisecond-precision timestamps and automatic span correlation IDs.

1. Add dependencies to `agent/Cargo.toml`.
   ```toml
   [dependencies]
   tokio = { version = "1.40", features = ["full"] }
   tracing = "0.1"
   tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
   serde_json = "1.0"
   reqwest = { version = "0.11", features = ["json"] }
   ```

2. Replace `agent/src/main.rs` with this skeleton.
   ```rust
   use tracing::{info, instrument};
   use std::time::Instant;

   #[tokio::main]
   async fn main() {
       // Initialize tracing with JSON output and no ANSI escapes
       tracing_subscriber::fmt()
           .json()
           .with_target(false)
           .with_current_span(true)
           .with_ansi(false)
           .init();

       // Simulate an agent that calls a tool every 3 s
       loop {
           agent_step().await;
           tokio::time::sleep(std::time::Duration::from_secs(3)).await;
       }
   }

   #[instrument(skip_all, fields(trace_id, agent_id = "agent-1"))]
   async fn agent_step() {
       let start = Instant::now();

       let tool_result = call_tool("search", "kubernetes agentic latency").await;
       let duration = start.elapsed().as_millis();

       info!(
           event = "tool_called",
           tool = "search",
           duration_ms = duration,
           input = "kubernetes agentic latency",
           result = &tool_result,
           "agent step completed"
       );
   }

   async fn call_tool(tool: &str, query: &str) -> String {
       // Simulate an external API call
       tokio::time::sleep(std::time::Duration::from_millis(120)).await;
       format!("{}: results for '{}'", tool, query)
   }
   ```

3. Build the agent image and scale to 10 replicas.
   ```bash
   docker compose build agent
   docker compose up -d --scale agent=10
   ```

4. Verify events landed in the search backend.
   ```bash
   curl -XGET "$OPENSEARCH_ENDPOINT/logs-agent-*/_search?pretty" -H 'Content-Type: application/json' -d'
   {
     "size": 5,
     "query": { "match_all": {} },
     "sort": { "@timestamp": { "order": "desc" } }
   }'
   ```
   A representative document:
   ```json
   {
     "@timestamp": "2026-05-14T12:35:01.123Z",
     "trace_id": "7f3a1b4c",
     "agent_id": "agent-5",
     "event": "tool_called",
     "tool": "search",
     "duration_ms": 142
   }
   ```

### Why these design choices

- `tracing` is used instead of `log` because it injects a span ID into every event automatically. Without that, correlating events across agents requires manual plumbing that most teams never finish.
- Events are emitted as JSON so the ingestion pipeline can parse them without regex. Regex parsing at ingest is a common source of CPU cost and silent parse failures.
- The log level is set to INFO. Debug-level spans on a busy agent fleet will overwhelm any backend; raise the level per-agent when you need detail, not globally.

### The event schema

| Field        | Type     | Example value            | Why it matters                          |
|--------------|----------|--------------------------|-----------------------------------------|
| `@timestamp` | ISO8601  | 2026-05-14T12:35:01.123Z | Preserves ordering across buffering     |
| `trace_id`   | UUID     | 7f3a1b4c                 | Correlates agent steps across services  |
| `agent_id`   | string   | agent-5                  | Identifies which agent produced the log |
| `event`      | string   | tool_called              | Enables filtering in the backend        |
| `tool`       | string   | search                   | Helps debug tool-specific issues        |
| `duration_ms`| integer  | 142                      | Reveals performance regressions         |

If you skip `trace_id`, expect to spend days correlating logs that are tens of seconds out of sync. It is the single field that pays for itself fastest.

## Step 3 — the sidecar: bounded buffering and edge cases

The sidecar is where most of the operational difficulty lives. Three failure modes show up repeatedly:

- **Agent restarts lose the last buffered line.** Tailing the container's stdout file descriptor directly avoids the gap that Docker's log driver can introduce.
- **An unreachable search backend causes OOM.** Without a bounded queue, the sidecar accumulates events until the container is killed. A fixed-size in-memory queue with backpressure is the fix.
- **Oversized payloads are rejected by the pipeline.** Tool results can exceed the pipeline's per-record limit. Truncate at a known threshold and emit a separate event so the truncation is visible.

1. Replace `sidecar/index.js` with a version that handles all three.
   ```javascript
   const { Tail } = require('tail');
   const { Client } = require('@opensearch-project/opensearch');

   const client = new Client({ node: process.env.OPENSEARCH_ENDPOINT });
   const INDEX = 'logs-agent';

   // Bounded in-memory queue: drop oldest when full to protect the process
   const MAX_QUEUE = 100;
   const queue = [];
   let processing = false;

   // Tail the container's stdout directly
   const tail = new Tail('/proc/1/fd/1', { fromBeginning: false, follow: true });

   tail.on('line', (line) => {
     try {
       const obj = JSON.parse(line);
       obj['@timestamp'] = new Date().toISOString();
       if (queue.length >= MAX_QUEUE) {
         queue.shift(); // drop oldest; count this in a metric you actually monitor
       }
       queue.push(obj);
       if (!processing) flush();
     } catch (e) {
       console.error('Parse error', e);
     }
   });

   async function flush() {
     processing = true;
     while (queue.length > 0) {
       const batch = queue.splice(0, 100);
       try {
         await client.bulk({
           body: batch.flatMap((r) => [{ index: { _index: INDEX } }, r]),
         });
       } catch (err) {
         console.error('Bulk push failed', err);
         // Re-queue on failure, respecting the bound
         queue.unshift(...batch);
         while (queue.length > MAX_QUEUE) queue.pop();
         break;
       }
     }
     processing = false;
   }

   process.on('SIGTERM', async () => {
     tail.unwatch();
     await flush();
     process.exit(0);
   });
   ```

2. Install dependencies.
   ```bash
   cd sidecar && npm install @opensearch-project/opensearch tail
   ```

3. Mount the container's stdout in `docker-compose.yml` so the sidecar can tail it.
   ```yaml
   services:
     sidecar:
       build: ./sidecar
       volumes:
         - /proc/1/fd/1:/proc/1/fd/1:ro
       environment:
         - OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT}
         - REGION=${REGION}
       depends_on:
         - agent
   ```

4. Add payload truncation in the agent. Replace `call_tool` in `agent/src/main.rs`.
   ```rust
   async fn call_tool(tool: &str, query: &str) -> String {
       tokio::time::sleep(std::time::Duration::from_millis(120)).await;
       let result = format!("{}: results for '{}'", tool, query);
       const MAX_LEN: usize = 12_000;
       if result.len() > MAX_LEN {
           info!(
               event = "payload_truncated",
               original_len = result.len(),
               truncated_len = MAX_LEN,
               "payload too large"
           );
           result.chars().take(MAX_LEN).collect()
       } else {
           result
       }
   }
   ```

5. Restart the stack.
   ```bash
   docker compose down && docker compose up -d --build
   ```

### Verifying the edge cases

- Kill an agent container and confirm the sidecar reconnects within a couple of seconds.
- Point `OPENSEARCH_ENDPOINT` at an unroutable address (for example `http://192.0.2.1:9200`) and confirm the sidecar buffers without OOM. Watch RSS with `docker stats`.
- Push a payload larger than the truncation threshold and confirm a `payload_truncated` event appears in the backend.

The queue size is the hardest value to tune. A `t3.large` has 8 GB of RAM shared with the agent and metrics containers; a queue of 100 small JSON events is trivial, but 10,000 events of a few kilobytes each will not fit. Measure RSS under load before raising the bound.

## Step 4 — queue depth, latency, and alerts

Logs alone cannot answer the questions that matter during an incident:

- Which agents are stuck?
- How deep is the queue feeding the agents?
- What is the p99 latency of tool calls?

A small metrics service answers all three without instrumenting the agents directly.

1. Create `metrics/main.py` using FastAPI and the Prometheus client.
   ```python
   from fastapi import FastAPI
   from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
   from fastapi.responses import Response
   import redis.asyncio as redis

   app = FastAPI()

   AGENT_STEPS = Counter(
       'agent_steps_total',
       'Total agent steps completed',
       ['agent_id']
   )
   TOOL_LATENCY = Histogram(
       'agent_tool_latency_seconds',
       'Tool call latency in seconds',
       buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
       labelnames=['tool']
   )
   AGENT_QUEUE_DEPTH = Gauge(
       'agent_queue_depth',
       'Current queue length for agent work items'
   )

   redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

   @app.get('/metrics')
   async def metrics():
       AGENT_QUEUE_DEPTH.set(await redis_client.llen('agent_queue'))
       return Response(
           content=generate_latest(),
           media_type=CONTENT_TYPE_LATEST
       )

   @app.post('/event')
   async def ingest_event(body: dict):
       AGENT_STEPS.labels(body.get('agent_id', 'unknown')).inc()
       duration = body.get('duration_ms', 0) / 1000.0
       TOOL_LATENCY.labels(body.get('tool', 'unknown')).observe(duration)
       return {'status': 'ok'}
   ```

2. Pin the Python dependencies in `metrics/requirements.txt`.
   ```txt
   fastapi==0.115.0
   prometheus-client==0.21.0
   redis==5.0.1
   uvicorn==0.30.1
   ```

3. Have the agent post events to the metrics service. In `agent/src/main.rs`:
   ```rust
   use reqwest::Client;

   async fn send_event(client: &Client, body: serde_json::Value) -> Result<(), reqwest::Error> {
       client
           .post("http://metrics:8000/event")
           .json(&body)
           .send()
           .await?;
       Ok(())
   }
   ```

4. Add the metrics service to Compose and start it.
   ```bash
   docker compose up -d metrics
   ```

5. Configure Prometheus to scrape it every 15 seconds.
   ```yaml
   scrape_configs:
     - job_name: 'metrics'
       scrape_interval: 15s
       static_configs:
         - targets: ['metrics:8000']
   ```

6. Add an alert rule for sustained queue growth.
   ```yaml
   - alert: AgentQueueBackedUp
     expr: agent_queue_depth > 5000
     for: 2m
     labels:
       severity: critical
     annotations:
       summary: "Agent queue depth > 5000"
       description: "Queue depth is {{ $value }}; agents may time out."
   ```

### How to measure whether any of this is helping

Do not trust a vendor's or a blog post's latency numbers. Measure your own. The instrumentation above gives you everything you need:

- **Detection time.** Record the wall-clock time when a stuck agent first appears and when the alert fires. Compare before and after the queue-depth metric is live. The difference is your detection improvement.
- **Tool latency percentiles.** The histogram buckets above give you p50, p95 and p99 directly from Prometheus. Query `histogram_quantile(0.99, rate(agent_tool_latency_seconds_bucket[5m]))` and compare against a baseline captured before the change.
- **Ingest cost.** Watch CPU on the ingestion pipeline and bytes-per-day in the backend. Compare JSON ingestion against a regex-based alternative by running both on the same event stream for an hour.
- **Queue depth under load.** Load-test the agent fleet and graph queue depth over time. If it grows monotonically, the consumers are slower than the producers and no amount of dashboarding will fix that.

The point of measuring is to establish a baseline you can defend in a postmortem, not to produce a table.

## Failure modes to plan for

- **Timestamps rewritten by the sidecar.** Setting `@timestamp` at ingestion time replaces the agent's own timestamp. That is fine if the agent's clock is unreliable and the transport is fast, but it destroys ordering if the sidecar batches. Decide which timestamp is authoritative and document it.
- **Unbounded queues.** Any queue without a maximum will eventually exhaust memory. Drop oldest, drop newest, or block — but choose deliberately and emit a counter for drops.
- **Silent parse failures.** A single malformed line should not stop the pipeline. Log parse errors with the offending line and a counter, not just to stderr.
- **Alerts that fire constantly.** An alert that fires more than a couple of times a week gets muted. Tune the threshold against real traffic before you page anyone.
- **Sampling away the tail.** Sampling high-latency events is exactly the wrong thing to do when latency is the symptom. Sample the boring events, keep the tails.

## A decision checklist

Before you ship any agent observability stack, be able to answer these:

- What is the authoritative timestamp, and where is it set?
- What is the maximum size of every in-memory buffer, and what happens when it fills?
- Which metric tells you an agent is stuck, and how long after it gets stuck does that metric change?
- What is your sampling policy, and what evidence justifies it?
- What does a payload that exceeds the pipeline's limit do — truncate, drop, or fail?
- Which alert would page you at 3 a.m., and how often has it fired in the last week?
- What is the per-GB cost of your storage class, and how many days of retention does your budget buy?

## FAQ

**How do I run this on Kubernetes instead of Docker Compose?**

Split the stack into three Deployments: `agent`, `sidecar`, and `metrics`. Run the sidecar as a sidecar container in the agent pod so it can tail the same file descriptor. Replace the Prometheus static config with a ServiceMonitor. The storage-class decision is the same and is still the hard-to-reverse one.

**Why not use a log-only backend instead of a search backend?**

A log-only backend is often cheaper and faster for pure log search, but it usually lacks native trace correlation. If you need logs, traces and metrics in one query surface, a search backend is simpler than stitching two systems together. If you only need logs, the cheaper option wins.

**Can I sample events to reduce cost?**

Yes, but sample deliberately. Sampling high-latency events removes the evidence you need most during an incident. A common pattern is to keep all error and slow events and sample successful fast ones. Whatever you choose, record the sampling rate as a field on every event so downstream queries can correct for it.

**What if the ingestion pipeline drops support for my architecture?**

Pin the image tag explicitly and keep the pipeline configuration in version control. A migration is then a one-line image tag change plus a rebuild. Do not rely on floating tags for anything in the ingest path.

## What to do in the next 30 minutes

Open your current agent service, pick one tool call, and add a single structured log line that includes `trace_id`, `tool`, `duration_ms`, and an ISO-8601 timestamp with milliseconds. Deploy it to one replica. Then query your backend for that event and confirm the timestamp survived the round trip. If it did not, you have found the first thing to fix — and you have found it before an incident, not during one.
