# CDC pipeline death spiral: one bad LLM call can crash…

A change data capture (CDC) pipeline that calls an LLM is a chain of queues. Each link — the replication slot, the connector, the broker, the consumer, the HTTP client, the inference server — has a bounded capacity. When one link accepts more work than the next link can drain, the excess does not disappear; it accumulates as lag, memory, or blocked threads. The failure mode is rarely a crash. It is a slow, cascading stall that starts at the LLM call and ends with a Postgres replication slot that will not advance.

This article covers why that happens, how to detect it, and where to place backpressure so a single large LLM response cannot take down the pipeline behind it.

## Why LLM bursts are different from normal traffic

Most CDC pipelines are designed around a predictable event rate. A row changes, a small event is emitted, a consumer processes it in milliseconds. The variance is low and the payloads are small.

LLM calls invert both assumptions:

- **Latency variance is enormous.** A short prompt may return in 200ms; a long one may take many seconds. A consumer that blocks on each call will see its throughput collapse when prompts get longer.
- **Output size is unbounded unless you bound it.** A model asked to "expand this into chunks" can legitimately return tens of thousands of tokens. That is a single logical event that produces a very large physical payload.
- **Concurrency is easy to create accidentally.** A retry loop, a fan-out, or a batch consumer can each multiply the number of simultaneous in-flight requests without anyone intending it.

The result is that the LLM layer acts as a variable-rate source feeding a fixed-rate sink. Without an explicit bound between them, the sink absorbs the mismatch until it fails.

## The failure mode: a stalled replication slot

The canonical symptom is a replication slot that stops advancing. The sequence usually looks like this:

1. A large LLM response arrives at the CDC consumer.
2. The consumer's HTTP client buffers the response, and the event loop or worker thread blocks while it does so.
3. Because the consumer is blocked, it stops acknowledging messages from the broker.
4. The broker's consumer lag grows, but more importantly, the connector's sink stops draining.
5. The connector stops confirming progress to Postgres.
6. Postgres retains WAL segments for the unconfirmed slot. `pg_replication_slots` shows a growing `restart_lsn` gap, and the slot's retained WAL grows.
7. If the stall is long enough, or the disk fills, the slot is invalidated and must be recreated — which typically means a fresh snapshot.

Note that the root cause is upstream of Postgres. Postgres is behaving correctly: it retains WAL because a consumer told it to. The fix belongs at the point where unbounded work enters the pipeline.

### What to instrument

You cannot fix what you cannot see. The minimum useful set of signals:

- **Replication lag in bytes and seconds.** Read `pg_current_wal_lsn()` against the slot's `confirmed_flush_lsn`, and export the difference. Alert on a sustained increase, not a single spike.
- **Slot retained WAL.** `pg_replication_slots` exposes the WAL retention per slot. A monotonic rise is the earliest reliable warning.
- **Consumer lag per partition.** From the broker's own metrics or the client library.
- **In-flight LLM requests and their duration.** A histogram of call duration and a gauge of concurrent calls. The gauge is the one that predicts stalls.
- **Response size distribution.** A histogram of output bytes or tokens. The tail of this distribution is what breaks buffers.

A useful exercise is to plot response size against consumer processing time on the same axis. If the two track each other closely, the consumer is doing work proportional to payload size and has no bound on either.

## Backpressure at each layer

Backpressure means refusing or delaying new work when downstream capacity is exhausted. It must exist at every link, because a single unbounded link is enough to propagate a stall.

### 1. Bound the generation itself

The cheapest place to limit burst size is at the inference server, before tokens are generated. Most serving stacks expose a maximum batch size or maximum sequence length. Setting these to values your downstream can actually absorb converts an unbounded problem into a bounded one.

```python
from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    max_model_len=32768,
    max_num_batched_tokens=8192,   # caps tokens processed per step
    enforce_eager=True,            # avoids graph capture buffering large batches
    disable_log_requests=True,
    tensor_parallel_size=2,
)

sampling_params = SamplingParams(
    max_tokens=2048,               # caps output length per request
    temperature=0.3,
    top_p=0.9,
)
```

Two distinct bounds are in play here. `max_num_batched_tokens` limits how much work the server schedules per step; `max_tokens` limits the length of any single response. Both matter. A server that schedules 8,192 tokens per step will still return a very long single response if `max_tokens` is unset.

Verify the effect by measuring, not by assuming. Log the output token count per request and confirm the 99th percentile is below your configured cap. If it is not, the cap is not being applied where you think it is.

### 2. Bound concurrency at the caller

A semaphore around the LLM call is the simplest effective control. It converts "however many requests happen to be in flight" into a fixed number.

```python
import asyncio
from prometheus_client import Histogram

llm_duration = Histogram(
    "llm_call_duration_seconds",
    "Duration of LLM calls",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
llm_tokens = Histogram(
    "llm_output_tokens",
    "Number of tokens in LLM output",
    buckets=[100, 1000, 5000, 10000, 20000, 50000],
)

_semaphore = asyncio.Semaphore(4)

async def safe_generate(text: str) -> str:
    async with _semaphore:
        with llm_duration.time():
            outputs = llm.generate(prompt=text, sampling_params=sampling_params)
            if not outputs:
                raise ValueError("no LLM output")
            response = outputs[0].outputs[0].text
            llm_tokens.observe(len(response.split()))
            return response
```

Choose the semaphore size from measurement: saturate the downstream, find the concurrency at which p99 latency starts to climb faster than throughput, and set the limit below that point. A value that is too high provides no protection; a value that is too low wastes capacity.

The `llm_output_tokens` histogram is what tells you whether the bound is real. If the tail keeps growing, the semaphore is limiting concurrency but not response size.

### 3. Bound the connector's batch and fetch sizes

On the CDC side, the connector's batch and fetch settings determine how much data the sink takes on at once. Large defaults are fine when events are small; they are dangerous when a single event can be megabytes.

```yaml
name: "postgres-connector"
connector.class: "io.debezium.connector.postgresql.PostgresConnector"
tasks.max: "4"
database.hostname: "postgres-primary.internal"
database.port: "5432"
database.user: "cdc_user"
database.dbname: "app_db"
database.server.name: "app"
table.include.list: "public.events"
slot.name: "cdc_slot"
plugin.name: "pgoutput"
snapshot.mode: "initial"

max.batch.size: 1000
max.poll.records: 500
fetch.max.bytes: 52428800   # 50 MB
poll.interval.ms: 100

topic.prefix: "cdc_events"
key.converter: "org.apache.kafka.connect.json.JsonConverter"
value.converter: "org.apache.kafka.connect.json.JsonConverter"
```

The exact parameter names and defaults vary by connector and version, so confirm them against the documentation for the version you run rather than copying values blindly. The principle is stable: cap how many records and how many bytes the sink will accept in one pass.

### 4. Bound the broker's per-message size

A broker that accepts arbitrarily large messages will eventually hand one to a consumer that cannot process it. Setting a maximum message size forces oversized payloads to fail at produce time, where they are cheap to handle, rather than at consume time, where they block a partition.

```bash
kafka-configs.sh --alter --topic cdc_events \
  --config max.message.bytes=10485760 \
  --bootstrap-server kafka-broker:9092
```

If a legitimate event exceeds the limit, the right answer is usually to store the payload elsewhere and pass a reference, not to raise the limit.

### 5. Bound the compute layer's concurrency

If the consumer runs on a serverless platform, its concurrency limit is a backpressure mechanism whether or not you treat it as one. Reaching the limit causes throttling, which is preferable to unbounded queue growth but still produces a stall if the source keeps producing.

```yaml
functions:
  cdc_processor:
    handler: handler.process
    memorySize: 1800
    timeout: 30
    reservedConcurrency: 200
```

Two cautions. First, reserved concurrency is a ceiling, not a target; setting it very high removes the protection entirely. Second, a throttled consumer still leaves the replication slot unconfirmed, so the slot retains WAL during the throttle window. Pair the limit with an alarm on throttles so the condition is visible.

## A worked example

Consider a pipeline with the following stated assumptions. These are illustrative round numbers chosen to make the arithmetic visible, not measurements.

- The replication slot retains WAL at a rate of 2 MB/s while the consumer is stalled.
- The consumer stalls for 30 seconds while buffering one large response.
- The consumer's normal drain rate is 5 MB/s.

During the stall, WAL accumulates at 2 MB/s for 30 seconds:

```
2 MB/s * 30 s = 60 MB retained
```

After the stall clears, the consumer must both process new events and catch up on the backlog. If it drains at 5 MB/s while new WAL arrives at 2 MB/s, the net drain is:

```
5 MB/s - 2 MB/s = 3 MB/s
```

Recovering 60 MB at a net 3 MB/s takes:

```
60 MB / 3 MB/s = 20 s
```

So a 30-second stall produces roughly 50 seconds of degraded operation. If stalls occur more often than every 50 seconds, the backlog never clears and the slot's retained WAL grows without bound.

This is the arithmetic that makes backpressure non-optional. The relevant question is not whether a stall happens but whether the recovery time is shorter than the interval between stalls. Measure both: the stall duration from your duration histogram, and the recovery rate from your lag metric.

## How to verify a fix

A configuration change is a hypothesis. To test it:

1. **Reproduce the condition deliberately.** Send a request at the maximum size your system claims to support and watch the lag metrics. If nothing moves, the bound is not where you think it is.
2. **Compare the response size distribution before and after.** The tail should be truncated at the configured cap. If the tail is unchanged, `max_tokens` is not being applied.
3. **Watch the concurrency gauge under load.** It should sit at the semaphore limit and not above it. A gauge that exceeds the limit means requests are bypassing the semaphore.
4. **Measure recovery time.** After a deliberate stall, record how long the lag takes to return to baseline. This is the number that determines your safety margin.
5. **Confirm the slot does not retain WAL indefinitely.** Query `pg_replication_slots` before and after; the retained WAL should return to a steady state.

## A decision checklist

Before adding an LLM call to a CDC pipeline, confirm each of the following:

- [ ] The inference server has both a per-request output cap and a per-step batch cap.
- [ ] The caller has a concurrency limit sized from measurement, not guesswork.
- [ ] The HTTP client's buffer limits are known and larger than the maximum response you will accept.
- [ ] The connector's batch and fetch sizes are set explicitly rather than left at defaults.
- [ ] The broker enforces a maximum message size.
- [ ] The compute layer's concurrency is bounded and its throttling is alarmed.
- [ ] Replication lag, retained WAL, consumer lag, in-flight requests, and response size are all exported as metrics.
- [ ] There is an alarm on retained WAL growth, not just on lag.
- [ ] There is a documented recovery procedure for an invalidated slot.

## FAQ

**Why not just make the consumer faster?**
Faster consumers raise the drain rate but do not bound the arrival rate. A sufficiently large burst will still exceed any fixed drain rate. Backpressure bounds the input; throughput improvements only delay the failure.

**Is a semaphore enough on its own?**
No. A semaphore limits how many calls are in flight, but a single call with an unbounded `max_tokens` can still return a payload larger than the consumer's buffer. You need both a concurrency bound and a size bound.

**What if the LLM output legitimately needs to be large?**
Store the large payload out of band and pass a reference through the CDC stream. The pipeline's job is to move change events reliably, not to carry multi-megabyte documents.

**How do I choose the semaphore size?**
Increase concurrency until p99 latency rises faster than throughput, then set the limit below that point. The exact number depends on the model, the hardware, and the request mix, so it must be measured on your system.

**Does raising the replication slot's WAL retention help?**
It delays the failure. The slot will retain more WAL before becoming a problem, but the underlying mismatch between arrival and drain rates is unchanged. Treat increased retention as a safety margin, not a fix.

## Do this in the next 30 minutes

Query your replication slot's retained WAL twice, 60 seconds apart, and export both values. If the second reading is higher than the first while your pipeline is nominally idle, you have an unconfirmed slot and an unbounded link somewhere upstream — start by checking the in-flight LLM request count and the response size histogram.
