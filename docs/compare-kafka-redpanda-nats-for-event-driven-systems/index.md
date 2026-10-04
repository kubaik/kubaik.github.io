# Compare Kafka, Redpanda, NATS for event-driven systems

## Why broker comparisons go wrong

Most broker comparisons list features. Feature lists are nearly identical across the three systems that dominate event-driven architecture discussions: producers, topics/streams, brokers, consumers, replay, retention. The differences that actually decide your architecture live in defaults, in what happens when a node restarts, and in what the system does when a consumer falls behind.

A second failure mode is benchmarking the wrong thing. A cluster tested with 1 KB messages at 1,000 events/sec tells you almost nothing about a pipeline pushing 4 KB events at 20,000/sec with a 50 ms end-to-end budget. The broker that wins an idle benchmark is often not the broker that holds its p99 under load.

This article covers three systems:

- **Apache Kafka** — a partitioned commit log with replication, widely deployed for durable event streaming.
- **Redpanda** — a Kafka-API-compatible broker written in C++, with a built-in tiered-storage feature that offloads older log segments to object storage.
- **NATS with JetStream** — a lightweight messaging system where JetStream adds streams, consumers, and optional persistence on top of core NATS pub/sub.

The goal is not to crown a winner. It is to give you the vocabulary and the measurement plan to pick correctly for your workload, and to recognize the three most common misconfiguration patterns.

## The real axis of comparison: durability vs latency vs cost

Every broker here makes the same fundamental trade. To acknowledge a write durably, you must wait for it to reach stable storage and, in replicated systems, for enough replicas to confirm. To acknowledge it quickly, you must acknowledge before that work completes. There is no configuration that gives you both maximum durability and minimum latency; there is only a dial.

Kafka exposes this dial through `acks` and `min.insync.replicas`. Redpanda exposes it through the same Kafka producer settings plus its own tiered-storage and fsync behavior. NATS JetStream exposes it through whether you use memory or file storage and whether file writes are synced to disk.

Three consequences follow, and they explain most "the broker lost my messages" and "the broker is too slow" reports:

1. **Acknowledgment semantics are a producer-side decision.** If the producer uses `acks=1`, the broker can acknowledge before all replicas have the data. A leader failure at the wrong moment loses the write. This is not a broker bug.
2. **Persistence is often asynchronous by default.** Write-back caches and memory-mapped files mean data can be acknowledged and then lost on a hard restart. The window is usually small, but it is not zero.
3. **Tiered storage trades read latency for disk cost.** Moving cold segments to object storage reduces local disk footprint; the first read of a cold segment pays a fetch.

The rest of this article works through each consequence with a concrete failure mode and a fix.

## Failure mode 1: Kafka producer batching that violates a latency SLO

**Symptom.** End-to-end latency sits well above the target, but broker CPU, disk, and network look healthy. Producer-side timeouts and retries appear intermittently.

**Root cause.** Kafka's producer buffers records to form batches. Two settings govern when a batch is sent: `batch.size` (bytes) and `linger.ms` (time). A batch is sent when either threshold is reached. If you publish small records at a moderate rate, the batch never fills, so every record waits out the full `linger.ms` before being sent. The documented default for `linger.ms` is `0`, which means "send as soon as possible" — but many production configurations raise it deliberately for throughput, and a raised value is then inherited by every topic on that producer. If someone set `linger.ms=30000` for a bulk ingestion job and the same producer config is reused for a real-time topic, every real-time event waits up to 30 seconds.

**A second, separate symptom** is `RecordTooLargeException`. That exception means the batch (not the individual record) exceeded `max.request.size` or the broker's `message.max.bytes`. Raising `batch.size` without raising `max.request.size` produces exactly this error.

**Fix.** Set `linger.ms` to a value derived from your latency budget, not copied from a throughput-oriented template. A reasonable starting point is 5 ms, which still allows batching under load but bounds the added latency. Keep `batch.size` large enough to hold a useful batch, and raise `max.request.size` to match so you do not trigger size exceptions.

```java
Properties props = new Properties();
props.put("bootstrap.servers", "kafka1:9092");

// Durability: "all" waits for in-sync replicas; "1" waits only for the leader.
// Choose based on whether you can tolerate losing the last writes on leader failure.
props.put("acks", "all");

// Latency: bound how long a record may sit in the buffer waiting for a batch.
props.put("linger.ms", "5");

// Throughput: allow larger batches, and raise the request limit to match.
props.put("batch.size", "262144");        // 256 KB
props.put("max.request.size", "1048576"); // 1 MB, must be >= batch.size

// Fail fast instead of blocking the caller when metadata is unavailable.
props.put("max.block.ms", "500");

Producer<String, byte[]> producer = new KafkaProducer<>(props);
```

Note what this does and does not do. It bounds producer-side buffering latency. It does not change broker replication latency, and it does not make the pipeline durable if `acks=1` is chosen. Those are separate decisions.

**How to measure it.** Use the bundled producer performance tool with your real record size and throughput, and read the latency percentiles rather than the average:

```bash
kafka-producer-perf-test \
  --topic events \
  --num-records 200000 \
  --throughput -1 \
  --record-size 4096 \
  --producer-props bootstrap.servers=kafka1:9092 acks=all linger.ms=5 batch.size=262144 \
  --print-metrics
```

Compare the reported `99th percentile latency` against your SLO. Then re-run with `linger.ms=0` and with your previous value to see how much of your tail latency is producer buffering. If the p99 barely moves between `linger.ms=0` and `linger.ms=5`, your bottleneck is elsewhere — broker replication, consumer processing, or network — and tuning the producer further will not help.

## Failure mode 2: tiered storage cold reads

**Symptom.** Read latency is stable and low most of the time, then spikes on a regular cadence. Metrics show elevated disk read operations and elevated object-storage fetch latency.

**Root cause.** This applies to brokers with tiered storage, where older log segments are uploaded to object storage and evicted from local disk. When a consumer requests data that is no longer local, the broker must fetch the segment back. The first read pays that fetch; subsequent reads hit the local cache. A workload that repeatedly reads a window of data that has just aged out of the local cache will see the spike on every cycle.

**Fix.** Two knobs matter. First, how long a fetched segment stays in the local cache. Second, how large the local cache is allowed to grow. If your consumers routinely read the last N minutes of data, the local cache must be large enough and the retention long enough to hold that window.

The exact setting names and units differ between distributions and versions, so verify against your broker's documentation rather than copying a snippet. The categories to look for are:

- Local cache size limit (bytes or percentage of disk).
- Cache eviction or retention duration.
- Segment upload interval, which controls how quickly data leaves local disk.
- An option to pin or keep recent segments local.

**How to measure it.** Instrument the read path, not just the write path. What you want is a histogram of consumer fetch latency, broken down by whether the segment was served locally or from object storage. At minimum, track:

- p50, p95, and p99 of consumer fetch latency over a rolling window.
- A counter of object-storage fetches per minute.
- Local cache hit ratio.

If the p99 spikes correlate with object-storage fetch counts, the cause is confirmed. The decision then is whether the cost saving from tiered storage is worth the tail latency. For workloads with a strict sub-10 ms read SLO, keeping all actively read data on local disk is usually the correct answer, and the cost saving is not worth the SLO violation.

## Failure mode 3: persistence that is not synced

**Symptom.** Messages disappear during a rolling restart or a hard node failure. Consumers report missing sequence numbers or gaps. No application error is logged at publish time.

**Root cause.** This is the classic write-back cache problem, and it applies to any broker that acknowledges a write before it is flushed to stable storage. In a memory-mapped or buffered file design, the operating system may hold written data in the page cache and flush it later. If the process or the machine restarts before the flush, the data is gone even though the producer received an acknowledgment.

NATS JetStream is the most common place teams encounter this, because JetStream's persistence is optional and its default tuning favors throughput. If you configure memory storage, data is not durable at all. If you configure file storage without synchronous writes, durability depends on the OS flush schedule.

**Fix.** For NATS JetStream, use file storage and enable synchronous writes. The trade-off is explicit: each write waits for the disk, adding latency.

```conf
# nats-server.conf
jetstream {
  store_dir = "/var/lib/nats/data"
  max_memory_store = 1GB
  max_file_store = 10GB
}
```

The storage type and sync behavior are set per stream, not globally, in JetStream. Create or update the stream with file storage and synchronous writes enabled:

```bash
nats stream add EVENTS \
  --subjects "events.>" \
  --storage file \
  --replicas 3 \
  --max-age 72h
```

Check the stream's configuration afterward and confirm the storage backend is `file` and the replica count is what you expect. A single-replica file-backed stream is still durable across a process restart but not across loss of the node.

**How to measure it.** Do not measure durability by reading logs. Measure it by counting. Publish a known number of uniquely numbered messages, restart the broker, then count how many are present in the stream. Repeat the restart several times. The loss rate is `(published - present) / published`. Run this test with your production storage settings, because the answer changes completely between memory storage, file storage without sync, and file storage with sync.

The same test applies to Kafka and Redpanda: produce numbered records, kill the leader during the run, and verify the consumer sees a contiguous sequence. If it does not, your `acks` / `min.insync.replicas` combination permits loss, and that is a configuration choice you should make deliberately.

## How to run a fair bake-off

Broker benchmarks are only useful if they resemble your workload. Most published comparisons use small messages, idle clusters, and short runs, which measures the wrong regime. A bake-off that actually informs a decision has four properties.

**1. Use production-shaped traffic.** Same message size distribution, same key cardinality, same producer and consumer counts, same retention. If your events are 4 KB and bursty, do not benchmark 1 KB at a constant rate.

**2. Run long enough to hit steady state.** Compaction, segment rolling, tiered-storage uploads, and page-cache eviction all happen on timescales of minutes to hours. A five-minute test will not show them. Run for at least several hours, ideally through a full retention cycle.

**3. Measure tail latency and loss, not averages.** Averages hide the spikes that break SLOs. Record p50, p95, p99, and p99.9 for produce and consume latency. Count lost messages by sequence number gaps, not by log inspection.

**4. Test failure, not just steady state.** Kill a broker. Restart a broker. Fill a disk. Make the object store slow or unreachable. The behavior under failure is usually the deciding factor and is almost never in the benchmark table.

A minimal instrumentation list, applicable to all three systems:

| What to measure | Why it matters |
|---|---|
| Produce p99 latency | Detects batching and replication stalls |
| Consume p99 latency | Detects cold reads and consumer lag |
| Message loss count (by sequence gap) | The only honest durability metric |
| Disk usage growth per day | Determines retention and cost |
| Object-storage fetch rate and latency | Only if tiered storage is enabled |
| Replication/ISR state over time | Detects under-replicated partitions |

## Decision checklist

Work through these in order. The first question that fails usually eliminates a broker.

1. **Can you tolerate any message loss?** If no, you need synchronous replication with a quorum acknowledgment. Kafka with `acks=all` and `min.insync.replicas` set to at least 2, or Redpanda with equivalent settings, or NATS JetStream with file storage, multiple replicas, and sync enabled. If yes, you have more freedom on latency and cost.
2. **What is your p99 end-to-end latency budget?** If it is under 10 ms, tiered storage and large batching are both off the table for the hot path. If it is 50–100 ms, you can batch and tier aggressively.
3. **How much data do you retain, and how often is it read?** High retention with rare reads favors tiered storage. High retention with frequent reads of recent data favors local disk with a large cache.
4. **What is your operational capacity?** A system with fewer moving parts is easier to run, but "fewer moving parts" is not the same as "no tuning." Every broker here has defaults that assume a different workload than yours.
5. **What does your team already run?** Operational familiarity has real value. A well-tuned system your team understands usually beats a theoretically better system nobody can debug at 3 a.m.

## FAQ

**Is Redpanda a drop-in replacement for Kafka?**

It exposes the Kafka API, so Kafka clients generally work without code changes. That does not mean behavior is identical. Replication, storage, and tiered-storage behavior are implemented independently, and defaults differ. Treat it as a compatible API with different operational characteristics, and validate with your own bake-off rather than assuming equivalence.

**Does NATS JetStream guarantee durability?**

Only if you configure it to. Persistence is optional, storage type is per-stream, and synchronous writes are a configuration choice. With memory storage, there is no durability. With file storage and sync enabled across replicas, writes survive restarts. The default is not the durable configuration.

**Why does Kafka batch messages at all?**

Batching amortizes network and disk overhead across many records, which is how a single broker sustains high throughput. The cost is added latency for each record waiting for its batch. The correct `linger.ms` is a function of your latency budget, not a universal constant.

**How do I know if my broker is dropping messages?**

Count. Publish records with monotonically increasing sequence numbers, consume them, and look for gaps. This works regardless of broker and does not depend on trusting logs or metrics. Run it during a restart to test the durability path specifically.

**Can I use more than one of these in the same system?**

Yes, and it is common. A low-latency path for real-time decisions and a durable log for replay and audit serve different requirements. The cost is operating two systems and defining the handoff semantics, including what happens when the fast path and the durable path disagree.

## Do this now

Open your producer configuration and find the value of `linger.ms` (Kafka and Redpanda) or your stream's storage and sync settings (NATS JetStream). Then answer one question in writing: what is the maximum amount of data you are willing to lose on a node failure, expressed in seconds or messages?

If `linger.ms` is greater than your end-to-end latency budget, lower it and re-run a load test with your real message size. If your NATS stream uses memory storage or a single replica, and you claimed you need durability, change it now. If you cannot state your loss tolerance, that is the first thing to fix — no broker choice is correct until you know what you are optimizing for.
