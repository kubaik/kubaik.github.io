# MCP servers: real-time pipelines without agents

Legacy modernisation projects repeatedly hit the same wall. A claims system still runs on green screens and a proprietary terminal emulator, fronted by a monolithic Java application on a single 4-core VM in a regional data centre. The business wants fraud alerts within 500ms of a claim filing. The brick wall appears when a lightweight streaming server is bolted onto the legacy COBOL copybooks: the layouts change monthly, the JVM heap exhausts under load, and the network team will not open ports for WebSocket upgrades.

The constraint is rarely bandwidth. It is data contract entropy. Most streaming tutorials assume both ends of the wire are under your control and can adopt Protobuf or Avro. Legacy systems do not play that game. They expose fixed-length, EBCDIC-encoded, packed-decimal fields through CICS BMS maps that expect 3270 data streams. The bottleneck is the impedance mismatch between modern messaging formats and 1970s data layouts. What is needed is a way to turn COBOL copybooks into a streaming interface that can feed real-time fraud models without rewriting the mainframe.

This article covers how to evaluate that interface, what to instrument, and which architectural shapes actually fit a single constrained VM.

## The real constraints, stated plainly

Before comparing tools, write down the constraints. A typical set looks like this:

- **Latency budget:** 500ms end-to-end for a fraud alert, measured from claim submission to alert emission.
- **Hardware:** one 4-core VM, roughly 32 GB RAM, no additional nodes permitted.
- **Heap cap:** about 1 GB for any JVM-based component, because the legacy application already owns the rest.
- **Network:** no new inbound ports beyond what security already approved. Assume only the broker port is reachable.
- **Data contract:** copybook layouts change on a monthly cadence, sometimes without notice.
- **Integration path:** CICS via EXCI, or IBM MQ over LU 6.2. An intermediate REST layer or a JNI shim is usually rejected at review.

Any candidate that violates one of these constraints is disqualified regardless of benchmark results. That framing matters, because a tool that is fast but needs three Kafka brokers is not a candidate at all.

## How to evaluate a candidate (four tests)

Every candidate should run through the same four tests. None of these require a production deployment; all can be run on a copy of the target VM.

### 1. Data contract survival test

Can the pipeline ingest the raw copybook layout without codegen or schema registry maintenance?

The practical way to measure drift is to compute a hash of the copybook source and attach it to each message as metadata. At runtime, compare the incoming hash against the last known hash. Log a mismatch with the field offsets that shifted.

A worked example: a copybook defines a claim record with a `CLAIM-ID` of `PIC X(12)` at offset 0, a `FILED-DATE` of `PIC 9(8)` at offset 12, and a `AMOUNT` of `PIC S9(7)V99 COMP-3` at offset 20. If a new field is inserted at offset 12, every downstream parser that assumed a fixed offset now reads garbage. A hash check catches this in one comparison; a schema registry catches it only if someone remembered to register the new schema, which is exactly the failure mode being avoided.

Any solution that forces a parallel schema registry or an IDL file to be maintained by hand fails this test.

### 2. Latency budget test

Measure end-to-end latency with a load generator that replays representative traffic. A synthetic generator that emits claim payloads of about 1.2 KB at a target rate is sufficient. Instrument three timestamps:

- `t_ingest`: when the record is read from the source.
- `t_convert`: when the EBCDIC bytes are decoded into a usable structure.
- `t_emit`: when the fraud alert leaves the pipeline.

The difference between `t_ingest` and `t_emit` is the number that matters. Report median and 99th percentile separately; a pipeline with a 150ms median and a 2-second tail will miss the SLA during exactly the traffic spikes that matter.

Run the generator at a rate above the expected peak, not at the expected peak. If the expected peak is 5,000 claims/s, test at 6,000 to see where the tail begins to degrade.

### 3. Memory footprint test

Measure resident set size (RSS), not just heap. Heap monitoring misses native buffers, memory-mapped files, and off-heap caches, all of which matter on a constrained VM.

On Linux, sample RSS while the load generator runs:

```bash
while true; do
  ps -o rss= -p $(pgrep -f fraud-pipeline) >> rss.log
  sleep 1
done
```

Then compute the maximum and the steady-state plateau. A candidate that plateaus at 600 MB is safe; one that creeps upward without bound is leaking and will fail eventually, even if it passes a short test.

### 4. Legacy integration test

Can the candidate talk to CICS via EXCI or to IBM MQ over LU 6.2 without an intermediate REST layer or a JNI shim? If not, it is disqualified. This test is binary and should be applied before any benchmarking, because it eliminates most candidates immediately.

## Architectural shapes that fit the constraints

There are three shapes that commonly survive all four tests on a single VM. They are described by their properties rather than by product names, because the specific products change faster than the shapes do.

### Shape A: In-process conversion with a data-structure store

A single process hosts both the message handling and a data-structure store. A small function, running inside that process, converts raw EBCDIC copybook bytes into a structured representation (for example, a JSON document) on ingest. The structured form is stored alongside the raw bytes.

- **Strengths:** one process to manage, no cross-process latency, direct access to the raw bytes for drift checks.
- **Weaknesses:** a single process is a single point of failure; module loading failures can take the whole process down; persistence options are limited to what the process supports.

### Shape B: Lightweight broker with an external sink

A lightweight message broker handles ordering and fan-out. A separate process or module consumes from the broker and writes to a persistence layer for replay.

- **Strengths:** ordering guarantees, subject-based routing, TLS termination, and a clear separation between transport and storage.
- **Weaknesses:** two processes to manage; file-backed broker storage can saturate a single disk under replay load; a broker outage pauses alerts unless a fallback path exists.

### Shape C: Stream processor with exactly-once semantics

A stream processor consumes from the legacy source, decodes copybook records with a custom deserialiser, and writes to a fraud topic. Windowing and late-event handling are handled by the processor.

- **Strengths:** exactly-once semantics, windowed scoring without an external database, mature handling of late events.
- **Weaknesses:** needs a cluster manager; the resident footprint of a minimal cluster typically exceeds what a single legacy VM can spare; JVM tuning becomes a project of its own.

## Comparison at a glance

The table below compares the three shapes on the dimensions that decide the outcome. Figures are illustrative targets, not measured results; substitute your own measurements from the tests above.

| Dimension | Shape A (in-process) | Shape B (broker + sink) | Shape C (stream processor) |
|---|---|---|---|
| Processes to manage | 1 | 2 | 2+ (cluster) |
| Typical median latency target | sub-200ms | sub-200ms | sub-250ms |
| RSS budget fit on 4-core VM | Good | Tight | Poor |
| Ordering guarantees | Limited | Strong | Strong |
| Exactly-once semantics | No | Depends on sink | Yes |
| Replay after restart | Depends on store | Yes, from broker | Yes |
| Copybook drift handling | In-process hash check | Consumer-side hash check | Deserialiser-side check |
| Operational complexity | Low | Medium | High |

The point of the table is not to crown a winner. It is to make the trade-off explicit: Shape A wins on footprint and simplicity, Shape B wins on durability and ordering, Shape C wins on correctness guarantees at the cost of operational weight.

## Worked example: converting a packed-decimal field

The single most common source of silent corruption when bridging copybooks is packed-decimal (`COMP-3`) handling. A worked example makes the failure mode concrete.

Consider `AMOUNT PIC S9(7)V99 COMP-3`. This field holds a signed value with seven integer digits and two decimal digits. Packed decimal stores two digits per byte, with the sign in the low nibble of the last byte. For a value of `1234567.89`, the digits are `123456789`, which is nine digits, so the field occupies five bytes: four full bytes of digit pairs plus a final byte containing the last digit and the sign nibble.

A naive parser that reads the field as an integer will produce `123456789` and then divide by 100, which happens to be correct here. But a parser that assumes ASCII digits will read the bytes as garbage, and a parser that ignores the sign nibble will silently drop negative amounts. Negative amounts matter: a fraud model that never sees refunds will flag legitimate reversals as anomalies.

The correct approach is to decode the field byte by byte, accumulate the digit pairs, and apply the sign from the final nibble. This is a small amount of code, but it must be tested against real copybook data, including negative values, values with leading zeros, and values at the field's maximum magnitude.

Instrument the conversion path to log any field whose decoded value falls outside the expected range for that field. A claim amount of `9999999.99` is legal; a claim amount of `999999999` after decoding indicates a field-offset error. Range checks catch offset drift that a hash check might miss if the copybook hash was updated but the parser was not.

## Failure modes to design against

These are the failure modes that show up repeatedly in production, in rough order of how often they cause incidents.

**Silent offset drift.** A copybook changes, the hash check is not wired into the alerting path, and the pipeline keeps running with misaligned fields. Mitigation: make a hash mismatch a hard failure that stops the pipeline and pages an operator, not a warning in a log file.

**Unbounded memory growth.** A parser allocates a new buffer per message and relies on garbage collection to reclaim it. Under sustained load, allocation outpaces collection and RSS climbs until the process is killed. Mitigation: reuse buffers where the language allows it, and monitor RSS with a hard alert threshold well below the VM's limit.

**Disk saturation during replay.** A file-backed broker stream is replayed after a restart, and the disk cannot keep up with the read rate. Mitigation: size the replay window to what the disk can sustain, or place the stream on a memory-backed filesystem if durability requirements allow it.

**Module load failure taking down the host process.** Some data-structure stores load extensions at startup; a corrupted or version-mismatched extension can crash the process rather than failing gracefully. Mitigation: pin extension versions explicitly, verify checksums at build time, and test the startup path in a staging environment that mirrors production.

**Clock skew between the legacy source and the pipeline host.** Timestamps used for windowing can be wrong if the two clocks drift. Mitigation: use the source timestamp for windowing, and monitor the offset between source and pipeline clocks.

## A decision checklist

Work through these in order. The first "no" answer eliminates the shape.

1. Can the candidate read CICS via EXCI or IBM MQ over LU 6.2 without an intermediate layer? If no, stop.
2. Does the candidate fit within the VM's RSS budget at the target peak rate, measured, not estimated? If no, stop.
3. Does the candidate's median and 99th percentile latency both fit the SLA at a rate above the expected peak? If no, stop.
4. Does the candidate detect copybook drift and fail loudly? If no, add that capability before proceeding.
5. Does the candidate survive a restart without losing messages that the fraud model needs? If no, add a persistence or replay path.
6. Can the team operate it with the staff available? A pipeline that needs a dedicated cluster administrator is not a fit for a team that does not have one.

## Frequently asked questions

**How do I convert a COBOL copybook to a structured format without losing precision?**

Decode the field types explicitly rather than treating the record as a byte blob. Packed decimal needs byte-by-byte decoding with sign handling. Zoned decimal needs EBCDIC-to-ASCII conversion per digit. Binary fields need endianness handling. Store the raw bytes alongside the decoded form so that a decoding bug can be diagnosed without re-reading the source. The conversion cost per record is small; measure it rather than assuming it.

**Can a lightweight broker guarantee ordering for fraud detection?**

Some brokers offer ordered consumers that assign a monotonically increasing sequence number, which is sufficient for windowing. The catch is that file-backed storage on a single disk can saturate under replay load. Measure the disk's sustained read rate and size the replay window accordingly, or use a memory-backed filesystem if the durability trade-off is acceptable.

**What is the smallest deployment that survives a failover?**

For a data-structure store, a common minimum is three nodes with a quorum of two, which survives one node failure. For a message broker, a three-node cluster with replication factor two is a common starting point. Both require more than one VM, which may violate the single-VM constraint. If the constraint is absolute, accept that failover is not available and design the pipeline to restart quickly and replay from the source.

**How do I benchmark latency without deploying the full pipeline?**

Measure each stage separately first. For the conversion stage, write a small harness that decodes a fixed set of copybook records in a loop and reports per-record time. For the transport stage, use the broker's built-in benchmark tool if it has one, or a simple publisher and subscriber pair. Compose the stage measurements into an end-to-end estimate, then validate the estimate with a full-pipeline test before committing. Stage measurements are cheap; full-pipeline tests are not.

## What to do in the next 30 minutes

Pick one representative copybook, write a hash of its source, and add a runtime check that compares the incoming hash against the last known value and logs a structured error on mismatch. This is the single highest-value change you can make today, because it converts a silent corruption failure into a loud, diagnosable one.
