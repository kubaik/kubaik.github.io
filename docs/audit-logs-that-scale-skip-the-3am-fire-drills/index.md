# Audit logs that scale: skip the 3AM fire drills

Official documentation for audit logging covers the mechanics well. What it rarely covers is what happens six months into production, when the pipeline has quietly become a second production system with its own latency budgets and cost profile. The failure modes below are common enough to be predictable, and the architecture that avoids them is well understood. What follows is the reasoning a senior engineer would pass to a colleague hitting this for the first time.

## The conventional wisdom (and why it's incomplete)

Audit logging is often built like a tax form: a legal checkbox with no bearing on the product itself. A team ships a centralized log collector, attaches a wrapper in every service, and assumes the worst-case line volume is whatever a capacity calculator predicts. That works until it doesn't.

The hidden assumption is that audit logs are write-only: something you ingest and forget. In practice, they become a second production system. Compliance teams demand long retention windows; performance teams watch p99 latency spike when the garbage collector pauses every 30 seconds because the heap is full of multi-megabyte JSON blobs. The standard three-tier architecture—app → collector → warehouse—collapses under its own weight when any one tier hiccups.

A common failure mode is the "too-many-fields" schema. Teams start with a simple `{user_id, action, timestamp, metadata}` and then bolt on `client_ip`, `user_agent`, `request_id`, `geo`, `device_id`, and a dozen custom fields for each new regulation. By the time a regulatory reporting obligation comes due, the metadata object is mostly nulls and nested JSON, and every query against it times out in the search tier. Worse, schema drift breaks downstream parsers; one missing field in a nightly ETL job and the entire compliance report is red.

So the conventional stack—a search engine for interactive queries, object storage for cold retention, and a data lake for analytics—is not wrong; it's just incomplete. The missing piece is treating audit logs as a real-time data product with latency budgets, cost controls, and idempotency guarantees. The part that trips people up is that the same pipeline must satisfy two masters: auditors who want an exact, tamper-evident copy of every event, and engineers who need sub-second query times for incident response.

## What actually happens when you follow the standard advice

A typical incident pattern recurs across companies of very different sizes.

Consider a service running on Kubernetes with a log shipper forwarding JSON lines into a search cluster. The cluster indexes millions of events per minute at peak, on the order of hundreds of gigabytes per day uncompressed. During a marketing push, the on-call engineer notices p99 search latency climb from a few hundred milliseconds to multiple seconds. On investigation:

- Heap usage on the search data nodes spikes because each log line is large and the JVM spends a meaningful fraction of CPU in garbage collection.
- The rollover policy keeps many indices open; shard allocation pressure causes yellow cluster states.
- A misconfigured lifecycle policy leaves indices on hot nodes far longer than intended, inflating storage cost by an order of magnitude relative to the cold tier.

The compliance team, meanwhile, receives an access request under GDPR Article 15. The data subject wants all logs related to their account exported within 30 days. The pipeline exports a multi-gigabyte JSON file, and the ETL job that flattens it fails after 90 minutes with an out-of-memory error. The exported file is missing events because the date-range filter in the query was off by one hour due to a daylight-saving transition.

That is the standard advice in action: centralized logging, a search engine, object storage. It satisfies compliance in theory, but in practice it creates a second production fire drill every quarter.

## A different mental model

Auditors care about three things: completeness, integrity, and non-repudiation. Engineers care about latency, cost, and correctness. The mental model that bridges both is to treat audit logs as **immutable event streams** first and **searchable records** second.

Start with an append-only log. A distributed commit log (Kafka is the common choice) or a managed stream service are the usual options. Every microservice writes to a topic such as `audit.v1`, partitioning by `user_id` or `tenant_id` so downstream consumers can scale independently. The payload is a **strict schema** (Avro or Protobuf) with fixed fields: `event_id`, `user_id`, `action`, `resource`, `timestamp`, `metadata`. No late-arriving fields, no deeply nested objects—a flat key-value map with a documented maximum size per event.

Next, run **two parallel pipelines** off the same stream:

1. **Realtime pipeline**: a small consumer group writes to a columnar store optimised for point queries. ClickHouse is a typical choice. Because the schema is strict and flat, ingestion latency is low and storage is meaningfully smaller than JSON.

2. **Cold pipeline**: a separate consumer writes to immutable object storage in Parquet format, partitioned by `year=YYYY/month=MM/day=DD/hour=HH`. This satisfies a long retention requirement without search-engine overhead.

Integrity is handled by **cryptographic hashes**. Each event carries a hash of its payload plus the previous event's hash, forming a hash chain. The columnar table stores the hash alongside the event, letting auditors verify the chain without touching cold storage.

Cost control comes from **tiered retention**. Hot data (for example, the last 7 days) lives on SSD-backed nodes; warm data (7–90 days) moves to object storage with Zstd compression; cold data (beyond 90 days) lands in an archival storage class.

This architecture flips the conventional model: the expensive search engine isn't the primary store; it's a cache. Compliance audits run against Parquet in object storage, which is cheaper and easier to verify than search indices.

## How to measure whether this matters for you

Rather than quoting benchmark numbers, instrument the pipeline you already have. The following measurements are the ones that decide the architecture.

**Line size.** Sample 10,000 events from your shipper and compute the mean and p99 of the serialized payload size. Compare JSON against the same events encoded in Avro or Protobuf. The ratio you observe is your compression headroom, and it is usually the single largest lever on storage cost.

**Ingest-to-query latency.** Emit a synthetic event with a known `event_id`, then poll the query tier until it appears. Record the delta. Do this at 1×, 2×, and 5× your current peak rate. The point at which the delta degrades non-linearly is your practical ceiling.

**Garbage collection pressure.** On the search tier, expose JVM GC metrics and chart the fraction of CPU spent in collection against ingest rate. When GC time exceeds roughly 10 % of wall-clock CPU, latency becomes unpredictable regardless of how much heap you add.

**Query latency by pattern.** Separate point lookups (equality on `user_id`) from full-text and aggregation queries. Measure p50, p95, and p99 for each. Architectures that excel at one often fail at the other, and the split tells you whether a single store can serve both.

**Export cost.** Time a full subject-access export for a single user over a 30-day window. Record wall-clock time, peak memory, and output size. If the job's peak memory scales with the number of events rather than the number of users, it will eventually fail.

Run these measurements before choosing a stack. The numbers you collect are more useful than any published benchmark, because they reflect your schema, your query mix, and your hardware.

## The cases where the conventional wisdom IS right

Not every team needs the dual-stream model. Three situations still suit a search-engine-centric stack:

1. **Small scale**: If peak volume is low (hundreds of events per minute) and retention is under six months, a single-node search cluster with a lifecycle policy and a frozen tier is simpler and cheaper than maintaining a commit log plus a columnar store.

2. **Search-heavy workloads**: If the primary use case is full-text search across unstructured logs—debugging user flows, for example—a search engine still wins on query flexibility.

3. **Regulatory sandbox**: In industries where the regulator provides a standard schema and a mandated search interface, the compliance team may insist on a search engine regardless. Fighting it adds overhead without benefit.

Even here, the worst failures are mitigable:
- Cap the log line size at a documented maximum (2 KB is a common ceiling).
- Use a hot-warm architecture with an explicit retention boundary.
- Run a nightly integrity job that snapshots event identifiers and hashes the payload.

## How to decide which approach fits your situation

Use the **4-question filter** to pick your stack:

| Question | Dual-stream (commit log + columnar + object storage) | Search-engine-centric | Notes |
|---|---|---|---|
| Peak events/minute | High (tens of thousands) | Low (thousands) | Dual-stream scales horizontally; a single search node has a practical ceiling that depends on shard count and heap |
| Retention | >6 months | <6 months | Dual-stream cost advantage grows with retention; search cold tiers get expensive above roughly 1 TB |
| Query pattern | Point lookups by user_id or resource | Full-text search or aggregations | Dual-stream excels at equality filters; search engines excel at regex and text |
| Team skillset | Commit log, columnar store, Parquet | Search engine, lifecycle policies | Dual-stream requires more DevOps muscle; search engines are easier to hire for |

A useful rule of thumb: if projected audit volume exceeds roughly 1 TB/year or the query latency target is under 100 ms, default to the dual-stream model. If not, the search-engine-centric stack remains viable.

## Objections and responses

**Objection 1**: "Adding a commit log doubles the operational surface. We already run PostgreSQL and Redis; we don't want another system."

Response: You're trading one surface for two—but the new surfaces are **simpler**. A commit log is a dumb append-only pipe; a columnar store is SQL over immutable parts. Both are often easier to operate than a search cluster with its JVM tuning, shard allocation headaches, and Lucene merge storms. The real cost is not the systems; it's the pager duty when the search tier browns out at 3 AM.

**Objection 2**: "Our auditors insist on a search engine because their tooling only reads those indices."

Response: Give them a read-only replica. A columnar store can export a subset of the audit table to a search index nightly via Parquet. The replica is smaller (hot data only) and read-only, so it cannot corrupt the primary chain. Most audit tooling only needs the last 30 days anyway.

**Objection 3**: "Protobuf/Avro adds complexity; JSON is universal."

Response: JSON is not universal—it's slow and bloated. Protobuf and Avro typically compress substantially better than JSON and parse faster. The complexity is front-loaded in schema evolution; once the schema is locked, downstream parsers are trivial. Teams that stick with JSON usually end up adding a schema registry anyway, so they're not saving complexity.

**Objection 4**: "We already have billions of events; migrating is impossible."

Response: Migrate in place. Start a dual-write: every service writes to both the old topic and the new commit log. Run a streaming job that backfills the new columnar table from the old index using scroll queries. Once the columnar table is caught up and the hashes verify, flip the consumer to read from the commit log only. The cutover takes one maintenance window and the old index can be kept as a backup for 30 days.

## A worked example: verifying a hash chain

Suppose each event carries a SHA-256 hash of its payload concatenated with the previous event's hash. The genesis event uses a fixed sentinel value, such as 32 zero bytes, as its "previous hash."

Event 1: `payload = {"event_id":"a1","user_id":"u1","action":"login"}`, `prev_hash = 0000...0000`. Compute `h1 = SHA256(payload || prev_hash)`.

Event 2: `payload = {"event_id":"a2","user_id":"u1","action":"read"}`, `prev_hash = h1`. Compute `h2 = SHA256(payload || h1)`.

To verify a range, fetch the events in order, recompute each hash from the stored payload and the previous stored hash, and compare. If any recomputed hash differs from the stored hash, the chain is broken at that point—either an event was modified, deleted, or inserted.

A verification query can return the aggregate state of a range:

```sql
SELECT
  count() AS event_count,
  min(hash) AS min_hash,
  max(hash) AS max_hash
FROM audit.v1
WHERE user_id = 'user123'
  AND timestamp >= now() - INTERVAL 30 DAY;
```

This query returns the event count and the extreme hashes for the range. It does not by itself prove integrity—`min` and `max` are not order-sensitive—but it gives an auditor a cheap fingerprint to compare against a locally recomputed chain. The full verification walks the events in `timestamp` order and recomputes each hash.

A Python verification script is short:

```python
import hashlib

def verify(events, genesis=b"\x00" * 32):
    prev = genesis
    for e in sorted(events, key=lambda x: x["timestamp"]):
        payload = e["canonical_payload"].encode()
        expected = hashlib.sha256(payload + prev).hexdigest()
        if expected != e["hash"]:
            return False, e["event_id"]
        prev = e["hash"]
    return True, None
```

The `canonical_payload` field must be a deterministic serialization (sorted keys, no whitespace) so that recomputation is reproducible across languages.

## What to do differently when starting over

If designing an audit log pipeline from scratch, the following choices pay off:

1. **Enforce schema evolution from day one**. Use a schema registry with compatibility set to BACKWARD. Reject any log line that violates the schema at the producer level; this prevents silent data corruption.

2. **Use a single schema for the entire audit stream**, not per-service schemas. This keeps downstream SQL simple and avoids the "schema soup" problem where every team defines its own metadata fields.

3. **Run a nightly integrity job** that reads the latest 24 hours from the columnar store, recomputes the hash chain, and writes the min/max hashes to a dedicated `audit_integrity` table. This table is tiny and lets you detect chain breaks without scanning the full dataset.

4. **Add a dead-letter topic** for malformed events. Instead of dropping them silently, route them to a `dead_letter` topic so schema mismatches can be debugged before they poison production.

5. **Use tiered storage in the columnar store**: keep 7 days on SSD, 90 days on HDD, and archive to object storage for anything older. This reduces storage cost without sacrificing query performance for recent data.

A Terraform sketch of the storage and schema layer:

```hcl
resource "clickhouse_table" "audit_v1" {
  name         = "audit.v1"
  engine       = "MergeTree()"
  order_by     = "(event_id, timestamp)"
  partition_by = "toYYYYMM(timestamp)"
  settings = {
    storage_policy = "tiered"
  }
  columns = [
    { name = "event_id",  type = "UUID" },
    { name = "user_id",   type = "String" },
    { name = "action",    type = "String" },
    { name = "resource",  type = "String" },
    { name = "timestamp", type = "DateTime64(3)" },
    { name = "hash",      type = "FixedString(32)" },
    { name = "metadata",  type = "Map(String, String)" },
  ]
}
```

Note that the hash column here is a fixed 32 bytes, which corresponds to a raw SHA-256 digest. If you store the hex representation instead, use `FixedString(64)`.

## Summary

The conventional audit logging stack—search-centric, JSON-heavy, and retention-focused—fails under real load because it treats logs as a second-class citizen. A dual-stream architecture—a commit log for the event pipe, a columnar store for hot queries, and Parquet in object storage for cold retention—satisfies both compliance and performance by making immutability and integrity primary concerns rather than afterthoughts.

The deciding factor is scale and retention: once audit logs exceed roughly 1 TB/year or the latency target drops below 100 ms, the search-centric stack starts costing more in operational toil than it saves in setup time. For everyone else, the conventional stack is still viable—just cap your log line size, enforce schemas, and run a nightly integrity check.

The part that trips people up is assuming that audit logs are write-only. In reality, they're a second production system with their own latency budgets, cost controls, and uptime guarantees. Build them like one.

## Frequently Asked Questions

**How do you export audit logs for a GDPR subject access request?**

Export from the Parquet files in object storage, not from the search tier. A local columnar store or a query engine over Parquet can run:

```sql
SELECT * FROM audit.v1
WHERE user_id = 'user123'
  AND timestamp >= '2026-01-01 00:00:00'
  AND timestamp <= '2026-01-31 23:59:59'
INTO OUTFILE 'user123_audit.jsonl'
FORMAT JSONEachRow
```

The Parquet files are already partitioned by date, so the query is fast and the output is machine-readable. If the regulator insists on a search index, replicate the last 30 days to a read-only index nightly using a simple consumer.

**What is a typical audit log line size?**

A well-tuned audit line in Avro or Protobuf averages a few hundred bytes after compression. JSON lines in the wild are often several times larger. The difference is schema overhead and repeated field names. Teams that stick with JSON and keep adding custom fields often exceed 2 KB per line, which drives up storage and query costs.

**How do you verify audit log integrity without a search engine?**

Use a hash chain. Each event carries a hash of its payload plus the previous event's hash. Store the previous hash alongside the event. Nightly, recompute the hash of the batch for the last 24 hours and compare it to the stored final hash. A mismatch indicates corruption or deletion. The verification script is under 100 lines of Python.

**What retention settings work for the commit log?**

Set `retention.ms` to a value that covers your replay window—24 hours is a common choice—and let the cold pipeline own anything older. Avoid `retention.bytes`, which interacts poorly with partition splits. If you need to replay, use the Parquet files in object storage as the source of truth rather than commit-log retention.

**Do you need a schema registry?**

Only if your producers and consumers evolve independently. If a single team owns both ends of the stream and schema changes are coordinated, a versioned schema file checked into the repository is sufficient. A registry becomes valuable when multiple teams write to the same topic and compatibility must be enforced at the producer.

## Take action in the next 30 minutes

Sample 10,000 events from your current audit pipeline, compute the mean and p99 serialized size, and re-encode the same events in Avro or Protobuf. The ratio you measure is your storage headroom, and it will tell you within half an hour whether the architecture question is worth a deeper investigation.
