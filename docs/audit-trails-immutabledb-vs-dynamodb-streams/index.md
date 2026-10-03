# Audit trails: ImmutableDB vs DynamoDB Streams

## The audit trail problem, stated precisely

An audit trail for an automated decision system has to answer three questions after the fact:

1. What did the system decide, and what inputs produced that decision?
2. Can the record be trusted to be the same one that was written at the time?
3. Can the decision be replayed and inspected quickly enough to be useful during an incident?

Most teams get the first question right early and discover the other two much later, usually when an auditor or an incident review asks for something the original design never anticipated. A common failure mode is a plain relational table with a JSON column: it is easy to write, easy to query, and quietly hostile to both tamper evidence and bulk replay. A table with tens of millions of rows and nested JSON payloads will make an export script crawl, and nothing in the schema prevents an application bug or a well-meaning engineer from mutating a historical row.

Two architectural families address this:

- **Append-only, hash-chained logs.** Records are written once, never updated, and each record carries a hash derived from its contents and its predecessor. This is the model used by event stores, ledger systems, and change-data-capture logs with cryptographic linking.
- **Managed change streams on a database.** The database itself emits an ordered stream of item changes, and a consumer persists or forwards them. DynamoDB Streams is the canonical AWS example.

These are not interchangeable. They differ in where the trust boundary sits, what a "restore" actually means, and how the cost scales. The rest of this article works through the mechanics, a worked cost model, and a decision checklist.

## Where the trust boundary sits

The single most useful way to separate these designs is to ask: *who guarantees that the record is unchanged?*

With an append-only hash-chained log, the guarantee is **structural**. Each record's hash covers its own payload and the previous record's hash. To alter record *N* undetectably, an attacker must recompute every hash from *N* to the head of the log. If the head hash is published or stored somewhere outside the log's write path, that becomes computationally infeasible. Verification is a pure function: given the records and a trusted head hash, anyone can check the chain without trusting the storage layer.

With DynamoDB Streams, the guarantee is **operational**. DynamoDB does not let you mutate an item in place without generating a `MODIFY` stream record, and point-in-time recovery (PITR) lets you restore a table to any second within a rolling window (the documented maximum is 35 days). But the trust argument is "the database and its access controls prevented tampering," not "the data is self-verifying." If an operator with sufficient IAM permissions writes directly to the table, the stream faithfully records that the write happened — it does not prove the resulting item is the one the application intended.

That distinction matters in regulated environments. Some auditors accept database access controls plus a complete change stream as sufficient evidence. Others want a cryptographic proof that a specific record has not been altered since it was written. These are different requirements and they lead to different architectures.

A middle path exists and is often the right answer: keep DynamoDB as the system of record, and have the stream consumer compute a hash chain and write the chained digest into a separate, write-once destination (an S3 object with Object Lock, for example). This gets tamper evidence without running a separate log cluster. The cost is that the hash chain is only as trustworthy as the consumer that builds it, and any gap in stream processing creates a gap in the chain.

## Option A: append-only hash-chained logs

The general shape of this design:

- A writer appends a record containing the payload plus a sequence number and a hash of `(previous_hash || payload)`.
- Readers can verify any prefix of the log against a known head hash.
- Compaction, if it exists at all, is an explicit operator action that rewrites the log in a way that preserves verifiability, not a background process that silently drops records.
- Replay means reading the log from a chosen point and feeding records into a consumer that reconstructs state.

**What this buys you.** Tamper evidence is intrinsic. Replay is a first-class operation because the log *is* the source of truth, not a side effect of it. Schema evolution is usually handled by tagging records with a version and having readers ignore fields they do not recognise, which is more forgiving than it sounds in practice.

**What it costs you.** You are now operating a stateful system. If you run it yourself, you own replication, disk capacity, shard or partition planning, and upgrade coordination. Write latency is bounded by the synchronous append and the replication factor you choose. Schema changes that affect the hash input — for example, adding a field that participates in the digest — require a coordinated rollout, because old and new writers must agree on what is being hashed.

**A worked verification example.** Suppose each record stores `seq`, `payload`, and `hash`, where `hash = SHA256(seq || prev_hash || payload)`. To verify records 1000 through 5000 given a trusted `hash` for record 999:

1. Fetch records 1000–5000 in order.
2. For each record, recompute `SHA256(seq || prev_hash || payload)` and compare to the stored `hash`.
3. Set `prev_hash` to the stored hash and continue.
4. After record 5000, compare the final computed hash to a head hash obtained from a trusted source.

The cost is linear in the number of records and dominated by hashing and I/O. To measure it on your own data, instrument the verification loop with a counter and a timer, run it against a representative slice of your log, and record records-per-second and peak RSS. Do this on the same instance class you would use in production, because the answer depends heavily on payload size and disk throughput.

**Failure modes to design against.**

- *Unbounded growth.* Append-only means append-only. Without a retention or archival policy, storage costs grow monotonically and replay times grow with them. Decide the retention window before you write the first record.
- *Head hash handling.* If the head hash is stored only inside the log, an attacker who can rewrite the log can rewrite the head too. Store periodic head hashes somewhere the log writer cannot modify.
- *Consumer lag.* A replay consumer that falls behind during an incident is the moment you need it most. Alert on consumer lag as a first-class metric, not an afterthought.

## Option B: DynamoDB Streams

DynamoDB Streams emits an ordered record of item-level changes (`INSERT`, `MODIFY`, `REMOVE`) for a table. A consumer — a Lambda function, a Kinesis adapter, or a long-running poller — reads those records and does something with them.

**The mechanics that matter for audit trails.**

- Stream records are ordered *within a shard*, and shards are derived from partition keys. There is no global ordering across the table. If your audit requirement is "decisions in the exact order the system made them," you must either encode ordering into the item (a monotonic sequence per agent) or accept per-key ordering only.
- Stream records are retained for 24 hours. This is a hard limit. If your consumer is down for longer than that, the records are gone. Anything that needs longer retention must be persisted elsewhere by the consumer.
- Point-in-time recovery is a separate feature and must be enabled explicitly. The documented window is 35 days. PITR restores the *table*, not the stream — you cannot replay stream events from a PITR restore.
- TTL deletes items and generates `REMOVE` stream records. If your consumer is not prepared for deletions, TTL will quietly introduce gaps in downstream state.
- A Lambda consumer processes shards concurrently. Without care, this means out-of-order processing across shards, which can corrupt a downstream state reconstruction that assumed global order.

**What this buys you.** No servers to run, scaling handled by the platform, and a natural fit if DynamoDB is already your primary store. The change stream is generated by the same system that holds the data, so there is no dual-write consistency problem for the primary record.

**What it costs you.** The trust argument rests on IAM and database controls rather than cryptography. The 24-hour stream retention is a real operational constraint. And the cost model is less obvious than it first appears, because it is dominated by storage and by the compute that consumes the stream.

## A worked cost comparison

Cost comparisons are only meaningful with stated assumptions. The figures below are **illustrative** and use round numbers so the arithmetic is checkable; substitute your own region's prices and your own traffic.

Assume 2.4 million decisions per day, or roughly 876 million per year. Assume each stored decision record is about 1.2 KB after encoding, giving roughly 1.05 TB of logical data per year.

**Append-only log, self-managed.** Assume three replicas on a mid-size instance class at $0.13 per instance-hour:

- Compute: 3 × $0.13 × 24 × 365 = $3,416 per year.
- Storage: 1.05 TB × 3 replicas = 3.15 TB. At $0.10 per GB-month: 3,150 × $0.10 × 12 = $3,780 per year.
- Cross-AZ or egress traffic: assume 1 TB per month at $0.09 per GB = 1,000 × $0.09 × 12 = $1,080 per year.
- Total: roughly **$8,276 per year**, or about **$9.45 per million decisions**.

**DynamoDB Streams.** Assume on-demand capacity and a Lambda consumer:

- Storage: 1.05 TB at $0.25 per GB-month = 1,050 × $0.25 × 12 = $3,150 per year.
- Stream reads: 876 million records at $0.02 per 100,000 reads = 8,760 × $0.02 = $175 per year.
- Lambda: 876 million invocations at $0.20 per million = $175, plus duration. At 128 MB and 100 ms average, GB-seconds = 876,000,000 × 0.1 × 0.125 = 10,950,000 GB-s. At $0.0000166667 per GB-s = $183. Total Lambda roughly **$358 per year**.
- Total: roughly **$3,683 per year**, or about **$4.20 per million decisions**.

The gap is large, but two caveats matter. First, the append-only figure assumes you already have Kubernetes or equivalent operational maturity; if you do not, the true cost includes the engineering time to run it, which usually dwarfs the infrastructure line. Second, the DynamoDB figure assumes on-demand pricing at modest write rates. Sustained high write throughput on DynamoDB is often cheaper with provisioned capacity, but provisioned capacity requires forecasting and can be expensive if you over-provision.

To get real numbers for your own workload, use `aws ce get-cost-and-usage` grouped by service over a representative month, and compare against a modelled append-only deployment in your own region. Do not trust anyone else's per-million figure, including the ones above.

## Choosing between them

| Question | Append-only hash-chained log | DynamoDB Streams |
|---|---|---|
| Does the record prove it is unaltered? | Yes, structurally, if the head hash is stored externally | No; relies on IAM and database controls |
| Global ordering guarantee | Yes, by construction | Per-partition-key only |
| Stream retention | Your retention policy | 24 hours |
| Restore granularity | Replay from any point in the log | PITR restores the table, window up to 35 days |
| Schema evolution | Versioned records; hash-input changes need coordination | Add an attribute, deploy the consumer |
| Operational burden | You run the log | Managed |
| Cost driver | Compute and replicated storage | Storage plus consumer compute |

A practical decision checklist:

1. **Does a regulator or auditor require cryptographic proof of non-alteration?** If yes, you need a hash chain, whether inside a dedicated log or built by a stream consumer into write-once storage. If no, this whole axis can be deprioritised.
2. **Do you need global ordering across all decisions?** If yes, per-partition ordering in DynamoDB Streams is not sufficient on its own; you need either a dedicated sequenced log or an explicit ordering field plus a consumer that respects it.
3. **How long can your consumer be down before you lose data?** If more than 24 hours is plausible, DynamoDB Streams alone will not retain the events. Persist them somewhere durable.
4. **What is your replay time budget during an incident?** Measure it, do not estimate. Write a script that reconstructs state from a representative slice and time it end to end, including any deserialisation and database writes.
5. **What is your team's operational capacity?** A self-managed log cluster is a real commitment. If nobody owns it, it will drift.
6. **What is your retention obligation?** Append-only logs grow forever unless you design retention. DynamoDB TTL is convenient but generates `REMOVE` records that your consumer must handle.

## A concrete replay consumer for DynamoDB Streams

The following script reads stream records into a local SQLite database for offline inspection. It is deliberately simple; note the comments about ordering and retention.

```python
import boto3
import sqlite3
import json

REGION = "us-east-1"
TABLE = "agent_decisions"

dynamodb = boto3.client("dynamodb", region_name=REGION)
conn = sqlite3.connect("decisions.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY,
        agent_id TEXT,
        decision_json TEXT,
        decision_time TEXT,
        stream_seq TEXT
    )
""")

# NOTE: this reads a single shard. A real consumer must enumerate all shards
# and process them independently; ordering is guaranteed only within a shard.
desc = dynamodb.describe_stream(TableName=TABLE)
shards = desc["StreamDescription"]["Shards"]
if not shards:
    raise SystemExit("no shards available; stream may be disabled")

iterator = dynamodb.get_shard_iterator(
    TableName=TABLE,
    ShardId=shards[0]["ShardId"],
    ShardIteratorType="TRIM_HORIZON",  # oldest retained record, max 24h back
)["ShardIterator"]

count = 0
while iterator is not None:
    batch = dynamodb.get_records(ShardIterator=iterator, Limit=1000)
    for rec in batch["Records"]:
        if rec["eventName"] == "REMOVE":
            # TTL and deletes both surface here; decide explicitly what to do.
            continue
        image = rec["dynamodb"].get("NewImage", {})
        conn.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?)",
            (
                image["id"]["S"],
                image["agent_id"]["S"],
                json.dumps(image.get("decision", {})),
                image["decision_time"]["S"],
                rec["dynamodb"].get("SequenceNumber", ""),
            ),
        )
        count += 1
    conn.commit()
    iterator = batch.get("NextShardIterator")
    if count >= 10000:
        break

print(f"replayed {count} decisions into decisions.db")
```

Two things this script deliberately does not do, and which a production consumer must: enumerate and checkpoint every shard, and handle the 24-hour retention boundary by falling back to a PITR restore or a separate archive when the iterator expires.

## Verifying a hash chain

If you build an append-only log, verification is the operation you will be asked to demonstrate. The shape is always the same:

1. Obtain a trusted head hash from outside the log's write path.
2. Fetch records from a known-good starting point to the head, in order.
3. Recompute each record's hash from its payload and its predecessor's hash.
4. Compare the final computed hash to the trusted head.

Time this on your own hardware. The number depends on payload size, serialisation format, and disk throughput, and it will be different from any figure quoted in an article. A useful measurement is records verified per second, plus the wall-clock time to verify a full day's worth of records. If that time exceeds your incident response budget, verify incrementally rather than from genesis.

## Frequently asked questions

**Can DynamoDB Streams serve as the audit log itself, without persisting records elsewhere?**

Only if a 24-hour retention window satisfies your requirement. Stream records are not retained beyond that, and a consumer outage longer than the window means permanent data loss for those events. For most audit requirements, the stream is a transport, not a store.

**Does point-in-time recovery let you replay stream events?**

No. PITR restores the table to a prior state. Streams capture changes going forward and are retained for 24 hours. They are separate mechanisms with separate windows.

**How do you handle GDPR erasure with an append-only log?**

Append-only and "delete this user's data" are in tension. The usual resolutions are to store personal data outside the log and reference it by an opaque identifier that can be deleted, or to encrypt per-subject data and discard the key. Both preserve the integrity of the log while making the personal data unrecoverable. Plan for this before you write the first record, not after the first erasure request.

**Is DynamoDB Streams cheaper than running your own log?**

On infrastructure line items, usually yes, and the worked example above shows why. The comparison changes if you already operate the infrastructure the log would run on, or if the engineering time to build and maintain a stream consumer exceeds the infrastructure delta. Model both with your own traffic and your own loaded engineering cost.

**What about Kinesis Data Streams?**

Kinesis is a general-purpose ordered log with configurable retention (up to 365 days) and much higher throughput ceilings than DynamoDB Streams. It is a reasonable choice when you need longer retention than 24 hours or when you are already building an event pipeline. It does not, by itself, give you tamper evidence; you would still need to hash-chain the records if that is a requirement.

## Your next 30 minutes

Pick one decision your system makes in production, then write a script that reconstructs that decision's full context from your current audit trail and times how long it takes. Run it against your production data, not a sample. If it takes longer than your incident response budget, or if it cannot reconstruct the context at all, you have found the gap that matters — and you now have a concrete number to bring to the decision about which architecture to adopt.
