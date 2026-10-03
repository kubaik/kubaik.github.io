# Log audit costs without breaking compliance

Audit logging sits in an awkward spot between two requirements that pull in opposite directions. Compliance regimes want durable, tamper-evident, queryable records that survive for months or years. Application teams want the write path to stay fast, because an audit write that blocks a user request turns a compliance feature into an outage. The failure mode that shows up most often is not missing encryption or a bad retention policy — it is write amplification and query latency discovered during an audit, when changing the architecture is expensive.

This article compares the two patterns that dominate the solution space, explains where each one hits a ceiling, and gives a way to decide between them without relying on vendor benchmarks. Every number below is either a documented default, arithmetic from stated assumptions, or explicitly labelled illustrative.

## The two dominant patterns

**Pattern A — PostgreSQL as system of record plus an audit table.** Audit rows are written into a dedicated table, usually partitioned or converted into a hypertable, and shipped off-box through logical replication or a change-data-capture pipeline. The appeal is operational: one database, one backup strategy, one access-control model, and SQL that every engineer already knows.

**Pattern B — PostgreSQL as transactional source, a columnar store as the analytical sink.** Raw events land in an append-only columnar table (ClickHouse is the common choice, though the category includes other columnar engines). Materialized views roll raw events into compliance-shaped tables. The appeal is query speed across wide date ranges and the ability to store raw payloads without flattening them up front.

Both can satisfy SOC 2 Type II, ISO 27001, and financial-sector operational resilience requirements. They fail differently, and they fail at different volumes.

## Pattern A: PostgreSQL with an audit table

The usual implementation is a `SECURITY DEFINER` trigger function attached to every table that needs auditing, writing into a single partitioned `audit_log` table. Logical replication then publishes that table to a downstream consumer that writes immutable files to object storage.

A workable schema, using only core PostgreSQL plus the `pgcrypto` extension for hashing:

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE audit_log (
  id BIGSERIAL,
  event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id TEXT NOT NULL,
  action TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  old_data JSONB,
  new_data JSONB,
  ip_address INET,
  user_agent TEXT,
  prev_hash BYTEA,
  row_hash BYTEA,
  PRIMARY KEY (id, event_time)
) PARTITION BY RANGE (event_time);

CREATE TABLE audit_log_2026_01
  PARTITION OF audit_log
  FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

Note two deliberate choices. First, the partition key is part of the primary key, which PostgreSQL requires for partitioned tables. Second, the hash is not a generated column. A generated column cannot reference other rows, so it cannot express a hash chain. A chain has to be computed in application code or in a trigger that reads the previous row's hash:

```sql
CREATE OR REPLACE FUNCTION trg_audit() RETURNS TRIGGER AS $$
DECLARE
  _prev BYTEA;
  _payload TEXT;
BEGIN
  SELECT row_hash INTO _prev
  FROM audit_log
  ORDER BY id DESC
  LIMIT 1;

  _payload := concat_ws('|',
    TG_OP,
    TG_TABLE_NAME,
    coalesce(to_jsonb(NEW)::text, to_jsonb(OLD)::text),
    encode(coalesce(_prev, ''::bytea), 'hex'));

  INSERT INTO audit_log (
    user_id, action, entity_type, entity_id,
    old_data, new_data, ip_address, user_agent, prev_hash, row_hash
  ) VALUES (
    current_user, TG_OP, TG_TABLE_NAME,
    coalesce(NEW.id, OLD.id)::text,
    CASE WHEN TG_OP = 'INSERT' THEN NULL ELSE to_jsonb(OLD) END,
    CASE WHEN TG_OP = 'DELETE' THEN NULL ELSE to_jsonb(NEW) END,
    inet_client_addr(),
    current_setting('application_name', true),
    _prev,
    digest(_payload, 'sha256')
  );
  RETURN coalesce(NEW, OLD);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER trg_user_audit
  AFTER INSERT OR UPDATE OR DELETE ON users
  FOR EACH ROW EXECUTE FUNCTION trg_audit();
```

Two caveats worth stating plainly. `inet_client_addr()` returns NULL for local connections, so the trigger above will record NULL rather than a fabricated loopback address. And the `SELECT ... ORDER BY id DESC LIMIT 1` inside the trigger serializes concurrent inserts on the audit table — that is the price of a strict hash chain. If strict chaining is not required, drop `prev_hash` and compute a per-row hash instead; if it is required, expect the audit table to become a write bottleneck and plan accordingly.

### Where Pattern A breaks

The dominant cost is write amplification. Every audited transaction produces at least one extra row, and if the row is then replicated, the same bytes are written again to the WAL and again on the subscriber. A single logical replication slot streams changes from one publication with one worker process; the documented behavior is that a slot is consumed by one active consumer at a time, so scaling the publisher means adding publications and slots, not threads.

The second cost is `ALTER TABLE`. Adding a column to a large partitioned table takes an access-exclusive lock on each partition in turn, and the audit table is usually the largest table in the database.

The third cost is retention. Dropping a partition is cheap; deleting rows with `DELETE` is not, and an audit table with a rolling delete job will bloat unless autovacuum keeps up.

## Pattern B: a columnar analytical sink

The columnar pattern treats the transactional database as the source of truth and the analytical store as an immutable, query-optimized copy. Events are written once, in append-only fashion, and never updated in place.

A representative schema:

```sql
CREATE TABLE events_raw (
  event_time DateTime64(3),
  user_id String,
  action LowCardinality(String),
  entity_type LowCardinality(String),
  entity_id String,
  old_data String,
  new_data String,
  ip_address IPv4,
  user_agent String,
  row_hash String,
  source_date Date MATERIALIZED toDate(event_time)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (toStartOfHour(event_time), entity_type, user_id)
TTL event_time + INTERVAL 366 DAY;
```

Ingestion is typically batched: either a client inserting `JSONEachRow` in blocks of a few thousand rows, or a Kafka sink connector batching on size and time. Batching is not an optimization here — it is mandatory. Columnar engines write one part per insert, and a high rate of tiny inserts produces a backlog of parts that background merges cannot keep up with. The documented remedy is to batch, or to enable asynchronous inserts so the server coalesces small writes.

### Where Pattern B breaks

- **Small inserts.** A stream of single-row inserts will stall merges and degrade read performance. This is the most common operational mistake with columnar audit stores.
- **Late-arriving and corrected events.** Columnar engines are append-only in practice; updates are expensive. If an audit record must be corrected, the correction is a new row, and every query has to account for that.
- **Coordination.** A production cluster needs a coordination service for replication and distributed DDL. That is a second stateful system to operate, back up, and upgrade.
- **Deletes.** Retention via TTL is cheap because it drops whole partitions. Point deletes for a data-subject request are not, and columnar engines generally do not offer row-level security equivalent to PostgreSQL's.

## Comparing the two honestly

Do not trust throughput numbers from an article, including this one. The table below lists what to measure and how, not what the answer is.

| Dimension | What to measure | How to measure it |
|---|---|---|
| Sustained write throughput | Events/sec at a fixed p99 insert latency | Replay a captured production event stream at increasing rates; record the rate at which p99 latency crosses your SLO |
| Write amplification | Extra bytes written per event | Compare `pg_stat_user_tables.n_tup_ins` growth for audited tables against application-level event counts; compare WAL generation rate (`pg_stat_wal`) before and after enabling auditing |
| Point-in-time query latency | Time to return N rows for a `(user_id, entity_type, time range)` predicate | Run the same query shape against both stores on identical data; vary the range from one day to thirty |
| Storage per million events | Bytes on disk after compression | Insert a fixed synthetic dataset, force a merge or vacuum, and measure the on-disk size |
| Retention cost | Cost of dropping one month of data | Time the partition drop or TTL expiry and measure the CPU spike it causes |
| Recovery time | Time to restore and replay to a point in time | Restore from backup in a staging environment and measure end to end |

A few structural differences are worth stating without numbers:

- **Partition pruning.** Both systems prune by time. A columnar store additionally prunes by the sort key within a partition, which is why wide-range scans with a selective predicate tend to return faster there.
- **Schema changes.** Adding a column is a metadata operation in a columnar engine and a locking operation in PostgreSQL. For an audit table that grows monotonically, this difference compounds.
- **Access control.** PostgreSQL offers row-level security, which matters when the audit table lives in the same database as tenant data. Columnar stores generally implement access control at the database or table level.
- **Backup and restore.** PostgreSQL backup tooling is mature and well understood. Columnar stores usually back up to object storage and restore by copying parts, which is fast but less familiar to on-call engineers.

## A cost model you can fill in yourself

Infrastructure cost comparisons age badly. Instead, build the model from four inputs and recompute it when any input changes.

1. **Peak events per second.** Take the 99th percentile over a month, not the average.
2. **Bytes per event after compression.** Measure it, do not estimate.
3. **Retention window in months.** This is usually set by policy, not by engineering.
4. **Replication factor.** One for a single-node analytical store, three for a typical replicated cluster.

Illustrative arithmetic: at 2,000 events/sec sustained, one year holds roughly `2000 × 60 × 60 × 24 × 365 ≈ 6.3 × 10^10` events. At 200 bytes per event after compression, that is about 12.6 TB before replication. At a replication factor of three, 37.8 TB. Those figures are illustrative — substitute your own measured bytes-per-event and retention window, and the same multiplication gives you a storage budget you can defend.

Then compare the operational side. A columnar cluster is at least one additional stateful system to run, plus a coordination service in a distributed deployment. A PostgreSQL audit table is zero additional systems but adds write load to the database that already serves user traffic. The right comparison is not "which is cheaper" but "which failure mode can this team absorb at 3 a.m."

## Decision checklist

Work through these in order; the first few usually settle it.

- **What is peak sustained event rate, not average?** Below roughly 2,000/sec, a well-partitioned PostgreSQL audit table is usually sufficient and adds no new systems.
- **Does the audit query pattern require wide date ranges with selective filters?** If auditors routinely ask for a single user across ninety days, a columnar store will answer that faster and with less load on the transactional database.
- **Is a strict hash chain required, or only per-row integrity?** A strict chain serializes writes. If the requirement is per-row tamper evidence, avoid the chain and keep the write path parallel.
- **Who is on call?** Adding a columnar cluster adds a second system with its own failure modes. If the team has no one who has operated one, that is a real cost.
- **How often do audit schemas change?** Frequent schema evolution favors a columnar store or a schemaless raw-event table.
- **What does the auditor actually accept?** Ask before designing. Daily signed exports to object storage satisfy many regimes; point-in-time interactive queries satisfy a smaller set.

## Failure modes to design against

**Write amplification discovered late.** The audit trigger doubles or triples write volume on the busiest tables. Instrument `pg_stat_user_tables` and WAL generation before enabling auditing on a production table, not after.

**Hash chain as a bottleneck.** A trigger that reads the previous row's hash serializes inserts. Measure the throughput ceiling in staging with realistic concurrency; if it is below your peak event rate, use per-row hashes and rely on signed exports for chain integrity.

**Small inserts into a columnar store.** Batch on the client or enable server-side asynchronous inserts. Monitor the parts count; a rising count of small parts is the early warning.

**Retention that does not actually delete.** Verify that partition drops or TTL expiry are happening, and measure the CPU cost of the merge or drop. A retention policy that exists only in a document is not a retention policy.

**No restore rehearsal.** An audit log that cannot be restored to a point in time is not evidence. Restore into staging on a schedule and record the elapsed time.

## FAQ

**Can PostgreSQL alone handle audit logging at scale?**
Yes, up to a point. The constraint is not PostgreSQL's storage engine but the write amplification of the audit path and the single-consumer behavior of logical replication slots. Partitioning by time, avoiding a strict hash chain, and archiving partitions to object storage extends the ceiling considerably.

**Do I need Kafka to feed a columnar audit store?**
No. A batch writer that reads from the transactional database and inserts in blocks of a few thousand rows works for moderate volumes. Kafka becomes worthwhile when you already operate it or when you need multiple independent consumers of the same event stream.

**How do I prove an audit log has not been tampered with?**
Two commonly used approaches: per-row hashes stored alongside the row, or a hash chain where each row includes the previous row's hash. Both are only as strong as the storage of the hashes themselves, so the usual design also writes signed, immutable exports to object storage on a schedule, where the retention policy prevents modification.

**Is a columnar store a replacement for the transactional database?**
No. It is an analytical sink. The transactional database remains the source of truth for application state; the columnar store holds the audit record of changes to that state.

**What is the single most useful thing to measure first?**
Write amplification: extra rows or bytes written per application event. It determines whether the current design will survive the next traffic increase, and it is measurable in a few minutes.

## Do this in the next 30 minutes

Run this against a staging copy of your production database and record the output before you change anything:

```sql
SELECT
  schemaname,
  relname,
  n_tup_ins,
  n_tup_upd,
  n_tup_del,
  CASE
    WHEN n_tup_ins = 0 THEN NULL
    ELSE round((n_tup_ins + n_tup_upd + n_tup_del)::numeric / n_tup_ins, 3)
  END AS rows_written_per_insert
FROM pg_stat_user_tables
WHERE relname LIKE '%audit%'
ORDER BY n_tup_ins DESC;
```

A ratio close to 1 means the audit path is writing roughly one row per insert. A ratio well above 1 means the audit path is multiplying write volume, and that number — not a vendor benchmark — is the one to bring to the next architecture discussion.
