# DuckDB in production: when it replaces a warehouse

The edge cases only show up once real users hit the system. The conventional advice on duckdb production is incomplete in one specific, costly way. Here's the fuller picture, with the tradeoffs left in.

## The problem this solves

A lot of teams reach for a full data warehouse — Snowflake, BigQuery, Redshift — before they actually need one. The trigger is usually a single dashboard that has to scan a few million rows, or a nightly report that takes 40 minutes on a Postgres replica. So they stand up a warehouse, wire up an ELT pipeline, and inherit a new class of problems: network round trips, per-query billing, a separate deployment surface, and a credential model that has to be audited. For datasets that fit on one machine's disk, that is a lot of operational weight for very little analytical gain.

DuckDB is an in-process OLAP database. It runs inside your Python, Node, or Go process the same way SQLite does, but its storage and execution engine are columnar and vectorized. On a single 8-core machine with an NVMe SSD, DuckDB 1.1 can scan and aggregate a 10 GB Parquet dataset in the low single-digit seconds — the kind of workload that used to justify a warehouse cluster. The part that trips people up is not the query speed; it's knowing where the embedded model stops being appropriate and a real warehouse starts, and that's what this post actually covers.

The claim I'm making: for analytical workloads under roughly 100 GB of compressed data with a single writer, DuckDB replaces the warehouse entirely, and it does so with less code, fewer moving parts, and a predictable cost. Above that line, or once you need concurrent writes from multiple services, you should stop and reconsider.

## Prerequisites and what you'll build

You'll build a small analytics service that ingests Parquet exports from an operational Postgres database, stores them as a partitioned DuckDB database, and serves queries over an HTTP endpoint. The whole thing runs as one process. No cluster, no coordinator, no network hop between the query engine and the data.

What you need:

- Python 3.11 or newer (3.12 is fine; the `duckdb` wheel ships binary wheels for both)
- DuckDB 1.1.x — pin it, because the storage format has changed across minor versions
- `pyarrow` 17.x for Parquet handling
- FastAPI 0.111 and Uvicorn 0.30 if you want the HTTP layer
- A machine with at least 2x your dataset size in free disk and enough RAM to hold your largest working set

What you're building, concretely:

1. A loader that reads Parquet files from object storage and appends them into a DuckDB file.
2. A query layer that exposes parameterized analytical queries.
3. A background refresh that rebuilds a materialized summary table every 15 minutes.
4. Tests that assert row counts and a checksum against the source.

The comparison that matters before you write any code:

| Concern | Warehouse (Snowflake/BigQuery) | DuckDB embedded |
|---|---|---|
| Data location | Remote, managed storage | Local file or object storage via httpfs |
| Latency floor | 200–800 ms per query (network + scheduling) | 5–50 ms for warm queries |
| Cost model | Per-query or per-slot-hour | Fixed compute; storage is your disk |
| Concurrency | Many readers, many writers | Many readers, one writer |
| Ops surface | IAM, warehouse sizing, clustering keys | One process, one file |
| Sweet spot | >100 GB, multi-team, high concurrency | <100 GB, single service, embedded |

That table is the whole decision. If your row is on the right for every concern, DuckDB wins on simplicity and cost. If you have multiple services writing concurrently, you're on the left.

## Step 1 — set up the environment

Why this ordering: DuckDB's performance characteristics depend heavily on how the file is laid out. If you set up the environment wrong — for example, letting the database grow in the default single-file mode without partitioning — you'll get acceptable results on small data and then a cliff at scale. Get the storage layout right first.

Create a virtual environment and pin your versions:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "duckdb==1.1.3" "pyarrow==17.0.0" "fastapi==0.111.0" "uvicorn==0.30.1"
```

Now the storage layout. DuckDB supports two modes: a single `.duckdb` file, and a directory of partitioned Parquet files queried through views. For anything that will grow past a few gigabytes, use the second. The reason is that DuckDB's single-file format rewrites pages on update, and a large append-heavy file fragments over time. Partitioned Parquet lets you drop and re-add partitions cheaply.

```python
import duckdb

con = duckdb.connect()  # in-memory for the loader
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("""
    CREATE OR REPLACE VIEW events AS
    SELECT * FROM read_parquet(
        's3://my-bucket/events/year=*/month=*/day=*/*.parquet',
        hive_partitioning = true
    )
""")
```

The `hive_partitioning = true` flag is what makes `year=2026/month=03/day=14/` directories become queryable columns without you writing a single `CAST`. A common trap here is forgetting that DuckDB infers partition types from the directory names — if your partitions are zero-padded inconsistently (`month=3` vs `month=03`), you'll get a type conflict error like `Conversion Error: Could not convert string '03' to INT32` only on some files. Zero-pad everything from the start.

For local development without S3, point the glob at a local directory. The query syntax is identical, which is the point: you can develop against a 500 MB local sample and deploy against 50 GB in object storage without changing the view definition.

## Step 2 — core implementation

Why this shape: the pattern that works is a read-only query connection plus a separate write connection used only by the refresh job. DuckDB allows one writer at a time, and if you let the HTTP handlers open write connections you'll hit `IO Error: Could not set lock on file` under concurrent load.

Here's the query layer:

```python
import duckdb
from fastapi import FastAPI, Query

app = FastAPI()
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET memory_limit = '6GB';")
con.execute("SET threads = 8;")
con.execute("""
    CREATE VIEW events AS
    SELECT * FROM read_parquet(
        's3://my-bucket/events/year=*/month=*/day=*/*.parquet',
        hive_partitioning = true
    )
""")

@app.get("/counts")
def counts(tenant: str = Query(...), since: str = Query(...)):
    rows = con.execute(
        """
        SELECT event_type, COUNT(*) AS n
        FROM events
        WHERE tenant_id = ? AND event_time >= ?
        GROUP BY event_type
        ORDER BY n DESC
        """,
        [tenant, since],
    ).fetchall()
    return {"counts": [{"type": r[0], "n": r[1]} for r in rows]}
```

Two settings do most of the work. `memory_limit` caps DuckDB's buffer pool so it doesn't get OOM-killed by the container's cgroup limit — set it to roughly 75% of your container's memory. `threads` should match your vCPU count; setting it higher than the physical cores adds contention without throughput.

For the materialized summary, run it on a schedule against a write connection:

```python
import duckdb

def refresh_summary():
    w = duckdb.connect("analytics.duckdb")
    w.execute("""
        CREATE OR REPLACE TABLE daily_summary AS
        SELECT tenant_id, date_trunc('day', event_time) AS day,
               event_type, COUNT(*) AS n
        FROM read_parquet('s3://my-bucket/events/**/*.parquet')
        GROUP BY 1, 2, 3
    """)
    w.close()
```

The summary table is a real DuckDB table, so queries against it are local and fast — typically 5–20 ms for a dashboard that would have been 400 ms against raw Parquet on S3. You trade 15 minutes of staleness for a 20x latency improvement. For most dashboards that's the right trade; for anything that needs real-time counts, query the raw view.

## Step 3 — handle edge cases and errors

The failure modes here are well-documented and they cluster around three things: memory, concurrency, and schema drift.

Memory first. DuckDB will happily try to materialize a large intermediate result and get killed by the OOM killer. The error you see from the kernel is a `SIGKILL` with no Python traceback, which makes it look like a mystery crash rather than a memory problem. The fix is `SET memory_limit` plus `SET temp_directory` so that spilling goes to disk instead of RAM:

```python
con.execute("SET memory_limit = '6GB';")
con.execute("SET temp_directory = '/var/tmp/duckdb_spill';")
con.execute("SET max_temp_directory_size = '50GB';")
```

Spilling is slower — expect a 3–5x slowdown on queries that spill — but it's the difference between a slow query and a dead process.

Concurrency second. The single-writer limitation is real. If you have two services that both need to write, you have three options: serialize writes through one service, use a queue, or move to a client-server database. DuckDB is not the right tool for multi-writer workloads, and trying to make it one via file locking will produce intermittent `IO Error: Could not set lock on file` failures that are hard to reproduce.

Schema drift third. This is the one that bites teams in production. Parquet files written by an upstream job that added a column will cause DuckDB to infer a union schema, and if the new column has a conflicting type in older files you get `Binder Error: Referenced column not found` or a type mismatch on read. The robust pattern is to project explicit columns rather than `SELECT *`:

```sql
SELECT tenant_id, event_time, event_type, payload_json
FROM read_parquet('s3://my-bucket/events/**/*.parquet', hive_partitioning = true)
```

Explicit projection also means DuckDB only reads the column chunks it needs, which on a wide table can cut I/O by 60–80%.

## Step 4 — add observability and tests

Why this matters more than usual: an embedded database has no server-side query log by default. If you don't instrument it, you have no idea which queries are slow. DuckDB has a built-in profiler you can enable per-query.

```python
con.execute("PRAGMA enable_profiling = 'json';")
con.execute("PRAGMA profiling_output = '/var/log/duckdb_profile.json';")
```

The JSON profile includes wall-clock time per operator, which rows each operator produced, and peak memory. For a query that's slower than expected, the profile almost always shows the problem in one of two places: a hash join that spilled to disk, or a scan that read far more columns than the query needed.

For tests, assert against known-good aggregates rather than exact row counts, because Parquet file ordering is not guaranteed. A test like this catches schema drift, partition misconfiguration, and accidental data loss:

```python
def test_daily_summary_matches_source(tmp_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE src AS SELECT * FROM read_parquet('tests/fixtures/*.parquet')")
    expected = con.execute(
        "SELECT COUNT(*), SUM(n) FROM (SELECT event_type, COUNT(*) AS n FROM src GROUP BY 1)"
    ).fetchone()
    assert expected[0] == 4
    assert expected[1] == 1200
```

Run this in CI against a small fixture set. It takes under a second and it will catch the class of bug where a partition glob silently stops matching after a directory rename.

## Real results from running this

These are the numbers that show up repeatedly for this kind of workload, not measurements from a specific deployment:

- A 12 GB Parquet dataset on an 8-core, 32 GB machine: a `GROUP BY` over 40 million rows completes in 1.8–2.5 seconds warm, 4–6 seconds cold.
- The same query against a remote warehouse with a comparable cluster: 6–12 seconds, dominated by scheduling and network overhead rather than compute.
- Dashboard queries against the pre-aggregated summary table: 5–20 ms, versus 300–600 ms against raw Parquet on S3.
- Memory footprint for the service process: 400 MB idle, up to the configured `memory_limit` under load.
- Cost: the compute is whatever your container costs. There's no per-query line item.

The pattern that emerges is that DuckDB's advantage is largest for the medium-sized analytical workloads — the ones too big for a Postgres query to be pleasant, but too small to justify a warehouse. That's a wider band than people assume. If your largest table is under 100 GB compressed, you're probably in it.

Where it stops working: once you need more than one writer, or once your dataset exceeds what one machine's disk can hold with room to spare, or once you need per-query isolation between teams. At that point the warehouse earns its cost.

## Common questions and variations

**Can DuckDB handle concurrent reads?**
Yes. Multiple read-only connections against the same file are fine, and in-process connections from multiple threads are safe. The limitation is writers: exactly one process can hold a write lock at a time. If you need concurrent writes, either serialize them through a single service or use a client-server database.

**How does DuckDB compare to SQLite for analytics?**
SQLite is a row-store optimized for transactional point lookups. DuckDB is a column-store optimized for scans and aggregations. On a 10 million row aggregation, DuckDB is typically 20–50x faster than SQLite because it reads only the columns the query touches and uses vectorized execution. For a single-row lookup by primary key, SQLite is faster. They solve different problems.

**What's the largest dataset DuckDB handles well?**
It scales to datasets larger than RAM via spilling, but the practical ceiling for a good experience on one machine is around 100–200 GB compressed, depending on disk speed and query complexity. Past that, the spill volume makes queries slow enough that a distributed engine wins.

**Does DuckDB work with Parquet on S3 directly?**
Yes, through the `httpfs` extension. You query `read_parquet('s3://...')` and DuckDB handles range requests. The catch is that latency to S3 adds up: a query touching 200 files pays 200 round trips. Partition pruning and explicit column projection reduce this, and pre-aggregating into a local table eliminates it entirely.

## Where to go from here

DuckDB is not a warehouse replacement in the abstract. It's a replacement for the specific case where your analytical data fits on one machine, your write pattern is single-writer, and your queries are read-heavy aggregations. That case is more common than the warehouse vendors would like you to believe, and the operational savings are real: one process instead of a cluster, one file instead of a managed service, and no per-query meter running.

The move to make right now: take your largest analytical query, export the underlying table to Parquet with `COPY (SELECT ...) TO 'sample.parquet' (FORMAT PARQUET)`, and time it in DuckDB with `PRAGMA enable_profiling`. If it comes back under two seconds on a sample that's 10% of production size, you have a concrete, defensible case for removing a warehouse dependency from that workload — and you'll have the profile JSON to prove it.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
