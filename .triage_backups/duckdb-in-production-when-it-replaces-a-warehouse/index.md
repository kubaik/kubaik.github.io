# DuckDB in production: when it replaces a warehouse

## The decision this article is about

Teams reach for a full data warehouse — Snowflake, BigQuery, Redshift — before they need one. The trigger is usually a single dashboard that has to scan a few million rows, or a nightly report that takes 40 minutes on a Postgres replica. So they stand up a warehouse, wire up an ELT pipeline, and inherit a class of problems: network round trips, per-query billing, a separate deployment surface, and a credential model that has to be audited. For datasets that fit on one machine's disk, that is a lot of operational weight for very little analytical gain.

DuckDB is an in-process OLAP database. It runs inside your Python, Node, or Go process the same way SQLite does, but its storage and execution engine are columnar and vectorized. The part that trips people up is not query speed; it's knowing where the embedded model stops being appropriate and a real warehouse starts. That boundary is what this article covers.

The claim: for analytical workloads under roughly 100 GB of compressed data with a single writer, an embedded columnar engine can replace a warehouse for that workload entirely, with less code, fewer moving parts, and predictable cost. Above that line, or once multiple services need to write concurrently, the calculus changes.

## The comparison that decides it

Before writing any code, evaluate each concern below against your actual workload. The table is the decision.

| Concern | Cloud warehouse | Embedded columnar engine |
|---|---|---|
| Data location | Remote managed storage | Local file, or object storage via an HTTP filesystem extension |
| Latency floor | Network + scheduling overhead per query | Local scan; no network hop |
| Cost model | Per-query or per-slot-hour | Fixed compute; storage is your disk |
| Concurrency | Many readers, many writers | Many readers, one writer |
| Ops surface | IAM, warehouse sizing, clustering keys | One process, one file |
| Sweet spot | Multi-team, high concurrency, very large data | Single service, embedded, data that fits one machine |

If every row lands on the right, the embedded engine wins on simplicity and cost. If multiple services write concurrently, you are on the left.

Two caveats on the table. First, "latency floor" is not a fixed number — it depends on your network path and the warehouse's queue depth at query time. Second, "sweet spot" for size is not a hard limit; it is a practical ceiling discussed in the sizing section below.

## Prerequisites and what you'll build

You'll build a small analytics service that ingests Parquet exports from an operational Postgres database, stores them as partitioned Parquet, and serves queries over an HTTP endpoint. The whole thing runs as one process. No cluster, no coordinator, no network hop between the query engine and the data.

What you need:

- Python 3.11 or newer
- A recent DuckDB release — pin it, because the storage format has changed across minor versions
- `pyarrow` for Parquet handling
- An ASGI web framework (FastAPI is used below) and an ASGI server if you want the HTTP layer
- A machine with at least 2x your dataset size in free disk and enough RAM to hold your largest working set

What you're building, concretely:

1. A loader that reads Parquet files from object storage and exposes them as a view.
2. A query layer that exposes parameterized analytical queries.
3. A background refresh that rebuilds a materialized summary table every 15 minutes.
4. Tests that assert aggregates against the source.

## Step 1 — set up the environment

DuckDB's performance depends heavily on how the data is laid out. If the layout is wrong — for example, letting the database grow in the default single-file mode without partitioning — you'll get acceptable results on small data and then a cliff at scale. Get the storage layout right first.

Create a virtual environment and pin your versions:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "duckdb" "pyarrow" "fastapi" "uvicorn"
```

Pin exact versions in your `requirements.txt` or lockfile. The DuckDB storage format has changed between minor releases, and a file written by one version may not open cleanly in another.

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

The pattern that works is a read-only query connection plus a separate write connection used only by the refresh job. DuckDB allows one writer at a time, and if you let the HTTP handlers open write connections you'll hit `IO Error: Could not set lock on file` under concurrent load.

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

The summary table is a real DuckDB table, so queries against it are local and fast — typically 5–20 ms for a dashboard that would have been 400 ms against raw Parquet on S3. You trade 15 minutes of staleness for a large latency improvement. For most dashboards that's the right trade; for anything that needs real-time counts, query the raw view.

## Step 3 — handle edge cases and errors

The failure modes cluster around three things: memory, concurrency, and schema drift.

**Memory.** DuckDB will try to materialize a large intermediate result and get killed by the OOM killer. The error you see from the kernel is a `SIGKILL` with no Python traceback, which makes it look like a mystery crash rather than a memory problem. The fix is `SET memory_limit` plus `SET temp_directory` so that spilling goes to disk instead of RAM:

```python
con.execute("SET memory_limit = '6GB';")
con.execute("SET temp_directory = '/var/tmp/duckdb_spill';")
con.execute("SET max_temp_directory_size = '50GB';")
```

Spilling is slower — expect a 3–5x slowdown on queries that spill — but it's the difference between a slow query and a dead process.

**Concurrency.** The single-writer limitation is real. If two services both need to write, you have three options: serialize writes through one service, use a queue, or move to a client-server database. DuckDB is not the right tool for multi-writer workloads, and trying to make it one via file locking will produce intermittent `IO Error: Could not set lock on file` failures that are hard to reproduce.

**Schema drift.** Parquet files written by an upstream job that added a column will cause DuckDB to infer a union schema, and if the new column has a conflicting type in older files you get `Binder Error: Referenced column not found` or a type mismatch on read. The robust pattern is to project explicit columns rather than `SELECT *`:

```sql
SELECT tenant_id, event_time, event_type, payload_json
FROM read_parquet('s3://my-bucket/events/**/*.parquet', hive_partitioning = true)
```

Explicit projection also means DuckDB only reads the column chunks it needs, which on a wide table can cut I/O substantially. The exact reduction depends on the ratio of projected columns to total columns; on a table with 40 columns where you read 8, the scan reads roughly one fifth of the column data.

## Step 4 — add observability and tests

An embedded database has no server-side query log by default. If you don't instrument it, you have no idea which queries are slow. DuckDB has a built-in profiler you can enable per-query.

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

The two asserted numbers are illustrative — replace them with the aggregates of your own fixture set. Run this in CI against a small fixture set. It takes under a second and it will catch the class of bug where a partition glob silently stops matching after a directory rename.

## How to benchmark this for your own workload

Rather than trust published numbers, measure. The workflow takes about 30 minutes.

1. Export a representative slice to Parquet:

```sql
COPY (SELECT * FROM your_table WHERE event_time >= '2026-01-01') TO 'sample.parquet' (FORMAT PARQUET);
```

2. Record the sample's size on disk and its row count.
3. Enable the profiler and run your largest analytical query against the sample.
4. Read the profile JSON and note wall-clock time, peak memory, and whether any operator spilled.
5. Extrapolate linearly to full size, then multiply by 1.5–2x to account for the non-linear effects of spilling and cache misses. If the extrapolated number is under two seconds, the workload is a strong candidate for an embedded engine.

What to instrument in production once you deploy:

- Per-query wall-clock time, tagged by route and tenant.
- Peak memory per query, from the profile JSON.
- Spill volume per query — if this is nonzero on a hot path, your `memory_limit` is too low or your query needs a pre-aggregation.
- Process RSS, to confirm the container limit is not being approached.

Compare those numbers against the same queries running against your current warehouse, measured the same way. The comparison is what justifies the migration, not a vendor's published benchmark.

## Sizing and where the model breaks

The practical ceiling for a good experience on one machine is roughly 100–200 GB of compressed data, depending on disk speed and query complexity. The limiting factors are:

- **Disk throughput.** A full scan of 200 GB at 2 GB/s takes 100 seconds before any computation. If your queries scan most of the dataset, you are disk-bound regardless of how fast the engine is.
- **Spill volume.** Once working sets exceed `memory_limit`, queries spill to `temp_directory`. Spilling is correct but slow, and the slowdown compounds with the number of concurrent queries.
- **Write throughput.** Single-writer means one refresh job at a time. If your refresh takes longer than the interval between refreshes, you have a scheduling problem no amount of tuning fixes.

Where an embedded engine stops working:

- More than one writer, or writers in more than one process.
- Datasets that exceed what one machine's disk can hold with room to spare.
- Per-query isolation between teams — an embedded engine has no query scheduler, so one expensive query competes with every other query in the process.
- Regulatory requirements that mandate a managed service with an audit trail.

At that point the warehouse earns its cost. The decision is not "which is better" but "which constraint binds first."

## Common questions

**Can DuckDB handle concurrent reads?**
Yes. Multiple read-only connections against the same file are fine, and in-process connections from multiple threads are safe. The limitation is writers: exactly one process can hold a write lock at a time. If you need concurrent writes, either serialize them through a single service or use a client-server database.

**How does DuckDB compare to SQLite for analytics?**
SQLite is a row-store optimized for transactional point lookups. DuckDB is a column-store optimized for scans and aggregations. On a large aggregation, DuckDB is typically much faster because it reads only the columns the query touches and uses vectorized execution. For a single-row lookup by primary key, SQLite is faster. They solve different problems.

**What's the largest dataset DuckDB handles well?**
It scales to datasets larger than RAM via spilling, but the practical ceiling for a good experience on one machine is around 100–200 GB compressed, depending on disk speed and query complexity. Past that, the spill volume makes queries slow enough that a distributed engine wins.

**Does DuckDB work with Parquet on S3 directly?**
Yes, through the `httpfs` extension. You query `read_parquet('s3://...')` and DuckDB handles range requests. The catch is that latency to S3 adds up: a query touching 200 files pays 200 round trips. Partition pruning and explicit column projection reduce this, and pre-aggregating into a local table eliminates it entirely.

**What about schema evolution in the source?**
DuckDB infers a union schema across files. Additive changes are usually fine; conflicting types across files are not. The mitigation is explicit column projection plus a test that asserts the expected columns exist and have the expected types before the refresh job runs.

## The move to make in the next 30 minutes

Take your largest analytical query, export a representative slice to Parquet, and time it in DuckDB with the profiler enabled:

```sql
COPY (SELECT * FROM your_table LIMIT 10000000) TO 'sample.parquet' (FORMAT PARQUET);
```

```python
import duckdb
con = duckdb.connect()
con.execute("PRAGMA enable_profiling = 'json';")
con.execute("PRAGMA profiling_output = '/tmp/profile.json';")
con.execute(open("your_query.sql").read()).fetchall()
```

Open `/tmp/profile.json` and look at the total wall-clock time and the peak memory. If the query completes in under two seconds on a sample that's roughly 10% of production size, you have a concrete, defensible case for removing a warehouse dependency from that workload — and you'll have the profile JSON to prove it.
