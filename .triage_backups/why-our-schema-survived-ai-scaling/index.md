# Designing a Schema That Survives Embedding Model Upgrades

Most write-ups about vector search stop exactly where the interesting part starts: at the first working query. The harder problem is what happens six months later, when the embedding model changes, the dimension count shifts, and a table that has served production traffic for a year starts rejecting writes.

This article is about the data-modeling decisions that determine whether a relational schema survives that transition. The focus is deliberately narrow: how embeddings are stored next to transactional data, how they are versioned, and how to tell whether the design is holding up.

## The error, and why the message misleads

A common failure looks like a generic PostgreSQL type error:

```
psycopg2.errors.DataError: column "embedding" has type "vector" but expression is of type "jsonb"
LINE 4: INSERT INTO transaction_embeddings (transaction_id, embedding) VALUES ($1, $2)
```

The message points at a type conflict. The actual cause is usually a modeling decision: an application started persisting the raw JSON response from an embedding service into a column declared as `vector(N)`, or a model upgrade changed the output dimensionality while the column kept its original width. The database then either rejects the insert or attempts an implicit cast that costs more than it looks like it should.

The same class of error appears in several disguises:

- A dimension mismatch raised by the client library rather than the database, because the SDK validates length before sending.
- Inserts succeeding but reads returning nothing, because old rows have a different effective dimension than new query vectors.
- Table growth that outpaces the row count, because raw JSON payloads were stored as text alongside the vector.

None of these are exotic. They are the predictable result of treating an embedding as "just another column."

## The three decisions that cause it

### 1. A `vector` column with no version tag

The `vector` type enforces a fixed dimension. If the column is declared `vector(1536)` and a later model returns 768 floats, every insert carrying the new model's output fails. There is no ambiguity in the database's behavior here — the type is doing exactly what it was told.

The problem is that the schema records the dimension but not *which model* produced the values. Two models can share a dimension and still produce incompatible vector spaces; cosine similarity between a vector from model A and a vector from model B is meaningless even when the arithmetic succeeds. Storing the dimension alone is not enough to make the table self-describing.

### 2. Raw API responses kept "for debugging"

A common early pattern is to store the entire JSON response in a `jsonb` column so nothing is lost. This is defensible during prototyping. It becomes a liability when the table reaches production scale, because:

- JSON payloads are far larger than the vectors they contain, inflating row size and the cost of every sequential scan.
- Queries against the vector column must cast or extract from JSON at runtime if the two representations coexist.
- The redundant payload drifts out of sync with the extracted vector, so there is no single source of truth.

### 3. Vectors co-located with transactional data past the point where it helps

Keeping embeddings in the same table as transactions is genuinely convenient: joins are free, transactions are atomic, and there is one backup story. That convenience has a ceiling. Once similarity search dominates the workload, the query planner's choices for a table that also serves OLTP traffic stop being the right choices for approximate nearest-neighbor search. The exact point where this happens depends on row width, index configuration, and query mix — it is not a fixed row count.

## Fix 1: version the embedding, not just the dimension

Add an explicit model identifier alongside the vector and enforce dimensionality at write time. A check constraint is usually preferable to a trigger because it is declarative and visible in schema dumps.

```sql
ALTER TABLE transaction_embeddings
  ADD COLUMN model_version VARCHAR(64) NOT NULL DEFAULT 'ada-002';

ALTER TABLE transaction_embeddings
  ADD CONSTRAINT embedding_dimension_matches_model CHECK (
    (model_version = 'ada-002'                AND vector_dims(embedding) = 1536) OR
    (model_version = 'text-embedding-3-small' AND vector_dims(embedding) = 768)
  );
```

Two notes on the details, because they are easy to get wrong:

- `vector_dims()` is the pgvector function for reading a vector's dimensionality. Do not use `array_length()` on a `vector` column; it is defined for arrays, not for the `vector` type, and will either error or return NULL depending on the version.
- The default value on `model_version` exists only to backfill existing rows. New writes should always set it explicitly, and the default should be dropped once the backfill is complete so a forgotten parameter fails loudly instead of silently labeling data as `ada-002`.

The constraint turns a late, confusing failure into an immediate, specific one. It does not by itself make cross-model queries correct — that still requires filtering by `model_version` in every similarity query, which is the subject of the next section.

## Fix 2: backfill legacy rows and drop the redundant payload

If a `jsonb` response column exists, migrate it in bounded batches rather than one transaction. A single long transaction on a large table holds locks, bloats the WAL, and cannot be resumed if it fails partway.

```python
import json
import psycopg2

BATCH_SIZE = 1000

conn = psycopg2.connect(dsn="dbname=fintech")
conn.autocommit = False
cur = conn.cursor()

while True:
    cur.execute(
        """
        SELECT id, response
        FROM transaction_embeddings
        WHERE embedding IS NULL AND response IS NOT NULL
        ORDER BY id
        LIMIT %s
        FOR UPDATE SKIP LOCKED
        """,
        (BATCH_SIZE,),
    )
    rows = cur.fetchall()
    if not rows:
        break

    for row_id, response in rows:
        payload = json.loads(response)
        vector = payload["data"][0]["embedding"]
        cur.execute(
            """
            UPDATE transaction_embeddings
            SET embedding = %s, model_version = %s
            WHERE id = %s
            """,
            (vector, payload["model"], row_id),
        )

    conn.commit()
    print(f"committed batch of {len(rows)}")

cur.close()
conn.close()
```

`FOR UPDATE SKIP LOCKED` is what makes this safe to run while the application is live: the migration only locks rows it is actively updating, and a second worker can run concurrently without deadlocking. Committing per batch keeps transaction size bounded and makes the job resumable — if it crashes, re-running it simply picks up the rows that still have a NULL embedding.

Before dropping the `response` column, verify that nothing else reads it:

```sql
SELECT pg_total_relation_size('transaction_embeddings') AS total_bytes;

SELECT count(*) FROM transaction_embeddings WHERE embedding IS NULL;
```

The first query gives a before/after size comparison; the second must return zero. Only then is the column safe to drop.

## Fix 3: offload similarity search when it stops being a side job

Moving vectors to a dedicated store is the right call when similarity queries dominate the workload, not when a specific row count is crossed. The signal to watch is the shape of the query plan, not the table size.

A useful decision checklist:

- **Do similarity queries compete with OLTP traffic for the same buffer pool?** If cache hit rates on transactional tables drop when the vector workload spikes, the two are interfering.
- **Has the vector index outgrown memory?** An index that no longer fits in RAM turns every query into a disk read. Check the index size against available shared buffers.
- **Are you rebuilding the index often enough that it affects write throughput?** Frequent rebuilds on a large table are a sign the storage engine is being asked to do two incompatible jobs.
- **Do you need filtering combined with similarity?** Pre-filtering by metadata is a first-class feature in dedicated vector stores and an awkward join in PostgreSQL.

If two or more of these are true, a separate store is justified. The migration pattern is the same regardless of which store you choose:

1. Create the destination index with the correct dimension and distance metric.
2. Stream existing vectors in bulk, carrying `model_version` as filterable metadata.
3. Point the application's similarity reads at the new store, keeping the relational table as the source of truth for transactional data and the canonical vector record.

## How to verify the change worked

Verification should be mechanical, not impressionistic. The following checks cover the failure modes above.

**Schema audit.** Dump the schema and confirm every `vector` column has a companion version column and a constraint:

```bash
pg_dump --schema-only mydb | grep -A2 'vector('
```

**Dimensionality audit.** Confirm no row violates its declared model's dimension:

```sql
SELECT model_version, vector_dims(embedding) AS dims, count(*)
FROM transaction_embeddings
GROUP BY 1, 2;
```

Every row in the output should match a known (model, dimension) pair. Any unexpected combination is a row that escaped the constraint, which usually means it was written before the constraint existed.

**Latency measurement.** Measure the similarity query in isolation, before and after any change, using the same query text and the same data. `EXPLAIN (ANALYZE, BUFFERS)` is more informative than wall-clock timing alone because it separates planning time from execution time and shows whether the index is being used or a sequential scan has crept in. Record the plan, not just the number.

**Storage accounting.** Compare `pg_total_relation_size` before and after the migration. The reduction should be attributable to the dropped payload column; if the table grew, something else is accumulating.

**Contract test.** Pin the embedding dimension in a test that calls the real API with a fixed input and asserts the returned length. This is the only check that catches a provider-side change before it reaches production.

## Prevention checklist

| Practice | Failure it prevents |
|---|---|
| Version column on every vector column | Silent mixing of incompatible vector spaces |
| Check constraint tying dimension to version | Late, confusing insert failures |
| Batched, resumable migrations | Lock contention and unrestartable backfills |
| Dimension contract test in CI | Provider-side model changes |
| Query plans recorded alongside latency numbers | Regressions hidden by caching |
| Similarity queries filtered by `model_version` | Meaningless cross-model results |

The last row is the one most often missed. A version column that is written but never read in queries provides no protection at all — the vectors are still being compared across incompatible spaces, and the results are still wrong, just quietly.

## Frequently asked questions

**Can two models share a dimension and still be incompatible?**

Yes. Dimensionality is a necessary but not sufficient condition for comparability. Vectors from different models occupy different spaces even at the same width, so cosine similarity between them is not meaningful. This is why the version tag matters independently of the dimension check.

**Should the version column be a string or a foreign key?**

A string is simpler and survives model deprecation without a migration. A foreign key to a models table is worth it when you need to track deprecation dates, cost per token, or which models are still approved for production. Start with a string; add the table when you have a second reason to query model metadata.

**Is a check constraint or a trigger better for dimension validation?**

A check constraint is declarative, appears in schema dumps, and is enforced by the planner. A trigger can express logic a constraint cannot, such as looking up the expected dimension from another table. Prefer the constraint unless you genuinely need the lookup.

**When is it worth moving to a separate vector store?**

When similarity queries interfere with transactional performance, when the index no longer fits in memory, or when you need metadata pre-filtering as a first-class operation. Table size alone is a weak signal.

**What about storing the raw API response for debugging?**

Store it in object storage keyed by request ID, or in a separate table that is periodically truncated. Keeping it inline with the vector doubles the row width and creates a second, drifting source of truth.

**How should a model upgrade be rolled out?**

Add the new model as a new `model_version` value, backfill vectors in the background, and switch queries over once coverage is complete. Keep the old vectors until the new ones have been validated in production. Deleting the old model's rows is the last step, not the first.

---

**Next step:** run `SELECT model_version, vector_dims(embedding), count(*) FROM transaction_embeddings GROUP BY 1, 2;` against your own database and check that every row's dimension matches a model you recognize. Any row that does not is a latent upgrade failure waiting for the next model change.
