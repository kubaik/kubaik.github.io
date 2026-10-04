# Soft deletes fail: what breaks first

Soft deletes look like a two-line change: add a `deleted_at` column, set it to `NULL` for active rows, set it to `NOW()` when a user deletes something, and add a global scope that hides deleted rows. That works in a CRUD app with a thousand rows. It stops working in a predictable way as the table grows, and the failure is easy to misdiagnose because the symptom looks like a hardware problem rather than a schema problem.

## The one-paragraph version

Soft deletes are a toggle, but deletion is a lifecycle. Once a large fraction of rows in a hot table are marked deleted, every index on that table carries dead weight, every query that touches it pays for a filter it cannot skip, and backups and replicas grow without bound. The fix is not hard deletion either — it is moving rows between tiers (live, archive, cold) so that each tier has an index design suited to the queries that actually run against it. This article explains the failure mode, gives a worked refactor, and lists the checks worth running before you commit to the change.

## Why the pattern confuses people

Tutorials present soft deletes as a boolean. The `deleted_at` column is nullable, so `NULL` means active and a timestamp means deleted. This is compact and reversible, and it is genuinely the right answer for small tables where deletion is rare and audit history matters.

The confusion comes from conflating *logical deletion* with *soft deletes*. Logical deletion is a requirement: "we must be able to show that this record existed and was removed." Soft deletes are one implementation of that requirement, and a leaky one. The leak is that the deleted marker lives in the same table, and therefore in the same indexes, as the live data. Every query planner decision, every index page, and every sequential scan now includes rows that no query wants.

A typical failure sequence looks like this:

1. The table is small, queries are fast, and the global scope is invisible.
2. Row count grows past the point where the planner prefers an index scan for the common access path.
3. Deleted rows accumulate. The index on the hot column now has to be traversed past many entries that will be discarded by the `deleted_at IS NULL` filter.
4. The planner notices the filter is not selective and falls back to a sequential scan, or picks an index that returns far more rows than the query needs.
5. Latency rises. Teams add more indexes, which makes writes slower and the problem worse.

The step that surprises people is 4. Adding `WHERE deleted_at IS NULL` to every query does not make the query faster; it changes which rows are returned, not how many are read. The planner still has to read the deleted entries and discard them.

## What actually breaks, in order

The failures arrive roughly in this order, and each one has a different fix.

**Index bloat first.** Every index that includes the deleted rows grows. A B-tree index on `(user_id, status)` that also carries `deleted_at` as a trailing column or an included column has to store entries for deleted rows. If deleted rows are 15% of the table, roughly 15% of each index is dead weight that every scan must traverse. Writes pay for it too: every insert updates more index pages, and every vacuum has more to do.

**Planner misestimates second.** The planner uses statistics to choose between a sequential scan and an index scan. When a filter like `deleted_at IS NULL` is not correlated with the indexed column, the estimated selectivity is poor, and the chosen plan can be much worse than the estimate suggests. This is the classic case where a query is fast in staging (few deleted rows) and slow in production (many deleted rows) with identical SQL.

**Foreign key chains third.** If `orders.user_id` references `users.id`, and `users` also uses soft deletes, then a query that needs the user's email has to join through a table that is itself mostly deleted rows. The filter propagates: you write `WHERE u.deleted_at IS NULL` in a join, and now both sides of the join are filtering.

**Backups and replicas fourth.** Backups include deleted rows. If 20% of your table is soft-deleted, 20% of every backup is data no query will ever return. Read replicas carry the same rows. If you replicate the whole table, you replicate the dead weight.

**Analytics fifth.** A reporting query that scans the table has to exclude deleted rows, and if it forgets, the numbers are wrong. This is the failure mode that tends to surface last, because it produces incorrect results rather than slow ones.

## A worked example

The following is a simplified order system. The numbers below are illustrative, chosen so the arithmetic is easy to follow; substitute your own row counts and timings. Assume a table with 5,000,000 rows, of which 800,000 are soft-deleted, on a general-purpose instance.

### Step 1: the naive soft delete

```python
# models.py
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    status = Column(String(20), index=True)
    created_at = Column(DateTime, server_default=func.now())
    deleted_at = Column(DateTime, nullable=True)
```

```python
# queries.py
from sqlalchemy import select
from models import Order

def get_active_orders(session, user_id: int):
    stmt = (
        select(Order)
        .where(Order.user_id == user_id)
        .where(Order.status == 'paid')
        .where(Order.deleted_at.is_(None))
        .order_by(Order.created_at.desc())
    )
    return session.execute(stmt).scalars().all()
```

The problem here is not the SQL. It is that the index on `user_id` returns rows for both live and deleted orders, and the `status` and `deleted_at` filters are applied after the index lookup. When deleted rows dominate the index range for a given user, the index scan reads far more pages than it needs.

### Step 2: measure before you change anything

Before adding indexes or refactoring, measure. Three commands give you most of what you need on PostgreSQL.

```sql
-- 1. How much of the table is dead weight?
SELECT
  count(*) FILTER (WHERE deleted_at IS NOT NULL) AS deleted_rows,
  count(*) AS total_rows,
  round(100.0 * count(*) FILTER (WHERE deleted_at IS NOT NULL) / count(*), 1) AS pct_deleted
FROM orders;

-- 2. What is the planner actually doing?
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders
WHERE user_id = 42 AND status = 'paid' AND deleted_at IS NULL
ORDER BY created_at DESC;

-- 3. Which indexes are carrying the dead weight?
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE relname = 'orders'
ORDER BY pg_relation_size(indexrelid) DESC;
```

If the `EXPLAIN` output shows a `Seq Scan` with a `Filter` on `deleted_at IS NULL`, or an index scan whose `Rows Removed by Filter` is a large fraction of `Rows Removed by Index Recheck`, you have the leak. The ratio of `Rows Removed by Filter` to rows returned is the number to watch: it tells you how much work the query does that no caller asked for.

### Step 3: refactor to a lifecycle

Split the table into a live table and an archive table. The live table has no `deleted_at` column at all, because rows that are deleted are moved out of it. The archive table is partitioned by time, so old partitions can be dropped or exported without rewriting the table.

```python
# lifecycle_models.py
from sqlalchemy import Column, Integer, String, DateTime, Date, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class OrderLive(Base):
    __tablename__ = 'orders_live'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    status = Column(String(20), index=True)
    created_at = Column(DateTime, server_default=func.now())

class OrderArchive(Base):
    __tablename__ = 'orders_archive'
    __table_args__ = {
        'postgresql_partition_by': 'RANGE (created_at)'
    }
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    status = Column(String(20), index=True)
    created_at = Column(DateTime, nullable=False)
    archived_at = Column(Date, server_default=func.current_date())
```

The live query no longer mentions `deleted_at`, because there is nothing to filter:

```python
# lifecycle_queries.py
from sqlalchemy import select
from lifecycle_models import OrderLive

def get_active_orders(session, user_id: int):
    stmt = (
        select(OrderLive)
        .where(OrderLive.user_id == user_id)
        .where(OrderLive.status == 'paid')
        .order_by(OrderLive.created_at.desc())
    )
    return session.execute(stmt).scalars().all()
```

The move itself is a batched job. The important property is idempotency: if the job crashes halfway, rerunning it must not duplicate rows. The version below uses a single transaction per batch, so a crash rolls back the whole batch.

```python
# lifecycle_job.py
from datetime import datetime, timedelta
from sqlalchemy import select, insert, delete
from lifecycle_models import OrderLive, OrderArchive

def archive_old_orders(session, batch_size: int = 10_000) -> int:
    cutoff = datetime.utcnow() - timedelta(days=30)
    moved = 0
    while True:
        rows = session.execute(
            select(OrderLive).where(OrderLive.created_at < cutoff).limit(batch_size)
        ).scalars().all()
        if not rows:
            break
        ids = [r.id for r in rows]
        session.execute(
            insert(OrderArchive).values([
                {
                    'id': r.id,
                    'user_id': r.user_id,
                    'status': r.status,
                    'created_at': r.created_at,
                }
                for r in rows
            ])
        )
        session.execute(delete(OrderLive).where(OrderLive.id.in_(ids)))
        session.commit()
        moved += len(rows)
    return moved
```

Two details matter here. First, the `insert` and `delete` are in the same transaction, so a row is never in both tables or in neither. Second, the loop re-selects each batch rather than holding a cursor open, so it does not block writers for the duration of the job.

### Step 4: verify the improvement

Re-run the same `EXPLAIN (ANALYZE, BUFFERS)` from step 2 against `orders_live`. The thing to compare is not wall-clock time in isolation, which varies with cache state, but the plan shape and the buffer counts. A query that previously reported a large `Rows Removed by Filter` should now report near zero, and the plan should show an index scan on the composite index with no filter step.

## A decision checklist

Not every table needs a lifecycle. Use this list to decide.

- **Is the deleted fraction above roughly 5% of the table?** Below that, the index bloat is usually tolerable and the refactor is not worth the operational cost. Measure with the query from step 2.
- **Is the table on the hot path for writes?** Every index on a soft-deleted table is written on insert. If the table takes thousands of writes per second, the extra index entries are a direct write-amplification cost.
- **Do you need to query deleted rows?** If the answer is never, the archive may not need to be in the same database at all. If the answer is "for compliance, on request," an archive table with a time partition is enough.
- **Do you have a retention policy?** A lifecycle without a retention rule is just a second table that grows forever. Decide how long archive rows live before they move to cold storage or are dropped.
- **Can your application tolerate a read-from-two-places period?** The migration in the next section requires it.

## Migrating without downtime

The safe migration is a dual-write, backfill, cutover sequence. It is well documented for schema changes in general; the soft-delete case is a specific instance.

1. **Create the new tables.** `orders_live` and the partitioned `orders_archive`. Add the composite index you want on `orders_live`.
2. **Dual-write.** Change the application to write every new order to both `orders` and `orders_live`. Reads still come from `orders`.
3. **Backfill in batches.** Copy rows from `orders` into `orders_live` (for rows younger than the archive cutoff) and `orders_archive` (for older rows). Run in batches with a small sleep between them so you do not saturate I/O.
4. **Verify counts.** Compare `count(*)` on `orders` against `count(*)` on `orders_live` plus `count(*)` on `orders_archive`. They should match once backfill completes.
5. **Cut reads over.** Point reads at `orders_live`. Watch error rates and latency. If something is wrong, point reads back at `orders`.
6. **Cut writes over.** Stop writing to `orders`. Keep it in place, read-only, until you are confident.
7. **Drop the old table.** Only after a full backup and a waiting period you are comfortable with.

The step people skip is 4. Without a count comparison, a partial backfill looks identical to a complete one until a query returns a missing row.

## Common misconceptions

**"Soft deletes are free because the data is still there."** They are not free. Each deleted row occupies space in the heap and in every index that covers it. The cost is paid on every scan, every vacuum, and every backup.

**"Adding `WHERE deleted_at IS NULL` makes the query fast."** It makes the query correct. The filter is applied after rows are read, so the read cost is unchanged. Whether it helps at all depends on the plan; in many cases it does not change the plan shape, only the rows returned.

**"Soft deletes are required for analytics."** Analytics needs a consistent view of historical data. That can come from an archive table, a materialized view that unions live and archive, or a separate analytics store fed by change data capture. None of those require the deleted flag to live in the hot table.

**"You cannot change this without downtime."** The dual-write and backfill pattern above is the standard approach and does not require downtime. The constraint is application complexity, not database availability.

## FAQ

**How do I know if soft deletes are hurting a specific query?**

Run `EXPLAIN (ANALYZE, BUFFERS)` and look at `Rows Removed by Filter`. If that number is large relative to the rows returned, the query is reading rows it will discard. Then compare the plan against the same query on a table without the deleted rows.

**What if I need to keep deleted rows for compliance?**

Keep them, but in a separate table or a separate storage tier. The compliance requirement is that the data is retained and retrievable, not that it lives in the same table as live data.

**Does this work with an ORM?**

Yes, but you will have two model classes instead of one, and your application code must choose which to query. That is the cost of removing the leak. Some ORMs support a view that unions live and archive for read-only paths.

**What about foreign keys between live and archive tables?**

Foreign keys cannot span a live table and an archive table cleanly. The common approaches are to denormalize the referenced value into the live table (for example, store `user_email` on `orders_live`), or to accept that archive queries do not enforce referential integrity and are read-only.

**How often should the archive job run?**

As often as your retention boundary requires. If rows must leave the live table after 30 days, run the job at least daily, and alert if the oldest row in `orders_live` exceeds the boundary by more than one job interval.

## Action

Run this against your production database now and read the top ten rows:

```sql
SELECT query, calls, total_exec_time, mean_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

For the top query, run `EXPLAIN (ANALYZE, BUFFERS)` on it. If the plan contains a `Filter` on `deleted_at IS NULL` with a high `Rows Removed by Filter`, you have found the table that should be your first lifecycle candidate.
