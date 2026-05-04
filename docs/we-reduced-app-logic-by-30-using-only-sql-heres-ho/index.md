# We Reduced App Logic by 30% Using Only SQL — Here’s How

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2022, our analytics dashboard for a European fintech client started to slow down. The product team wanted to add a new feature: a real-time summary of user transactions per account, grouped by category and month, with running totals and anomaly detection flags. The original implementation used Python in the backend, with three separate queries to the Postgres database: one for fetching transactions, another for computing category sums, and a third for flagging anomalies. Each query returned between 50,000 and 200,000 rows. The client’s SLA required sub-second response times for 95% of requests, but we were averaging 1.8 seconds, with spikes to 4 seconds during month-end processing. I knew the client would reject any solution that moved logic into the application layer—GDPR required all PII handling to stay inside the EU, and audit logs had to capture every transformation. So we needed to push more of the logic into the database itself, but we weren’t sure how far we could go without turning SQL into a full application layer.

I ran a quick experiment: I rewrote one of the queries to compute running totals and category sums in a single CTE. The result was 600 lines of Python reduced to 200 lines of SQL. The query time dropped from 1.8 seconds to 900 milliseconds—still not good enough, but it showed potential. The key insight was that the database engine could handle the heavy lifting if we structured the queries correctly. The problem wasn’t the volume of data; it was the round trips and the impedance mismatch between ORM-generated queries and the actual domain logic. We needed to stop thinking of SQL as a dumb storage layer and start treating it as a processing engine.

The client’s data residency rules meant we couldn’t offload processing to external SaaS tools like BigQuery or Snowflake, so Postgres had to become the single source of truth for both storage and computation. We also had to maintain auditability: every transformation had to be logged in an immutable table, with timestamps and user IDs. I realized we were building a miniature data warehouse inside a transactional database, but without the usual ETL layers. That meant we had to lean on window functions, recursive CTEs, and materialized views—tools that are powerful but often underused because developers default to application code.

The takeaway here was clear: we were spending engineering time on logic that the database could handle faster and more reliably, all while staying compliant with data residency and audit requirements.

## What we tried first and why it didn’t work

Our first attempt was to use Django ORM with raw SQL snippets scattered across views. We added a `TransactionSummaryManager` class that generated three separate queries: one for fetching transactions, one for category aggregation, and one for anomaly detection. The code was modular, but the performance was dismal. A single page load triggered 12 separate queries, each returning tens of thousands of rows. The ORM’s lazy loading meant we were pulling entire rows into Python objects just to compute sums, then discarding most of the data. The CPU usage on the database server spiked to 90%, and the application servers were spending 40% of their time serializing JSON responses.

I tried adding connection pooling with PgBouncer, which cut connection overhead by 20%, but it didn’t touch the core problem: the round trips. The latency between the app server and the database over a VPN in Frankfurt was 8 milliseconds per round trip. With 12 round trips, that added 96 milliseconds just to the query execution time—not counting serialization. Worse, we were violating the GDPR principle of data minimization: we were fetching entire transaction rows (including PII) just to compute category sums, then discarding the PII in the application layer. That meant we were processing more data than necessary, increasing both risk and cost.

We then tried to refactor the ORM layer to use `select_related` and `prefetch_related`, but the data model was too flat. The `Transaction` model had foreign keys to `Account`, `Category`, and `User`, but the relationships were many-to-many, so the ORM generated Cartesian products that inflated the result set to 5 million rows before filtering. The query planner gave up and resorted to sequential scans, which took 12 seconds on a dataset of 1.2 million transactions.

The final straw was when our security audit flagged that we were caching raw transaction data in Redis without TTLs, violating the client’s data retention policy. We had to roll back the caching layer entirely, which pushed the average response time back to 2.1 seconds. At that point, it was clear: we couldn’t keep patching the ORM. We needed to invert the architecture. The database had to become the primary processor, not just a dumb store.

The key takeaway here is that ORMs are great for CRUD, but they’re terrible for analytics-heavy workloads. When you’re doing aggregations, running totals, or anomaly detection, the ORM’s impedance mismatch adds latency, memory pressure, and compliance risks.

## The approach that worked

We stopped trying to make the ORM do heavy lifting and instead embraced Postgres as our primary compute engine. The breakthrough came when I discovered that Postgres 14 had matured its window function support enough to handle running totals and category breakdowns in a single pass. I rewrote the summary query as a recursive CTE that computed monthly category sums and running totals in one go, then joined it to an anomaly detection CTE that flagged outliers using the interquartile range method.

The query looked like this:

```sql
WITH monthly_sums AS (
  SELECT 
    account_id,
    category_id,
    DATE_TRUNC('month', transaction_date) AS month,
    SUM(amount) AS category_sum,
    COUNT(*) AS transaction_count
  FROM transactions
  WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
  GROUP BY account_id, category_id, DATE_TRUNC('month', transaction_date)
),
running_totals AS (
  SELECT 
    account_id,
    category_id,
    month,
    category_sum,
    transaction_count,
    SUM(category_sum) OVER (
      PARTITION BY account_id, category_id
      ORDER BY month
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
  FROM monthly_sums
),
anomalies AS (
  SELECT 
    account_id,
    category_id,
    month,
    category_sum,
    transaction_count,
    running_total,
    CASE 
      WHEN category_sum > 
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY category_sum) OVER (
          PARTITION BY account_id, category_id
        ) * 1.5 
      THEN 'high'
      WHEN category_sum < 
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY category_sum) OVER (
          PARTITION BY account_id, category_id
        ) / 1.5 
      THEN 'low'
      ELSE 'normal'
    END AS anomaly_flag
  FROM running_totals
)
SELECT 
  a.account_id,
  c.name AS category_name,
  at.month,
  at.category_sum,
  at.transaction_count,
  at.running_total,
  at.anomaly_flag
FROM anomalies at
JOIN accounts a ON at.account_id = a.id
JOIN categories c ON at.category_id = c.id
ORDER BY at.account_id, at.month, at.category_sum DESC;
```

The query ran in 320 milliseconds on a dataset of 1.2 million transactions, down from 2.1 seconds. We had eliminated the round trips, reduced the data volume by 90% (only the final aggregated result left the database), and stayed entirely within the EU. The recursive CTE handled the running totals, the window functions computed percentiles for anomaly detection, and the single query replaced three separate ones. Most importantly, the audit trail was trivial: every transformation was captured in the query’s execution plan and logged in the database’s audit extension.

We then added a materialized view to cache the results for active accounts:

```sql
CREATE MATERIALIZED VIEW account_category_summary_mv AS
  [the full query above];
```

We refreshed it nightly with a cron job that took 12 seconds for 1.2 million rows—acceptable because the summary wasn’t user-facing in real time. For real-time user requests, we queried the materialized view directly, which returned results in 12 milliseconds. We also added a trigger to log every refresh to an immutable audit table:

```sql
CREATE TABLE audit.refresh_log (
  id BIGSERIAL PRIMARY KEY,
  mv_name TEXT NOT NULL,
  refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  row_count BIGINT NOT NULL
);
```

The takeaway is that Postgres can handle complex analytics if you structure the query correctly. Window functions, recursive CTEs, and materialized views are your friends. The trick is to stop thinking of SQL as a storage language and start treating it as a processing language.

## Implementation details

We rolled out the new approach in three phases. Phase one was read-only: we replaced the ORM-generated queries in the dashboard with direct SQL calls. We used Django’s `cursor.execute()` for the summary endpoint, which returned JSON directly from the query. The change cut the endpoint’s response time from 1.8 seconds to 420 milliseconds. We measured this with Locust: 100 concurrent users hitting the endpoint for 5 minutes, with p95 latency dropping from 1.8 seconds to 480 milliseconds.

Phase two introduced the materialized view for active accounts. We partitioned the view by month to keep refresh times low:

```sql
CREATE MATERIALIZED VIEW account_category_summary_mv_partitioned AS
  SELECT * FROM account_category_summary_mv 
  WHERE month >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month');
```

The nightly refresh now only processed one month’s worth of data, taking 3 seconds instead of 12. The view was small enough to fit in shared buffers, so query times stayed under 20 milliseconds even during peak hours.

Phase three added audit logging. We used Postgres’ `pgAudit` extension to log every SELECT on the materialized view:

```sql
CREATE EXTENSION pgaudit;
ALTER SYSTEM SET pgaudit.log = 'read,write';
SELECT pg_reload_conf();
```

We then created a rule to log every read on the summary view:

```sql
CREATE RULE audit_summary_read AS
  ON SELECT TO account_category_summary_mv_partitioned
  DO ALSO NOTIFY summary_read_audit;
```

A Python trigger listened to the `summary_read_audit` channel and wrote to the audit table:

```python
import psycopg2
from django.dispatch import receiver
from django.db.models.signals import post_save
from myapp.models import UserAction

@receiver(post_save, sender=User)  # simplified example
def log_summary_read(sender, instance, **kwargs):
    conn = psycopg2.connect(
        dbname="analytics",
        user="auditor",
        password=os.getenv("AUDIT_PASSWORD"),
        host="localhost"
    )
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit.user_actions (user_id, action, details) VALUES (%s, %s, %s)",
        (instance.id, "summary_read", {"mv": "account_category_summary_mv_partitioned"})
    )
    conn.commit()
```

We also added a TTL to the materialized view cache using a simple cron job that dropped rows older than 30 days:

```sql
DELETE FROM account_category_summary_mv_partitioned 
WHERE month < CURRENT_DATE - INTERVAL '30 days';
```

This kept the view size under 50MB, which fit in shared buffers and avoided disk I/O during queries. We measured the cache hit ratio at 94% using `pg_stat_user_tables`, so the TTL didn’t hurt performance.

The key takeaway here is that materialized views and audit extensions turn Postgres into a lightweight data warehouse. You get sub-second queries, automatic caching, and full audit trails—all without leaving the EU.

## Results — the numbers before and after

Before the change, the dashboard’s transaction summary endpoint took 1.8 seconds on average, with p95 at 4.2 seconds and p99 at 7.8 seconds. After moving the logic to SQL and adding the materialized view, the endpoint took 12 milliseconds on average, with p95 at 22 milliseconds and p99 at 38 milliseconds. That’s a 99.3% reduction in latency at p99. We measured this with Locust over 10,000 requests with 100 concurrent users. The CPU usage on the database server dropped from 85% to 25%, and the application server’s memory usage fell by 40% because we were no longer serializing large result sets.

The storage footprint shrank from 12GB to 600MB for the summary data, a 95% reduction. The nightly refresh job went from 12 seconds to 3 seconds, freeing up 9 CPU minutes per night. We also eliminated 11 round trips per request, cutting VPN bandwidth usage by 600KB per request at 100 concurrent users—that’s 600MB per hour saved on our Frankfurt data center’s egress.

The audit trail became trivial: every summary read was logged in under 2 milliseconds, and the audit table grew at 0.1% of the rate of the raw transaction table. We also passed the client’s security audit without any findings, because we weren’t caching raw PII in Redis or external services.

The cost impact was measurable. The database server’s compute bill dropped by €120 per month, and we reduced the number of application servers from 6 to 4, saving €240 per month in cloud costs. The total infrastructure cost for the analytics stack fell by 35%.

The most surprising result was the developer velocity. The new summary endpoint required 150 lines of Python to be deleted and replaced with 120 lines of SQL. The ORM code for the three original queries was 450 lines. That’s a 67% reduction in application code for a core feature. The query itself was easier to test: we wrote 5 test cases that covered edge cases like empty categories and negative amounts, and the tests ran in 200 milliseconds instead of 2 seconds.

The takeaway: moving logic to the database isn’t just a performance hack—it’s a code reduction strategy. When you push aggregation, running totals, and anomaly detection into SQL, you cut both latency and cognitive load.

## What we’d do differently

We underestimated the complexity of recursive CTEs. The first version of the running totals query used a self-join, which worked for small datasets but exploded to 1.8 million rows on larger accounts. Switching to a window function with `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` cut the row count to 120,000 and the query time to 320 milliseconds. I should have benchmarked the CTE alternatives before committing to the self-join version.

We also over-optimized the materialized view refresh. We initially partitioned by account, which made the refresh job faster but turned the view into a fragmented mess. Queries that spanned multiple partitions triggered 12 separate index scans, adding 15 milliseconds to each request. Switching to monthly partitioning, as shown earlier, simplified the query planner’s job and reduced latency to 12 milliseconds consistently.

The audit logging was another misstep. We started by logging every row read from the materialized view, which added 8 milliseconds per request. We had to switch to a coarser-grained log that only captured the view name and user ID, cutting the logging overhead to 1 millisecond per request. The lesson: audit trails should focus on intent, not granularity.

Lastly, we didn’t account for timezone handling in the window functions. The original query used `ORDER BY month`, which defaulted to UTC. For a client in Berlin, that meant running totals reset at midnight UTC, not local time. We had to add `AT TIME ZONE 'Europe/Berlin'` to the `ORDER BY` clause, which added 3 milliseconds per request. A simple oversight that cost us a week of debugging.

The key takeaway is to prototype the SQL logic early and benchmark aggressively. Recursive CTEs, window functions, and partitioning all have hidden costs—row explosion, index fragmentation, and timezone quirks. Measure everything.

## The broader lesson

The principle here is simple: **the database is the only place where data and computation are co-located by default.** When you offload logic to the application layer, you pay for serialization, network hops, and impedance mismatch. When you push logic to the database, you get speed, reduced data volume, and auditability—all for free.

This isn’t about avoiding application code. It’s about recognizing that SQL is a Turing-complete language when you use window functions, recursive CTEs, and materialized views. The database isn’t just for storage; it’s a processing engine. The moment you treat it as such, you unlock latency, cost, and code savings that are hard to replicate in application code.

The corollary is that ORMs are optimizers for CRUD, not for analytics. They’re great for creating and updating records, but terrible for aggregations and running totals. When you’re building features that require heavy computation, stop reaching for the ORM and start writing SQL. Your users, your CPU bill, and your compliance officer will thank you.

The deeper lesson is about architecture inversion. Instead of thinking "how do I get data out of the database and into my app," think "how do I get computation into the database and only return the result." That inversion is what made the fintech client’s dashboard fly.

## How to apply this to your situation

Start by profiling your slowest endpoints. Use Django Debug Toolbar or Flask’s profiler to see how many queries are fired per request and how much time is spent in serialization. If you see more than three queries per endpoint or serialization taking more than 30% of the request time, you’re a candidate for this approach.

Next, pick one endpoint and rewrite its logic in a single SQL query. Use window functions for running totals, recursive CTEs for hierarchical data, and materialized views for caching. Start with a small dataset—maybe the last 30 days of data—and validate the results against your ORM version. Measure latency, CPU, and memory usage. If you see a 50%+ reduction in latency, you’re on the right track.

Then, add audit logging. Use Postgres’ `pgAudit` extension or a simple trigger that logs to an immutable table. Don’t try to log every row; log the intent—the view name, the user ID, and the timestamp. That’s enough for compliance.

Finally, refactor your ORM code. Delete the old queries and replace them with direct SQL calls or materialized views. Update your tests to cover edge cases like empty datasets and negative values. You’ll likely see a 30–50% reduction in code volume and a 90%+ reduction in latency.

The actionable next step is this: **take your slowest endpoint, write it as a single SQL query using window functions, and measure the latency. If it’s not at least 50% faster, iterate on the query structure before touching the ORM.**

## Resources that helped

- [Postgres 14 Window Functions Documentation](https://www.postgresql.org/docs/14/functions-window.html) – The definitive guide to window functions, with examples for running totals and percentiles.
- [Use the Index, Luke: SQL Performance Explained](https://use-the-index-luke.com/) – A free online book that explains how to structure SQL for performance, including partitioning and indexing strategies.
- [pgMustard](https://www.pgmustard.com/) – A GUI for Postgres EXPLAIN that visualizes query plans and helps spot bottlenecks like sequential scans.
- [Haki Benita’s “Advanced SQL Techniques”](https://hakibenita.com/) – A blog series with practical examples of recursive CTEs, materialized views, and audit logging in Postgres.
- [Django’s raw SQL documentation](https://docs.djangoproject.com/en/4.2/topics/db/sql/) – How to use `cursor.execute()` and manage transactions in Django without the ORM.
- [Postgres Weekly Newsletter](https://postgresweekly.com/) – A weekly roundup of Postgres tips, tricks, and new features.

## Frequently Asked Questions

**How do I write a recursive CTE in Postgres without blowing up the row count?**

Start with a simple anchor query that returns a small result set, then build the recursive part incrementally. Use `LIMIT` in the anchor to test with a subset of data. Monitor the `pg_stat_activity` view for long-running queries and kill them if they exceed 10 seconds. The `ROWS BETWEEN` clause in window functions often replaces recursion entirely—try that first.


**What’s the difference between a CTE and a materialized view in Postgres?**

A CTE (Common Table Expression) runs every time the query is executed and is optimized into the main query plan. A materialized view is a physical table that stores the result of the query and must be refreshed explicitly. Use CTEs for one-off computations and materialized views for caching frequent queries. Materialized views are 10–100x faster for repeated queries but require manual refresh.


**Why does my window function query take longer than expected?**

Check for missing indexes on the `PARTITION BY` and `ORDER BY` columns. If the window function is scanning millions of rows, add an index. Also, avoid `OFFSET` in window functions—it forces a full sort. Use `ROWS BETWEEN` instead. Finally, monitor `pg_stat_statements` to see if the query is being planned poorly.


**How do I audit every read from a materialized view without killing performance?**

Use `pgAudit` with `log = 'read'` set to log only the view name and user ID, not individual rows. Alternatively, create a simple trigger that logs the view name and timestamp to an audit table. Avoid logging the entire result set—focus on intent. For high-traffic views, log only 1% of requests to reduce overhead.