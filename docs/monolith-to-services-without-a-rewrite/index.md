# Monolith to services without a rewrite

## The conventional wisdom, and where it breaks down

The standard playbook for splitting a monolith goes like this: identify bounded contexts, draw the boundaries on a whiteboard, then carve out services one at a time behind a strangler fig facade. The promised benefits are cleaner code, independent deployments, and a path to horizontal scalability.

That playbook is not wrong, but it is incomplete. It assumes the monolith is already modular in its data model, the team has DevOps capacity to spare, and the infrastructure can absorb the overhead of network calls between what used to be function calls. Many real systems satisfy none of those assumptions.

The missing piece in most write-ups is **data coupling**. When a large fraction of your queries join across what you want to be separate services, you are not extracting a service. You are extracting a distributed monolith, and a distributed monolith typically has worse latency and harder debugging than the original. A common failure mode is a team spending months extracting a "User Service" only to find that every other service still queries the `users` table for address history, preferences, and purchase summaries. The result is not independence. It is a mesh of cross-service queries that used to be a single indexed join.

## Three failure modes worth understanding before you plan anything

**Failure mode 1: the invisible lock.** A service is extracted behind an internal API gateway. Staging looks fine. In production, the service starts timing out on a small percentage of requests. The root cause is often a write lock on a shared table that the monolith holds during a transaction. The extracted service queries that table directly for validation. The lock was always there; it just used to be invisible because the caller was inside the same process and the same transaction. After extraction, the lock surfaces as tail latency. Service extraction does not change the data model. It relocates the contention to a boundary where it is harder to reason about.

**Failure mode 2: duplicated load, not reduced load.** A monolith running on one large database instance is split so that a payment module runs on its own application servers. The database bill stays flat because the same queries still hit the same tables. The compute bill rises because you now run two application tiers instead of one. Nothing was scaled down. The team justifies the cost as "investment in scalability," but if traffic is flat, there is no scalability being bought. The right question is not "how many services do we have" but "what resource is actually saturated, and does a service boundary relieve it."

**Failure mode 3: the latency tax.** In-process function calls are measured in fractions of a millisecond. An HTTP round trip inside a VPC is typically single-digit to low tens of milliseconds. That difference is the latency tax, and it is paid on every cross-service call. Teams routinely underestimate it because staging traffic is low and the network is quiet. The tax scales with fan-out: a request that touches five services serially pays it five times.

## A different mental model: start with coupling points

Instead of starting with contexts, start with **coupling points** — the places where data or behavior is shared across what you think are separate features. The goal is not to extract services. The goal is to reduce coupling so that a future extraction does not drag half the monolith along with it.

### Step 1: map your queries

On PostgreSQL, `pg_stat_statements` ranks statements by total execution time. That ranking, not your mental model of the code, is the honest map of where the database spends its work:

```sql
-- Top 20 statements by total time
SELECT
  queryid,
  calls,
  total_exec_time,
  mean_exec_time,
  rows,
  left(query, 120) AS query_snippet
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

A frequent finding is that one or two statements dominate. A single reporting view that joins a dozen tables, written years ago for a dashboard nobody uses anymore, can account for a large share of total database time. Dropping or materializing that view is a database change, not a service change, and it can remove more load than any extraction would.

To see whether a specific query crosses feature boundaries, run:

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT ... ;
```

Look at which relations appear in the plan. If a query you associate with billing touches `users`, `addresses`, and `orders`, those tables are coupling points between billing and whatever else owns them.

### Step 2: identify transaction boundaries

If two operations must be atomic with respect to each other, they are coupled. Creating a user and sending a welcome email inside one transaction is a coupling. You cannot extract the email path without either accepting duplicate emails, accepting lost emails, or introducing a coordination mechanism.

The standard coordination mechanism is the **outbox pattern**. The monolith writes the business row and an event row to an `outbox` table in the same local transaction. A background worker reads unpublished rows and publishes them to a broker. Consumers subscribe. The atomicity guarantee is preserved by the single local transaction; the delivery is at-least-once, so consumers must be idempotent.

A sketch of the outbox table:

```sql
CREATE TABLE outbox (
  id           bigserial PRIMARY KEY,
  aggregate    text        NOT NULL,
  event_type   text        NOT NULL,
  payload      jsonb       NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  published_at timestamptz
);

CREATE INDEX outbox_unpublished_idx
  ON outbox (created_at)
  WHERE published_at IS NULL;
```

And the publisher loop, showing the idempotency contract rather than a specific broker:

```python
def publish_pending(batch_size: int = 100) -> int:
    with db.transaction():
        rows = db.execute(
            """
            SELECT id, aggregate, event_type, payload
            FROM outbox
            WHERE published_at IS NULL
            ORDER BY created_at
            LIMIT %s
            FOR UPDATE SKIP LOCKED
            """,
            (batch_size,),
        ).fetchall()

        for row in rows:
            broker.publish(
                topic=f"{row['aggregate']}.{row['event_type']}",
                key=str(row["id"]),
                value=row["payload"],
            )
            db.execute(
                "UPDATE outbox SET published_at = now() WHERE id = %s",
                (row["id"],),
            )
    return len(rows)
```

Two details matter. `FOR UPDATE SKIP LOCKED` lets multiple publisher workers run without stepping on each other. Publishing before marking as published means a crash between the two produces a duplicate, which is why consumers must be idempotent — that is the at-least-once contract, not a bug.

The coupling is broken at the event layer without any service extraction. The email path can later become its own process, or stay a worker inside the monolith, and the contract does not change.

### Step 3: measure before you move

Before extracting anything, establish a baseline and then simulate the extraction.

Instrument at minimum:

- P50, P95, and P99 latency for the endpoints you intend to split.
- Query counts per request (to catch N+1 patterns that appear when a join becomes a remote call).
- Database connection pool saturation.
- Error rate by endpoint.

Then run a load test that replays production-shaped traffic against a staging clone. A tool such as Locust or k6 can drive the load; the choice matters less than replaying a realistic mix. Record the baseline numbers.

Next, simulate the extraction. Route a fraction of the relevant calls through a mock service that adds a fixed delay equal to your expected network hop, and re-run the same load. If P95 latency rises by more than your acceptable budget — a common internal target is 20ms — the extraction will hurt performance unless the boundary is redesigned.

The point of the simulation is that it is cheap. A mock that sleeps for 20ms takes an afternoon. A production extraction that turns out to be a latency regression takes a quarter.

## A worked example, with the arithmetic shown

Suppose a monolith serves 5,000 requests per second, and a candidate "notifications" module is invoked on 30% of those requests. That is:

```
5,000 req/s × 0.30 = 1,500 calls/s to the notifications module
```

In-process, each call costs about 0.3ms of CPU. After extraction, each call becomes an HTTP round trip. Suppose the measured RTT inside the VPC is 20ms:

```
1,500 calls/s × 20ms = 30,000 ms/s = 30 CPU-seconds of waiting per wall-clock second
```

That waiting is not CPU work, but it is concurrency demand: to sustain 1,500 in-flight calls at 20ms each, you need roughly:

```
1,500 calls/s × 0.020 s = 30 concurrent in-flight requests
```

A thread-per-request server needs 30 threads just to hold those calls open, plus the threads for the rest of the request. A monolith that was comfortably using 40 worker threads may now need 70 or more, depending on how much of the request path is blocked. If the pool is not sized for that, requests queue and P99 latency degrades long before CPU saturates.

Now consider the same numbers with a 5ms RTT (same-region, same-AZ, connection reuse):

```
1,500 calls/s × 5ms = 7,500 ms/s = 7.5 CPU-seconds of waiting per second
1,500 calls/s × 0.005 s = 7.5 concurrent in-flight requests
```

The difference between 20ms and 5ms is a factor of four in concurrency demand, which is why the network path matters as much as the number of services.

These figures are illustrative, not measured. The method is what transfers: instrument the call rate, measure the RTT, multiply, and compare against your thread pool and connection pool limits.

## When the conventional wisdom is right

The strangler fig approach works well when the monolith is already modular. If each domain already has its own Django app (or Rails engine, or Spring module) with its own models and URL namespace, the extraction is largely a deployment change. The code does not need to be untangled because it was never tangled.

It also works when the coupling is **resource-based** rather than data-based. If a search workload is starving the primary database of I/O, moving search to its own cluster with dedicated storage relieves a real bottleneck. The service boundary follows the resource boundary, which is a clean line to draw.

It works when the module has a genuinely different scaling profile: a batch job that needs burst CPU, a media transcoder that needs GPU, a workload with a different runtime. The boundary is justified by the resource, not by an architectural preference.

And it works when you are replacing a legacy subsystem outright. A rewrite is sometimes unavoidable — for example, replacing a nightly batch system with a real-time one. In that case the strangler fig is less a migration strategy than a way to land the rewrite incrementally, keeping the old path alive until the new one is proven.

## A decision checklist

Work through these in order. Stop at the first one that applies.

1. **Is there a query or view that dominates database time?** If `pg_stat_statements` shows one statement consuming a large share of total time, fix that first. It is often a stale view or a missing index, and the fix is measured in hours, not months.
2. **Do more than roughly 30% of your top queries join across the proposed service boundary?** If yes, the boundary is wrong or premature. Refactor the schema first: split schemas, remove cross-feature joins, introduce read replicas for read-heavy paths.
3. **Are there atomic operations spanning the boundary?** If yes, introduce an outbox pattern and make consumers idempotent before extracting anything.
4. **What is the measured RTT between the proposed service and its callers?** If it is above single-digit milliseconds, the latency tax will be felt. Consider co-locating, or reducing call frequency with batching.
5. **What is the simulated P95 delta?** If it exceeds your budget, do not extract. The simulation is cheap; the regression is not.
6. **Does the team have observability and rollback in place?** Distributed tracing, per-service dashboards, and a rehearsed rollback procedure are prerequisites, not nice-to-haves. Without them, an incident becomes an outage.
7. **Is the business case resource-based or preference-based?** If the driver is "we should have microservices," the answer is usually no. If the driver is a saturated resource or a genuinely independent scaling need, the answer may be yes.

A comparison of the three common approaches:

| Approach | Primary change | Typical effort | Main risk | When it fits |
|---|---|---|---|---|
| Database-first refactor | Schema, indexes, views, read replicas | Weeks | Underestimating query rewrites | High data coupling, one dominant query |
| Outbox plus events | Async boundaries, consumer idempotency | Weeks to a couple of months | Duplicate delivery if consumers are not idempotent | Atomic operations spanning a proposed boundary |
| Strangler fig extraction | Deployment topology, internal APIs | Months | Latency tax, distributed debugging | Already-modular monolith, resource-driven scaling |

The effort and risk columns are qualitative. The only way to make them quantitative for your system is to run the audit and the simulation described above.

## Objections, and honest responses

**"We need independent deployments for CI/CD."** Independent deployments are a means, not an end. Feature flags and canary releases deliver most of the deployment-risk reduction without splitting the codebase. If the CI pipeline is slow because the test suite is slow, fix the test suite. Splitting the service does not make the tests faster; it makes them distributed.

**"The monolith is spaghetti and we cannot test it."** Extracting services from a codebase you do not understand usually produces distributed spaghetti. The order of operations is wrong. Write characterization tests first — tests that capture current behavior, bugs included — then refactor internally. Only when the module has a stable, tested interface is extraction a mechanical change rather than a rewrite.

**"We need to scale this module independently."** Check whether the module needs independent *compute* or independent *data*. Read replicas, connection pooling, and dedicated queue workers often solve the compute case inside the monolith. A separate service is justified when the scaling requirement is genuinely different in kind — a different runtime, a different storage engine, a different hardware profile.

**"Leadership wants microservices."** Architecture that does not serve a measurable bottleneck tends to be rolled back. The useful move is to translate the request into a resource question: what is saturated, and what change relieves it? Sometimes the answer is a service. Often it is not, and the data to show that is the audit described above.

## Frequently asked questions

**Can a monolith be migrated to services with zero downtime?**
Zero downtime is achievable only if there is no shared mutable state and no atomic operation spanning the boundary. If the monolith uses one database with foreign keys across the proposed services, the migration requires either a dual-write period or a schema refactor first. Dual-write is not zero downtime; it is a longer window during which two systems must agree.

**What is the most common mistake in monolith-to-services migrations?**
Extracting a service before reducing data coupling. The extracted service ends up querying the same tables as the monolith, so the boundary adds latency without adding independence. The second most common mistake is not measuring the latency tax before committing.

**How do you decouple a monolith before extracting services?**
Start with a query audit using `pg_stat_statements`. Identify the statements that dominate total execution time and check whether they join across proposed boundaries. Then split schemas, remove cross-feature joins, add read replicas for read-heavy paths, and introduce an outbox pattern for operations that must be atomic. Extract services only after the data layer is decoupled.

**When should you not extract a service from a monolith?**
When the monolith's P95 latency is already within budget and the extraction would add more than your latency budget per call. When the team lacks observability and rollback. When the module shares foreign keys or complex joins with other modules. When the only driver is architectural preference rather than a saturated resource.

## Summary

The conventional advice to extract services early assumes a modular monolith, a mature team, and infrastructure headroom. Many systems have none of those. The failure mode is a distributed monolith: same data coupling, now with network latency and harder debugging. The safer sequence is to audit coupling at the query level, break atomic dependencies with an outbox pattern, measure the latency tax with a simulated extraction, and only then decide whether a service boundary is justified.

## Do this in the next 30 minutes

Run this against your production PostgreSQL database and read the top 20 rows:

```sql
SELECT
  calls,
  round(total_exec_time::numeric, 1) AS total_ms,
  round(mean_exec_time::numeric, 2)  AS mean_ms,
  left(query, 100)                  AS query_snippet
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

For each row, note which tables appear in the query. If more than about a third of the top 20 join tables that you consider separate features, your next task is a schema refactor, not a service extraction.
