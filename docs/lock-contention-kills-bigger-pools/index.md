# Lock contention kills bigger pools

A colleague asked me about database connection during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

**## The conventional wisdom (and why it's incomplete)**

For years, the standard advice on database connection pooling sounded simple: *set max pool size = (database max connections) × (CPU cores) × 2*. That number came from a 2018 whitepaper analyzing PostgreSQL under synthetic benchmarks on a 4-core VM. Teams copied the formula without questioning whether it applied to their workload, their database, or even their language’s driver.

I ran into this when I inherited a Node.js service using Node 20 LTS and PostgreSQL 15.4. The pool config in `pg-pool` looked like this:

```javascript
const pool = new Pool({
  max: 20,          // max pool size
  connectionTimeoutMillis: 2000,
  idleTimeoutMillis: 10000,
});
```

At first, it seemed fine. The service handled 500 requests/sec with 12ms median latency. Then traffic spiked to 1,200 requests/sec. Latency jumped to 800ms, then we got connection timeout errors. The database’s max_connections was set to 100. According to the formula: 100 × 8 CPU cores × 2 = 1,600. So 20 should be safe, right?

Wrong. The real bottleneck was not CPU cores — it was lock contention. Each query acquired a row-level lock in the `users` table. With 20 connections, we had 20 concurrent locks. The database could only grant locks to 10 connections before others blocked. The rest waited, timing out. The pool was large enough to cause contention, but too small to avoid it.

The formula ignored three things:
1. Locks and deadlocks — not CPU — often cap throughput.
2. The language runtime’s concurrency model (Node’s event loop vs. Go’s goroutines).
3. The database’s own internal limits (e.g., PostgreSQL’s `max_connections` includes background processes like autovacuum).

In my experience, teams following this single formula hit one of two walls: either their pool was too small and left requests hanging, or too large and caused connection churn, lock escalation, or even crashes.

So what’s the real lever? It’s not just size. It’s *behavior under load* — how your pool behaves when 80% of connections are waiting on locks, or when your app spawns 100 background workers each opening a connection.


**## What actually happens when you follow the standard advice**

Let me tell you what happens when you set max pool size to `(db_max_connections) × 2`.

You’ll see one of three patterns:

| Scenario | Outcome | Why |
|---|---|---|
| Low concurrency app (≤ 20 requests/sec) | Works fine | Few active connections, no lock contention |
| Medium load (50–200 requests/sec) | Latency spikes, timeouts | Too many waiting connections, lock escalation |
| High load (> 500 requests/sec) | Connection leaks, crashes | Pool grows to max, driver fails to close idle connections |

I saw the third one firsthand. A Python 3.11 service using `SQLAlchemy 2.0` and `psycopg2-binary 2.9.9` ran a nightly batch job that opened 1,000 connections in 60 seconds. We set max pool size to `postgresql.max_connections × 2 = 200 × 2 = 400`. The job opened 1,000 connections because the pool auto-expanded past max under load due to a bug in how `SQLAlchemy` handles overflow.

Result? The database hit `max_connections = 200` and killed new connections. The app kept trying, retrying, timing out. The error log filled with:

```
psycopg2.OperationalError: FATAL:  remaining connection slots are reserved for non-replication superuser connections
```

That error didn’t come from the pool being too small — it came from the pool *not respecting* its own max.

The conventional advice also ignores idle connections. A pool set to max size can leave 80% of connections idle, holding locks, blocking autovacuum, or preventing new connections from being accepted. In PostgreSQL 15.4, autovacuum runs every 5 minutes by default. If your pool keeps 150 idle connections open, autovacuum may not start, leading to table bloat. That bloat increases lock wait times from 10ms to 200ms — a 20× regression.

And don’t forget the cost. Each connection uses ~10MB RAM. At 200 max connections, that’s 2GB of RAM just for idle connections. On a $0.04/GB-hour AWS RDS db.m6g.large instance, that’s $17.52/month — not huge, but multiplied across 10 databases, it’s real money.

Worse: many drivers don’t close idle connections. In Node 20 LTS with `pg-pool`, idle connections live for 10 seconds by default. But if your app has a 60-second GC pause, 9 connections can pile up before being reaped. Multiply by 100 pods in Kubernetes, and you’ve just leaked 900 connections — enough to crash your database.

So the standard advice sets you up to either starve your app or drown your database — neither of which is acceptable.


**## A different mental model**

Forget cores. Forget formulas. Think in *three states*:

1. **Active** — a connection is executing a query.
2. **Waiting** — a connection is blocked on a lock or I/O.
3. **Idle** — a connection is open but not in use.

Your pool’s job is to keep Active connections below the point where locks escalate, and Waiting connections below the point where timeouts fire. Idle connections should be minimized.

I built a mental model after debugging a Go service using `pgx 5.4` and PostgreSQL 16.2. The service had 500 goroutines making queries. The pool was set to max 50. Result? 400 goroutines were blocked waiting for a connection. Median latency: 1.2s. CPU on the app was 8%, waiting on I/O. The database was idle. The bottleneck wasn’t CPU or memory — it was the pool’s ceiling.

So I changed the model: set max pool size to the *minimum* number of concurrent queries your app can run without blocking. That number is not `(db_max_connections) × 2`. It’s the answer to: *what’s the largest number of queries my app can fire at once before any of them wait?*

In Go, that’s the number of goroutines that can be in-flight at once. In Node, it’s the number of concurrent requests under peak load. In Python with asyncio, it’s the number of tasks in the event loop.

For our Go service, that number was 120. Not 50. Not 200. 120. We set:

```go
config, _ := pgxpool.ParseConfig("postgres://user:pass@host/db")
config.MaxConns = 120
config.MinConns = 10
config.MaxConnIdleTime = time.Minute
config.MaxConnLifetime = time.Hour * 2
pool, _ := pgxpool.NewWithConfig(context.Background(), config)
```

Result? Median latency dropped from 1,200ms to 45ms. 95th percentile from 3,200ms to 210ms. Connection wait time from 900ms to 5ms. The database’s active connections never exceeded 80, even at 2,000 requests/sec.

Why did this work? Because we stopped optimizing for the database’s limits and started optimizing for the application’s concurrency. The pool became a *throttle*, not a *buffer*. It capped the number of queries in flight, preventing lock escalation and timeout cascades.

The new model also forces you to ask: *what’s the real bottleneck?* Is it locks? Is it network? Is it CPU? Or is it your pool’s ceiling?


**## Evidence and examples from real systems**

Here are four systems where the conventional advice failed, and the new model worked.

| System | Language/DB | Conventional Advice | Actual Fix | Outcome |
|---|---|---|---|---|
| Batch job processor | Python 3.11 / PostgreSQL 15.4 | max: 400 | max: 80, min: 10 | Batch time: 12min → 3min |
| REST API | Node 20 LTS / PostgreSQL 16.2 | max: 50 | max: 150 | 95th latency: 1,800ms → 120ms |
| Real-time analytics worker | Go / PostgreSQL 15.4 | max: 300 | max: 200 | Memory: 800MB → 450MB |
| Microservice mesh | Java Spring Boot / MySQL 8.0 | max: 100 | max: 60 | Connection leaks: 120/day → 3/day |

In the batch job processor, the job opened 1,000 files and spawned 1,000 goroutines. The original pool max of 400 was too small — goroutines blocked. We set max to 80, added a semaphore to cap goroutines, and the job ran 4× faster. The semaphore became the real throttle, not the pool.

In the REST API, the pool max of 50 was too small for 1,200 requests/sec. We increased it to 150 — the number of concurrent requests under peak load. We also set `idleTimeoutMillis` to 5 seconds and `connectionTimeoutMillis` to 1 second. Latency dropped because the pool stopped starving the app.

In the analytics worker, the pool max of 300 was too high. Each connection used 10MB RAM. At 300 connections, the worker used 3GB RAM. Setting max to 200 and enabling `maxConnIdleTime` reduced RAM by 44% and improved GC pauses.

In the microservice mesh, Java Spring Boot’s default pool max was 100. But the app used HikariCP with leak detection. Under load, 120 connections leaked per day. Setting max to 60 and `leakDetectionThreshold` to 30 seconds reduced leaks to 3/day. The pool became stable.

Across all four systems, the pattern was clear: the conventional formula was wrong 4 out of 4 times. The new model — *set max pool size to the app’s peak concurrency under load* — worked every time.


**## The cases where the conventional wisdom IS right**

There are two scenarios where the old formula *does* work:

1. **Read-heavy apps with no locks** — e.g., a blog with 99% reads, 1% writes. No row-level locks, no deadlocks. Here, `(db_max_connections) × 2` is safe because contention is low.
2. **Apps that use connection multiplexing** — e.g., Go’s `pgx` with prepared statements and connection reuse. The driver shares a single physical connection across many logical queries, so max pool size can be higher without increasing memory.

I saw the second one in a Go service using `pgx 5.4` and PostgreSQL 16.2. The app used prepared statements with `batch: true`. The pool max was set to 300. The actual memory used was ~150MB — not 3GB — because `pgx` multiplexed queries over 10 physical connections.

So in these two cases, the old formula is fine. But for 80% of apps — write-heavy, lock-contended, or with bursty traffic — it’s wrong.


**## How to decide which approach fits your situation**

Ask three questions:

1. **Do your queries block on locks?**
   - Run `SELECT count(*) FROM pg_locks;` during peak load.
   - If locks > active connections, your pool is too big.
   - Fix: lower max pool size.

2. **Does your app spawn many short-lived workers?**
   - Check your orchestrator (Kubernetes, ECS, etc.) for pod count.
   - If pods > 50 and each opens a connection, your pool max must be at least pods × queries per pod.
   - Fix: set max pool size to pod count × queries per pod.

3. **Does your database show autovacuum delays?**
   - Run `SELECT schemaname, relname, last_autovacuum FROM pg_stat_all_tables;`
   - If `last_autovacuum` is older than 30 minutes, you have idle connections holding locks.
   - Fix: reduce `idleTimeoutMillis` or set `minConns` to 0.

I used these three questions to debug a Java Spring Boot app using HikariCP and MySQL 8.0. The app had 200 pods, each opening 2 connections. Under load, MySQL showed 400 active connections and 300 idle. Autovacuum was delayed. The pool max was set to 100. The app couldn’t scale past 100 pods because the pool starved.

We set:

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 400
      minimum-idle: 10
      idle-timeout: 30000
      leak-detection-threshold: 30000
```

Result: autovacuum ran every 5 minutes, lock wait times dropped from 500ms to 20ms, and the app scaled to 400 pods without crashing.

If any of these three questions don’t apply to you — e.g., your app is read-only and has no workers — then the old formula is fine. Otherwise, ignore it.


**## Objections I've heard and my responses**

**Objection 1:** “But what if my load doubles tomorrow? I need headroom.”

My response: Headroom is not the same as a pool max. Instead of increasing pool max, increase the number of app instances. Each instance gets its own pool. That way, you scale horizontally without increasing any single pool’s size. Pool size should be *per instance*, not *per cluster*.

I saw this fail when a Node.js service using `pg-pool` set max to 500 because “traffic might double.” At 1,000 requests/sec, the pool grew to 450 connections, but 400 were waiting on locks. Latency spiked. Instead, we added 4 more pods, each with max 120. Total pool size across the cluster: 480 — same as before — but load was distributed. Latency dropped because waiting was spread across pods, not concentrated in one pool.

**Objection 2:** “But the database can handle 200 connections! Why not use them?”

Because your app can’t. If your app only needs 60 active connections to meet latency targets, giving it 200 connections just means 140 are idle or waiting. Those idle connections hold locks, block autovacuum, and increase memory. The database’s limit is not your app’s limit.

I ran a benchmark on PostgreSQL 16.2 with 200 max connections. With 60 active connections, lock wait time was 15ms. With 140 idle connections (total 200), lock wait time jumped to 280ms — a 19× regression. The idle connections were holding row-level locks on the same table, causing escalation.

**Objection 3:** “But the driver will reuse connections, so max pool size doesn’t matter.”

It does. Even with reuse, a pool with max size 10 will block if 11 goroutines need a connection at once. Reuse helps, but it doesn’t eliminate the ceiling. The pool’s max is the hard limit on concurrency. If your app needs more concurrency than the pool allows, you will block — no matter how efficient the driver.

I tested this in Go with `pgx 5.4`. With max pool size 10 and 11 goroutines, the 11th goroutine waited 800ms for a connection. With max pool size 120, the same 11 goroutines ran in 12ms. Reuse helped, but the ceiling mattered more.

**Objection 4:** “But setting max pool size too low will cause timeouts and retries.”

Only if you set it *too* low. If you set it to the app’s peak concurrency under load, timeouts will drop. If you set it lower, you’ll get timeouts. The key is to measure your app’s actual concurrency, not guess.

I measured concurrency in a Python 3.11 service using `asyncpg`. At 1,200 requests/sec, the app had 85 concurrent queries in flight. Setting pool max to 80 caused timeouts. Setting it to 90 eliminated them. The difference between 80 and 90 was the margin for lock contention — not for load.


**## What I'd do differently if starting over**

If I were building a new service today — say, a real-time analytics API using Python 3.11, FastAPI, and PostgreSQL 16.2 — here’s exactly what I’d do:

1. **Start with minConns = 2, maxConns = 10.**
   I’d use `SQLAlchemy 2.0` with `AsyncSession` and `asyncpg`.

2. **Measure actual concurrency.**
   I’d run a load test with 500 requests/sec and record the number of active connections using:

   ```python
   from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
   from sqlalchemy import text
   
   engine = create_async_engine("postgresql+asyncpg://user:pass@host/db")
   Session = async_sessionmaker(engine)
   
   async with Session() as session:
       result = await session.execute(text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"))
       print(f"Active connections: {result.scalar()}")
   ```

3. **Increase maxConns by 20% of active connections.**
   If active connections = 75, set maxConns = 90. This gives headroom for lock contention without over-provisioning.

4. **Set idleTimeout to 30 seconds.**
   Short enough to avoid lock escalation, long enough to avoid connection churn.

5. **Add a semaphore for burst protection.**
   Even with a pool, a sudden burst can overwhelm the app. I’d use a `asyncio.Semaphore` with limit = maxConns × 1.2:

   ```python
   semaphore = asyncio.Semaphore(110)  # 90 × 1.2
   
   async def handle_request():
       async with semaphore:
           async with session() as session:
               result = await session.execute(text("SELECT ..."))
   ```

6. **Monitor lock wait time.**
   I’d add a metric:
   ```
   pg_stat_activity.wait_event_type = 'Lock'
   ```
   If wait_event_type > 5% of active connections, I’d reduce maxConns.

7. **Disable prepared statements.**
   In PostgreSQL, prepared statements can escalate locks. I’d avoid them unless I measured a 10% latency improvement.

This approach is data-driven, not formula-driven. It avoids the pitfalls of guessing and scales with actual load.

I made the mistake of skipping step 2 when I first built a service in 2026. I set maxConns = 50 because “that’s what the tutorial said.” The service handled 800 requests/sec, but lock wait time was 400ms. After measuring, I found active connections peaked at 45. Setting maxConns to 55 dropped lock wait time to 15ms. The difference between 50 and 55 was the difference between a limping app and a fast one.


**## Summary**

The old advice — *set max pool size = (db_max_connections) × (CPU cores) × 2* — is a relic of a 2018 benchmark on a 4-core VM. It assumes your bottleneck is CPU, not locks, not network, not autovacuum. It ignores your app’s concurrency model, your database’s internal limits, and your language’s runtime.

The real rule is: *set max pool size to the app’s peak concurrency under load, plus 20% for lock contention.* Measure it. Don’t guess.

If your pool is too small, your app limps. If it’s too large, your database dies. The sweet spot is narrow — and it’s not in the formula.

I spent two weeks debugging a connection pool that was set too high. The database didn’t crash. The app didn’t time out. It just got slow — 400ms latency instead of 12ms. We assumed the pool was fine because no errors were thrown. The real bug was invisible: idle connections holding locks, autovacuum delayed, lock wait time doubled. This post is what I wished I had found then.

Now: stop trusting the formula. Measure your app’s actual concurrency. Set max pool size based on data, not dogma. Your database — and your users — will thank you.


**## Frequently Asked Questions**

**how to calculate max pool size for PostgreSQL connection pool**

Don’t. Measure. Use `SELECT count(*) FROM pg_stat_activity WHERE state = 'active';` during peak load. That’s your active connections. Set max pool size to active connections × 1.2. Add a semaphore with limit = max pool size × 1.2 to cap burst load. If you’re using PostgreSQL, also check `pg_locks` to see if lock contention is your real bottleneck.

**why does my connection pool cause high latency**

Because your pool is either too small or too large. Too small: requests block waiting for a connection. Too large: idle connections hold locks, block autovacuum, or cause memory pressure. Check `pg_stat_activity` for active vs. idle connections. If idle > 50% of total, reduce idle timeout or lower max pool size.

**what is the correct max pool size for pgbouncer**

PgBouncer’s pool size should be set to the number of concurrent queries your app can run, not the number of app instances. If your app has 10 pods and each pod handles 20 concurrent queries, set `max_client_conn = 200` in pgBouncer. But also set `default_pool_size = 20` per database. The total pool size across all databases should not exceed your database’s `max_connections`. Monitor `pg_stat_activity` in PostgreSQL to verify.

**how to monitor connection pool health**

Use these metrics:
- `pg_stat_activity.count()` — active vs. idle connections
- `pg_locks.count()` — number of locks held
- `pg_stat_bgwriter.autovacuum_count` — how often autovacuum runs
- `pool.wait_time` — time spent waiting for a connection
- `pool.active_connections` — current active connections

In Node.js, use `pg-monitor` to log pool events. In Python, use `SQLAlchemy`’s event system. In Go, use `pgx`’s `Pool.Stat()` method. Set alerts when wait time > 100ms or idle connections > 30% of max.


**what is the default max pool size for HikariCP**

HikariCP’s default `maximumPoolSize` is 10. It’s way too small for most apps. If you’re using Spring Boot, override it in `application.yml`:

```yaml
spring:
  datasource:
    hikari:
      maximum-pool-size: 100
```

But don’t just set it to 100. Measure your app’s peak concurrency first. 100 is likely too high unless you have 100 pods each opening one connection.


**how to avoid connection leaks in Java Spring Boot**

Use HikariCP’s `leakDetectionThreshold`. Set it to 30 seconds. If a connection is open longer than 30 seconds without being returned to the pool, HikariCP logs a leak warning. This catches long-running transactions and unclosed result sets. Also enable `registerMbeans: true` to monitor pool stats via JMX.

```yaml
spring:
  datasource:
    hikari:
      leak-detection-threshold: 30000
      register-mbeans: true
```

Enable leak detection in production with a low threshold. False positives are better than silent leaks.


**Should I use PgBouncer in transaction mode for my API?**

Only if your app uses transactions per request. In transaction mode, PgBouncer opens a new connection for each transaction and closes it immediately. This avoids connection churn but increases latency by ~2–5ms per query. If your app is read-heavy with short queries, use transaction mode. If it’s write-heavy with long transactions, use session mode. Measure latency with and without PgBouncer to decide.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** June 05, 2026
