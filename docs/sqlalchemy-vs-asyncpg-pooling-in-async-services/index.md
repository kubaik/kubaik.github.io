# SQLAlchemy vs asyncpg: pooling in async services

## The pool is the service's real concurrency limit

Every async Python service that talks to Postgres ends up making the same set of decisions, usually in the wrong order. Someone picks SQLAlchemy because the ORM is familiar. Someone else picks asyncpg because it is faster. Then the service goes to production behind a pgbouncer, or doesn't, and weeks later requests start timing out under load with an error that reads like a network problem but isn't one.

The comparison that matters isn't SQLAlchemy versus asyncpg in the abstract. SQLAlchemy's async support uses asyncpg as one of its supported drivers — the choice is not between two unrelated stacks so much as a choice about how much of the connection lifecycle is managed for you. The real question is: **who owns the pool, how big is it, and what happens when it runs dry?**

That question has different answers depending on whether you're on a managed Postgres with a hard connection cap, a shared host where the database is a neighbor rather than a dedicated machine, or a container platform where every replica opens its own pool. This article walks through the mechanics, the failure modes, and a decision framework you can apply to your own service.

## Two layers, not two competitors

The framing that trips people up: SQLAlchemy's `create_async_engine` with an `asyncpg` URL does not replace asyncpg's pool with SQLAlchemy's. It builds on top of it. When you write:

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@db.internal:5432/app",
    pool_size=10,
    max_overflow=5,
    pool_timeout=10,
    pool_recycle=1800,
)
```

you are configuring SQLAlchemy's `AsyncAdaptedQueuePool`. Underneath, SQLAlchemy asks the asyncpg dialect to open raw asyncpg connections and wraps them. The `pool_size` and `max_overflow` numbers are SQLAlchemy's, not asyncpg's. If you instead call `asyncpg.create_pool()` directly, you get asyncpg's pool with its own `min_size` and `max_size` semantics, and no ORM, no `Session`, no unit of work.

So the honest comparison is:

- **SQLAlchemy async engine** — ORM, query construction, migrations via Alembic, session lifecycle, and a pool you configure with SQLAlchemy's parameters.
- **asyncpg pool directly** — raw protocol driver, prepared statements, binary result decoding, and a pool you configure with asyncpg's parameters. No ORM layer.

Both end up holding Postgres backend processes open. Both can exhaust `max_connections`. Both need the same discipline around checkout and release.

## What asyncpg's pool actually guarantees

asyncpg's `Pool` is a pool of connections created eagerly up to `min_size`, growing lazily to `max_size`. When you acquire a connection, you either get one from the idle set, one is created for you (if under `max_size`), or you wait. The acquire timeout is configurable; if you don't set it, a saturated pool will block the caller until a connection frees up or the wait is cancelled.

The properties that matter in production:

- **Prepared statement caching is per-connection.** asyncpg caches prepared statements on each connection. This is a large part of why it's fast on repeated queries — but it means a connection that has been open a long time accumulates state. Recycle limits matter.
- **No automatic transaction management.** You write `async with conn.transaction():` or you don't. There is no session, no flush, no identity map. This is a feature for some services and a footgun for others.
- **The connection is not thread-safe.** asyncpg connections are bound to the event loop they were created on. If your service uses multiple event loops (e.g. a background worker thread with its own loop), you need a separate pool per loop. This is a real source of "attached to a different loop" errors.

A minimal correct pattern:

```python
import asyncpg

async def fetch_users(pool: asyncpg.Pool, limit: int = 100):
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, email FROM users ORDER BY id LIMIT $1",
            limit,
        )
    return rows

# at startup
pool = await asyncpg.create_pool(
    dsn="postgresql://user:pass@db.internal:5432/app",
    min_size=2,
    max_size=10,
    command_timeout=30,
)
```

The `async with pool.acquire()` is the whole discipline. Every code path that touches the database must go through it, and the block must not span an `await` on something slow (an HTTP call, a queue put) unless you genuinely want to hold the connection for that long. Holding a connection across an unrelated await is the single most common way teams exhaust a pool without high query load.

## What SQLAlchemy's async engine adds

SQLAlchemy's async engine wraps the driver pool with a `QueuePool`-style layer that supports `pool_size`, `max_overflow`, `pool_timeout`, `pool_recycle`, and `pool_pre_ping`. The session layer adds a unit of work: you `add()` objects, `flush()` to get IDs, `commit()` to persist. For an async service, the relevant pattern is `AsyncSession` with an explicit context manager:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from sqlalchemy import select
from models import User

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_user(session: AsyncSession, user_id: int) -> User | None:
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def create_user(session: AsyncSession, email: str) -> User:
    user = User(email=email)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
```

`expire_on_commit=False` is not optional in async code. With the default `True`, attribute access after commit triggers a lazy load — which in async context raises `MissingGreenlet` because there's no greenlet to run the I/O. This error is one of the most common first-week surprises for teams moving a sync SQLAlchemy codebase to async, and it is not a bug in SQLAlchemy; it's a consequence of the object model assuming synchronous attribute access.

The `pool_pre_ping=True` option issues a lightweight check on checkout to detect dead connections — useful behind a pgbouncer or a network that drops idle TCP sessions. It costs a round trip per checkout, which is fine at moderate rates and measurable at high ones.

## Head-to-head on the axes that decide it

| Axis | SQLAlchemy async engine | asyncpg pool directly |
|---|---|---|
| Pool configuration | `pool_size`, `max_overflow`, `pool_timeout`, `pool_recycle`, `pool_pre_ping` | `min_size`, `max_size`, acquire timeout, `command_timeout` |
| Query construction | ORM + Core expression language | SQL strings with `$1` placeholders |
| Transaction handling | `AsyncSession` with explicit `begin()` / `commit()` | `async with conn.transaction():` |
| Prepared statement caching | Per-connection, via driver | Per-connection, native |
| Schema migrations | Alembic integrates directly | Separate tooling |
| Type mapping | Python types via SQLAlchemy type system | asyncpg's own codecs, `json`/`jsonb` need explicit handling |
| Overhead per query | Higher — ORM layer, session bookkeeping | Lower — direct protocol |
| Loop affinity | Engine bound to loop | Pool bound to loop |
| Learning curve for existing Django/Flask teams | Familiar | Requires understanding raw SQL and manual mapping |

The performance gap between the two is real but often overstated for services that spend most of their request time in Python business logic or in other network calls. Where it shows up: high-throughput read endpoints returning many small rows, bulk inserts, and any path where the ORM's identity map and lazy-load machinery cost more than the query itself. Where it doesn't: services doing a handful of queries per request behind a 100ms external API call.

Because the gap is workload-specific, measure it rather than assume it. Two practical approaches:

1. **Microbenchmark the query path in isolation.** Run the same query N times through each stack against a seeded database, discard warm-up iterations, and compare the distribution of per-query latencies, not just the mean. Include the checkout/acquire step, since that is where pool contention appears.
2. **Instrument the real service.** Wrap the database call in a timer and emit a histogram keyed by query shape. Compare p50 and p99 for the same endpoint before and after a stack change, with the same traffic profile. A difference that disappears once you add the surrounding HTTP call is not worth a rewrite.

## Pool sizing: the arithmetic teams skip

The number that causes incidents is not `pool_size`. It's `pool_size × replicas × (1 + max_overflow_ratio)`. Postgres has a hard `max_connections` (default 100 on many managed tiers, though plan-dependent). If you have 8 replicas with `pool_size=10` and `max_overflow=5`, your theoretical peak is 8 × 15 = 120 — over the cap. Under load, the 101st connection attempt fails with `FATAL: sorry, too many clients already`, and Postgres starts refusing new connections from every client, including your migration tool and your on-call engineer's psql session.

Assume, illustratively, a service with 4 replicas, `pool_size=10`, `max_overflow=10`, and a managed Postgres with `max_connections=100` and 10 reserved for admin: 4 × 20 = 80. That leaves 10 for migrations and humans. Add a rolling deploy where old and new pods overlap and you temporarily have 8 replicas = 160 — over the limit. This is why the pool size calculation must account for your *deployment strategy*, not just steady state.

The fix is one of:

1. **Reduce pool size per replica and accept queueing.** A pool of 5 with `pool_timeout=5` will queue requests rather than fail them, which is usually better than a hard connection refusal.
2. **Introduce a connection pooler** (pgbouncer in transaction mode) between the app and Postgres. The app's pool then talks to pgbouncer, and pgbouncer multiplexes many client connections onto far fewer server connections. This changes the failure mode: instead of `too many clients`, you get pgbouncer queueing, which is visible in its `SHOW POOLS` output.
3. **Use a managed pooler** offered by your database provider. The mechanics are similar; the operational surface differs.

pgbouncer in transaction mode has a specific consequence for SQLAlchemy and asyncpg: **prepared statements cannot be assumed to persist across transactions**, because a transaction may land on a different server connection. asyncpg's statement cache becomes a liability here. The usual resolutions are to disable statement caching (`statement_cache_size=0` in the asyncpg connect args, passed through SQLAlchemy's `connect_args`) or to use session mode. Disabling the cache costs some throughput; session mode costs multiplexing. Neither is free.

## The failure modes, in the order they bite

**1. Connection held across an external await.** A handler acquires a session, then calls a third-party API, then queries. Under load, every in-flight request holds a connection while waiting on the external API. The pool drains, `pool_timeout` fires, and the error looks like a database problem. The fix is to scope the session to the database work only — acquire, query, release, then call the API.

**2. `MissingGreenlet` after commit.** Covered above; `expire_on_commit=False` or explicit `refresh()` after commit.

**3. Event loop mismatch in background workers.** A pool created on the main loop is used from a worker loop. asyncpg raises an error about the connection being attached to a different loop. The fix is a per-loop pool, created inside the worker's `asyncio.run()`.

**4. Stale connections after a network blip.** Idle TCP connections through a NAT or a load balancer get dropped silently. The next query fails with a connection-reset error. `pool_pre_ping=True` (SQLAlchemy) or a periodic `SELECT 1` (asyncpg) detects this; `pool_recycle` bounds how long a connection can be idle before being discarded.

**5. Fork-safety.** Creating the engine or pool at module import time and then forking (gunicorn with preload, for example) means the child inherits a pool pointing at connections the parent owns. The child's first query fails or, worse, corrupts protocol state. Create the engine in a startup hook that runs after the fork, not at import.

**6. `max_overflow` masking saturation.** Overflow connections are created on demand and disposed when returned. If your overflow is large, you don't see pool exhaustion until you've already blown past the server's `max_connections`. A smaller overflow with a clear `pool_timeout` surfaces the problem earlier and more legibly.

**7. Transactions left open.** A session that begins a transaction and never commits or rolls back keeps its connection in `idle in transaction` state, which holds locks and blocks vacuum. This is invisible in pool metrics until the pool is empty. Set a `statement_timeout` and an `idle_in_transaction_session_timeout` on the server side so the database reclaims these connections itself.

## Who should pick which

**Pick SQLAlchemy's async engine when:**

- You have an existing SQLAlchemy codebase and the migration cost of rewriting queries is real.
- You use Alembic for migrations and want schema changes to stay in the same toolchain.
- Your team's mental model is ORM-first and the queries are not the bottleneck.
- You need the unit-of-work pattern — multiple related writes that must commit together with identity tracking.

Its weaknesses: the ORM layer adds overhead, the async session has sharp edges (`expire_on_commit`, lazy loading, `MissingGreenlet`), and the pool configuration is one more layer between you and the driver. Debugging a pool exhaustion incident means understanding both SQLAlchemy's pool and asyncpg's pool.

**Pick asyncpg directly when:**

- Your service is query-heavy and the ORM overhead is measurable against your latency budget.
- You are comfortable writing SQL and mapping rows explicitly.
- You want the smallest possible dependency surface and the clearest pool semantics.
- You are building a data pipeline or a worker where the ORM's identity map provides no value.

Its weaknesses: no migrations story (you'll use Alembic standalone or another tool), manual type mapping for `jsonb` and arrays, no relationship loading, and every query is a string you have to keep correct. Teams that pick asyncpg for speed and then rebuild half an ORM on top of it have usually made the wrong trade.

**A hybrid is legitimate:** SQLAlchemy for the write path where the ORM earns its keep, asyncpg directly for a read-heavy hot path. The cost is two pools, two sets of connection accounting, and two places to check when connections run out.

## What to check before you change anything

Before choosing or resizing, measure the current state. The two numbers that matter are your **observed peak concurrent connections** and your **p99 time spent waiting for a connection**. The first you can get from Postgres:

```sql
SELECT count(*), state
FROM pg_stat_activity
WHERE datname = 'app'
GROUP BY state;
```

Run it during your busiest window. If `active` plus `idle in transaction` is close to your pool size, you are saturated. If `idle` is large, your pool is oversized and you are wasting server connections.

The second — connection wait time — is not exposed by default in SQLAlchemy. You can instrument it by timing the `session.begin()` or `pool.acquire()` call in a middleware and emitting a histogram. If the p99 of that metric is non-trivial relative to your query p99, the pool is the bottleneck, not the database.

**The next 30 minutes:** open your service's database configuration file, count the replicas in your production deployment, multiply by `pool_size + max_overflow`, and compare the result to your Postgres `max_connections` minus reserved admin slots. If the product exceeds the cap, you have found a latent incident — reduce the pool size or add a pooler before the next traffic spike finds it for you.
