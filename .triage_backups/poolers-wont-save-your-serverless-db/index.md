# Poolers won’t save your serverless DB

## Why pooler guides stop where the interesting part starts

Most connection pooler write-ups assume steady web traffic and long-lived application servers. Serverless and agent workloads break both assumptions, and the guidance rarely gets updated. This article covers what actually changes: where the latency goes, which costs appear, which failure modes are common, and how to measure all of it on your own system rather than trusting a vendor's slide.

Two clarifications before going further. First, "pooler" here means a process that keeps persistent backend connections open and hands them to clients — PgBouncer, ProxySQL, Amazon RDS Proxy, and similar. Second, "multiplexer" means a proxy that terminates many logical client connections onto a small number of physical backend connections without per-client TLS renegotiation. The distinction matters because the two designs fail differently.

## What a pooler actually does, and why serverless flips the contract

A pooler sits between clients and the database and keeps a set of persistent connections open so clients avoid the TCP and TLS handshake on every request. Under steady traffic from long-lived application servers, that is a clear win: the handshake cost is amortized across thousands of queries.

Serverless runtimes invert the assumptions the design was tuned for:

- Clients are short-lived and appear in bursts. A function instance may run for a few hundred milliseconds and then disappear.
- The client-facing connection is short-lived even when the backend connection is not. If the pooler sits behind a load balancer that terminates TLS, every new function instance still pays a fresh TCP and TLS handshake to the pooler.
- Idle time dominates. An agent that queries the database, waits several seconds for a model response, then queries again spends most of its connection lifetime idle. Poolers that bill or meter by connection-open time charge for that idle period.

The result is that in bursty workloads the pooler can be a net addition to the critical path rather than a subtraction from it. The backend connection is reused; the client-side cost is not.

A useful way to think about it: a pooler optimizes the *backend* side of the connection. Serverless latency problems usually live on the *client* side and in the network path between the client and the pooler. Fixing one does not fix the other.

## Where the latency actually goes

Break a serverless request into phases and measure each one rather than reasoning about the total.

**Connect phase.** The function opens a TCP socket to the pooler endpoint. If TLS is enforced at the pooler or at a load balancer in front of it, the handshake costs one or more round trips. If the pooler is behind an application load balancer, that adds another hop. The exact cost depends on region, availability zone placement, and whether the client and pooler are co-located, so measure it rather than assuming a number.

**Authentication phase.** Some poolers re-authenticate each new client session even when the backend connection is reused. Look for whether your pooler requires client credentials on every new connection and whether that check is cached.

**Query phase.** The pooler routes the query to a backend connection. In transaction pooling mode the backend is held only for the duration of a transaction; in session mode it is held for the whole client session. Transaction mode is usually the better fit for short bursts, but it requires care with session-level state such as temporary tables, advisory locks, and `SET` commands, which do not survive across transactions on a shared backend.

**Idle phase.** If the pooler or your provider meters connection-open time, the idle phase is where cost accumulates. This is invisible in latency graphs and shows up only on the bill.

To measure the connect phase in isolation, run a trivial `SELECT 1` in a loop from a cold function and record the time from socket open to first row returned, then compare against the same query issued over an already-open connection. The difference is your per-request connection overhead. Do this in the same region and availability zone as production, because cross-AZ placement can dominate the result.

## The costs that do not appear in the pricing page

Latency is the visible problem; cost is the one that surprises teams at the end of the month. Three cost centers are worth instrumenting:

1. **Per-connection charges.** Some managed poolers bill per million proxied connections. If your function opens a connection per invocation, that count tracks invocations, not queries.
2. **Data transfer.** If the pooler runs in a different availability zone or VPC from the client or the database, every byte crosses a boundary that may be billed. This is easy to miss because it is a line item on the network bill, not the database bill.
3. **Idle connection time.** Where the provider meters open connections, idle time is billed time.

The honest way to evaluate a pooler's cost is to compute it from your own traffic shape. As a worked example with illustrative numbers: suppose a workload issues 100 million function invocations per month, each opening one connection to a pooler, and the pooler charges $0.02 per million connections. That is 100 × $0.02 = $2 per month in connection charges — small. Now suppose each invocation transfers 8 KB through the pooler and the cross-AZ transfer rate is $0.01 per GB in each direction. That is 100,000,000 × 8 KB = 800,000,000 KB = 800 GB, costing 800 × $0.01 × 2 = $16 per month. Still small at this volume. The point is not that any specific figure is large; it is that you must substitute your own invocation count, payload size, and provider rates, because the relative weight of connection charges versus transfer charges changes completely with payload size. A workload streaming large JSON blobs is dominated by transfer; a workload issuing tiny queries is dominated by connection count.

Build a small spreadsheet with invocation count, average payload size, connection charge per million, and transfer rate per GB, and compute both. That is more reliable than any published benchmark, including the ones in this article's earlier drafts.

## A multiplexer is not a pooler

The alternative to a pooler in a bursty serverless environment is a multiplexer: a lightweight proxy that terminates many logical client connections onto a small number of physical backend connections, without requiring a fresh TLS handshake per client.

The important properties:

- The client-to-proxy leg is cheap and can be terminated at a gateway that is already in the request path, removing a separate TLS negotiation.
- The proxy-to-database leg is a small, fixed number of long-lived connections, typically one or a few per shard.
- Logical clients share physical connections, which means session-level state cannot be assumed to persist between statements. This is the same constraint as transaction pooling mode and it is the source of most multiplexer bugs.

Where a pooler reduces backend connection count, a multiplexer also reduces the *client-side* handshake cost by removing the extra TLS hop. That is the part that shows up in p50 latency.

## Implementation walkthrough

The following pattern replaces a per-invocation connection to a pooler with a connection to a multiplexer that sits behind the API gateway, with TLS terminated at the gateway.

### Step 1: Choose the right mechanism for your database

- **PostgreSQL, self-managed or RDS:** a connection multiplexer such as PgBouncer in transaction mode, or a purpose-built multiplexer proxy. Evaluate whether your workload can tolerate shared backend sessions.
- **MySQL or Aurora MySQL:** ProxySQL supports multiplexing; verify its prepared-statement and session-state behavior against your workload.
- **Aurora Serverless:** the Aurora Data API is an HTTP-based interface that does not require a persistent connection from the client at all. For many serverless workloads this removes the pooler question entirely.
- **Managed pools:** Amazon RDS Proxy is a managed option. It still sits in the network path, so measure the added hop against your baseline.

Do not choose a tool by version number from an article; check the project's current release notes and confirm the features you need exist in the version you deploy.

### Step 2: Configure transaction-level connection reuse

For a PgBouncer-style proxy, the settings that matter most for bursty workloads are the pool mode and the timeouts:

```ini
[pgbouncer]
pool_mode = transaction
server_reset_query = DISCARD ALL
server_idle_timeout = 60
query_wait_timeout = 5
```

`pool_mode = transaction` releases the backend connection at the end of each transaction, which matches short bursts. `server_reset_query = DISCARD ALL` clears session state between clients so one agent cannot see another's temporary tables or settings. `query_wait_timeout` bounds how long a client waits for a free backend before failing, which prevents unbounded queueing under burst.

The trade-off is explicit: transaction mode is faster and cheaper under bursty load, but any workload that relies on session-level state — prepared statements held open across transactions, advisory locks, `SET` applied outside a transaction — will break. Audit your queries for those patterns before switching.

### Step 3: Terminate TLS at the gateway

If the function already talks to an API gateway or load balancer, terminate TLS there and connect to the proxy over a private network path. This removes one handshake from the function's critical path. The function code then connects without client-side TLS to the proxy, relying on network isolation for confidentiality.

This is a real security decision, not a free win. It is only appropriate when the function-to-proxy leg stays inside a private network with no untrusted hops. Document the decision and the network boundary it depends on.

### Step 4: Client code

The client should open one connection per invocation and close it, letting the proxy manage reuse. Do not build a connection pool inside the function; it will be discarded when the instance is recycled and adds complexity for no benefit.

```python
import os
import pg8000

def lambda_handler(event, context):
    conn = pg8000.connect(
        host=os.environ["PG_PROXY_HOST"],
        port=int(os.environ.get("PG_PROXY_PORT", "5432")),
        database=os.environ["PG_PROXY_DB"],
        user=os.environ["PG_PROXY_USER"],
        password=os.environ["PG_PROXY_PASSWORD"],
        ssl_context=None,  # TLS terminated at the gateway on a private path
    )
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT state FROM agents WHERE id = %s",
            (event["agent_id"],),
        )
        row = cursor.fetchone()
        return {"state": row[0] if row else None}
    finally:
        conn.close()
```

The `finally` block matters. If the handler raises before closing, the connection is released only when the proxy's own timeout fires, which under a burst of failures can tie up backend slots.

### Step 5: Size the proxy and set alarms

Run at least two proxy instances in different availability zones and route to them with health checks. Instrument these signals:

- Active backend connections versus configured maximum.
- Client wait time for a backend connection (the metric that predicts queueing before it becomes visible in p99).
- Proxy CPU and memory, since transaction-mode proxies spend cycles on session reset.

Set an alarm on client wait time, not just on connection count. Connection count rising is normal under load; wait time rising means clients are queuing.

## Measuring whether any of this helped

A comparison table with invented numbers is worse than no table, because it will not match your workload. Build your own with these steps:

1. **Establish a baseline.** Record p50, p95, and p99 latency for a representative endpoint, plus monthly cost broken into compute, database, and data transfer. Keep the raw data, not just the summary.
2. **Instrument the connect phase separately.** Log the time from function start to first successful query, and the time from connection open to first row. The difference isolates connection overhead from query time.
3. **Change one thing.** Switch from the pooler to the multiplexer, or from the multiplexer to the Data API, without also changing instance sizes or query patterns.
4. **Re-measure over a full traffic cycle.** Bursty workloads have quiet periods; a one-hour sample can miss the burst that matters.
5. **Compare cost line by line.** A latency win that moves spend from the database bill to the network bill is not necessarily a win.

The metric most likely to improve is p50, because removing a handshake from every request helps the common case. p99 improvements depend on whether the old design was queueing under burst; if it was not, expect little change.

## Failure modes

**Shared session state.** The most common bug in transaction-mode pooling and multiplexing is assuming session state persists. Temporary tables, `SET` commands issued outside a transaction, advisory locks, and `LISTEN`/`NOTIFY` all break. Run `DISCARD ALL` between clients and audit your queries for these patterns.

**Long transactions blocking other clients.** In transaction mode, a client holding a transaction open occupies a backend. A slow query or an application bug that leaves a transaction open can starve every other client on that backend. Set `idle_in_transaction_session_timeout` in PostgreSQL so abandoned transactions are terminated, and set a query timeout in the proxy.

**Prepared-statement cache growth.** If every client uses a unique query string, a prepared-statement cache keyed by query text can grow without bound. Cap the cache size in the proxy configuration and monitor eviction rate; a rising eviction rate with flat throughput means the cache is thrashing.

**Connection leaks from crashed clients.** A function that exits without closing its connection leaves the proxy holding a slot until a timeout fires. Bound this with an aggressive `query_wait_timeout` and alert on the gap between active connections and expected concurrency.

**Health-check load during scale-out.** When many proxy instances start at once, health checks can briefly dominate traffic. Use an application-level health endpoint rather than a TCP connect check so the check exercises the same path as real requests.

**Audit-trail gaps.** When many logical clients share one physical connection, the database sees one backend session. If you need per-client attribution, set `application_name` per client and log it:

```sql
SET application_name = 'agent-12345';
SELECT state FROM agents WHERE id = $1;
```

Then query the server's activity view to attribute work:

```sql
SELECT pid, usename, application_name, query_start, state
FROM pg_stat_activity
WHERE application_name LIKE 'agent-%';
```

This adds a small per-query overhead and requires every client to set the field. Where a compliance regime requires one connection per authenticated user, multiplexing may be disallowed outright; confirm before designing around it.

**Cross-shard routing.** If the proxy shards by table and a single logical request touches tables on different shards, the proxy must route to more than one backend, adding latency and complicating transactions. Co-locating frequently joined tables on the same shard is a schema decision that is expensive to reverse after launch. Make it before, not after.

## When not to use a multiplexer

- **Workloads with heavy session state.** If your application relies on temporary tables or session-level settings, transaction-mode multiplexing will break it. Use session pooling and accept the cost.
- **High per-client query rates.** A single agent issuing many queries per second will monopolize its backend slot. Pooling with per-client isolation may be more appropriate.
- **Strict per-user audit requirements.** If regulations require a distinct database connection per authenticated user, multiplexing is not compatible.
- **Reads that must be current.** If the proxy routes reads to replicas, an agent may see stale data. Route reads that must be current to the primary.
- **Client libraries that assume one connection per thread.** Some drivers will not work correctly through a multiplexer. Verify before committing.

In these cases a traditional pooler is the right tool. Isolate it in its own network segment and use private endpoints so traffic does not cross a billed boundary unnecessarily.

## What to do in the next 30 minutes

Open your proxy or pooler configuration and find the client wait timeout — the setting that governs how long a client waits for a backend connection. In PgBouncer it is `query_wait_timeout`; in other proxies it has a different name. If the value is greater than 5 seconds or unset, set it to 5 seconds and redeploy. Then check whether you have a metric for client wait time. If you do not, add one. That single metric will tell you whether your proxy is absorbing bursts or queueing them, and it is the first thing to look at when latency degrades under load.
