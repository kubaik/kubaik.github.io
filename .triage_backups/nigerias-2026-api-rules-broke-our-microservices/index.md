# Designing Fintech Backends Around Audit Deadlines

## Why the standard microservices playbook can conflict with audit rules

Most fintech architecture guidance says the same thing: split the backend into microservices, give each service its own database, and expose REST or GraphQL APIs. The stated benefits are independent deploys and horizontal scalability. That advice is reasonable in general, but it becomes questionable when a regulator imposes a hard, wall-clock deadline on how quickly every financial transaction must be reconstructable from logs.

Suppose a rule states that every financial transaction must be fully traceable across every component within 30 seconds of the initiating request. Now consider a transfer that a wallet routes to a bank through three services: authentication, ledger, and settlement. Each service owns its own database and its own transaction log. A trace ID is generated at the edge and propagated downstream. If any hop is delayed — a broker lag, a replication lag, a retry against a slow bank API — the audit reader can observe a window in which the transaction ID exists upstream but not yet downstream. Depending on how the regulator defines "traceable", that window can be recorded as a gap.

The important nuance is that the deadline is not a performance target you can tune away. It is a property of the whole path, including components you do not control: card switches, bank APIs, and third-party processors. You cannot shard your way out of a clock.

## A concrete failure mode: fan-out plus sampling

Consider a small transfer. The authentication service validates a token in roughly 40 ms. The ledger service updates a balance in roughly 120 ms. The settlement service calls a bank over HTTP. That call is the problem: it may respond in 300 ms on a good day and time out after several seconds on a bad one. If the client retries three times, the elapsed time before the transaction is recorded as settled can reach double-digit seconds.

Now add an audit reader that samples or polls on an interval — say every five seconds. At a poll boundary, the reader may see the transaction ID in the auth log and the ledger log but not yet in the settlement log. Whether that counts as a violation depends on the rule's wording, but in practice auditors often treat "not yet visible" and "missing" as the same thing unless the system can prove otherwise.

A second, subtler failure mode is replication lag. If the audit reader queries a logical replica rather than the primary, any lag between commit on the primary and visibility on the replica is a window in which a committed transaction is invisible to the auditor. At moderate load, that lag may be a few hundred milliseconds. At peak load, it can grow by an order of magnitude. The absolute number may look small next to a 30-second budget, but it is not the only source of delay in the path, and the delays compound.

A third failure mode is broker configuration. If a producer is configured with `acks=1`, a leader election can delay acknowledgement while a new leader is chosen. During that window, the message exists in the producer's buffer but not in any log the auditor can read. The fix is not to make the broker faster; it is to stop treating the broker as the audit source of truth.

## A different mental model: the transaction boundary

Instead of splitting services by domain (auth, ledger, settlement, notification), split by transaction boundary. A transaction boundary is any unit of work that must complete atomically with respect to a regulatory deadline. For a financial operation with a 30-second traceability requirement, the boundary typically includes the ledger write and any synchronous external call that must succeed or fail together with it.

The practical implication is that the ledger and the settlement call should live in the same deploy unit, sharing one database connection and one transactional outbox. Authentication and notification can usually live outside the boundary, because they do not need to commit atomically with the money movement.

This is not a call for a monolithic codebase. It is a call for a modular monolith with explicit transaction boundaries. You can keep separate modules, separate packages, and even separate containers, provided the boundary components share a single durable log that the audit reader can consume.

The mechanism that makes this work is the transactional outbox:

1. Begin a database transaction.
2. Write the business state change (for example, the balance update).
3. Write a row into an `outbox` table in the same transaction, containing the event payload and the trace ID.
4. Commit. At this point the state change and the intent to publish are durable together.
5. A separate publisher process reads the `outbox` table and emits to the message broker. If publishing fails, the row remains and is retried.

Because the business change and the outbox row commit atomically, there is no window in which money moved but no event exists. The audit reader can consume the outbox table or the database's write-ahead log directly, rather than waiting for the broker.

## Reading the write-ahead log instead of the broker

Every relational database that supports crash recovery writes a write-ahead log (WAL) before applying changes to data files. In PostgreSQL, that log is the authoritative record of what committed. Logical decoding, whether via built-in logical replication slots or a change-data-capture tool, can stream committed changes to a consumer.

The architectural point is simple: the WAL is the source of truth for "did this transaction commit", and the broker is a transport for downstream consumers. If your audit reader consumes the broker, it inherits the broker's latency, its reordering behaviour, and its failure modes. If it consumes the WAL or a logical replica fed from the WAL, it inherits only database replication latency, which you can measure and bound.

To measure the relevant lag, instrument two timestamps: the commit timestamp recorded by the database, and the timestamp at which the audit reader observes the row. The difference is your audit visibility latency. Track it as a histogram, not an average, because the tail is what breaks compliance. A useful command-level check is to compare `pg_current_wal_lsn()` on the primary with the last received LSN reported by the replica's WAL receiver, and to alert when the byte gap exceeds a threshold you have tied to your deadline.

## What this looks like in code

A minimal outbox write inside a single transaction, using the Node `pg` driver:

```js
// db.js
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
});

module.exports = { pool };
```

```js
// ledger.js
const { pool } = require('./db');

async function applyTransfer({ traceId, fromAccount, toAccount, amountMinor }) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    await client.query(
      'UPDATE accounts SET balance_minor = balance_minor - $1 WHERE id = $2',
      [amountMinor, fromAccount]
    );
    await client.query(
      'UPDATE accounts SET balance_minor = balance_minor + $1 WHERE id = $2',
      [amountMinor, toAccount]
    );

    await client.query(
      `INSERT INTO outbox (trace_id, event_type, payload)
       VALUES ($1, $2, $3)`,
      [traceId, 'transfer.applied', JSON.stringify({ fromAccount, toAccount, amountMinor })]
    );

    await client.query('COMMIT');
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}

module.exports = { applyTransfer };
```

The publisher polls the outbox and marks rows as sent. It must not delete rows until the audit reader has confirmed consumption, or you lose the ability to replay.

```js
// publisher.js
const { pool } = require('./db');
const { producer } = require('./kafka');

async function publishBatch() {
  const client = await pool.connect();
  try {
    const { rows } = await client.query(
      `SELECT id, trace_id, event_type, payload
         FROM outbox
        WHERE published_at IS NULL
        ORDER BY id
        LIMIT 100
        FOR UPDATE SKIP LOCKED`
    );

    for (const row of rows) {
      await producer.send({
        topic: 'ledger.events',
        messages: [{
          key: row.trace_id,
          value: JSON.stringify({
            traceId: row.trace_id,
            type: row.event_type,
            payload: row.payload,
          }),
        }],
      });

      await client.query(
        'UPDATE outbox SET published_at = now() WHERE id = $1',
        [row.id]
      );
    }
  } finally {
    client.release();
  }
}

module.exports = { publishBatch };
```

The publisher is idempotent from the broker's perspective only if consumers deduplicate on `trace_id`. Do not assume exactly-once delivery; assume at-least-once and make consumers idempotent.

## Choosing between a single deploy unit and a sharded monolith

A common objection is that a single deploy unit cannot scale. That is true for a single process on a single machine, but the transaction boundary does not require a single process — it requires a single consistency domain. You can shard the ledger by account or user ID, provided each shard is an independent deploy unit with its own database, its own outbox, and its own WAL. Each shard is internally a small monolith. This is sometimes called a sharded monolith, and it scales horizontally without breaking the atomicity of each shard.

The trade-off is cross-shard operations. A transfer between accounts on different shards is no longer a single transaction; it becomes a two-phase operation or a saga with compensating actions. That reintroduces the audit-gap problem at the shard boundary. For most retail payment volumes, a single primary with synchronous replication to a standby handles the load, and the operational simplicity is worth more than the theoretical ceiling. Shard only when measurements show the primary is the bottleneck, and design the cross-shard path explicitly rather than discovering it later.

## Caching without serving stale balances

Caching read-heavy data such as user profiles is uncontroversial. Caching balances is not, because a stale balance read by an authentication or authorisation service can lead to an incorrect decision. If you cache balances, you need a reliable invalidation channel. A common approach is to publish an invalidation event to a durable stream whenever the ledger commits, and have cache consumers subscribe and evict. The write path pays a small latency cost; the read path avoids stale data.

The failure mode to watch for is silent invalidation loss. If the invalidation channel is best-effort, a dropped message leaves a stale entry indefinitely. Use a durable, replayable channel and monitor consumer lag. A short time-to-live on cached balances is a cheap backstop, but it is not a substitute for correct invalidation.

## Measuring whether your design actually meets the deadline

Do not accept a claim about audit latency without a measurement. What to instrument:

- Commit-to-visible latency: the difference between the database commit timestamp and the timestamp at which the audit reader can query the row. Record it as a histogram with p50, p95, p99, and max.
- Outbox backlog: the count and age of unpublished rows. A growing backlog means the publisher is falling behind, which is a leading indicator of audit gaps.
- Replication lag: the byte gap between primary and replica WAL positions, converted to a time estimate using recent throughput.
- External call duration: histogram of every third-party call, with explicit timeouts. A call that can hang indefinitely will eventually break the deadline.
- Trace completeness: for each transaction, the set of required hops that have been observed. Track the fraction of transactions with all hops observed within the deadline, as a rolling window.

To run a controlled comparison, generate a fixed workload with a load tool, apply a known perturbation (for example, a brief broker pause or a replica pause), and compare the trace-completeness metric before and after. The comparison is only meaningful if the workload, the perturbation, and the measurement window are identical between runs.

## Decision checklist

Use this when deciding whether a component belongs inside the transaction boundary:

1. Does the component participate in a state change that a regulator or auditor must reconstruct within a fixed deadline? If yes, keep it inside the boundary.
2. Does the component make a synchronous external call that must succeed or fail together with the ledger write? If yes, keep it inside the boundary and wrap the call in a timeout and circuit breaker.
3. Is the component read-only and tolerant of eventual consistency? If yes, it can live outside the boundary, reading from a replica.
4. Is the component non-financial (notifications, marketing, analytics)? If yes, it can be split freely.
5. Can the component be sharded along a key that keeps each shard's operations self-contained? If yes, sharding is safe; if not, cross-shard coordination reintroduces the gap.
6. Have you measured commit-to-visible latency under peak load, including the tail? If not, you do not yet know whether the design meets the deadline.

## Common objections, addressed

**"Microservices let us deploy faster."** They do, for services outside the boundary. For services inside the boundary, a change requires re-validating the audit path, which costs time regardless of how the code is packaged. The speed advantage is real but applies unevenly.

**"A single primary cannot handle our volume."** Measure before assuming. A single primary with synchronous replication handles substantial throughput for ledger-style workloads, which are typically small, indexed updates rather than large scans. If measurements show otherwise, shard along a self-contained key.

**"Event sourcing solves this."** Event sourcing makes the event log the source of truth, which is conceptually aligned with auditability, but it adds a durable write before the business transaction commits. That extra write is another source of latency on the critical path. The outbox pattern achieves similar auditability with one durable write instead of two.

**"What if the ledger process crashes mid-transaction?"** With a single database transaction, a crash before commit means the transaction did not happen; a crash after commit means it did, and the WAL contains it. Synchronous replication to a standby ensures the committed record survives the loss of the primary. Configure a shutdown hook that stops accepting new work and lets in-flight transactions finish or roll back, and verify the behaviour by killing the process under load in a test environment.

## A worked example with arithmetic

Here is an illustrative calculation, using round numbers chosen for clarity, not measured data.

Suppose the deadline is 30 seconds from the initiating request. Break the path into stages with assumed worst-case durations:

- Edge authentication and request validation: 0.1 s
- Ledger transaction commit: 0.05 s
- Synchronous replication acknowledgement to standby: 0.02 s
- Audit reader observes the committed row: 0.2 s
- Settlement call to the bank, with a 5 s timeout and one retry: up to 10 s

The total worst case is 0.1 + 0.05 + 0.02 + 0.2 + 10 = 10.37 seconds, comfortably within 30 seconds. Now suppose the bank call has no timeout and occasionally hangs for 25 seconds. The total becomes 25.37 seconds, which is still within 30 but leaves almost no margin for a broker pause or a replication spike. If the bank call hangs for 30 seconds, the deadline is missed regardless of how fast everything else is.

The lesson from the arithmetic is that the largest term dominates. Optimising the 0.05-second ledger commit while leaving an unbounded external call is wasted effort. Bound the largest term first.

## Action for the next 30 minutes

Open the repository for your highest-traffic financial path and list every synchronous call made between the incoming request and the database commit. For each call, note whether it has an explicit timeout. Any call without a timeout is an unbounded term in your worst-case latency budget. Add a timeout to the longest one now, and record the change so you can measure the effect on your commit-to-visible latency histogram at the next load test.
