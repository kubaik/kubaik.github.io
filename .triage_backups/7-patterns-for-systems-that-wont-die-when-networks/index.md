# 7 patterns for systems that won’t die when networks

## Why network failures break otherwise correct systems

A service can pass every unit test, deploy cleanly, and still collapse the moment a dependency becomes slow rather than unreachable. The failure mode is rarely a clean connection refused. It is a socket that accepts a request, holds it open, and returns nothing for thirty seconds. Every caller waiting on that socket occupies a connection, a worker thread, and a database session. Within minutes the healthy service is saturated by work that will never complete.

The classic trigger is a retry loop with no ceiling. A client library that retries 5xx responses with exponential backoff and no maximum attempt count will keep a failing dependency under load indefinitely, which prevents it from recovering. The retry curve looks reasonable on a whiteboard and is catastrophic in production.

This article covers seven patterns for making a service survive partial failures: network partitions, slow dependencies, database failovers, and instances that die mid-operation. For each pattern it describes what it does, where it breaks, and how to decide whether it belongs in your system. The patterns are ordered roughly by the cost of adopting them, from least to most invasive.

Two terms are used throughout:

- **At-least-once delivery** means a message may be processed more than once. Duplicates are possible, so handlers must tolerate them.
- **Idempotent operation** means applying it twice has the same effect as applying it once. It is the property that makes at-least-once delivery safe.

Most of the patterns below are ways of converting an unreliable transport into a system that is correct despite duplicates, delays, and reordering.

## How to evaluate a resilience pattern honestly

Vendor benchmarks and blog-post tables are usually measured under conditions that do not resemble a real workload. If you want numbers, generate them against your own system. Three measurements matter.

**Recovery time after a dependency failure.** Instrument the latency of each outbound call and record the time from the first failure to the point where error rates return to baseline. If you use Prometheus, a histogram on the client call duration plus a counter on failures is enough. Compare the 50th and 99th percentiles separately; a pattern that recovers quickly on average but stalls on the tail is not resilient.

**Duplicate-processing rate.** Count the number of times each logical operation is executed. If the count exceeds one under fault injection, your handler is not idempotent, and every at-least-once transport in your stack is a latent bug.

**Operational cost.** Count the lines of production code the pattern adds, the new infrastructure it requires, and the number of new failure modes it introduces. A pattern that eliminates one failure mode but adds three is usually a net loss.

A simple way to run these measurements is a fault-injection harness. A script that periodically closes connections, adds latency, or kills a replica while a load generator runs at a steady rate will surface most of the failure modes described below. The specific tooling matters less than running the harness long enough to see the second-order effects — a five-minute test will miss the cascading failures that appear after ten minutes of sustained pressure.

The rest of this article assumes you have such a harness, or are willing to build one. Without it, every pattern below is a guess.

## Pattern 1: Bounded retries with jitter and a dead-letter queue

**What it does.** Every outbound call has a maximum attempt count, an exponential backoff, and randomized jitter added to each delay. When the attempt count is exhausted, the operation is written to a dead-letter queue (DLQ) for later inspection rather than dropped or retried forever.

**Why jitter matters.** If a thousand clients all back off on the same schedule, they retry in lockstep and produce a thundering herd every time the backoff expires. Adding a random component to each delay spreads the retries out. A common approach is "full jitter": sleep for a random duration between zero and the computed backoff.

**Where it breaks.** A DLQ is not a solution; it is a place to put problems you have not solved. If nothing consumes the DLQ, it grows without bound and the failures are invisible. Alert on DLQ depth. Also beware of retrying non-idempotent operations: a retried payment charge that succeeds on the first attempt but times out on the response will be charged twice.

**How to decide.** Use bounded retries on every network call, without exception. The attempt count should be small — three or four is typical — and the total time budget should be less than the caller's own timeout. If the operation is not idempotent, either make it idempotent (see Pattern 2) or do not retry it at all.

## Pattern 2: Idempotency keys

**What it does.** The client generates a unique key for each logical operation and sends it with the request. The server stores the key alongside the result. If the same key arrives again, the server returns the stored result instead of re-executing the operation.

**Where it breaks.** The key store itself becomes a dependency. If the store is unavailable, you must decide whether to fail closed (reject the request) or fail open (process it and risk a duplicate). Failing closed is usually correct for financial operations. The bigger risk is unbounded key retention: keys must have a time-to-live, and that TTL must be longer than the maximum retry window of any client. If a client can retry a refund for five days and the key expires after one, the refund can be replayed.

**Implementation sketch.** The key store is typically a fast key-value store with atomic set-if-not-exists semantics. The sequence is: attempt to reserve the key, and if the reservation succeeds, execute and store the result; if it fails, return the stored result.

```javascript
import Fastify from 'fastify';
import Redis from 'ioredis';

const app = Fastify({ logger: true });
const redis = new Redis(process.env.REDIS_URL);

app.post('/charge', async (req, reply) => {
  const idempotencyKey = req.headers['idempotency-key'];
  if (!idempotencyKey) {
    return reply.status(400).send({ error: 'idempotency-key header required' });
  }

  const cacheKey = `idem:${idempotencyKey}`;
  // Reserve the key atomically. NX means "only set if it does not exist".
  const reserved = await redis.set(cacheKey, 'in-progress', 'EX', 86400, 'NX');
  if (!reserved) {
    const existing = await redis.get(cacheKey);
    if (existing === 'in-progress') {
      // A concurrent request holds the key. Tell the client to retry later.
      return reply.status(409).send({ error: 'request in progress' });
    }
    return reply.status(200).send(JSON.parse(existing));
  }

  try {
    const result = await processCharge(req.body);
    await redis.set(cacheKey, JSON.stringify(result), 'EX', 86400);
    return reply.status(201).send(result);
  } catch (err) {
    // Release the reservation so the client can retry.
    await redis.del(cacheKey);
    throw err;
  }
});
```

The reservation step is what distinguishes a correct implementation from a naive one. Reading the key, then writing it, leaves a window in which two concurrent requests both see a miss and both execute the operation.

**How to decide.** Any endpoint that a client may retry, and any endpoint that mutates state, should accept an idempotency key. The cost is one extra lookup per request; the benefit is that every other pattern in this article becomes safe to use.

## Pattern 3: The transactional outbox

**What it does.** Instead of writing to the database and publishing a message as two separate operations, the application writes the business row and an outbox row in a single database transaction. A separate process reads the outbox table and publishes the messages, marking each row as sent.

**Why it matters.** The naive approach — commit, then publish — has a window in which the commit succeeds and the publish fails, producing a database that is inconsistent with the message stream. The reverse order has the opposite problem. The outbox eliminates the window by making both writes atomic.

**Where it breaks.** The outbox table grows forever unless rows are deleted after publication. The publisher must be idempotent, because it may publish the same row twice if it crashes after publishing but before marking the row as sent. And the publisher is a new component that can fail independently of the application.

```java
@Entity
@Table(name = "orders_outbox")
public class OrderOutbox {
  @Id
  private String eventId;
  private String aggregateId;
  private String eventType;
  @Column(length = 4000)
  private String payload;
  private Instant createdAt;
  private Instant publishedAt; // null until the publisher confirms
}

// In the same transaction as order creation:
@Transactional
public Order createOrder(OrderRequest request) {
  Order order = orderRepository.save(Order.from(request));
  outboxRepository.save(new OrderOutbox(
      UUID.randomUUID().toString(),
      order.getId(),
      "OrderCreated",
      objectMapper.writeValueAsString(order),
      Instant.now(),
      null
  ));
  return order;
}
```

The publisher then polls for rows where `publishedAt IS NULL`, publishes each, and updates `publishedAt`. Because the publisher may crash between publishing and updating, consumers must deduplicate using the `eventId`.

**How to decide.** Use the outbox whenever a database write must reliably produce a message. If your system already tolerates lost messages, you do not need it. If it does not, the outbox is usually cheaper than the alternative of distributed transactions.

## Pattern 4: Saga orchestration with compensating actions

**What it does.** A long-running business process is broken into a sequence of local transactions. A central orchestrator invokes each step and, if a later step fails, invokes a compensating action for each completed step in reverse order.

**Why it matters.** Distributed transactions across services are impractical at scale, and two-phase commit has a well-known failure mode: if the coordinator crashes after the prepare phase, participants can be left holding locks indefinitely. The saga replaces atomicity with a sequence of reversible steps.

**Where it breaks.** Compensating actions are not rollbacks. A refund is not the inverse of a charge; it is a separate operation that can itself fail, and it may be visible to the customer. The orchestrator must be durable, or a crash mid-saga leaves the system in an indeterminate state. Debugging a saga that failed at step three of five requires reconstructing the state of five services from their logs.

A specific hazard is a compensating action that is not idempotent. If the orchestrator retries a compensation, it may apply it twice. Every compensation should carry an idempotency key, exactly like any other mutating operation.

```go
func CheckoutWorkflow(ctx workflow.Context, order Order) (string, error) {
  ao := workflow.ActivityOptions{
    ScheduleToCloseTimeout: 10 * time.Minute,
    RetryPolicy: &temporal.RetryPolicy{
      MaximumAttempts: 3,
    },
  }
  ctx = workflow.WithActivityOptions(ctx, ao)

  var paymentResult string
  err := workflow.ExecuteActivity(ctx, ChargePayment, order.Payment).Get(ctx, &paymentResult)
  if err != nil {
    return "", err
  }

  var inventoryResult string
  err = workflow.ExecuteActivity(ctx, ReserveInventory, order.Items).Get(ctx, &inventoryResult)
  if err != nil {
    // Compensate: refund payment. This call must itself be idempotent.
    compensateCtx := workflow.WithActivityOptions(ctx, ao)
    _ = workflow.ExecuteActivity(compensateCtx, RefundPayment, order.Payment).Get(compensateCtx, nil)
    return "", err
  }

  return "order-confirmed", nil
}
```

**How to decide.** Use a saga when a business process spans more than two or three services and each step has a meaningful inverse. For a two-step process, a simpler pattern — usually the outbox plus an idempotent consumer — is easier to operate. Orchestration is generally easier to debug than choreography, because the state of the process lives in one place.

## Pattern 5: Optimistic concurrency control

**What it does.** Each row carries a version number or timestamp. An update includes the version the client read. If the version has changed, the update affects zero rows and the client is told to retry.

**Why it matters.** Two clients editing the same record concurrently will otherwise overwrite each other, producing a lost update. Optimistic control avoids holding locks during the read, so it scales with read traffic rather than serializing on it.

**Where it breaks.** The application must handle conflicts. A retry loop with a bounded attempt count is required; without it, users see errors they do not understand. Under very high contention on a single row, optimistic control can livelock, with every attempt conflicting. The usual remedy is to serialize writes to a single hot row through a queue rather than through the database.

```python
from sqlalchemy import Column, Integer, String, DateTime, func, update
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    version = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, onupdate=func.now())

def update_product(session, product_id, expected_version, new_name):
    stmt = (
        update(Product)
        .where(Product.id == product_id, Product.version == expected_version)
        .values(name=new_name, version=expected_version + 1)
        .returning(Product)
    )
    result = session.execute(stmt)
    session.commit()
    row = result.scalar_one_or_none()
    if row is None:
        raise VersionConflictError("Product was updated by another writer")
    return row
```

The critical detail is that the version check and the increment happen in the same statement. A separate read-then-write reintroduces the race the pattern exists to prevent.

**How to decide.** Use optimistic control for high-read, low-contention write workloads: catalogs, profiles, counters. For rows that are written constantly by many clients, prefer a queue or a single-writer service.

## Pattern 6: Event sourcing

**What it does.** State is stored as an ordered, append-only log of events rather than as a mutable row. The current state is derived by replaying events. Read models are built by projecting the event log into query-friendly shapes.

**Why it matters.** The event log is an audit trail by construction, and a read model can be rebuilt from scratch if it becomes corrupt. Rebuilding a projection is usually far faster than restoring from a backup, because it only requires replaying the log.

**Where it breaks.** Schema evolution is the hard part. Events are immutable, so a change to their shape means every consumer must handle both old and new versions. Projections accumulate assumptions about event order and shape, and those assumptions break silently when a new event type appears. The operational cost of an event-sourced system is significantly higher than that of a CRUD system, and it is rarely justified by audit requirements alone.

**How to decide.** Adopt event sourcing when the history of changes is itself a product requirement — regulatory audit, temporal queries, or the ability to rebuild arbitrary read models. Do not adopt it merely because it sounds more rigorous than storing rows.

## Pattern 7: Compensating transactions driven by change data capture

**What it does.** The database's write-ahead log is read by a change-data-capture (CDC) consumer, which reacts to specific changes by triggering compensating actions. For example, a payment row moving to a `failed` state triggers a refund row.

**Why it matters.** The compensating action is decoupled from the original request, so a slow refund does not block the payment path. The WAL is the source of truth, so no event can be missed as long as the consumer's offset is tracked.

**Where it breaks.** CDC introduces lag between the change and the reaction. Under load, that lag can grow to seconds or more. Any business process that assumes a refund happens immediately after a failure will be wrong. CDC also requires elevated database privileges and adds a component whose failure is silent unless offset lag is monitored.

```sql
-- Publisher configuration (requires a restart to take effect)
ALTER SYSTEM SET wal_level = logical;
-- Then restart PostgreSQL.

CREATE PUBLICATION payment_events FOR TABLE payments;
```

```sql
-- Subscriber
CREATE SUBSCRIPTION refund_sub
CONNECTION 'host=db.internal port=5432 dbname=payments user=repl password=...'
PUBLICATION payment_events;
```

**How to decide.** Use CDC-driven compensation when side effects must be reversed automatically and the reversal can tolerate a delay. If the reversal must be immediate, it belongs in the request path, not in a CDC consumer.

## A comparison of the seven patterns

The table below compares the patterns on the dimensions that determine adoption cost. The values are qualitative; the point is to make the trade-offs comparable, not to provide a ranking.

| Pattern | Adds infrastructure | Tolerates duplicates | Typical failure mode | Best fit |
|---|---|---|---|---|
| Bounded retries + DLQ | A queue for the DLQ | Only if operation is idempotent | DLQ grows unnoticed | Every network call |
| Idempotency keys | A fast key-value store | Yes, by design | Key TTL shorter than retry window | Any retryable mutation |
| Transactional outbox | A publisher process | Yes, via event IDs | Outbox table grows unbounded | Database write must emit a message |
| Saga orchestration | A durable workflow engine | Only if steps are idempotent | Non-idempotent compensation | Multi-service business process |
| Optimistic concurrency | None | No | Livelock on a hot row | Low-contention shared records |
| Event sourcing | An event store | Yes, via event IDs | Schema evolution breaks projections | Audit or temporal queries required |
| CDC compensation | A CDC consumer | Yes, via event IDs | Offset lag grows silently | Automatic reversal of side effects |

A useful rule of thumb: the first three patterns are almost always worth adopting, the middle two depend on your workload, and the last two should be adopted only when a specific requirement demands them.

## Failure modes to design against explicitly

**Retry storms.** A client that retries without a bound will keep a failing dependency down. Bound every retry and add jitter.

**Thundering herds.** Many clients that back off on the same schedule will retry simultaneously. Jitter is the fix.

**Duplicate side effects.** At-least-once delivery is the norm in every practical transport. Every consumer must be idempotent, and every mutating endpoint should accept an idempotency key.

**Unbounded queues.** A DLQ, an outbox table, or a poison queue that is never drained will eventually exhaust its storage. Alert on depth, not just on error rate.

**Silent offset lag.** A CDC consumer that falls behind produces no errors, only stale data. Monitor the lag explicitly.

**Non-idempotent compensation.** A refund that is applied twice is worse than a refund that is applied late. Compensations need idempotency keys like any other mutation.

## A decision checklist

Work through these questions in order. The first "yes" usually determines the pattern.

1. Does the operation mutate state and can the client retry it? If yes, implement idempotency keys first. Nothing else is safe without them.
2. Does a database write need to reliably produce a message? If yes, use the transactional outbox.
3. Does the business process span more than two services, with meaningful inverses at each step? If yes, use saga orchestration.
4. Are concurrent writers likely to edit the same row? If yes, add optimistic concurrency control, and route writes to hot rows through a queue.
5. Is the history of changes a product requirement? If yes, consider event sourcing — and budget for the operational cost.
6. Must side effects be reversed automatically, and can the reversal tolerate a delay? If yes, CDC-driven compensation is viable.

Notice that the first question is not about which pattern is most sophisticated. It is about the property that makes every other pattern safe. Idempotency is the foundation; the rest are ways of coping with the delays and duplicates that a network partition produces.

## FAQ

**Does an idempotency key need to be stored forever?**
No. It needs to be stored longer than the maximum time a client might retry the same operation. If clients retry for up to 24 hours, a 48-hour TTL is reasonable. The risk of a too-short TTL is a duplicated side effect; the risk of a too-long TTL is unbounded storage growth.

**Can a saga be replaced with the outbox pattern?**
Sometimes. If the process has only two steps and the second step is a message publication, the outbox plus an idempotent consumer is simpler than a saga. Sagas become worthwhile when there are three or more steps and each has a meaningful inverse.

**Is optimistic locking safe under high contention?**
It is safe in the sense that it will not lose updates, but it may livelock: every attempt conflicts and no writer makes progress. For a single hot row, serialize writes through a queue instead.

**How much CDC lag is acceptable?**
That depends entirely on the business process. A refund that arrives ten seconds late is usually acceptable; a refund that arrives ten minutes late may not be. Measure the lag under peak load and decide against a stated requirement, not against a benchmark.

**Do these patterns require a specific database or message broker?**
No. They are transport-agnostic. The outbox requires a database with transactions and a way to poll or tail a table; idempotency requires a key-value store with atomic set-if-not-exists; CDC requires a database that exposes its change log. The specific products are interchangeable.

## What to do in the next 30 minutes

Open the source of the busiest mutating endpoint you own. Search for every outbound network call it makes and check whether each one has a bounded retry count and jitter. If any call retries without a limit, add the limit now — that single change will prevent the most common cascading failure, and it takes minutes to implement.
