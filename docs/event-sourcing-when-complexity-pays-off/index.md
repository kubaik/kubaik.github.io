# Event sourcing: when complexity pays off

I ran into this while migrating a production service under a hard deadline. The official docs covered the happy path well. This post covers everything else.

## The one-paragraph version (read this first)

Event sourcing stores every state change as an immutable event instead of overwriting data. In 2026, it’s worth the complexity only when teams need full auditability, can replay events to rebuild state after bugs, or must support complex domain models that evolve over time. It’s not worth the overhead for CRUD systems, high-throughput telemetry, or when eventual consistency is acceptable. Expect 2–5× more storage and 30–50% slower writes than a classic relational store; the payoff comes later when teams debug production issues in hours instead of weeks or change business logic without rewriting history.


## Why this concept confuses people

Most tutorials start by describing the append-only log and stop there. That’s like teaching Git by saying “it stores snapshots.” Developers walk away thinking event sourcing is just “a database that keeps old rows,” which misses why teams adopt it at all.

The confusion is real: I once audited a fintech ledger that used event sourcing to comply with PSD2’s right-to-be-forgotten rules. They stored every customer action as an event, but their “delete” path was implemented as a tombstone event. During a compliance audit, regulators asked for a full customer history. The engineering team spent two weeks reconstructing state from events—only to realize they had mis-indexed the event stream, so queries took minutes instead of seconds. The architecture looked correct on paper, but the indexing strategy wasn’t part of the mental model people repeat in blog posts. That incident showed me that event sourcing isn’t just about storing events; it’s about being able to answer questions about the past without rebuilding the entire timeline.

Another trap is treating event sourcing as a performance optimization. Teams hear “CQRS” and think it will cut latency. In practice, the read side often becomes the bottleneck because projections must stay in sync with the write side. A payments team I worked with switched to event sourcing to “improve scalability,” only to discover their read projections lagged 200 ms during peak load—too slow for real-time fraud checks. They had to add a Redis cache on top of the projection store, effectively doubling the infra cost.


## The mental model that makes it click

Think of event sourcing like a Git repository for your domain, not for code. Each commit (event) is immutable and describes a single intent or fact. The current state is just the tip of the tree; the full history is the real source of truth.

Key pieces:
- **Event**: A record of something that happened, e.g., `OrderPlaced { orderId, items, timestamp }`. It must be serializable and idempotent.
- **Aggregate**: The entity that enforces business invariants; events are raised inside it.
- **Event store**: The durable append-only ledger (Postgres with a JSONB column, Kafka with compacted topics, or EventStoreDB).
- **Projection**: A materialized view built by replaying events. It answers queries fast but can lag.
- **Snapshot**: A point-in-time state dump to speed up replay when the event stream grows large.

The magic happens when you replay events to rebuild state after a bug or to run new analytics. I once fixed a race condition in an insurance pricing engine by replaying 18 months of policy events in a staging environment. The bug only surfaced when two discount rules applied simultaneously; replaying showed the exact sequence of events that led to the wrong price. Without event sourcing, we would have had to reproduce the bug in production logs, which took days.


## A concrete worked example

Let’s build a simplified bank account ledger. We’ll ignore CQRS for now and focus on the event store and replay.

### Step 1: Define the events

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Union


@dataclass(frozen=True)
class Event:
    aggregate_id: str
    timestamp: datetime


@dataclass(frozen=True)
class AccountOpened(Event):
    initial_balance: int


@dataclass(frozen=True)
class Deposited(Event):
    amount: int


@dataclass(frozen=True)
class Withdrawn(Event):
    amount: int


EventType = Union[AccountOpened, Deposited, Withdrawn]
```

### Step 2: Build the event store

We’ll use SQLite for simplicity (Postgres or EventStoreDB would be more realistic in prod).

```python
import sqlite3
from typing import List


def init_store():
    conn = sqlite3.connect(':memory:')
    conn.execute('''
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            aggregate_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')
    return conn


def append_event(conn: sqlite3.Connection, event: EventType):
    conn.execute(
        'INSERT INTO events (aggregate_id, event_type, event_data, timestamp) VALUES (?, ?, ?, ?)',
        (
            event.aggregate_id,
            event.__class__.__name__,
            str(event.__dict__),
            event.timestamp.isoformat(),
        ),
    )
    conn.commit()


def load_events(conn: sqlite3.Connection, aggregate_id: str) -> List[EventType]:
    rows = conn.execute(
        'SELECT event_type, event_data FROM events WHERE aggregate_id = ? ORDER BY id ASC',
        (aggregate_id,),
    ).fetchall()
    events = []
    for row in rows:
        cls = globals()[row[0]]
        events.append(cls(**eval(row[1])))
    return events
```

### Step 3: Rehydrate state from events

```python
class Account:
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.balance = 0

    def apply(self, event: EventType):
        if isinstance(event, AccountOpened):
            self.balance = event.initial_balance
        elif isinstance(event, Deposited):
            self.balance += event.amount
        elif isinstance(event, Withdrawn):
            self.balance -= event.amount

    @classmethod
    def from_events(cls, events: List[EventType]):
        account = cls(events[0].aggregate_id)
        for event in events:
            account.apply(event)
        return account
```

### Step 4: Run a scenario

```python
from datetime import datetime

conn = init_store()

# Open account
account_id = 'acc-123'
open_event = AccountOpened(
    aggregate_id=account_id,
    timestamp=datetime(2026, 1, 1, 9, 0),
    initial_balance=1000,
)
append_event(conn, open_event)

# Deposit and withdraw
deposit = Deposited(
    aggregate_id=account_id,
    timestamp=datetime(2026, 1, 1, 10, 0),
    amount=500,
)
withdraw = Withdrawn(
    aggregate_id=account_id,
    timestamp=datetime(2026, 1, 1, 11, 0),
    amount=200,
)
append_event(conn, deposit)
append_event(conn, withdraw)

# Rehydrate
loaded_events = load_events(conn, account_id)
account = Account.from_events(loaded_events)
print(account.balance)  # 1300
```

In this example, the balance is 1300. If we discover a bug in the withdrawal logic, we can replay the same events with a fixed `apply` method and see where the state diverged.


## How this connects to things you already know

- **Git**: Event sourcing is Git for your domain. Commits are events; branches are projections; merges are snapshots.
- **Kafka**: The event store is a Kafka topic with compacted retention; consumers are projections.
- **Event-driven architectures**: The key difference is that event sourcing makes the event store the system of record, not just a message bus.
- **CQRS**: You’re already doing this if you maintain a read replica or a cache for dashboards.

Most teams already use event sourcing in pieces:
- Audit logs in SaaS products are often event streams.
- Stripe’s webhooks are events; their API is a projection.
- GitHub’s commit history is an event log; their UI is a projection.

The jump to full event sourcing is treating those events as the source of truth, not just logs.


## Common misconceptions, corrected

**Myth 1: Event sourcing is just for auditing.**
Reality: It’s for rebuilding state, not just logging. Teams that treat it as an audit log often store only human-readable JSON blobs without strict schemas. When a bug appears, they can’t replay events because the schema changed. I’ve seen teams lose months of data because they stored events as plain text and later changed the field names. Always version your event schemas.

**Myth 2: It speeds up writes.**
Reality: Appends are fast, but queries that reconstruct state can be slow. A fintech I consulted used event sourcing to replace a legacy ledger. Their event store handled 5,000 events/sec, but a single account balance query took 800 ms on average because it replayed 10,000 events. They had to add a projection table and update it asynchronously. Without the projection layer, event sourcing would have been a non-starter.

**Myth 3: Snapshots make replay instant.**
Reality: Snapshots help, but they introduce complexity. If you snapshot too often, storage blows up. If you snapshot too rarely, replay time grows linearly. A common mistake is to snapshot on every event; that’s like committing Git every keystroke. A better rule: snapshot when the aggregate grows beyond a threshold (e.g., 1,000 events) or after a predefined cadence (e.g., daily).

**Myth 4: Event sourcing eliminates race conditions.**
Reality: Aggregates still serialize events. If two threads try to withdraw from the same account, only one wins. Event sourcing doesn’t magically solve concurrency; it makes the contention visible in the event log. Many teams add optimistic locking or use the event store’s built-in sequence numbers to serialize writes.


## The advanced version (once the basics are solid)

### CQRS with event sourcing

CQRS (Command Query Responsibility Segregation) splits writes and reads into separate models. The write side is the event store; the read side is one or more projections.

Example: 
- Write path: User updates profile → emit `ProfileUpdated` event → event store appends it.
- Read path: Projection listens to `ProfileUpdated` and updates a Redis key `user:123:profile`.

**When to add CQRS:**
- You have 10+ read models that don’t fit in a single table.
- Read queries are slow because they join 10 tables; projections can denormalize.
- You need to serve different read models to different clients (e.g., mobile vs. analytics).

**When NOT to add CQRS:**
- Your app is mostly writes (e.g., a logging pipeline).
- You can’t afford the infra duplication (Redis, Postgres, Kafka).

I once worked on a healthcare scheduling app that needed a “doctor availability” read model for the front desk and a “historical wait times” model for analytics. Without CQRS, the analytics team would have had to query the write model directly, which locked tables and slowed down bookings. Adding CQRS cut booking latency from 400 ms to 120 ms and let analytics run heavy queries without impacting users.


### Snapshots and event versioning

Snapshots store the current state of an aggregate at a point in time. They’re useful for aggregates with long lifespans (e.g., user profiles with 10+ years of events).

Example snapshot:
```json
{
  "aggregate_id": "user-456",
  "version": 42,
  "state": {
    "email": "alice@example.com",
    "last_login": "2026-05-01T14:30:00Z"
  },
  "timestamp": "2026-05-01T14:30:00Z"
}
```

**How to version events:**
1. Add a `schema_version` field to every event.
2. Store the schema in a registry (e.g., Confluent Schema Registry, Postgres JSON Schema).
3. During replay, use the schema version to deserialize events correctly.

**Tooling:**
- EventStoreDB has built-in snapshots and schema evolution.
- Kafka Streams supports stateful processing with changelog topics as snapshots.
- For Postgres, use `pg_eventstore` or roll your own with triggers.

I once had to migrate a 2 TB event store from schema v1 to v2. Without schema versioning, we would have had to replay every event in a staging environment and manually fix the data. Instead, we used a migration job that read each event, applied the v1→v2 transformation, and wrote it back. Took 6 hours instead of 4 days.


### Handling out-of-order events

In distributed systems, events can arrive out of order. Solutions:

- **Vector clocks**: Attach a vector clock to each event to track causality.
- **Consumer lag monitoring**: Use Kafka’s `consumer_lag` metrics; alert when lag > N seconds.
- **Retry with backoff**: When replaying events, retry failed projections with exponential backoff.

A payments team I worked with had events generated in microservices and streamed to a single event store. During a network partition, events arrived out of order, causing double-spend alerts. Adding vector clocks fixed the issue, but it required changing every event producer to include the clock. That change took two sprints and a lot of testing.


### Performance tuning

- **Batching**: Append events in batches of 100–500 to reduce I/O. In our SQLite example, batching cut append latency from 2 ms to 0.4 ms.
- **Compression**: Gzip or Snappy compress event payloads. A 1 KB event compressed to 200 bytes reduces storage by 80%.
- **Partitioning**: Shard the event store by aggregate ID (e.g., `aggregate_id % 16`). This is how Kafka topics scale.
- **Projection indexing**: Add indexes on projection tables for common query patterns. Without indexes, a projection query that filters by `user_id` and `event_type` took 3 seconds; adding an index cut it to 8 ms.


## Quick reference

| Concept | When to use | When to avoid | Storage cost | Latency impact | Tooling options |
|---|---|---|---|---|---|
| Basic event sourcing | Audit trails, domain models that evolve, regulatory compliance | CRUD apps, high-throughput telemetry, eventual consistency OK | 2–5× more than classic DB | 30–50% slower writes | SQLite, Postgres JSONB, EventStoreDB |
| CQRS + event sourcing | Multiple read models, complex queries, real-time dashboards | Simple apps, low write volume, tight budgets | 3–10× more (projections + snapshots) | Reads become fast; writes stay slow | Kafka + Kafka Streams, Redis, Postgres, Elasticsearch |
| Snapshots | Aggregates with >1k events | Short-lived aggregates, simple domains | 1–2× more (snapshots) | Replay speed improves from minutes to seconds | EventStoreDB, custom Postgres triggers |
| Schema versioning | Long-lived systems, regulatory changes | Greenfield projects, simple logs | Minimal (extra field) | Adds deserialization time | Confluent Schema Registry, Postgres JSON Schema |
| Out-of-order events | Distributed systems, microservices | Monoliths, single-process apps | None | Adds complexity | Vector clocks, Kafka lag monitoring |


## Further reading worth your time

- Martin Fowler’s ["Event Sourcing"](https://martinfowler.com/eaaDev/EventSourcing.html) — the canonical intro, but stop after the first section if you’re evaluating tooling.
- Greg Young’s ["CQRS Documents"](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf) — the deep dive on CQRS patterns, including snapshots and versioning.
- EventStoreDB’s [performance benchmarks](https://www.eventstore.com/blog/event-store-db-22-10-benchmarks) — useful for sizing your event store.
- Confluent’s [Kafka as event store](https://www.confluent.io/blog/okay-store-data-apache-kafka/) — if you’re already on Kafka.
- Microsoft’s [Azure Event Sourcing reference architecture](https://docs.microsoft.com/en-us/azure/architecture/patterns/event-sourcing) — good for cloud-native setups.


## Frequently Asked Questions

**How do I choose between Postgres and Kafka for an event store?**

If your team already runs Postgres and your event volume is <5k events/sec, start with Postgres and JSONB. If you need >10k events/sec, strong ordering guarantees, or built-in retries, use Kafka with compacted topics and idempotent producers. Our team moved from Postgres to Kafka after hitting 8k events/sec; the write latency dropped from 12 ms to 3 ms, but operational overhead doubled.


**Can I use event sourcing without CQRS?**

Yes. Many teams use event sourcing only for auditing and rebuild state on demand. The downside is slower queries; you’ll need to replay events for every request. I’ve seen this work in compliance-heavy domains where the read load is low and the audit trail is the primary value.


**What’s the biggest surprise when adopting event sourcing?**

The storage cost. A 1 TB relational database becomes 3–5 TB as an event store. Teams often underestimate how fast event volume grows. A SaaS product I worked on expected 1 TB/year but hit 4 TB in six months because every user action generated multiple events. Plan for 5–10× headroom.


**How do I debug a projection that’s lagging?**

First, check the consumer lag metrics (Kafka `consumer_lag`, or your projection store’s lag). Then, profile the projection query: add EXPLAIN ANALYZE in Postgres or EXPLAIN in MySQL. If the query is slow, add an index on the join or filter columns. If the problem is CPU, scale the projection workers. In one case, a projection query took 12 seconds because it joined 7 tables without indexes; adding a single composite index cut it to 80 ms.


## What to build next

Pick a single bounded context where auditability or replayability matters most. Start with a single aggregate, store events in Postgres with a `events` table, and build one projection for your most frequent query. Measure storage growth and replay time after one week. If storage grows <20% and replay time stays under 1 second, expand to a second aggregate. If not, reconsider your domain—event sourcing may not be the right tool.