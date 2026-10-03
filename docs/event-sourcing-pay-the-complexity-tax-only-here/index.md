# Event sourcing: pay the complexity tax only here

Event sourcing is an append-only log of immutable domain events from which all current state is derived. It solves a narrow set of problems well and adds real operational cost. This article covers where the pattern pays off, where a temporal table is sufficient, and how to measure the tradeoff in your own system before committing.

## The one-paragraph version

Event sourcing records every state change as an immutable event in an append-only log instead of updating rows in place. It is a good fit when you need to reconstruct any past state, run new read models without touching production writes, or evolve schemas without rewriting history. The documented costs are schema evolution discipline, snapshot storage, projection lag, and a mental model shift from CRUD to append-only. Use it when your domain has high replay or audit value and your read side can tolerate eventual consistency. Avoid it if you only need to know who changed what, or if your team has no process for versioning event contracts.

## Why this concept confuses people

**Terminology overload.** Event sourcing, CQRS, message brokers, and change data capture are related but distinct. A system that publishes domain events to a broker while its read models still query a mutable table is not event-sourced; it is a CRUD system with notifications. CQRS splits reads from writes; event sourcing makes the write side an append-only log. You can have either without the other.

**Cost myopia.** The visible cost is storage; the real cost is schema evolution, snapshot management, and projection operations. Storage prices change and vary by provider, so estimate from your own bill rather than from a table in an article.

**The myth of simple audit.** Many compliance requirements are satisfied by temporal tables (system-versioned tables in PostgreSQL 15+, for example) or by change streams. Event sourcing is overkill unless you also need deterministic replay, for example to reconstruct state at an arbitrary timestamp or to rebuild a projection in a new shape.

## The mental model

Think of your system as a replayable tape rather than a mutable whiteboard.

- **Mutable whiteboard (CRUD):** you overwrite yesterday's values with today's. Answering "what did the balance look like at 14:07 yesterday?" requires backups or logs that were not designed for that query.
- **Replayable tape (event sourcing):** every change is appended. To see the balance at 14:07, replay the tape up to that point. The tape never changes; you replay it differently.

This changes design decisions:

- **Commands** express intent and are validated before producing events.
- **State** is a derived view computed from events.
- **Schema evolution** is additive: new event types are appended; old events stay immutable.

Versioning discipline is what makes replay deterministic. A schema registry that enforces backward compatibility at publish time prevents silent corruption that only surfaces during a replay months later.

## A worked example: a minimal event-sourced wallet

The example below uses Python, FastAPI, and SQLite as a local event store. It is small enough to run in a few minutes and shows the core mechanics: append, replay, project.

### Step 1: The event contract

```python
SCHEMA = {
    "type": "record",
    "name": "WalletEvent",
    "fields": [
        {"name": "wallet_id", "type": "string"},
        {"name": "event_id", "type": "string"},
        {"name": "event_type", "type": {"type": "enum", "name": "EventType", "symbols": ["Deposited", "Withdrawn"]}},
        {"name": "amount", "type": "double"},
        {"name": "timestamp", "type": {"type": "long", "logicalType": "timestamp-millis"}},
        {"name": "version", "type": "int"}
    ]
}
```

### Step 2: Event store schema

```python
import sqlite3, uuid, time

conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        wallet_id TEXT NOT NULL,
        event_id TEXT NOT NULL UNIQUE,
        event_type TEXT NOT NULL,
        amount REAL NOT NULL,
        timestamp INTEGER NOT NULL,
        version INTEGER NOT NULL
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_events_wallet ON events (wallet_id, seq)")
```

### Step 3: Append and replay

```python
def current_version(wallet_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) FROM events WHERE wallet_id = ?", (wallet_id,)
    ).fetchone()
    return row[0] or 0

def append_event(wallet_id: str, event_type: str, amount: float) -> str:
    event_id = str(uuid.uuid4())
    version = current_version(wallet_id) + 1
    conn.execute(
        "INSERT INTO events (wallet_id, event_id, event_type, amount, timestamp, version) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (wallet_id, event_id, event_type, amount, int(time.time() * 1000), version)
    )
    return event_id

def get_balance(wallet_id: str) -> float:
    rows = conn.execute(
        "SELECT event_type, amount FROM events WHERE wallet_id = ? ORDER BY seq ASC",
        (wallet_id,)
    ).fetchall()
    balance = 0.0
    for event_type, amount in rows:
        if event_type == "Deposited":
            balance += amount
        elif event_type == "Withdrawn":
            balance -= amount
    return balance
```

### Step 4: HTTP endpoints

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/wallets/{wallet_id}/deposit")
def deposit(wallet_id: str, amount: float):
    append_event(wallet_id, "Deposited", amount)
    return {"ok": True}

@app.get("/wallets/{wallet_id}")
def get(wallet_id: str):
    return {"wallet_id": wallet_id, "balance": get_balance(wallet_id)}
```

### Step 5: Run it

```bash
pip install fastapi uvicorn
uvicorn main:app --reload
curl -X POST "http://localhost:8000/wallets/w1/deposit?amount=100"
curl http://localhost:8000/wallets/w1
# => {"wallet_id":"w1","balance":100.0}
```

### A failure mode worth knowing

The code above is correct for a single process, but it has a race condition. Two concurrent deposits for the same wallet can both read the same `current_version` and write duplicate version numbers. The fix is a unique constraint on `(wallet_id, version)` plus a retry on conflict, or a transactional read-modify-write with a row lock. This is the kind of issue that shows up under load and not in a demo. In production, this is typically handled by optimistic concurrency control: the append operation checks the expected version and fails if it has changed since the last read.

## The tradeoff, in concrete terms

The decision is not "event sourcing or nothing." The realistic choices are:

1. **CRUD with an audit log.** Cheapest to build, hardest to replay.
2. **CRUD with temporal tables.** Adds point-in-time queries without changing the write model.
3. **Event sourcing.** Adds deterministic replay and flexible projections, at the cost of schema discipline and projection operations.

A useful way to frame the tradeoff is in terms of the questions you need to answer:

| Question | CRUD + audit log | Temporal tables | Event sourcing |
|---|---|---|---|
| What is the current state? | Yes | Yes | Yes (via projection) |
| What was the state at time T? | Painful | Yes | Yes |
| Rebuild a read model in a new shape? | No | No | Yes |
| Deterministic replay of domain intent? | No | No | Yes |
| Schema changes without rewriting history? | N/A | Limited | Yes, with versioned events |
| Operational complexity | Low | Low | High |

## How to measure whether it pays off in your system

Before committing, instrument three things and compare them against your requirements.

**1. Replay time.** Write a script that replays all events for a representative aggregate and measures wall-clock time. For an aggregate with N events, replay time is roughly `N × per-event-processing-time`. If per-event processing is 50 microseconds, 1 million events take about 50 seconds. If your p99 requirement for rebuilding a projection is 30 seconds, you need snapshots or parallelism. Measure it; do not estimate.

**2. Projection lag.** If you run projections asynchronously, instrument the difference between the event's append timestamp and the time the projection is updated. Track p50 and p99. A projection lag of a few hundred milliseconds is often acceptable; a lag of minutes usually is not.

**3. Storage growth.** Measure bytes per event for your actual payloads, then multiply by your expected event rate. For example, at 1,000 events per second and 500 bytes per event, that is 500 KB/s, or about 43 GB per day. Apply your provider's storage rate to get a monthly figure. Add snapshot storage on top of that.

Run these three measurements on a staging environment with a realistic event volume. If replay time and projection lag fit your requirements without snapshots and without exotic infrastructure, event sourcing may be a reasonable fit. If not, the complexity tax is probably not worth paying yet.

## Common misconceptions

**"Events are just logs."** Events are immutable facts with a defined order per aggregate. If events are not keyed by aggregate identifier, replay can produce non-deterministic results. Partitioning by aggregate id is not optional.

**"We can skip snapshots."** Snapshots trade storage for replay time. Whether you need them depends on your replay time budget, which you should measure. A common pattern is to snapshot every N events, where N is chosen so that replay from the last snapshot fits within your latency budget.

**"Event sourcing equals CQRS."** They are orthogonal. CQRS splits read and write models; event sourcing makes the write side append-only. You can have CQRS with a mutable write model (read replicas, for example) and event sourcing with a single read model.

**"Schema evolution is free."** Renaming a field or changing a type is a breaking change in most serialization formats unless handled carefully. The safe pattern is to add new fields with defaults, deprecate old ones over a migration window, and rebuild projections from the full event history to verify correctness.

## A decision checklist

Before adopting event sourcing, answer these questions honestly:

- Does a regulator, auditor, or business process require deterministic replay of domain intent, not just a record of changes?
- Can you name a read model that you would rebuild from events if the schema changed? If not, you may not need event sourcing.
- Does your team have a process for versioning event contracts and reviewing breaking changes?
- Have you measured replay time for your largest aggregate?
- Have you measured projection lag under realistic load?
- Do you have a plan for snapshots, and do you know the storage cost?
- Is eventual consistency acceptable for every consumer of the read models?

If you answer "no" to any of the first three, event sourcing is likely premature. If you answer "no" to any of the last four, you are not ready to operate it.

## FAQ

**Why not just use change data capture from the database?**
Change data capture gives you a stream of row-level changes. Those changes are tied to the current table schema and do not carry domain intent. Replaying them after a schema change is fragile. Event sourcing records domain events with versioned contracts, so replay remains deterministic across schema changes.

**What is the minimum write volume that justifies event sourcing?**
There is no universal threshold. The relevant question is whether you need replay or flexible projections, not how many writes you have. A low-volume system with a strict regulatory replay requirement may justify event sourcing; a high-volume system with no replay requirement may not.

**How do I handle breaking schema changes?**
Add new fields with defaults rather than renaming or retyping existing fields. Deprecate old fields over a migration window. Rebuild projections from the full event history in staging to verify that the new schema produces the same results. Use a schema registry that rejects breaking changes at publish time if your serialization format supports one.

**Can I use event sourcing with serverless functions?**
Yes, but batch events to reduce per-invocation overhead, and make handlers idempotent because retries may redeliver events. Watch for partial batch failures and ensure your append operation is safe to retry.

## Your next 30 minutes

Pick the aggregate in your system that auditors or support engineers ask about most often. Time how long it takes to answer: "What did this aggregate look like at a specific past timestamp?" If the answer requires restoring a backup, parsing binlogs, or writing a one-off script, record that time and the steps involved. Then compare it against the replay-time measurement described above. That comparison, not a general preference for or against event sourcing, is what should drive the decision.
