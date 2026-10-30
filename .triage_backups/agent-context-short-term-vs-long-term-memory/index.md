# Agent context: short-term vs long-term memory

## The mistake this article is about

A common context-engineering failure in agent systems is treating context as a cache when the application's correctness depends on it being a ledger. The code usually looks reasonable: a JSON blob keyed by agent ID, a TTL to bound growth, a read-modify-write on every step. It passes tests because tests run on fast, stable networks and short sessions. It fails in production because the failure mode is silent — the agent does not error out when context goes missing. It proceeds with whatever it can reconstruct, which is often nothing, and generates a plausible-sounding action that never happened.

This article compares two families of approaches to agent context:

- **Short-term memory:** fast key-value storage of the current context document, with a TTL and an append-only change log. The canonical implementation is a cache like Redis with a JSON-capable data type.
- **Long-term memory:** an append-only event log in a relational database, with optional vector embeddings for semantic retrieval over history. The canonical implementation is PostgreSQL with a vector extension and, if you need time-series rollups, a time-series extension.

The point is not that one is universally correct. The point is that the choice is a durability decision, and durability decisions should be made by identifying what the agent must never forget, not by benchmarking read latency on a laptop.

## What "context" actually means for an agent

Before comparing storage, separate the things people lump under "context." They have different durability requirements.

**Working state.** The current turn's scratchpad: the user's latest message, the tool calls in flight, the intermediate reasoning. This is genuinely ephemeral. Losing it means retrying one step.

**Conversation history.** The ordered record of what the user and agent said. Usually needed for the current session, sometimes needed across sessions.

**Durable facts and commitments.** The user's account number, the transaction the agent already initiated, the promise it made, the approval it received. Losing this is not a retry — it is a correctness violation. An agent that forgets it already submitted a refund may submit a second one.

**Audit record.** What the agent did, when, why, and with what inputs. Needed for debugging, dispute resolution, and compliance. This must survive restarts and must not be mutable.

A short-term store is appropriate for the first two categories. A long-term store is required for the last two. Most production incidents attributed to "the agent hallucinated" are actually failures in the third category: the agent lost a durable fact and confabulated a replacement.

## Option A: short-term memory in a fast key-value store

The pattern: store the agent's context as a single JSON document under a key like `agent:{id}:context`. Every step reads the whole document, mutates it in memory, and writes it back. A TTL bounds memory growth. An append-only stream records each write so the document can be rebuilt if it is evicted or lost.

```python
import os
import redis

r = redis.Redis(
    host=os.environ["REDIS_HOST"],
    port=6379,
    password=os.environ.get("REDIS_PASSWORD"),
    decode_responses=True,
)

CONTEXT_TTL_SECONDS = 3600
STREAM_MAXLEN = 1000

def load_context(agent_id: str) -> dict:
    ctx = r.hgetall(f"agent:{agent_id}:context")
    if ctx:
        return ctx
    # Fall back to replaying the change log.
    events = r.xrevrange(f"agent:{agent_id}:stream", count=100)
    return rebuild_context_from_events(events)

def save_context(agent_id: str, context: dict) -> None:
    r.hset(f"agent:{agent_id}:context", mapping=context)
    r.expire(f"agent:{agent_id}:context", CONTEXT_TTL_SECONDS)
    r.xadd(
        f"agent:{agent_id}:stream",
        {"event": "update", "data": serialize(context)},
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
```

Two details in that snippet matter more than they look.

First, `maxlen` on the stream is not optional. An unbounded append-only log inside a cache will eventually consume the memory you were trying to save. Trimming is what keeps this pattern viable.

Second, `rebuild_context_from_events` can only reconstruct what the stream still contains. If the stream was trimmed and the context document was evicted, the reconstruction is incomplete. The agent has no way to know this unless you make it check — and a check requires knowing what the correct state was, which is exactly the information you lost.

### Where this pattern is genuinely good

- High concurrency with a latency budget measured in single-digit milliseconds per step.
- Sessions short enough that the TTL never fires mid-session.
- Context that is reconstructible from an external source of truth (the user's message, an upstream API) if it is lost.
- Internal tooling where a wrong answer costs a retry, not a refund.

### Where it breaks

The characteristic failure is **eviction under memory pressure**. When the store hits its memory limit, the configured eviction policy decides what to drop. If that policy is not `noeviction`, the store will happily discard live agent contexts to make room. The agent then reads an empty or partial document and proceeds.

A second failure is **partial writes**. A read-modify-write cycle on a JSON document is not atomic unless you make it atomic. Two concurrent steps on the same agent can interleave and produce a document that reflects neither step's intent. In a cache this is easy to miss because the write succeeds — it just writes the wrong thing.

A third is **TTL expiry during a long step**. If a step takes longer than expected (a slow tool call, a retry loop), the context can expire between the read and the write-back. The write-back then recreates the key with a partial document and a fresh TTL, silently discarding the prior state.

None of these produce an exception. They produce an agent that confidently acts on a state that never existed.

## Option B: long-term memory as an append-only event log

The pattern: never overwrite context. Instead, append immutable events — `UserMessage`, `ToolCall`, `ToolResult`, `CommitmentMade` — and derive the current context by folding over the relevant events. Add a vector column so history can be retrieved by semantic similarity rather than only by time.

```python
import os
from datetime import datetime, timezone

import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    host=os.environ["PG_HOST"],
    port=5432,
    dbname=os.environ["PG_DATABASE"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
)
register_vector(conn)

def add_event(agent_id: str, event_type: str, payload: str) -> None:
    embedding = generate_embedding(payload)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_events (agent_id, event_type, payload, embedding, ts)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (agent_id, event_type, payload, embedding, datetime.now(timezone.utc)),
        )
    conn.commit()

def get_relevant_context(agent_id: str, query: str, limit: int = 10):
    embedding = generate_embedding(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT payload, ts, embedding <=> %s AS distance
            FROM agent_events
            WHERE agent_id = %s
              AND ts > NOW() - INTERVAL '7 days'
            ORDER BY distance
            LIMIT %s
            """,
            (embedding, agent_id, limit),
        )
        return cur.fetchall()
```

Note the operator: pgvector's distance operators are `<=>` (cosine), `<->` (L2), and `<#>` (negative inner product). Ordering ascending by `<=>` gives nearest-first. A `similarity()` function does not exist in pgvector; using one is a common copy-paste error.

### Why append-only matters

The value of this pattern is not the database. It is the invariant: **an event, once written, is never modified or deleted.** That invariant gives you three things a cache cannot.

1. **Reconstruction is exact.** The current state is a pure function of the event log. If a projection is wrong, you recompute it — you do not have to guess what was lost.
2. **Audit is free.** The log *is* the audit record. You do not need a separate pipeline to answer "what did the agent do and why."
3. **Idempotency is enforceable.** Before performing a side effect, check whether an event recording that side effect already exists. This is how you prevent the double-refund class of bug, which no amount of cache tuning fixes.

### Where this pattern is genuinely good

- Any agent that touches money, identity, medical data, or legal commitments.
- Sessions that span hours or days, where TTL-based expiry is not an option.
- Systems that need to answer retrospective questions ("which agents discussed X last month").
- Systems where you expect to change the agent's logic and want to replay history against the new logic.

### Where it breaks

The characteristic failure is **retrieval quality**, not durability. Semantic search over an event log returns the nearest events by embedding distance, which is not the same as the most relevant events. If the fold that reconstructs context is wrong, the agent gets a coherent but incorrect picture. Durability guarantees the data is there; it does not guarantee the agent looks at the right part of it.

The second failure is **write amplification and cost**. Every step writes an event plus an embedding. Embedding generation is a per-call cost and a per-call latency. At high step volumes this dominates the bill, and it is the line item teams most often forget to model.

The third is **schema evolution**. Adding a field to an event payload is easy. Changing the meaning of an existing field is not, because old events cannot be rewritten. Version your event types from day one.

## A worked example: the double refund

Consider an agent that processes refund requests. The user asks for a refund of $240. The agent validates, calls the payment API, and confirms.

**Short-term memory path.** Step 1 writes `{refund_pending: 240}` to the context key. Step 2 calls the payment API. Step 3 writes `{refund_completed: 240}`. If the process crashes between step 2 and step 3, the context still says `refund_pending`. On restart, the agent sees a pending refund and retries the API call. The user is refunded twice. The context store was never wrong about what it knew — it simply never learned about the side effect.

**Long-term memory path.** Step 2 writes a `RefundInitiated` event with a client-generated idempotency key before calling the API. Step 3 writes `RefundCompleted`. On restart, the agent folds the log, sees `RefundInitiated` without a matching `RefundCompleted`, and queries the payment provider for the status of that idempotency key rather than blindly retrying. The duplicate is prevented by the log, not by the cache's availability.

The lesson generalizes: **the moment your agent performs a side effect it cannot undo, the record of that side effect must be durable before the side effect happens.** This is the write-ahead logging rule, and it applies to agents exactly as it applies to databases.

## How to measure your own system

Do not adopt a benchmark from an article. Measure the two things that actually decide this: context loss rate and step latency under your real conditions.

**Context loss rate.** Instrument a counter that increments whenever the agent reads a context that fails a validity check you define (for example, a monotonic sequence number that should never go backwards, or a required field that is missing). Log the agent ID and the step. After a week of normal traffic, the ratio of loss events to total steps is your loss rate. If it is not zero, the short-term store is losing state, and no amount of latency tuning compensates.

**Step latency distribution.** Record the wall-clock time of each step, from context read to context write, and keep the full distribution — not just the mean. The relevant question is the tail: p95 and p99 under load, not the median on an idle machine.

```bash
# Redis: is anything being evicted?
redis-cli INFO stats | grep -E "evicted_keys|keyspace_hits|keyspace_misses"

# Redis: how close is this key to expiring?
redis-cli TTL agent:1234:context

# Redis: how large is the keyspace?
redis-cli INFO memory | grep used_memory_human

# Postgres: how much of the query time is the vector index?
EXPLAIN (ANALYZE, BUFFERS)
SELECT payload, ts, embedding <=> '[...]' AS distance
FROM agent_events
WHERE agent_id = '1234' AND ts > NOW() - INTERVAL '7 days'
ORDER BY distance LIMIT 10;
```

A non-zero `evicted_keys` on a store holding live agent context is the signal that matters most. It means the store has already decided which agents to forget.

## A comparison that is actually a comparison

The table below compares the two patterns by property, not by invented measurement. "Higher" and "lower" are relative to each other, not to a benchmark.

| Property | Short-term (cache) | Long-term (event log) |
|---|---|---|
| Read latency for current state | Lower | Higher |
| Write latency per step | Lower | Higher |
| Survives process restart | Only if persistence is configured and fsync is not deferred | Yes, by construction |
| Survives store eviction | No | Yes |
| Exact reconstruction after loss | No | Yes |
| Audit trail | Requires a separate pipeline | Inherent |
| Idempotency enforcement | Manual, racy | Natural |
| Schema evolution | Trivial (schemaless document) | Requires versioned events |
| Per-step cost at high volume | Storage and network only | Storage, network, plus embedding generation |
| Operational surface | Small | Larger (indexes, vacuum, query plans) |

The two rows that decide most cases are "survives store eviction" and "idempotency enforcement." Everything else is a tuning problem.

## A hybrid that is usually the right answer

The two patterns are not mutually exclusive, and the common production shape is both:

- The **event log is the source of truth.** Every state-changing action is appended before it is performed.
- The **cache is a derived projection** of recent events, rebuilt from the log on a miss, and safe to lose at any moment.

The critical property is that the cache is *never* authoritative. If it disagrees with the log, the log wins and the cache is rebuilt. This turns the cache's failure modes from correctness bugs into latency blips.

```python
def load_context(agent_id: str) -> dict:
    cached = r.hgetall(f"agent:{agent_id}:context")
    if cached and int(cached.get("log_offset", -1)) == latest_log_offset(agent_id):
        return cached
    # Cache is missing or stale: rebuild from the authoritative log.
    events = fetch_events_since(agent_id, cached.get("log_offset") if cached else None)
    context = fold_events(events)
    save_context(agent_id, context, log_offset=latest_log_offset(agent_id))
    return context
```

The `log_offset` field is what makes this safe. Without it, a stale cache entry is indistinguishable from a fresh one, and the agent acts on yesterday's state.

## Decision checklist

Work through these in order. The first "yes" that applies to a durable fact ends the discussion.

1. Does the agent perform any irreversible side effect — a payment, a message sent, a record created? If yes, the record of that side effect must be durable before the effect occurs.
2. Would a user be harmed if the agent forgot something it previously committed to? If yes, that commitment is a durable fact.
3. Does any regulation or contract require you to reconstruct what the agent did? If yes, you need an immutable log.
4. Do sessions routinely outlive a reasonable TTL? If yes, TTL-based expiry is not viable, and the cache cannot be the source of truth.
5. Can the context be reconstructed from an external system of record on demand? Only if the answer is an unambiguous yes may the cache be authoritative.
6. Is your team able to operate a relational database — backups, index maintenance, query-plan debugging? If not, that is a real constraint, and it argues for starting with the log in the simplest possible form (a single table, no vector column) rather than for skipping durability.

If none of the first five apply, a cache-only design is defensible. Add an eviction monitor and treat any eviction as a page-worthy event.

## FAQ

**Can I use the cache as the source of truth if I enable persistence?**
Persistence protects against process restart, not against eviction, memory pressure, or a corrupted snapshot. The failure mode you are worried about — the agent acting on missing state — is not addressed by persistence alone.

**Do I need vector search at all?**
No. The durability property comes from the append-only log, not from embeddings. Many systems start with a plain table of events ordered by time and add vector search only when "most recent N events" proves insufficient. That is a cheaper and simpler starting point.

**How do I bound the size of the event log?**
Archive old events to cold storage on a schedule, but keep the invariant that archived events are immutable and retrievable. Never delete events that record an irreversible side effect. If an event must be removed for privacy reasons, replace it with a tombstone that preserves the fact that something happened, even if the payload is gone.

**Is the cache ever worth keeping?**
Yes, as a projection. The cost of the cache is small compared to the cost of a wrong action, and it removes the log read from the hot path for the common case.

**What about time-series extensions for rollups?**
They are useful when you need continuous aggregates over high-volume event streams — for example, per-hour counts of tool calls. They are an optimization on top of the log, not a substitute for it. Add them when a specific query is too slow, not preemptively.

## One thing to do in the next 30 minutes

Pick one agent in your system and answer this question with evidence, not memory: **if the context store were wiped right now, which of this agent's actions would it repeat?**

To find out, check whether the store is evicting anything:

```bash
redis-cli INFO stats | grep evicted_keys
```

If that number is non-zero, the store has already been discarding live contexts. Then grep your agent code for the side-effecting calls — payment, email, record creation — and check whether each one is guarded by a durable record written *before* the call. Any unguarded call is a duplicate-action bug waiting for a restart. Write the guard, or move the record to a log, before you tune anything else.
