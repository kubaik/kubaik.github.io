# Supervisor vs swarm vs debate vs pipeline

Most multi-agent orchestration guides assume a clean environment and a patient timeline. The failure mode that matters in production rarely appears in those guides: the orchestrator itself becomes the outage. A retry storm, a health check that returns 200 while the queue behind it grows without bound, a restart loop that consumes a cluster — these are orchestration failures, not model failures.

The gap between a tutorial and a production system is not about scaling. It is about **recovery**. A multi-agent system that survives production needs three properties that toy examples skip:

1. A deterministic way to restart failed agents without cascading retries.
2. A circuit breaker that stops the orchestrator from spamming the message broker.
3. A way to replay a conversation when a downstream service times out, rather than only retrying the last call.

A common failure mode illustrates why: a single `TimeoutError` from an external API triggers thousands of retries in under a minute, and the retry queue grows faster than the supervisor can drain it. The supervisor becomes the denial-of-service vector against its own dependencies. The fix is rarely more RAM. It is a circuit breaker and a maximum retry count encoded in the supervisor's state machine.

Multi-agent orchestration is not about making agents smarter. It is about making the **orchestrator dumber** — deliberately limiting its power so it cannot destroy itself when something downstream fails.

## How the four patterns work under the hood

### Supervisor: the strict parent

A supervisor pattern treats agents like child processes: it spawns them, monitors their health, and replaces them if they crash. The key property is **idempotent restarts**.

In practice, the supervisor keeps a heartbeat table in a shared store with three columns: `agent_id`, `last_seen`, and `status`. If `status` is `unhealthy` for three consecutive heartbeats — nine seconds at a three-second interval — the supervisor kills the agent and spawns a fresh instance with the same configuration. No memory, no state, no drama.

The supervisor also enforces a **max restart budget**: for example, five crashes per agent per hour. After that, it blacklists the agent and alerts the on-call engineer. This prevents the supervisor from looping forever on a broken agent. A corrupted container image is a classic trigger: without a budget, the supervisor will restart the same broken agent indefinitely, and each restart consumes scheduling and I/O resources.

The supervisor's state machine is tiny:
```python
from dataclasses import dataclass
from enum import Enum, auto

class AgentStatus(Enum):
    HEALTHY = auto()
    UNHEALTHY = auto()
    BLACKLISTED = auto()

@dataclass
class AgentHeartbeat:
    agent_id: str
    status: AgentStatus
    last_seen: float
```

### Swarm: the anarchist collective

A swarm pattern removes the supervisor entirely. Agents broadcast their presence via mDNS or a gossip protocol, and any agent can handle a task. This is seductive until you notice that **no agent has a global view**. A swarm can lose half its nodes and the remaining agents will keep working — until they try to talk to a dead peer and hang indefinitely.

The only way to survive production with a swarm is to bake **ephemeral state** into every message. If AgentA sends a message to AgentB and AgentB never replies, AgentA must eventually assume AgentB is dead and either reroute the message to another agent or fail the task gracefully.

The mechanism that makes this work is a `last_ack` timestamp on every task plus a TTL. If a task's `last_ack` is older than the TTL — 30 seconds is a reasonable starting point — the swarm marks the agent as dead and reroutes. Without that timestamp, routing decisions are based on stale presence data, and messages flow to nodes that stopped responding minutes ago.

The swarm's simplicity is also its fragility. Without a supervisor, the design bets on agents being **stateless** and the network being **reliable**. Neither holds in practice.

### Debate: the courtroom drama

A debate pattern turns orchestration into a **consensus protocol**. Agents argue over the best answer to a question, and the final output is the consensus view. This works for subjective tasks such as summarizing a document, and falls apart for **deterministic tasks** such as calculating a total.

A workable debate protocol is a round-robin tournament with a quorum. Each agent produces an answer, then the next agent critiques it. After N rounds, the system picks the answer with the highest average score from all critiques. If no answer reaches a two-thirds quorum, the task fails.

The catch: **agents do not retain their own answers** between rounds. The system must store intermediate state externally, keyed by task and round — for example, `debate:{task_id}:round:{round_num}`.

Debate is the most expensive of the four patterns in both latency and compute. It trades CPU and wall-clock time for **quality**. Use it when correctness on a subjective judgment outweighs speed.

### Pipeline: the waterfall that never dries up

A pipeline pattern treats agents like stages in a factory assembly line. Each agent does one thing well, and the output of one agent feeds the input of the next. The key to survival is **backpressure**.

In practice, each pipeline stage has:
- a bounded queue (the bound is a design choice; 100 items is a common starting point)
- a per-stage timeout
- a retry policy with exponential backoff
- a dead-letter queue for items that fail all retries

A typical five-stage pipeline: validation, enrichment, business logic, database write, and webhook delivery.

The failure mode that catches teams is **a downstream stage blocking an upstream one**. The enrichment stage sends items to the database-write stage, but database writes slow to 50 items/second. The enrichment stage's queue grows, and its memory usage balloons. The supervisor does not notice, because the enrichment stage's health check still returns 200 — the process is alive and answering probes. It is just drowning.

The fix is to add **queue depth metrics** to the health check. If a stage's queue depth exceeds 80% of its max size, the stage is marked unhealthy and the supervisor restarts it. This is a few lines of code and turns an invisible failure into a visible one.

| Pattern    | Pros                          | Cons                          | Best for                          |
|------------|-------------------------------|-------------------------------|-----------------------------------|
| Supervisor | Simple, restarts on failure   | Single point of failure       | Reliable, low-latency workflows   |
| Swarm      | No single point of failure    | No global state, hard to debug| Highly available, stateless jobs |
| Debate     | Higher quality output         | Expensive, slow               | Subjective tasks, consensus tasks |
| Pipeline   | Predictable flow              | Backpressure surprises        | Ordered, multi-step workflows     |

## Step-by-step implementation

### Supervisor in Go with Redis

A minimal supervisor spawns agents and restarts them on failure, using Redis for heartbeats and a small state machine.

```go
package supervisor

import (
    "context"
    "log"
    "time"

    "github.com/redis/go-redis/v9"
)

type Agent struct {
    ID         string
    Command    string
    MaxRestart int
}

type Supervisor struct {
    redisClient *redis.Client
    agents      map[string]*Agent
    maxRestart  int
    interval    time.Duration
}

func NewSupervisor(redisAddr string) *Supervisor {
    return &Supervisor{
        redisClient: redis.NewClient(&redis.Options{Addr: redisAddr}),
        agents:      make(map[string]*Agent),
        maxRestart:  5,
        interval:    3 * time.Second,
    }
}

func (s *Supervisor) Monitor(ctx context.Context) {
    ticker := time.NewTicker(s.interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            s.checkHeartbeats(ctx)
        }
    }
}

func (s *Supervisor) checkHeartbeats(ctx context.Context) {
    keys, err := s.redisClient.Keys(ctx, "heartbeat:*").Result()
    if err != nil {
        log.Printf("redis keys error: %v", err)
        return
    }

    for _, key := range keys {
        agentID := key[len("heartbeat:"):]
        lastSeen, err := s.redisClient.Get(ctx, key).Float64()
        if err != nil {
            log.Printf("redis get error for %s: %v", agentID, err)
            continue
        }

        if time.Since(time.Unix(int64(lastSeen), 0)) > 9*time.Second {
            s.restartAgent(ctx, agentID)
        }
    }
}

func (s *Supervisor) restartAgent(ctx context.Context, agentID string) {
    agent, ok := s.agents[agentID]
    if !ok {
        return
    }

    restartCount, err := s.redisClient.Incr(ctx, "restart:count:"+agentID).Result()
    if err != nil {
        log.Printf("redis incr error: %v", err)
        return
    }

    if restartCount > int64(agent.MaxRestart) {
        s.redisClient.Set(ctx, "blacklist:"+agentID, "true", 1*time.Hour)
        log.Printf("blacklisted %s after %d restarts", agentID, agent.MaxRestart)
        return
    }

    // Spawn new agent (pseudo-code)
    go s.spawnAgent(agent)
    log.Printf("restarted %s (attempt %d)", agentID, restartCount)
}
```

Three properties matter here:
- Shared state lives in Redis, not in an in-memory map, so it survives a supervisor restart.
- Restart counts are **persistent**, so they survive agent crashes.
- Blacklist durations are **short** — one hour is long enough to cool down and short enough to recover.

Note that `KEYS` is dangerous on a large keyspace because it scans everything. In production, use `SCAN` with a cursor, or maintain a set of active agent IDs instead of discovering them by pattern.

### Swarm in Node.js with NATS

A minimal swarm uses NATS for message routing and heartbeats.

```javascript
// agent.js
import { connect } from 'nats.ws'
import { randomUUID } from 'node:crypto'
import { setTimeout } from 'timers/promises'

const natsUrl = process.env.NATS_URL || 'nats://localhost:4222'
const agentId = process.env.AGENT_ID || randomUUID()

const nc = await connect({ servers: natsUrl })
const js = nc.jetStream()

// Heartbeat loop
setInterval(async () => {
  await js.publish('gossip.agents', JSON.stringify({
    type: 'heartbeat',
    agentId,
    timestamp: Date.now(),
  }))
}, 3000)

// Task processing
const sub = nc.subscribe('tasks.>', { callback: async (err, msg) => {
  if (err) {
    console.error('NATS error:', err)
    return
  }

  let task
  try {
    task = JSON.parse(msg.data.toString())
  } catch (e) {
    console.error('Malformed task payload:', e)
    msg.term() // Do not redeliver unparseable messages
    return
  }

  try {
    const result = await processTask(task)
    await js.publish(`results.${task.taskId}`, JSON.stringify({ result }))
    msg.ack()
  } catch (e) {
    // No retry logic here — swarm relies on upstream to reroute
    console.error('Task failed:', e)
    msg.ack() // Explicit ack to prevent redelivery
  }
}})
```

The critical detail: **no agent blocks indefinitely**. If an agent cannot process a task, it acks the message and lets the upstream decide what to do. This keeps the swarm alive when nodes fail. Note the distinction between `ack` and `term`: a malformed message should be terminated, not acked, so it does not consume redelivery budget.

### Debate in Python with FastAPI and Redis

A debate pattern with round-robin scoring:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
import json

app = FastAPI()
redis_client = redis.from_url("redis://localhost:6379")

class DebateRound(BaseModel):
    task_id: str
    round_num: int
    answers: list[str]

@app.post("/debate/start")
async def start_debate(prompt: str):
    task_id = f"debate:{prompt[:8]}"
    await redis_client.set(f"debate:{task_id}:prompt", prompt)
    await redis_client.set(f"debate:{task_id}:round", 0)
    await redis_client.set(f"debate:{task_id}:status", "active")
    return {"task_id": task_id}

@app.post("/debate/round")
async def debate_round(round: DebateRound):
    await redis_client.set(
        f"debate:{round.task_id}:round:{round.round_num}",
        json.dumps(round.model_dump())
    )
    await redis_client.incr(f"debate:{round.task_id}:round")
    return {"ok": True}

@app.get("/debate/result/{task_id}")
async def get_result(task_id: str):
    status = await redis_client.get(f"debate:{task_id}:status")
    if status != "active":
        raise HTTPException(status_code=404, detail="Debate ended")

    last_round = int(await redis_client.get(f"debate:{task_id}:round"))
    scores = {}

    for r in range(last_round):
        data = json.loads(
            await redis_client.get(f"debate:{task_id}:round:{r}")
        )
        for ans in data["answers"]:
            scores[ans] = scores.get(ans, 0) + 1

    quorum = await redis_client.incr(f"debate:{task_id}:quorum")
    if quorum >= (2 * last_round / 3):
        winner = max(scores.items(), key=lambda x: x[1])[0]
        await redis_client.set(f"debate:{task_id}:status", "finished")
        await redis_client.set(f"debate:{task_id}:winner", winner)
        return {"winner": winner}

    raise HTTPException(status_code=202, detail="No quorum yet")
```

This is illustrative scaffolding, not a finished service. The quorum counter increments on every result read, which is almost certainly not the intended semantics — quorum should be computed from the round data, not incremented as a side effect of a GET. Treat the snippet as a sketch of the data model: one key per round, one status key, one winner key.

The debate pattern's latency is dominated by **Redis round trips**. Each round writes a JSON blob of answers and critiques. Pipelining helps, but each round still adds a network hop. For four rounds, that is four sequential write-then-read cycles before a winner emerges — expensive for high-throughput systems.

### Pipeline in Rust with Tokio and PostgreSQL

A pipeline with backpressure and bounded queues:

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use sqlx::postgres::PgPoolOptions;

#[derive(Clone)]
struct Pipeline {
    tx: mpsc::Sender<String>,
    queue_depth: usize,
}

impl Pipeline {
    async fn new(pool: sqlx::PgPool) -> Self {
        let (tx, mut rx) = mpsc::channel(100); // bounded queue
        tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                if let Err(e) = process_stage_a(&pool, &item).await {
                    eprintln!("Stage A failed: {}", e);
                    // Dead-letter queue
                    if let Err(e) = sqlx::query("INSERT INTO dead_letters (payload) VALUES ($1)")
                        .bind(&item)
                        .execute(&pool)
                        .await
                    {
                        eprintln("Dead letter failed: {}", e);
                    }
                }
            }
        });
        Self { tx, queue_depth: 100 }
    }
}

async fn process_stage_a(pool: &sqlx::PgPool, item: &str) -> Result<(), sqlx::Error> {
    sleep(Duration::from_millis(10)).await; // Simulate work
    let _ = sqlx::query("INSERT INTO stage_a (payload) VALUES ($1)")
        .bind(item)
        .execute(pool)
        .await?;
    Ok(())
}
```

The pipeline's health check is simple: if the channel's `len()` exceeds 80% of its capacity, the supervisor marks the stage as unhealthy. This prevents memory blowups and keeps the pipeline flowing. Note that `mpsc::Sender` does not expose `len()` directly — you need to share the receiver's length via an `Arc<AtomicUsize>` updated on send, or use a channel type that reports depth.

## How to measure performance instead of guessing

Published benchmark tables for multi-agent patterns are close to useless, because performance depends on message size, serialization format, network topology, and the ratio of I/O to compute. Measure your own system. The instrumentation is straightforward:

**What to instrument:**
- Per-stage queue depth, sampled every second.
- End-to-end latency at P50, P95, and P99, recorded at the point where a task enters and where its result is acknowledged.
- Error rate split by category: timeout, malformed input, downstream rejection, and internal exception.
- Retry count per task, with a histogram rather than an average.
- Broker metrics: publish rate, deliver rate, pending messages, and memory used by the stream.
- Shared-store metrics: operations per second, CPU utilization, and blocked clients.

**What to compare:**
Run the same workload through each pattern with identical payloads and identical downstream services. Hold concurrency constant. Then vary one dimension at a time — payload size, worker count, downstream latency — and record where each pattern breaks first.

**What to expect, qualitatively:**
- The supervisor adds one heartbeat write per agent per interval plus one restart decision per failure. Its overhead scales with agent count, not request count.
- The swarm adds one broadcast per heartbeat per agent, and its routing decisions depend on the freshness of presence data. Overhead scales with agent count squared if every agent gossips to every other.
- The debate adds one round of full fan-out per debate round. Its cost scales with rounds × agents × payload size.
- The pipeline adds one queue operation per item per stage. Its overhead is linear in items and stages, and its failure mode is queue growth, not CPU.

The one number worth watching above all others is **queue depth**. Latency rises before error rate does, and queue depth rises before latency does. If you instrument only one thing, instrument queue depth.

## Failure modes that appear in real deployments

### 1. The heartbeat table becomes a hotspot

A single-threaded shared store serializes every heartbeat write. At high heartbeat rates, the store's CPU saturates and clients block. The fix is to shard heartbeat keys by agent ID hash:
```python
# Before
key = f"heartbeat:{agent_id}"

# After
shard = hash(agent_id) % 16
key = f"heartbeat:{shard}:{agent_id}"
```
Sharding distributes writes across slots and reduces contention. Measure the store's CPU before and after to confirm the effect; the improvement depends on your client library's connection pooling as much as on the key layout.

### 2. The message broker silently drops messages under load

Streaming brokers typically have a configured maximum memory or disk budget for retained messages. When the stream fills, behavior depends on the retention policy: some configurations drop old messages, some block publishers, and some reject new publishes. None of these is obvious from the client side. Set explicit limits, monitor stream size against them, and alert at 80% of the configured maximum.

### 3. Agent memory leaks compound across restarts

In the supervisor pattern, agents restart periodically. If an agent leaks memory at a steady rate, each restart resets the process and hides the leak from the operating system's view. The fix is to log resident memory every 30 seconds and alert on a positive slope across restarts, not just on absolute usage. A leak in a JSON parsing library, for example, will show up as a sawtooth pattern in a memory graph — rising within each process lifetime, dropping at restart, rising again.

### 4. Debate quorum deadlocks on ambiguous prompts

Debate patterns assume agents converge. Ambiguous prompts cause agents to disagree indefinitely, and the quorum never forms. A tiebreaker agent with a stricter rubric resolves most cases, at the cost of an extra round trip and additional state writes. The alternative is to fail fast: set a maximum round count and return the highest-scoring answer with a confidence flag, rather than blocking forever.

### 5. Pipeline stages block each other invisibly

A downstream stage slows; an upstream stage's queue grows; the upstream stage's health check still returns 200 because the process is alive. The supervisor sees a healthy fleet while memory climbs toward the limit. The fix is to make queue depth part of the health signal:
```go
func (s *Supervisor) checkHealth(ctx context.Context, agentID string) bool {
    queueDepth, err := s.redisClient.LLen(ctx, "queue:"+agentID).Result()
    if err != nil {
        log.Printf("redis error: %v", err)
        return false
    }
    return queueDepth < 80 // 80% of max
}
```

## Tool categories and what to look for

Rather than a list of specific versions, here is what each category needs to provide:

| Category               | Role                              | What to verify                          |
|------------------------|-----------------------------------|------------------------------------------|
| Shared state store     | Heartbeats, locks, debate rounds  | Atomic increments, TTL support, predictable latency under contention |
| Message broker         | Task routing, streams, retries    | Explicit retention limits, ack semantics, dead-letter support |
| Supervisor runtime     | Process lifecycle, restarts       | Concurrency primitives, graceful shutdown, structured logging |
| Agent runtime          | Async I/O, lightweight processes  | Non-blocking I/O, memory visibility, fast startup |
| Metrics and dashboards | Queue depth, latency, errors      | Histograms, not just averages; alerting on rate of change |

Two notes on tooling choices. A general-purpose distributed log is usually the wrong tool for inter-agent messaging: it adds commit latency and operational weight that a lightweight pub/sub system does not. And a binary RPC framework is often unnecessary for agent-to-agent calls where payloads are small and schemas are loose; JSON over a pub/sub transport is simpler to debug and usually fast enough.

## When this approach is the wrong choice

### 1. You need sub-10 ms latency

Multi-agent orchestration adds overhead from message routing, serialization, and shared-state lookups. If your P99 budget is under 10 ms, use a single process with in-memory queues or a monolith.

### 2. Your agents are stateful

Supervisor, swarm, and pipeline patterns assume agents are **stateless** or **ephemeral**. If agents must persist session state, put that state in a database or a stream with explicit durability guarantees, and keep the agents themselves disposable.

### 3. You are cost-constrained at high volume

Debate costs multiples of the other patterns because every round fans out to every agent and writes intermediate state. At high request rates, the difference between debate and supervisor is the difference between a viable budget and an unviable one. Do the arithmetic with your own token and compute prices before committing.

### 4. Your team does not know the implementation language

Building orchestration in a language the team does not maintain well creates technical debt that outlives the prototype. If the stack is JVM-based, use a JVM actor or workflow library rather than introducing Go or Rust for the orchestrator alone.

### 5. The workflow is a straight line

If the workflow is "call A, then B, then C", a pipeline pattern is overkill. A workflow engine or a simple script with retries is easier to operate.

## Decision checklist

Work through these in order. The first "yes" usually decides the pattern.

1. Is the task deterministic with a single correct answer? If yes, debate is the wrong tool.
2. Is the workflow a fixed sequence of stages? If yes, use a pipeline and invest in queue-depth monitoring.
3. Do agents need to share mutable state? If yes, you need a supervisor or an external state store; a pure swarm will not work.
4. Must the system survive the loss of any single node without coordination? If yes, a swarm is the only fit, and you must accept weaker debugging.
5. Is output quality more important than latency and cost? If yes, debate with a round cap and a tiebreaker.
6. None of the above? Start with a supervisor. It is the easiest to reason about and the easiest to instrument.

## FAQ

**Do pipelines need a supervisor?**
Yes, in practice. The supervisor's job in a pipeline is not to manage agents but to watch queue depth and restart saturated stages. Without it, a slow downstream stage turns into an out-of-memory kill.

**How many rounds should a debate run?**
Start with two. Most disagreements resolve after one critique round, and additional rounds add cost faster than they add agreement. Set a hard cap and return the best-scoring answer when the cap is hit.

**What is the right heartbeat interval?**
It depends on how fast you need to detect failure versus how much write load you can tolerate. A three-second interval with a three-strike rule detects failure in roughly nine seconds. Shorten the interval only if your shared store can absorb the write rate.

**Can you combine patterns?**
Yes, and most production systems do. A supervisor managing a set of pipeline stages is a common combination. The failure modes compose too: you inherit the supervisor's single point of failure and the pipeline's backpressure problems.

**How do you debug a swarm?**
Add correlation IDs to every message and log every routing decision with the presence data that informed it. Without that, a lost message is untraceable, because no component has a global view of what happened.

## The takeaway

The pattern matters less than the recovery properties around it. A supervisor with a restart budget and queue-depth health checks will outperform a swarm with none, even on workloads the swarm is theoretically better suited to. Build the instrumentation first — queue depth, latency histograms, retry counts — then pick the pattern that makes those numbers easy to interpret.

## Do this in the next 30 minutes

Pick your busiest multi-agent workflow and add one metric: queue depth at the point where tasks enter each stage. Emit it every second, and set an alert at 80% of the maximum queue size. If you do not have a maximum, choose one now. That single metric will surface backpressure problems before they become outages, and it is the cheapest change in this article.
