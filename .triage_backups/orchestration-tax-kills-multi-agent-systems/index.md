# Orchestration tax kills multi-agent systems

Multi-agent systems often fail for a reason that has nothing to do with the agents. The coordination layer — schedulers, retry managers, heartbeats, lock services, shared state stores — accumulates latency, memory, and failure modes that grow with the number of agents. This is the orchestration tax: work performed purely to keep agents from stepping on each other.

The tax is easy to miss in a prototype. A handful of agents coordinate cheaply, and the framework's defaults look fine. The problem appears when agent count rises and the coordination path becomes the hottest path in the system. A typical failure mode is coupling between agent logic and orchestration logic: an agent retries a failed call, the orchestration layer duplicates the work or double-counts the retry, and other agents block waiting on a stuck one.

This article separates agent logic from coordination logic, describes patterns that remove the coordination layer where possible, and explains how to measure the tax in your own system. It is not a framework ranking. The right question is not "which orchestrator is fastest" but "do these agents need to coordinate at all?"

## What the orchestration tax actually is

Orchestration tax has four observable components:

1. **Added latency.** Time spent waiting on locks, scheduler decisions, heartbeats, state serialization, or history writes before an agent can do useful work.
2. **Added memory.** Heap and RSS growth attributable to the coordination runtime rather than the agent's own state.
3. **Failure coupling.** The degree to which one agent's crash, stall, or retry storm propagates to others through the coordination layer.
4. **Configuration surface.** The amount of custom code and configuration required to keep agents from interfering with each other.

The tax is not a fixed constant. It scales with contention. A scheduler that adds 5 ms at 10 agents may add 80 ms at 500 agents if it serializes decisions or holds a global lock. That nonlinearity is why prototypes mislead.

The most useful mental model: every coordination mechanism is a shared resource, and shared resources are where multi-agent systems develop queues. If agents can proceed independently, the cheapest coordination layer is no layer.

## A worked example: where the latency comes from

Consider an illustrative system with these stated assumptions:

- 500 agents, each handling 4 tasks per second.
- Each task does 20 ms of real work.
- The coordination layer performs one serialized state write per task.
- That write takes 2 ms of service time and is served by a single-threaded component.

Total task rate is 500 × 4 = 2,000 tasks per second. The coordination component can serve 1 / 0.002 = 500 writes per second. Utilization is 2,000 / 500 = 4.0, which is above 1.0 — the queue is unstable and latency grows without bound. Even at 400 agents (1,600 tasks/s), utilization is 3.2 and the same conclusion holds.

To get utilization to a comfortable 0.7, the coordination component would need to serve 2,000 / 0.7 ≈ 2,857 writes per second, i.e. roughly 0.35 ms per write, or the writes would need to be sharded across about 6 independent instances. This arithmetic is the entire argument for removing the central coordinator: a component that looks trivial at low volume becomes the bottleneck at scale, and the fix is either sharding or elimination.

The same arithmetic applies to memory. If each agent holds 1 MB of coordination state (mailbox, supervision tree, history buffer), 500 agents consume 500 MB before any agent does work. If the coordination state is per-agent and bounded, the number is flat; if it grows with retry history, it grows without bound.

## Pattern 1: event-driven agents with idempotency keys

Each agent publishes immutable events to a log and consumes only events it has not processed. There is no central scheduler; agents coordinate through the stream. Correctness depends on idempotency keys.

```python
import redis.asyncio as redis

class IdempotentAgent:
    def __init__(self, agent_id: str, redis_url: str):
        self.agent_id = agent_id
        self.redis = redis.from_url(redis_url)
        self.ttl_ms = 300_000  # 5 minutes

    async def process(self, payload: dict) -> bool:
        key = f"idemp:{self.agent_id}:{payload['task_id']}"
        # SET NX is atomic: only the first caller wins.
        acquired = await self.redis.set(key, "1", px=self.ttl_ms, nx=True)
        if not acquired:
            return False  # duplicate; another worker owns it
        try:
            await self._do_work(payload)
        except Exception:
            # Release the key so a retry can proceed.
            await self.redis.delete(key)
            raise
        return True
```

Design notes that matter in practice:

- The key must be derived from the task identity, not from a timestamp alone. Keys that embed a coarse timestamp window collide when retries run longer than the window; keys that embed only a task ID never expire and grow the keyspace forever.
- The TTL must exceed the maximum expected processing time. If work can take 10 minutes, a 5-minute TTL allows a second worker to start while the first is still running.
- Releasing the key on failure is what makes retries possible. Without it, a transient failure permanently poisons the task.

**Strength:** no coordination runtime, so latency stays near the agent's native latency. **Weakness:** correctness depends entirely on key design, and the system must tolerate at-least-once delivery. **Best fit:** independent tasks where eventual consistency is acceptable.

## Pattern 2: work-stealing queues with agent-local state

A single queue holds tasks; agents pull work when idle. No central scheduler, no heartbeats.

```python
import redis.asyncio as redis

async def worker(queue_key: str, agent_id: str, redis_url: str):
    r = redis.from_url(redis_url)
    processing_key = f"processing:{agent_id}"
    while True:
        # BRPOPLPUSH is atomic: exactly one worker receives each task.
        task = await r.brpoplpush(queue_key, processing_key, timeout=10)
        if task is None:
            continue
        try:
            await process_task(task)
        finally:
            await r.lrem(processing_key, 1, task)
```

The `processing:{agent_id}` list doubles as a recovery record: if an agent dies, another process can scan stale processing lists and requeue their contents.

**Strength:** scales roughly linearly with agent count because the queue is the only shared resource. **Weakness:** uneven task sizes cause starvation unless you use weighted pull rates or separate queues per task class. **Best fit:** high-throughput systems with variable task size.

## Pattern 3: CRDT-based agents with local-first sync

Agents keep local copies of shared state and converge via conflict-free replicated data types, typically synchronized with a gossip protocol. No coordinator decides the final value; the merge function does.

```python
# Illustrative pseudocode for a grow-only counter stored as JSON.
# Each agent increments its own slot; reads sum the slots.
import redis.asyncio as redis

async def increment(r: redis.Redis, agent_id: str):
    await r.hincrby("counter", agent_id, 1)

async def read_total(r: redis.Redis) -> int:
    slots = await r.hgetall("counter")
    return sum(int(v) for v in slots.values())
```

A grow-only counter is the simplest CRDT and the easiest to reason about: merges are commutative and idempotent, so replaying an update is harmless. More complex CRDTs (maps, sequences, sets) have more expensive merge logic and larger state.

**Strength:** survives partitions and agent restarts without a coordinator. **Weakness:** state grows with the number of writers and the number of operations unless you prune. **Best fit:** shared state that must converge despite unreliable networks.

## Pattern 4: lightweight actors on the existing event loop

Each agent is a coroutine with a mailbox. There is no separate runtime scheduler; the language's event loop is the scheduler.

```python
import asyncio
from dataclasses import dataclass

@dataclass
class Message:
    sender: str
    payload: dict

class Agent:
    def __init__(self, name: str, max_pending: int = 100):
        self.name = name
        self.mailbox = asyncio.Queue(maxsize=max_pending)

    async def run(self):
        while True:
            msg = await self.mailbox.get()
            try:
                await self.handle(msg)
            except Exception as exc:
                print(f"{self.name} failed on {msg.payload}: {exc}")
            finally:
                self.mailbox.task_done()

    async def handle(self, msg: Message):
        await asyncio.sleep(0.1)  # stand-in for real work

async def main():
    agents = [Agent(f"agent-{i}") for i in range(100)]
    tasks = [asyncio.create_task(a.run()) for a in agents]
    await agents[0].mailbox.put(Message(sender="client", payload={"task": 1}))
    await asyncio.sleep(1)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main())
```

The bounded `maxsize` is the important detail. Without it, a slow agent's mailbox grows until the process runs out of memory. With it, producers block or drop, which is backpressure — and backpressure is the mechanism that prevents one slow agent from consuming the whole heap.

**Strength:** very low per-agent memory because there is no supervision tree, no history buffer, and no serialization boundary. **Weakness:** no built-in supervision or distribution; a crash takes down its mailbox unless you wrap it. **Best fit:** teams already running an async runtime who want minimal coordination overhead.

## Pattern 5: FaaS with durable execution

Each agent runs as a short-lived function, with a durable workflow engine providing retries and timeouts.

```yaml
# Illustrative Step Functions Express workflow.
StartAt: AgentTask
States:
  AgentTask:
    Type: Task
    Resource: arn:aws:lambda:us-east-1:123456789012:function:agent-worker
    TimeoutSeconds: 30
    Retry:
      - ErrorEquals: ["States.ALL"]
        IntervalSeconds: 1
        MaxAttempts: 3
    End: true
```

**Strength:** no long-running scheduler process to operate; the platform handles placement. **Weakness:** cold starts add jitter, and the workflow engine serializes state and writes execution history, which is itself orchestration tax. For very short agents, the tax can exceed the work. **Best fit:** bursty workloads where agents are short-lived and idempotent.

## How to measure orchestration tax in your own system

Do not trust published numbers, including any in this article. Measure.

1. **Build a baseline with no coordination.** Run N agents that each perform the same fixed unit of work and record p50, p95, and p99 latency, plus RSS and CPU per agent. Use a fixed workload so runs are comparable.
2. **Add exactly one coordination mechanism.** Scheduler, retry manager, or shared state store — one at a time. Re-run the identical workload.
3. **Subtract.** The difference in p99 latency and per-agent memory is the tax for that mechanism. Run at three agent counts (e.g. 10, 100, 500) to see whether the tax is flat or growing.
4. **Instrument the coordination path specifically.** Count lock acquisitions, serialization calls, history writes, and scheduler decisions per task. These counters explain the latency difference; raw latency alone does not.
5. **Stress the shared resource.** Push task rate above the coordination component's service rate and confirm the queue grows as predicted. This is the failure mode that only appears at scale.

Useful commands and tools: `py-spy top` or `py-spy record` for Python CPU attribution, `perf` for native hotspots, and a load generator that reports p99 rather than mean. Record the exact agent count, task rate, and machine type with every measurement, because the tax is a function of all three.

## Decision checklist

Work through these in order:

- **Can agents proceed independently?** If yes, remove the coordination layer and use idempotency keys. This is the highest-leverage decision.
- **Is task identity stable across retries?** If not, fix that before adding any orchestration. No coordinator can deduplicate work it cannot name.
- **Is the shared resource's service rate above the aggregate task rate?** If not, shard it or eliminate it. Check with the utilization arithmetic shown above.
- **Does any agent's failure need to stop another agent?** If yes, you need real coordination. If no, failure isolation is cheaper than coordination.
- **Do you need strict ordering or compensating transactions?** If yes, a durable workflow engine is justified. Accept the tax and budget for it.
- **Is per-agent memory bounded?** If retry history or mailboxes grow without limit, add a cap before scaling agent count.

## Comparison of coordination approaches

The table below compares the patterns by the properties that matter. Figures are qualitative because the tax depends on workload, agent count, and implementation.

| Approach | Coordination model | Failure isolation | Per-agent memory | Operational cost |
|---|---|---|---|---|
| Event-driven + idempotency | None (log) | High | Low, bounded by key TTL | Requires a durable log |
| Work-stealing queue | Single shared queue | High | Low | Requires a reliable queue |
| CRDT local-first | Merge function | High under partition | Grows with writers/ops | Requires pruning policy |
| Lightweight actors | Event loop | Medium (per-mailbox) | Very low | None beyond the runtime |
| FaaS + durable workflow | Platform workflow engine | Medium | Very low while idle | Cold-start jitter, history writes |
| Durable workflow engine (self-hosted) | Central history store | Medium | High per worker | Significant; needs a cluster |

The general rule: as you move down the table, coordination guarantees increase and so does the tax. Choose the lowest row that satisfies your correctness requirements.

## Common failure modes

- **Timestamp-only idempotency keys.** Collide when retries run longer than the timestamp window, causing duplicate work. Use task identity plus a window, and set the TTL above the maximum processing time.
- **Unbounded mailboxes.** A single slow agent accumulates messages until the process exhausts memory. Always bound the queue and define the backpressure behavior.
- **Unbounded CRDT state.** Merge metadata grows with the number of writers and operations. Prune or snapshot on a schedule.
- **Retry storms.** A failing dependency triggers synchronized retries across all agents, which is itself a coordination load. Add jitter and a circuit breaker.
- **Coordination state as a single point of failure.** If the scheduler or lock service is down, no agent can proceed. Prefer designs where coordination failure degrades throughput rather than halting it.
- **Measuring mean instead of tail.** Coordination tax shows up in p99 first. Mean latency hides it until the system is already in trouble.

## FAQ

**How do I prevent duplicate work with idempotency keys?**
Derive the key from the task identity, not from wall-clock time alone. Store it with a TTL longer than the maximum processing time, and release it on failure so retries can proceed. If retries can exceed the TTL, use a longer TTL or a lease with explicit renewal.

**What if two agents pick the same task from a queue?**
Atomic pop-and-move operations (such as `BRPOPLPUSH` in Redis) ensure only one agent receives a given task. The processing list records ownership so a crashed agent's tasks can be requeued.

**Can CRDTs handle state that grows without bound?**
No. Merge metadata grows with writers and operations. Set a pruning policy — TTLs, snapshots, or a sliding window — before the state becomes the dominant memory consumer.

**Why does FaaS still incur orchestration tax?**
The workflow engine serializes state and writes execution history for every step. Cold starts add jitter. For agents that run for only a few seconds, that overhead can exceed the useful work.

**What is the simplest way to test orchestration tax?**
Run N agents doing fixed work with no coordination and record p99 and RSS. Add one coordination mechanism and repeat. The difference is the tax. Repeat at 10, 100, and 500 agents to see how it scales.

## Next step

Open your agent task handler and make the work idempotent: derive a key from the task identity, store it with a TTL longer than the maximum processing time, and release it on failure. Then run the same workload at 10, 100, and 500 agents with and without your current coordination layer, and compare p99 latency and per-agent RSS. If the tax grows with agent count, you have found the bottleneck — and the fix is to remove or shard the shared resource, not to tune the agents.
