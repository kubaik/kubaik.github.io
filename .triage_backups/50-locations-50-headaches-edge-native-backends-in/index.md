# 50 locations, 50 headaches: edge-native backends in

The conventional advice on edge-native backends is incomplete. It holds in the simple case and breaks in a specific way under load. This article describes the fuller picture: what changes when an API runs from many points of presence (POPs), which patterns bound staleness, and how to measure the things that actually go wrong.

## The one-paragraph version

Running an API from 50 edge locations is not horizontal scaling — it is geographic scaling, and it changes latency, consistency, cost, and debugging at the same time. The tooling has matured enough that pushing code to many POPs no longer requires rewriting an application. The trap is reusing a single-region mental model. Replication lag, eventual consistency, and cold-start costs do not scale the way instance count does, because state is now distributed across a network you do not control.

## Why this concept confuses people

Most developers learn to scale by adding instances inside one region. Sharding, keeping a single source of truth, and reading one cloud bill are all well-understood. Edge-native inverts that: every POP is both a cache and a potential source of truth, and the network becomes part of the data layer. Teams get stuck on three questions:

1. **Where is my data?** It may be in Singapore, but only if the last write landed there. Otherwise a reader in São Paulo sees a stale value.
2. **Why did the bill grow faster than traffic?** Edge function invocations are priced differently from regional ones, and replication writes multiply egress by the number of replicas.
3. **How do you debug a race condition that only reproduces in one POP under specific packet loss?**

The confusion is not technical detail — it is a shift from "scale up" to "scale out, everywhere."

## The mental model that makes it click

Think of the edge as a globally distributed CDN with compute attached. Each POP is a small data center that can run your API, but it is not always in sync with the others. The key abstraction is **eventual consistency with bounded staleness**.

- **Reads** can be served from the nearest POP, but they may return stale data.
- **Writes** fan out to replicas, but only become durable once the authoritative copy accepts them.

In practice, three patterns cover most systems:

| Pattern | Staleness bound | Cost profile | Best for |
|---|---|---|---|
| Active-active with CRDTs | Low (sub-second) | High — every replica merges | Collaborative editing, game state |
| Leader-based writes with read-through cache | Bounded by cache TTL | Medium | User profiles, inventory, sessions |
| Read-through cache with TTL only | Bounded by TTL | Low | Product catalogs, static content |

A common failure mode is choosing leader election per POP. Under high packet loss, a consensus group can lose and regain leadership repeatedly, so clients in a distant region observe their state "resetting" to whichever leader last won. The usual fix is to pin the leader to a single home region and let every POP read from it, caching locally with a short TTL. Staleness then becomes a number you chose rather than a number you discovered.

## A concrete worked example

Build a global counter that increments a value and returns it. The example uses a Node HTTP handler and a Postgres connection pool; the deployment target is a platform that runs the app in multiple regions with a regional Postgres primary.

### Step 1: App skeleton

```javascript
// src/index.js
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

export default async function handler(req) {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    const { rows } = await client.query(
      'SELECT value FROM global_counter WHERE id = 1 FOR UPDATE'
    );
    const next = rows[0].value + 1;
    await client.query('UPDATE global_counter SET value = $1 WHERE id = 1', [next]);
    await client.query('COMMIT');
    return new Response(String(next));
  } catch (err) {
    await client.query('ROLLBACK');
    throw err;
  } finally {
    client.release();
  }
}
```

Note the explicit transaction. The original version issued `FOR UPDATE` and the update as two separate statements outside a transaction, which releases the row lock between them and allows lost updates. Wrapping both in `BEGIN`/`COMMIT` is what makes the increment correct.

### Step 2: Configuration

```toml
# fly.toml
app = "global-counter"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[[services]]
  protocol = "tcp"
  internal_port = 3000

  [[services.ports]]
    port = 80
    handlers = ["http"]

[metrics]
  port = 9090
  path = "/metrics"
```

### Step 3: Deploy to many regions

```bash
flyctl deploy --config fly.toml --strategy rolling
flyctl scale count 50 --process-group app
```

### Step 4: Measure, do not guess

The point of this step is to produce numbers you can defend. Before changing anything, record:

- **P50/P95/P99 latency per POP**, not globally. A global average hides the slowest region.
- **Replication lag** between the primary and each read replica, sampled every few seconds.
- **Write throughput** on the primary, plus its network egress.
- **Row lock wait time**, which shows up as autovacuum contention on hot rows.

A load generator such as `vegeta` or `k6` can drive the traffic:

```bash
vegeta attack -rate 100 -duration 30s -targets targets.txt | vegeta report
```

Then compare the per-POP P99 against the replication lag series. If P99 in a distant region tracks replication lag, the bottleneck is consistency, not compute. If P99 tracks row-lock wait, the bottleneck is the write path itself.

### Step 5: Apply the fix and re-measure

Switch to leader-based writes in one home region, and let every POP read through a local cache with a 2-second TTL. Then re-run the same load test and the same lag sampling. The claim to verify is narrow: does P99 in the distant region now track the cache TTL rather than the replication lag? If yes, staleness is bounded. If no, the cache is not absorbing reads and the problem is elsewhere.

## How this connects to things you already know

If you have used Redis Cluster, the sharding-plus-leader idea will feel familiar. Edge-native extends it to every POP. The differences matter:

- **No single control plane.** Redis Cluster has one CLI that can reshard. Edge POPs are largely independent and are usually managed through GitOps or a provider control plane.
- **Network partitions are normal.** In one region, the control plane is assumed reachable. Across many POPs, a partition in one location is a routine condition to handle, not an outage.
- **Cold starts and egress cost money.** A regional function is cheap per invocation; an edge function is priced higher. But the larger cost is replication writes: a 1 KB write fanned out to 50 replicas is 50 KB of egress per write. At 1,000 writes/sec, that is 50 MB/s of egress, or roughly 4.3 TB/day. Multiply by the provider's per-GB egress rate to see whether fan-out-to-all is viable for your write volume.

A common mistake is putting an edge function in front of a regional Redis Cluster and assuming the bill stays flat. It does not, because every write now crosses the network once per replica. Leader-based writes with a local read cache cut that traffic to one write per replica-set rather than one per POP.

## Common misconceptions, corrected

1. **"Edge functions are always cheaper."** False. They are cheap for simple, read-mostly logic. If writes fan out to every POP, you pay bandwidth and CPU in every region.
2. **"Eventual consistency is fine for everything."** Not when users observe the staleness. In a shopping cart, stale data means overselling. In a banking app, it means double spends. Bounded staleness — a TTL or a lag ceiling you enforce — is the minimum bar.
3. **"Every POP is a mini-region."** It is not. POPs have limited CPU, memory, and disk. A small edge isolate is not equivalent to a general-purpose VM, and memory-heavy workloads such as a full-text index will fail fast.
4. **"Debugging is the same everywhere."** It is not. Packet capture, host metrics, and tracing are exposed differently by each provider, and the tooling changes faster than regional equivalents.
5. **"One database everywhere is fine."** It can be, until the primary's network link saturates. A single primary replicating to 50 regions has a hard ceiling set by its uplink. Beyond that, you need sharding or a multi-region database with its own consensus layer.

## The advanced version

Once bounded staleness works, the next problems are multi-region transactions and global rollbacks.

### Multi-region transactions

Use a saga with compensating actions. Each write records its intent and its inverse. If a step fails, a coordinator replays the compensating actions in reverse.

```javascript
// saga.js
import { Pool } from 'pg';
const pool = new Pool({ connectionString: process.env.SAGA_DB });

async function runSaga(steps) {
  const client = await pool.connect();
  const completed = [];
  try {
    await client.query('BEGIN');
    for (const step of steps) {
      await step.execute(client);
      completed.push(step);
    }
    await client.query('COMMIT');
  } catch (err) {
    await client.query('ROLLBACK');
    for (const step of completed.reverse()) {
      if (step.compensate) await step.compensate(client);
    }
    throw err;
  } finally {
    client.release();
  }
}
```

Two details matter. First, only steps that actually completed should be compensated — the original version reversed the entire list, including steps that never ran. Second, compensation itself must be idempotent, because the coordinator may retry after a crash.

### Global rollbacks

A schema or logic change pushed to every region needs a reverse path. The standard approach is a change-data-capture stream from each region into a home-region log, so a rollback service can replay changes in reverse order. The measurement that matters is **time to rollback**: how long from detection to every region running the previous version. Instrument the detection timestamp and the last-region-confirmed timestamp; the difference is your real rollback time, and it is usually dominated by detection, not propagation.

### Cost control

Use these levers deliberately, and measure each one before and after:

| Lever | Latency effect | Bill effect | When to use |
|---|---|---|---|
| Cache reads at POP with short TTL | Small increase | Lower egress and compute | Catalogs, static content |
| Fan out writes to a small subset of POPs | Higher for distant writers | Much lower egress | Profiles, session state |
| CRDTs for small, hot state | Low | Higher — merge cost | Collaborative apps |
| Leader-based writes with read-through cache | Bounded by TTL | Lower | Inventory, user data |

The trade-off is explicit: you are choosing a staleness ceiling in exchange for lower cost and predictable latency. Pick the ceiling first, then pick the pattern.

## Quick reference

- **Latency:** Expect P95 in the low hundreds of milliseconds and P99 up to roughly a second when staleness is bounded by a TTL. Measure per POP.
- **Cost:** Driven mostly by egress from write fan-out, not by invocation count. Compute egress as (write size) × (replica count) × (writes/sec).
- **Throughput:** A single primary has a hard ceiling set by its uplink. Shard when you approach it.
- **Staleness:** Choose a ceiling — sub-second for CRDTs, cache-TTL for leader-based reads, seconds for plain caching.
- **Debugging:** Start at the leader. If the leader is healthy, compare per-POP P99 against replication lag and row-lock wait.
- **Pattern:** Leader-based writes plus a read-through cache with a TTL is the simplest correct default.
- **Fallback:** Route a failed POP to the next closest one, and measure the failover time rather than assuming it.

## Frequently asked questions

**What surprises teams most about going edge-native?**
Latency usually drops, but replication lag is rarely planned for. A reader in one region may see a value written in another region a second earlier. The design implication is to choose a staleness bound rather than fight for strong consistency everywhere.

**How many POPs should receive writes?**
Start with the smallest set that covers your write-heavy users, measure the latency penalty and the egress cost, and expand only if the numbers justify it. Fan-out-to-all is rarely the cheapest option.

**Is Postgres the only source of truth?**
No. Any database with a documented replication model works, but the model must be understood. A cache is not a source of truth for writes, regardless of how fast it is.

**How do you debug a race condition in one POP?**
Start at the leader. If the leader is clean, capture packets at the affected POP and look for retransmits, buffer sizes, and connection churn. Correlate those with the application's lock waits.

**What do teams get wrong most often?**
Treating every POP as a mini-region. POPs are resource-constrained and have higher latency to any control plane. Design for bounded staleness instead of assuming strong consistency is available.

## Close the gap in 30 minutes

Open your API's read path and add a one-second TTL cache in front of the most frequently read query, then measure two things before and after: P95 latency for that endpoint, and egress bytes per minute. If latency drops and egress stays flat, the cache is doing useful work and you have a measured starting point for the rest of the design.
