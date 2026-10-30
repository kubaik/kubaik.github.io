# Survive network drops: agentic workflows that keep running

The conventional advice on agentic workflows — retry with backoff, use a queue, wrap it in a transaction — works in the simple case and breaks in a specific way under load. This article explains the failure mode and the pattern that avoids it.

## The one-paragraph version

Teams operating where connectivity is intermittent lose days debugging workflows that hang forever when a single hop fails. The patterns that survive partitions rest on one idea: **treat every agent as a state machine with an append-only log**, store that log durably, and replay from the last known good state when connectivity returns. This keeps billing reminders, delivery confirmations, and multi-party approvals running across long outages. It is not about sophisticated AI agents — it is about durable execution, idempotent commands, and deterministic retries. If a workflow cannot survive a tower outage, it will not survive a 500ms latency spike in a cloud region either.

## Why this concept confuses people

Most engineers start with the wrong mental model: they treat agents like stateless microservices that retry on failure. That leads to backoff storms, duplicate invoices, and database contention when the network hiccups.

A typical failure mode looks like this. A REST retry loop fires on every timeout. A fiber cut or tower outage lasts an hour. When the link returns, every pending retry fires at once against a database that was sized for steady-state traffic. The result is duplicate side effects — double disbursements, double SMS charges — and a reconciliation problem that costs more to clean up than the outage itself. The lesson: retries alone are not enough. You need **deterministic replay** from a durable log.

A second trap is over-engineering. Distributed stream processors and managed workflow services are excellent for high-throughput pipelines, but they add per-hop latency and a fixed monthly cost that is hard to justify for a workflow processing a few hundred events per day. A team can spend a week trying to fit a streaming platform into a workload that only needed a table and a loop.

A third source of confusion is conflating two different problems: **network partitions** (temporary loss of connectivity) and **permanent failures** (a host is gone for good). The correct responses differ. If an agent cannot distinguish a flaky link from a dead instance, it will misbehave under real conditions: it will retry forever against a host that will never come back, or it will give up on a command that would have succeeded thirty seconds later.

## The mental model that makes it click

Think of the agent as a **deterministic state machine** with three layers:

1. **Command log** — append-only, durable storage
2. **Executor** — applies commands idempotently
3. **Supervisor** — monitors progress and retries deterministically

The log is the source of truth. Every command — "send SMS to user 42 for approval" — is appended with a monotonically increasing sequence number. The executor reads the log and applies each command at most once. When connectivity returns, the supervisor resumes from the last committed sequence, skipping already-applied commands. This is the same pattern log-structured systems use internally, but it can be implemented with a local embedded database and a polling loop.

Analogy: a bank teller with a carbon-copy ledger. When the power flickers, the teller closes the ledger, waits, then picks up where they left off — no double entries, no missing transactions. That is durable execution.

The key insight: **idempotency keys alone are not enough**. If the key is a random UUID generated per attempt, two retries produce two different keys and both effects land. The key must be derived from the log position — the sequence number — so that a retry of sequence N is indistinguishable from the original attempt at sequence N.

## A concrete worked example

A minimal approval workflow for a delivery operation. Requirements:

- Accept a delivery request
- Send an SMS to the driver for approval
- If approved, update the delivery status in PostgreSQL
- Survive intermittent mobile data, SMS gateway timeouts, and occasional power loss

Stack: Python 3.11, SQLite (WAL mode), Redis for pub/sub coordination.

### Step 1: The command log

```python
# log.py
import sqlite3
import pickle
from dataclasses import dataclass

@dataclass
class Command:
    seq: int
    name: str
    payload: dict

class CommandLog:
    def __init__(self, path="commands.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commands (
                seq INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                payload BLOB NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        self.conn.commit()

    def append(self, cmd: Command):
        self.conn.execute(
            "INSERT INTO commands (seq, name, payload, status) VALUES (?, ?, ?, ?)",
            (cmd.seq, cmd.name, pickle.dumps(cmd.payload), "pending"),
        )
        self.conn.commit()

    def pending_upto(self, seq: int):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT seq, name, payload FROM commands "
            "WHERE seq <= ? AND status IN ('pending', 'failed') ORDER BY seq",
            (seq,),
        )
        return [Command(r[0], r[1], pickle.loads(r[2])) for r in cur.fetchall()]

    def mark(self, seq: int, status: str):
        self.conn.execute("UPDATE commands SET status = ? WHERE seq = ?", (status, seq))
        self.conn.commit()
```

Two details matter here. First, the primary key on `seq` makes duplicate appends impossible: a repeated insert with the same sequence number fails loudly rather than silently duplicating. Second, `synchronous=NORMAL` under WAL is a deliberate durability trade-off — it survives process crashes and most power loss, but a hard power cut can lose the last few committed transactions. If that is unacceptable, use `synchronous=FULL` and accept the write latency.

### Step 2: The executor

```python
# executor.py
import subprocess
import time
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

class Executor:
    def __init__(self, log):
        self.log = log

    def run(self):
        max_seq = self.log.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM commands"
        ).fetchone()[0]
        for cmd in self.log.pending_upto(max_seq):
            try:
                if cmd.name == "send_sms":
                    result = subprocess.run(
                        [
                            "curl", "-m", "30",
                            "https://sms-gateway.example/api/send",
                            "-d", f"phone={cmd.payload['phone']}",
                            "-d", f"msg={cmd.payload['msg']}",
                            "-d", f"idem={cmd.seq}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=35,
                    )
                    if result.returncode == 0:
                        self.log.mark(cmd.seq, "done")
                        r.publish("approvals", f"sent:{cmd.payload['delivery_id']}")
                    else:
                        self.log.mark(cmd.seq, "failed")
            except Exception:
                self.log.mark(cmd.seq, "failed")
                time.sleep(1)
```

The `idem={cmd.seq}` parameter is the crucial line. A well-behaved SMS gateway that honours an idempotency token will collapse duplicate submissions of the same sequence number into a single delivered message. If the gateway does not support idempotency tokens, the sequence number still gives you a stable key you can deduplicate against in your own records after the fact.

### Step 3: The supervisor

```python
# supervisor.py
import time
from log import CommandLog
from executor import Executor

class Supervisor:
    def __init__(self):
        self.log = CommandLog()
        self.executor = Executor(self.log)
        self.last_seq = 0

    def poll(self):
        while True:
            max_seq = self.log.conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM commands"
            ).fetchone()[0]
            if max_seq > self.last_seq:
                self.executor.run()
                self.last_seq = max_seq
            time.sleep(2)

if __name__ == "__main__":
    Supervisor().poll()
```

### How it survives a partition

1. A command is appended with `seq=1` and status `pending`.
2. The supervisor picks it up and attempts the SMS send.
3. The link drops mid-request. The subprocess times out after 35 seconds.
4. The executor marks the command `failed` and commits.
5. The supervisor sleeps, then polls again. It sees `seq=1` as `failed` and retries.
6. When connectivity returns, the send succeeds and the command is marked `done`.

No duplicate side effects, no lost state. Note that step 5 retries the *same* sequence number, which is what makes the retry safe: the downstream system sees the same idempotency token it saw before.

## Measuring whether the pattern is actually working

Claims about durability are only meaningful if they are measured. Instrument these four counters in the supervisor and expose them on a `/metrics` endpoint:

- `commands_appended_total` — should equal the number of user-initiated actions
- `commands_applied_total` — should equal `commands_appended_total` once the backlog drains
- `commands_failed_total` — a rising value with a flat `applied` value means a persistent downstream fault, not a transient one
- `replay_lag_seconds` — the age of the oldest pending command

The invariant to alert on is `commands_applied_total - commands_appended_total`. It should be zero at steady state and should return to zero within one outage window after connectivity is restored. If it grows monotonically, the executor is not making progress; if it goes negative, you have a duplicate-application bug and the idempotency key is not doing its job.

For the SMS gateway specifically, compare your `commands_applied_total` against the gateway's own delivery report count over the same window. A mismatch in either direction is the signal you need: more deliveries than commands means duplicates, fewer means silent drops.

## How this connects to things you already know

If you have used consumer groups in a log-based broker, you have already used durable execution. The broker's log is an append-only command store, and the consumer offset is the sequence number. The difference is scale and operational cost: a broker cluster is the right tool when you need many consumers, retention policies, and cross-team fan-out. A single-node embedded database is the right tool when you have one writer and a few hundred commands a day.

If you have built a task queue pipeline with retries, you have fought the same problem. A task ID is an idempotency key, but it is usually generated per attempt rather than per logical action. Two workers can therefore pick up the same logical action under different IDs and both succeed, creating duplicates. Late acknowledgement helps with at-least-once delivery, but it does not give you deterministic replay from a clean state.

If you have used a managed state-machine service, you have used a state machine. Its execution history is an append-only log. The trade-off is cost per state transition and vendor coupling versus the operational burden of running your own log. For low-volume workflows, the per-transition pricing can dominate the bill; for high-volume ones, it is often cheaper than the engineering time to build and operate the equivalent.

The pattern also applies to frontend state. A Redux-style reducer is the executor and the store is the log. When the browser reloads, the reducer replays from the last committed state. That is durable execution in the small.

## Common misconceptions, corrected

**Misconception 1: "Just use exponential backoff."**
Backoff reduces load but does not solve idempotency. If two agents retry the same logical command, both can succeed. Backoff also has a synchronisation failure mode: retries that start at the same time and double at the same rate stay aligned, so they arrive in bursts rather than spreading out. Adding jitter helps with the burst; only sequence-based identity fixes the duplication.

**Misconception 2: "A message queue solves this."**
Queues are excellent for decoupling producers from consumers. They do not by themselves give you deterministic replay. If a consumer crashes after taking a message but before acknowledging it, the message is redelivered — which is correct at-least-once behaviour, but it means your handler must be idempotent anyway. The command log makes the same guarantee explicit and gives you a total order to replay from.

**Misconception 3: "A database transaction makes it atomic."**
Transactions give you atomicity within a database. They do not give you durability across a partition between your process and that database. If the connection drops mid-transaction, the transaction rolls back and the intent is lost unless it was recorded somewhere durable first. The command log is that somewhere.

**Misconception 4: "Serverless is simpler."**
Serverless removes host management but introduces cold starts, per-invocation timeouts, and egress costs that are easy to underestimate. For a workflow that runs continuously and polls a local database, a small always-on instance is often both cheaper and easier to reason about. Serverless is the better choice when traffic is genuinely bursty and you can tolerate cold-start latency.

## The advanced version

Once the command log is reliable, these layers add capability without changing the core model.

### 1. Event sourcing with snapshots

Instead of replaying every command from the beginning, store a snapshot of derived state every N commands and replay only from the snapshot. This bounds restart time as the log grows. The trade-off is snapshot storage and the complexity of keeping the snapshot format compatible across code changes. A common approach is to snapshot every few thousand commands and to keep the snapshot schema versioned alongside the code.

### 2. Distributed supervisors

For high availability, run multiple supervisors and elect a single leader that appends to the log. Followers replay from the log and take over if the leader fails. Consensus systems such as etcd or Consul provide the leader election primitive. The cost is operational: you now run a quorum of nodes and must reason about split-brain scenarios. This is worth it when the workflow is business-critical and a single node is a single point of failure; it is not worth it for a workflow where a few minutes of downtime is acceptable.

### 3. Rate limiting with token buckets

Instead of a fixed sleep between retries, use a token bucket so retries are spaced to match the downstream rate limit. This keeps the workflow within gateway limits without hard-coding a sleep interval that is either too slow or too fast.

```python
import time
from redis import Redis

r = Redis(host="localhost", port=6379, db=0)

def throttle(key, max_tokens, refill_sec):
    now = time.time()
    tokens = float(r.get(key) or max_tokens)
    last_refill = float(r.get(f"{key}:ts") or now)
    elapsed = now - last_refill
    tokens = min(max_tokens, tokens + elapsed / refill_sec)
    if tokens >= 1:
        r.set(key, tokens - 1)
        r.set(f"{key}:ts", now)
        return True
    return False
```

Note the fix from the naive version: `tokens` must be a float and the refill must be proportional to elapsed time, otherwise the bucket either never refills or refills in whole-token jumps that do not match the intended rate.

### 4. Dead letter queue for poison commands

If a command keeps failing — an invalid phone number, a malformed payload — retrying forever wastes capacity and hides the bug. After a bounded number of attempts, move the command to a separate table or stream and alert. The supervisor should skip dead-lettered sequences on subsequent polls.

### 5. Observability

Expose the counters from the measurement section above and alert on `replay_lag_seconds` exceeding a threshold that reflects your acceptable staleness. A lag that climbs steadily is the earliest signal that the executor is wedged, well before users notice missing approvals.

## Choosing an approach

| Approach | Fits when | Ordering guarantee | Operational burden |
|---|---|---|---|
| Command log + executor | One writer, hundreds to low thousands of commands/day | Total order via sequence number | Low: one embedded database |
| Log-based broker | Many consumers, high throughput, cross-team fan-out | Per-partition total order | High: cluster to run and tune |
| Managed state machine | Low-volume workflows already in that cloud | Per-execution total order | Low, but per-transition cost |
| Task queue with retries | Background jobs where at-least-once is acceptable | None by default | Medium |
| Consensus-backed supervisors | Business-critical, multi-node availability required | Total order via leader | High: quorum and failover |

The decision hinges on two questions. How many commands per day? If the answer is in the hundreds, an embedded database plus a polling loop is almost always the cheapest correct answer. How bad is a duplicate? If a duplicate disbursement is a compliance incident, the sequence-number identity is not optional regardless of which transport you choose.

## FAQ

**How do you prevent duplicate approvals when the network drops and retries fire?**
Derive the idempotency key from the log sequence number rather than generating a fresh one per attempt. Append the command once. On retry, re-send the same sequence number. Downstream systems that honour the token collapse the duplicates; systems that do not can be deduplicated against the log after the fact.

**What is the simplest durable execution pattern for a small VM?**
An embedded database in WAL mode for the command log, a short polling loop for the supervisor, and a single executor process. Keep the command handler idempotent and derive keys from sequence numbers. This is a few hundred lines of code and one process to operate.

**Why not just use a message queue?**
A queue solves decoupling and delivery, not identity. Redelivery after a crash is expected behaviour, so the handler must still be idempotent. A command log gives you the same at-least-once delivery plus a total order to replay from, which makes recovery reasoning much simpler.

**How do you handle power loss?**
Use WAL mode with `synchronous=NORMAL` for a balance of durability and write latency, or `FULL` if losing the last few transactions is unacceptable. Snapshot derived state periodically to separate storage so a disk failure does not require replaying the entire history. A short-lived UPS reduces the frequency of hard cuts but does not remove the need for the durability setting to be correct.

## Do this in the next 30 minutes

Create the log, append one command, run the supervisor, kill it mid-flight, and restart it. Then verify two things: the command was applied exactly once, and the sequence number in the log matches the idempotency token your downstream service received. That single test tells you whether your retry path is deterministic or whether it is quietly duplicating work.

```bash
mkdir agentic-workflow && cd agentic-workflow
python -m venv venv && source venv/bin/activate
pip install redis prometheus-client
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

Paste the three files above into the directory, start Redis, and run `python supervisor.py`. In a second terminal:

```python
from log import CommandLog, Command
log = CommandLog()
log.append(Command(seq=1, name="send_sms",
                   payload={"phone": "+254712345678", "msg": "Approve? Reply YES"}))
```

Watch the supervisor process it. Interrupt the supervisor with Ctrl+C while the send is in flight, restart it, and confirm the command still completes exactly once. That is durable execution in ten minutes, with no cluster and no managed service.
