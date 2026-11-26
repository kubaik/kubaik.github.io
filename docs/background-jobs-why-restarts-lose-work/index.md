# Background jobs: why restarts lose work

A background job that "runs fine" in development and "occasionally loses work" in production is almost never a bug in the job's business logic. It is a bug in the contract between the job and the process that hosts it. The job assumes it will be allowed to finish. The process assumes it can be killed at any moment. Only one of those assumptions is true.

This post is about that contract. It explains why restarts — deploys, OOM kills, autoscaling, node drains, spot reclaims — silently eat work, and what a job design has to look like to survive them. The real problem it solves: you cannot tell from reading most job code whether a restart will duplicate a side effect, drop a task, or corrupt state, because the code was written as if the process were immortal.

## The confusion: "the queue is durable, so the job is durable"

Most teams reach for a durable queue — SQS, RabbitMQ, Redis with persistence, a database-backed table — and conclude the problem is solved. It is not. The queue's durability only guarantees that a message survives. It says nothing about what happens between the moment a worker receives the message and the moment the side effect is committed.

That gap is where work dies. A worker dequeues a message, calls an external API, and is killed before it records that the call succeeded. On restart, the message is redelivered. Now the external API has been called twice. Or the worker acknowledges the message first and then does the work; on kill, the message is gone and the work never happened. Both designs are common, both are wrong in different directions, and neither is visible from the queue's dashboard.

The confusion is treating delivery as execution. Delivery is a transport concern. Execution is a state-machine concern. Restart-safety lives in the second one.

## The underlying model: at-least-once delivery plus idempotent effects

Nearly every durable queue you will use in practice — SQS, RabbitMQ with manual acks, Kafka consumer groups, a `jobs` table polled by workers — gives you at-least-once delivery. Not exactly-once. The moment a consumer crashes after doing work but before acknowledging, the system has no way to know whether the work happened, so it redelivers. That is the correct behavior; the alternative is losing messages.

So the model is fixed: **messages will be delivered more than once, and your job must produce the same end state whether it runs once or five times.** That property is idempotency, and it is the only thing that makes restarts safe.

There are three places idempotency has to be enforced, and teams usually do only the first:

1. **The job's own state.** Writing a row twice should not create two rows. Use a natural key or a unique constraint.
2. **External side effects.** Sending an email, charging a card, calling a third-party API. These are not idempotent by default and need an idempotency key or a deduplication record.
3. **Downstream consumers.** If the job emits an event, that event will also be duplicated, and its consumers inherit the same obligation.

A useful mental model: a restart-safe job is a transaction with a checkpoint. The checkpoint records "I have completed step N for task T." On redelivery, the worker reads the checkpoint and skips completed steps. The checkpoint must be written in the same durable store as the effect it describes — or the effect must itself be idempotent — otherwise you have just moved the race condition.

### Where exactly the kill lands

It helps to enumerate the windows. Consider a worker that does: receive → call API → write DB → ack. A kill can land:

| Kill point | Queue behavior | Result without idempotency |
|---|---|---|
| Before receive | Message stays queued | Fine |
| After receive, before API call | Visibility timeout expires, redelivered | Fine (no effect yet) |
| After API call, before DB write | Redelivered | **Duplicate API call** |
| After DB write, before ack | Redelivered | **Duplicate DB write** |
| After ack | Message gone | Fine |

Two of five windows are dangerous, and both are invisible in normal operation because they only open during a crash. That is why restart bugs are so often described as "intermittent" or "only on deploy."

## Worked example: a database-backed job that survives a kill

The simplest durable job system most teams can reason about is a table in the same database as the data it touches. No extra infrastructure, and the checkpoint and the effect can share a transaction. Here is a minimal correct version in Python using PostgreSQL via `psycopg`. It assumes a `jobs` table with a `status` column and a `dedupe_key` unique constraint.

```sql
CREATE TABLE jobs (
    id           BIGSERIAL PRIMARY KEY,
    dedupe_key   TEXT NOT NULL UNIQUE,
    payload      JSONB NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    attempts     INT  NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX jobs_pending_idx ON jobs (status, locked_until)
    WHERE status IN ('pending', 'running');
```

```python
import json, time, uuid
import psycopg

DSN = "postgresql://app:app@localhost/app"

# The side effect must be idempotent. Here we write to a table with a
# unique constraint on (job_id), so a retry is a no-op, not a duplicate.
def do_work(cur, job_id: int, payload: dict) -> None:
    cur.execute(
        """
        INSERT INTO ledger (job_id, amount)
        VALUES (%s, %s)
        ON CONFLICT (job_id) DO NOTHING
        """,
        (job_id, payload["amount"]),
    )


def claim_one(conn) -> dict | None:
    # SELECT ... FOR UPDATE SKIP LOCKED lets multiple workers poll safely.
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status = 'running',
                attempts = attempts + 1,
                locked_until = now() + interval '5 minutes'
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'pending'
                   OR (status = 'running' AND locked_until < now())
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, payload, attempts
            """
        )
        row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        return {"id": row[0], "payload": row[1], "attempts": row[2]}


def run_once(conn) -> bool:
    job = claim_one(conn)
    if job is None:
        return False
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                do_work(cur, job["id"], job["payload"])
                cur.execute(
                    "UPDATE jobs SET status = 'done' WHERE id = %s",
                    (job["id"],),
                )
    except Exception:
        # Leave status='running'; the locked_until expiry will make it
        # eligible again. Bounded retries are handled by a separate sweeper.
        conn.rollback()
        raise
    return True


if __name__ == "__main__":
    with psycopg.connect(DSN) as conn:
        while True:
            if not run_once(conn):
                time.sleep(1)
```

Three properties make this survive a restart:

- **The claim is atomic.** `UPDATE ... RETURNING` with `FOR UPDATE SKIP LOCKED` means two workers cannot claim the same row, and a killed worker's row becomes claimable again when `locked_until` passes.
- **The effect and the status update share a transaction.** If the process dies between them, both roll back and the job is retried cleanly.
- **The effect itself is idempotent.** The `ON CONFLICT DO NOTHING` on `ledger` means even if the same job somehow runs twice, the ledger does not double-count. This third property is what protects you from the case the transaction cannot cover: a crash *after* commit but *before* the client learns the commit succeeded.

That last point is subtle and worth stating plainly: a database transaction makes the *database* consistent, but the worker can still be killed after `COMMIT` returns and before it does anything else. The job status is `done`, so on redelivery it will be skipped — fine. But if the design instead acked the queue *before* committing, the effect would be lost. Order matters: commit the effect, then ack.

## The patterns that actually break under restart

A short catalog of failure modes that show up repeatedly, with the mechanism behind each.

**Ack-before-work.** The worker receives a message, immediately deletes it from the queue, then does the work. This is the single most common cause of lost jobs. It looks efficient — no redelivery, no duplicate work — and it fails exactly when you need it not to. The fix is ack-after-commit, accepting that duplicates are possible.

**Non-idempotent external calls.** Charging a card, sending an email, creating a ticket in a third-party system. These have no natural dedupe. The fix is an idempotency key passed to the provider (Stripe, for example, documents an `Idempotency-Key` header for exactly this) or a local `sent_events` table with a unique constraint, written in the same transaction as the local state change. Without one of these, every restart during a deploy is a chance to double-charge.

**Visibility timeout shorter than the job.** SQS's default visibility timeout is 30 seconds. A job that takes 90 seconds will be redelivered while the first worker is still running it, and now two workers are doing the same work concurrently. This is not a crash — it happens in normal operation — but it produces the same duplicate effects and is often misdiagnosed as a restart bug. Either extend the timeout or send heartbeats to extend it during long jobs.

**In-memory retry state.** A worker keeps `attempts` in a local variable and retries in a loop. On kill, the count resets. The job can loop forever across restarts, or give up too early. Retry counts belong in the durable record, not the process.

**Cron with no overlap guard.** A cron job that runs every minute but sometimes takes two minutes will overlap with itself. On a restart, the scheduler may fire immediately, adding a third copy. The fix is a lock (a row lock, an advisory lock, or a distributed lock with a TTL longer than the expected runtime) held for the duration of the job.

**Side effects in the middle of a batch.** A job processes 1,000 rows, updating each. It crashes at row 600. On restart, it starts at row 1 unless the batch is checkpointed. The fix is to make the batch resumable: process in pages keyed by a monotonic id, or record `last_processed_id` durably after each page.

## Common misreadings

**"Exactly-once delivery will fix this."** No mainstream broker gives you exactly-once end-to-end, because the boundary between the broker and your side effect is outside its control. Kafka's "exactly-once semantics" applies to Kafka-to-Kafka pipelines within its transaction model; the moment you call an external API, you are back to at-least-once. Design for duplicates.

**"Idempotency means the job can run twice."** It means the *end state* is the same after two runs as after one. A job that sends an email twice is not idempotent even if its database row is. Idempotency has to be evaluated at the level of every observable effect, not just the primary write.

**"The queue's retry policy handles failures."** Retry policies handle *exceptions*. They do not handle a process that vanishes, because there is no exception to catch. Only the visibility timeout or lock expiry brings a killed job back, and only if the worker did not ack first.

**"We use a managed queue, so this is handled."** Managed queues handle transport, redelivery, and dead-lettering. They cannot know whether your side effect committed. That responsibility is always the application's.

**"Graceful shutdown solves it."** SIGTERM handlers help — they let in-flight jobs finish or release their locks — but Kubernetes sends SIGKILL after its grace period, OOM kills are immediate, and node failures do not send signals at all. Graceful shutdown reduces the window; it does not close it.

## Designing the job contract

A restart-safe job has a small set of properties. Treat this as the checklist to apply to any job before it ships:

- **Durable claim.** The job is claimed in a store that survives process death, with a lock or visibility timeout longer than the expected runtime.
- **Checkpointed progress.** Long jobs record how far they got, so a restart resumes rather than restarts.
- **Idempotent effects.** Every external call has an idempotency key or a local dedupe record; every write has a unique constraint or an upsert.
- **Commit before ack.** The queue message is acknowledged only after the effect is durable.
- **Bounded retries in durable state.** `attempts` lives in the row, not the process, and a dead-letter path exists for poison messages.
- **Observable lag.** You can answer "how old is the oldest unprocessed job?" from a dashboard, not from logs.

The last one matters more than it sounds. Most restart incidents are discovered by a user noticing missing data, not by monitoring. A simple query — `SELECT max(now() - created_at) FROM jobs WHERE status = 'pending'` — turned into an alert, catches the class of failure where jobs are being enqueued but not processed, which is what a misconfigured worker or an expired lock looks like from the outside.

## Frequently Asked Questions

**How do I make a background job idempotent when it calls an external API?**

Pass an idempotency key that is stable across retries — usually the job's durable id — and rely on the provider to deduplicate. If the provider does not support idempotency keys, keep a local table of completed side effects with a unique constraint on the job id, and check it before calling. The local table must be written in the same transaction as the state change that depends on the call, or you reintroduce the race.

**Why does my job run twice after every deploy?**

Almost always because the worker was killed after doing the work but before acknowledging the message, so the queue redelivered it. That is correct at-least-once behavior. The duplicate is not the bug; the missing idempotency is. Check whether the job's effects have unique constraints or idempotency keys, and whether the ack happens after the commit.

**What is the difference between a visibility timeout and a lock?**

A visibility timeout is a broker concept: after it expires, the message becomes visible to other consumers, whether or not the original worker is still running. A lock is an application concept: it prevents concurrent execution explicitly. Visibility timeouts are simpler but can cause concurrent duplicates if the job outruns the timeout; locks are more precise but need their own expiry to avoid deadlock after a crash. Many systems use both.

**Should retries be handled by the queue or by the job?**

By the job, for anything where the retry decision depends on the error type. The queue's retry policy is blunt: it retries on any failure until a max count, then dead-letters. A job that distinguishes transient errors (timeouts, 429s) from permanent ones (validation failures) should record attempts durably and decide for itself, using the queue only as the transport and dead-letter backstop.

## The next 30 minutes

Open the file that defines your most business-critical background job and find the line where it acknowledges or deletes the queue message. If that line appears before the database commit or the external API call returns, you have found a job that loses work on restart — move the ack after the commit, add a unique constraint or idempotency key on the effect, and then check the job's visibility timeout against its p99 runtime.

## Official documentation

- [PostgreSQL documentation](https://www.postgresql.org/docs/)
- [Redis documentation](https://redis.io/docs/)
- [Kubernetes documentation](https://kubernetes.io/docs/home/)
- [Apache Kafka documentation](https://kafka.apache.org/documentation/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
