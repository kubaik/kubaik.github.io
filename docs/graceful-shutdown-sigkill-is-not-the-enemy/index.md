# Graceful shutdown: SIGKILL is not the enemy

Most teams treat SIGKILL as the villain in their shutdown story. They add signal handlers, preStop hooks, and terminationGracePeriodSeconds values plucked from a blog post, then wonder why they still lose in-flight jobs. The contrarian take: SIGKILL is not the problem. The problem is that your worker has no concept of a shutdown deadline, and no way to checkpoint work that cannot finish in time. If you fix that, SIGKILL becomes a harmless safety net. If you don't, no amount of signal handling will save you.

This post argues that graceful shutdown is not a signal-handling problem — it is a state-durability problem. The signal is just the messenger. The real work is making every long-running job resumable or abortable within a bounded time, and then treating the orchestrator's kill signal as a deadline you must beat, not a courtesy you can ignore.

## The common advice: catch SIGTERM, drain, exit

The standard recipe goes like this: register a SIGTERM handler, stop accepting new work, wait for in-flight work to finish, then exit. In Kubernetes, add a preStop hook to sleep a few seconds so the load balancer stops sending traffic before the process starts shutting down. Set terminationGracePeriodSeconds high enough that your longest job can finish. Done.

This advice is not wrong. It is incomplete in a way that quietly fails at the worst moment. The recipe assumes your work is short enough to finish within the grace period, and that the grace period is long enough to cover your p99 job duration. For a request-serving API, that is often true. For a long-running worker — a batch processor, a queue consumer, a video transcoder, a report generator — it is almost never true. Jobs run for minutes or hours. The grace period is measured in seconds. The math does not close.

When the math does not close, teams do one of two things. They set terminationGracePeriodSeconds to something enormous, like an hour, which makes every deploy take an hour and makes the orchestrator's eviction logic sluggish. Or they accept that jobs get killed mid-flight and rely on retries. Retries work only if the job is idempotent and cheap to restart. Many are not.

## Why the drain-and-wait pattern breaks for long jobs

The drain-and-wait pattern has three failure modes that show up repeatedly in production.

**The unbounded wait.** A worker receives SIGTERM, stops pulling new messages, and waits for the current job. The job has no timeout. It hangs on a network call, a lock, or a slow downstream. The grace period expires. The orchestrator sends SIGKILL. The job is lost, and because the worker never acknowledged the message, the queue redelivers it — possibly to another worker that hits the same hang. This is the classic poison-message loop, except the message is not poisonous; the shutdown logic is.

**The in-flight acknowledgement gap.** Many queue clients acknowledge a message only after the handler returns. If the process is killed between the handler finishing and the acknowledgement being sent, the message is redelivered even though the work was done. For non-idempotent work, that is a duplicate side effect. For idempotent work, it is wasted compute. Either way, the shutdown path did not coordinate with the acknowledgement path.

**The preStop sleep illusion.** The preStop hook that sleeps for five seconds is meant to let the load balancer deregister the pod. For a worker consuming from a queue, there is no load balancer in the path. The sleep does nothing except delay shutdown. Worse, teams copy the preStop hook from an HTTP service manifest into a worker manifest and believe they have solved a problem they never had.

The common thread: the pattern treats shutdown as a sequence of signals and sleeps. It does not treat it as a deadline that forces the worker to decide, at a specific moment, whether to finish, checkpoint, or abort.

## The better mental model: shutdown as a deadline, not a signal

A worker's shutdown logic should look like a deadline-aware state machine, not a signal handler. The signal is an input. The output is a decision made against a clock.

Here is the model. When the worker starts a job, it records a deadline: the job must either complete or reach a checkpoint by `start_time + max_job_duration`. When SIGTERM arrives, the worker computes the remaining time until the orchestrator's kill deadline (terminationGracePeriodSeconds minus a safety margin). It then compares the job's remaining work against that budget. If the work fits, it finishes. If it does not, it checkpoints and exits cleanly. If it cannot checkpoint, it aborts and lets the queue redeliver — but only after marking the attempt so a poison message does not loop forever.

The key shift: the worker owns the deadline. It does not wait passively for SIGKILL. It makes an explicit decision at a known time, and that decision is testable.

This model requires two things that the drain-and-wait pattern does not: a checkpoint mechanism, and a job-level timeout that is shorter than the orchestrator's grace period. Both are engineering work. Neither is optional if jobs are long.

## Checkpointing: the part nobody wants to build

Checkpointing is where most teams stall, because it sounds like distributed-systems research. In practice, for most workers, it is a small, boring change: persist enough state that a restarted job can resume without redoing everything.

The simplest form is a progress marker. A worker processing a large file writes the byte offset it has reached to a durable store every N records. On restart, it reads the offset and seeks. A worker generating a report writes partial results to object storage keyed by job ID and page number. On restart, it skips pages that already exist.

The harder form is transactional checkpointing, where the job's side effects and its progress marker must commit together. If they do not, a crash between the side effect and the marker causes duplicate work. This is why idempotency keys matter: the side effect carries a key, and the downstream deduplicates. The checkpoint then only needs to be approximately right; the idempotency key handles the boundary.

A minimal, runnable example in Python shows the shape. It uses a signal handler only to set a flag, and a deadline check inside the work loop. The checkpoint is a JSON file; in production it would be Redis, Postgres, or object storage.

```python
import json
import os
import signal
import time
from pathlib import Path

CHECKPOINT = Path("checkpoint.json")
shutdown_requested = False

def handle_sigterm(signum, frame):
    global shutdown_requested
    shutdown_requested = True

signal.signal(signal.SIGTERM, handle_sigterm)

def load_offset():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())["offset"]
    return 0

def save_offset(offset):
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"offset": offset}))
    os.replace(tmp, CHECKPOINT)  # atomic on POSIX

def process(record):
    time.sleep(0.01)  # stand-in for real work

def run(records, grace_seconds=20):
    offset = load_offset()
    deadline = time.monotonic() + grace_seconds
    for i in range(offset, len(records)):
        if shutdown_requested and time.monotonic() > deadline:
            save_offset(i)
            print(f"checkpointed at {i}, exiting")
            return
        process(records[i])
        if i % 100 == 0:
            save_offset(i)
    save_offset(len(records))
    print("completed")

if __name__ == "__main__":
    run(list(range(10000)))
```

The important details: the signal handler does almost nothing (sets a flag), the deadline is computed from `time.monotonic()` not wall clock, and the checkpoint write is atomic via `os.replace`. The handler does not call `save_offset` directly, because doing I/O in a signal handler is unsafe. The loop checks the flag and the deadline at a safe point.

This example is illustrative, not a library. The point is the shape: signal sets a flag, loop checks flag and clock, checkpoint is atomic, exit is clean.

## Where the common advice is still right

The drain-and-wait pattern is correct for short-lived work. If your p99 job duration is well under the grace period, the simple recipe is fine, and adding checkpointing is over-engineering. The decision rule is arithmetic: if `p99_job_duration + safety_margin < terminationGracePeriodSeconds`, drain-and-wait works. If not, you need the deadline model.

The preStop hook is also right for HTTP services behind a load balancer, where the delay lets endpoint propagation catch up. It is wrong for pure queue consumers, where there is no endpoint to propagate. Copying it into a worker manifest is a common mistake, but the hook itself is not the problem; applying it to the wrong workload is.

And SIGTERM handling is still necessary. The deadline model does not remove the signal; it gives the signal a job to do. Without a handler, the process dies immediately on SIGTERM and never gets the chance to checkpoint. The handler is the trigger; the deadline logic is the response.

## A decision rule for choosing a shutdown pattern

Use this table to pick the pattern that fits the workload. The rows are workload shapes; the columns are the properties that determine which pattern applies.

| Workload shape | Typical job duration | Checkpointable? | Recommended pattern |
|---|---|---|---|
| HTTP request server | milliseconds to seconds | N/A | Drain-and-wait with preStop |
| Short queue consumer | seconds | Usually not needed | Drain-and-wait, ack after handler |
| Long batch job | minutes to hours | Yes | Deadline model with periodic checkpoints |
| Long batch job | minutes to hours | No | Deadline model with abort-and-redeliver plus attempt cap |
| Streaming consumer | continuous | Offset-based | Commit offsets on shutdown, not per message |

The rule in words: if the job cannot finish inside the grace period, it must either checkpoint or abort. If it can neither checkpoint nor abort safely, the design is wrong, and no shutdown pattern will fix it. That is the uncomfortable conclusion. Graceful shutdown is a constraint on job design, not a patch you apply at the process level.

For jobs that cannot checkpoint, the abort path needs an attempt counter. Without one, a job that always times out during shutdown will be redelivered forever. The counter can live in the message metadata or a side table. After N attempts, route to a dead-letter queue and alert. This is the same poison-message logic that applies to any retry loop; shutdown is just one source of retries.

## The orchestrator settings that actually matter

Three settings determine whether the deadline model has room to work. They are worth getting right before writing any handler code.

**terminationGracePeriodSeconds** is the total budget from SIGTERM to SIGKILL. Set it to `max_expected_shutdown_time + safety_margin`. If your checkpoint write takes up to two seconds and you want to give in-flight work ten seconds to finish, set it to fifteen. Do not set it to an hour to avoid thinking about the problem; that makes deploys slow and evictions sluggish.

**The preStop hook** should be omitted for queue consumers unless you have a specific reason. For HTTP services, a short sleep (a few seconds) is reasonable to let endpoint updates propagate. The hook runs before SIGTERM, so it consumes part of the grace period. Account for it in the budget.

**The queue's visibility timeout** must be longer than the longest expected processing time, including shutdown. If a worker checkpoints and exits, the message may still be invisible to other consumers until the visibility timeout expires. If the timeout is shorter than the job duration, another worker will pick up the message while the first is still working, causing duplicate processing. This is a documented behavior of visibility-timeout-based queues, and it is a common source of duplicate side effects during deploys.

A worked example makes the arithmetic concrete. Assume a grace period of 30 seconds, a preStop sleep of 5 seconds, a checkpoint write of up to 2 seconds, and a desired in-flight finish window of 10 seconds. The budget after preStop is 25 seconds. Subtracting the checkpoint write leaves 23 seconds for work. If the job's remaining work is under 23 seconds, finish; otherwise checkpoint. These are illustrative assumptions, not measured values, but the arithmetic is the point: every second in the budget has a name.

## Testing shutdown without waiting for a deploy

Shutdown logic is rarely tested because it only runs during deploys or evictions. That is backwards. It should be tested more than the happy path, because it runs at the worst possible time.

The cheapest test is a local signal. Start the worker, send SIGTERM with `kill -TERM <pid>`, and assert that it checkpoints and exits within the budget. This catches the obvious bugs: handlers that block, checkpoints that are not atomic, loops that ignore the flag.

The next test is a kill during checkpoint. Send SIGKILL at a random point during the checkpoint write and assert that the next start reads a consistent state. This is where non-atomic writes show up. If the checkpoint is written in place rather than via a temp file and rename, a kill mid-write leaves a truncated file, and the next start either crashes or resumes from a corrupt offset.

The third test is the orchestrator path. In Kubernetes, deleting a pod triggers the real sequence: preStop, SIGTERM, grace period, SIGKILL. Running this in a staging cluster with a synthetic long job is the only way to confirm that the grace period and the job's deadline logic agree. A unit test cannot catch a mismatch between `terminationGracePeriodSeconds` and the worker's internal budget.

A useful assertion: after the test, the job should be either completed or checkpointed, never lost. If it is lost, the shutdown path has a gap, and the gap is usually between the side effect and the acknowledgement or checkpoint.

## Frequently Asked Questions

**How long should terminationGracePeriodSeconds be for a worker?**

It should be the sum of your maximum checkpoint write time, your desired in-flight finish window, and a safety margin. For a worker that checkpoints in under two seconds and finishes in-flight work in ten, fifteen to twenty seconds is reasonable. Setting it to an hour to avoid the problem makes deploys slow and evictions sluggish, and it does not fix jobs that cannot finish anyway.

**Why does my worker lose jobs on deploy even with a SIGTERM handler?**

Because the handler waits for the job to finish, and the job outlasts the grace period. The orchestrator sends SIGKILL, and the job is lost. The fix is not a longer handler; it is a deadline check inside the job loop that checkpoints or aborts before the grace period expires. The handler sets a flag; the loop makes the decision.

**What is the difference between graceful shutdown and checkpointing?**

Graceful shutdown is the process of stopping work cleanly when asked. Checkpointing is the mechanism that makes it possible for long jobs. Without checkpointing, graceful shutdown for a long job is just a slower way to lose it. With checkpointing, shutdown becomes a pause and resume rather than a loss.

**When should a worker abort instead of checkpoint?**

When it cannot persist enough state to resume safely, or when the remaining work is too large to checkpoint within the budget. In that case, abort and let the queue redeliver, but cap the attempts. Without a cap, a job that always aborts during shutdown becomes a poison message that loops forever.

## The one thing to check first

Open your worker's main loop and find where it handles SIGTERM. If the handler does anything other than set a flag, that is the first problem. If the loop has no deadline check against the orchestrator's grace period, that is the second. Fix the flag first, then add the deadline check, then add checkpointing. The next 30 minutes: run `kubectl get deployment <worker> -o jsonpath='{.spec.template.spec.terminationGracePeriodSeconds}'` and compare it to your worker's maximum checkpoint write time plus in-flight finish window. If the grace period is smaller, you have found the gap that loses jobs on every deploy.

## Official documentation

- [PostgreSQL documentation](https://www.postgresql.org/docs/)
- [Redis documentation](https://redis.io/docs/)
- [Kubernetes documentation](https://kubernetes.io/docs/home/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
