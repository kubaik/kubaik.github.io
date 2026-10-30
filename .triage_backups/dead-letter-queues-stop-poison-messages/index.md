# Dead-letter queues: stop poison messages

A background worker that retries a poison message forever is worse than no worker at all. It burns CPU, fills logs, and hides the real failure behind a wall of stack traces. On a single small instance acting as the only worker for a backlog of payment webhooks, one poison message can degrade the whole pipeline for hours — and the failure often surfaces only when a customer reports that a top-up never reflected.

The problem is not that messages fail. The problem is that most queue setups treat every failure as transient. A malformed JSON payload, a missing database column, or a bug in a third-party API client will never succeed on retry, no matter how many times you try. Without a dead-letter queue (DLQ) and a retry cap, those messages become immortal. They sit at the head of the queue, blocking everything behind them — a head-of-line block that turns a 200ms job into a 20-minute outage.

The part that trips people up is not building the DLQ itself. It is designing the retry policy, the visibility timeout, and the alerting so that poison messages are isolated in seconds, not hours — and so you can debug them without SSHing into a box. That is what this post covers.

## Prerequisites and what you'll build

You'll build a small but complete background job processor using Python 3.11, Redis 7.2, and RQ 1.15 (Redis Queue). RQ is a reasonable fit for teams that already run Redis and do not want the operational overhead of Celery or the cost of a managed queue like AWS SQS. It is lightweight, has a built-in failure registry, and runs on a small VPS.

You'll need:

- Python 3.11 or newer
- Redis 7.2 running locally (or a managed Redis instance)
- RQ 1.15 (`pip install rq==1.15.0`)
- A text editor and a terminal

You'll implement:

1. A job function that simulates a flaky third-party API call.
2. A worker wrapper that retries transient failures but stops after 3 attempts.
3. A dead-letter queue that stores the full job payload and exception traceback.
4. Structured logging and a simple metric for failed jobs.
5. A test suite using pytest 7.4 that proves poison messages do not block the queue.

The total code is under 150 lines and will run on a Raspberry Pi 4 with 2GB RAM.

## Step 1 — set up the environment

Before writing any job logic, get Redis and RQ installed and confirm the worker can start. This step is boring, but skipping it means you'll debug import errors instead of queue semantics.

Create a project directory and a virtual environment:

```bash
mkdir poison-queue && cd poison-queue
python3.11 -m venv .venv
source .venv/bin/activate
pip install rq==1.15.0 redis==5.0.1 pytest==7.4.4
```

Start Redis 7.2 locally. If you're on Ubuntu 22.04, the default package is Redis 6.x, so use the official Redis APT repository or run it via Docker:

```bash
docker run -d --name redis-poison -p 6379:6379 redis:7.2-alpine
```

Create a `config.py` that reads the Redis URL from an environment variable. Hardcoding `localhost:6379` is fine for a demo but breaks the moment you deploy to a container where Redis is a separate service.

```python
# config.py
import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10
```

Why `MAX_RETRIES = 3`? Most transient failures — a 503 from a payment gateway, a brief DNS blip — resolve within two attempts. Three attempts with a 10-second delay gives you a 30-second retry window. If a job still fails after that, it is almost certainly not transient, and retrying further just delays the inevitable DLQ entry. You can tune this per job type later.

Now create a `worker.py` that will run the RQ worker. You'll add the DLQ logic in the next step, but get the skeleton running first.

```python
# worker.py
import logging
from rq import Worker, Queue, Connection
from redis import Redis
from config import REDIS_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

redis_conn = Redis.from_url(REDIS_URL)

if __name__ == "__main__":
    with Connection(redis_conn):
        worker = Worker([Queue("default")])
        worker.work()
```

Run `python worker.py` in one terminal. You should see `Worker rq:worker:... started`. If you see a connection refused error, Redis is not running or the URL is wrong. Fix that before moving on.

## Step 2 — core implementation

Now write the job function and the retry logic. The key design decision is: retries happen inside the job, not at the queue level. RQ's built-in retry mechanism re-enqueues the job with a delay, but it does not give you fine-grained control over which exceptions are retryable. For poison message handling, you want to catch specific exceptions, retry those, and send everything else straight to the DLQ.

Create `jobs.py`:

```python
# jobs.py
import time
import random
import logging
from config import MAX_RETRIES, RETRY_DELAY_SECONDS

logger = logging.getLogger(__name__)

class TransientError(Exception):
    """Raised for errors that might succeed on retry."""
    pass

class PermanentError(Exception):
    """Raised for errors that will never succeed."""
    pass

def process_payment(payload: dict):
    """Simulate a payment processing job."""
    attempt = payload.get("attempt", 0) + 1
    payload["attempt"] = attempt

    # Simulate a flaky external API
    if random.random() < 0.3:
        raise TransientError("Payment gateway timeout")

    # Simulate a malformed payload that will never succeed
    if "amount" not in payload or payload["amount"] <= 0:
        raise PermanentError(f"Invalid amount: {payload.get('amount')}")

    logger.info("Payment processed for %s", payload.get("user_id"))
    return {"status": "ok", "attempt": attempt}
```

Notice the two exception types. `TransientError` is retryable. `PermanentError` is not — it should go straight to the DLQ. This distinction is the core of poison message handling. Without it, you either retry everything (wasting time on permanent failures) or retry nothing (losing transient failures that would have succeeded).

Now wrap the job in a retry loop. In RQ, you can do this inside the job function itself, which keeps the retry logic testable and avoids RQ's retry queue semantics.

```python
# jobs.py (continued)
def process_payment_with_retry(payload: dict):
    """Retry transient errors, fail fast on permanent ones."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return process_payment(payload)
        except TransientError as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            if attempt == MAX_RETRIES:
                raise  # let RQ mark it as failed
            time.sleep(RETRY_DELAY_SECONDS)
        except PermanentError as e:
            logger.error("Permanent failure: %s", e)
            raise  # no retry, go straight to failure
```

Note that `time.sleep` inside a worker blocks that worker's slot for the duration. With a 10-second delay and three attempts, a single failing job can occupy a worker for ~20 seconds. If that matters for your throughput, run retries as delayed re-enqueues instead, and track attempt counts in the job payload.

Enqueue a job from a Python shell:

```python
from redis import Redis
from rq import Queue
from jobs import process_payment_with_retry

q = Queue("default", connection=Redis.from_url("redis://localhost:6379/0"))
q.enqueue(process_payment_with_retry, {"user_id": "user_123", "amount": 5000})
```

Run the worker and watch the logs. You'll see retries for transient errors and immediate failures for permanent ones. But right now, failed jobs just sit in RQ's failed registry. You need to move them to a DLQ with enough context to debug.

## Step 3 — handle edge cases and errors

RQ stores failed jobs in a `FailedJobRegistry`. By default, it keeps the last 10,000 failures. That is fine for debugging, but it is not a DLQ — it does not isolate poison messages, and it does not give you a separate queue to monitor. More importantly, if a job fails because of a bug in your code, RQ will keep it in the failed registry, but your worker will keep pulling new jobs. The poison message does not block the queue — but it also does not get the attention it needs.

The real poison message problem shows up when you use a queue that blocks on failure, like a simple `BRPOPLPUSH` loop. In that design, a single bad message can halt the entire worker. RQ avoids that by design, but you still need a DLQ for two reasons: first, to separate poison messages from transient failures that exhausted retries; second, to trigger alerts when the DLQ grows.

Implement a DLQ using a separate Redis list. When a job fails permanently, push the full payload and exception traceback to `dlq:payments`.

```python
# dlq.py
import json
import traceback
import logging
from datetime import datetime, timezone
from redis import Redis
from config import REDIS_URL

logger = logging.getLogger(__name__)
redis_conn = Redis.from_url(REDIS_URL)

def send_to_dlq(queue_name: str, payload: dict, exc: Exception):
    """Push a failed job to the dead-letter queue with full context."""
    entry = {
        "queue": queue_name,
        "payload": payload,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    redis_conn.lpush(f"dlq:{queue_name}", json.dumps(entry))
    logger.error("Job sent to DLQ: %s", entry["error"])
```

Now modify the job to call `send_to_dlq` when retries are exhausted or a permanent error occurs.

```python
# jobs.py (updated)
from dlq import send_to_dlq

def process_payment_with_retry(payload: dict):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return process_payment(payload)
        except TransientError as e:
            if attempt == MAX_RETRIES:
                send_to_dlq("payments", payload, e)
                raise
            time.sleep(RETRY_DELAY_SECONDS)
        except PermanentError as e:
            send_to_dlq("payments", payload, e)
            raise
```

A common gotcha here: if `send_to_dlq` itself fails (Redis is down, serialization error), you'll lose the failure entirely. Wrap it in a try/except that logs to stderr as a last resort. In production, you'd also want a fallback to a local file or a cloud logging service.

Another edge case: duplicate DLQ entries. If a job is retried manually or re-enqueued, you might get the same payload in the DLQ twice. Add a deduplication key based on a hash of the payload and error type, and use Redis `SETNX` with a 24-hour expiry.

```python
def send_to_dlq_dedup(queue_name: str, payload: dict, exc: Exception):
    import hashlib
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    dedup_key = f"dlq:dedup:{queue_name}:{key}"
    if not redis_conn.set(dedup_key, "1", nx=True, ex=86400):
        logger.info("Duplicate DLQ entry skipped: %s", key)
        return
    send_to_dlq(queue_name, payload, exc)
```

This adds a few lines of overhead but prevents a flood of identical failures from filling your DLQ and drowning out unique errors. Note that the dedup key ignores the exception type and message, so two genuinely different failures with identical payloads will collapse into one entry. If that matters, include `type(exc).__name__` in the hash input.

## Step 4 — add observability and tests

A DLQ you never look at is just a slower way to lose messages. You need two things: a metric you can alert on, and a test that proves poison messages do not block the queue.

For metrics, use Redis counters. Increment `metrics:dlq:payments:count` every time you push to the DLQ. Then set up an alert: if the counter increases by more than 5 in 10 minutes, page someone. On a small team, that someone might be you, and the alert might be a Telegram message from a cron job. That is fine — the important part is that you know within minutes, not days. Pick the threshold by measuring your normal DLQ arrival rate for a week first; a fixed number chosen without a baseline will either page constantly or never fire.

```python
# metrics.py
from redis import Redis
from config import REDIS_URL

redis_conn = Redis.from_url(REDIS_URL)

def increment_dlq_metric(queue_name: str):
    redis_conn.incr(f"metrics:dlq:{queue_name}:count")
```

Call this inside `send_to_dlq`. Now write a pytest test that enqueues a poison message and verifies it ends up in the DLQ without blocking a subsequent good job.

```python
# test_poison.py
import pytest
from redis import Redis
from rq import Queue, SimpleWorker
from jobs import process_payment_with_retry
from config import REDIS_URL

@pytest.fixture
def redis_conn():
    conn = Redis.from_url(REDIS_URL)
    conn.flushdb()
    yield conn
    conn.flushdb()

def test_poison_message_goes_to_dlq(redis_conn):
    q = Queue("default", connection=redis_conn)
    # Enqueue a job with an invalid amount (permanent error)
    q.enqueue(process_payment_with_retry, {"user_id": "u1", "amount": -100})
    worker = SimpleWorker([q], connection=redis_conn)
    worker.work(burst=True)
    dlq_entries = redis_conn.lrange("dlq:payments", 0, -1)
    assert len(dlq_entries) == 1
    assert "Invalid amount" in dlq_entries[0].decode()

def test_good_job_processes_after_poison(redis_conn):
    q = Queue("default", connection=redis_conn)
    q.enqueue(process_payment_with_retry, {"user_id": "u1", "amount": -100})
    q.enqueue(process_payment_with_retry, {"user_id": "u2", "amount": 5000})
    worker = SimpleWorker([q], connection=redis_conn)
    worker.work(burst=True)
    # The good job should have succeeded
    assert redis_conn.llen("dlq:payments") == 1  # only the poison one
```

Run `pytest test_poison.py -v`. Both tests should pass in a couple of seconds. If the second test fails because the good job also went to the DLQ, check your retry logic — you are probably treating a transient error as permanent. Note that the good-job test has a 30% chance of hitting the simulated transient failure, so it can be flaky; set `random.seed()` in the test, or make the flakiness injectable, before relying on it in CI.

Now add a simple health check script that reports DLQ depth. Run it every 5 minutes from cron.

```bash
#!/bin/bash
# healthcheck.sh
DEPTH=$(redis-cli LLEN dlq:payments)
if [ "$DEPTH" -gt 10 ]; then
  echo "DLQ depth is $DEPTH — investigate" | mail -s "DLQ alert" ops@example.com
fi
```

On a small VPS, this is enough to catch poison messages before they become a crisis.

## How to measure the effect yourself

Rather than trust a table of numbers from someone else's machine, instrument your own. The measurements that matter are:

- **Time to isolate a poison message.** Log a timestamp when a job first fails and again when it lands in the DLQ. The difference is your detection latency. Compare that against how long it would take to notice via manual log inspection.
- **Worker CPU under a poison loop.** Run `top -b -n 1 | grep python` or read the worker's CPU from `/proc/<pid>/stat` while a permanent failure is being retried indefinitely versus after the retry cap is applied. This shows whether retries are starving other jobs.
- **Queue latency.** Measure the time between `enqueue` and job start for a known-good job while a poison message is present. This is the head-of-line blocking cost.
- **Recovery time.** Time how long it takes to re-enqueue every DLQ entry after a fix. With a replay script, this is bounded by your script's throughput; without one, it is bounded by how long it takes someone to reconstruct payloads by hand.

A controlled test with 1,000 jobs, 10% of which are poison, is a reasonable harness. Your numbers will vary with payload size, Redis latency, and worker count, but the direction is consistent: a retry cap plus automatic DLQ routing turns a multi-minute detection window into seconds, and it stops a single bad message from consuming a worker indefinitely.

## Common questions and variations

**How do I replay jobs from the dead-letter queue?**

Write a small script that pops entries from the DLQ and re-enqueues them. Always add a `replayed_at` field so you can track how many times a job has been replayed. If a job fails again after replay, it goes back to the DLQ — but now you have two entries, which helps you spot patterns.

```python
import json
from datetime import datetime, timezone
from redis import Redis
from rq import Queue
from config import REDIS_URL
from jobs import process_payment_with_retry

redis_conn = Redis.from_url(REDIS_URL)

def replay_dlq(queue_name: str, limit: int = 10):
    q = Queue(queue_name, connection=redis_conn)
    for _ in range(limit):
        entry = redis_conn.rpop(f"dlq:{queue_name}")
        if not entry:
            break
        data = json.loads(entry)
        data["payload"]["replayed_at"] = datetime.now(timezone.utc).isoformat()
        q.enqueue(process_payment_with_retry, data["payload"])
```

**What's the difference between a DLQ and a retry queue?**

A retry queue holds jobs that failed transiently and are waiting to be retried. A DLQ holds jobs that have exhausted retries or failed permanently. Mixing them means you cannot tell whether a job is waiting or dead. Keep them separate.

**Should I use a managed queue with a built-in DLQ instead of Redis?**

If you are already on a cloud provider and can absorb the per-request cost, a managed queue gives you a DLQ with automatic redrive and no worker process to babysit. The trade-offs are usually a message size limit and the need to serialize payloads as bytes or JSON rather than arbitrary Python objects. For small teams without a cloud billing relationship, Redis plus RQ is a practical alternative that runs on any VPS.

**How do I handle poison messages that are too large for Redis?**

Redis has a 512MB value limit, but you should keep messages well under 1MB. If your payload is larger, store it in object storage (or a local file) and put only the reference in the queue. The DLQ entry should include the storage key, not the full payload.

## A decision checklist before you deploy

- Does every job type have an explicit retry cap? An unbounded retry is a poison-message generator.
- Are exceptions classified as transient or permanent at the point they are raised, not inferred later?
- Does the DLQ entry contain the payload, the exception type, the traceback, and a timestamp?
- Is there a metric on DLQ depth, and an alert threshold chosen from a measured baseline?
- Is there a replay script, and has it been run against a real DLQ entry at least once?
- Is `send_to_dlq` itself protected against failure, so a broken DLQ does not swallow the original error?
- Is deduplication in place if the same payload can fail repeatedly?

## Your next 30 minutes

Open your current worker code and add a single `MAX_RETRIES` constant with a value of 3. Then wrap your job function in a `try/except` that catches one specific transient exception, retries up to the cap, and re-raises everything else. Run your existing test suite. If a poison message still blocks the queue, you will see it in the test output — and now you have a place to put the DLQ logic.
