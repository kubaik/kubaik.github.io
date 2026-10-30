# Build agent workflows that survive restarts

Tutorials for agent frameworks tend to show the happy path: a request arrives, a model is called, a result is returned. Production is different. A worker gets evicted mid-call, a provider ships a new response shape, a network partition causes a retry storm, and suddenly a workflow that looked robust on paper is silently dropping tasks. This article walks through a minimal but durable agent pipeline: how to persist task state, how to resume after a restart, how to detect schema drift, and how to measure whether any of it actually helped.

The stack used here is deliberately boring: Python, FastAPI for the HTTP edge, Redis streams for durable queuing, Pydantic for schema validation, and `httpx` for outbound calls. The same patterns apply to any managed queue and any model provider.

## Prerequisites and what you will build

You will need:

- Python 3.11 or newer
- A Redis 7.x instance (local Docker is fine)
- FastAPI for the HTTP interface
- Pydantic v2 for schema validation
- `httpx` for outbound HTTP
- `structlog` (or any structured logging library) for JSON logs
- A model provider that exposes an HTTP API

The workflow you will build:

- Accepts tasks over REST and enqueues them durably
- Processes them with bounded retries and exponential backoff
- Survives container restarts without losing in-flight work
- Detects provider schema drift and fails fast instead of retrying forever
- Emits structured JSON logs so tasks can be replayed after an outage

Keep the whole thing small. A pipeline you can read end to end in one sitting is easier to debug at 2 AM than one split across a dozen abstractions.

## Step 1 — environment and task schema

Create a virtual environment and install the stack:

```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install "fastapi[all]" "redis" "pydantic>=2" "httpx" "structlog"
```

Run Redis locally with append-only persistence enabled. AOF matters here: without it, a Redis restart can lose the last few seconds of enqueued tasks.

```bash
docker run -d --name redis-durable -p 6379:6379 redis:7-alpine \
  redis-server --save 60 1 --appendonly yes --appendfsync everysec
```

Verify the connection:

```bash
redis-cli ping
```

You should see `PONG`. If you are on a managed Redis service, make sure automatic failover is enabled and that the instance has persistence turned on; a cache-only configuration is not a durable queue.

Now define the task schema. This is the first layer of durability: everything needed to resume a task after a restart lives in this object.

```python
# app.py
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Task(BaseModel):
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    input: str
    output: Any = None
    status: TaskStatus = TaskStatus.queued
    attempt: int = 0
    max_attempts: int = 3
    created_at: float = Field(default_factory=time.time)
    input_schema_fingerprint: str = ""
```

Two notes on this schema:

- `attempt` and `max_attempts` are stored with the task, not tracked in memory. If a worker dies after incrementing `attempt`, the next worker sees the incremented value and does not restart the retry budget.
- `input_schema_fingerprint` is a placeholder for detecting provider drift. It is populated in Step 3.

## Step 2 — a durable queue on Redis streams

Redis streams give you persistence, consumer groups, and message acknowledgement. That is the minimum you need for at-least-once delivery across restarts.

```python
# queue.py
import json
import time

import redis.asyncio as redis

from app import Task, TaskStatus

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
    health_check_interval=1,
)

STREAM = "task_stream"
GROUP = "agents"


async def ensure_group() -> None:
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


async def enqueue(task: Task) -> str:
    msg_id = await r.xadd(STREAM, {"payload": task.model_dump_json()})
    return msg_id


async def dequeue(consumer: str, timeout_ms: int = 5000) -> tuple[str, Task] | None:
    entries = await r.xreadgroup(
        GROUP,
        consumer,
        {STREAM: ">"},
        count=1,
        block=timeout_ms,
    )
    if not entries:
        return None
    _stream, messages = entries[0]
    msg_id, data = messages[0]
    task = Task(**json.loads(data["payload"]))
    task.status = TaskStatus.processing
    await update_task(task)
    return msg_id, task


async def ack(msg_id: str) -> None:
    await r.xack(STREAM, GROUP, msg_id)


async def update_task(task: Task) -> None:
    await r.hset(
        f"task:{task.id}",
        mapping={
            "payload": task.model_dump_json(),
            "updated_at": str(time.time()),
        },
    )


async def get_task(task_id: str) -> Task | None:
    payload = await r.hget(f"task:{task_id}", "payload")
    if payload is None:
        return None
    return Task(**json.loads(payload))
```

The important design decision: the task is stored in two places. The stream entry is the durable queue message; the hash at `task:<id>` is the current state. A worker that dies mid-processing leaves the stream entry unacknowledged, so another consumer can claim it via `XAUTOCLAIM` and resume from the last persisted state.

Two failure modes worth naming explicitly:

1. **A worker that acks before doing the work.** If you `XACK` immediately after `dequeue`, a crash loses the task. Ack only after the task reaches a terminal state.
2. **A worker that never acks.** If the worker crashes after `dequeue` but before `ack`, the message stays in the consumer group's pending entries list (PEL). It will not be redelivered to a new consumer unless something calls `XAUTOCLAIM` or `XCLAIM`. A periodic reaper that scans the PEL for entries older than a threshold is the standard fix.

The `health_check_interval=1` argument tells the client to send a `PING` every second. During a Redis failover, this shortens the window in which a worker believes it still has a healthy connection.

## Step 3 — the HTTP edge and the processor

Wire the queue into FastAPI:

```python
# main.py
from fastapi import FastAPI, HTTPException

from app import Task
from queue import ensure_group, enqueue, get_task

app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    await ensure_group()


@app.post("/tasks")
async def create_task(input: str) -> dict:
    task = Task(input=input)
    msg_id = await enqueue(task)
    return {"task_id": task.id, "message_id": msg_id}


@app.get("/tasks/{task_id}")
async def read_task(task_id: str) -> Task:
    task = await get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404)
    return task
```

Start the server:

```bash
uvicorn main:app --reload --port 8000
```

Post a task:

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"input":"extract the date"}'
```

You get a `task_id` back immediately. The task is queued and survives a restart of the FastAPI process, because the queue state lives in Redis, not in the process.

Now the processor. This is where most durability bugs live.

```python
# processor.py
import asyncio
import hashlib
import json
import os

import httpx
import structlog

from app import Task, TaskStatus
from queue import ack, enqueue, update_task

logger = structlog.get_logger()

EXPECTED_KEYS = {"choices", "usage", "model"}


def fingerprint(schema: dict) -> str:
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True).encode()
    ).hexdigest()[:16]


SCHEMA = {"result": "string"}


async def process_task(task: Task, msg_id: str) -> None:
    task.attempt += 1
    task.status = TaskStatus.processing
    task.input_schema_fingerprint = fingerprint(SCHEMA)
    await update_task(task)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                os.environ["MODEL_URL"],
                headers={"Authorization": f"Bearer {os.environ['MODEL_KEY']}"},
                json={
                    "model": os.environ.get("MODEL_NAME", "default"),
                    "messages": [{"role": "user", "content": task.input}],
                },
            )
            resp.raise_for_status()
            body = resp.json()

            unexpected = set(body.keys()) - EXPECTED_KEYS
            if unexpected:
                raise ValueError(f"unexpected response keys: {sorted(unexpected)}")

            task.output = body["choices"][0]["message"]["content"]
            task.status = TaskStatus.completed
            await update_task(task)
            await ack(msg_id)
            logger.info("task_completed", task_id=task.id)

    except Exception as e:
        logger.exception("task_error", task_id=task.id, error=str(e))
        if task.attempt >= task.max_attempts:
            task.status = TaskStatus.failed
            await update_task(task)
            await ack(msg_id)
            logger.error("task_failed_permanently", task_id=task.id)
        else:
            delay = min(2 ** task.attempt, 60)
            await asyncio.sleep(delay)
            await enqueue(task)
            await ack(msg_id)
            logger.warning("task_requeued", task_id=task.id, delay=delay)
```

Three things to notice:

- **Unknown response keys are treated as an error, not ignored.** This is the schema-drift tripwire. If a provider starts returning an extra field, the task fails fast with a clear message instead of silently producing malformed output.
- **The retry delay is capped at 60 seconds.** Without the cap, the delay doubles on every attempt: 2s, 4s, 8s, 16s, 32s, 64s. Capping prevents a single task from sitting idle for minutes while the queue backs up.
- **`ack` happens only on terminal states.** Completed and permanently failed tasks are acked; requeued tasks are acked after the new stream entry is written. That ordering matters: write the new entry first, then ack the old one. If the process dies between the two, the task is processed twice, not lost. At-least-once beats at-most-once for agent work.

## Step 4 — replay, observability and tests

Structured logs make replay possible. Configure `structlog` to emit JSON:

```python
# logging_config.py
import logging

import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
```

A replay script scans the task hashes and reprocesses anything stuck in `queued` or `processing`:

```python
# replay.py
import asyncio
import json

from app import Task, TaskStatus
from queue import r, update_task
from processor import process_task


async def replay(statuses: set[TaskStatus]) -> int:
    cursor = 0
    replayed = 0
    while True:
        cursor, keys = await r.scan(cursor, match="task:*", count=100)
        for key in keys:
            payload = await r.hget(key, "payload")
            if not payload:
                continue
            task = Task(**json.loads(payload))
            if task.status in statuses:
                await process_task(task, msg_id=f"replay-{task.id}")
                replayed += 1
        if cursor == 0:
            break
    return replayed


if __name__ == "__main__":
    count = asyncio.run(replay({TaskStatus.queued, TaskStatus.processing}))
    print(f"replayed {count} tasks")
```

Tests should simulate the failures you actually care about: worker crash, provider timeout, schema drift. A minimal pytest suite:

```python
# test_workflow.py
import pytest
from fastapi.testclient import TestClient

from main import app
from queue import r

client = TestClient(app)


@pytest.fixture(autouse=True)
async def clear_redis():
    await r.flushdb()
    yield


def test_task_is_durable_across_restart(monkeypatch):
    resp = client.post("/tasks", json={"input": "ping"})
    task_id = resp.json()["task_id"]

    # Simulate a worker crash: the stream entry is never acked.
    # A reaper (not shown) would reclaim it via XAUTOCLAIM.
    from replay import replay
    import asyncio

    asyncio.run(replay({__import__("app").TaskStatus.queued}))

    resp = client.get(f"/tasks/{task_id}")
    assert resp.json()["status"] in {"completed", "failed"}
```

### How to measure this properly

Do not trust any single number quoted from someone else's deployment. Measure your own. The instrumentation you need:

- **Completion rate.** Count tasks that reach `completed` versus total tasks enqueued, bucketed by hour. A drop after a deploy is the first sign of a regression.
- **Time in `processing`.** Record `updated_at` on every state change. A task that sits in `processing` for longer than your p99 provider latency is either stuck or the worker died.
- **Retry distribution.** Histogram of `attempt` values at completion. A long tail means your backoff is too aggressive or the provider is flaky.
- **PEL depth.** `redis-cli XPENDING task_stream agents` shows how many messages are unacked. A growing PEL means workers are dying or the reaper is not running.
- **Replay count.** Log how many tasks the replay script picks up. Zero is the goal; a nonzero steady state means something upstream is leaving tasks behind.

Run a load test with a fixed input set and a known provider latency. Compare completion rate, p95 latency, and PEL depth before and after a change. If you cannot reproduce a failure locally, inject it: kill the worker process mid-call, return a malformed response from a stub provider, and drop the Redis connection during `XADD`.

## Failure modes to plan for

A short checklist of the ones that bite teams most often:

- **Schema drift.** A provider adds, removes, or renames a field. Detect it with a strict allow-list of expected keys and fail fast. Log the unexpected keys so the fix is obvious.
- **Thundering herd on recovery.** When a provider comes back after an outage, every queued task retries at once. Add jitter to the backoff: `delay = min(2 ** attempt, 60) * (0.5 + random.random())`.
- **Non-idempotent side effects.** At-least-once delivery means a task can run twice. If the task charges a card, sends an email, or writes to a downstream system, include an idempotency key derived from `task.id` and let the downstream system de-duplicate.
- **Unbounded streams.** Redis streams grow forever unless you trim them. Use `XADD ... MAXLEN ~ 10000` or a periodic `XTRIM`. The hash entries at `task:*` also need a TTL or a retention job.
- **Consumer name collisions.** Two workers using the same consumer name will steal each other's pending entries. Generate a unique name per process, for example `f"worker-{os.getpid()}"`.
- **Clock skew.** `created_at` and `updated_at` come from the worker's clock. If workers run on hosts with unsynchronized clocks, your latency measurements will be wrong. Prefer Redis-side timestamps for ordering.

## When to reach for something heavier

Redis streams are a good fit when you want a small, self-contained durable queue with predictable semantics. They are a worse fit when:

- You need cross-region replication with strong consistency guarantees.
- You need per-message visibility timeouts, dead-letter queues, and scheduled delivery out of the box.
- You are already running a managed message broker and do not want a second stateful system to operate.

In those cases, the same task schema and processor logic apply; only the queue adapter changes. Keep `enqueue`, `dequeue`, `ack`, and `update_task` behind a small interface so swapping the backend is a one-file change.

## Take action in the next 30 minutes

Open your agent project and add a single field to your task model: `attempt: int = 0`. Persist it alongside the task payload before every outbound call, and increment it on the persisted copy rather than in memory. Then write one test that kills the worker process mid-call and asserts the task is either retried or marked failed, never silently lost. If you cannot make that test pass, the rest of the durability work is decoration.
