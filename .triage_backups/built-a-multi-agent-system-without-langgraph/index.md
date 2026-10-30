# Built a multi-agent system without LangGraph

Multi-agent LLM pipelines fail in boring ways: an upstream API starts returning 504s, a worker dies mid-job, a retry storm hammers a flaky endpoint, or two consumers process the same message twice. Framework choice matters less than whether the design answers those four cases. This article builds a three-agent research pipeline on plain Python, Redis Streams, and a managed retry layer, and shows how to measure whether it is actually working.

## The problem with graph frameworks for small teams

Graph orchestration libraries exist to solve real problems: cyclic state machines, human-in-the-loop interrupts, checkpointed resumption, visual debugging. If a workflow genuinely needs those, adopting one is reasonable.

The friction appears when a team adopts a graph framework for a pipeline that is really a linear chain of three or four steps. Common failure modes reported by teams in that situation:

- **Version churn in checkpointing APIs.** Pre-1.0 orchestration libraries change persistence formats between minor releases. Code that pins to an internal checkpoint schema breaks on upgrade.
- **Opaque deadlocks.** When the scheduler stalls, the queue state is often not exposed through public APIs, so debugging means reading library internals.
- **Impedance mismatch with existing infrastructure.** If retries, queues, and observability already exist in the platform (Step Functions, SQS, CloudWatch, an existing tracer), a second orchestration layer duplicates them.

None of that makes graph frameworks bad. It means the decision should be driven by whether the workflow is genuinely graph-shaped. A pipeline that is "research, then validate, then summarize, with retries" is a chain. A pipeline where any agent can hand off to any other agent, with approval gates and resumable checkpoints, is a graph.

A useful decision checklist before adding a framework:

1. Does any step need to loop back to an earlier step based on output? If no, a chain suffices.
2. Does a human need to inspect and approve mid-run?
3. Must a partially completed run resume after a process restart, with state persisted per node?
4. Do you need a visual trace of the graph structure, not just spans?
5. Is the team already operating a queue and retry system it trusts?

Two or more "yes" answers point toward a graph framework. Otherwise, a queue plus a retry layer is usually less code and less surface area.

## Architecture overview

The pipeline:

- Three agents: **researcher**, **validator**, **summarizer**.
- Redis Streams for task queues, consumer groups, and results.
- A managed state machine (AWS Step Functions is one option; Temporal and Argo Workflows are alternatives in the same category) for retries and execution history.
- FastAPI for the HTTP entry point.
- Lambda or a container for the worker compute.

Why Redis Streams rather than a plain message queue: streams support consumer groups, which give fan-out to N workers with per-message acknowledgement, plus a pending-entries list that survives worker crashes. A standard queue can do this too, but streams keep ordering per stream and give you `XADD`/`XREADGROUP`/`XACK` in one primitive.

Important caveat: **Redis Streams ordering is per stream, not global.** If ordering across agents matters, route everything through a single stream with one consumer group and make job handlers idempotent.

## Step 1 — environment setup

Create a project and pin dependencies:

```bash
uv init multi_agent_research --python 3.11
cd multi_agent_research
uv add fastapi uvicorn redis httpx pydantic-settings
uv add --dev pytest pytest-asyncio httpx black
```

Environment file:

```
HF_TOKEN=<your token>
REDIS_URL=redis://localhost:6379/0
```

A note on the Redis URL: the trailing `/0` selects database 0. Omitting it defaults to database 0 in most clients, but some connection helpers have historically parsed the path differently, and code that assumes one DB while writing to another produces the confusing symptom of "keys disappear after restart." Always specify the database explicitly.

Local Redis for testing:

```bash
docker run -d --name redis72 -p 6379:6379 redis:7.2-alpine
```

Minimal FastAPI app in `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}
```

The hard-to-reverse decision at this stage is the **Redis key layout**. Once writers depend on `agent:researcher:job:{job_id}`, migrating is a coordinated rewrite. Pick a scheme and document it before the first deploy.

## Step 2 — agent interface and queue

Define a single agent class. Keeping one class parameterized by name keeps the retry and timeout logic in one place.

```python
import os
import httpx
from typing import Any

class Agent:
    def __init__(self, name: str, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.name = name
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_prompt(self, data: dict[str, Any]) -> str:
        return f"You are a {self.name}. {data.get('prompt', '')}"

    async def run(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        client = await self._ensure_client()
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
        r = await client.post(
            "https://api-inference.huggingface.co/models/" + self.model,
            json={"inputs": self._build_prompt(input_payload), "parameters": {"max_tokens": 512}},
            headers=headers,
        )
        r.raise_for_status()
        return {"agent": self.name, "output": r.json()[0]["generated_text"]}
```

Enqueue helper:

```python
import json
import os
from uuid import uuid4
import redis.asyncio as redis

async def enqueue_job(topic: str, payload: dict[str, Any]) -> str:
    job_id = str(uuid4())
    payload["job_id"] = job_id
    rc = redis.from_url(os.getenv("REDIS_URL"))
    await rc.xadd(topic, {"payload": json.dumps(payload)})
    await rc.aclose()
    return job_id
```

Worker loop using a consumer group. `XREADGROUP` with `>` delivers only new messages; `XACK` removes them from the pending list. Unacknowledged messages remain in the pending entries list and can be reclaimed with `XAUTOCLAIM` if a worker dies.

```python
import asyncio
import json
import os
import redis.asyncio as redis

GROUP = "agents"

async def ensure_group(rc, stream: str, group: str) -> None:
    try:
        await rc.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

async def worker(name: str, stream: str, consumer: str) -> None:
    rc = redis.from_url(os.getenv("REDIS_URL"))
    await ensure_group(rc, stream, GROUP)
    agent = Agent(name)
    try:
        while True:
            messages = await rc.xreadgroup(
                GROUP, consumer, {stream: ">"}, count=1, block=5000
            )
            if not messages:
                continue
            _, entries = messages[0]
            message_id, data = entries[0]
            payload = json.loads(data[b"payload"].decode())
            try:
                result = await agent.run(payload)
            except Exception as e:
                # Leave unacked so it can be reclaimed; do not delete.
                print(f"agent={name} job={payload['job_id']} error={e}")
                continue
            await rc.xadd(
                f"results:{name}",
                {"job_id": payload["job_id"], "result": json.dumps(result)},
            )
            await rc.xack(stream, GROUP, message_id)
    finally:
        await agent.aclose()
        await rc.aclose()

if __name__ == "__main__":
    asyncio.run(worker("researcher", "tasks:research", "worker-1"))
```

Two deliberate choices in that loop:

- On failure, the message is **not** acknowledged and **not** deleted. It stays pending for reclamation. Deleting on failure is the classic bug that turns a transient error into silent data loss.
- The agent's HTTP client is created once and closed in a `finally` block. Creating an `AsyncClient` per call leaks file descriptors and memory under load.

## Step 3 — retries and timeouts

There are two layers where retries can live, and mixing them is a common source of confusion.

**In-process retries** are cheap and appropriate for transient network errors. Bound them tightly, use exponential backoff with jitter, and cap the total time so the worker does not exceed its own execution timeout.

```python
import asyncio
import random

MAX_ATTEMPTS = 3
BASE_DELAY = 0.5

async def safe_run(agent: Agent, payload: dict) -> dict | None:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await agent.run(payload)
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                return None
            delay = BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay + random.uniform(0, delay * 0.1))
    return None
```

The arithmetic is worth spelling out: with `BASE_DELAY = 0.5` and `MAX_ATTEMPTS = 3`, waits are roughly 0.5s then 1.0s, for a worst-case in-process budget of ~1.5s plus request time. That must fit inside the worker timeout with headroom. If the worker timeout is 15s and each request can take 10s, three attempts cannot fit — the retry layer has to move outward.

**Out-of-process retries** belong to the state machine. A Step Functions task with a `Retry` block retries the whole invocation, which is correct when the failure is "the worker died" rather than "the HTTP call blipped." Example definition:

```json
{
  "Comment": "Multi-agent research workflow",
  "StartAt": "Research",
  "States": {
    "Research": {
      "Type": "Task",
      "Resource": "arn:aws:states:us-east-1:123456789012:function:multi-agent-research",
      "Next": "Validate"
    },
    "Validate": {
      "Type": "Task",
      "Resource": "arn:aws:states:us-east-1:123456789012:function:multi-agent-research",
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 2,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ],
      "Next": "Summarize"
    },
    "Summarize": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:multi-agent-research",
      "End": true
    }
  }
}
```

Note the change from `States.ALL` to `States.TaskFailed`. Retrying on `States.ALL` also retries deterministic errors such as a malformed payload, which wastes the retry budget on failures that will never succeed. Enumerate the transient error classes explicitly.

The rule of thumb: **retry in-process for network jitter, retry out-of-process for worker death.** If both layers retry the same failure, you multiply attempt counts and can produce a retry storm against an already-degraded upstream.

## Step 4 — failure modes and how to handle them

### Duplicate processing

Redis Streams deliver at-least-once. A worker that processes a message and crashes before `XACK` will have that message reclaimed and processed again. The fix is idempotency at the job level, not at the queue level: derive a deterministic result key from `job_id` and use `SET key value NX` (or an upsert) so a duplicate write is a no-op.

```python
async def store_result(rc, job_id: str, result: dict) -> None:
    # NX makes duplicate writes harmless.
    await rc.set(f"result:{job_id}", json.dumps(result), nx=True)
```

### Poison messages

A message that fails deterministically will be reclaimed forever. Track an attempt counter in the message payload or in a Redis hash keyed by message ID, and after N attempts move it to a dead-letter stream and acknowledge the original.

```python
MAX_DELIVERIES = 5

async def handle_failure(rc, stream: str, group: str, msg_id: str, payload: dict, err: str) -> None:
    key = f"attempts:{msg_id}"
    attempts = await rc.incr(key)
    await rc.expire(key, 86400)
    if attempts >= MAX_DELIVERIES:
        await rc.xadd("deadletter", {"job_id": payload["job_id"], "error": err})
        await rc.xack(stream, group, msg_id)
```

### Upstream timeouts

Wrap the model call in a short timeout and fall back to a cached result keyed by a hash of the prompt. This converts a hard failure into a degraded-but-successful response, which is usually preferable for a summarization pipeline.

```python
import hashlib
from fastapi import HTTPException

async def call_model(prompt: str) -> str:
    key = "cache:" + hashlib.sha256(prompt.encode()).hexdigest()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
                json={"inputs": prompt},
                headers={"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"},
            )
            r.raise_for_status()
            text = r.json()[0]["generated_text"]
            await _cache_set(key, text)
            return text
    except Exception:
        cached = await _cache_get(key)
        if cached is not None:
            return cached
        raise HTTPException(status_code=503, detail="model unavailable")
```

### Retry storms

Two mitigations: jitter (shown above) and a concurrency cap on the worker. On Lambda, `reserved_concurrency` caps simultaneous invocations so a backlog cannot overwhelm the upstream. On ECS or Kubernetes, cap replicas and use a queue-depth-based autoscaler rather than a CPU-based one — CPU stays low while the queue grows.

## Step 5 — observability and tests

Instrument three things before optimizing anything: per-agent latency, failure rate by error class, and queue depth. Without those, "the system is slow" is unfalsifiable.

Structured logging with a correlation ID per job:

```python
import logging
import json

log = logging.getLogger("agent")

def log_event(job_id: str, agent: str, event: str, **fields) -> None:
    log.info(json.dumps({"job_id": job_id, "agent": agent, "event": event, **fields}))
```

An integration test using a real Redis container. The `asyncio.sleep` in the original version is a flaky-test generator; poll instead.

```python
import asyncio
import pytest
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def redis_url():
    with RedisContainer("redis:7.2-alpine") as r:
        host = r.get_container_host_ip()
        port = r.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"

@pytest.mark.asyncio
async def test_research_agent(redis_url, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_url)
    job_id = await enqueue_job("tasks:research", {"prompt": "test"})
    # Poll for the result rather than sleeping a fixed interval.
    for _ in range(50):
        result = await _read_result(redis_url, job_id)
        if result is not None:
            break
        await asyncio.sleep(0.1)
    assert result is not None
```

### How to measure the numbers that matter

Do not trust anyone's published latency or cost figures, including any in this article. Measure your own:

- **End-to-end latency.** Add a `time.perf_counter()` around the request handler and emit a histogram. Percentiles require a histogram, not an average — a mean of 850ms can hide a p99 of 8s.
- **Per-agent latency.** Same instrumentation inside `Agent.run`, tagged by agent name.
- **Failure rate by error class.** Count exceptions by type (timeout, HTTP 5xx, validation) rather than a single "errors" counter.
- **Queue depth and pending count.** `XLEN` for stream length and `XPENDING` for unacknowledged messages. Rising pending count means workers are dying or too slow.
- **Cost.** Multiply invocation count by per-invocation price for each service. Keep the arithmetic visible in a spreadsheet rather than a hardcoded number in a dashboard.

A minimal load test that gives real numbers:

```bash
# Fire 100 requests and capture per-request total time.
for i in $(seq 1 100); do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8000/run
done | sort -n | awk '{a[NR]=$1} END {print "p50="a[int(NR*0.5)], "p99="a[int(NR*0.99)]}'
```

## Comparison: chain-with-queue vs. graph framework

| Concern | Queue + state machine | Graph framework |
|---|---|---|
| Linear pipelines | Minimal code, uses existing infra | Adds a dependency and a scheduler |
| Cyclic or conditional routing | Awkward; needs a router step | First-class |
| Human-in-the-loop interrupts | Manual (store state, resume) | Built in |
| Checkpointing / resume | State machine execution history | Framework-managed, version-sensitive |
| Debugging a stall | Inspect queue + execution history | Depends on framework internals |
| Operational familiarity | Reuses existing queues and alarms | New system to learn and monitor |

Neither column is universally better. The table is a prompt to check which rows matter for your workflow.

## FAQ

**Can this scale past one instance?**
Yes. Run multiple workers in the same consumer group; Redis distributes messages across consumers. Scale on pending-entries count or stream length, not CPU. Redis itself becomes the bottleneck eventually — monitor its CPU and connection count, and shard by stream name if needed.

**Should I use a standard message queue instead of Redis Streams?**
If the platform already runs one and the team knows it, reuse it. Streams are convenient because consumer groups, pending lists, and fan-out come from one primitive, but a standard queue with visibility timeouts and a dead-letter queue provides equivalent semantics. Adding a second queueing system purely for this pipeline is usually a net negative.

**How do I resume a partially completed job?**
Persist the job's stage to a durable store keyed by `job_id` after each successful step, and make each step idempotent. On restart, read the stage and replay from there. The state machine's execution history also provides this for the workflow layer.

**What about checkpoints for long-running agents?**
If a single agent run can exceed the worker timeout, split it into multiple steps with explicit state persisted between them. That is the point at which a graph framework's checkpointing starts to earn its complexity.

## Action for the next 30 minutes

Add a latency histogram and a pending-entries gauge to your current pipeline, then run one load test:

```bash
for i in $(seq 1 50); do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8000/run
done | sort -n | awk '{a[NR]=$1} END {print "p50="a[int(NR*0.5)], "p99="a[int(NR*0.99)]}'
```

Then check `XPENDING` on your task stream. If p99 is more than roughly three times p50, or pending count is climbing, the retry and acknowledgement logic is where the next hour of work should go — not the framework choice.
