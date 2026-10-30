# Agent frameworks: stop rewriting the same glue code

Agent codebases tend to converge on the same shape: a thin slice of genuinely novel logic (prompting, tool selection, domain rules) wrapped in a thick layer of plumbing that has nothing to do with the product. The plumbing is where incidents live. This article is about the plumbing: what to abstract, what to leave alone, and how to tell whether the abstraction is paying for itself.

## The short version

Most of the code in a typical agent service is not agent logic. It is retry loops, credential fetches, state persistence, and per-provider rate-limit handling. Three abstractions cover the bulk of it:

1. A job queue adapter with deterministic, per-provider retries.
2. A credentials service that issues short-lived tokens and logs rotations.
3. A minimal DAG schema plus a compiler that emits code for whichever runtime you actually deploy on.

The design constraint that matters: the abstractions must be opinionated enough to delete the boilerplate, but thin enough that swapping the underlying queue, secret store, or runtime is a configuration change rather than a rewrite.

## Why this is confusing at first

The first mistake is treating an agent framework as "more async." Async runtimes in Python 3.11+ or Node 20 LTS handle concurrency. They do not handle retries that respect a provider's `Retry-After` header, credential rotation without a process restart, or a step that blocks for hours waiting on a human. Those are orchestration concerns, and they sit above the event loop.

The second trap is assuming a cron job or a serverless scheduler is a sufficient orchestrator. Cron has no idempotency keys, so a duplicate trigger becomes duplicate work. Serverless functions have execution time limits that are shorter than a realistic human-in-the-loop step, so the workflow either fails or has to be split in ways the scheduler does not model.

A recurring failure mode is the retry loop that never converges. A service receives a 429, sleeps a fixed interval, and that interval is simultaneously too short for one provider, too long for another, and blind to the `Retry-After` header the provider actually sent. A second failure mode is credential sprawl: each tool has its own `.env`, and rotating one key means touching many repositories. A third is state management without an idempotency strategy — intermediate results stored in object storage or a relational table, where a retried job is indistinguishable from a new request downstream.

All three have the same root cause: the orchestration logic is duplicated per integration instead of centralized.

## The mental model

Model an agent as a directed acyclic graph. Each node is one of: an API call, a human step, or a conditional branch. Edges carry retry policy, timeouts, and credential lookups. The traversal logic — scheduling ready nodes, honoring retry budgets, persisting state between steps — is generic. It should not be written per agent.

The three abstractions below are the minimum surface area that removes the duplication.

**1. Job queue with deterministic retries.** "Deterministic" here means three concrete properties: the retry delay honors the provider's `Retry-After` header when present; each provider has its own retry budget rather than a global one; and the adapter emits metrics (attempts, delays, failures by provider) that you can alert on. In practice this is a thin adapter over whatever queue you already run — Celery, RQ, SQS, or a Redis-backed queue — that normalizes retry behavior across providers.

**2. Credentials as a service.** A single endpoint that issues short-lived tokens, refreshes them on access, and logs every rotation. The pattern is the same one Vault and cloud secret managers implement; the point is that agents need rotation at runtime, not at deploy time, because a long-running agent can outlive a static key.

**3. A minimal DAG compiler.** A YAML or JSON schema describing the graph, plus a code generator that emits either in-process code, Kubernetes Jobs, or serverless functions. The compiler injects retry budgets, credential lookups, and idempotency keys so the agent author never writes them.

## A worked example

Scenario: a support agent that (a) classifies a ticket, (b) fetches customer history from a CRM, (c) optionally asks a human for clarification, and (d) updates the ticket status.

The DAG schema:

```yaml
# agent.yaml
steps:
  - id: classify
    action: llm
    input: ticket_text
    output: classification
    retries:
      budget: 3
      backoff: exponential
      per_provider:
        openai: 2000ms
        anthropic: 4000ms

  - id: fetch_history
    action: crm_api
    input: customer_id
    output: history
    depends_on: classify
    retries:
      budget: 5
      backoff: linear

  - id: ask_human
    action: human_review
    input: classification, history
    output: resolution
    only_if: classification == "needs_human"

  - id: update_ticket
    action: crm_api
    input: ticket_id, resolution
    depends_on: ask_human OR fetch_history
    retries:
      budget: 2
```

From this schema the compiler emits three artifacts: a module containing the retry loop (honoring `Retry-After` per provider), a deployment artifact for the target runtime (a scheduled job, a queue consumer, or a set of functions), and a small credentials client that fetches short-lived tokens.

**Reasoning about the retry budget.** Take the `classify` step with `budget: 3` and exponential backoff starting at 2s for one provider. The delays are 2s, 4s, 8s — three retries, total worst-case added latency 14s, plus the original attempt. That is arithmetic from stated assumptions, not a measurement. If your provider's rate-limit window is 60s, a 14s worst case will exhaust the budget before the window resets; you would raise the base delay or the budget. The point of putting the budget in the schema is that this calculation is reviewable in a pull request instead of buried in a `sleep()` call.

**Reasoning about idempotency.** `update_ticket` writes to the CRM. If the worker crashes after the write but before acknowledging the message, the queue redelivers and the write happens twice. The compiler therefore injects an idempotency key derived from the step ID and the run ID, and the CRM call includes it. If the CRM does not support idempotency keys, the step must instead be preceded by a read that checks current state — a pattern the schema can express but the compiler cannot invent for you.

## How this relates to tools you already know

If you have used Airflow or Temporal, the DAG-and-retry model will be familiar. Airflow requires a metadata database and a scheduler process; Temporal requires a cluster. The abstractions here are deliberately smaller: a schema, a queue adapter, and a code generator. The trade is that you give up the operational maturity of those systems in exchange for something you can read end to end in an afternoon. For teams running a handful of agents, that trade is often correct; for teams running hundreds of workflows with complex visibility requirements, it usually is not.

The credentials piece overlaps with the twelve-factor principle of storing config in the environment. The difference is timing: twelve-factor assumes secrets are injected at deploy time, while an agent that runs for hours needs to refresh credentials mid-flight without a restart.

## Common misconceptions

**"The framework already handles retries."** Many agent libraries ship some retry behavior. The question to ask is whether it honors `Retry-After`, whether the budget is per-provider or global, and whether the delay is configurable per provider. If the answer to any of those is no, you will end up writing the retry layer yourself, and it is better to do that once in a queue adapter than once per tool.

**"One retry budget is enough."** Providers differ in how their rate-limit windows reset and in whether they send `Retry-After` at all. A single global budget tuned for the fastest-resetting provider will hammer the slowest one; tuned for the slowest, it will under-utilize the fastest. Per-provider budgets are the minimum viable configuration.

**"The runtime choice is permanent."** It is not, provided the DAG schema is runtime-agnostic. The compiler emits a different deployment artifact per target; the schema does not change. That is the main argument for keeping the schema declarative rather than expressing the graph directly in code.

## Adding a circuit breaker

Once the queue adapter is stable, the next useful addition is a circuit breaker that disables a provider for a cooldown period after a failure threshold. It belongs in the adapter, not in agent code, so that every agent benefits without changes.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


@dataclass
class _BreakerState:
    failure_count: int = 0
    last_failure: datetime | None = None


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    timeout: timedelta = timedelta(minutes=5)
    _state: dict[str, _BreakerState] = field(default_factory=dict)

    def _get(self, provider: str) -> _BreakerState:
        if provider not in self._state:
            self._state[provider] = _BreakerState()
        return self._state[provider]

    def allow_request(self, provider: str) -> bool:
        cb = self._get(provider)
        now = datetime.now(timezone.utc)
        if cb.last_failure and (now - cb.last_failure) < self.timeout:
            return False
        if cb.failure_count >= self.failure_threshold:
            return False
        return True

    def record_success(self, provider: str) -> None:
        cb = self._get(provider)
        cb.failure_count = 0
        cb.last_failure = None

    def record_failure(self, provider: str) -> None:
        cb = self._get(provider)
        cb.failure_count += 1
        cb.last_failure = datetime.now(timezone.utc)
```

Two details worth noting. First, `datetime.utcnow()` is deprecated as of Python 3.12; use `datetime.now(timezone.utc)` as above. Second, the breaker as written is per-process. If you run multiple workers, each has its own view of failures, which is usually acceptable but worth knowing before you rely on it for a global rate-limit guard.

## Human-in-the-loop without blocking a worker

A human step should not occupy a worker slot. The standard pattern is to persist the pending step, return immediately, and resume when a callback arrives.

```python
from datetime import datetime, timezone
from uuid import uuid4
import json

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


class HumanStepRequest(BaseModel):
    input_data: dict
    ttl_seconds: int = 86400  # 24 hours


@app.post("/human-step")
async def create_human_step(request: HumanStepRequest):
    step_id = str(uuid4())
    key = f"human_step:{step_id}"
    await r.json().set(key, "$", {
        "input": request.input_data,
        "status": "waiting",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    await r.expire(key, request.ttl_seconds)
    return {"step_id": step_id, "callback_url": f"/human-step/{step_id}/result"}


@app.post("/human-step/{step_id}/result")
async def complete_human_step(step_id: str, result: dict):
    key = f"human_step:{step_id}"
    if not await r.exists(key):
        raise HTTPException(status_code=404, detail="Step not found or expired")
    await r.json().set(key, "$.result", result)
    await r.json().set(key, "$.status", "completed")
    await r.lpush("human_step_callbacks", json.dumps({"step_id": step_id, "result": result}))
    return {"status": "ok"}
```

The `TTL` on the key is what prevents orphaned pending steps from accumulating. A separate worker consumes `human_step_callbacks` and resumes the DAG. Note that the callback endpoint must be authenticated; a human step that accepts unauthenticated results is an authorization hole, not just a design smell.

## A credentials client with a local cache

The cache is what keeps the credentials service off the hot path. The trade is a bounded window in which a revoked token is still used.

```python
from datetime import datetime, timedelta, timezone

import hvac


class VaultCredentials:
    def __init__(self, vault_url: str, role_id: str, secret_id: str, cache_ttl: timedelta = timedelta(minutes=5)):
        self.client = hvac.Client(url=vault_url)
        self.client.auth.approle.login(role_id=role_id, secret_id=secret_id)
        self.cache: dict[str, dict] = {}
        self.cache_ttl = cache_ttl

    def get_token(self, provider: str) -> str:
        now = datetime.now(timezone.utc)
        cached = self.cache.get(provider)
        if cached and (now - cached["fetched_at"]) < self.cache_ttl:
            return cached["token"]

        path = f"secret/data/providers/{provider}"
        secret = self.client.secrets.kv.v2.read_secret_version(path=path)
        token = secret["data"]["data"]["api_key"]
        self.cache[provider] = {"token": token, "fetched_at": now}
        return token
```

Set the cache TTL well below the token lifetime. If tokens live one hour and the cache is five minutes, a revoked token is usable for at most five minutes after revocation, plus whatever propagation delay the issuer has. That number belongs in your threat model, not in a comment.

## Measuring whether the abstraction helps

Do not trust a claimed speedup. Instrument these four things before and after the change, and compare on the same workload:

- **Retry attempts per provider per day.** Emit a counter at the point where the adapter decides to retry. A drop here means the per-provider budgets are doing work.
- **Wall-clock latency per step, split by step type.** Histogram, not average. Retry delays show up in the tail, which the mean hides.
- **Credential fetches per hour versus cache hits.** A high miss rate means the cache TTL is too short or the token lifetime is too short.
- **Time from "new tool needed" to "tool in production."** This is the metric the abstraction is actually for. Track it in your issue tracker, not in code.

To get the retry numbers without spending on real API calls, run the DAG against a mock provider that returns 429 with a known `Retry-After` value and assert that the recorded delay matches. That test is cheap, deterministic, and catches the most common regression: someone changing a backoff constant.

## Failure modes the abstraction does not fix

**Queue unavailability.** If the queue backend is unreachable, the adapter cannot help. A local durable fallback (an on-disk queue) adds resilience but also adds a second source of truth and a reconciliation problem. Decide deliberately whether that complexity is worth it for your workload.

**Provider policy changes.** When a provider changes its rate limit, the per-provider budget in the schema is stale until someone updates it. The abstraction makes the change one line, but it does not make it automatic. If your provider publishes limits in a machine-readable form, a scheduled job that reconciles the schema against it is worth building; otherwise, alert on sustained 429 rates instead.

**Runtime-specific code.** A DAG that calls into a heavy native library may not compile to every target. The compiler should fail loudly with a clear message rather than emitting something that breaks at runtime.

**Clock and timezone assumptions.** A schedule expressed in UTC will fire at the wrong local time for users in other zones. Add a `timezone` field to the schema and emit the zone explicitly in the generated cron expression rather than assuming UTC.

## A decision checklist

Before adopting these abstractions, answer:

- How many agents or workflows do you run today, and how many do you expect in a year? Below roughly five, the abstraction may cost more than it saves.
- How many distinct API providers do your agents call? One provider means per-provider retry budgets buy you little.
- Do any steps require human input, and can they run longer than your serverless timeout?
- Do you already run a queue you trust, or would this introduce a new piece of infrastructure?
- Who is on call when the credentials service is down, and what is the documented fallback?
- Can you measure the four metrics above before and after, or will you be guessing?

If you cannot answer the last question, the change is not ready to ship — you will not be able to tell whether it worked.

## Next step: do this in the next 30 minutes

Open your agent codebase and count the places where a retry delay is hard-coded or a credential is fetched inline. Pick the single most-duplicated one. Write a failing test that asserts the retry delay honors a `Retry-After` header of `0`, then make it pass by routing that call through one shared adapter function. You now have one adapter, one test, and a baseline count of the remaining call sites — which is exactly the evidence you need to decide whether to continue.
