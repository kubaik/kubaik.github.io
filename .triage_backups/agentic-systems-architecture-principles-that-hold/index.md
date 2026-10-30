# Agentic systems: architecture principles that hold

Evaluation and reliability work has a reputation for being boring, which is exactly why it rarely gets budgeted. The failure mode surfaces later, usually at the worst possible time. This article covers what happens after the happy path: the orchestration concerns that decide whether an agentic system stays operable as it grows.

Agentic and multi-model systems break the tidy assumptions that microservices and serverless architectures were built on. A single user request can fan out into dozens of model calls, each with its own latency tail, token budget, and failure mode. The principles that survive are the ones that treat nondeterminism as a first-class constraint rather than an exception.

This walkthrough covers the architecture patterns that hold up when a single model call becomes a graph of agents, tools, and fallback models. You will build a small orchestrator in Python that enforces timeouts, budgets, and idempotency — the three levers that keep agentic systems from becoming unobservable money sinks. The result is a runnable pattern that can be dropped into an existing service.

## The problem this solves

A typical multi-model system starts as a chain: prompt → model → tool → model → response. Each hop adds latency variance and a new failure surface. When a downstream model times out, a naive retry re-runs every prior step, including side-effecting tool calls. When a model returns a malformed tool call, the orchestrator either crashes or silently drops the step. When traffic spikes, the system burns through token budgets before anyone notices.

The core problem is that agentic systems lack a built-in notion of a transaction boundary. One has to be imposed — a unit of work with a deadline, a cost ceiling, and a stable identity — before more agents are added.

## Prerequisites and what you'll build

You need Python 3.11+ and familiarity with `asyncio`. No specific model provider is required; the code uses a `call_model`-style interface that can be backed with any HTTP client. You will build:

- A `Step` abstraction with timeout and retry policy.
- A `RunContext` that carries a deadline and token budget through nested calls.
- An idempotency key derived from the request, not the attempt.
- Structured logging that records every model call with its outcome and token usage.

The pattern is provider-agnostic and works whether the system calls one model or routes across several.

## Step 1 — Define the unit of work before writing any agent code

The common sequence is to start with the agent loop and bolt on reliability later. Reverse that. Define a `RunContext` that every model and tool call receives. The context carries three invariants:

1. **Deadline** — an absolute monotonic timestamp. Every step checks remaining time before starting.
2. **Budget** — a token ceiling for the entire run, decremented by each call.
3. **Idempotency key** — a stable string derived from the original request payload, not from the attempt number.

```python
import time
import hashlib
import json
from dataclasses import dataclass

class BudgetExceeded(Exception):
    def __init__(self, used: int, budget: int):
        super().__init__(f"token budget exceeded: {used} > {budget}")
        self.used = used
        self.budget = budget

@dataclass
class RunContext:
    deadline: float          # time.monotonic() value
    token_budget: int
    idempotency_key: str
    tokens_used: int = 0

    @classmethod
    def from_request(cls, payload: dict, timeout_s: float = 30.0, budget: int = 20_000):
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:32]
        return cls(
            deadline=time.monotonic() + timeout_s,
            token_budget=budget,
            idempotency_key=key,
        )

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def charge(self, tokens: int) -> None:
        self.tokens_used += tokens
        if self.tokens_used > self.token_budget:
            raise BudgetExceeded(self.tokens_used, self.token_budget)
```

Why this matters: without a deadline, a single slow model call can hold a worker for minutes. Without a budget, a retry loop can multiply cost silently. Without a stable idempotency key, a step that has already caused a side effect cannot be retried safely.

**Gotcha:** deriving the idempotency key from the attempt number is a common trap. If the key changes on retry, downstream services treat the retry as a new request and duplicate the side effect. Derive it from the input payload and the step name, and pass it unchanged through every retry.

## Step 2 — Wrap every model and tool call in a guarded step

A step is the only place where external calls happen. It enforces the context invariants and records a structured outcome. Keep the step generic; the model-specific logic lives in the callable passed in.

```python
import asyncio
import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger("agent")

class StepFailed(Exception):
    def __init__(self, step: str, reason: str):
        super().__init__(f"{step}: {reason}")
        self.step = step
        self.reason = reason

async def guarded_step(
    name: str,
    ctx: RunContext,
    fn: Callable[[RunContext], Awaitable[tuple[Any, int]]],
    max_attempts: int = 3,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        remaining = ctx.remaining()
        if remaining <= 0:
            raise StepFailed(name, "deadline exceeded before attempt")
        try:
            result, tokens = await asyncio.wait_for(fn(ctx), timeout=remaining)
            ctx.charge(tokens)
            log.info("step_ok", extra={
                "step": name, "attempt": attempt,
                "tokens": tokens, "key": ctx.idempotency_key,
            })
            return result
        except asyncio.TimeoutError:
            log.warning("step_timeout", extra={
                "step": name, "attempt": attempt, "remaining": remaining,
            })
            if attempt == max_attempts:
                raise StepFailed(name, "timeout after retries")
        except BudgetExceeded:
            raise
        except Exception as exc:
            log.warning("step_error", extra={
                "step": name, "attempt": attempt, "error": repr(exc),
            })
            if attempt == max_attempts:
                raise StepFailed(name, f"failed after retries: {exc!r}")
            await asyncio.sleep(min(2 ** attempt * 0.1, 1.0))
    raise StepFailed(name, "unreachable")
```

The `asyncio.wait_for` call is the enforcement point for the deadline. Passing `remaining` rather than a fixed timeout means the step cannot outlive the run. The budget check happens after the call returns because token usage is only known then; if the call would exceed the budget, the charge raises and the run aborts before the next step.

**Gotcha:** wrapping a synchronous SDK in `asyncio.wait_for` does not cancel it. The coroutine returns, but the underlying thread or socket may keep running. Use an async-native client, or run the sync call in a thread pool with `asyncio.to_thread` and accept that cancellation is cooperative.

**A note on retry safety:** the retry loop above retries any exception, including ones raised by non-idempotent tools. If a tool cannot deduplicate, mark it as non-retryable and let the run fail so the caller can decide. Retrying a charge that may have succeeded is worse than failing loudly.

## Step 3 — Make tool calls idempotent and observable

Tool calls are where agentic systems cause real-world side effects: writing rows, sending messages, charging cards. The orchestrator cannot guarantee exactly-once execution across a network boundary, so the tool itself must be idempotent. Pass the run's idempotency key and the step name to the tool, and require the tool to deduplicate on that pair.

```python
async def charge_card(ctx: RunContext, amount_cents: int, customer: str):
    # The downstream API must accept an idempotency key header.
    # This example assumes a generic HTTP client `post`.
    key = f"{ctx.idempotency_key}:charge_card:{customer}"
    resp = await post(
        "/charges",
        json={"amount": amount_cents, "customer": customer},
        headers={"Idempotency-Key": key},
        timeout=ctx.remaining(),
    )
    return resp.json(), 0  # tool calls do not consume model tokens
```

Two rules make this reliable:

- The key must be stable across retries of the same logical step. Concatenating the run key with the step name and the target resource gives you that.
- The tool must return the original result on a duplicate key, not an error. Most payment APIs and many internal services support this; if yours does not, add a dedupe table keyed on the idempotency string.

For observability, log every step with the run key, step name, attempt, outcome, and token count. A structured log line per step is enough to reconstruct a run. Avoid logging full prompts by default; log a hash and the token count instead.

## Step 4 — Route across models without branching the whole graph

Multi-model systems usually start with a single provider and add a fallback later. The mistake is to branch the agent graph per provider. Instead, keep one graph and make the model call a parameterized step. The router decides which model to call based on the remaining budget, the step's required capability, and the provider's current health.

```python
from dataclasses import dataclass

@dataclass
class ModelSpec:
    name: str
    cost_per_1k_tokens: float  # illustrative
    max_context: int

ROUTES = {
    "cheap": ModelSpec("small-model", 0.0005, 32_000),
    "strong": ModelSpec("large-model", 0.0100, 200_000),
}

async def call_model(ctx: RunContext, tier: str, prompt: str):
    spec = ROUTES[tier]
    if ctx.token_budget - ctx.tokens_used < 2_000:
        tier = "cheap"
        spec = ROUTES[tier]
    # provider-specific HTTP call omitted; return (text, tokens_used)
    text, tokens = await provider_call(spec.name, prompt, timeout=ctx.remaining())
    return text, tokens
```

The router is a function, not a fork in the graph. That keeps the step names stable, which keeps the idempotency keys stable, which keeps retries safe. When a third provider is added, a `ModelSpec` and a routing rule are added; the graph does not change.

**Gotcha:** routing on remaining budget alone causes tier flapping under load — a run starts on the strong model, drops to cheap mid-way, and produces inconsistent output. Route once at the start of the run based on the request class, and only downgrade on explicit budget pressure with a logged reason.

## Step 5 — Bound fan-out and concurrency

Agents that spawn sub-agents or parallel tool calls can multiply load unpredictably. A single request that fans out to twenty parallel model calls will saturate a provider's rate limit and trigger retries across the whole graph. Bound fan-out at the orchestrator: a semaphore per run and a global concurrency limit per provider.

```python
class ProviderLimiter:
    def __init__(self, max_concurrent: int):
        self._sem = asyncio.Semaphore(max_concurrent)

    async def run(self, coro):
        async with self._sem:
            return await coro

limiter = ProviderLimiter(max_concurrent=8)

async def parallel_steps(ctx: RunContext, steps):
    tasks = [limiter.run(guarded_step(name, ctx, fn)) for name, fn in steps]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

`return_exceptions=True` is deliberate: one failed sub-step should not cancel its siblings. Collect the exceptions and decide at the graph level whether the run can continue. A common pattern is to require at least one successful step per group and to degrade gracefully otherwise.

**Gotcha:** `asyncio.gather` with `return_exceptions=True` swallows `CancelledError` in some Python versions, which can leave tasks running after the parent is cancelled. Check the Python version's behavior and prefer `asyncio.TaskGroup` (3.11+) when you want structured cancellation. If you use `TaskGroup`, a failure in one task cancels the others — the opposite of the fan-out semantics above. Pick one deliberately.

## Step 6 — Add a verification harness before you add agents

Nondeterministic model output cannot be unit-tested directly, but the orchestration around it can. The harness replaces `call_model` with a scripted fake that returns fixed outputs, raises timeouts, and exceeds budgets on demand. That lets you assert the properties that matter: the run respects its deadline, the budget is never exceeded, and a retried step reuses the same idempotency key.

```python
import pytest

@pytest.mark.asyncio
async def test_deadline_enforced():
    ctx = RunContext.from_request({"q": "hi"}, timeout_s=0.05)

    async def slow(ctx):
        await asyncio.sleep(0.2)
        return "late", 10

    with pytest.raises(StepFailed) as exc:
        await guarded_step("slow", ctx, slow, max_attempts=1)
    assert "timeout" in exc.value.reason

@pytest.mark.asyncio
async def test_budget_aborts_run():
    ctx = RunContext.from_request({"q": "hi"}, budget=100)

    async def expensive(ctx):
        return "ok", 500

    with pytest.raises(BudgetExceeded):
        await guarded_step("expensive", ctx, expensive)
```

These two tests catch the most common regressions: a step that ignores the deadline and a call that silently overspends. Add a third that asserts the idempotency key is identical across retries — that is the property that protects side effects.

```python
@pytest.mark.asyncio
async def test_key_stable_across_retries():
    ctx = RunContext.from_request({"q": "hi"})
    seen = []

    async def flaky(ctx):
        seen.append(ctx.idempotency_key)
        if len(seen) == 1:
            raise RuntimeError("transient")
        return "ok", 5

    await guarded_step("flaky", ctx, flaky, max_attempts=2)
    assert len(seen) == 2
    assert seen[0] == seen[1]
```

## How to verify it works

Run the harness with a deliberately slow fake and a budget smaller than one call. The run should fail fast with `StepFailed` or `BudgetExceeded`, not hang. Then run it with a fake that fails once and succeeds on retry; the log should show two attempts with the same `key` field. If the key differs between attempts, the key derivation is wrong — fix it before wiring in a real provider.

To measure the system in staging rather than in tests, instrument three things: the wall-clock duration of each run (compare against the deadline), the token count per run (compare against the budget), and the number of attempts per step (a rising retry rate is an early signal of provider degradation). Emit these as structured log fields or metrics, then plot the distribution rather than the mean. The tail — p95 and p99 run duration — is where the deadline is actually doing work.

## Where the principles hold and where they do not

| Principle | Holds for | Breaks when |
|---|---|---|
| Deadline per run | Any multi-step graph | A step legitimately needs more time than the user's SLA allows — split the run |
| Token budget | Cost control across providers | Output length is unbounded and unknown until the call returns — cap `max_tokens` |
| Stable idempotency key | Retried side effects | The tool cannot deduplicate — add a dedupe table or make the tool read-only |
| Bounded fan-out | Parallel tool calls | A sub-agent must run for minutes — move it to a queue, not a request |
| Single graph, routed models | Multi-provider fallback | Providers have incompatible tool-call schemas — normalize at the step boundary |

The principles are not new. They are the same ones that made microservices survivable: timeouts, idempotency, bounded concurrency, and structured logs. What changed is that the failure modes are now probabilistic, so the enforcement has to be explicit in code rather than assumed from the transport.

## A worked failure analysis

Consider a three-step run — retrieve, summarize, charge — with a 30-second deadline and a 20,000-token budget. The retrieve step takes 2 seconds. The summarize step calls the strong model and uses 8,000 tokens in 6 seconds. The charge step calls the payment API, which times out at 20 seconds.

Without the patterns above, the orchestrator retries the whole chain. The retrieve and summarize steps re-run, consuming another 8,000 tokens and 6 seconds, and the charge is attempted a second time with a fresh key — a duplicate charge. Total elapsed time exceeds the deadline and the budget is nearly exhausted by one retried step.

With the patterns above, the charge step is wrapped in `guarded_step` with the run's key. The first attempt times out after the remaining deadline (roughly 22 seconds), the retry reuses the same key, and the payment API returns the original result instead of charging again. If the retry also times out, the run fails with `StepFailed` and the caller receives a clear error rather than a silent duplicate. The budget is charged once for the summarize step, not twice.

The difference is not the model or the provider. It is whether the orchestrator has a transaction boundary.

## Decision checklist before adding another agent

- Does every external call go through a guarded step? If not, add the wrapper first.
- Is the idempotency key derived from the request payload, not the attempt? Confirm with the retry test above.
- Is the deadline absolute and shared across nested calls, or is each call given its own fixed timeout? The latter lets chains exceed the SLA.
- Is fan-out bounded per run and per provider? Check the semaphore values against the provider's documented rate limits.
- Does the graph branch per provider, or is the model a parameterized step? Branching multiplies the step names and breaks key stability.
- Are token counts and run durations logged as structured fields? Without them, budget regressions are invisible until the bill arrives.

## Frequently Asked Questions

**How do you set a timeout for an LLM call?**
Set it from the remaining run deadline, not as a fixed constant. Pass `ctx.remaining()` to the HTTP client or `asyncio.wait_for`. A fixed timeout per call lets a chain of slow calls exceed the overall SLA even though each call individually succeeded.

**Should retries re-run the whole agent graph or just the failed step?**
Just the failed step, and only if the step is idempotent. Re-running the graph duplicates every prior side effect. If a step is not idempotent, make it so or fail the run and let the caller retry with a new idempotency key.

**How do you stop an agent from looping forever?**
Bound the number of steps per run and the total token budget. A loop that keeps calling tools will eventually hit the budget and raise. Log the step count so you can see loops forming before they exhaust the budget.

**Does routing across models break tool calling?**
It can, because providers differ in how they represent tool calls. Normalize tool-call schemas at the step boundary so the graph sees one shape. Test the normalization with a fake for each provider before routing live traffic.

## Where to go from here

The next 30 minutes: open your orchestrator's main entry point and add a `RunContext.from_request` call at the top, then pass it to your first model call. Do not refactor the whole graph — just make the deadline and budget visible in one place. Once that compiles and the existing tests pass, you have the seam you need to add idempotency keys and bounded fan-out without rewriting the agent logic.
