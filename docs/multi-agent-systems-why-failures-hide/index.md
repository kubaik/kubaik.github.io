# Multi-agent systems: why failures hide

The interesting failure question in a multi-agent system is rarely answered in the FAQ. It lives in the incident log. Most write-ups stop exactly where the interesting part starts.

## The problem: failures that do not announce themselves

Multi-agent systems are usually pitched as a way to parallelise work: one agent plans, another retrieves, a third acts, and a coordinator stitches the results together. The failure modes that matter rarely show up in a single request. They show up after thousands of requests, when a retry loop has quietly tripled token spend, when two agents are deadlocked waiting on each other's output, or when a coordinator keeps re-dispatching the same subtask because a downstream agent returns a plausible-looking but wrong answer.

A common trap is treating agent orchestration like a normal microservice call graph. It is not. A microservice that returns HTTP 500 is obviously broken. An agent that returns a fluent, confident paragraph is not obviously broken, even when it is completely wrong. That asymmetry is the root of why these failures hide.

Three properties make early detection hard. First, agents are stochastic: the same input can produce different tool calls on successive runs, so a test that passed once proves little. Second, failures are often silent — a hallucinated field propagates through three agents before a human notices. Third, cost and latency failures compound: a 200 ms per-call overhead looks fine until there are 12 agents and a retry storm turns it into 4 seconds p99.

The observability you already have — request logs, error rates, p95 latency — is blind to most of this. What is needed is per-agent tracing, semantic validation, and budget enforcement. That is what this article covers.

## Prerequisites and what you will build

You will build a small but realistic multi-agent pipeline with three roles: a planner, a retriever, and a synthesizer. The coordinator dispatches subtasks, enforces a per-request token budget, and records a trace span per agent call. The pipeline is then deliberately broken in the ways that happen in production, and each break is mapped to the signal that catches it.

Assumptions: you are comfortable with Python, have used an LLM API before, and can run Docker. The examples use Python 3.12, FastAPI, the OpenTelemetry SDK, Redis, and pytest. The LLM calls are stubbed behind an interface so any provider can be swapped in; the orchestration logic is what matters here.

You will need:

- Python 3.12 with `uv` or `pip`
- Redis running locally (`docker run -p 6379:6379 redis:7.2`)
- An OpenTelemetry collector, or just the console exporter for local runs
- Around 30 minutes to get the skeleton running

The end result is a pipeline that fails loudly and early instead of silently and expensively. The key design decision, and the one worth defending: put the budget check and the schema check inside the coordinator, not inside each agent. Agents should be dumb executors. If each agent independently decides whether it is over budget, you get inconsistent enforcement and no single place to reason about cost.

A comparison of where to put guardrails:

| Guardrail location | Catches early? | Cost of change | Failure visibility |
|---|---|---|---|
| Inside each agent | No — each sees partial state | High, N places to edit | Fragmented logs |
| In the coordinator | Yes — sees full graph | Low, one place | Single trace |
| In a sidecar proxy | Partially — no semantic view | Medium | Network-level only |
| Post-hoc batch audit | No — too late | Low | Delayed by hours |

The coordinator is the only component with the full picture, so that is where enforcement belongs.

## Step 1 — set up the environment

Before writing orchestration code, get tracing wired up. The reason is simple: if tracing is added after the pipeline works, it gets added around the happy path, and the failure paths — exactly the ones that need to be visible — are missed.

Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "fastapi==0.115.*" "redis==5.0.*" \
  "opentelemetry-sdk==1.27.*" \
  "opentelemetry-exporter-otlp==1.27.*" \
  "pytest==8.2.*" "pydantic==2.9.*"
```

Now set up a tracer that exports to the console so spans are visible without a collector. The important detail is that each agent call gets its own span with attributes for token count, latency, and a semantic hash of the output. That semantic hash is what lets you detect the silent-wrong-answer failure mode later.

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agent-pipeline")
```

A gotcha worth flagging now: `SimpleSpanProcessor` is synchronous and will slow down the pipeline noticeably under load. Use `BatchSpanProcessor` in anything resembling production. It is kept simple here only so spans are readable in the terminal.

Finally, start Redis and confirm it is reachable. It is used for two things: a per-request token counter and a short-lived cache of agent outputs keyed by a hash of the input, which cuts duplicate work when the coordinator retries.

```bash
docker run --rm -p 6379:6379 redis:7.2
redis-cli ping   # expect: PONG
```

## Step 2 — core implementation

Now the coordinator. The design principle: every agent call goes through one function that checks budget, records a span, validates output against a Pydantic model, and writes to Redis. No agent calls the LLM directly.

```python
# coordinator.py
import hashlib, json, time
from typing import Any
from pydantic import BaseModel, ValidationError
from tracing import tracer
from redis_client import r

TOKEN_BUDGET = 40_000  # per request, across all agents

class AgentResult(BaseModel):
    content: str
    confidence: float
    sources: list[str] = []

def semantic_hash(text: str) -> str:
    # Normalise whitespace so trivial formatting diffs don't defeat caching
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]

async def call_agent(agent_fn, name: str, payload: dict, request_id: str) -> AgentResult:
    key = f"budget:{request_id}"
    used = int(r.get(key) or 0)
    if used >= TOKEN_BUDGET:
        raise RuntimeError(f"token budget exceeded for {request_id}: {used}")

    with tracer.start_as_current_span(f"agent.{name}") as span:
        start = time.perf_counter()
        raw = await agent_fn(payload)
        latency_ms = (time.perf_counter() - start) * 1000

        try:
            result = AgentResult.model_validate(raw)
        except ValidationError as e:
            span.set_attribute("agent.schema_error", str(e)[:200])
            raise

        r.incrby(key, raw.get("tokens", 0))
        span.set_attribute("agent.tokens", raw.get("tokens", 0))
        span.set_attribute("agent.latency_ms", latency_ms)
        span.set_attribute("agent.semantic_hash", semantic_hash(result.content))
        span.set_attribute("agent.confidence", result.confidence)
        return result
```

The budget check reads from Redis rather than a local variable because the coordinator may run on multiple workers. This is the first place a naive implementation fails: if the counter lives in process memory, two workers each think they have the full budget and you overspend by 2x. That factor is arithmetic, not a measurement: two independent counters, each allowed to reach the same ceiling, permit twice the intended total.

Now the graph. The planner produces subtasks, the retriever fetches sources, the synthesizer combines them. Each is called through `call_agent`.

```python
async def run_pipeline(query: str, request_id: str) -> dict:
    plan = await call_agent(planner, "planner", {"query": query}, request_id)
    subtasks = json.loads(plan.content)["subtasks"][:5]  # cap fan-out

    retrieved = []
    for i, task in enumerate(subtasks):
        out = await call_agent(
            retriever, "retriever", {"task": task, "idx": i}, request_id
        )
        retrieved.append(out.content)

    final = await call_agent(
        synthesizer,
        "synthesizer",
        {"query": query, "context": retrieved},
        request_id,
    )
    return {"answer": final.content, "confidence": final.confidence}
```

That `[:5]` cap is not cosmetic. A planner that decides to emit 40 subtasks is a frequently reported cause of runaway cost in agent postmortems. Capping fan-out at the coordinator is cheap insurance.

## Step 3 — handle edge cases and errors

This is where multi-agent systems differ most from normal services. Four failure modes matter, and each needs a different signal.

**Deadlock.** Two agents each wait for the other's output. In a synchronous pipeline this shows up as a request that never returns. The fix is a hard timeout on every `call_agent`, plus a global request deadline. A common failure here is a timeout that is set per-agent but not enforced across the graph, so 12 agents each taking 29 seconds under a 30-second limit still blows past any sane client timeout. Set a wall-clock deadline for the whole request and check it before every dispatch.

**Retry storm.** The coordinator retries a failed agent, the retry fails, and each retry re-runs the planner because the whole graph is retried. Token usage triples. The fix is idempotency keys plus caching: hash the agent input, check Redis, and only call the model on a miss. With a 60-second TTL you absorb most retry storms without serving stale answers.

**Silent wrong answers.** The synthesizer returns a confident paragraph citing a source that the retriever never returned. Schema validation passes because the shape is correct. The only signal is cross-checking sources — verify that every source string in the synthesizer's output appears in the retriever's outputs. If it does not, fail the request rather than returning it.

**Schema drift.** An agent that used to return `confidence` as a float starts returning it as a string after a prompt change. Pydantic catches this immediately, which is exactly why validation belongs in the coordinator. Without it, the drift propagates and you get a `TypeError` three layers down with no useful context.

A realistic error you will see from the budget check:

```
RuntimeError: token budget exceeded for req_8f3a: 41208
```

If that fires in production, it usually means either the fan-out cap is too high or an agent is looping. Check the trace for repeated `agent.retriever` spans with the same `semantic_hash` — that is the signature of a loop.

## Step 4 — add observability and tests

Observability for agents is not the same as observability for services. Three additions matter: per-agent token accounting, semantic-hash tracking, and a confidence floor.

Track semantic hashes across a request. If the same hash appears more than twice for the same agent, you have a loop. This is a two-line check and it catches the most expensive failure mode.

```python
from collections import Counter

def check_for_loops(spans: list[dict]) -> list[str]:
    warnings = []
    per_agent = {}
    for s in spans:
        if not s["name"].startswith("agent."):
            continue
        per_agent.setdefault(s["name"], []).append(s["attributes"]["agent.semantic_hash"])
    for name, hashes in per_agent.items():
        dupes = [h for h, c in Counter(hashes).items() if c > 2]
        if dupes:
            warnings.append(f"loop suspected in {name}: {dupes}")
    return warnings
```

For tests, do not try to assert on exact model output. Assert on invariants: budget never exceeded, no duplicate semantic hashes beyond threshold, every cited source present in retrieved context, and p95 latency under a threshold. These are stable across model versions, which output-string assertions are not.

```python
import pytest

@pytest.mark.asyncio
async def test_budget_enforced(monkeypatch):
    monkeypatch.setattr("coordinator.TOKEN_BUDGET", 100)
    with pytest.raises(RuntimeError, match="token budget exceeded"):
        await run_pipeline("test query", "req_test")

@pytest.mark.asyncio
async def test_no_loop_on_typical_query():
    result = await run_pipeline("what is redis persistence", "req_ok")
    assert result["confidence"] >= 0.5
```

Run these under `pytest -x --asyncio-mode=auto`. The budget test is the one that pays for itself; it is the only test that reliably catches an unbounded fan-out before it hits production.

## How to measure this on your own pipeline

Rather than trusting numbers from someone else's stack, instrument the pipeline and read the distributions yourself. What to capture:

1. **Per-span attributes.** Already in the code above: `agent.tokens`, `agent.latency_ms`, `agent.semantic_hash`, `agent.confidence`. Export these to a store you can query (the console exporter is fine for a smoke test, but you want a backend that supports aggregation).
2. **Request-level totals.** Sum `agent.tokens` across all spans sharing a `request_id`. That sum, not the per-call figure, is what the budget governs.
3. **Latency percentiles.** Compute p50, p95, and p99 over the *request*, not the span. A pipeline's tail is dominated by fan-out and retries, so span-level percentiles will flatter you.
4. **Loop indicator.** Count duplicate `semantic_hash` values per agent per request. A run of three or more identical hashes is the loop signature.
5. **Source-integrity rate.** Fraction of synthesizer outputs whose cited sources all appear in the retriever outputs. This is the metric that catches hallucinated citations, and it should be near 1.0.

To compare configurations, run the same fixed query set through the pipeline with and without each guardrail, and diff the request-level token sum and p99 latency. That is a controlled comparison; anything else is a guess. The distribution of tokens per request is typically long-tailed, so report the median and the 95th percentile rather than the mean.

## Common questions and variations

**Does this work with more than three agents?** Yes, but fan-out grows fast. Cap the total number of agent calls per request, not just subtasks. A hard cap of 15 calls per request is a reasonable starting point; raise it only when traces show the extra calls are productive.

**What if the agents run on different providers?** The coordinator pattern is provider-agnostic as long as each agent returns the same Pydantic shape. Normalise at the agent boundary, not in the coordinator, so the coordinator stays simple.

**How do I handle streaming?** Streaming complicates budget enforcement because the token count is not known until the stream ends. Count tokens incrementally and abort the stream if the budget is exceeded mid-flight. This is awkward but necessary — running streaming agents without a hard abort is asking for an unbounded bill.

**Is Redis the right place for the counter?** For low-to-moderate traffic, yes. At high request rates the Redis round-trip becomes a bottleneck and you want a local counter with periodic flush, accepting slightly loose enforcement in exchange for throughput. The exact crossover depends on your Redis latency and request volume; measure it rather than assuming.

## Decision checklist before you ship

- [ ] Every agent call goes through a single dispatch function.
- [ ] The dispatch function checks the token budget before calling the model.
- [ ] A wall-clock deadline is captured at request start and checked before every dispatch.
- [ ] Fan-out is capped, both per-plan and in total calls per request.
- [ ] Agent output is validated against a schema at the coordinator boundary.
- [ ] Agent inputs are hashed and cached so retries do not re-run the graph.
- [ ] Semantic hashes are recorded per span and duplicates are flagged.
- [ ] Cited sources are cross-checked against retrieved sources.
- [ ] Tests assert on invariants, not on exact model output.
- [ ] Token and latency distributions are exported and reviewed per release.

## Where to go from here

The one thing to do in the next 30 minutes: open your coordinator (or wherever you dispatch agent calls) and add a wall-clock deadline check before every dispatch. If there is no coordinator, add the check at your top-level handler. The exact line is a comparison of `time.monotonic()` against a deadline captured at request start, raising if exceeded. That single check catches deadlocks and runaway fan-out, which together account for most of the expensive agent failures described in postmortems. Do it before you add anything else.
