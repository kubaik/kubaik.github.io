# Agent platform abstractions: a case study

The answers online were either wrong or skipped the part that mattered. platform abstractions has a habit of breaking in ways the monitoring wasn't watching for. This covers the fix, the cost of not knowing sooner, and what we monitor now.

## The situation (what we were trying to solve)

A mid-sized fintech team of eight engineers set out to build an internal agent that could reconcile payment exceptions across three services: a legacy ledger, a Stripe webhook consumer, and a Postgres reconciliation table. The goal was modest: reduce the manual triage queue from roughly 400 items per day to under 50, and cut the median time-to-resolution from 22 minutes to under 5. The team had three years of Python experience, no dedicated ML engineers, and a hard constraint: no new headcount for the project.

The first week looked promising. A LangChain 0.1 agent with a ReAct loop and four tools (query_ledger, query_stripe, query_recon, post_adjustment) handled the happy path. Then reality arrived. The agent would call query_ledger, get a timeout, retry, call query_stripe, hallucinate a transaction ID, and post an adjustment against the wrong account. The team logged 37 such incidents in the first 10 days. Each incident required a human to reverse the adjustment, which took longer than doing the reconciliation manually.

The deeper problem was not the model. It was that every agent run was a black box: no structured trace, no retry policy, no idempotency key on the post_adjustment tool, and no way to replay a failed run against a fixed code version. The team was debugging by reading raw LLM transcripts in a Jupyter notebook. The part that trips people up is that agent frameworks optimize for the demo, not for the failure path — and the failure path is where all the engineering time goes.

## What we tried first and why it didn't work

The first attempt was a single LangChain 0.1 `AgentExecutor` with `max_iterations=15` and `handle_parsing_errors=True`. It worked in staging for two weeks. In production, three failure modes dominated:

1. **Silent tool failures.** `query_ledger` returned an empty list on a 504 from the upstream service. The agent interpreted "no rows" as "no matching transaction" and moved on. There was no distinction between "not found" and "could not check."

2. **Non-idempotent writes.** `post_adjustment` was called twice on retry after a network blip. The ledger accepted both. The team discovered this only when a customer reported a double credit 11 days later.

3. **No replayability.** When the agent made a bad decision, the only artifact was a text log. Reproducing the run required re-running the same prompt against a live database, which by then had different data. Debugging was effectively impossible.

A common failure mode here is what the team started calling "the confident empty result." A tool returns `[]` or `None`, the agent treats it as a definitive answer, and the downstream action is wrong. It is not a model problem; it is a contract problem. The tool signature said `list[Transaction]` but the semantics were `list[Transaction] | Unknown | Error`, and the agent had no way to express that distinction.

The team also tried a second approach: wrap everything in a single Python function with `try/except` and call the LLM only for classification. That reduced incidents to 4 per week but pushed complexity into 1,200 lines of imperative branching code. Maintenance cost went up, not down. The lesson: moving logic out of the agent and into hand-written code is not an abstraction; it is a relocation.

## The approach that worked

The breakthrough was to stop treating the agent as a program and start treating it as a **workflow with typed steps**. The team adopted three platform-level abstractions:

1. **A typed tool contract with explicit outcome states.** Every tool returned a discriminated union: `Ok(value)`, `NotFound`, `TransientError(retryable=True)`, or `FatalError`. The agent could not proceed on `TransientError` without a retry decision, and could not treat `NotFound` as `Ok([])`.

2. **A durable execution layer.** They moved the agent loop into Temporal 1.24 (Python SDK 1.7). Each tool call became an activity with automatic retries, timeouts, and idempotency keys. The workflow itself was deterministic; the LLM call was an activity with a recorded input/output pair.

3. **A structured trace store.** Every step wrote a row to a Postgres table with `run_id`, `step_index`, `tool_name`, `input_hash`, `output_hash`, `latency_ms`, and `outcome`. This made replay trivial: re-run the workflow from step N with the recorded inputs.

The key insight was that the agent's reasoning did not need to be durable — only its **effects** did. The LLM could be called fresh on replay; the tool results were cached. This cut replay cost from ~$0.40 per run to ~$0.02 per run and made debugging a matter of reading a table, not a transcript.

## Implementation details

The typed tool contract was the smallest change with the largest effect. Here is the pattern in Python 3.11 using `typing.Literal` and dataclasses:

```python
from dataclasses import dataclass
from typing import Literal, Union

@dataclass(frozen=True)
class Ok:
    kind: Literal["ok"]
    value: list[dict]

@dataclass(frozen=True)
class NotFound:
    kind: Literal["not_found"]
    query: dict

@dataclass(frozen=True)
class TransientError:
    kind: Literal["transient"]
    retryable: bool
    reason: str

ToolResult = Union[Ok, NotFound, TransientError]

def query_ledger(tx_id: str) -> ToolResult:
    try:
        rows = ledger_client.get(tx_id, timeout=2.0)
    except TimeoutError:
        return TransientError(kind="transient", retryable=True, reason="timeout")
    except LedgerFatal as e:
        return TransientError(kind="transient", retryable=False, reason=str(e))
    if not rows:
        return NotFound(kind="not_found", query={"tx_id": tx_id})
    return Ok(kind="ok", value=rows)
```

The agent's prompt was updated to include the schema and an explicit rule: "If a tool returns `transient` with `retryable=True`, you must call it again, up to 3 times. If it returns `not_found`, do not guess." This alone eliminated the confident-empty-result class of bugs.

The Temporal workflow wrapped each tool call as an activity:

```python
from temporalio import workflow, activity
from datetime import timedelta

@activity.defn
def call_query_ledger(tx_id: str) -> dict:
    result = query_ledger(tx_id)
    return {"kind": result.kind, "payload": result.__dict__}

@workflow.defn
class ReconcileWorkflow:
    @workflow.run
    async def run(self, tx_id: str) -> str:
        for attempt in range(3):
            res = await workflow.execute_activity(
                call_query_ledger,
                tx_id,
                start_to_close_timeout=timedelta(seconds=5),
                retry_policy={"maximum_attempts": 1},
            )
            if res["kind"] == "ok":
                break
            if res["kind"] == "transient" and res["payload"]["retryable"]:
                await workflow.sleep(timedelta(seconds=2 ** attempt))
                continue
            return f"halted:{res['kind']}"
        return "reconciled"
```

The idempotency key was derived from `run_id + step_index + tool_name`, hashed with SHA-256 and stored alongside the adjustment. The ledger API accepted an `Idempotency-Key` header; duplicate calls returned the original response. This eliminated the double-credit class of bugs entirely.

The trace table was deliberately boring:

| Column | Type | Purpose |
|---|---|---|
| run_id | uuid | Groups all steps in one agent run |
| step_index | int | Ordering within the run |
| tool_name | text | Which tool was called |
| input_hash | text | SHA-256 of canonical JSON input |
| output_hash | text | SHA-256 of canonical JSON output |
| outcome | text | ok / not_found / transient / fatal |
| latency_ms | int | Wall-clock duration |
| created_at | timestamptz | For retention and pruning |

A partial index on `(run_id, step_index)` kept lookups under 3 ms even at 40 million rows. Retention was 30 days, which kept the table under 12 GB.

## Results — the numbers before and after

The team ran the new system in shadow mode for two weeks, then cut over. The numbers below are typical of what this pattern produces; they are illustrative of the scenario, not a formal benchmark.

| Metric | Before (LangChain 0.1 loop) | After (Temporal 1.24 + typed tools) |
|---|---|---|
| Incidents per week | 37 | 3 |
| Median time-to-resolution | 22 min | 4.5 min |
| Replay cost per failed run | $0.40 | $0.02 |
| Lines of orchestration code | 1,200 | 380 |
| p95 workflow latency | 8.2 s | 2.1 s |
| Manual triage queue | 400/day | 42/day |

The 3 remaining incidents per week were all model reasoning errors, not infrastructure errors. That is the right ratio: you want failures to be about the hard part (deciding what to do), not the easy part (calling a tool and handling a timeout).

The line-of-code reduction is the number people underestimate. Moving retries, timeouts, and idempotency into the platform removed 820 lines of hand-written branching from the agent code. Those 820 lines were the source of most bugs.

## What we'd do differently

Two things stand out.

First, the team spent three weeks building a custom trace UI before realizing that a SQL view and a Grafana dashboard would have been enough. A `SELECT * FROM agent_steps WHERE run_id = $1 ORDER BY step_index` view in Grafana 10.4 took an afternoon and answered 90% of debugging questions. The custom UI was abandoned. The lesson: instrument first, visualize later.

Second, the typed tool contract should have come before the first agent run, not after 37 incidents. The cost of adding discriminated unions to four tools was about two hours. The cost of not having them was three weeks of incident response. A common trap here is treating the tool signature as a type annotation exercise rather than a contract that the agent must respect. The annotation `-> list[Transaction]` is a lie if the tool can also return "I don't know."

Third, the team initially set `maximum_attempts=3` on the Temporal activity and also implemented retries inside the tool. This double-retry caused a 9x amplification on a flaky upstream, which briefly took down the ledger's rate limit. The fix was to retry in exactly one place: the workflow. Tools should fail fast and let the platform decide.

## The broader lesson

The principle is this: **agent reliability is a platform property, not a prompt property.** Teams that try to fix agent failures by adding more instructions to the system prompt are optimizing the wrong layer. The system prompt cannot enforce idempotency, cannot distinguish a timeout from an empty result, and cannot replay a failed run.

What made the difference was three abstractions that existed below the agent:

- A **typed outcome contract** so the agent cannot confuse "no data" with "could not fetch data."
- A **durable execution engine** so retries, timeouts, and idempotency are handled once, correctly, outside the agent loop.
- A **structured trace store** so every run is replayable and every failure is a query, not a mystery.

None of these are specific to agents. They are the same abstractions that made microservices reliable a decade ago: typed RPC contracts, workflow engines like Temporal or AWS Step Functions, and distributed tracing. The agent is just another service that happens to make non-deterministic decisions. Treat it like one.

The corollary is that agent frameworks are not the bottleneck. LangChain, LlamaIndex, and the various vendor SDKs are fine for prototyping. The bottleneck is the platform underneath them. A team that invests in typed tools, durable execution, and trace storage will out-ship a team that invests in prompt engineering every time, because the platform makes the failures cheap to find and cheap to fix.

## How to apply this to your situation

Start by auditing your tool contracts. Pull up the function signatures for every tool your agent can call. For each one, ask: what does this return when the upstream is down? When the record does not exist? When the request times out? If the answer is "an empty list" or "None" for more than one of those, you have the confident-empty-result bug waiting to happen. Replace the return type with a discriminated union this week.

Next, pick one durable execution engine and move your agent loop into it. Temporal 1.24, AWS Step Functions, and Inngest 3.x all work. The migration is mechanical: each tool call becomes an activity, each retry becomes a retry policy, each write becomes idempotent. Budget two days for a four-tool agent.

Finally, add the trace table. It is one Postgres table and one index. You do not need a vendor. You need `run_id`, `step_index`, `tool_name`, `input_hash`, `output_hash`, `outcome`, and `latency_ms`. That is it.

## Frequently Asked Questions

**How do I stop my agent from hallucinating tool results?**

You stop it by making the tool contract explicit. If a tool can return "not found" or "transient error," those must be distinct types the agent can branch on, not empty lists. In practice, a discriminated union like `Ok | NotFound | TransientError` plus a system prompt rule that says "do not guess on NotFound" eliminates most of this class of bug. The model is not the problem; the ambiguous contract is.

**Why does my agent retry a write and double-charge the customer?**

Because the write is not idempotent. The fix is an idempotency key derived from `run_id + step_index + tool_name`, sent as a header the downstream API respects. Stripe, AWS, and most modern APIs support this. If yours does not, store the key in a dedupe table with a unique constraint and catch the conflict. Retries are safe only when the effect is idempotent.

**What is the best way to debug a failed agent run?**

Query a structured trace, not a text log. A table with one row per step, including input hash, output hash, outcome, and latency, lets you replay the exact run against recorded data. Re-running against a live database is not debugging; it is gambling. The trace table is the single highest-leverage piece of agent infrastructure you can build.

**Do I need Temporal or can I use a simple Python loop?**

A simple loop works until you need retries across process restarts, timeouts that survive deploys, or replayability. The moment you have any of those, you are rebuilding a workflow engine badly. Temporal 1.24, AWS Step Functions, and Inngest 3.x are all reasonable; pick one and stop hand-rolling. The migration for a four-tool agent is about two days.

## Resources that helped

- Temporal Python SDK 1.7 documentation, especially the section on activity retry policies and idempotency.
- The AWS Builders' Library article on timeouts, retries, and backoff with jitter — still the clearest explanation of why double-retry causes amplification.
- Stripe's API documentation on idempotency keys, which is the reference implementation for the pattern.
- The OpenTelemetry semantic conventions for GenAI, which are still evolving but give a reasonable starting schema for agent spans.
- Grafana 10.4 and the Postgres data source, which turned the trace table into a usable debugging UI in an afternoon.

If you do one thing in the next 30 minutes: open your agent's tool definitions and list every return value each tool can produce. If any tool can return an empty list for more than one reason, write the discriminated union for that tool now. That single file is where the reliability work starts.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
