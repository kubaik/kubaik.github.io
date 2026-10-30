# Agent platform abstractions: a case study

## The failure mode nobody instruments for

Most agent incidents do not come from the model reasoning badly. They come from the plumbing underneath the agent being unobservable. A tool returns an empty list, the agent reads it as a definitive answer, and a downstream write lands in the wrong place. Nothing in the monitoring stack fires, because from the metric layer's point of view the request succeeded.

A representative scenario: an internal agent reconciles payment exceptions across a legacy ledger, a webhook consumer, and a reconciliation table. The tool set is small — `query_ledger`, `query_stripe`, `query_recon`, `post_adjustment` — and the happy path works in staging. In production, three failure classes dominate:

1. **Silent tool failures.** `query_ledger` returns an empty list after a 504 from upstream. The agent interprets "no rows" as "no matching transaction" and proceeds. There is no distinction between "not found" and "could not check."
2. **Non-idempotent writes.** `post_adjustment` is called twice after a network blip. The ledger accepts both. The duplicate is discovered days later when a customer reports a double credit.
3. **No replayability.** When the agent makes a bad decision, the only artifact is a text transcript. Reproducing the run requires replaying the same prompt against a live database whose contents have since changed.

A common name for the first class is **the confident empty result**. A tool returns `[]` or `None`, the agent treats it as an answer, and the downstream action is wrong. This is not a model problem; it is a contract problem. A signature like `list[Transaction]` cannot express `list[Transaction] | Unknown | Error`, so the agent has no way to branch on the distinction.

The instinctive fix — wrap everything in one Python function with `try/except` and call the model only for classification — reduces incidents but pushes complexity into hundreds of lines of imperative branching. Maintenance cost goes up, not down. Moving logic out of the agent and into hand-written code is not an abstraction; it is a relocation.

## The reframe: workflow with typed steps

The productive shift is to stop treating the agent as a program and start treating it as a **workflow with typed steps**. Three platform-level abstractions carry most of the reliability:

1. **A typed tool contract with explicit outcome states.** Every tool returns a discriminated union: `Ok(value)`, `NotFound`, `TransientError(retryable=...)`, or `FatalError`. The agent cannot proceed on a transient error without a retry decision, and cannot treat `NotFound` as `Ok([])`.
2. **A durable execution layer.** The agent loop moves into a workflow engine. Each tool call becomes an activity with automatic retries, timeouts, and idempotency keys. The workflow itself is deterministic; the model call is an activity with a recorded input/output pair.
3. **A structured trace store.** Every step writes a row containing the run identifier, step index, tool name, input hash, output hash, latency, and outcome. Replay becomes mechanical: re-run the workflow from step N with the recorded inputs.

The key insight is that the agent's **reasoning** does not need to be durable — only its **effects** do. The model can be called fresh on replay; tool results are cached. This makes debugging a matter of reading a table rather than a transcript.

## Implementation: a typed tool contract

The typed contract is the smallest change with the largest effect. In Python 3.11, using `typing.Literal` and dataclasses:

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

The agent's system prompt must then include the schema and an explicit rule: "If a tool returns `transient` with `retryable=True`, call it again, up to 3 times. If it returns `not_found`, do not guess." That single rule removes the confident-empty-result class of bugs, because the ambiguity is gone before the model ever sees the result.

## Implementation: durable execution

Wrapping each tool call as a workflow activity moves retries, timeouts, and replay out of the agent loop. The example below uses the Temporal Python SDK shape; the same structure applies to any durable execution engine.

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

Note the deliberate `maximum_attempts=1` on the activity. Retries are handled in the workflow loop, not by the engine's activity retry policy. This matters: retrying in two places multiplies the effective attempt count. With three activity attempts and three workflow iterations, a flaky upstream sees nine calls instead of three — a 3x amplification that can trip a downstream rate limit.

The idempotency key for writes is derived from `run_id + step_index + tool_name`, hashed with SHA-256, and sent as an `Idempotency-Key` header (or stored in a dedupe table with a unique constraint if the downstream API does not support one). Duplicate calls then return the original response instead of creating a second effect.

## Implementation: the trace table

Deliberately boring schema:

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

A composite index on `(run_id, step_index)` keeps lookups fast as the table grows; the exact index type and retention window depend on volume and compliance requirements. Retention of 30 days is a common starting point for operational data, but the right number is whatever satisfies the audit and debugging windows for the specific domain.

## How to measure whether any of this helped

Benchmark tables for this pattern are usually fabricated, so treat any published one with suspicion. The honest approach is to instrument the four numbers that matter and compare before and after on the same workload:

- **Incidents per week, split by cause.** Tag each incident as `infra` (timeout, duplicate write, parse failure) or `reasoning` (model chose the wrong action given correct inputs). The ratio is the signal. A healthy system has most failures in the `reasoning` bucket, because that is the part you cannot engineer away.
- **Replay cost per failed run.** Measure by summing the token cost of steps re-executed during a replay. With a trace store, replay re-runs only the steps after the divergence point, so cost scales with distance from the failure, not with total run length.
- **Duplicate-effect rate.** Count rows in the downstream system that share an idempotency key. This should be zero by construction. If it is not, the key derivation or the downstream dedupe is broken.
- **p95 end-to-end workflow latency.** Measure at the workflow boundary, not the model boundary. Retry sleeps and activity scheduling are part of the user-visible latency.

The command-level version: run the same 100 recorded inputs through the old loop and the new workflow, diff the trace tables, and count how many runs produce a different terminal outcome. That diff is the actual regression suite.

## Failure modes that survive the refactor

Typed contracts and durable execution do not eliminate every class of bug. Four that commonly remain:

**Double retry.** Retries configured in both the tool and the workflow engine. Fix: exactly one retry layer, and it should be the one that can see the whole run.

**Idempotency key collisions.** If the key is derived from a value that is not stable across retries (a timestamp, a random UUID generated inside the tool), deduplication silently fails. Fix: derive the key from workflow-level identifiers that are recorded before the call.

**Trace table growth.** Without retention and partitioning, the trace table becomes the largest object in the database and slows every query. Fix: partition by `created_at` and drop old partitions rather than running `DELETE`.

**Non-deterministic workflow code.** If the workflow body reads the wall clock, generates randomness, or calls the network directly, replay produces a different execution path than the original. Fix: all side effects go in activities; the workflow body stays pure.

## Decision checklist

Before adding a tool to an agent, answer these:

- What does this tool return when the upstream is down? When the record does not exist? When the request times out? If the answer is "an empty list" or `None` for more than one of those cases, the contract is ambiguous.
- Is this tool's effect idempotent, or does it need an idempotency key?
- If this tool fails mid-run, can the run be replayed from the failure point without re-executing earlier steps?
- Where does the retry live — tool, workflow, or both? (The answer should be exactly one.)
- What row does this tool write to the trace table, and can a failed run be reconstructed from those rows alone?

## FAQ

**How do I stop an agent from hallucinating tool results?**

Make the tool contract explicit. If a tool can return "not found" or "transient error," those must be distinct types the agent can branch on, not empty lists. A discriminated union like `Ok | NotFound | TransientError` plus a system-prompt rule that says "do not guess on `NotFound`" removes most of this class of bug. The model is not the problem; the ambiguous contract is.

**Why does an agent retry a write and double-charge a customer?**

Because the write is not idempotent. The fix is an idempotency key derived from workflow-level identifiers, sent as a header the downstream API respects or stored in a dedupe table with a unique constraint. Retries are safe only when the effect is idempotent.

**What is the best way to debug a failed agent run?**

Query a structured trace, not a text log. A table with one row per step — input hash, output hash, outcome, latency — lets you replay the exact run against recorded data. Re-running against a live database is not debugging; it is gambling.

**Do I need a full workflow engine, or can I use a simple Python loop?**

A simple loop works until you need retries that survive process restarts, timeouts that survive deploys, or replayability. The moment you need any of those, you are rebuilding a workflow engine badly. Durable execution engines exist precisely for this; pick one and stop hand-rolling.

## The broader lesson

Agent reliability is a platform property, not a prompt property. Teams that try to fix agent failures by adding more instructions to the system prompt are optimizing the wrong layer. A system prompt cannot enforce idempotency, cannot distinguish a timeout from an empty result, and cannot replay a failed run.

The three abstractions that matter — typed outcome contracts, durable execution, structured traces — are not specific to agents. They are the same abstractions that made microservices reliable: typed RPC contracts, workflow engines, and distributed tracing. An agent is just another service that happens to make non-deterministic decisions. Treat it like one.

The corollary is that agent frameworks are not the bottleneck. They are fine for prototyping. The bottleneck is the platform underneath them, and the platform is what makes failures cheap to find and cheap to fix.

If you do one thing in the next 30 minutes: open your agent's tool definitions and list every return value each tool can produce. If any tool can return an empty list for more than one reason, write the discriminated union for that tool now. That single file is where the reliability work starts.
===END===
