# Evaluate agents with Spark vs custom harness

## The evaluation harness is part of the system under test

A common failure mode in multi-agent projects is treating the evaluation harness as neutral infrastructure. The harness shapes what you can observe, what you can replay, and how quickly you can tell whether a failure came from the agent or from the test rig. When the harness itself is a distributed system, debugging becomes a distributed-systems problem.

Two broad approaches dominate practice:

- **Dataframe-oriented harnesses** built on a distributed batch/stream engine (Apache Spark is the common choice). Agent interactions are modelled as rows in a table, and each agent is a transformation over those rows.
- **Custom harnesses** written as ordinary application code — typically Python with `pytest` and `asyncio` — where each agent is a function or class and the orchestration is explicit.

Both can work. They fail in different ways, and they impose different costs. This article covers how each works, where each breaks, how to measure the difference on your own workload, and how to choose.

## What a harness actually has to do

Before comparing implementations, separate the responsibilities. A harness that only runs the happy path is a demo, not an evaluation system. A useful harness does at least five things:

1. **Drives the agent chain** with a controlled input, including tool responses that are stubbed or recorded.
2. **Records every intermediate artifact** — prompts, tool calls, tool results, retries, and final output — in a form you can diff between runs.
3. **Asserts on outcomes** that matter: task success, schema validity, latency budget, token spend, and safety constraints.
4. **Replays** a stored conversation against a changed prompt, model, or tool implementation.
5. **Reports failures** with enough context that the failing step is identifiable without re-running.

Any harness that skips (2) or (5) will cost you more in debugging than it saves in automation. This is the axis on which the two approaches actually differ.

## Option A: dataframe-oriented harnesses

In this model, each interaction is a row. Columns typically include an input, an agent identifier, a serialized tool-call payload, and an intermediate state blob. Agents are user-defined functions (UDFs) applied to those rows, and the conversation history is materialized into a table.

A simplified chain looks like this:

```python
from pyspark.sql import functions as F

interactions = spark.read.format("parquet").load("/checkpoints/interactions")

chain_result = (
    interactions
    .withColumn(
        "next_agent",
        F.when(F.col("agent_id") == "user", F.lit("classifier"))
         .when(F.col("agent_id") == "classifier", F.lit("planner"))
         .when(F.col("agent_id") == "planner", F.lit("tool_caller"))
         .otherwise(F.lit("user"))
    )
    .withColumn(
        "new_output",
        F.expr("agent_udf(next_agent, input_text, intermediate_state)")
    )
    .write.format("parquet")
    .mode("append")
    .save("/checkpoints/interactions")
)
```

Note the format: plain Parquet, not a specific table format. If you want snapshot isolation or time travel, that comes from whichever table format your platform provides — evaluate it separately, because it changes the operational story.

**Where this model earns its keep:**

- **Bulk replay.** Re-running a fixed corpus of conversations against a new prompt version is a batch job, and batch engines are good at batch jobs. Partitioning by conversation ID lets the engine skip work whose inputs have not changed, if you structure the job that way.
- **Existing operational tooling.** If your organization already runs a Spark cluster, the harness inherits its scheduler, its metrics, and its on-call rotation. That is a real cost saving, not a hypothetical one.
- **Schema enforcement.** Serializing intermediate state into typed columns forces you to define what an agent actually produces. That discipline catches drift early.

**Where it breaks:**

- **Serialization tax on every tool call.** Each UDF invocation crosses a process or language boundary. For agents that call a fast external service, that overhead can dominate the actual work. It is measurable: instrument wall-clock time inside the UDF versus total task time, and the difference is your serialization and scheduling cost.
- **Statelessness assumptions.** UDFs are expected to be pure functions of their inputs. Agents that hold state across turns — a scratchpad, a growing plan, a session-scoped cache — do not fit that model without awkward checkpointing that reintroduces the problem you were trying to solve.
- **Error attribution.** A stage graph tells you which task failed. It does not tell you why the model produced a bad tool call. You still need the raw prompt and response, which means the harness must log them somewhere queryable, which means you have built a second system.

## Option B: custom Python harnesses

The custom approach treats agents as ordinary async functions and the harness as ordinary test code.

```python
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class Agent:
    name: str
    endpoint: str

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def __call__(self, message: str) -> str:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(
                self.endpoint,
                json={"messages": [{"role": "user", "content": message}]},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

async def run_chain(initial: str, agents: list[Agent]) -> str:
    message = initial
    for agent in agents:
        message = await agent(message)
    return message
```

**Where this model earns its keep:**

- **Full control over retries and timeouts.** Retry policy lives next to the call it governs. When a timeout occurs, the exception carries the request that caused it.
- **Non-JSON tools.** Legacy SQL engines, gRPC services, and paginated APIs are awkward as dataframe columns but trivial as function calls.
- **Testability.** `pytest` fixtures can substitute a fake agent, assert on the exact prompt sent, and run deterministically. Coverage tooling works without modification.

**Where it breaks:**

- **Orchestration sprawl.** Every new agent adds mocks, fixtures, and retry configuration. Without discipline, the harness grows faster than the system it tests.
- **Weak default observability.** A bare `httpx.ReadTimeout` tells you nothing about which step in a five-agent chain timed out, or what the preceding four steps produced. This is a design failure, not an inherent limitation — but it is the default, and defaults win.
- **No built-in replay.** Replaying a recorded conversation requires you to have stored it. If the harness only logs on failure, you cannot reproduce successes that later become failures.

## The comparison that matters: observability, not throughput

Latency and throughput numbers from someone else's workload tell you almost nothing about yours, because both are dominated by your tool latency, your model latency, and your concurrency pattern. What transfers between teams is the shape of the failure.

| Dimension | Dataframe-oriented harness | Custom Python harness |
|---|---|---|
| Bulk replay of a fixed corpus | Natural fit; engine handles partitioning | Requires you to build the runner |
| Per-step latency overhead | Serialization and scheduling add fixed cost per UDF call | Near-zero framework overhead |
| Stateful agents | Awkward; needs external checkpointing | Straightforward |
| Non-JSON tools | Requires wrapping | Direct |
| Failure attribution | Stage/task graph plus whatever you logged | Whatever you logged |
| Onboarding cost | Low if the team already runs the engine | Low for Python teams |
| Operational surface | The cluster, its scheduler, its upgrades | Your process, your deployment target |

The two rows that decide most projects are **failure attribution** and **operational surface**. Teams rarely regret the framework's raw speed. They regret spending a week unable to tell whether an agent was wrong or the harness was.

## How to measure this on your own workload

Do not adopt either approach on the strength of a table. Measure. The procedure below takes an afternoon and produces numbers that are actually yours.

**Step 1 — Build a fixed corpus.** Capture 200–500 real or realistic conversations, including at least twenty known-hard cases. Store each as a record containing the initial input and the recorded tool responses. This corpus is the only thing both harnesses must consume identically.

**Step 2 — Instrument three timings per step.** For every agent invocation, record:

- `t_total` — wall clock from the moment the step is scheduled to the moment its output is available.
- `t_work` — wall clock spent inside the model or tool call.
- `t_framework` — `t_total - t_work`. This is the harness's own cost.

Report the median and the 95th percentile of `t_framework`. This is the single most useful number in the comparison, and it is the one that external benchmarks never give you.

**Step 3 — Count failures by class.** Run the corpus and classify every failure as: model error, tool error, harness error, or timeout. A harness that produces many "harness error" classifications is telling you something about itself.

**Step 4 — Measure time-to-root-cause.** Pick five failures at random. Time how long it takes, using only the harness's artifacts, to state the failing step and the reason. This is subjective but it is the metric that predicts your maintenance burden.

**Step 5 — Measure replay cost.** Change one prompt. Re-run the corpus. Record wall-clock time and compute cost. Compare against the time to run the full corpus from scratch. If replay is not meaningfully cheaper, the replayability argument for the heavier harness does not apply to you.

**Step 6 — Project cost with explicit assumptions.** Cloud cost arithmetic is simple once the assumptions are stated. For a serverless function billed per GB-second, the monthly cost is:

```
cost = invocations × duration_seconds × memory_GB × price_per_GB_second
```

For a provisioned cluster billed per vCPU-hour:

```
cost = vCPUs × hours_per_month × price_per_vCPU_hour × utilization_factor
```

State every input. If you cannot state the utilization factor, you do not yet know your cluster cost — you know its ceiling. Treat any comparison that omits the utilization factor as illustrative rather than measured.

## Failure modes worth designing against

**Text normalization silently destroying signal.** Any harness that normalizes text before an agent sees it — stripping non-ASCII, lowercasing, trimming whitespace — can mask prompt-injection payloads and encoding-based attacks. The failure is invisible because the normalization is intentional. Mitigation: log both the raw and normalized input, and assert that they differ only in ways you intended.

**Retry budgets set to zero.** A retry decorator with `stop_after_attempt(1)`, or a cluster configured to fail fast, turns a transient network blip into a test failure. Teams then spend hours investigating agent logic. Mitigation: assert on retry counts as part of the test, not just on final output.

**Unbounded agent loops.** Multi-agent chains that terminate on model output rather than a step limit can run for dozens of rounds. Mitigation: enforce a hard step cap in the harness and fail the test when it is hit. The cap belongs in the harness, not in each agent.

**Non-determinism mistaken for regression.** Sampling temperature, tool-side caching, and clock-dependent prompts all produce run-to-run variation. Mitigation: pin the seed where the API allows it, and run the corpus three times before declaring a regression.

**Harness drift.** The harness and the production orchestrator diverge until the harness tests a system that no longer exists. Mitigation: have the harness import the production orchestration code rather than reimplementing it.

## A decision checklist

Work through these in order. The first one that applies usually settles it.

1. **Do your agents hold state across turns?** If yes, a stateless-transform harness will fight you. Prefer the custom harness, or budget explicitly for external checkpointing.
2. **Do your agents call non-JSON tools?** If yes, prefer the custom harness.
3. **Does your team already operate a distributed batch engine?** If no, the operational cost of adopting one is usually larger than the engineering cost of writing the harness. Prefer custom.
4. **Is bulk replay of a large fixed corpus a core workflow?** If yes, and the answer to (3) is also yes, the dataframe harness is a reasonable default.
5. **Is your p99 latency budget tight relative to your tool latency?** Measure `t_framework` first. If it is a meaningful fraction of the budget, that decides it.
6. **Do you need to hand artifacts to an auditor?** Both approaches can produce them, but the custom harness requires you to design the export. Budget for that work rather than assuming it is free.

If none of these apply — a single-agent prototype with one model call and no tools — use neither. A handful of `pytest` tests around the one function is sufficient, and adding infrastructure at that stage is pure cost.

## A worked example of the reasoning

Suppose a team runs a three-agent chain: classify, retrieve, summarize. The retrieval step calls an internal service with a documented p99 of 80 ms. The summarization step calls a model with a p99 of 900 ms. The chain's latency budget is 1.5 seconds at p99.

The dominant term is the model call at 900 ms. Framework overhead of 60 ms per step across three steps is 180 ms — about 12% of the budget. That is not free, but it is not the deciding factor either. The deciding factor is what happens when the summarizer times out: with a stateless-transform harness, the failure surfaces as a failed task in a stage graph, and the prompt that caused it must be retrieved separately. With a custom harness, the exception can carry the prompt directly if the code is written that way.

So the recommendation for this team is: choose based on failure attribution, not latency. If they already run a batch engine and their logging is good, either works. If their logging is poor, the heavier harness will not save them, because the harness is not where the logging lives.

That reasoning — identify the dominant term, then ask whether the framework's overhead is material relative to it — generalizes. Run it with your own numbers before adopting anyone's default.

## What to do in the next 30 minutes

Open your current harness and add one instrumentation point: wrap each agent invocation so it records `t_total` and `t_work`, and appends `t_framework` to a list. Run your existing test suite. Print the median and 95th percentile of `t_framework`. If that number is under 5% of your per-step latency budget, framework overhead is not your problem and you should choose on observability grounds instead. If it is over 20%, you have found your constraint — and you found it with your own data rather than someone else's benchmark.
