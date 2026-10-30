# SLOs for agentic features: beyond latency

Agentic features fail in ways that traditional SLOs never catch. A tool-calling agent can return HTTP 200 in 300ms, log zero errors, and still be completely wrong: it picked the wrong tool, hallucinated a parameter, or looped three times before giving up. Error-rate and latency dashboards stay green while users quietly lose trust.

This is a step-by-step guide to defining, implementing, and alerting on SLOs that measure whether the agent did its job, using OpenTelemetry and Prometheus. By the end you'll have a working SLI pipeline that tracks task success, tool-call validity, and loop termination — the three signals that matter most for agents.

## The problem this solves

Classic request/response SLOs assume a binary mapping between "the call returned" and "the work was done." Agents break that assumption in three specific ways.

**Semantic failure with a clean status code.** The model returns a well-formed response that answers the wrong question, calls a tool that exists but is inappropriate, or produces a plausible-looking parameter set that the downstream API rejects silently.

**Partial recovery.** A tool call fails, the agent retries with corrected arguments, and the task succeeds. From the outside this is a success; from a per-call error-rate perspective it is a failure. Any SLI that counts the intermediate error as a failure will be permanently noisy.

**Non-termination.** The agent hits a max-iteration cap or a wall-clock timeout and returns an empty or truncated answer. This looks like a latency or timeout event, but the underlying cause is usually a prompt or tool-design problem, and it needs its own signal.

The rest of this article builds an SLI pipeline that distinguishes these cases. The approach is framework-agnostic: it depends on structured spans, not on any particular agent library.

## Prerequisites and what you'll build

You need:

- An agent loop (any framework — LangChain, LlamaIndex, or hand-rolled) that can emit structured spans.
- An OpenTelemetry SDK for your language. Examples below use Python; the span and metric semantics are the same in Go, Java, or JavaScript.
- A Prometheus-compatible metrics backend (Prometheus itself, Grafana Cloud, or VictoriaMetrics).
- Working familiarity with SLI (indicator), SLO (objective), and error budget.

What you'll build:

1. Three custom SLIs: **task success rate**, **tool-call validity rate**, and **loop termination rate**.
2. A span-instrumentation layer that produces the attributes those SLIs consume.
3. Recording and alerting rules that fire on error-budget burn, not on raw error counts.
4. A test harness that validates the SLI logic against synthetic traces.

Every step below maps to a metric queryable in Grafana within an hour of wiring it up.

## Step 1 — Define what "success" means for your agent

Before writing any code, define a binary outcome per agent invocation. This is the hardest part, and it's where most teams stall.

For a customer-support agent, success might be: the ticket was resolved without escalation AND the user didn't re-open it within 24 hours. For a code-generation agent: the patch compiles AND passes the existing test suite. For a research agent: the final answer cites at least one source from an allowed domain list.

Write this as a function. It doesn't have to be perfect — it has to be consistent. Two common traps are defining success too broadly ("user seemed happy") or too narrowly ("exact string match on expected output"). Both produce SLIs that either never fire or fire constantly.

Here's a workable definition for a tool-using agent:

```python
# sli_definitions.py
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentOutcome:
    task_id: str
    success: bool
    tool_calls_valid: bool
    terminated_cleanly: bool
    error: str | None = None


def evaluate_outcome(trace: dict[str, Any]) -> AgentOutcome:
    """
    Given a completed agent trace, return a binary outcome.
    This is the single source of truth for all SLIs.
    """
    spans = trace["spans"]

    # Success: the agent produced a final answer AND every tool error
    # was followed by a retry that the agent recovered from.
    final_answer = trace.get("final_answer")
    tool_error_indices = [
        i for i, span in enumerate(spans)
        if span["name"].startswith("tool.") and span.get("error")
    ]
    recovered = all(
        any(s["name"] == "llm.retry" for s in spans[i + 1:])
        for i in tool_error_indices
    )
    success = bool(final_answer) and (not tool_error_indices or recovered)

    # Tool-call validity: every tool call had a schema-valid argument set.
    tool_calls = [s for s in spans if s["name"].startswith("tool.")]
    tool_calls_valid = all(
        s.get("attributes", {}).get("args_valid", False)
        for s in tool_calls
    )

    # Termination: the agent stopped because it decided to, not because
    # it hit a max-iteration cap or a timeout.
    terminated_cleanly = trace.get("termination_reason") == "agent_finished"

    return AgentOutcome(
        task_id=trace["task_id"],
        success=success,
        tool_calls_valid=tool_calls_valid,
        terminated_cleanly=terminated_cleanly,
        error=trace.get("error"),
    )
```

This function is the contract. Every SLI below derives from it. If the definition of success changes, it changes here, and the rest of the pipeline follows.

One design note: `tool_calls_valid` uses `all(...)`, so a single invalid argument set marks the whole trace invalid. That is deliberate for a validity SLI — you want to know the rate at which the agent produces schema-correct calls at all. If you'd rather track per-call validity, emit a counter per tool span instead and aggregate at query time.

## Step 2 — Instrument the agent loop with OpenTelemetry

SLIs cannot be computed from unstructured logs alone — they need spans with the attributes `evaluate_outcome` reads. OpenTelemetry is the right substrate because it is vendor-neutral and has SDKs for the languages agents are typically written in.

The key structural decision: emit one root span per agent invocation, one child span per LLM call, and one child span per tool call. That tree is what makes recovery and termination computable.

```python
# agent_instrumentation.py
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


tracer = trace.get_tracer("agent.runtime")


def run_agent(task_id: str, user_input: str, tools: dict) -> str:
    with tracer.start_as_current_span("agent.invoke") as root:
        root.set_attribute("task.id", task_id)
        root.set_attribute("agent.tool_count", len(tools))

        messages = [{"role": "user", "content": user_input}]
        max_iterations = 10
        termination_reason = "max_iterations"

        for iteration in range(max_iterations):
            with tracer.start_as_current_span("llm.call") as llm_span:
                llm_span.set_attribute("llm.iteration", iteration)
                response = call_llm(messages)
                llm_span.set_attribute("llm.stop_reason", response.stop_reason)

            if response.stop_reason == "end_turn":
                termination_reason = "agent_finished"
                root.set_attribute("termination_reason", termination_reason)
                root.set_attribute("final_answer", response.content)
                return response.content

            if response.tool_call:
                tool_name = response.tool_call.name
                with tracer.start_as_current_span(f"tool.{tool_name}") as tool_span:
                    try:
                        args = validate_args(tool_name, response.tool_call.arguments)
                        tool_span.set_attribute("args_valid", True)
                        result = tools[tool_name](**args)
                        tool_span.set_status(Status(StatusCode.OK))
                    except Exception as exc:
                        tool_span.set_attribute("args_valid", False)
                        tool_span.record_exception(exc)
                        tool_span.set_status(Status(StatusCode.ERROR))
                        result = f"Error: {exc}"
                messages.append({"role": "tool", "content": result})

        root.set_attribute("termination_reason", termination_reason)
        root.set_status(Status(StatusCode.ERROR, "max_iterations"))
        return ""
```

Two details matter. First, `args_valid` is set on the tool span itself, which makes the tool-call-validity SLI computable without re-parsing logs. Second, `termination_reason` is set on the root span, which distinguishes "agent finished" from "agent gave up."

A common trap: teams instrument only the LLM call and forget tool spans. They can then measure latency and token usage but not whether the agent used tools correctly. If you use a framework that auto-instruments LLM calls, check whether it also emits tool spans — many do not, and you'll need to add them manually.

A second trap is cardinality. Never put `task.id`, user input, or free-text tool arguments on a span attribute that becomes a metric label. Attributes like `task.id` are fine on spans (they're useful for trace lookup) but must be dropped or aggregated before metrics are derived from them, or the metric backend will explode. The counters in Step 3 use a fixed `agent.name` label for exactly this reason.

## Step 3 — Convert spans to SLIs

Raw spans are not metrics. They must be aggregated into counters Prometheus can scrape. The OpenTelemetry Collector's `spanmetrics` connector handles standard dimensions (latency, error status) but knows nothing about `args_valid` or `termination_reason`. Two options exist:

1. Use the Collector's `transform` processor to rewrite span attributes into metric labels, then feed `spanmetrics`.
2. Emit metrics directly from the application alongside the spans.

For agentic SLIs, option 2 is usually simpler, because the outcome logic already lives in application code. Here's the pattern:

```python
# sli_metrics.py
from opentelemetry import metrics

meter = metrics.get_meter("agent.sli")

task_success = meter.create_counter(
    "agent.task.success.total",
    description="Count of agent tasks that succeeded",
)
task_failure = meter.create_counter(
    "agent.task.failure.total",
    description="Count of agent tasks that failed",
)
tool_valid = meter.create_counter(
    "agent.tool.call.valid.total",
    description="Tool calls with schema-valid arguments",
)
tool_invalid = meter.create_counter(
    "agent.tool.call.invalid.total",
    description="Tool calls with schema-invalid arguments",
)
terminated_clean = meter.create_counter(
    "agent.termination.clean.total",
    description="Agent runs that terminated by decision",
)
terminated_forced = meter.create_counter(
    "agent.termination.forced.total",
    description="Agent runs that hit max iterations or timeout",
)


def record_outcome(outcome):
    attrs = {"agent.name": "support_agent"}
    if outcome.success:
        task_success.add(1, attrs)
    else:
        task_failure.add(1, attrs)
    if outcome.tool_calls_valid:
        tool_valid.add(1, attrs)
    else:
        tool_invalid.add(1, attrs)
    if outcome.terminated_cleanly:
        terminated_clean.add(1, attrs)
    else:
        terminated_forced.add(1, attrs)
```

Call `record_outcome` once per agent invocation, after the root span closes. The counters are cheap; aggregation happens at scrape time in Prometheus.

## Step 4 — Define SLOs and error budgets

Now there are time series. An SLO is a ratio over a rolling window. For a 30-day window at 99% success:

```promql
# SLI: task success rate over 30 days
sum(rate(agent_task_success_total[30d]))
/
sum(rate(agent_task_success_total[30d]) + rate(agent_task_failure_total[30d]))
```

```promql
# Error budget remaining (as a fraction of allowed failures)
1 - (
  (1 - (
    sum(rate(agent_task_success_total[30d]))
    /
    sum(rate(agent_task_success_total[30d]) + rate(agent_task_failure_total[30d]))
  )) / 0.01
)
```

For tool-call validity, the SLO should generally be tighter: invalid arguments are a schema or prompt bug, not a user-facing failure, and they're more actionable. For loop termination, the target depends on the max-iteration setting; if the cap is 10 and the agent routinely needs 12, that's a design problem, not a runtime problem.

| SLI | Illustrative SLO | Rationale |
|-----|-----------------|-----------|
| Task success | 99% | User-facing; failures are visible |
| Tool-call validity | 99.9% | Schema/prompt bug; should be near-zero failure |
| Clean termination | 95% | Some forced terminations are acceptable on hard tasks |

The targets above are illustrative starting points, not documented defaults — calibrate them against your own baseline. The important property is that each SLI has a target and a window, and that the error budget is computed from the same window.

### Choosing a window and a target

Two decisions determine whether an SLO is useful or decorative.

**Window length.** A 30-day rolling window is the common default because it smooths daily traffic variation and makes the error budget meaningful. A 7-day window reacts faster but produces noisier burn-rate alerts. A 1-day window is usually too short for a low-traffic agent: with 500 invocations a day, a single failure moves the success rate by 0.2%, so a 99.9% target would be violated by two failures.

**Target from baseline, not aspiration.** Measure the current success rate for two weeks before setting the target. If the agent is at 97% and the product needs 99%, the SLO should be 99% and the gap is a roadmap item — not a number pulled from a template. Setting 99.9% on a system that has never exceeded 98% produces a permanently red dashboard that everyone learns to ignore.

## Step 5 — Alert on burn rate, not on raw failures

A single failed agent run is not an incident. A sustained burn of the error budget is. The standard pattern is multi-window burn-rate alerting, which fires when the budget is consumed faster than the SLO allows.

```yaml
# prometheus_rules.yml
groups:
  - name: agent_slo
    rules:
      - record: agent:task_success_rate:30d
        expr: |
          sum(rate(agent_task_success_total[30d]))
          /
          sum(rate(agent_task_success_total[30d]) + rate(agent_task_failure_total[30d]))

      - alert: AgentTaskSuccessBudgetBurnFast
        expr: |
          (1 - agent:task_success_rate:30d) > (14.4 * 0.01)
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Agent task success budget burning 14.4x faster than allowed"

      - alert: AgentTaskSuccessBudgetBurnSlow
        expr: |
          (1 - agent:task_success_rate:30d) > (6 * 0.01)
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Agent task success budget burning 6x faster than allowed"
```

The 14.4 and 6 multipliers come from the standard burn-rate math described in the Google SRE workbook. The arithmetic, for a 30-day window and a 1% error budget:

- A 14.4x burn sustained for 1 hour consumes `14.4 × (1 hour / 720 hours) = 2%` of the 30-day budget.
- A 6x burn sustained for 6 hours consumes `6 × (6 / 720) = 5%` of the budget.

Those two thresholds give a fast page and a slow ticket. The exact numbers depend on your window and tolerance; the point is to alert on the rate of budget consumption, not on individual failures.

## How to verify the pipeline works

Waiting for a real incident to learn whether the SLOs are wired correctly is not a plan. Build a synthetic trace generator that produces known-good and known-bad outcomes, then assert that the derived metrics move as expected.

```python
# test_sli_pipeline.py
from sli_definitions import evaluate_outcome


def test_success_path():
    trace = {
        "task_id": "t1",
        "final_answer": "done",
        "termination_reason": "agent_finished",
        "spans": [
            {"name": "llm.call", "attributes": {}},
            {"name": "tool.search", "attributes": {"args_valid": True}},
        ],
    }
    outcome = evaluate_outcome(trace)
    assert outcome.success is True
    assert outcome.terminated_cleanly is True
    assert outcome.tool_calls_valid is True


def test_invalid_tool_args():
    trace = {
        "task_id": "t2",
        "final_answer": "done",
        "termination_reason": "agent_finished",
        "spans": [
            {"name": "tool.search", "attributes": {"args_valid": False}},
        ],
    }
    outcome = evaluate_outcome(trace)
    assert outcome.tool_calls_valid is False
    assert outcome.success is True  # agent recovered


def test_forced_termination():
    trace = {
        "task_id": "t3",
        "final_answer": "",
        "termination_reason": "max_iterations",
        "spans": [],
    }
    outcome = evaluate_outcome(trace)
    assert outcome.terminated_cleanly is False
    assert outcome.success is False
```

Run these in CI. If someone changes the outcome logic and breaks the SLI contract, the tests fail before the change ships.

To verify the metrics layer rather than the outcome logic, exercise the pipeline end to end: emit a batch of synthetic traces through the real instrumentation, then query the counters directly and compare against the expected counts. In Prometheus, `sum(agent_task_success_total)` after a known batch should equal the number of synthetic successes. This catches the failure mode where the outcome function is correct but `record_outcome` is never called on the early-return path.

## Failure modes to watch for

**The SLI that never fires.** If the success definition is too lenient (for example, "the agent returned any string"), the SLI sits at 100% and the SLO is meaningless. Symptom: no error-budget consumption for weeks while users complain. Fix: tighten the definition until the baseline rate drops below 100%.

**The SLI that always fires.** If success requires an exact string match on expected output, non-deterministic agents will fail constantly and the alert will be muted within a day. Fix: use a rubric or evaluator, and sample.

**Cardinality explosion.** Adding `task.id` or raw arguments as a metric label will multiply time series until the backend falls over. Fix: fixed labels on counters; keep high-cardinality data on spans only.

**Evaluator drift.** If success is judged by an LLM evaluator with a fixed prompt, and the model behind that evaluator is upgraded, the measured success rate can shift without any change to the agent. Fix: pin the evaluator model version and re-baseline deliberately, treating evaluator changes as SLO changes.

**Retries polluting success.** If a task succeeds on the third attempt, it is still a success. Count retries as a separate SLI (retry rate) for efficiency visibility, but don't let them pollute the success metric.

## Common questions

**How do I handle agents that legitimately take multiple attempts?**

Define success at the task level, not the attempt level. A task that succeeds on the third retry is a success. Track retry rate separately if you want efficiency visibility.

**What about non-deterministic outputs where "correct" is subjective?**

Use a rubric-based evaluator and record its verdict as a span attribute such as `eval.passed`. The SLI becomes the rate of `eval.passed`. This adds cost, so sample: evaluate a fixed percentage of runs and extrapolate, or evaluate everything for a calibration week and then sample.

**Should I use spanmetrics or custom counters?**

Custom counters when the outcome depends on multiple spans, which is the usual case for agents. Spanmetrics is fine for per-span latency and error rate, but it cannot express "this tool call was valid AND the agent recovered from the error." That is a cross-span judgment and belongs in application code.

**How does this work with managed observability platforms?**

OpenTelemetry is the portability layer. If you emit OTLP, you can point at any backend that speaks it. The SLI math is the same; only the query language changes. The risk is vendor-specific features creeping into alerting rules — keep the recording rules in PromQL where possible so a backend migration doesn't require rewriting alerts.

## Where to go from here

Pick one agent feature and write its `evaluate_outcome` function today. Don't try to cover every agent in the system at once. Choose the one with the most user-visible failures, define success in a single function, and instrument it with the three counters above. Then open `prometheus_rules.yml` and add the burn-rate alert. That's a 30-minute change that will tell you more about your agent's real reliability than any latency dashboard.

---

**Do this in the next 30 minutes:** open your agent's code, write a single `evaluate_outcome(trace) -> AgentOutcome` function that returns `success`, `tool_calls_valid`, and `terminated_cleanly` as booleans, and commit it with the three unit tests from the verification section above. No instrumentation needed yet — the function is the contract, and everything else follows from it.
