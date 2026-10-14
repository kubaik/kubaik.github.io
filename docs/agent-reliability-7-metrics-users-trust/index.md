# Agent reliability: 7 metrics users trust

## The problem with "reliability" as a single number

For agents that book appointments, answer support tickets, or move money, there is a recurring dashboard problem. Uptime is green. Latency is fine. Error rate is near zero. And yet users keep abandoning the thing mid-task.

The usual metrics — request success rate, p95 latency, token throughput — describe the *service*, not the *agent*. An agent can return HTTP 200 with a perfectly formatted response that a user will not trust, because it contradicted itself, forgot a constraint, or silently did the wrong thing. Those failures don't show up in an APM.

This article covers seven reliability metrics that correlate with what users experience as trust: did the agent do what it said, did it stay inside its boundaries, and did it fail in a way the user could recover from. None of these require a data science team. All of them can be computed from logs most agent services already write.

## What this list actually solves

The core issue is a mismatch between *system* reliability and *agent* reliability. System reliability asks: did the request complete? Agent reliability asks: was the outcome correct, consistent, and recoverable from the user's point of view?

A concrete example. A support agent is asked to refund an order. It calls the refund tool, gets a success response, and tells the user the refund is processed. But the tool returned success for a *partial* refund because of a currency rounding issue, and the user expected the full amount. Request success rate: 100%. User trust: broken. The metric that would have caught this is not error rate — it's a task-level outcome check.

The list below is organized around four questions that map to user trust:

1. Did the agent complete the *task*, not just the *call*?
2. Did it stay within the boundaries the user assumed?
3. Was it consistent when asked the same thing twice?
4. When it failed, could the user tell and recover?

## Evaluation criteria

Each metric below is judged against the same four criteria. The criteria matter more than the specific metric choice.

- **Computable from existing logs.** A metric that requires a new tracing pipeline and a labeling team will not survive the next sprint. Prefer things derivable from request/response logs and tool-call records.
- **Correlates with a user-visible outcome.** The metric should move when a real user would have had a worse experience. A metric that only moves when infrastructure hiccups is a system metric, not an agent metric.
- **Has a documented failure mode.** You should be able to say "when this number goes bad, here is what is happening." Without that, it's a vanity metric.
- **Works with partial instrumentation.** Many teams run on limited budgets and shared infrastructure. A metric that demands a full distributed-tracing deployment before it returns anything useful is a poor fit.

Deliberately excluded: any metric that requires a ground-truth label for every conversation. Human labeling is valuable but slow, and it belongs in a separate evaluation loop, not a reliability dashboard.

## The 7 metrics

### 1. Task completion rate (not request success rate)

**What it does:** Measures the fraction of user-initiated tasks that reach a defined terminal state the user asked for — booking confirmed, ticket resolved, refund issued — rather than the fraction of API calls that returned 200.

**Strength:** It's the closest single number to "did the user get what they came for." It catches the class of failure where every tool call succeeds but the task silently doesn't. A team that tracks only request success will see 99.9% and miss that a meaningful share of refunds were partial.

**Weakness:** "Terminal state" must be defined per task type, and that definition drifts. A booking that gets rescheduled is neither success nor failure. In practice you end up with a small taxonomy — completed, abandoned, escalated, errored — and "abandoned" ends up doing a lot of work.

**Best for:** Any agent with a small number of well-defined task types. If "done" can be written down, this is the top-line metric.

### 2. Constraint violation rate

**What it does:** Counts the fraction of tasks where the agent took an action outside a boundary the user or the business assumed — spending above a limit, contacting an external party, deleting something, or acting on stale data.

**Strength:** This maps to *trust* rather than *satisfaction*. Users forgive a failed task; they do not forgive an agent that did something they didn't authorize. It's also cheap to compute if tool calls are logged with their arguments: a violation is a predicate over those arguments.

**Weakness:** Constraints must be enumerated, and the enumeration is never complete. The first version will miss the constraint that actually bites. Treat it as a living list, and add a constraint every time a new violation is found.

**Best for:** Agents that touch money, permissions, or external systems. If the agent can only read, this metric is less urgent.

### 3. Determinism under repetition

**What it does:** Sends the same input (or a close paraphrase) multiple times and measures how often the agent produces materially different outcomes. Not identical text — identical *decisions*.

**Strength:** Users notice inconsistency immediately. If asking the same question twice yields two different answers, trust drops even when both answers are individually reasonable. This metric makes that visible. It's also a good proxy for whether prompts and tools are stable.

**Weakness:** LLM-based agents are not deterministic by default, and forcing determinism can make them worse at handling edge cases. The useful version is *bounded* variation: the decision should be stable, the phrasing can vary. Setting that threshold is a judgment call.

**Best for:** Agents that answer questions or make recommendations, where users will naturally re-ask. Less critical for one-shot batch jobs.

### 4. Recovery rate after failure

**What it does:** Of the tasks that hit an error or an ambiguous state, what fraction did the user successfully complete — either by retrying, by a fallback path, or by the agent self-correcting?

**Strength:** This separates agents users keep using from agents users abandon. A 5% failure rate with a 90% recovery rate is a far better user experience than a 2% failure rate with a 20% recovery rate. It also rewards good error messages and clear handoff to a human, which are cheap to improve.

**Weakness:** Requires linking a failed attempt to a later successful one, which means session- or user-level correlation. If logs are per-request, a correlation ID has to be added first. Worth it.

**Best for:** Every agent. If only one metric from this list is tracked, track this one alongside task completion.

### 5. Grounding accuracy on tool results

**What it does:** Measures how often the agent's final answer is consistent with the tool results it actually received, rather than with what the model "remembers" or assumes. A refund agent that says "refunded $40" when the tool returned $39.60 is a grounding failure.

**Strength:** This catches hallucination in its most damaging form — not invented facts, but *contradicted* facts. Users trust an agent that reports what it did accurately. It's computable if both the tool output and the final message are logged: extract the key figure from each and compare.

**Weakness:** Extraction is fuzzy. "$39.60" and "thirty-nine sixty" are the same value. A small normalization step is needed, and it will have false positives. Budget for tuning.

**Best for:** Agents whose output includes numbers, dates, IDs, or statuses derived from tools. That's most transactional agents.

### 6. Escalation precision

**What it does:** Of the tasks the agent escalated to a human (or to a fallback), what fraction actually needed escalation? And conversely, of the tasks that should have been escalated, what fraction were?

**Strength:** Escalation is the safety valve for trust. Users tolerate an agent that says "I can't do this, let me get a human" far more than one that confidently does the wrong thing. Precision here tells you whether the valve is working or whether the agent is just giving up too easily.

**Weakness:** "Should have been escalated" requires a ground-truth judgment, which is expensive. A practical compromise is to track the *precision* side (escalations that were unnecessary) automatically, and sample the *recall* side manually.

**Best for:** Agents in regulated or high-stakes domains — finance, health, anything with a compliance surface.

### 7. Time-to-trust

**What it does:** Measures how many interactions a new user needs before they stop double-checking the agent — for example, before they stop re-reading the output or verifying it against the source system.

**Strength:** This is the only metric on the list that directly measures trust rather than a proxy for it. It's also the one that tells you whether reliability work is actually landing. A drop in time-to-trust after a fix is strong evidence the fix mattered.

**Weakness:** It needs behavioral instrumentation — click tracking, edit rate, re-query rate — and it's noisy. It's also slow: a cohort of new users has to be observed. Treat it as a quarterly metric, not a daily one.

**Best for:** Products with returning users and enough traffic to form cohorts. With ten users, this metric doesn't exist yet.

## The strongest default, and why

If only one thing can be shipped, ship **task completion rate plus recovery rate**, computed from a single correlation ID per task.

The reasoning: task completion answers "did it work," recovery answers "when it didn't, did the user get there anyway." Together they cover the two failure modes users actually feel. Constraint violations and grounding errors are important, but they're rarer and more domain-specific. Determinism and time-to-trust are valuable but expensive. Start with the pair that's cheap and universal.

A minimal implementation looks like this. Assume the agent logs one JSON line per request with a `task_id`, `user_id`, `ts`, `status`, and `tool_calls` field. Both metrics can be computed with a short script:

```python
import json
from collections import defaultdict

def load_events(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def task_outcomes(events):
    # A task is complete if any of its events reached a terminal success state.
    # A task is recovered if it had a failure followed by a success.
    by_task = defaultdict(list)
    for e in events:
        by_task[e["task_id"]].append(e)

    completed = 0
    recovered = 0
    failed_unrecovered = 0

    for task_id, evs in by_task.items():
        evs.sort(key=lambda e: e["ts"])
        saw_failure = False
        saw_success = False
        for e in evs:
            if e["status"] == "error":
                saw_failure = True
            if e["status"] == "completed":
                saw_success = True
        if saw_success:
            completed += 1
            if saw_failure:
                recovered += 1
        elif saw_failure:
            failed_unrecovered += 1

    total = len(by_task)
    return {
        "task_completion_rate": completed / total if total else 0.0,
        "recovery_rate": recovered / completed if completed else 0.0,
        "unrecovered_failures": failed_unrecovered,
    }
```

This is deliberately simple. It assumes logs already have a `task_id` and a terminal status. If they don't, adding those two fields is the highest-leverage change available. The script is illustrative — status names will differ — but the shape is the point: one pass over events, grouped by task, counting outcomes.

One caveat: this treats "completed" as a binary. In reality a third state is needed for "user abandoned," which usually shows up as a task with no terminal event within a timeout window. Add that once the basic version is running.

### How to verify the numbers mean anything

A metric that never moves is indistinguishable from a metric that is broken. Before trusting either number, confirm it can detect a known failure. Inject one: in a staging environment, force a tool to return an error status on a fraction of calls and confirm `unrecovered_failures` rises in proportion. Then force a tool to return a valid response with a wrong value and confirm that task completion *does not* move — that gap is exactly why metric 5 exists.

For grounding accuracy, the measurement procedure is: log the tool response payload and the final user-facing message for every task; extract the salient fields (amounts, dates, IDs, statuses) from each; normalize both sides (parse numbers, resolve relative dates against the request timestamp, canonicalize IDs); compare field by field; count tasks where any extracted field disagrees. Report the disagreement rate and sample twenty disagreements per week to estimate the false-positive rate of the extractor itself. If the false-positive rate is above roughly a fifth of flagged items, the extractor needs work before the number is actionable.

## Honorable mentions worth knowing about

A few metrics that didn't make the main list:

- **Tool-call correctness rate.** Fraction of tool calls with arguments that pass schema and business validation. Useful as a leading indicator — it often degrades before task completion does. But it's a system metric in disguise, and high correctness doesn't imply user trust.
- **Contradiction rate across turns.** How often the agent's later statements contradict its earlier ones within a session. This is a sharper version of determinism, but it requires turn-level parsing and is easy to over-count.
- **Human handoff latency.** Time from "agent decides to escalate" to "human picks up." Not strictly an agent metric, but users experience it as part of agent reliability. Worth tracking if escalation is common.

## Options that look appealing but fail in practice

Some metrics get proposed in every reliability discussion and then quietly get dropped. Here's why, so they can be skipped.

**Overall "hallucination rate."** It sounds like the metric you want, but it's not operational. "Hallucination" spans everything from a wrong date to a fabricated policy, and the ground truth requires a human to check every output. Teams that try to track it as a dashboard metric end up with a number nobody trusts. Better to decompose it: grounding accuracy (metric 5) covers the transactional cases, and a sampled human review covers the rest.

**Average user rating as a reliability metric.** Ratings are lagging, sparse, and biased toward extreme experiences. They're useful for product direction, but they won't tell you that the refund agent started rounding wrong on Tuesday. Keep ratings, but don't let them stand in for reliability measurement.

**Token-level confidence scores.** Model-reported confidence is not calibrated to correctness, and treating it as a reliability signal leads to false comfort. If a confidence signal is wanted, build it from outcome data — for example, the historical success rate of tasks that look like this one — not from the model's self-report.

**"Number of retries" as a health metric.** Retries are a symptom, not a cause. A high retry count could mean flaky tools or a confused agent, and the metric can't distinguish them. Track recovery rate instead; it captures the user-relevant consequence.

## How to choose based on your situation

A short decision framework:

| Situation | Start with | Add next |
|---|---|---|
| Early prototype, few users | Task completion rate | Recovery rate |
| Agent touches money or permissions | Constraint violation rate | Grounding accuracy |
| High-stakes / regulated domain | Escalation precision | Recovery rate |
| Q&A or recommendation agent | Determinism under repetition | Grounding accuracy |
| Established product, returning users | Time-to-trust | Constraint violations |

The pattern: start with the metric that matches the worst plausible failure, then add the one that matches the second-worst. Instrumenting all seven at once produces seven half-built metrics and no action.

## Frequently Asked Questions

**How do I measure agent reliability without a dedicated data team?**
Start with logs that already exist. If the agent writes one line per request, add a `task_id` and a terminal status, then compute task completion and recovery with a script like the one above. That's a day of work, not a quarter. Metrics that require human labeling — escalation recall, hallucination rate — should wait until the automated ones are running and trusted.

**Why doesn't an uptime dashboard catch agent failures?**
Because uptime measures whether the service answered, not whether the agent did the right thing. An agent can return a well-formed response that contradicts its own tool results or violates a user constraint, and every system metric stays green. Agent reliability needs outcome-level metrics — task completion, constraint violations, grounding accuracy — that live above the HTTP layer.

**What's the difference between task completion rate and request success rate?**
Request success rate counts individual API calls that returned without error. Task completion rate counts user-initiated tasks that reached the outcome the user asked for. A single task might involve five tool calls; if four succeed and one returns a wrong-but-valid result, request success is 100% and the task is a failure. Track both, but treat task completion as the one that reflects user experience.

**How often should these metrics be re-measured?**
Daily for task completion, recovery, and constraint violations — they're cheap and belong on a dashboard. Weekly for determinism and grounding accuracy, since they need sampling. Quarterly for time-to-trust, which needs cohorts. The cadence should match how fast the metric can move meaningfully; a daily time-to-trust number will just be noise.

**Can these metrics work with a small user base?**
Yes for the log-derived ones — task completion, recovery, constraint violations, and grounding all work at low volume because they're computed per task, not per cohort. Determinism needs enough traffic to sample repeats. Time-to-trust needs enough new users to form a cohort, so it's the one to defer when the user base is small. The absence of the expensive metrics shouldn't stop the cheap ones from shipping.

## Final recommendation

Agent reliability is not a service metric, and the gap between the two is where user trust leaks out. The fix is not a bigger observability stack — it's a task-level outcome metric and a recovery metric, both computable from logs that can be added today.

Your next 30 minutes: open the agent's request logging code and add a `task_id` field that stays constant across all calls made on behalf of a single user request, plus a terminal `status` field on the final event. That one change unlocks task completion rate, recovery rate, and most of the list above. Everything else can wait until that's in place.
