# Postmortems for AI agents: human review vs blind replay

## The failure mode this article addresses

An agent runs for hours before anyone notices it is quoting stale prices from a local PDF that a retrieval pipeline ranked above live pricing data. The output looked plausible. No stack trace fired. The guardrail passed. By the time a human notices, the failure has been live for a long time and the evidence is scattered across prompt versions, retrieved chunks, tool responses, and model provider state.

Traditional incident response assumes a reproducible chain of calls. Agent failures often live in the prompt, the retrieval context, or internal state that never reached the logs. That mismatch is why teams keep re-litigating the same question: should a human review the failure, or should a machine replay it?

Two approaches dominate:

- **Human-loop postmortems.** A reviewer re-runs the agent with the same inputs, inspects intermediate steps, and validates the output before it reaches the user. The human is in the loop at review time.
- **Blind-replay postmortems.** The system captures the entire trace (prompts, tool calls, memory state, outputs) and replays it deterministically in a staging environment. No human is involved at review time; the replay produces a pass/fail regression result.

This article compares them on detection speed, resolution speed, false positives, developer experience, and cost. Every number below is either a documented default, arithmetic shown from stated assumptions, or explicitly labelled illustrative. Where a benchmark would normally appear, you get the instrumentation recipe instead, because your numbers will differ from anyone else's.

## What a trace actually contains

Before comparing approaches, it helps to be precise about the artifact both depend on. A useful agent trace records, at minimum:

- The exact system prompt and any prompt template version identifier.
- The full message history sent to the model, including tool schemas.
- Every tool call: name, arguments, raw response, and wall-clock duration.
- The retrieval step: query, index snapshot identifier, top-k chunks with scores.
- Model parameters: temperature, top_p, seed, max tokens, and the provider's model version string.
- The final output, plus any guardrail verdicts and their thresholds.

If any of these are missing, neither approach can do its job. A reviewer cannot judge a retrieval failure without the chunks; a replay engine cannot reproduce a call without the exact arguments and the index snapshot. The single highest-leverage investment in agent postmortems is making the trace complete, not choosing between the two methods.

## Option A: human-loop postmortems

### How it works

The agent records a trace and flags outputs that fail a guardrail or trigger an alert. A reviewer replays the trace, annotates the incident, and decides whether to patch the prompt, the retrieval index, or a tool configuration. The human is in the loop at review time, not at runtime.

A typical stack captures traces to durable storage, surfaces them in a review UI, and notifies an on-call engineer through whatever channel the team already uses. The reviewer's job is to validate the output, inspect retrieved chunks, and classify the failure as prompt-related, retrieval-related, or tool-related.

### Where it shines

Human review is strongest when correctness is semantic and hard to assert programmatically: support replies, summaries, marketing copy, anything where a domain expert can judge quality faster than a rule can. It is also the right default while prompts are still changing weekly, because the reviewer generates labelled examples you can later use for evaluation sets or fine-tuning.

The other advantage is override capability. A toxicity or PII filter that fires on a legitimate medical term can be overridden by a reviewer who understands the context. Logging every override is the standard way to tighten guardrails over time: overrides are your ground-truth labels for "the filter was wrong."

### The failure mode to watch

Reviewer cognitive load is the dominant cost, and it is easy to underestimate. A raw trace with thirty retrieved chunks and a dozen tool calls takes real effort to read. When traces are long, on-call engineers skip reviews, and a skipped review is indistinguishable from a passing one in most dashboards.

The mitigation is a summary view containing only the prompt, the top few retrieved chunks, the tool calls, and the final output, with a link to the raw trace for the minority of cases that need it. This is a UI change, not an architecture change, and it is usually the single cheapest improvement available.

### Illustrative sketch

The following is a simplified illustration of the shape of a trace collector, not a working integration with any specific library. Treat it as pseudocode for the data model.

```python
# Illustrative only: shape of a human-review trace collector.
from dataclasses import dataclass, field

@dataclass
class ReviewTrace:
    trace_id: str
    prompt_version: str
    messages: list
    tool_calls: list
    retrieved_chunks: list
    final_output: str
    guardrail_verdicts: list
    reviewer: str | None = None
    status: str = "pending"
    overrides: list = field(default_factory=list)

def on_guardrail_failure(trace: ReviewTrace, store, notifier):
    trace.status = "pending"
    store.save(trace)
    notifier.notify(trace.trace_id)
    return trace.trace_id
```

The important design choices are visible here: the prompt version is stored alongside the trace, guardrail verdicts are recorded rather than just acted on, and reviewer overrides are captured as data.

## Option B: blind-replay postmortems

### How it works

Blind replay captures the full trace automatically and replays it deterministically in a staging environment. The replay engine must be deterministic in every input it touches: the same model version, the same vector index snapshot, the same tool stubs, the same sampling parameters. In practice that means pinning dependencies to exact versions and recording or caching model responses so a replay does not depend on a live provider.

Determinism is the whole game. If any input varies between the original run and the replay, the comparison is meaningless. The usual sources of non-determinism are:

- Sampling: temperature above zero, or a provider that does not honour a seed. Setting temperature to zero and a fixed seed removes most of it, but not all providers guarantee this.
- Retrieval: a live index that has been rebuilt or re-embedded since the original run.
- Tools: live web searches, rate-limited APIs, and anything with wall-clock or randomness in its response.
- Model version: providers deprecate and silently update models. Pin the version string and record it in the trace.

### Where it shines

Blind replay is strongest when the output is not directly user-facing, or when user impact is low: internal enrichment, classification, routing, batch summarisation. It is also the only practical way to catch retrieval drift, because drift is invisible to guardrails. If an index is rebuilt with a new embedding model but an old snapshot is still wired into the pipeline, outputs degrade gradually and continue to sound plausible. A similarity assertion against a known query set catches this; a human reading one output usually does not.

The second advantage is deployment gating. Once a replay suite exists, it runs on every prompt or index change, which turns postmortems into regression tests. That is a genuine workflow shift: engineers fix the prompt or the index rather than triaging a ticket.

### The failure mode to watch

Prompt drift breaks blind replay. When the system prompt changes, every old trace is replayed against the new prompt, and assertions written for the old behaviour fire on differences that are harmless or even intended. The result is a flood of spurious failures, which is worse than no signal because it trains the team to ignore the suite.

The mitigation is a prompt-diff step: compare the prompt version recorded in the trace against the current prompt, and only re-run assertions that are still relevant to the changed sections. Assertions should be scoped to behaviour, not to exact string matches on the output.

### Illustrative sketch

The following illustrates the shape of a replay configuration. It is not tied to a specific product.

```yaml
# Illustrative only: shape of a replay configuration.
replay:
  llm_cache: true
  model: "<pinned-model-version>"
  index_snapshot: "2026-06-15T14:30Z"
  temperature: 0
  seed: 12345
  assertions:
    - type: retrieval_similarity
      min_score: 0.85
    - type: p95_latency_ms
      max: 5000
    - type: output_schema
      strict: true
```

## How to measure both approaches on your own traffic

Published benchmarks for agent postmortems are close to worthless because the numbers depend entirely on your trace completeness, your guardrail design, and your incident mix. Measure it yourself. Here is what to instrument.

**Time-to-detect.** Timestamp the moment the failure occurs (the trace's final output timestamp) and the moment an alert or ticket is created. For human-loop, detection depends on a guardrail firing or a user reporting. For blind replay, detection happens when the replay suite next runs. Record the distribution, not the mean; the tail is what hurts.

**Time-to-resolve.** From the detection timestamp to the timestamp the fix is deployed and verified. Break this into triage time and fix time, because the two approaches differ mainly in triage.

**False-positive rate.** Count flagged incidents that a reviewer or a replay assertion marked as not-a-real-failure, divided by total flagged incidents. You need a consistent definition of "real failure" or this metric drifts.

**Reviewer minutes per incident.** Instrument the review UI to record time-on-page. Self-reported estimates are consistently too low.

**Replay determinism rate.** Run the same trace twice through the replay engine and diff the outputs. If they differ, the replay is not deterministic and every result from it is suspect. Track this as a first-class metric.

**Cost.** Reviewer time is `reviewer_minutes / 60 × loaded_hourly_rate`, including benefits and overhead, not just salary. Replay infrastructure is the marginal cost of the nodes or SaaS tier you actually run, divided by incidents processed.

A controlled experiment is straightforward: inject synthetic failures of known type into a staging agent, then run both approaches against the same set and compare. Fifty failures across prompt, retrieval, and tool categories is enough to see whether one approach is clearly ahead for your workload. The point is not to reproduce anyone else's numbers; it is to replace assumptions with your own measurements before committing.

## Head-to-head comparison

The table below compares the two approaches qualitatively. No numeric results are given because they are workload-specific; the right-hand column tells you what to measure to fill in the blanks for your team.

| Dimension | Human-loop | Blind replay | What to measure |
|---|---|---|---|
| Detection | Depends on guardrail firing or user report | Automatic when the replay suite runs | Time from failure to first alert |
| Triage speed | Bounded by reviewer availability | Bounded by replay runtime | Triage minutes per incident |
| Semantic quality judgement | Strong | Weak; assertions only | Rate of "plausible but wrong" outputs caught |
| Retrieval drift detection | Weak; drift is invisible to guardrails | Strong; similarity assertions catch it | Similarity score delta on a fixed query set |
| Determinism required | No | Yes, absolutely | Replay determinism rate (same trace twice) |
| Setup cost | Low; uses existing incident tooling | High; pinning, caching, snapshots | Engineer-weeks to first reliable replay |
| Marginal cost per incident | Reviewer time | Compute plus tooling | Cost per incident at your volume |
| Scales with volume | Poorly; reviewer hours are linear | Well; compute is elastic | Cost curve as incidents per month grows |
| Handles live external tools | Yes | Poorly | Fraction of incidents involving live tools |
| Best fit | User-facing, semantic, low volume | Internal, high volume, deterministic | Your incident mix |

The pattern that emerges is not "one wins." Human review wins on semantic judgement and setup cost. Blind replay wins on detection latency, scale, and catching drift. Most mature teams end up with both: replay as the default gate, human review reserved for the category of failures where correctness is a judgement call.

## A worked decision example

Consider a support agent handling 5,000 tickets per day across three markets. Assume, illustratively, that the team sees 300 flagged incidents per month, that a reviewer spends 6 minutes per incident, and that the loaded reviewer rate is $45 per hour.

- Reviewer cost: `300 × (6 / 60) × $45 = 300 × 0.1 × $45 = $1,350` per month.
- If a summary view halves review time to 3 minutes: `300 × 0.05 × $45 = $675` per month. The saving is $675 per month for a UI change.
- If the false-positive rate is 20%, then 60 of those 300 reviews are wasted: `60 × 0.05 × $45 = $135` per month spent on non-failures. Improving guardrails attacks that line directly.

Now the replay side. Assume two small nodes at $0.05 per node-hour, running continuously: `2 × 0.05 × 24 × 30 = $72` per month, plus whatever SaaS tier the team uses for trace storage. If the replay suite catches 40% of the same incidents automatically and those no longer need a reviewer, reviewer cost drops to `180 × 0.05 × $45 = $405` per month, and total cost is `$405 + $72 = $477`, versus $675 for review-only. The break-even is sensitive to the reviewer rate and the catch rate, which is exactly why you should compute it with your own numbers rather than adopting someone else's threshold.

The decision rule that falls out: if `replay_monthly_cost < reviewer_cost_saved + value_of_faster_detection`, run replay as the default. Otherwise keep review as the default and use replay only for the categories it handles well.

## Decision checklist

Work through these in order. The first "no" usually settles it.

1. **Is the output user-facing and safety-critical?** Medical, financial, legal, or anything where a wrong answer causes harm. If yes, a human must review semantic correctness. Replay alone is not sufficient.
2. **Can you replay deterministically today?** Do you pin model versions, cache responses, and snapshot your index? If not, estimate the work before choosing replay; it is usually measured in engineer-weeks, not days.
3. **What is your incident volume?** Below roughly 100 to 150 incidents per month, reviewer time is usually cheaper than replay infrastructure. Above that, compute the break-even with your own rates.
4. **Do you have domain experts available?** If not, human review degrades into rubber-stamping, and replay is the safer default.
5. **How often does the prompt change?** Frequent prompt changes make replay noisy unless you invest in prompt-diffing. If prompts change weekly and you have no diffing, start with review.
6. **Do incidents involve live external tools?** Web search, rate-limited APIs, and other non-reproducible dependencies make replay unreliable. Route those to review.
7. **Do you already have incident tooling?** Existing alerting, ticketing, and on-call rotation make review cheaper to adopt. Replay needs its own harness.

## Implementation notes that apply to both

**Version everything.** Prompt version, index snapshot, model version, tool schemas, and guardrail thresholds should all be recorded in the trace. Without this, neither approach can reproduce anything, and you cannot tell whether a change caused an improvement.

**Separate detection from diagnosis.** Detection can be automated cheaply: schema validation, similarity thresholds, latency bounds, and output comparison. Diagnosis is where humans add value. Mixing the two is why reviewer queues get long.

**Scope assertions to behaviour.** An assertion that the output equals a stored string will fire on every harmless rephrasing. Assert on structure, on required facts being present, and on retrieval quality, not on exact text.

**Track override rates.** Every human override of a guardrail is a label. If a guardrail is overridden often, it is miscalibrated. If it is never overridden, check whether reviewers are actually reading it.

**Measure determinism first.** Before trusting any replay result, run the same trace twice and diff. A replay suite with a 70% determinism rate produces 30% noise, and noise destroys trust in the suite faster than missing failures does.

## FAQ

**How do I set up deterministic replay for an agent in production?**

Start by pinning every input: exact model version string, temperature zero, a fixed seed where the provider supports it, a snapshot identifier for the vector index, and stubs or recordings for every tool. Cache model responses so a replay does not depend on a live provider. Then verify determinism empirically: run the same trace twice and diff the outputs. Only after that rate is 100% should you write assertions, because assertions on a non-deterministic replay measure noise.

**What is the biggest mistake teams make with human-loop postmortems?**

Underestimating reviewer cognitive load. Teams assume a reviewer can skim a prompt and an output, but in practice long traces cause skipping, and a skipped review looks like a pass in most dashboards. The fix is a summary view with the prompt, a few retrieved chunks, the tool calls, and the final output, plus a link to the raw trace.

**When does human review genuinely outperform replay?**

When correctness is a judgement call rather than a checkable property, when the output is safety-critical, when incident volume is low enough that reviewer time is cheaper than infrastructure, or when incidents depend on live external tools that cannot be reproduced.

**How do I handle prompt drift in a replay suite?**

Record the prompt version in every trace, and before re-running assertions, diff the trace's prompt against the current one. Re-run only the assertions that remain relevant to the changed sections. Without this step, every prompt edit produces a wave of spurious failures, and the team learns to ignore the suite.

**Can we run both?**

Yes, and most teams that operate agents at any scale do. Replay runs as a deployment gate on every prompt or index change, catching drift and regressions automatically. Human review handles the subset of incidents where the failure is real but the correct behaviour is a judgement call. The split is usually decided by incident category, not by incident volume.

**What if our agent calls a live search API?**

Replay cannot reproduce it faithfully. Record the tool response in the trace and stub it during replay, so you are testing the agent's reasoning over a fixed response rather than the live tool. Failures caused by the tool itself need a different diagnostic path, usually logging and rate analysis rather than replay.

## What to do in the next 30 minutes

Open your agent's incident log for the last 30 days and count three things: total flagged incidents, how many were resolved by an automated check versus a human reading the output, and how many traces are missing a prompt version or index snapshot identifier. If the third number is greater than zero, stop reading and fix trace completeness first. Neither postmortem approach works on incomplete traces, and that is the gap most teams find when they actually look.
