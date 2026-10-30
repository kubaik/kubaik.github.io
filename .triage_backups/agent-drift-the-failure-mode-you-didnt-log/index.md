# Agent drift: the failure mode you didn't log

An agent passes its evaluation suite on Tuesday, ships a slightly different distribution of answers on Wednesday, and by Friday a subset of users receives responses that are technically valid but contextually wrong. Nothing in the stack flagged it. Latency is flat, token counts are flat, error rates are flat. This is agent drift, and the reason it goes unlogged is that most observability is built to catch a different failure.

## The gap between what the docs say and what production needs

Agent drift is commonly defined as an agent's behavior gradually diverging from its intended specification over time. That definition is accurate and nearly useless for building anything. The documentation for many agent frameworks treats drift as something addressed with better prompts or a stronger model. Production behavior suggests otherwise.

The part that trips people up is that drift is usually not a model failure. It is a distribution failure. The model is doing what it was asked; the input distribution, the context assembly, or the surrounding system changed. A recurring failure mode: a support agent begins summarizing tickets more aggressively once a prompt cache warms up, because the cached prefix contains an example that biases toward brevity. The model weights did not change. The context did. Because evaluation suites test final answers rather than intermediate state, the shift stays invisible until a user reports that the agent "sounds different."

Most agent observability is built for the wrong failure. Teams instrument latency, token counts, and error rates, all of which stay flat during drift. The signal that matters is behavioral: are the agent's decisions still clustering around the same points as last week? That is a different measurement problem, and it is the one this article covers.

## Detection and containment are two different problems

Drift handling is really two separate problems bolted together: detection (did the behavior change?) and containment (what do we do about it?). Conflating them produces a dashboard nobody acts on.

Detection works by sampling the agent's decision points and comparing them to a reference distribution. For a tool-calling agent, the decision points are which tool was selected, what arguments were passed, and in what order. For a retrieval-augmented agent, they are the retrieval set and the claim structure of the answer. Comparing full text is expensive and noisy. Comparing embeddings of the decision trace is better but still costly. Cheaper still, and often sufficient, is comparing categorical features: tool name, argument schema shape, retrieval count, answer length bucket.

Containment is the harder half. Once drift is detected, there are three broad options: roll back the prompt or model version, clamp the agent to a narrower action space, or route the drifted traffic to a human. Treating containment as a manual runbook is the common mistake. Containment needs to be a policy the agent runtime enforces, such as a circuit breaker that trips when the drift score crosses a threshold.

A concrete example: an agent that books internal meeting rooms. Its tool calls are `check_availability`, `reserve_room`, and `send_invite`. Normal drift is a shift in the argument distribution, for instance more 30-minute slots than 60-minute slots. Pathological drift is the agent calling `reserve_room` before `check_availability`, a sequencing violation that produces double-bookings. Categorical drift detection catches the first; a state-machine guard catches the second. Both are needed.

The useful framing: drift detection is a statistical problem, but containment is a state machine problem. Teams that try to solve both with the same tool end up with either noisy alerts or brittle guards.

## A minimal implementation

The following is a minimal implementation of the architecture described above. It uses Python 3.11, sentence-transformers 3.0 for the optional embedding check, and Redis 7.2 for storing reference distributions. The agent runtime is a simple loop; the drift detector runs as a sidecar that samples a fraction of traces.

First, the trace collector. Every agent decision is logged as a structured event with a categorical signature:

```python
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class DecisionTrace:
    trace_id: str
    step: int
    tool_name: str
    arg_keys: tuple[str, ...]
    arg_shape: str  # e.g. "str:int:int"
    retrieval_count: int
    answer_len_bucket: str  # "short" | "medium" | "long"

def signature(trace: DecisionTrace) -> str:
    # Categorical signature — cheap to compare, stable across model versions
    parts = [
        trace.tool_name,
        ",".join(sorted(trace.arg_keys)),
        trace.arg_shape,
        str(trace.retrieval_count),
        trace.answer_len_bucket,
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
```

The signature is deliberately coarse. Comparing raw embeddings of full traces yields high-dimensional noise; comparing categorical signatures yields a distribution that can actually be tested. A signature space of a few hundred distinct values is typically the sweet spot: small enough to compute a stable histogram, large enough to catch real shifts.

Next, the reference distribution and the drift score. Population stability index (PSI) is a reasonable choice because it is interpretable. The conventional reading, inherited from credit risk monitoring, is that PSI below 0.1 is stable, 0.1 to 0.25 is a warning, and above 0.25 is a drift event. The metric maps cleanly onto agent traces.

```python
import math
from collections import Counter

def psi(reference: Counter, current: Counter, epsilon: float = 1e-6) -> float:
    total_ref = sum(reference.values())
    total_cur = sum(current.values())
    keys = set(reference) | set(current)
    score = 0.0
    for k in keys:
        ref_pct = (reference.get(k, 0) + epsilon) / total_ref
        cur_pct = (current.get(k, 0) + epsilon) / total_cur
        score += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return score

# Common starting thresholds:
# PSI < 0.10  -> stable, no action
# 0.10 - 0.25 -> log warning, increase sampling
# > 0.25      -> trip circuit breaker, route to fallback agent
```

Containment is a state machine layered on top, and it lives in the agent runtime rather than in the detector. When PSI crosses the chosen threshold for two consecutive windows, the runtime can swap the agent's tool set to a read-only subset and route write operations to a queue for human review. That is the difference between detection and containment: the detector emits a signal, the runtime enforces a policy.

One more piece: the sequencing guard. This catches pathological drift that categorical PSI misses.

```python
ALLOWED_TRANSITIONS = {
    "check_availability": {"reserve_room", "send_invite"},
    "reserve_room": {"send_invite"},
    "send_invite": set(),
}

def enforce_sequence(history: list[str], next_tool: str) -> bool:
    if not history:
        return next_tool == "check_availability"
    last = history[-1]
    return next_tool in ALLOWED_TRANSITIONS.get(last, set())
```

If `enforce_sequence` returns False, the runtime rejects the tool call and re-prompts the agent with the valid next actions. This is cheap, on the order of microseconds, and catches the double-booking class of failure that PSI will never see, because the distribution of tool calls can look entirely normal while the ordering is wrong.

## What this costs in practice

Rather than quoting benchmark numbers, it is more useful to describe what to instrument and how to compare. The figures below are illustrative of what the architecture above tends to cost, expressed as a worked example with stated assumptions rather than measurements from a named system.

Assume an agent handling 40,000 conversations per day, sampled at 5 percent, over 15-minute windows. That is roughly 2,000 conversations per day entering the detector, or about 83 per window. If each conversation produces on average 6 decisions, a window contains roughly 500 traces. A PSI computation over a histogram of a few hundred keys is a few milliseconds of pure CPU work; the dominant cost is reading the window from storage.

Trace signature computation is a hash over a handful of short strings, which runs in well under a millisecond and requires no model call. Storage is the signature string plus a counter per window; with a 30-day retention policy and a few hundred distinct signatures, this stays in the tens of megabytes. The circuit breaker, being in-process, adds no network hop.

The optional embedding-based deep check is the expensive component. Running a sentence-transformer model such as `all-MiniLM-L6-v2` on a single trace costs on the order of tens to low hundreds of milliseconds depending on hardware, which is why it is usually reserved for a 1 percent sample or run offline.

To measure the tradeoff on your own traffic, instrument three things: the wall-clock time of `signature()`, the wall-clock time of `psi()` per window, and the count of distinct signatures per window. Then compare the drift events flagged by the categorical detector against those flagged by the embedding check over the same period. The comparison that matters is not raw accuracy but the marginal value of the embedding check: how many drift events does it catch that the categorical detector misses, and what does each of those cost.

A common finding when teams run this comparison is that the categorical PSI detector catches the large majority of drift events that the embedding check also catches, at a small fraction of the cost. The embedding check is still worth keeping for the residual cases, usually semantic drift where the tool distribution is stable but the content of arguments shifted. If only one can be afforded, build the categorical detector first. It is boring and it works.

Latency impact on the agent itself is typically negligible: the signature is computed inline, the PSI runs asynchronously, and the sequencing guard adds under a millisecond. Total overhead is small compared to a typical LLM call.

## Failure modes worth designing for

The first failure mode is reference distribution rot. The reference histogram is built from some period of past traffic. If the user base shifts, whether from a new customer segment or a seasonal spike, the reference is no longer valid and PSI will fire constantly. The usual response is to raise the threshold, which defeats the purpose. The fix is to rebuild the reference on a rolling window, but only from traces that passed human review or explicit quality checks. Rebuilding from all traffic bakes drift into the baseline.

The second is the silent tool rename. Renaming `reserve_room` to `book_room` in a prompt update changes every signature overnight, and PSI goes to its maximum. This is not drift; it is a schema change. The fix is a signature versioning scheme: hash the tool schema rather than the tool name, and treat schema changes as explicit reference resets. Treating a signature as stable when the underlying schema is not is a common trap.

The third, and the one that most directly causes bad user experiences, is drift in the fallback path. When the circuit breaker trips, traffic routes to a fallback agent, usually a simpler and more constrained one. If the fallback has its own drift, and it will because it receives less attention, the problem has simply moved. The fallback needs its own reference distribution and its own PSI check. Teams that skip this discover that the safety net produces the same bad answers, just slower.

A concrete example of the third: an agent that handles refund requests starts drifting toward over-approval. The circuit breaker trips and traffic routes to a rule-based fallback. The fallback approves refunds under a fixed dollar threshold automatically, a rule that was correct when written but now covers most requests because of a pricing change. Users get inconsistent treatment: some approved instantly, some routed to manual review, with no visible logic. The drift was contained; the user experience was not.

## Choosing components

There is a real temptation to build everything from scratch. For storage and metrics layers, existing tools are usually adequate. The detection logic itself is domain-specific and is often worth writing directly.

| Component | Role | Where it fits | Where it does not |
|---|---|---|---|
| Redis | Signature storage, sliding windows, TTL | Fast key-value histograms with expiry | Not a time-series database; avoid long-range analytical queries |
| Prometheus | PSI as a gauge, alerting rules | Threshold alerts on low-cardinality metrics | Not for high-cardinality trace IDs |
| sentence-transformers | Semantic drift checks on a sample | Offline or sampled semantic comparison | Too slow for inline use |
| LangGraph | Explicit state machine for containment | Encoding allowed transitions | Its tracing is not drift detection |
| OpenTelemetry | Trace propagation across agent steps | Correlating decisions to a request | Does not understand agent semantics |
| A batch drift-reporting library | PSI and drift reports for batches | Building reference distributions, validating thresholds | Not designed for streaming agent traces |

A practical split: use an offline drift library to build reference distributions and validate thresholds, and keep the hot path to a short PSI function plus a key-value store. The hot path does not need a framework.

## When this approach is the wrong choice

This architecture assumes the agent has a stable action space. If the agent dynamically generates tools, or if the tool set changes weekly, categorical drift detection will produce constant false positives and will likely be turned off within a month. In that case, semantic drift detection at the intent level is needed instead, which is a harder problem and usually requires human labeling.

It is also the wrong choice for low-volume agents. If an agent handles fewer than roughly 1,000 decisions per day, the histograms are too sparse for PSI to be meaningful. A 15-minute window containing a dozen traces produces a PSI score dominated by noise. For low-volume agents, use a simple per-trace check, such as whether the trace violated the sequencing guard, and skip distributional detection.

Finally, this approach should not be applied where drift is expected and desired. A recommendation agent that adapts to user behavior is supposed to drift. Applying a stability threshold to it fights the adaptation it was built for. The distinction is whether drift is a bug or a feature, and that is a product decision rather than a technical one. Drift detection applies to a specific class of agents: those with a fixed action space and a correctness definition that does not change over time.

## An honest assessment

The notable finding is not that drift happens; that is well known. The notable finding is how cheap useful detection can be. It is easy to assume that embeddings, vector databases, and a streaming pipeline are required. What often works instead is a hash of a categorical signature, a short PSI function, and a key-value store with a rolling TTL. The expensive parts are reference distribution hygiene and the containment policy, neither of which is a machine learning problem.

A second point worth pushing back on: drift is often framed as an AI safety issue when it is mostly an operational one. The bad user experiences that result from drift are usually not the agent going rogue; they are the agent doing something slightly different for a week while everyone assumed the evaluation suite covered it. Evaluation suites test the answers that someone thought to test. Drift detection tests the answers that nobody did.

For teams building agents, the sequencing guard is often the highest-return piece. It is a small amount of code, it catches the failures that actually hurt users, such as double-bookings, over-approvals, and out-of-order writes, and it requires no statistical machinery. It is worth building before the dashboard.

## Frequently asked questions

**How is agent drift detected without embeddings?**

Use categorical signatures of the agent's decision points: tool name, argument keys, argument shape, retrieval count, answer length bucket. Hash the combination and track the histogram over time. Compare windows using population stability index. This catches many drift events at a fraction of the cost of embedding-based detection, and it runs inline in under a millisecond per decision.

**What PSI threshold should be used for agent drift alerts?**

A reasonable starting point is 0.10 as a warning and 0.25 as a drift event, the conventional thresholds from credit risk monitoring. Requiring two consecutive windows above 0.25 before tripping a circuit breaker reduces the false positive rate substantially. Tune from there based on traffic volume and tolerance for alerts.

**Why does the drift detector fire on every prompt update?**

Usually because the signature includes fields that change with prompt updates, typically tool names or argument schemas. Hash the tool schema rather than the tool name, and treat any schema change as an explicit reference distribution reset. Without signature versioning, every prompt change looks like drift, and the alerts get ignored.

**When is drift detection not worth building?**

When the agent handles fewer than roughly 1,000 decisions per day, when its action space changes frequently, or when drift is the intended behavior, as with adaptive recommenders. In those cases, use per-trace guards such as sequencing checks instead of distributional detection. The full PSI pipeline is generally overkill below roughly 10,000 decisions per day.

## One action for the next 30 minutes

Open the agent's trace log and extract the last 500 decisions. For each one, write the tool name and the argument keys as a single string. Count the distinct strings. If the count is under a few hundred, the categorical drift detector described above is buildable today: that count is the signature space, and it is small enough that a key-value store and a PSI function will produce a usable signal within a week. Start with the sequencing guard, not the dashboard.
