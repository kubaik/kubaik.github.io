# Agent audit trails: event log vs state snapshots

## The two questions an audit trail must answer

Compliance teams ask *what* an agent decided and *why*. Engineers ask *where* it broke. Those questions pull audit trail designs in opposite directions, and the tension usually surfaces only when an auditor requests something the logs cannot reconstruct.

Two dominant approaches exist. Event sourcing keeps an append-only log of every decision. State snapshots capture full agent state periodically. Neither is universally correct. The right choice depends on whether the primary consumer is a regulator asking "prove this decision was justified" or an on-call engineer asking "why did step 7 loop for nine minutes."

This article compares both against the same requirements: regulatory auditability, debugging fidelity, storage cost, and replay capability. It closes with a decision framework and one action to take in the next 30 minutes.

## Why this comparison matters now

Agent systems — LLM-driven or classical planners — make sequences of decisions that are hard to reconstruct after the fact. A single user request might trigger a retrieval call, a tool invocation, a policy check, and a final answer. Each step depends on the previous one's output.

Regulatory regimes in finance, healthcare, and general data protection expect a record of automated decisions that affect individuals. The EU AI Act's logging expectations for high-risk systems and GDPR Article 22's provisions on automated decision-making point in the same direction: the ability to explain a decision after the fact.

Debugging has the opposite requirement. When an agent produces a wrong answer, the engineer wants the *full state* at the moment of failure — the prompt, the retrieved context, the tool outputs, the internal reasoning trace. A log of events tells you what happened but not always what the system *believed* at that moment.

The naive solution is to log everything. That works until storage costs and query latency make the log useless in practice. The interesting question is which structure preserves the most explanatory power per byte.

## Option A: Event sourcing for agent decisions

Event sourcing treats every decision as an immutable fact appended to a log. The log is the source of truth; any current state is a projection derived by replaying events in order.

For an agent, the event stream looks roughly like this:

```json
{"ts":"2026-01-14T09:12:03.221Z","type":"request.received","agent_id":"a-81f","input_hash":"sha256:9c1f..."}
{"ts":"2026-01-14T09:12:03.884Z","type":"tool.invoked","tool":"policy_lookup","args_hash":"sha256:44ab...","latency_ms":663}
{"ts":"2026-01-14T09:12:04.102Z","type":"decision.made","choice":"approve","policy_ref":"PL-2291","confidence":0.91}
```

Each event is small. The full sequence for a typical agent run might be a few kilobytes. The log is append-only, which makes it cheap to write and easy to ship to cold storage.

The strength of event sourcing is *causal reconstruction*. Given a decision, you can walk backwards through the log and see every input that influenced it. That is exactly what a regulator wants: a defensible chain from input to output.

The weakness is *state reconstruction cost*. To answer "what did the agent know at step 7," you replay events 1 through 7. For short runs this is trivial. For long-running agents with hundreds of steps, replay becomes a query problem. Teams commonly solve this with periodic snapshots layered on top of the event log — a hybrid, discussed below.

Event sourcing also requires discipline. Every state change must emit an event. If a code path mutates state without logging, the log is no longer authoritative and replay produces a different result than what actually happened.

A second, subtler weakness applies specifically to LLM agents: event sourcing assumes replay reproduces state. For classical systems with deterministic operations, that holds. For LLM-driven agents with temperature above zero, or with tools that hit external APIs, replay does not reproduce the original state — it produces a *new* state that may differ. The log tells you what happened; it does not let you re-experience it. This point is developed further in the debugging section.

## Option B: State snapshots for agent decisions

State snapshots capture the full agent state at a point in time — typically at the end of each step, or on a fixed interval.

```json
{
  "snapshot_id": "s-2026-01-14-091204-102",
  "agent_id": "a-81f",
  "step": 7,
  "context_window": ["...full retrieved documents..."],
  "scratchpad": "...internal reasoning...",
  "tool_results": ["...full tool outputs..."],
  "pending_actions": [],
  "decision": {"choice": "approve", "policy_ref": "PL-2291"}
}
```

Snapshots are large — often tens to hundreds of kilobytes each, dominated by retrieved context and reasoning traces. They are also self-contained: to answer "what did the agent know at step 7," you read one record.

The strength is *debugging fidelity*. An on-call engineer can open a single snapshot and see the full picture without replaying anything. For LLM agents especially, where the prompt and retrieved context are the explanation, this is enormously valuable.

The weakness is *cost and drift*. Snapshots duplicate information across steps. A 200-step run with 50 KB snapshots is 10 MB per run. At moderate volume that becomes a storage line item. Snapshots also lose the *between-step* story: they show state at boundaries, not the events that caused transitions.

## Head-to-head: regulatory auditability

Regulatory auditability favors event sourcing. The reason is not technological but structural: regulators ask questions of the form "show me the sequence of decisions that led to this outcome." An append-only log answers that directly. A series of snapshots requires diffing adjacent records to reconstruct the sequence, which is possible but fragile — if a snapshot is missing, the diff is wrong.

Event sourcing also gives you *tamper evidence* more naturally. A hash-chained log (each event includes the hash of the previous) makes retroactive edits detectable. Snapshots stored as individual blobs do not have this property unless you add it explicitly.

That said, snapshots can be made auditable. If each snapshot includes a reference to the previous snapshot's hash and a monotonic step counter, you get a chain. The cost is that you are now maintaining two structures — the snapshot and the chain metadata — which is more work than the event log alone.

A concrete failure mode: a team stores snapshots in a mutable object store with no versioning. An auditor asks whether snapshot 7 was modified after the fact. Without versioning or hashing, the honest answer is "we cannot prove it wasn't." Event sourcing with hash chaining avoids this question entirely.

## Head-to-head: debugging fidelity

Debugging favors snapshots. When an agent produces a wrong answer, the engineer wants the full context — the exact prompt, the exact retrieved documents, the exact tool outputs. A snapshot delivers all of this in one read. An event log delivers it only after replay, and replay of an LLM call is not deterministic.

This is the point teams miss. Event sourcing assumes replay reproduces state. For classical systems with deterministic operations, that holds. For LLM-driven agents with temperature above zero, or with tools that hit external APIs, replay does not reproduce the original state — it produces a new state that may differ.

Snapshots sidestep this. They record the actual state at the time, including the actual LLM output, regardless of determinism.

A practical pattern: log events for the causal chain, but also snapshot the *inputs and outputs* of each non-deterministic call. This is the hybrid, and for LLM agents it is usually the right answer.

| Requirement | Event log | State snapshots |
|---|---|---|
| Causal chain reconstruction | Direct | Requires diffing |
| Point-in-time state inspection | Requires replay | Direct |
| Tamper evidence | Natural (hash chain) | Needs explicit chaining |
| Storage per run (illustrative) | ~KB | ~MB |
| Handles non-determinism | No (replay diverges) | Yes (records actual) |
| Query latency for "why" | Proportional to run length | Constant |

## Head-to-head: cost model

The cost difference is dominated by what you store. Consider an illustrative agent run with 50 steps, each step producing 2 KB of events and 40 KB of snapshot data. That is 100 KB of events and 2 MB of snapshots per run — a 20x ratio. The numbers are illustrative, not measured.

At low volume neither matters. At high volume the ratio determines whether audit storage is a rounding error or a line item. Object storage is cheap per gigabyte, but query costs and egress dominate once you actually read the data during incidents or audits.

The event log's cost advantage shrinks when you add snapshots for debugging. The snapshot's cost advantage shrinks when you add chaining metadata and indexes for audit queries. In practice, teams that need both properties end up storing both, and the cost question becomes "how much snapshot data do we actually need?"

A useful discipline: snapshot only the fields that are (a) non-deterministic, (b) expensive to recompute, or (c) required by a specific audit question. Storing the full context window on every step is rarely necessary; storing the context window at decision points is usually sufficient.

To measure the real ratio in your own system rather than relying on the illustrative figures above, instrument three things: bytes written per event, bytes written per snapshot, and the number of each per run. Compare against your actual query patterns — how often incident responders read snapshots versus how often auditors read the event chain — and price the reads, not just the writes.

## A decision framework

Rather than pick one, answer three questions about your system:

1. **Is your agent's behavior deterministic?** If yes, an event log alone is sufficient — replay reproduces state. If no (LLM with temperature, external tool calls, time-dependent logic), you need snapshots or at least recorded outputs for the non-deterministic steps.

2. **Who is the primary consumer of the audit trail?** If regulators or auditors, prioritize the event log with hash chaining. If on-call engineers, prioritize snapshots at decision points. If both, you need both, and you should say so explicitly in your architecture doc rather than pretending one structure serves both.

3. **What is your retention requirement?** Regulatory retention is often measured in years. Debugging retention is often measured in days to weeks. Storing full snapshots for years is expensive; storing events for years is cheap. A common pattern is short-retention snapshots plus long-retention events, with the snapshots referenced from the events.

The framework produces four outcomes:

- Deterministic + engineers-only → event log, no snapshots
- Deterministic + regulators → event log with hash chaining
- Non-deterministic + engineers-only → snapshots at decision points, minimal event log
- Non-deterministic + regulators → hybrid: hash-chained event log as the spine, snapshots at decision points referenced from events

Most production agent systems land in the fourth box.

## Recommendation and caveats

For most teams building agents that touch regulated decisions, the hybrid is the right default. Use a hash-chained event log as the authoritative record. Emit an event for every state transition. At decision points — moments where the agent commits to an action with external consequences — write a snapshot and reference it from the event.

```python
import hashlib, json, time

class AuditTrail:
    def __init__(self, sink):
        self.sink = sink
        self.prev_hash = "0" * 64

    def _emit(self, event_type, payload):
        event = {
            "ts": time.time(),
            "type": event_type,
            "payload": payload,
            "prev": self.prev_hash,
        }
        body = json.dumps(event, sort_keys=True).encode()
        event["hash"] = hashlib.sha256(body).hexdigest()
        self.prev_hash = event["hash"]
        self.sink.write(event)
        return event["hash"]

    def decision(self, choice, snapshot):
        snap_ref = self.sink.write_snapshot(snapshot)
        return self._emit("decision.made", {
            "choice": choice,
            "snapshot_ref": snap_ref,
        })
```

The caveats matter. Hash chaining gives tamper *evidence*, not tamper *prevention* — anyone with write access to the sink can rewrite the chain, and detection requires an external anchor. For real tamper resistance, periodically publish the latest hash to a separate system (a timestamping service, a separate account, a write-once store).

Snapshots also raise a data protection question. If the snapshot includes retrieved documents or user input, it inherits whatever retention and access rules apply to that data. An audit trail is not exempt from GDPR or similar regimes just because it is labeled "audit." Teams commonly need a redaction pass or a separate retention policy for snapshot payloads versus event metadata.

Finally, neither approach solves the *quality* problem. A complete audit trail of a bad decision is still a bad decision. The trail makes the decision explainable; it does not make it correct.

## Final verdict

Event sourcing wins on auditability, cost, and tamper evidence. Snapshots win on debugging fidelity and handling non-determinism. For LLM-driven agents in regulated contexts, the honest answer is that you need both, and the interesting engineering work is minimizing how much snapshot data you actually have to keep.

The teams that get this right tend to make one structural decision early: the event log is the source of truth, and snapshots are derived artifacts referenced from it. Teams that treat snapshots as the primary record end up reverse-engineering causality from diffs, which is exactly the work the event log was supposed to save.

## Frequently Asked Questions

**How long should audit trails for agent decisions be retained?**

Retention is driven by the regulation that applies to your domain, not by engineering preference. Financial services rules in many jurisdictions require multi-year retention of decision records. GDPR does not specify a fixed period but requires that retention be justified and limited. The practical pattern is to retain the hash-chained event log for the regulatory period and retain snapshots for a shorter window, typically aligned with your incident response and debugging needs. Document the retention policy explicitly so auditors can see it was a decision, not an omission.

**Why does replaying an event log not reproduce LLM agent behavior?**

Replay assumes the same inputs produce the same outputs. LLM inference with non-zero temperature is stochastic, so the same prompt can produce different completions. External tools may also return different results on replay if their underlying state changed. The event log records what happened; it does not guarantee you can re-execute it. This is why snapshots that record actual outputs are necessary for debugging non-deterministic agents.

**What is the minimum viable audit trail for a compliance review?**

The minimum is an append-only record that answers three questions for any decision: what was the input, what was the decision, and what rule or policy justified it. That can be as simple as structured events with a policy reference. What pushes teams beyond the minimum is the follow-up question: "prove this record was not altered after the fact." Hash chaining addresses that. Without it, a compliance review may accept the record but flag the absence of tamper evidence as a gap.

**When should snapshots be taken instead of events?**

Snapshots are the better primary structure when the agent is non-deterministic and the primary consumer is a debugging engineer rather than a regulator. If you cannot replay to reproduce state, the log alone cannot answer "what did the agent see." In that case, snapshot at every decision point and keep a lightweight event log for the causal chain. If your agent is deterministic and the audit question is about sequence, events alone are sufficient.

## Take action in the next 30 minutes

Open your agent's decision-emitting code path and check one thing: does every state transition that affects an external outcome produce a record? If any transition can happen without emitting an event, your event log is not authoritative and replay will diverge from reality. Find the first such transition, add the emit call, and commit it. That single change is the difference between a log and an audit trail.
