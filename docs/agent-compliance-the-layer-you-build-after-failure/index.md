# Agent compliance: the layer you build after failure

The workaround gets copy-pasted forward long after the original reason is forgotten. Benchmarks for governance layer that don't mention their failure conditions aren't worth much. This post covers what comes after the happy path.

## The conventional wisdom (and why it's incomplete)

The conventional wisdom says that agent systems need governance from day one. Put a policy engine in front of every tool call, log every decision, require human approval for anything that touches production data, and you'll never have a compliance incident. This sounds responsible. It's also the reason so many agent platforms stall at prototype stage: the governance layer becomes a tax on every iteration, and teams either abandon the project or bolt on a minimal, brittle gate that passes audits rather than prevents failures.

The incomplete part is the assumption that governance is a *preventive* control. In practice, most compliance-relevant failures in agent systems are not prevented by pre-execution gates. They are discovered after the fact — by a customer complaint, a regulator inquiry, a security review, or a sudden spike in a cost dashboard. The governance layer that actually matters is the one you build *after* that failure. It's the reconstruction layer: the ability to answer, within hours, what the agent did, why it did it, what data it touched, and what should have stopped it.

This post argues that the post-failure governance layer is the real deliverable. Pre-execution policy is necessary but insufficient. The teams that recover fastest from a compliance incident are not the ones with the most elaborate approval workflows — they are the ones with immutable, queryable traces that let them reconstruct the decision path and patch the specific gap. The part that trips people up is that most agent frameworks optimize for *action* (tool calls, model invocations) and treat *reconstruction* as an afterthought, which is exactly backwards when compliance is on the line.

## What actually happens when you follow the standard advice

Following the standard advice means building a policy engine, a human-in-the-loop queue, and a logging pipeline. In a typical stack — say, an agent built on LangChain 0.1.x calling OpenAI's gpt-4-turbo through a FastAPI 0.110 backend — you end up with something like this:

```python
# policy_gate.py — the standard pre-execution check
from fastapi import HTTPException
from pydantic import BaseModel

class ToolCall(BaseModel):
    tool: str
    args: dict
    user_id: str

POLICY = {
    "read_customer_record": {"allowed_roles": ["support", "admin"]},
    "issue_refund": {"allowed_roles": ["admin"], "max_amount": 500},
}

def check_policy(call: ToolCall, role: str):
    rule = POLICY.get(call.tool)
    if not rule:
        raise HTTPException(403, f"tool {call.tool} not governed")
    if role not in rule["allowed_roles"]:
        raise HTTPException(403, "role not permitted")
    if "max_amount" in rule and call.args.get("amount", 0) > rule["max_amount"]:
        raise HTTPException(403, "amount exceeds policy")
    return True
```

This works for the obvious cases. It fails for the ones that actually cause compliance incidents. A common [failure mode: the agent](/agent-drift-the-failure-mode-you-didnt-log/) calls `read_customer_record` with a `user_id` that belongs to a different tenant because the retrieval step mixed up context from a shared vector store. The policy gate sees a permitted tool and a permitted role. It passes. The data leak is not in the tool call; it's in the *argument provenance*. Pre-execution policy almost never inspects how an argument was constructed.

Another common pattern: the agent chains three permitted calls — `read_customer_record`, `summarize`, `send_email` — and the summary includes PII that the email tool then sends to an external address. Each call passes policy individually. The composite action violates data-handling rules. This is the classic confused-deputy problem, and it's well documented in agent security literature. Pre-execution gates that evaluate calls in isolation cannot catch it.

What happens next is predictable. The incident surfaces weeks later. The team scrambles to reconstruct what happened. If they logged only the final tool call, they cannot. If they logged every call but not the model's reasoning or the retrieval context, they cannot explain *why* the agent chose that customer record. The governance layer that would have helped — a full decision trace — was never built, because the standard advice focused on stopping actions, not explaining them.

## A different mental model

Think of governance as two layers: the *brake* and the *flight recorder*. The brake is pre-execution policy. The flight recorder is post-execution reconstruction. Most teams invest 90% in the brake and 10% in the recorder. The right ratio, if compliance is a real requirement, is closer to 40/60.

The flight recorder has three properties that pre-execution policy cannot provide:

1. **Temporal completeness.** It captures the sequence, not just the final state. You can see that the agent retrieved document A, then document B, then synthesized a response that combined them. Policy gates see the synthesis call, not the retrieval chain.

2. **Argument provenance.** It records not just that `read_customer_record(user_id="cus_123")` was called, but that `cus_123` came from a vector search result with a similarity score of 0.82 against a query that included a different tenant's identifier. This is the difference between "the agent read a record" and "the agent read the wrong record because the retrieval step leaked context."

3. **Counterfactual clarity.** It lets you ask, after the fact, what would have stopped this. Was it a missing policy rule? A retrieval filter? A model prompt that should have included a tenant constraint? You can only answer that if you have the full trace.

The mental shift is from *preventing* to *reconstructing*. Prevention is probabilistic — you will miss something. Reconstruction is deterministic — if you logged it, you can replay it. The governance layer that matters is the one that turns an incident from a multi-week forensic mystery into a two-hour query.

## Evidence and examples from real systems

Consider a typical customer-support agent deployed on AWS Lambda (Python 3.11 runtime) with DynamoDB for session state and OpenSearch for retrieval. The agent has access to `get_order`, `issue_refund`, and `send_email`. A compliance-relevant failure occurs when the agent issues a refund to the wrong customer because the session ID was reused across two concurrent conversations — a race condition in the state store.

With only pre-execution policy, the trace looks like this:

```json
{"timestamp": "2026-01-15T10:23:41Z", "tool": "issue_refund", "args": {"order_id": "ord_9981", "amount": 120}, "result": "success"}
```

This tells you a refund happened. It does not tell you which conversation triggered it, which user was authenticated, or what the agent's reasoning was. Reconstructing the incident requires correlating Lambda request IDs, DynamoDB stream records, and CloudWatch logs — a process that typically takes 6–8 hours across two engineers.

With a flight-recorder layer, the trace includes the decision context:

```json
{
  "trace_id": "tr_7f3a9c",
  "span": "tool_call",
  "tool": "issue_refund",
  "args": {"order_id": "ord_9981", "amount": 120},
  "provenance": {
    "order_id": {"source": "retrieval", "doc_id": "doc_4412", "score": 0.91},
    "session_id": "sess_abc",
    "authenticated_user": "user_5521",
    "concurrent_session": "sess_xyz",
    "model_reasoning": "User requested refund for order mentioned in previous message."
  },
  "policy_checks": [{"rule": "refund_max_500", "passed": true}],
  "latency_ms": 412
}
```

With this, the reconstruction is a single query: find all traces where `session_id` was shared across `authenticated_user` values. The root cause — session reuse — is visible in seconds. The fix is a session key that includes the authenticated user ID, plus a policy rule that rejects tool calls when the session's user does not match the authenticated user.

The cost difference is not trivial. A post-incident forensic process that takes 8 engineer-hours at a fully loaded rate of roughly $95/hour costs about $760 per incident. A flight-recorder query that takes 15 minutes costs about $24. More importantly, the time-to-remediation drops from days to hours, which matters when a regulator is asking for a timeline.

A second example: an agent that summarizes customer emails and posts them to a Slack channel. The agent uses a retrieval step that pulls from a shared index. A compliance failure occurs when a summary includes a customer's full credit card number because the retrieval step surfaced a document containing it. Pre-execution policy permitted the `post_to_slack` call. The flight recorder shows that the retrieved document had a `pii_flag` field set to `true`, but the agent's prompt did not instruct it to redact. The fix is a pre-retrieval filter on `pii_flag` and a post-retrieval redaction step. Without the trace, the team would likely have blamed the model or added a generic "do not include PII" instruction, which is unreliable.

These are not exotic scenarios. They are the ordinary shape of agent failures: correct tool calls with incorrect context. The governance layer that catches them is the one that records context, not just calls.

| Approach | Catches wrong-tool calls | Catches wrong-context calls | Reconstruction time | Typical cost per incident |
|---|---|---|---|---|
| Pre-execution policy only | Yes | No | 6–8 hours | ~$760 |
| Logging tool calls only | Yes | No | 4–6 hours | ~$570 |
| Full decision trace | Yes | Yes | 15–30 minutes | ~$24–48 |
| Trace + automated replay | Yes | Yes | 5–10 minutes | ~$8–16 |

## The cases where the conventional wisdom IS right

Pre-execution policy is not useless. It is exactly right for a specific class of failures: those where the action itself is categorically prohibited, regardless of context. If your agent is never allowed to delete a production database, a policy gate that blocks `drop_table` is correct and sufficient. If your agent must never issue a refund above $500 without human approval, a policy gate is the right control. These are *action-level* rules, and they are cheap to enforce.

The conventional wisdom is also right about human-in-the-loop for high-stakes, low-frequency actions. If your agent can wire money, a human approval step is appropriate. The mistake is extending that logic to every action. A support agent that reads customer records thousands of times a day cannot have a human approve each read. The governance layer for that agent must be the flight recorder, not the approval queue.

Finally, the conventional wisdom is right that governance must be designed, not accreted. But "designed" should mean designing the trace schema and the query interface first, then adding policy rules as you discover gaps. Most teams do the reverse: they design policy rules first and treat logging as an infrastructure concern. That ordering is why post-incident reconstruction is so painful.

## How to decide which approach fits your situation

The decision hinges on two questions: how reversible are the agent's actions, and how sensitive is the data it touches?

If actions are reversible and data is low-sensitivity — for example, an agent that drafts internal documentation — pre-execution policy is probably enough. A simple allowlist of tools and a basic audit log will satisfy most reviews.

If actions are irreversible or data is sensitive — refunds, emails to customers, access to PII — you need the flight recorder. The rule of thumb: if a regulator could ask "show me exactly why the agent did this," you need a trace that answers that question without human archaeology.

A practical decision matrix:

| Action reversibility | Data sensitivity | Recommended governance |
|---|---|---|
| Reversible | Low | Pre-execution policy + basic audit log |
| Reversible | High | Pre-execution policy + decision trace |
| Irreversible | Low | Pre-execution policy + human approval for high-impact actions |
| Irreversible | High | Pre-execution policy + decision trace + human approval + automated replay |

For most teams building customer-facing agents, the answer is the third row or fourth row. That means the flight recorder is not optional. It is the primary control, and policy is the secondary one.

## Common objections, and responses

**"Decision traces are too expensive to store."** A trace with full context — retrieval documents, model reasoning, tool args — might be 2–5 KB per step. An agent that executes 50 steps per conversation and handles 10,000 conversations per day generates about 1–2.5 GB per day. At S3 Standard pricing (around $0.023 per GB-month), that's under $2 per month for storage. The compute to write it is negligible. The objection is usually about schema design effort, not cost.

**"We can't log model reasoning because it contains PII."** You can. Redact at write time using a deterministic tokenizer, or store reasoning in a separate encrypted store with stricter access controls. The trace does not need to contain raw PII; it needs to contain enough to reconstruct the decision. A hash of the PII plus a pointer to the source document is often sufficient.

**"Our agent framework doesn't support this."** Most frameworks — LangChain 0.1.x, LlamaIndex 0.10.x, CrewAI 0.30.x — have callback hooks or event emitters that let you capture steps. If yours doesn't, wrap the tool-calling interface. The flight recorder is a cross-cutting concern; it should not depend on framework support.

**"We'll add it after we have an incident."** That is the definition of building governance after a failure, which is the thesis of this post. The point is that you should build the *reconstruction* layer before you need it, because the incident will not wait for you to design a schema. The brake can be minimal; the recorder should be ready.

## What the alternative approach would change

If teams treated the flight recorder as the primary governance artifact, several things would change. First, the trace schema would be a first-class design document, reviewed alongside the agent's prompt and tool definitions. Second, incident response would be a query, not a project. Third, policy rules would be derived from trace analysis — you would see which contexts lead to violations and write rules that target those contexts, rather than guessing. Fourth, compliance reviews would shift from "show us your approval workflow" to "show us a trace of a real decision," which is a much stronger answer.

The cultural change is the hardest part. Pre-execution policy feels like control. The flight recorder feels like overhead until the first incident, at which point it feels like the only thing that mattered. The teams that internalize this build agents that are not just governed but *explainable*, which is increasingly the actual regulatory requirement.

## Summary

The governance layer that matters after a compliance-relevant failure is not the policy gate you built before it. It is the decision trace you built to reconstruct it. Pre-execution policy catches categorical violations; it does not catch wrong-context actions, which are the majority of agent failures. A flight recorder that captures sequence, argument provenance, and counterfactual clarity turns an 8-hour forensic project into a 15-minute query and reduces cost per incident from roughly $760 to under $50. The conventional wisdom is right about action-level rules and human approval for high-stakes actions, but wrong about treating logging as an afterthought. Build the trace schema first, derive policy from it, and treat reconstruction as the primary control.

Your next step: open your agent's tool-calling wrapper and add a single structured log line that captures the provenance of one argument — for example, the document ID and similarity score that produced an `order_id`. Run one conversation, then query that log for the provenance field. If you can answer "where did this argument come from" in under a minute, you have the beginning of a flight recorder. If you cannot, you have just found the gap that a compliance incident would expose.

## Frequently Asked Questions

**What is a compliance-relevant failure in an agent system?**
It is any action or outcome that violates a legal, regulatory, or contractual obligation — for example, a data leak, an unauthorized transaction, or a failure to honor a deletion request. These failures are often not caused by a prohibited tool call but by a permitted call with incorrect context, such as the wrong customer record or an unredacted PII field. The governance layer that addresses them must capture context, not just actions.

**How do I log agent decisions without storing PII?**
Store hashes or tokens instead of raw values, and keep a separate, access-controlled mapping if you need to resolve them. For model reasoning, redact at write time using a deterministic tokenizer or store it in an encrypted store with stricter IAM policies. The trace needs enough to reconstruct the decision path, not the raw data itself. This approach is compatible with GDPR and CCPA if the mapping store is properly governed.

**Why isn't pre-execution policy enough for agent governance?**
Pre-execution policy evaluates each tool call in isolation, so it cannot detect composite violations (e.g., three permitted calls that together leak data) or argument-provenance errors (e.g., a permitted tool called with an argument from the wrong tenant). Those are the failure modes that most often trigger compliance incidents. Policy is necessary for categorical rules but insufficient for context-dependent ones.

**What tools can help build a decision trace for agents?**
OpenTelemetry (spec 1.27) with custom spans is a common foundation, paired with a storage backend like ClickHouse 24.x or AWS OpenSearch. For agent-specific tracing, LangSmith and Arize Phoenix offer instrumentation, but you can also implement a lightweight wrapper around your tool-calling interface. The key is to define a schema that includes provenance, not just the call and result.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
