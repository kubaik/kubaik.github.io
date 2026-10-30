# Agent compliance in African fintech: the CBN gap

## The regulatory surface nobody named

Compliance writeups about autonomous agents tend to skip the part where the system breaks, because the failure is rarely in the code. It is in the mapping between what the agent does and what existing regulation already requires of the institution deploying it.

A common scenario in Lagos and other African fintech hubs: a payments team ships an agent to handle chargeback disputes. The agent reads transaction logs, drafts responses, and files them with the acquirer, with a human approving anything above a threshold. Resolution time for the auto-handled tier drops from a typical 36 hours to under 4 hours. The engineering works. Then a compliance review lands, and the question that stalls the rollout is not "does it work" but "who is the regulated entity when it acts."

That question is where most teams lose weeks. The engineering is tractable. The regulatory surface is not, and it is not the same surface a team in Berlin or San Francisco works against. A US team deploying an agent that touches money movement answers to a framework where the CFPB has published interpretive guidance on automated decisioning, where model risk management guidance (SR 11-7) is a known quantity, and where the worst case is usually a documented examination finding. An African fintech team answers to a patchwork: the Central Bank of Nigeria (CBN), the Nigeria Data Protection Commission (NDPC) under the NDPA 2023, the Securities and Exchange Commission where the agent touches anything resembling investment advice, and — if the agent moves money across borders — the recipient country's regulator plus whatever the correspondent bank's compliance desk decides it wants that week.

None of these regulators published a document titled "Rules for Autonomous Agents." The obligations are scattered across consumer protection circulars, data protection statutes, outsourcing guidelines, and payment system rules written before anyone imagined a system that could take an action without a human in the loop. The job is to map the agent's behavior onto those existing obligations, and the mapping is where teams lose weeks.

What follows is a worked case study of that mapping exercise: what a typical team tries first, why the obvious approach fails, and the architecture that survives a compliance review. Figures used are illustrative for a mid-size fintech processing on the order of 200,000 transactions a month, not audited results from a specific firm.

## The obvious first attempt and why it fails

The natural first design is to wrap the agent in a human approval step and call it done. Any action above a threshold goes to a human reviewer; everything below runs autonomously. A team sets the threshold at ₦50,000 and moves on.

This fails for three reasons that only become visible under review.

**The threshold is the wrong control.** Consumer protection frameworks generally do not care about transaction size when it comes to dispute handling — they care about whether the consumer received a fair hearing and whether the institution can produce a record of the decision. An agent that auto-rejects a ₦2,000 dispute has still made a regulated decision. The threshold controls financial exposure, not regulatory exposure, and those are different axes.

**The audit trail is a log file.** An agent that writes structured logs to stdout, which flow to a managed logging service, has logs. When a compliance officer asks "show me every decision this agent made on customer X between March 1 and March 31, and the basis for each," the answer requires a log query that takes 20 minutes to construct and returns unstructured text. The NDPA 2023 gives data subjects the right to meaningful information about automated decision-making, and "we have logs" is not the same as "we can produce a decision record."

**The model is a hosted endpoint.** A team calling a third-party LLM API is, from an engineering perspective, just making an HTTPS call. From a regulatory perspective, under CBN outsourcing guidelines, a regulated institution remains responsible for the activities of its service providers — and the service provider here is processing customer data outside the jurisdiction. That triggers a cross-border data transfer question under NDPA 2023 that the team may not have considered at all.

A common failure mode at this stage is treating compliance as a checkbox that lives in a document, separate from the system. The policy says the agent's decisions are reviewable. The system does not make them reviewable in any way a regulator would accept. The gap between policy and system is where rollouts stall — often weeks of back-and-forth, with the agent running in shadow mode the entire time.

## The three-layer architecture

The resolution is to stop treating the agent as a monolith and split it into three layers, each with its own regulatory posture.

**Layer 1: The decision engine.** This is the LLM call. It produces a recommendation, not an action. It is stateless, it receives only the minimum data needed, and its output is a structured object with a confidence score and a cited basis. Crucially, this layer is replaceable — the model provider can be swapped without changing anything downstream, which matters because no one can predict which provider will be acceptable to a regulator in 18 months.

**Layer 2: The policy gate.** This is deterministic code, not a model. It takes the recommendation and applies the institution's rules: is this action permitted at all, does it require human sign-off, does it need to be logged to the immutable store, does it trigger a customer notification. The policy gate is where the regulatory logic lives, and because it is code, it can be reviewed, tested, and version-controlled. When a regulator asks "what are your rules," the answer is a git tag.

**Layer 3: The execution layer.** This performs the action — filing the dispute response, sending the notification, updating the ledger. It writes to an append-only audit store before it acts, not after. If the action fails, the audit record still exists.

The key insight is that only Layer 1 is non-deterministic. Layers 2 and 3 are ordinary software with ordinary test coverage. This reduces the regulatory argument to a tractable question: "can you show that the deterministic gate prevents the model from taking any action outside your stated policy?" That is a question answerable with a test suite, not a promise.

This split also solves the cross-border data problem. The decision engine receives a redacted payload — no account numbers, no names, just the transaction pattern and the dispute reason code. The policy gate and execution layer run in-region, on infrastructure the institution controls. The LLM provider sees a fraction of the data, and the institution can document exactly what that fraction is.

## Implementation: the policy gate

The policy gate is the piece worth showing. This is Python 3.12 using Pydantic 2.7 for schema validation, so the recommendation object fails loudly if the model returns something malformed.

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime, timezone

class Action(str, Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    ESCALATE = "escalate"

class Recommendation(BaseModel):
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    basis_codes: list[str] = Field(min_length=1)
    amount_kobo: int

class Decision(BaseModel):
    recommendation: Recommendation
    final_action: Action
    requires_human: bool
    policy_version: str
    decided_at: datetime

POLICY_VERSION = "2026.03.1"

# Deterministic gate. No model calls here.
def apply_policy(rec: Recommendation, customer_tier: str) -> Decision:
    requires_human = False
    final = rec.action

    # Low confidence always escalates, regardless of amount.
    if rec.confidence < 0.72:
        final = Action.ESCALATE
        requires_human = True

    # Any rejection is a regulated decision. Human review required.
    if rec.action == Action.AUTO_REJECT:
        requires_human = True

    # High-value approvals need a second pair of eyes.
    if rec.action == Action.AUTO_APPROVE and rec.amount_kobo > 5_000_000:
        requires_human = True

    # Tier-1 customers get human review on anything contested.
    if customer_tier == "tier1" and rec.action != Action.AUTO_APPROVE:
        requires_human = True

    return Decision(
        recommendation=rec,
        final_action=final,
        requires_human=requires_human,
        policy_version=POLICY_VERSION,
        decided_at=datetime.now(timezone.utc),
    )
```

The important property is that `apply_policy` is pure. Given the same recommendation and customer tier, it returns the same decision. That makes it testable, and the test suite becomes the compliance artifact. A team would typically write tests against this function covering each branch and the boundaries. When an examiner asks how the institution ensures the agent cannot auto-reject a dispute without human review, the answer is a test named `test_auto_reject_always_requires_human` and a CI pipeline that fails if it is removed.

Note the confidence threshold: 0.72 is a policy choice, not a documented default. It should be set by measuring the model's calibration on a labelled sample of real disputes and choosing a cutoff where the false-escalation rate is acceptable to the operations team. The measurement is straightforward: run the model over a held-out set of past disputes, bucket outputs by confidence, and plot accuracy against confidence. Pick the threshold where accuracy meets the bar the business is willing to defend.

## Implementation: the audit store

The audit store is the second piece. It is append-only, and it is written before execution. A minimal version using PostgreSQL 16 with a write-only role:

```sql
CREATE TABLE agent_decisions (
    id            BIGSERIAL PRIMARY KEY,
    decision_id   UUID NOT NULL UNIQUE,
    customer_ref  TEXT NOT NULL,        -- pseudonymized, not the real ID
    policy_version TEXT NOT NULL,
    recommendation JSONB NOT NULL,
    final_action  TEXT NOT NULL,
    requires_human BOOLEAN NOT NULL,
    decided_at    TIMESTAMPTZ NOT NULL,
    executed_at   TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The application role can insert and select, never update or delete.
REVOKE UPDATE, DELETE ON agent_decisions FROM app_role;

-- Retention: NDPA 2023 and CBN record-keeping pull in different
-- directions. 7 years satisfies the stricter of the two for
-- financial records; personal data is minimized at write time.
```

The pseudonymization matters. The `customer_ref` is a token that maps to the real customer ID in a separate table with stricter access controls. The decision record itself contains no directly identifying information, which means producing a decision history for a data subject access request does not require exporting the whole table, and the cross-border exposure of the audit store is limited.

One gotcha worth naming: managed logging services commonly do not offer true append-only semantics. A role with write access to a log group can usually delete from it too. If the audit trail is a managed log service, the institution has a log, not an audit record. The distinction is exactly the kind of thing an examiner probes.

A second gotcha: the `REVOKE UPDATE, DELETE` statement above only works if the application connects as `app_role` and not as a superuser or table owner. Table owners retain the ability to drop or alter the table. In practice, the audit store should live in a separate database with separate credentials, and the application should hold only `INSERT` and `SELECT` on the one table. Verifying this is a two-minute exercise: connect as the application role and attempt a `DELETE`. If it succeeds, the control is not in place.

## What to measure, and how

Rather than trust a vendor's or a blog post's numbers, the useful exercise is to measure the properties that matter in the institution's own environment. Four measurements are worth running.

**Audit record production time.** Instrument the query path that produces a full decision record for a single customer over a date range. Time it. If it requires a general-purpose log search rather than an indexed lookup against `agent_decisions`, the number will be in the tens of minutes. The target is a sub-second indexed lookup, and the way to get there is to index on `(customer_ref, decided_at)`.

**Override rate.** Log every case where a human reviewer changed the agent's recommendation. The ratio of overrides to total reviews is the single best signal of whether human review is meaningful. A very low override rate combined with a very short median review time is strong evidence that the review is a formality, which regulators treat as functionally automated decisioning.

**Cross-border payload content.** For each call site to an external model, capture the outbound payload and run a regex sweep for account-number-like patterns, names, and identifiers. The count of matches should be zero. This is a CI check, not a one-off audit.

**Policy gate coverage.** Use coverage tooling on the policy gate module specifically. Branch coverage below 100% on this module means there is a code path that has never been exercised and therefore has never been reviewed.

## A decision checklist before shipping

Before an agent that touches customer money or customer data goes live, the following questions should have documented answers. If any answer is "we'll figure it out later," that is the finding.

- Can you list every action the agent can take that a customer would experience as a decision about them? Approving, rejecting, flagging, scoring, prioritizing — each is a regulated decision in most African jurisdictions, regardless of the amount attached.
- For each of those actions, is there a deterministic rule that governs it? Not a prompt. A rule, in code, under version control.
- Can you produce a complete decision record for any single decision the agent made last week — input, recommendation, rule applied, action taken, timestamps — without querying a general-purpose log?
- Is the audit store genuinely append-only for the credentials the application holds? Have you tested this by attempting a delete?
- Does the outbound payload to any external model contain directly identifying information? Have you verified this with a regex sweep in CI?
- Is there a data processing agreement in place with the model provider, with sub-processor disclosure and breach notification terms?
- Is the human review meaningful — can you show the override rate and the median review time?
- If the policy changes, how long does it take to deploy, and is the change traceable to a specific version?

## Frequently asked questions

**Does the CBN have specific rules for AI agents in fintech?**
Not under that name. The obligations come from the consumer protection framework, the outsourcing guidelines, and the payment system rules, all of which predate autonomous agents. The practical effect is that you map your agent's behavior onto existing obligations rather than pointing to an AI-specific rule. This is more work but it is not ambiguous — the underlying obligations are clear even when the technology is not named.

**Is calling a foreign LLM API a cross-border data transfer under NDPA 2023?**
If the payload contains personal data, yes, and the transfer has to be justified under the Act's provisions. The cleanest mitigation is to redact before the call so the payload contains no directly identifying information. That does not eliminate the question entirely — pseudonymous data can still be personal data — but it substantially narrows the exposure and makes the documentation far easier.

**How long do agent decision records need to be retained?**
Financial record-keeping obligations typically run to several years, and data protection law pulls in the other direction by requiring you not to keep personal data longer than necessary. The resolution is to minimize personal data at write time and retain the minimized record for the financial period. Storing full payloads for seven years is the wrong answer; storing a pseudonymous decision record for seven years is usually the right one.

**Can a human in the loop for everything avoid the problem?**
It avoids some of it, but a human approving a decision the agent made is still an automated decision if the human is rubber-stamping. Regulators look at whether the human review is meaningful. If the reviewer approves nearly all recommendations in under two seconds each, the review is a formality and the decision is functionally automated. Meaningful review means the reviewer can and does override, and the override rate is observable.

**Does this architecture work outside Nigeria?**
The pattern generalizes. The CBN, the NDPC, the SEC, and most other regulators ask variations of the same question: can you show that the system behaves within stated bounds, and can you produce a record of what it did. A deterministic gate answers the first. An append-only audit store answers the second. Neither requires the regulator to understand transformers. The jurisdiction-specific work is in the content of the policy gate — which rules apply, which thresholds are defensible — not in the architecture.

## The underlying principle

In a regulated environment, the agent's autonomy is not a property of the model — it is a property of the deterministic code around it. An agent does not become compliant because the model is safer. It becomes compliant because the model's output is constrained in what it can cause to happen, and because that constraint is inspectable.

This is why the three-layer split travels across jurisdictions. It also explains why the regional gap in automated-decisioning guidance is less of a disadvantage than it first appears. A team in a jurisdiction with mature guidance can point to a framework and say "we comply with X." A team in a jurisdiction where the guidance is implicit in older instruments has to do the mapping work itself. That is harder, but it is not worse — an implicit obligation that has been mapped and documented is more defensible than an explicit one that has merely been asserted. The work is the mapping, and the mapping is engineering work, not legal work. Treated that way, it gets done.

A final note on sequencing, because it is the mistake that costs the most. The right order is to write the policy gate first — before the model, before the prompt, before anything. The policy gate is the specification. It tells you what the model needs to output, what data it needs, and what the audit record must contain. Building the model first means reverse-engineering the gate from whatever the model happens to do, which is harder and produces worse controls. The second sequencing rule is to bring the compliance function in at the architecture-diagram stage, not the code-review stage. Compliance officers are not engineers, but a diagram that says "the model recommends, the gate decides, the ledger records" is readable by anyone, and delivered early it saves weeks. Delivered late, it reads as a retrofit.

## Action for the next 30 minutes

Open the agent's repository and search for every place it calls an external model API. For each call site, check whether the payload contains a customer name, account number, or any other directly identifying field. If it does, that is the first fix, and it is usually a one-function change — redact before the call, rehydrate after the response. Do that before anything else, because it is the change that most reduces cross-border exposure and it is the one that can ship today.
