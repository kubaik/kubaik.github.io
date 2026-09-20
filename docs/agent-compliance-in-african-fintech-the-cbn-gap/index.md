# Agent compliance in African fintech: the CBN gap

Most regulatory compliance writeups assume the reader has already made the mistake they're warning about. This is the version of the write-up that includes the part that broke. It works in the simple case and breaks in a specific way under load.

## The situation (what we were trying to solve)

A Lagos-based payments team shipped an autonomous agent in early 2026 to handle chargeback disputes for merchants on their platform. The agent read transaction logs, drafted responses, and — with a human approving anything above ₦50,000 — filed them with the acquirer. It worked. Dispute resolution time dropped from a typical 36 hours to under 4 hours for the auto-handled tier. [Then the compliance](/audit-logs-compliance-vs-performance/) review landed, and the question that stopped the rollout was not "does it work" but "who is the regulated entity when it acts."

This is the gap that catches almost every African fintech team building agents in 2026. The engineering is tractable. The regulatory surface is not, and it is not the same surface a team in Berlin or San Francisco is working against. A US team deploying an agent that touches money movement answers to a framework where the CFPB has published interpretive guidance on automated decisioning, where model risk management (SR 11-7) is a known quantity, and where the worst case is usually a documented examination finding. [An African fintech](/african-fintech-rules-broke-our-stack/) team answers to a patchwork: the Central Bank of Nigeria (CBN), the Nigeria Data Protection Commission (NDPC) under the NDPA 2023, the Securities and Exchange Commission where the agent touches anything resembling investment advice, and — if the agent moves money across borders — the recipient country's own regulator plus whatever the correspondent bank's compliance desk decides it wants that week.

The part that trips people up is that none of these regulators published a document titled "Rules for Autonomous Agents." The obligations are scattered across consumer protection circulars, data protection statutes, outsourcing guidelines, and payment system rules that were written before anyone imagined a system that could take an action without a human in the loop. Your job is to map the agent's behavior onto those existing obligations, and the mapping is where teams lose weeks.

What follows is a case study of that mapping exercise — what a typical team tries first, why the obvious approach fails, and the architecture that actually survives a compliance review. The numbers are realistic figures for a mid-size fintech processing on the order of 200,000 transactions a month, not a single team's audited results.

## What we tried first and why it didn't work

The first attempt was the natural one: wrap the agent in a human approval step and call it done. Any action above a threshold goes to a human reviewer; everything below runs autonomously. The team set the threshold at ₦50,000 and moved on.

This failed for three reasons that only became visible under review.

First, the threshold was the wrong control. The CBN's consumer protection framework does not care about transaction size when it comes to dispute handling — it cares about whether the consumer received a fair hearing and whether the institution can produce a record of the decision. An agent that auto-rejects a ₦2,000 dispute has still made a regulated decision. The threshold controlled financial exposure, not regulatory exposure, and those are different axes.

Second, the audit trail was a log file. The team's agent wrote structured logs to stdout, which went to CloudWatch. When the compliance officer asked "show me every decision this agent made on customer X between March 1 and March 31, and the basis for each," the answer required a log query that took 20 minutes to construct and returned unstructured text. The NDPA 2026 gives data subjects the right to meaningful information about automated decision-making, and "we have logs" is not the same as "we can produce a decision record."

Third — and this is the one that is genuinely hard — the agent's model was a hosted endpoint. The team was calling a third-party LLM API. Under the CBN's outsourcing guidelines, a regulated institution remains responsible for the activities of its service providers, and the service provider here was processing customer data outside the jurisdiction. That triggered a cross-border data transfer question under NDPA 2023 that the team had not even considered, because from an engineering perspective it was just an HTTPS call.

A common failure mode at this stage is to treat compliance as a checkbox that lives in a document, separate from the system. The team had a compliance policy. The policy said the agent's decisions were reviewable. The system did not make them reviewable in any way a regulator would accept. The gap between policy and system is where the rollout stalled — roughly six weeks of back-and-forth, with the agent running in shadow mode the entire time.

## The approach that worked

The resolution was to stop treating the agent as a monolith and split it into three layers, each with its own regulatory posture.

**Layer 1: The decision engine.** This is the LLM call. It produces a recommendation, not an action. It is stateless, it receives only the minimum data needed, and its output is a structured object with a confidence score and a cited basis. Crucially, this layer is replaceable — the team could swap the model provider without changing anything downstream, which matters because you cannot predict which provider will be acceptable to your regulator in 18 months.

**Layer 2: The policy gate.** This is deterministic code, not a model. It takes the recommendation and applies the institution's rules: is this action permitted at all, does it require human sign-off, does it need to be logged to the immutable store, does it trigger a customer notification. The policy gate is where the regulatory logic lives, and because it is code, it can be reviewed, tested, and version-controlled. When the regulator asks "what are your rules," you hand them a git tag.

**Layer 3: The execution layer.** This performs the action — filing the dispute response, sending the notification, updating the ledger. It writes to an append-only audit store before it acts, not after. If the action fails, the audit record still exists.

The key insight is that only Layer 1 is non-deterministic. Layers 2 and 3 are ordinary software with ordinary test coverage. This means the regulatory argument reduces to a tractable question: "can you show that the deterministic gate prevents the model from taking any action outside your stated policy?" That is a question you can answer with a test suite, not a promise.

This split also solved the cross-border data problem. The decision engine receives a redacted payload — no account numbers, no names, just the transaction pattern and the dispute reason code. The policy gate and execution layer run in-region, on infrastructure the team controls. The LLM provider sees a fraction of the data, and the team can document exactly what that fraction is.

## Implementation details

The policy gate is the piece worth showing. This is Python 3.12, using Pydantic 2.7 for the schema validation, because you want the recommendation object to fail loudly if the model returns something malformed.

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

The important property is that `apply_policy` is pure. Given the same recommendation and customer tier, it returns the same decision. That means it is testable, and the test suite is the compliance artifact. The team wrote 140 tests against this function, covering each branch and the boundaries. When the regulator's examiner asked how the institution ensures the agent cannot auto-reject a dispute without human review, the answer was a test named `test_auto_reject_always_requires_human` and a CI pipeline that fails if it is removed.

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

One gotcha worth naming: teams commonly discover that their cloud provider's managed logging service does not offer true append-only semantics. A role with write access to a log group can usually delete from it too. If your audit trail is CloudWatch Logs or an equivalent, you have a log, not an audit record. The distinction is exactly the kind of thing an examiner probes.

## Results — the numbers before and after

| Metric | Before (monolithic agent) | After (three-layer) |
|---|---|---|
| Dispute resolution time (auto tier) | ~36 hours | under 4 hours |
| Audit record production time | 20+ minutes (log query) | under 30 seconds (indexed query) |
| Data sent to third-party model | full transaction payload | redacted pattern + reason code |
| Policy change deployment | code change + review | git tag + CI, ~15 minutes |
| Compliance review duration | ~6 weeks, stalled | ~2 weeks, approved |
| Test coverage on regulatory logic | 0 tests | 140 tests |
| Customer data in cross-border transfer | names, account numbers | pseudonymous reference only |

The resolution-time improvement is not the point — it was already fast. The point is that the audit production time dropped from a 20-minute manual query to a sub-30-second indexed lookup, and that is the difference between a system a compliance officer can supervise and one they cannot. A supervisor who can pull a decision record in 30 seconds will actually spot-check the agent. One who needs 20 minutes will not, and an unsupervised agent is the regulatory problem.

The compliance review duration is the number that matters most to the business case. Six weeks of stalled rollout on a system that was already built is a real cost, and it is the cost that teams consistently fail to budget for when they plan agent deployments in this market.

## What we'd do differently

The biggest mistake was sequencing. The team built the agent first and asked the compliance question second. The right order is to write the policy gate first — before the model, before the prompt, before anything. The policy gate is the specification. It tells you what the model needs to output, what data it needs, and what the audit record must contain. Building the model first means reverse-engineering the gate from whatever the model happens to do, which is harder and produces worse controls.

The second thing is to involve the compliance function at the architecture diagram stage, not the code review stage. Compliance officers are not engineers, but they can read a diagram that says "the model recommends, the gate decides, the ledger records." That sentence, delivered early, saves weeks. Delivered late, it reads as a retrofit.

The third is to treat the model provider as a vendor with a contract, not an API with a key. The cross-border data question is real, and the answer is partly technical (redact before sending) and partly contractual (data processing agreement, sub-processor disclosure, breach notification terms). Teams that skip the contractual half discover during review that their provider's standard terms do not actually permit the processing they are doing.

## The broader lesson

The principle is this: in a regulated environment, the agent's autonomy is not a property of the model — it is a property of the deterministic code around it. You do not make an agent compliant by making the model safer. You make it compliant by constraining what the model's output can cause to happen, and by making that constraint inspectable.

This is why the three-layer split works across jurisdictions. The CBN, the NDPC, the SEC, and whatever regulator you have not thought of yet all ask variations of the same question: can you show that the system behaves within stated bounds, and can you produce a record of what it did. A deterministic gate answers the first. An append-only audit store answers the second. Neither requires the regulator to understand transformers.

The regional gap matters here. A team in a jurisdiction with mature automated-decisioning guidance can point to a framework and say "we comply with X." An African fintech team usually cannot, because the guidance is implicit in older instruments. That is harder, but it is not worse — an implicit obligation you have mapped and documented is more defensible than an explicit one you have merely asserted. The work is the mapping, and the mapping is engineering work, not legal work. Treat it that way and it gets done.

## How to apply this to your situation

Start by listing every action your agent can take that a customer would experience as a decision about them — approving, rejecting, flagging, scoring, prioritizing. Each of those is a regulated decision in most African jurisdictions, regardless of the dollar amount attached. Then, for each one, write the deterministic rule that governs it. Not the prompt. The rule.

If you cannot write the rule, you do not yet know what your agent is allowed to do, and that is the finding a regulator will write up.

The second step is to check where your audit records actually live and whether they are truly append-only. Pick one decision your agent made in the last week and try to produce a complete record of it — the input, the recommendation, the rule applied, the action taken, the timestamp — without querying a general-purpose log. If you cannot, you have a gap.

## Frequently Asked Questions

**Does the CBN have specific rules for AI agents in fintech?**
Not under that name. The obligations come from the consumer protection framework, the outsourcing guidelines, and the payment system rules, all of which predate autonomous agents. The practical effect is that you map your agent's behavior onto existing obligations rather than pointing to an AI-specific rule. This is more work but it is not ambiguous — the underlying obligations are clear even when the technology is not named.

**Is calling a foreign LLM API a cross-border data transfer under NDPA 2023?**
If the payload contains personal data, yes, and the transfer has to be justified under the Act's provisions. The cleanest mitigation is to redact before the call so the payload contains no directly identifying information. That does not eliminate the question entirely — pseudonymous data can still be personal data — but it substantially narrows the exposure and makes the documentation far easier.

**How long do I need to retain agent decision records?**
Financial record-keeping obligations typically run to several years, and data protection law pulls in the other direction by requiring you not to keep personal data longer than necessary. The resolution is to minimize personal data at write time and retain the minimized record for the financial period. Storing full payloads for seven years is the wrong answer; storing a pseudonymous decision record for seven years is usually the right one.

**Can I just put a human in the loop for everything and avoid the problem?**
You can avoid some of it, but a human approving a decision the agent made is still an automated decision if the human is rubber-stamping. Regulators look at whether the human review is meaningful. If the reviewer approves 99% of recommendations in under two seconds each, the review is a formality and the decision is functionally automated. Meaningful review means the reviewer can and does override, and you can show the override rate.

## Resources that helped

The Nigeria Data Protection Act 2023 text itself is the primary source and is worth reading in full rather than relying on summaries — the automated decision-making provisions are in the sections on data subject rights, not in a separate AI section. The CBN's consumer protection framework and its outsourcing guidelines are both published and both relevant; read them together, because the agent touches both. The CBN's payment system vision documents provide context on where the regulator's thinking is heading. For the engineering side, the append-only audit pattern is well documented in the event-sourcing literature, and PostgreSQL 16's role-based access controls are sufficient to implement it without a separate ledger product.

The action to take in the next 30 minutes: open your agent's repository and search for every place it calls an external model API. For each call site, check whether the payload contains a customer name, account number, or any other directly identifying field. If it does, that is your first fix, and it is usually a one-function change — redact before the call, rehydrate after the response. Do that before you do anything else, because it is the change that most reduces your cross-border exposure and it is the one you can ship today.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
