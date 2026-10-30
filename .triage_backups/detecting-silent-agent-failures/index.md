# Detecting silent agent failures

An agent that returns a 200 OK with a plausible-looking answer is the hardest failure mode to catch. There is no exception, no stack trace, no red dashboard — just a confident paragraph that happens to be wrong, or subtly harmful, or a hallucinated citation a customer will act on. This is a build guide: a small, runnable quality gate that sits between an agent's output and whatever consumes it, so "the agent responded" stops being the same thing as "the agent worked."

The mechanism is not novel. It is the same pattern used in content moderation and CI pipelines: define cheap, deterministic checks that catch obvious failures first, then layer a model-based judge on top for the cases that need judgment. What makes the ordering worth writing down is the failure mode it prevents — teams commonly build the expensive judge first, wire it to a threshold picked arbitrarily, and then discover it flags their own good outputs more often than the bad ones.

## The problem this solves

Agent output fails in three distinct ways, and they need different detectors.

**Format failures** are the easy ones. The model returned prose where JSON was needed, omitted a required field, or produced a value outside the allowed enum. These are deterministic and cheap to check. If structured output is not already validated against a schema, that is the first thing to fix and it is not really the topic of this post.

**Grounding failures** are the interesting middle. The output is well-formed and on-topic, but it asserts something that is not supported by the context it was given. A retrieval-augmented agent cites a document that does not contain the claim. A tool-using agent reports a number that does not match what the tool returned. This is where most silent damage happens, because the output looks exactly like a correct one.

**Policy failures** are the ones that get you in trouble with someone else. The output contains PII it should not have echoed, gives medical or legal advice the product is not licensed to give, or reproduces a competitor's trademark in a way legal will ask about. These are partly deterministic (regex for obvious patterns) and partly judgment calls.

A quality gate is a function that takes the agent's output plus its context and returns a verdict — pass, flag, or block — along with the reason. The reason matters as much as the verdict, because it is what you will need to debug the gate itself.

## Prerequisites and what you'll build

You need Python 3.10 or later and an API key for whatever model provider you already use. The example uses the OpenAI Python SDK's chat completions interface, which is the most widely mirrored shape across providers — on a different provider, the swap is usually a base URL and a model name.

What you will build:

1. A `Check` protocol so each detector is a small, independently testable unit.
2. Three deterministic checks: schema validation, PII pattern scan, and a groundedness check that compares claims against the retrieved context.
3. One model-based judge for the cases the deterministic checks cannot decide.
4. A runner that executes checks in cost order and short-circuits on the first block.
5. Logging that records every verdict so you can measure the gate's own false-positive rate.

Install what you need:

```bash
pip install openai pydantic
```

Pydantic is doing the schema work. It is a well-documented library and the version matters — Pydantic v2 changed the validator API significantly from v1, so pin it in your requirements file.

## Step 1 — define the contract before the checks

The mistake to avoid here is writing detectors against free-form output. If the agent can return anything, every check needs its own parser and they will disagree about what the output even is.

Define a Pydantic model for the expected output shape. This is your contract.

```python
from pydantic import BaseModel, Field
from typing import Literal

class AgentOutput(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)
    confidence: Literal["low", "medium", "high"]
    cited_sources: list[str] = Field(default_factory=list)
```

The `confidence` field is doing real work. Asking the model to self-report confidence is unreliable as a signal on its own — models are systematically overconfident — but it is useful as a routing hint. A `low` confidence answer and a `high` confidence answer should not go through the same gate.

If your agent already returns structured output via a provider's JSON mode or function calling, you have this for free. If it returns prose, wrap it: one cheap extraction call to force the shape, then validate. The extraction call is a real cost, but it is far cheaper than discovering downstream that the output cannot be parsed at all.

## Step 2 — write deterministic checks first

The ordering principle: run the cheapest check that can produce a verdict first, and stop as soon as you have one. A PII regex costs effectively nothing; a model-based judge costs a full inference call. If the regex already blocks the output, the judge is wasted money.

Define a small protocol so checks are uniform:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class Verdict:
    action: str  # "pass", "flag", "block"
    reason: str
    check_name: str

class Check(Protocol):
    name: str
    def run(self, output: AgentOutput, context: str) -> Verdict | None:
        """Return None to abstain, a Verdict to decide."""
        ...
```

The `None` return is the important design choice. A check that cannot decide should abstain, not default to `pass`. Defaulting to pass is how gates end up rubber-stamping everything.

Here is a PII scanner. This is deliberately crude — it is a first filter, not a compliance system:

```python
import re

class PIICheck:
    name = "pii_scan"

    # Illustrative patterns. Tune to your jurisdiction and data.
    PATTERNS = {
        "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
        "phone_na": re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
        "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def run(self, output: AgentOutput, context: str) -> Verdict | None:
        hits = [name for name, pat in self.PATTERNS.items()
                if pat.search(output.answer)]
        if hits:
            return Verdict("block", f"pii patterns: {','.join(hits)}", self.name)
        return None
```

Two things to be honest about. First, this will produce false positives — a support agent quoting a customer's own email back to them is not a leak, but this check cannot tell the difference. That is why the verdict is `block` with a reason string rather than a silent drop: the decision needs to be auditable. Second, regex PII detection has well-documented gaps; names, addresses, and non-US phone formats will slip through. Treat it as a seatbelt, not a cage.

## Step 3 — the groundedness check

This is the check that catches the failure mode the post is about. The output is well-formed, passes the regexes, and is still wrong because it claims something the context does not support.

The cheap version: for each sentence in the answer that contains a factual claim, check whether the cited source actually appears in the context. This catches fabricated citations, which are the most common and most embarrassing grounding failure.

```python
class CitationCheck:
    name = "citation_grounding"

    def run(self, output: AgentOutput, context: str) -> Verdict | None:
        if not output.cited_sources:
            # No citations to verify. Abstain — let the judge decide.
            return None
        missing = [s for s in output.cited_sources if s not in context]
        if missing:
            return Verdict(
                "block",
                f"cited sources not in context: {missing}",
                self.name,
            )
        return None
```

The trap here is substring matching. If the agent cites `"report.pdf"` and the context contains `"report.pdf"` inside a longer path, this passes. If the agent cites a title and the context contains the title with different whitespace, this fails a legitimate citation. Normalize before comparing — strip, lowercase, collapse whitespace — and accept that there will still be some noise.

The expensive version compares claims, not citations. Split the answer into sentences, embed each one, and check that the maximum cosine similarity against any context sentence exceeds a threshold. This catches the case where the citation is real but the claim attached to it is not. It costs an embedding call per sentence. For most solo-built products, the citation check plus the judge below is sufficient; add embeddings only after measuring that citation-checking misses real failures.

## Step 4 — the model-based judge

The judge runs only on outputs that survived the deterministic checks. Its job is to answer one question: is this answer supported by the context, and does it comply with the stated policy?

```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """You are a quality gate for an AI agent.
Given the CONTEXT and the AGENT OUTPUT, decide:
- supported: does every factual claim follow from the context?
- policy_ok: does the output avoid giving regulated advice,
  echoing PII, or making promises the product cannot keep?

Respond with JSON only: {"supported": bool, "policy_ok": bool,
"reason": "<one sentence>"}
"""

class JudgeCheck:
    name = "model_judge"

    def run(self, output: AgentOutput, context: str) -> Verdict | None:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content":
                 f"CONTEXT:\n{context}\n\nAGENT OUTPUT:\n{output.answer}"},
            ],
            temperature=0,
        )
        try:
            verdict = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            return Verdict("flag", "judge returned unparseable JSON", self.name)

        if not verdict.get("supported"):
            return Verdict("block", verdict.get("reason", "unsupported"), self.name)
        if not verdict.get("policy_ok"):
            return Verdict("flag", verdict.get("reason", "policy"), self.name)
        return None
```

Three things to get right. Set `temperature=0` — the judge should be as deterministic as it can be, or the gate's behavior will drift between runs. Use a smaller, cheaper model for the judge than for the agent; the judge's task is narrower and does not need the strongest model. And always parse defensively: JSON mode reduces but does not eliminate malformed output, and a judge that crashes on bad JSON is worse than no judge.

A common trap: using the same model for the agent and the judge. If the agent hallucinated because the model has a blind spot, the judge with the same weights may share it. Using a different model family for the judge is a cheap way to decorrelate the failures.

## Step 5 — the runner and its ordering

```python
CHECKS = [PIICheck(), CitationCheck(), JudgeCheck()]

def gate(output: AgentOutput, context: str) -> Verdict:
    for check in CHECKS:
        verdict = check.run(output, context)
        if verdict and verdict.action == "block":
            return verdict
    for check in CHECKS:
        verdict = check.run(output, context)
        if verdict and verdict.action == "flag":
            return verdict
    return Verdict("pass", "all checks passed", "runner")
```

The two-pass structure is deliberate but naive — it runs every check twice. In practice each check's verdict should be cached on the first pass. The point is the ordering rule: blocks short-circuit, flags accumulate.

## Worked example: tracing one output through the gate

Take a support agent asked "what is the refund window?" The retrieved context contains a policy page stating 30 days, plus a customer's email address from a ticket header. The agent answers: "Refunds are accepted within 14 days. Contact billing@example.com to start."

Trace it:

1. **Schema validation** passes — `answer` is a non-empty string, `confidence` is `"high"`, `cited_sources` is `["refund_policy"]`.
2. **PIICheck** matches the email pattern in the answer and returns `block`. The runner short-circuits; the judge never runs.
3. If the email had not been present, **CitationCheck** would look for `"refund_policy"` in the context. It is there, so the check abstains — note that the check cannot see that the *number* attached to the citation is wrong.
4. **JudgeCheck** would then compare the claim against the context. The context says 30 days, the answer says 14, so `supported` is false and the judge returns `block`.

The lesson is that the 14-day error is invisible to every deterministic check. Only the judge catches it, and only because the context actually contains the correct number. If the retrieval step had failed to fetch the policy page, the context would be empty, the citation check would abstain, and the judge would have nothing to compare against. That is the residual risk no gate can close: grounding checks are only as good as the context.

## How to verify it works

A gate that has not been tested is a gate you do not have. Build a small labeled set — a few dozen examples is enough to start — with three buckets: known-good outputs, known-bad outputs, and borderline cases. Run the gate over them and record, for each check, how many good outputs it blocked and how many bad outputs it let through.

Two numbers per check matter: false-positive rate on the good bucket and false-negative rate on the bad bucket. Compute them as simple ratios over the bucket size, for example:

- Good bucket: 40 examples. PIICheck blocks 3 of them. False-positive rate = 3 / 40 = 0.075.
- Bad bucket: 40 examples. PIICheck blocks 22 of them. False-negative rate = 18 / 40 = 0.45.

Those figures are illustrative, not benchmarks — the point is that both are arithmetic you can reproduce on your own labeled set. A check that blocks good outputs will get disabled by whoever is on call; a check that misses bad outputs is decoration. There is no universal acceptable threshold — it depends on what a wrong answer costs — but a threshold cannot be chosen before it has been measured.

Log every verdict with the check name, the reason, and a hash of the output. When a customer reports a bad answer, the question to answer is: did the gate see it, and what did it decide?

## Choosing block vs flag

| Situation | Action | Why |
| --- | --- | --- |
| Output triggers a hard schema failure | Block | Downstream code cannot parse it anyway |
| PII pattern matched | Block | Irreversible if it reaches the user |
| Citation absent from context | Block | The claim has no support at all |
| Judge says unsupported, deterministic checks passed | Block | This is the silent-failure case the gate exists for |
| Judge says policy risk but claim is grounded | Flag | A human can adjudicate before use |
| Judge returned unparseable JSON | Flag | Fail open for availability, but record it |

The rule of thumb: block when a wrong answer causes irreversible harm — sending money, deleting data, giving regulated advice. Flag when a human can review before the output is used. Defaulting everything to block trains operators to bypass the gate; defaulting everything to flag means nothing is ever reviewed. Reserve blocks for the small set of failures you would page someone about.

## Frequently Asked Questions

**How do I detect hallucination in agent output?**

It cannot be detected directly, because a hallucination is indistinguishable from a correct answer without external ground truth. What can be checked is grounding: does every claim follow from the context the agent was given? That is the CitationCheck and JudgeCheck above. This will not catch a claim that is true but unsupported, and it will not catch a claim that is wrong in a way the context also gets wrong. It catches the common case, which is a claim the context does not support at all.

**What is a good false-positive rate for an LLM judge?**

There is no published number that applies to your product, and anyone quoting one is quoting a different task. The useful approach is relative: measure the judge against the deterministic checks on the same labeled set. If the judge's false-positive rate is higher than the check it is replacing, it is not earning its cost. A judge with a tight, specific prompt tends to beat a general one, because the general prompt invites the judge to invent reasons to block.

**Should the quality gate block output or just flag it?**

Block when a wrong answer causes irreversible harm. Flag when a human can review before the output is used. The mistake is defaulting everything to block, which trains operators to bypass the gate, or everything to flag, which means nothing is ever reviewed. Reserve blocks for the small set of failures you would page someone about.

**How much does a model-based judge add to latency?**

It adds one inference call per gated output. If the agent already makes several calls, the judge roughly adds one more to the chain. The mitigation is the ordering rule: run deterministic checks first so the judge only sees outputs that passed them. If the judge is still too slow, run it asynchronously and flag rather than block — you trade immediate protection for throughput.

## Where to go from here

The gate above is deliberately minimal. The next additions, in order of value: cache judge verdicts by output hash so identical outputs are not re-judged; add a per-check timeout so a slow check cannot stall the pipeline; and wire the verdict log into a dashboard so the block rate over time is visible. A block rate that jumps is a signal that something upstream changed — a prompt edit, a model version bump, a new retrieval source.

Start with one action: open your agent's output-handling code and find the single place where the output is currently trusted. Add a `PIICheck` and a `CitationCheck` there, log the verdicts to a file, and run your existing test suite through it. That produces a measured false-positive rate for two checks before a single line of judge code is written, and that number will tell you whether the judge is worth building at all.
