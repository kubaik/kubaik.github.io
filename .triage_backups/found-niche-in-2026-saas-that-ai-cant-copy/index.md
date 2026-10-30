# Found niche in 2026: SaaS that AI can’t copy

## Why AI commoditizes some niches but not others

When a category of software can be replaced by a prompt, it usually is. Code generation, first-draft copywriting, and generic customer-support responses are the obvious examples. The pattern is not that AI destroys all software businesses; it is that AI exposes which parts of a workflow were always just text transformation.

Workflows built on unstructured interpretation — reading a PDF, deciding what a clause means, typing a date into a form — are the first to be automated away. Workflows built on structured verification — audit trails, versioned rules, deterministic calculations — tend to survive because the value is not in generating text. The value is in being able to prove, after the fact, that a specific rule was applied correctly on a specific date.

This article is about building in that second category. It uses regulatory compliance for small and medium enterprises as the worked example, but the reasoning applies to any domain where a single wrong output carries a real cost.

## The failure mode of an LLM-first compliance product

A common first attempt at regulatory software looks like this: ingest regulatory PDFs, parse them with a large language model, and emit alerts. The technical stack is unremarkable — object storage, a queue, an inference endpoint, and a notification service.

The failure mode is not performance. It is trust. Regulatory text is versioned, not continuous. A clause that was valid in January may be superseded in March. An LLM asked to summarize or extract deadlines produces plausible output, but plausible is not the same as correct. When the cost of an incorrect deadline is a revoked license or a fine, the customer cannot accept probabilistic output.

A typical incident: the system emits an alert that a license expires in three days. The compliance officer contacts the regulator and finds the license is current. The alert was a hallucination. That single event is often enough to lose the customer, and no amount of prompt engineering fixes the underlying issue, because the issue is architectural. The system has no mechanism to distinguish a verified fact from a generated one.

There is also a cost dimension. Inference on long regulatory documents is not free. If each request consumes a thousand or more tokens and the model is called per document per customer per day, the inference bill can exceed the price the customer is willing to pay. When margins compress, the product has no room to absorb a single churn event.

## The architectural alternative: deterministic rules with human review

The alternative is to keep the model out of the decision path entirely, or to confine it to a step where its output is reviewed before it becomes actionable.

The shape of the system:

1. **Ingestion.** A PDF is converted to structured text using an OCR and layout service. The output is blocks and lines, not meaning.
2. **Extraction.** A deterministic parser converts the structured text into candidate rules. This is where a model may help, but only as a proposal generator.
3. **Review.** A human — the customer's compliance officer — reviews the proposed rules, edits them, and approves them. The approval is recorded.
4. **Storage.** Approved rules are stored with explicit validity windows: a `valid_from` and a `valid_to` timestamp. When a regulation is amended, the old rule is archived rather than overwritten.
5. **Execution.** Deadlines, renewal dates, and required documents are computed from the stored rules by deterministic code. No model is involved at this stage.
6. **Notification.** Alerts are emitted through whatever channels the customer uses.

The important property is that every emitted alert can be traced back to a specific approved rule with a specific validity window and a specific reviewer. When a regulator asks why the system sent a particular notice, the answer is a record, not a model output.

This is sometimes described as human-in-the-loop. The more precise description is that the human owns legal interpretation and the system owns execution.

## A worked example: versioning a deadline

Consider a simplified rule. A license must be renewed annually, and the renewal window opens 60 days before expiry.

```go
// regulation.go
package rego

import "time"

// Regulation represents a single clause with versioning.
type Regulation struct {
    ID          string    `json:"id"`
    Title       string    `json:"title"`
    GazetteDate time.Time `json:"gazette_date"`
    Version     string    `json:"version"`
    Rules       []Rule    `json:"rules"`
}

type Rule struct {
    ID          string    `json:"id"`
    Description string    `json:"description"`
    Deadline    string    `json:"deadline"` // e.g. "2026-06-30"
    ValidFrom   time.Time `json:"valid_from"`
    ValidTo     time.Time `json:"valid_to"`
}
```

The `ValidTo` field is what prevents the failure mode described earlier. When a regulation is amended, the old rule is not deleted. It is given a `ValidTo` timestamp equal to the amendment date, and a new rule is inserted with a `ValidFrom` equal to that date.

A query for "what is the renewal deadline for this license today" then becomes a filter:

```
rules where ValidFrom <= today and (ValidTo is null or ValidTo > today)
```

If two rules match, that is a data error and should surface as an exception rather than silently picking one. This is the kind of check that is trivial in deterministic code and unreliable in a model.

## Parsing regulatory PDFs without trusting a model

OCR and layout analysis produce structured blocks. Converting those blocks into clauses is a parsing problem, and it is reasonable to use a model here — as long as the output is treated as a draft.

```python
# parser/textract_to_rego.py
import boto3
from typing import List, Dict

textract = boto3.client("textract", region_name="af-south-1")

def extract_regulations(pdf_bytes: bytes) -> List[Dict]:
    response = textract.analyze_document(
        Document={"Bytes": pdf_bytes},
        FeatureTypes=["LAYOUT"]
    )
    blocks = response["Blocks"]
    # Merge lines into paragraphs, then split into clauses.
    clauses = split_into_clauses(blocks)
    return clauses_to_rego_rules(clauses)
```

The `split_into_clauses` and `clauses_to_rego_rules` functions are where the real work lives, and they are the parts most likely to break. A common bug is date parsing. Regulatory PDFs in the same jurisdiction frequently use more than one date format. A parser that assumes `dd/mm/yyyy` will silently misread a `yyyy-mm-dd` document, and a misread date in a deadline field is exactly the class of error the architecture is supposed to prevent.

The mitigation is not a smarter model. It is explicit format detection with a fallback, plus a validation step that rejects any parsed date that falls outside a plausible range for the document's gazette date.

## Measuring whether the change actually helped

Claims about churn reduction or cost savings are only meaningful when the measurement method is stated. Here is how to instrument each of the metrics that matter for this kind of product.

**Churn.** Define the cohort precisely — for example, customers who signed up in a given month and were still paying at the end of the following month. Compute monthly churn as `(customers at start of month − customers at end of month) / customers at start of month`, excluding new signups. Report the number alongside the cohort size, because a churn rate computed on twelve customers is not the same evidence as one computed on two hundred.

**Inference cost.** Most inference providers expose per-request token counts. Log tokens in and tokens out per request, multiply by the published per-token price, and sum by day. Compare against the number of documents processed that day. The useful figure is cost per document, because that is what has to fit inside the price the customer pays.

**Latency.** Instrument the p50 and p95 of each stage separately: ingestion, parsing, rule evaluation, notification dispatch. A single end-to-end number hides which stage is the bottleneck. If rule evaluation is deterministic and in-process, it should be sub-millisecond; if it is not, something is wrong with the design.

**Support tickets.** Tag tickets by cause — parsing error, incorrect deadline, missing notification, billing. The ratio of "incorrect deadline" tickets to total active customers is the closest proxy for the trust problem this architecture is meant to solve.

**Audit survival.** This is the metric that matters most and is the hardest to measure. The practical version is: when a customer is audited, how many follow-up questions does the regulator ask about the system's records? Track this qualitatively per audit. A system that produces a clean, traceable record should produce fewer follow-ups than a manual process, but the only way to know is to ask the customer after each audit.

None of these require a benchmark table. They require logging and a defined denominator.

## Decision checklist before building

Use this to decide whether a domain is a good fit for a deterministic, human-reviewed system rather than an LLM-first product.

- **Is there a regulator or an auditor?** If a third party can demand to see how a decision was made, verifiability has value.
- **Is the source material versioned?** If rules change over time and the old version still matters for past periods, you need validity windows.
- **What is the cost of a wrong output?** If the answer is "the customer loses money or a license," probabilistic output is not acceptable in the decision path.
- **Is the domain narrow enough to model?** A single sector with a few hundred licensed businesses is easier to model correctly than "all regulated businesses."
- **Can a human review fit the workflow?** If the customer already employs someone who reads the regulation, the review step is not new work — it is the same work with better tooling.
- **Is the traffic steady or spiky?** Steady traffic favors a long-running process on a fixed instance; spiky traffic favors serverless. This affects latency and cost, not correctness.

If most of these are true, the deterministic architecture is likely to be more durable than an LLM-first one. If none are true, the domain may not need the audit trail, and a simpler product may be appropriate.

## Common mistakes when moving to deterministic rules

**Overwriting rules instead of versioning them.** The first instinct is to update a rule in place when a regulation changes. This destroys the ability to answer questions about past periods. Always insert a new version and close the old one.

**Building a custom review interface.** A pull-request workflow on a hosted Git provider already provides diffs, comments, approvals, and an immutable history. Rebuilding that interface is a large amount of work for little differentiation.

**Assuming a single date format.** Regulatory documents are inconsistent. Parse defensively and validate against the document's own metadata.

**Pricing per seat.** In a compliance product, the unit of value is the regulation being tracked, not the number of people logging in. A small organization may have few employees and many obligations. Pricing per regulation aligns the price with the value and scales with the customer's regulatory surface.

**Leaving the model in the notification path.** If a model can emit a deadline, it can emit a wrong deadline. Keep the model upstream of human review and downstream of nothing.

## FAQ

**Can a model be used anywhere in this architecture?**
Yes, in the extraction step, where its output is a draft that a human reviews before it becomes a rule. The constraint is that no model output should reach a customer without a human having approved it.

**How do you handle a regulation that changes frequently?**
The versioning model handles it without special cases. Each change produces a new rule with a new `ValidFrom` and closes the previous one with a `ValidTo`. The only additional work is ensuring the review step is fast enough that the customer is not exposed during the gap between publication and approval.

**What if the customer has no compliance officer?**
Then the review step is a service you provide, or the product is not a fit. A system that emits unverified deadlines is the failure mode this architecture exists to avoid.

**Does this approach scale to many jurisdictions?**
The rule storage and execution scale; the parsing does not, because every jurisdiction formats its documents differently. Budget for per-jurisdiction parser maintenance, and treat the parser as the part most likely to need ongoing attention.

**Is deterministic code always better than a model?**
No. For tasks where the output is judged on usefulness rather than correctness — summarization, drafting, search — a model is often the better tool. The distinction is whether a wrong output has a cost that the customer cannot absorb.

## Next step

Open a regulatory filing or compliance report that one of your target customers submits. Trace every data point in it back to its source, and note each step where a human reads a document and transcribes a value. That transcription step is the wedge: it is where errors enter, where audits focus, and where a deterministic, versioned, human-reviewed system provides value that a prompt cannot. Write the trace down in under 500 words. If you cannot, the domain is too broad to model yet.
