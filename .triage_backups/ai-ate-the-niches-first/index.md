# AI ate the niches first

## The conventional wisdom, and where it breaks

A durable piece of startup advice says: find an underserved vertical, automate the painful workflow, and charge $20–$50 per seat per month before a large platform notices. The examples usually offered are real-estate agent tooling, e-commerce plugin catalogs, or local gym management — verticals the big players supposedly ignore.

The standard playbook has five steps:

1. Pick a vertical with fragmented workflows.
2. Build a thin CRUD wrapper around a spreadsheet.
3. Add integrations to the incumbent tools of that vertical.
4. Charge a modest monthly subscription per practitioner.
5. Wait for competitors to copy the interface.

The playbook assumes the interface is the product. That assumption is what changed. When a capable model can produce the primary output of a workflow — a summary, a classification, a structured record, a first-draft document — from the same inputs your product accepts, the interface stops being the thing customers pay for. The platform that already holds the data can ship that generation as a free checkbox.

The defensible verticals are the ones where generation is not the bottleneck: messy proprietary data, a legally required human signature, or real-time shared state. Everything else trends toward being a feature.

## The failure mode, described generically

A recurring pattern in vertical SaaS looks like this. A product wraps a platform's API and adds analysis on top: it ingests a customer's billing events, support tickets, or scheduling data and returns a forecast, a report, or a recommended action. It charges a low monthly fee.

Then the platform ships a native feature that does the same analysis on the same data, at no extra cost, inside the surface the customer already uses. The wrapper's differentiation was convenience, not capability. Retention collapses not because the product got worse but because the reason to leave the host platform disappeared.

The diagnostic question is not "is my product good?" It is "what does the customer have to leave, install, or pay for in order to use me, and can the platform eliminate that step?" If the answer is "nothing much," the product is exposed.

The same logic applies to a second pattern: a product whose core value is a single generation step (upload a document, get a summary). The cost of that generation falls toward zero, and the platform that owns the document upload flow can absorb it.

## Three moats that hold

Vertical SaaS products that have held up against free integrated AI features generally sit on one of three moats.

**1. Proprietary or regulated data the platform cannot legally touch.** If the model cannot be trained on your customers' data without a data processing agreement and explicit consent, the incumbent faces a legal cost to replicate your output, not just an engineering cost. This is a real constraint, but note it is a legal moat, not a technical one — it holds only as long as the regulation and the enforcement do.

**2. A human-in-the-loop step that is legally required.** If the billable artifact must be signed, notarized, or attested by a licensed professional, no amount of model capability removes the human. The AI can draft; the human must sign. This moat is durable because it is a licensing constraint, not a capability constraint.

**3. Real-time multiplayer state.** If the product must synchronize edits across many concurrent users with low latency and strong consistency, it is a distributed-systems problem, not a generation problem. Models do not solve conflict resolution.

The rest of this article works through how to tell which category you are in, with a concrete worked example.

## A worked example: the notarization workflow

Consider a service that notarizes documents over a live video call. The workflow is: the customer uploads a PDF, schedules a short video session with a notary, the notary verifies identity and witnesses the signature, and the service returns a notarized PDF with an audit trail.

Why this holds up:

- The notarization artifact is legally binding, and the notary is a licensed human. A model cannot perform the notarization.
- The audit trail has evidentiary value; courts care about the chain of custody, not the quality of the generated text.
- The incumbent platforms that own document storage have no incentive to become licensed notaries in every jurisdiction.

The generation step — drafting the document, filling fields, formatting the output — is the easy part and is not where the value sits. The value sits in the legally recognized act.

Now contrast with a scheduling tool for a vertical like dental labs. The workflow is: receive a lab order, parse it, schedule it, notify the technician. Every step is text processing and calendar manipulation. A model embedded in the practice-management system the dentist already uses can do all of it. The scheduling tool has no moat because the output is generated text and the data already lives in the incumbent.

The general rule: if the primary user action is "give data, receive generated result," the platform absorbs it. If the primary user action requires a licensed human, a signature, or coordination between many simultaneous editors, it does not.

## A decision checklist

Answer each question for your product. Two or more "yes" answers means you are exposed.

| Question | Yes means |
|---|---|
| Can a model produce the primary output from the same inputs your product accepts? | Generation is commoditised |
| Is the primary output text, structured JSON, or a semi-structured document? | Output is model-shaped |
| Can the workflow run entirely inside a platform your customers already pay for? | The platform can absorb you |
| Is the input data already available to a major model or a public dataset? | No data moat |
| Could a competent engineer replicate 90% of the functionality in 30 days with a modest budget? | Low replication cost |

If you answered "yes" to two or more, the product is likely to become a feature. The response is either to add a moat (proprietary data under agreement, a required human signature, real-time sync) or to change the product so the moat is the core.

## Audit your own UI surface

A fast way to see how exposed a product is: count how much of your UI is upload, download, click, and report. If the bulk of the interaction is data in and generated result out, the product is model-shaped.

The following script walks a repository and counts occurrences of those interaction verbs in source files. It is a rough signal, not a verdict — adjust the patterns to your stack.

```bash
#!/usr/bin/env bash
# Count interaction verbs that suggest a generate-and-return UI.
# Run from the repository root.

set -euo pipefail

patterns='upload|download|click|button|report|generate|export'

total=0
while IFS= read -r file; do
  count=$(grep -Eoi "$patterns" "$file" | wc -l | tr -d ' ')
  if [ "$count" -gt 0 ]; then
    printf '%6d  %s\n' "$count" "$file"
    total=$((total + count))
  fi
done < <(find . -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' \) \
  -not -path './node_modules/*' -not -path './.next/*')

echo "----"
echo "total interaction-verb matches: $total"
```

What to do with the number: it is only meaningful relative to your total surface area. A better ratio is interaction-verb matches divided by total lines of UI code. Compute both, track the ratio over time, and watch whether the product is drifting toward a generate-and-return shape or away from it.

If the ratio is high and the primary output is generated, treat the product as a feature unless you can point to a specific moat.

## Instrumenting whether a platform feature is eating you

The generic failure mode above is not something you detect from a dashboard after the fact. You detect it by watching a few leading indicators:

- **Support ticket themes.** Count tickets that ask for a capability the host platform now offers natively. A rising count is the earliest signal.
- **Signup source.** If new signups increasingly come from outside the platform ecosystem, the platform's native feature is doing the work for you. If they increasingly come from inside it, you are competing with a free checkbox.
- **Churn reason codes.** Add a "switched to a platform-native feature" reason code and track it separately from price and product churn.

None of these require a benchmark. They require you to instrument the funnel and read the reasons, not just the rate.

## Where the conventional wisdom still holds

Three categories have resisted commoditisation in practice.

**Industrial telemetry with proprietary protocols.** If the data is raw bus frames or a vendor-specific industrial protocol, no public model can parse your factory's tags without a substantial fine-tune on labeled data you own. The moat is the labeling effort and the protocol access, not the model.

**Multi-tenant products with per-tenant schema customisation.** Verticals where each customer customises fields, workflows, and retention rules require re-creating the schema and the customer's macros to replicate. The cost is engineering hours, not model capability, and it scales with customer count.

**Regulatory reporting that requires a licensed signer.** The auditor or the licensed professional still signs the report. The model drafts; the human attests.

In all three, the moat is a cost the incumbent would have to pay — legal, engineering, or licensing — not a capability the incumbent lacks.

## FAQ

**How do I tell if my niche is exposed?**

Run the checklist above. If two or more answers are "yes," assume exposure and either add a moat or reposition.

**Is a data moat enough on its own?**

Only if the data is genuinely inaccessible to the incumbent — regulated, proprietary, or under a contract that prevents training. "Our data is better" is not a moat if the incumbent already holds the same data.

**Do regulatory moats erode?**

They can, if the regulation changes or if a certification pathway for automated attestation appears. Treat a regulatory moat as durable but not permanent, and re-check it annually.

**What if my product has no moat but good retention?**

Retention is evidence, not a moat. Check whether the retention survives the platform shipping the same feature for free. If you cannot answer that, you do not yet know whether the retention is structural.

**How much does it cost to fine-tune a model for a niche?**

It depends on labeling volume, base model, and whether you need on-premises inference for compliance. The dominant cost is usually labeling and evaluation, not compute. Estimate it by pricing your labeling hours at your own rate and adding the inference cost at your expected request volume.

## Next step in the next 30 minutes

Run the audit script above against your primary repository, then compute the ratio of interaction-verb matches to total lines of UI code. Write the ratio down with today's date. If the ratio is high and your primary output is generated text or a structured document, add a "switched to a platform-native feature" reason code to your churn survey and check it again in 30 days. That single number will tell you more about your exposure than any market analysis.
