# When velocity is hard to measure: evaluating engineering…

## The symptom: the usual signals stop answering the question

Every engineering manager eventually hits a review cycle where the standard signals fail. The sprint burndown looks fine, but the roadmap hasn't moved. Story points are up, but the same incidents keep recurring. Or the opposite: the board looks quiet, yet customer-reported bugs are down and the on-call rotation is calmer. When no single number reliably reflects output, review decisions start to feel arbitrary — and teams notice.

The real problem usually isn't a missing metric. It's that a single metric is being asked to represent a system with multiple, loosely coupled outputs. The practical fix is a small portfolio of evidence that triangulates output without pretending to be precise. What follows is a decision framework, a worked example, failure modes, and a checklist to apply before the next review.

## Why velocity breaks as a measure of output

Velocity — completed story points per sprint — was designed to help a team forecast its own capacity, not to measure value. When it becomes a management target, three predictable distortions appear:

1. **Point inflation.** If points are tied to performance reviews, the cheapest way to improve the number is to estimate higher. This isn't malicious; it's rational. The metric stops measuring work and starts measuring negotiation skill.
2. **Slicing distortion.** Large, valuable items get broken into many small items to inflate counts, while genuinely complex work gets avoided because it doesn't fit neatly into a sprint.
3. **Local optimization.** Teams optimize for "done" in the board sense — tickets closed — rather than "done" in the operational sense — deployed, observed, and stable.

None of this makes velocity useless. It makes velocity a *capacity signal*, not an *output signal*. Treating it as an output signal is a category error, like using CPU utilization to judge whether a service is healthy: high CPU can mean load or it can mean a busy loop.

The same logic applies to lines of code, commit counts, or tickets closed. Each is a proxy that can be gamed or can drift from what matters. The question isn't "which metric is best?" but "what portfolio of evidence is hard to game and cheap to collect?"

## A decision framework: three layers of evidence

Instead of one number, evaluate output across three layers, each answering a different question:

| Layer | Question it answers | Example signals | Gaming risk | Collection cost |
|-------|---------------------|-----------------|-------------|-----------------|
| **Delivery** | Are we shipping what we said we would? | Roadmap items completed, cycle time from "in progress" to "deployed," deployment frequency | Medium — can be gamed by shrinking scope | Low — already in your tracker and CI |
| **Stability** | Is what we shipped staying shipped? | Change failure rate, mean time to restore (MTTR), incident count by severity | Low — incidents are hard to hide | Medium — requires incident tracking and monitoring |
| **Outcome** | Is the work moving a business or user metric? | Adoption of the feature, support ticket volume for the area, conversion or retention for the surface | High — attribution is fuzzy | High — requires instrumentation and agreement on the metric |

Use all three. Delivery alone rewards shipping junk. Stability alone rewards doing nothing. Outcome alone is too noisy at short intervals and too easy to attribute wrongly. Together they form a triangle: if any two sides look good and the third doesn't, there is a specific conversation to have.

This framing restates the DORA metrics (deployment frequency, lead time for changes, change failure rate, time to restore) and combines them with product outcome thinking. The value is in using them as a *portfolio*, not as a scorecard.

## Worked example: same velocity, different output

Assume two teams, both at 40 points per sprint (illustrative numbers, not measured). Team A ships 8 small items per sprint; Team B ships 3 larger items. On velocity alone they look identical. Add the other layers:

- **Team A:** change failure rate 25% (2 of 8 deploys require a hotfix), MTTR 4 hours, one Sev-2 incident per sprint. Outcome: feature adoption flat because the feature keeps breaking.
- **Team B:** change failure rate 8% (roughly 1 in 12 deploys), MTTR 45 minutes, no Sev-2 in the last three sprints. Outcome: adoption up 10% quarter-over-quarter (assume the metric is instrumented and agreed in advance).

Same velocity, very different output. The conversation with Team A isn't "work harder" — it's "your change failure rate is consuming delivery capacity; look at test coverage and deploy size." The conversation with Team B isn't "great job" — it's "document what you're doing so it can spread."

The numbers are illustrative; the structure is the point. Velocity alone hides the difference. The three-layer view makes it legible without requiring a perfect metric.

### Doing the arithmetic honestly

Two definitions matter more than the numbers themselves:

- **Deployment frequency** = production deploys in the period ÷ working days in the period. If deploys are manual, count releases instead.
- **Change failure rate** = deploys requiring a hotfix or rollback within a defined window ÷ total deploys in the same period.

The window choice changes the result, so state it. A 24-hour window captures immediate breakage; a 7-day window captures slower failures (data corruption, edge-case regressions) but also catches unrelated fixes. Pick one, write it down, and keep it stable so trends are comparable. If the window changes, treat the series as reset rather than comparing across the change.

## Fix 1 — Replace a velocity-only agenda with a delivery-and-stability pair

The most common cause of "output can't be measured" is that the review agenda has one slot: velocity. Change the agenda. For each team, prepare two numbers before the review: **deployment frequency** and **change failure rate**. Both are computable from data most teams already have if they use a CI/CD system and an issue tracker.

A spreadsheet with two columns per team, updated weekly, is enough to start the conversation. The act of collecting the number changes what people pay attention to.

**Symptom this fixes:** reviews where the only question is "why is velocity down?" and the answer is always "estimates changed."

## Fix 2 — Instrument the outcome, even crudely

The less obvious cause is that outcome data exists but isn't connected to engineering work. A feature ships, and nobody checks whether support tickets for that area went down. The fix is to pick one outcome metric per major initiative and define it *before* the work starts.

A practical pattern: for each roadmap item, write a one-line success statement in the ticket — for example, "reduce password-reset tickets by 20% within 30 days of launch." It doesn't have to be perfectly measurable; it has to be specific enough that someone can check it. At the next review, ask "did we check?" If the answer is no, that's the finding — not a failure of the team, but a gap in the process.

Privacy constraints apply here. If an outcome metric involves user data, teams need a lawful basis for processing it and a retention policy. A crude but compliant approach is to use aggregated, anonymized counts (number of reset requests per day) rather than per-user tracking. Under the EU's ePrivacy Directive and GDPR, aggregate operational metrics often do not require consent while per-user behavioral tracking usually does, but the specifics depend on jurisdiction and implementation — confirm with whoever owns privacy compliance and document the basis in the ticket. This isn't bureaucracy; it's what makes the metric defensible if someone asks.

**Symptom this fixes:** "We shipped it, but we don't know if it helped."

## Fix 3 — Join incident and on-call data to delivery data

Stability data commonly lives in a different tool from delivery data, and nobody joins them. Join them monthly.

A minimal join: for each incident, record the service or feature area and the deploy that preceded it, if any. Then compute, per team, incidents per deploy and MTTR. A shared spreadsheet with three columns (date, service, linked deploy) is enough to see patterns.

What to look for is not "zero incidents" — that can signal under-deployment or hidden failures. Look for a *stable or declining* incident rate as deployment frequency increases. If deployment frequency goes up and incident rate goes up proportionally, the team is shipping faster but not safer; the next investment is testing and observability, not more features.

**Symptom this fixes:** "The team says they're busy, but the same things keep breaking."

## How to verify the new approach is working

Causality can't be proven from a few sprints, but three leading indicators are checkable:

1. **The review conversation changes.** Instead of "why is velocity down?", the questions become "what's our change failure rate?" and "did we check the outcome metric?" If the conversation doesn't change, the data isn't being used.
2. **Teams can explain their own numbers.** Ask a team lead to walk through deployment frequency and change failure rate without preparation. If they can't, the data isn't visible enough.
3. **At least one decision changes.** A roadmap item gets re-scoped because the outcome metric was unrealistic, or a testing investment gets prioritized because change failure rate is high. If nothing changes, the process has added reporting without adding value.

A minimal script can pull deployment counts from a CI system's API (GitHub Actions runs, GitLab pipelines) and incident counts from an incident tool, then print a weekly summary. The example below assumes two exported CSVs — deploys with a timestamp and service, incidents with a timestamp, service, and severity — and computes incidents per deploy. It is illustrative, not production-ready, but it shows the arithmetic.

```python
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# deploys.csv: timestamp,service
# incidents.csv: timestamp,service,severity

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

deploys = load("deploys.csv")
incidents = load("incidents.csv")

cutoff = datetime.now(timezone.utc) - timedelta(days=30)

def parse(ts):
    dt = datetime.fromisoformat(ts)
    # Treat naive timestamps as UTC so comparisons don't raise.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

deploy_counts = defaultdict(int)
for d in deploys:
    if parse(d["timestamp"]) >= cutoff:
        deploy_counts[d["service"]] += 1

incident_counts = defaultdict(int)
for i in incidents:
    if parse(i["timestamp"]) >= cutoff and i["severity"] in ("Sev1", "Sev2"):
        incident_counts[i["service"]] += 1

for service in sorted(deploy_counts):
    d = deploy_counts[service]
    i = incident_counts.get(service, 0)
    rate = (i / d) if d else 0.0
    print(f"{service}: {d} deploys, {i} incidents, {rate:.2f} incidents/deploy")
```

Run it weekly. Absolute numbers matter less than the trend. If incidents per deploy is rising while deploys are flat, something is degrading. If it's falling, whatever changed is working — or the sample is too small. Either way, there is now a basis for a conversation that isn't about story points.

Note that this script measures incidents per deploy, not change failure rate. Change failure rate requires linking each incident to the deploy that caused it, which is a stricter join. Incidents per deploy is a cheaper proxy that still reveals trend direction; use it to decide whether the stricter join is worth building.

## How to prevent the problem from recurring

Make the portfolio part of the operating rhythm, not a one-off exercise:

- **Quarterly:** review the three layers with each team. Pick one metric per layer to improve next quarter; don't try to improve all three at once.
- **Monthly:** join incident data with deploy data and share the result where the team already looks (a chat channel, a dashboard).
- **Per initiative:** write the outcome statement before work starts and check it after launch. If it can't be checked, say so explicitly and decide whether to instrument it next time.
- **In one-on-ones:** ask "what's the most valuable thing you shipped this month, and how do you know it was valuable?" The answer often reveals more than any dashboard.

The anti-pattern is turning the portfolio into a scorecard with targets. Once "change failure rate must be below X%" becomes a performance goal, teams will under-report incidents or avoid risky deploys. Use the numbers to ask better questions, not to rank people.

## Failure modes and related symptoms

- **"Velocity is high but nothing reaches production."** A delivery-layer failure: work is "done" in the tracker but not deployed. Check the definition of done and whether deploys are blocked by manual steps.
- **"We deploy often but customers keep reporting the same bug."** A stability-layer failure masked by high deployment frequency. Look at change failure rate and whether hotfixes are counted as deploys.
- **"We can't agree on an outcome metric."** Usually a sign the initiative is too vague. Split it into smaller pieces, each with a checkable statement.
- **"The team feels surveilled."** This happens when metrics are collected without context and used in reviews without warning. Introduce the portfolio in a team meeting, explain what it is and isn't for, and let the team see the data before management does.
- **"The numbers look great but the team is burning out."** None of these layers measure sustainability. Pair them with a simple check on unplanned work and after-hours pages; a team can hit delivery and stability targets by absorbing load that eventually shows up as attrition.

## When none of these work: escalation path

If the three layers still don't produce a clear picture, the problem is likely organizational rather than measurement-related. Two common cases:

1. **The team is dependent on another team.** If most work is blocked on an external dependency, no metric on this team will show output because the bottleneck is elsewhere. Escalate with data: show cycle time broken down by wait state. The conversation is about the dependency, not the team.
2. **The work is genuinely exploratory.** Research spikes and platform migrations don't produce shippable features at a steady rate. Measure *learning* instead: hypotheses tested, decisions made, risks retired. State explicitly that this is a different mode and set expectations accordingly.

If neither applies, ask for a skip-level review of the team's work with a peer manager. An outside read of the same evidence sometimes surfaces a pattern that has been normalized.

## Frequently Asked Questions

**How do you measure engineering productivity without velocity?**
Use a portfolio: delivery (deployment frequency, cycle time), stability (change failure rate, MTTR), and outcome (a specific metric per initiative). No single number is sufficient; the portfolio is designed to be triangulated, not summed. The goal is better questions in reviews, not a definitive score.

**What are good alternatives to story points for output?**
Count deployed items and their outcomes, not estimated effort. If forecasting is needed, keep story points for the team's internal use but don't use them as a management output metric. Cycle time and throughput measure flow, not value; pair them with stability and outcome data.

**How do you evaluate a team when the work is exploratory?**
Measure learning artifacts: hypotheses written, experiments run, decisions documented, risks retired. Set expectations that exploratory work has a different cadence and that "no shippable feature" is not the same as "no output." Agree on what evidence of progress looks like before the work starts.

**What if leadership demands a single number?**
Offer a composite only if the components are shown alongside it. A single number invites gaming; a composite with visible parts invites conversation. If forced, use deployment frequency as the headline and keep change failure rate as the guardrail — never report one without the other.

## Your next 30 minutes

Open your issue tracker and export the last 30 days of completed items for one team. Add two columns: "deployed?" (yes/no) and "outcome checked?" (yes/no). Count the no's. That count tells you which layer of the portfolio is weakest, and it's the first thing to fix before the next review.
