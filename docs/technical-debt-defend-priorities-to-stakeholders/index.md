# Technical debt: defend priorities to stakeholders

Technical debt has a reputation for being boring, which is exactly why it rarely gets budgeted. The hard part is not identifying debt; it is making the prioritisation defensible to people who do not read stack traces.

A familiar scene: a CTO asks why the checkout service still runs on a deprecated runtime while the team spends a sprint on test coverage. The engineering manager has a gut answer — "it's risky" — but no number, no date, and no story that survives a budget meeting. The debt never gets fixed, or it gets fixed in a panic after an outage. The real problem is not identifying technical debt; it is making the prioritisation defensible to people who don't read stack traces.

This article provides a framework that turns debt into a language stakeholders already understand: cost, risk, and time. It starts with the symptom that tells you your current approach is failing, then walks through the causes, the confirmation steps, the fixes, and the prevention. The output is a one-page artefact that can be pasted into a planning doc.

## The symptom: "we'll fix it after the launch"

The exact phrase varies, but the pattern is consistent: debt items are logged in a backlog that nobody outside engineering reads, and every prioritisation conversation ends with "not now." The engineering manager leaves the room with a promise to "keep an eye on it," and six weeks later the same item appears in a post-incident review.

A more measurable symptom is a growing gap between the debt backlog and the delivery roadmap. If debt items have no owner, no estimated effort, and no stated consequence of delay, they are not priorities — they are wishes. Stakeholders are not ignoring the list; they are responding rationally to a list that gives them nothing to trade off against feature work.

The core mistake is treating technical debt as an engineering problem that needs engineering justification. It is a business problem that needs a business case. The framework below is designed to produce that case without requiring a finance degree.

## Cause 1: debt is described in implementation terms

The most common reason debt gets deprioritised is that the description is unreadable to the people who control the budget. "Refactor the auth module" means nothing to a head of sales. "Migrate from a deprecated runtime before vendor support ends" means something, because it has a date and a consequence.

### How to confirm

Open the debt backlog and count how many items mention a business outcome (revenue, risk, compliance, customer-facing latency) versus a technical action. If the ratio is heavily skewed toward technical actions, this is the primary cause.

### The fix: rewrite each item as a decision

Convert each debt item into a one-line statement with four fields: what breaks if nothing is done, when it breaks, what it costs to fix now, and what it costs to fix later. The last field is the one stakeholders remember. Use ranges, not point estimates, and label them as estimates.

A simple template, usable in a spreadsheet or issue tracker:

```markdown
| Debt item | Consequence if deferred | Trigger date | Fix now (est.) | Fix later (est.) |
|-----------|--------------------------|--------------|----------------|------------------|
| Auth service on unsupported runtime | No security patches; incident response time doubles | Vendor EOL 2026-09-30 | 3 engineer-weeks | 6 engineer-weeks + incident cost |
| No integration tests for payments | Every release risks silent breakage; rollback rate high | Immediate | 2 engineer-weeks | 4 engineer-weeks + customer credits |
```

The numbers in the table are illustrative. The point is the structure: a date, a consequence, and a widening cost gap. Stakeholders can argue with a number; they cannot argue with "it's messy."

## Cause 2: no shared scoring model, so every debate restarts from zero

When each debt item is argued on its own merits, prioritisation becomes a series of one-off negotiations. The engineering manager ends up defending the same principles repeatedly, and the loudest stakeholder wins.

### How to confirm

If two people on the team would rank the same five debt items differently, there is no shared model. A quick test: ask each engineer to score the top five items on impact and effort without discussion. Wide disagreement confirms the cause.

### The fix: a two-axis score with a forcing function

Use a simple impact-versus-effort score, but add a forcing function: any item that blocks a committed roadmap goal or has a hard external deadline gets automatic priority regardless of score. This prevents the model from burying urgent work under a pile of medium-impact items.

A workable scoring model:

- **Impact (1–5):** how much does this item affect revenue, customer experience, security, or delivery speed? Anchor 5 to a current roadmap goal, 1 to no measurable effect.
- **Effort (1–5):** engineer-weeks, with 1 = under a week, 5 = over a month.
- **Reversibility (1–3):** how hard is it to undo if the choice is wrong? Higher means more caution.

Priority score = (Impact × 2) + Reversibility − Effort. The weights are arbitrary but must be fixed and published. The value is not mathematical precision; it is that the same inputs produce the same ranking every time, which ends the re-litigation.

| Framework | Best for | Weakness |
|-----------|----------|----------|
| Impact/Effort matrix | Fast triage of many items | Ignores deadlines and dependencies |
| Cost of delay | Stakeholder conversations about timing | Requires revenue estimates that may not exist |
| Risk-adjusted backlog | Security and compliance debt | Hard to compare against feature work |
| This two-axis score + forcing function | Mixed debt portfolios with some hard deadlines | Still needs calibration against real outcomes |

This table is a genuine comparison of trade-offs; pick the one that matches the stakeholder culture. If stakeholders think in money, cost of delay will land better. If they think in risk, lead with the risk-adjusted view.

## Cause 3: the cost of delay is invisible

The single most persuasive argument for debt work is the cost of waiting. But most teams never calculate it, so the default assumption is zero. When the cost of delay is zero, deferral is always rational.

### How to confirm

Check whether any debt item in the backlog has an estimated cost per sprint of delay. If none do, this cause is active.

### The fix: a worked example of cost of delay

Cost of delay does not require a finance team. It requires stated assumptions. Here is a worked example using only arithmetic and labelled assumptions.

Assume a payments service has no automated regression tests. The team releases weekly. Assume, based on incident history, that one in ten releases causes a rollback that takes four engineer-hours to resolve. Assume a fully loaded engineer cost of $60 per hour (illustrative).

- Expected rollback cost per release = 0.1 × 4 hours × $60 = $24.
- Over a year (52 releases) = $24 × 52 = $1,248.

That number alone will not win a budget meeting. Add the second-order cost: each rollback delays a planned feature by a day, and the feature is tied to a contract milestone. If the milestone penalty is $5,000 per week of delay, and one in ten releases causes a one-day delay, the expected penalty per release is 0.1 × (1/5) × $5,000 = $100. Annualised, that is $5,200. Now the total cost of delay is roughly $6,448 per year.

Compare that to the estimated fix cost of two engineer-weeks ($4,800 at the same illustrative rate). The payback period is under a year, and the risk reduction is a bonus. That is a defensible case. The numbers are illustrative; the method is the point. Replace them with your own incident history and contract terms.

Two practical notes on gathering the inputs. First, incident history is usually richer than people assume: ticketing systems, on-call logs, and deployment records already contain rollback counts and time-to-resolve. Second, when a number is genuinely unknown, state the assumption explicitly and show how the conclusion changes if it is wrong. A case built on visible assumptions can be challenged and improved; a case built on authority cannot.

## Cause 4: debt work has no visible delivery

Stakeholders fund visible progress. Debt work often produces no demo, no release note, and no metric movement in the short term, so it looks like a cost centre. This is a perception problem with a concrete fix.

### How to confirm

If the last three sprint reviews contained no debt-related demo or metric, this cause is active.

### The fix: attach a leading indicator to every debt item

Every debt item should have a metric that moves when the work is done. For test coverage debt, the indicator is rollback rate or escaped defect count, not coverage percentage. For runtime upgrade debt, the indicator is the number of unsupported dependencies. For architecture debt, the indicator is p95 latency or deploy frequency.

Publish these indicators in the same dashboard stakeholders already check for feature progress. When a debt item ships, show the before-and-after on that metric. This converts invisible work into a visible trend line, which is the only language a roadmap review understands.

## Cause 5: no capacity reservation, so debt work is always the flex

Even with a good case, debt work loses because it has no reserved capacity. Feature work has a committed roadmap; debt work has good intentions. When the sprint gets tight, debt is what gets cut.

### How to confirm

Look at the last four sprints. If debt work was planned and then dropped in more than one, there is no capacity reservation.

### The fix: a fixed debt budget with a rollover rule

Reserve a fixed percentage of engineering capacity for debt, and make it non-negotiable in the same way a feature commitment is. A common starting point is 15–20% of each sprint, but the exact number matters less than the rule that unused debt capacity does not roll over into feature work. If the team finishes debt early, they pull the next debt item; they do not pull a feature.

This is the hardest part to defend because it looks like a tax on delivery. The counter-argument is the cost-of-delay arithmetic above: a 20% reservation now is cheaper than an unplanned 40% outage later. Present it as insurance, not overhead.

## Failure modes that undermine the framework

A framework is only as good as its weak points. These are the failure modes most likely to appear once the model is in use.

**Score gaming.** When a score gates funding, scores drift upward. Every item becomes a 5 on impact. Countermeasure: require a named roadmap goal or incident for any impact score of 4 or 5, and review scores in a group rather than individually.

**Estimate theatre.** Ranges get collapsed into single dates that are then treated as commitments. Countermeasure: keep ranges visible in the artefact and state the confidence level alongside them. A "3–6 engineer-weeks, low confidence" estimate is more honest than a false precision of "4 weeks."

**The forcing function swallows everything.** If half the backlog is labelled a roadmap blocker, the automatic priority rule stops discriminating. Countermeasure: cap the number of items that can hold blocker status at any time, and require the roadmap owner to confirm the link.

**Metrics that reward the wrong thing.** Coverage percentage is the classic example: it can be raised without reducing risk. Countermeasure: prefer outcome metrics — rollback rate, escaped defects, unsupported dependency count, change failure rate — over activity metrics.

**Reservation erodes silently.** The capacity rule holds for two quarters, then a crunch arrives and debt capacity is quietly borrowed. Countermeasure: report on the reservation the same way as feature commitments, including when it was broken and what was traded for it.

## Prevention: make debt a standing agenda item, not a project

The frameworks above fix individual prioritisation decisions. Prevention means changing the default so debt does not accumulate silently.

Three practices that work:

1. **Debt review in every sprint planning.** Spend ten minutes scoring new debt items with the shared model. New items enter the backlog with a score, not just a description.
2. **A debt budget line in the quarterly plan.** Treat the reservation as a budget line item that stakeholders approve once, not a per-sprint negotiation.
3. **Post-incident debt capture.** Every incident review produces at least one debt item with a cost-of-delay estimate. This ties debt work directly to real failures, which is the most persuasive evidence available.

The goal is not zero debt; it is debt that is visible, scored, and funded. Teams that do this stop having the same argument every quarter.

## Frequently Asked Questions

**How do I prioritise technical debt when stakeholders only care about features?**

Translate debt into feature risk. Every debt item that threatens a committed feature gets a cost-of-delay estimate tied to that feature's deadline. Stakeholders care about features; show them how debt puts those features at risk. The two-axis score with a forcing function for roadmap blockers is designed for exactly this situation.

**What is a good technical debt prioritisation framework for a small team?**

Start with the impact/effort matrix plus a hard rule that any item with an external deadline or a roadmap blocker goes first. Small teams rarely have the data for cost-of-delay modelling, so keep the model simple and publish the weights. The value is consistency, not precision.

**How much engineering capacity should be reserved for technical debt?**

There is no universal number, but 15–20% per sprint is a common starting range. The more important rule is that the reservation is fixed and does not roll over into feature work. Adjust based on incident rate and the age of the oldest debt items.

**Why does technical debt keep getting deprioritised even when everyone agrees it matters?**

Because agreement without a cost of delay is not a priority. Debt loses to features by default when it has no date, no consequence, and no metric. The fix is to attach all three to every item and reserve capacity so debt work is not the flex when the sprint gets tight.

## Your next 30 minutes

Open the debt backlog and pick the three items that have been deferred the longest. For each, write one line with a consequence, a trigger date (or "immediate"), and an effort estimate in engineer-weeks. If a trigger date cannot be filled in, that is the first conversation to have with the system owner. Paste the three lines into the next planning doc and use them as the opening slide. That single artefact is the difference between a wish list and a defensible priority.
