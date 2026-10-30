# Remote salary: negotiate without sounding cheap

Most negotiation advice stops at "know your worth." That is not an instruction a developer can act on. What actually moves a remote rate conversation is a small set of documents that answer the client's real questions before they are asked: what does this person cost in my currency, what does the timezone gap cost me in hours, and what happens when something breaks at 2 AM their time.

This article describes how to build that packet, how to keep it consistent, and how to use it in writing so numbers do not get mangled across Slack threads.

## The problem: the client cannot map your number to their budget

A typical failure mode looks like this. A contractor quotes a monthly figure in USD. The client's finance owner needs to map that figure to a budget line, a headcount plan, and a currency. If the contractor cannot supply the mapping, the conversation stalls. It is rarely a rejection of the person; it is a rejection of an unanswerable question.

Three questions usually go unanswered:

1. Why does this rate correspond to this scope?
2. What does the client actually pay, in their currency and their time zone?
3. What is the plan when something breaks during the contractor's night?

A negotiation packet is three artifacts that answer those questions:

- A **cost sheet** that converts a target salary into the client's currency at two exchange rates (spot and a worse case).
- A **time sheet** that quantifies the timezone gap in hours and, separately, in money.
- A **risk sheet** that lists the scenarios that worry the client and what mitigates each one.

The rest of this article builds each one, then shows a worked negotiation example with all arithmetic shown.

## Prerequisites

- A public code host profile with at least a few repositories that build and run.
- A payment account that accepts USD, EUR, or GBP.
- A spreadsheet application, or just Python if you prefer to generate CSV.
- A quiet hour to collect real numbers.
- Willingness to write down your own cost of living and tax assumptions, because the packet only works if the inputs are honest.

The scripts below use Python 3.11 and only the standard library. Nothing needs to be installed.

## Step 1 — the cost sheet

Create a repository to hold the artifacts.

```bash
mkdir negotiation-kit && cd negotiation-kit
git init
git remote add origin git@github.com:YOURUSER/negotiation-kit.git
git pull origin main
```

Create three files: `cost-sheet.csv`, `time-sheet.csv`, `risk-sheet.csv`. Generate the cost sheet from a script so the numbers are reproducible and diffable.

```python
# cost_sheet.py
import csv
from decimal import Decimal, ROUND_HALF_UP

CONFIG = {
    "base_salary_usd": Decimal("7500"),
    "tax_rate": Decimal("0.25"),
    "exchange_rates": {
        "spot": Decimal("4.15"),
        "worst_case": Decimal("4.45"),
    },
}

rows = [["Metric", "Amount", "Currency", "Source"]]

rows.append(["Base salary (desired)", CONFIG["base_salary_usd"], "USD", "Negotiation target"])
rows.append(["Exchange rate (spot)", CONFIG["exchange_rates"]["spot"], "LOCAL/USD", "Set your own rate"])
rows.append(["Exchange rate (worst-case)", CONFIG["exchange_rates"]["worst_case"], "LOCAL/USD", "Set your own rate"])

base_spot = CONFIG["base_salary_usd"] * CONFIG["exchange_rates"]["spot"]
base_worst = CONFIG["base_salary_usd"] * CONFIG["exchange_rates"]["worst_case"]

rows.append(["Salary in local currency (spot)", base_spot.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "LOCAL", "Calculated"])
rows.append(["Salary in local currency (worst-case)", base_worst.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "LOCAL", "Calculated"])
rows.append(["Tax rate", f"{CONFIG['tax_rate'] * 100:.1f}", "%", "Your jurisdiction"])

take_home_spot = base_spot * (1 - CONFIG["tax_rate"])
take_home_worst = base_worst * (1 - CONFIG["tax_rate"])

rows.append(["Take-home (spot)", take_home_spot.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "LOCAL", "Calculated"])
rows.append(["Take-home (worst-case)", take_home_worst.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "LOCAL", "Calculated"])
rows.append(["Cost to client in USD", CONFIG["base_salary_usd"], "USD", "Same as base salary"])
rows.append([
    "Effective hourly rate (160h)",
    (CONFIG["base_salary_usd"] / 160).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
    "USD",
    "Calculated",
])

with open("cost-sheet.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
```

Run it:

```bash
python3.11 cost_sheet.py
```

Two design choices matter here. First, the exchange rate is a configuration value, not something you look up once and hardcode, because the rate moves. Second, the sheet records the source of every number. A client who cannot see where a figure came from will assume it is arbitrary.

The worst-case rate is not a prediction. It is a sensitivity check: if the rate moves against you by the amount you configured, what does your take-home become? That is the number that determines whether a contract is still worth signing in a bad month.

## Step 2 — the time sheet

The timezone gap has two components that are often conflated: overlap hours (when you and the client are both working) and delay hours (how long an async request waits because you are asleep). Overlap is the smaller and less important number.

```python
# time_sheet.py
from decimal import Decimal, ROUND_HALF_UP

HOURLY_RATE = Decimal("46.88")   # from cost-sheet.csv
OVERLAP_HOURS_PER_WEEK = Decimal("4")
DELAY_HOURS_PER_WEEK = Decimal("24")

monetised_delay = HOURLY_RATE * DELAY_HOURS_PER_WEEK
weekly_cost = HOURLY_RATE * (OVERLAP_HOURS_PER_WEEK + DELAY_HOURS_PER_WEEK)

print(f"Weekly overlap hours: {OVERLAP_HOURS_PER_WEEK}")
print(f"Weekly delay hours: {DELAY_HOURS_PER_WEEK}")
print(f"Monetised delay cost: ${monetised_delay:.2f}/week")
print(f"Total weekly cost: ${weekly_cost:.2f}/week")
```

Note the honesty constraint: the delay figure is an assumption, not a measurement. Say so in the sheet. A delay of 24 hours means a request filed Friday evening is answered Monday morning. If your client's work is genuinely async, that number is close to zero in practice, and the sheet should show that. If the client expects same-day turnaround, the number is real and should appear as a line item.

The value of this sheet is not the dollar figure. It is that it forces the conversation onto a concrete question: which tasks actually need same-day turnaround, and which can wait? Many "we need overlap" requests collapse once the client lists the tasks that require it.

## Step 3 — the risk sheet

```csv
Risk,Probability,Impact (hours),Mitigation cost (USD),Mitigation description
Emergency during your night,High,4,250,On-call rotation with a counterpart in the client's timezone
Third-party outage during your night,Medium,8,400,Automated rollback script and status page
Local machine or data loss,Low,24,300,Encrypted daily backups to object storage
Mid-sprint scope change,Medium,12,600,Prepaid buffer of hours
```

The probability column is a judgement, and should be labelled as one. What matters is that each row has a mitigation with a cost, so the client sees that the risk is being handled rather than absorbed silently.

Annualise the mitigation costs if the client wants a fixed monthly retainer, so the buffer is priced in rather than discovered later.

## Step 4 — the role brief

The packet needs a one-page summary the client can skim. Keep it to three questions: what problem you solve, how you solve it, and what availability and response behaviour the client can expect.

```markdown
# Role Brief: Full-Stack Engineer (Remote)

**Problem to solve:** Reduce critical-path latency for a billing service under peak load.

**Stack:**
- Node 20 LTS
- PostgreSQL with a connection pooler and read replicas
- Redis cluster for hot-path caching
- Terraform for infrastructure
- GitHub Actions for CI/CD

**Practices:**
- Incident response within a stated SLA
- Async standup in a written channel
- On-call rotation with a counterpart in the client's timezone
- Dashboards for latency, error rate, and saturation

**Availability:**
- Core overlap: state the hours explicitly
- Weekend standby: state whether it is included or billed
- Response SLA: separate P1 from P2 and P3

**Deliverables:**
- A latency target with a measurement method
- An uptime target with a measurement method
```

The deliverables section is where most briefs go wrong. "Improve latency" is not a deliverable. "P99 latency at or below X milliseconds, measured by this dashboard, over a rolling seven-day window" is. Write the measurement method into the brief, because it is the thing that will be argued about later.

## Step 5 — a worked pricing example

All figures below are illustrative. The point is the method, not the numbers.

Assume a desired base of 7,500 USD per month, an effective hourly rate of 46.88 USD (7,500 ÷ 160), a timezone gap of 4 overlap plus 24 delay hours per week, and a risk mitigation cost of 250 USD per month.

| Item | Monthly cost (USD) | Derivation |
|------|--------------------|------------|
| Base salary | 7,500 | Target |
| Timezone gap | 650 | 650 ≈ 46.88 × 4.33 weeks × 3.2 hours, rounded |
| Risk mitigation | 250 | Sum of mitigation rows |
| **Effective cost** | **8,400** | Rounded to the nearest 100 |

The timezone row deserves a note. The naive calculation is 46.88 × 28 hours per week, which is 1,312 USD per week and obviously wrong as a monthly figure. The reason it is wrong is that delay hours are not billed hours; they are a cost to the client in elapsed time, not in your labour. Pricing them at your full hourly rate double-counts. A more defensible approach is to price only the hours you are actually on call or actively working outside your normal window, and to present the delay figure separately as an operational cost the client can reduce by changing their own expectations.

That distinction is the single most common error in timezone pricing, and clients who have hired remote contractors before will spot it immediately.

Now the currency side. Suppose the local currency is quoted at 4.15 to the dollar at spot and 4.45 in the worse case. Then:

- Spot: 7,500 USD × 4.15 = 31,125 local units.
- Worst case: 7,500 USD × 4.45 = 33,375 local units.

At a 25% tax rate, the take-home is 75% of each:

- Spot take-home: 31,125 × 0.75 = 23,343.75 local units.
- Worst-case take-home: 33,375 × 0.75 = 25,031.25 local units.

Notice what this shows the client: your take-home in local currency is higher in the worst case, because the same dollar amount buys more local currency when the local currency weakens. That is the opposite of the risk for a client paying in dollars, and it is worth stating plainly, because it tells the client that currency movement in their favour is not a windfall you will renegotiate over.

## Step 6 — edge cases worth writing into the contract

**Currency risk.** If you are paid in USD but your costs are in local currency, a sharp move in the rate changes your real income. The mitigation is not a clause; it is a re-pricing cadence. State in the contract how often the rate is reviewed and what triggers a review.

**Payment rails.** Fees differ substantially between providers and between corridors. Before quoting a net figure, check the actual fee schedule for your corridor and compute the net. A percentage fee plus a fixed fee per withdrawal can remove a meaningful fraction of a monthly payment. Compare net take-home, never gross.

**Contract type.** A direct contractor agreement, a local entity, and an employer-of-record arrangement have different tax and compliance profiles. An EOR adds a markup, typically a percentage of the invoice, which the client will see. Decide which party absorbs that markup before you quote.

**Scope creep.** The client's brief will be vague. Convert it into a fixed-scope statement with measurable acceptance criteria before signing. A burn-down chart showing story points and sprints is a useful artifact, but the contract clause is what matters.

**Standby.** Define emergencies precisely. "Billing service down" is not precise; "payment failure rate above a stated threshold for more than a stated duration" is. Everything else goes into the next sprint.

## Step 7 — proving you can meet the SLA

Before sending a proposal that promises a latency or uptime target, have a dashboard that shows the target being met. The dashboard does not need to be elaborate. It needs three panels:

- Latency percentile over a rolling window.
- Uptime percentage over a rolling window.
- On-call response time.

A Grafana panel for P99 latency looks like this:

```json
{
  "title": "Billing P99 latency (ms)",
  "targets": [
    {
      "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{service=\"billing\"}[5m])) * 1000",
      "legendFormat": "P99"
    }
  ],
  "unit": "ms",
  "min": 0,
  "max": 200
}
```

Three checks are worth automating in the first week:

1. **Latency check.** Run 100 requests against the health endpoint and record the distribution, not just the mean. A mean of 80 ms with a P99 of 900 ms is a failing service.
2. **Uptime check.** A scheduled function that pings the endpoint on an interval and alerts on failure.
3. **On-call check.** A simulated incident that measures how long the response actually takes, including at an inconvenient hour.

Document the escalation path in the repository:

```markdown
# Escalation guide

- P1: service down or payment failures above the agreed threshold → incident channel, immediate page
- P2: latency above target or uptime below target → tracked issue, response within the agreed window
- P3: everything else → tracked issue, next sprint

Response SLA:
- P1: ≤ 15 minutes
- P2: ≤ 4 hours
- P3: ≤ 24 hours
```

## How to measure whether the packet worked

There is no benchmark table here because results depend entirely on the client, the market, and the role. What can be measured is the negotiation process itself. Instrument these:

- **Time to first substantive reply.** If the packet is clear, the client's next message should contain a question about scope or a counter-offer, not a request to clarify the numbers.
- **Number of clarification rounds.** Count them. More than two usually means the cost sheet is missing a mapping the client needs.
- **Conversion of the delay-hours line.** Did the client reduce their same-day expectations after seeing it? That is the sheet doing its job.
- **Renewal rate.** The packet is a long-term artifact; its value shows up at renewal, when the numbers can be updated rather than renegotiated from scratch.

Keep the CSVs in version control and diff them between negotiations. Over several contracts, the diffs show which assumptions you consistently get wrong, which is more useful than any single outcome.

## Common questions

**The client says the rate is well above local contractors. How should that be handled?**
Compare like with like. Local contractor rates and remote rates for the same role are different products: different overlap, different contract type, different tax treatment. Put the comparison in the cost sheet as an explicit row with the source of the local figure. If the client cannot supply a source, the comparison is not a comparison.

**What if there is no tax treaty between the client's country and mine?**
Withholding may apply at the source. Two common approaches: a gross-up clause, where the client pays the withholding so the net is unchanged, or a higher gross figure that nets to the target after withholding. Show the arithmetic. For example, to net 7,000 with a 30% withholding, the gross must be 7,000 ÷ 0.7 = 10,000. Put that calculation in the sheet rather than asserting the conclusion.

**Should an employer-of-record be used?**
An EOR handles payroll and compliance in exchange for a markup, often a percentage of the invoice. Whether that is worth it depends on whether the client is willing to absorb the markup and whether you want to run your own entity. Model both in the cost sheet so the client sees the difference rather than discovering it at contract time.

**What is needed to prove SLA capability?**
At minimum, a metrics source, a status page, and an alerting path. The specific tools matter less than that the status page is live and public before the proposal is sent. A broken link in a reliability proposal is a strong negative signal.

**How should early-career developers approach this?**
Negotiate scope before rate. A short paid trial with a clearly defined deliverable gives both sides evidence. The rate conversation is much easier after the client has seen the work than before.

## Where to go from here

The packet only works if the inputs are real. Start with the cost sheet, because every other number depends on the effective hourly rate.

Open `cost-sheet.csv`, set `base_salary_usd` to the number you actually want, and set the two exchange rates to today's spot rate and a rate roughly 5–10% worse. Run `python3.11 cost_sheet.py`, open the resulting CSV, and check two things: that the take-home figure matches what you would actually receive after tax and payment fees, and that the worst-case take-home is still a number you would accept. If the worst case is not acceptable, the base salary is too low, and you have found that out before the client did.
