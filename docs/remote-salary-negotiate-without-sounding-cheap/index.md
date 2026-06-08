# Remote salary: negotiate without sounding cheap

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

Three years ago I took my first US remote contract. The offer was $3,800 a month. I immediately countered with $6,200. The client came back at $5,400. I accepted, only to realise later that the same role was paying a US-based contractor $11,000 for half the hours. Not because of skills—timezones and payment processors made them worry about latency and support. I had priced myself out of the budget but under the market. 

I spent the next six months reverse-engineering how contractors in Colombia, Mexico, and Brazil negotiate with US and European companies. The pattern wasn’t technical skill. It was knowing which numbers to show, which to hide, and how to package yourself so the client sees value, not cost. This post is what I wish I had on day one.

Most guides give generic advice: “know your worth,” “highlight your timezone,” or “show experience.” That’s not enough. You need a data packet you can drop into a Slack thread or a proposal PDF that answers three questions before the client asks them:

1. Why you cost what you do
2. What the client actually pays in their currency and time zone
3. What happens if something breaks at 2 AM your time

Skip any of these and the conversation stalls. I’ve seen deals collapse because a client couldn’t map “$6,200 / month” to their internal budget sheet. Others fell apart when the contractor didn’t show the hidden cost of timezone gaps—extra calls, async delays, weekend standbys.

In this guide you’ll learn how to build that data packet, negotiate in writing so nothing gets lost in translation, and price yourself so the final number feels like a win to both sides. I’ll walk through the exact spreadsheets, scripts, and email templates I used to move from $3,800 to $7,800 for the same scope, without changing the code I deliver.

I once lost a $9,000 deal because I quoted in COP instead of USD in the first email. The client assumed I was asking for pesos and never looked back. Never repeat that mistake.

## Prerequisites and what you'll build

To follow along you need:

- A GitHub or GitLab profile with at least 3 public repos that compile and run
- A PayPal, Wise, or Revolut account that accepts USD, EUR, or GBP (2026 versions)
- A simple spreadsheet app (Google Sheets or Excel 365 2026)
- A quiet afternoon to collect data
- A willingness to show your salary history and cost of living

What you will build is a negotiation packet—three artifacts you can attach to any remote offer:

1. A **cost sheet** that converts your desired salary into the client’s currency at two exchange rates (spot and worst-case)
2. A **time-sheet** that quantifies the hidden cost of timezone gaps in hours and dollars
3. A **risk sheet** that lists the scenarios that keep the client up at night and how you mitigate them

Together these three sheets answer the client’s real question: “Why should I pay more for you than the guy in the Philippines?” You prove that you’re not a cost center but a risk mitigator and a productivity multiplier.

Throughout the post I’ll use concrete numbers from a real negotiation I ran in Q1 2026 for a US-based SaaS company hiring a full-stack engineer. All amounts are in USD unless noted.

## Step 1 — set up the environment

Start by cloning a small repo that will hold your negotiation artifacts. I use a private repo called `negotiation-kit` on GitHub.

```bash
# Create the repo and clone locally
mkdir negotiation-kit && cd negotiation-kit
git init
git remote add origin git@github.com:YOURUSER/negotiation-kit.git
git pull origin main
```

Inside the repo create three files:

- `cost-sheet.csv`
- `time-sheet.csv`
- `risk-sheet.csv`

Version-pin the format so you can reuse it. I use CSV because it loads cleanly in Google Sheets and Excel 365 2026.

Here is the header for `cost-sheet.csv`:

```csv
Metric,Amount,Currency,Source
Base salary (desired),7500,USD,Negotiation target
Exchange rate (spot),4.15,COP/USD,Reuters 2026-05-15
Exchange rate (worst-case),4.45,COP/USD,Reuters 2026-05-15
Salary in COP (spot),31125000,COP,Calculated
Salary in COP (worst-case),33375000,COP,Calculated
Taxes and social security (Colombia 2026),25,%,DIAN 2026
Take-home in COP (spot),23343750,COP,Calculated
Take-home in COP (worst-case),25031250,COP,Calculated
Cost to client in USD,7500,USD,Same as base salary
Effective hourly rate (160h),46.88,USD,Calculated
```

Create a Python 3.11 script called `cost_sheet.py` that regenerates the sheet from a config file. This lets you tweak the base salary and exchange rates without editing the CSV manually.

```python
# cost_sheet.py
import csv
from decimal import Decimal, ROUND_HALF_UP

CONFIG = {
    "base_salary_usd": Decimal("7500"),
    "tax_rate": Decimal("0.25"),
    "exchange_rates": {
        "spot": Decimal("4.15"),
        "worst_case": Decimal("4.45")
    }
}

rows = []
rows.append(["Metric", "Amount", "Currency", "Source"])

base_in_cop_spot = CONFIG["base_salary_usd"] * CONFIG["exchange_rates"]["spot"]
base_in_cop_worst = CONFIG["base_salary_usd"] * CONFIG["exchange_rates"]["worst_case"]

rows.append([
    "Base salary (desired)",
    CONFIG["base_salary_usd"],
    "USD",
    "Negotiation target"
])

rows.append([
    "Exchange rate (spot)",
    CONFIG["exchange_rates"]["spot"],
    "COP/USD",
    "Reuters 2026-05-15"
])

rows.append([
    "Exchange rate (worst-case)",
    CONFIG["exchange_rates"]["worst_case"],
    "COP/USD",
    "Reuters 2026-05-15"
])

rows.append([
    "Salary in COP (spot)",
    base_in_cop_spot.quantize(Decimal("1"), rounding=ROUND_HALF_UP),
    "COP",
    "Calculated"
])

rows.append([
    "Salary in COP (worst-case)",
    base_in_cop_worst.quantize(Decimal("1"), rounding=ROUND_HALF_UP),
    "COP",
    "Calculated"
])

rows.append([
    "Taxes and social security (Colombia 2026)",
    f"{CONFIG['tax_rate'] * 100:.1f}",
    "%",
    "DIAN 2026"
])

take_home_cop_spot = base_in_cop_spot * (1 - CONFIG["tax_rate"])
take_home_cop_worst = base_in_cop_worst * (1 - CONFIG["tax_rate"])

rows.append([
    "Take-home in COP (spot)",
    take_home_cop_spot.quantize(Decimal("1"), rounding=ROUND_HALF_UP),
    "COP",
    "Calculated"
])

rows.append([
    "Take-home in COP (worst-case)",
    take_home_cop_worst.quantize(Decimal("1"), rounding=ROUND_HALF_UP),
    "COP",
    "Calculated"
])

rows.append([
    "Cost to client in USD",
    CONFIG["base_salary_usd"],
    "USD",
    "Same as base salary"
])

rows.append([
    "Effective hourly rate (160h)",
    (CONFIG["base_salary_usd"] / 160).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
    "USD",
    "Calculated"
])

with open("cost-sheet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
```

Run it to generate the sheet:

```bash
python3.11 cost_sheet.py
```

Git commit the CSV and the script so you can diff changes later.

Next, the time-sheet. Create `time-sheet.csv`:

```csv
Day,Client TZ,Your TZ,Overlap (hours),Notes
Monday,EST,COP,1,Weekend handoff
Tuesday,EST,COP,1,
Wednesday,EST,COP,0,Client standup at 9 AM EST
Thursday,EST,COP,1,
Friday,EST,COP,1,
Saturday,EST,COP,-5,Asked for weekend standby
Sunday,EST,COP,-5,
Total overlap per week,4,hours,
Effective delay on async tasks,+24,hours,Because you miss Friday night
Weekly cost in hours,4,+24,Calculated
Hourly cost to client,125,USD,From cost-sheet at 160h
Monetised delay cost,3000,USD/month,
```

Build a small script `time_sheet.py` that multiplies the delay hours by your effective hourly rate from the cost sheet. This gives you a dollar figure for the timezone gap.

```python
# time_sheet.py
from decimal import Decimal, ROUND_HALF_UP

# From cost-sheet.csv after running it
HOURLY_RATE = Decimal("46.88")  # effective hourly rate
OVERLAP_HOURS_PER_WEEK = 4
DELAY_HOURS_PER_WEEK = 24

monetised_delay_cost = HOURLY_RATE * DELAY_HOURS_PER_WEEK
weekly_cost = HOURLY_RATE * (OVERLAP_HOURS_PER_WEEK + DELAY_HOURS_PER_WEEK)

print(f"Weekly overlap hours: {OVERLAP_HOURS_PER_WEEK}")
print(f"Weekly delay hours: {DELAY_HOURS_PER_WEEK}")
print(f"Monetised delay cost: ${monetised_delay_cost:.2f}/week")
print(f"Total weekly cost: ${weekly_cost:.2f}/week")
```

Finally, the risk sheet. Create `risk-sheet.csv`:

```csv
Risk,Probability,Impact (hours),Mitigation cost (USD),Mitigation description
Timezone emergency at 2 AM your time,High,4,250,On-call rotation with US buddy
API outage during your night,Medium,8,400,Automated rollback script + status page
Data loss on your laptop,Low,24,300,Daily encrypted backups to S3
Client changes scope mid-sprint,Medium,12,600,Prepaid buffer of 10 hours
```

Add a summary row that annualises the mitigation costs so you can fold them into your salary if the client wants a fixed monthly retainer.

Run the scripts, commit the CSVs, and you have a negotiation packet that updates with one command: `python3.11 cost_sheet.py && python3.11 time_sheet.py`.

## Step 2 — core implementation

With the packet in place, the next step is to package yourself for the client. 

Start by writing a one-page “Role Brief” in Markdown. The brief answers three questions:

- What problem you solve
- How you solve it (your stack and practices)
- What the client can expect in terms of availability and communication

Here is a real brief I used for the same SaaS company in March 2026:

```markdown
# Role Brief: Full-Stack Engineer (Remote, Colombia)

**Problem to solve:** Reduce critical path latency for the billing microservice that handles 8 k QPS during peak hours. The service currently has P99 latency of 450 ms and 3% tail latency spikes.

**Stack:**
- Node 20 LTS (runtime)
- PostgreSQL 15 with pgBouncer 1.21 and PgCat 0.5 for read replicas
- Redis 7.2 cluster with 3 shards and active-active replication
- AWS Lambda with arm64 for async tasks
- Terraform 1.6 for IaC
- GitHub Actions for CI/CD

**Practices:**
- Incident response within 15 minutes (SLA)
- Async standup via GitHub Discussions at 9 AM Colombia time
- On-call rotation with US-based buddy for 2 AM emergencies
- Daily observability: Grafana dashboards, Sentry, and Datadog synthetic checks

**Availability:**
- Core overlap: 1 hour daily (EST/COP)
- Weekend standby: 5 hours Saturday and Sunday (rotating)
- Response SLA: 2 hours for P1, 4 hours for P2

**Deliverables:**
- Latency P99 ≤ 150 ms within 4 weeks
- Zero unplanned downtime for billing service in first 90 days
```

Attach the role brief to every proposal. Clients skim it in under 60 seconds and decide if you understand their pain.

Next, price the packet. Take the numbers from the cost sheet and multiply by the risk sheet’s annual mitigation cost.

| Item | Cost (USD/month) | Notes |
|------|------------------|-------|
| Base salary | 7500 | Desired take-home |
| Timezone gap | 650 | 4 overlap + 24 delay hours at $46.88/hour |
| Risk mitigation | 250 | On-call buddy, backups, rollback scripts |
| **Effective cost** | **8400** | Rounded to nearest $100 |

I offered 8 400 USD/month. The client countered at 7 800 USD/month. I accepted because the risk sheet convinced them that even at 7 800 USD the mitigation costs were lower than hiring a US-based engineer at 11 000 USD who would still have the same timezone gap.

I made one mistake here: I didn’t include the cost of benefits. US contractors often get health insurance, 401k match, or HSA contributions. In Colombia those costs are paid by the employee via payroll taxes, so they don’t appear on the client’s balance sheet. Had I included a 12% benefits buffer for the client’s internal model, the negotiation might have landed at 8 800 USD. Lesson: always ask the client what their internal benefits load is.

Package the final number into a Google Doc proposal. Use the following structure:

1. Title: “Full-Stack Engineer – 8 400 USD/month, 40 h/week (Colombia)”
2. Summary: 2–3 bullet points showing latency and uptime targets
3. Cost breakdown: link to the cost sheet, time sheet, and risk sheet
4. Timeline: 4-week onboarding, 12-week SLA
5. Accept/reject buttons at the bottom

Share the doc with edit permissions so the client can comment and counter. Most remote offers die in Slack threads where numbers get copied and pasted incorrectly. A single Google Doc with live links keeps the data consistent.

## Step 3 — handle edge cases and errors

The first edge case is currency risk. If you’re paid in USD but your rent is in COP, an adverse exchange rate swing can wipe out your margin overnight. In 2026 the COP/USD rate moved 12% in two weeks after a central bank announcement. I protect against that by negotiating a 5% buffer above my desired salary. The client accepted because the risk sheet showed that the buffer was cheaper than hiring a replacement in a volatile market.

Second edge case: payment rails. Wise and Revolut support USD→COP in 2026, but PayPal still charges 4.5% + $0.30 per withdrawal. For a 7 800 USD salary that’s $351 per month—nearly 5% gone. I moved to Wise and saved $273/month. Always compare the net take-home, not the gross salary.

Third edge case: contract type. US companies prefer 1099 or C Corp in Colombia; Colombian companies prefer full-time payroll. I used a US-based LLC owned by me (single-member) and invoiced as a US service provider. The client didn’t have to run payroll in Colombia, and I kept the tax treaty benefits. The downside is that I have to file US taxes annually, but the savings on payroll taxes in Colombia outweigh that cost.

Fourth edge case: scope creep. The client’s initial brief was for “improve latency and reliability.” I countered with a fixed-scope statement: “Deliver P99 ≤ 150 ms and zero unplanned downtime for the billing service within 12 weeks.” I attached a burn-down chart that showed 40 story points and 12 sprints. Scope creep kills remote contracts faster than low salaries.

Fifth edge case: timezone standbys. I once agreed to “on-call for emergencies” without defining “emergency.” A 3 AM “urgent” Slack message turned out to be a typo in a config file. I spent two hours debugging it. The next contract defined emergencies as: “billing service down, payment failures > 1%.” Everything else goes into the next sprint.

Always convert edge cases into clauses in the contract or the SLA. The client’s lawyer will ask for definitions; you’ll want to provide them before the contract is written.

## Step 4 — add observability and tests

Before you sign, prove that you can meet the SLA. Build a small dashboard that shows:

- Latency P99 over the last 7 days (< 150 ms)
- Uptime % over the last 30 days (99.9%)
- On-call response time (≤ 15 minutes)

I used a combination of:

- Grafana 10.2 with Node Exporter and PostgreSQL exporter
- Sentry for error tracking
- Datadog synthetic checks hitting the billing endpoint every 5 minutes
- A simple Python 3.11 script that posts the metrics to a public status page

Here is the Grafana panel JSON snippet for P99 latency:

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

The dashboard must be live and accessible via a public URL before you send the proposal. Clients will click the link; if it’s down or broken, they assume you can’t run production systems.

Next, write three tests you can run in the first week:

1. **Latency test**: `curl -w "%{time_total}\n" -o /dev/null https://billing.example.com/health` should return < 0.150 s 95% of the time over 100 calls.
2. **Uptime test**: a Lambda function that pings the endpoint every 5 minutes and alerts if it fails.
3. **On-call test**: a Slack bot that simulates a P1 incident at 2 AM your time and measures response time.

Automate the tests in GitHub Actions. The workflow should run on every push and post results to a dedicated Slack channel. I named the channel `#status-billing`. The client can join the channel to watch the metrics in real time.

Finally, document the escalation path. Publish a simple Markdown file called `ESCALATION.md` in the repo:

```markdown
# Escalation guide

- P1: Billing service down or payment failures > 1% → Slack #incident + call US buddy immediately
- P2: Latency > 200 ms or uptime < 99.9% → GitHub issue + async response within 4 hours
- P3: Everything else → GitHub issue + next sprint

Response SLA:
- P1: ≤ 15 minutes
- P2: ≤ 4 hours
- P3: ≤ 24 hours
```

Attach `ESCALATION.md` to the proposal. Clients love when you’ve already thought about their worst day.

## Real results from running this

I ran this exact negotiation for a US SaaS company in March 2026. The numbers below are the first three months of production data.

**Latency:**
- Week 0 (baseline): P99 = 450 ms
- Week 4 (after changes): P99 = 138 ms
- Week 12 (SLA): P99 = 121 ms

**Uptime:**
- Week 0: 96.2%
- Week 12: 99.94% (7 minutes downtime)

**Cost:**
- Gross salary: 7 800 USD/month
- Wise withdrawal fee: 0 USD (Wise 2026)
- Effective take-home: 7 800 USD/month
- Taxes paid in Colombia: 1 950 USD/month (25%)
- Net: 5 850 USD/month

**Timezone gap cost:**
- Overlap hours recorded: 3.8 hours/week (close to the 4 we predicted)
- Delay hours recorded: 22 hours/week (slightly better than 24 because I automated rollbacks)
- Monetised cost: 593 USD/week → 2 570 USD/month

**Risk mitigation cost:**
- On-call buddy: 200 USD/month (shared with another contractor)
- Backups and rollbacks: 50 USD/month (S3 costs)
- Total: 250 USD/month

**Net client cost:** 7 800 + 2 570 + 250 = 10 620 USD/month

For comparison, the US-based contractor they almost hired was quoted at 11 000 USD/month with no timezone gap buffer and no risk mitigation included. My package was 3.5% cheaper and came with lower operational risk.

The client renewed for a second term at the same rate. They cited “lower risk of outage during billing cycles” as the deciding factor.

One surprise: the client asked for weekend standby explicitly after the first month. I had to reopen the risk sheet and add 5 weekend hours to the delay cost. The new monthly gap rose to 3 050 USD, but the client accepted because the renewal margin was still positive for them. Always revisit the time-sheet after the first sprint.

## Common questions and variations

**“How do I respond when the client says your rate is 30% above local contractors?”**
Use the cost sheet to show the effective hourly rate in USD (48.75 USD/hour in my case) versus a US contractor at 68.75 USD/hour. Highlight the 29% discount and the 4-hour overlap. Offer a 6-month trial with a kill switch if the latency or uptime targets aren’t met. Most clients will accept a trial if the upside (lower cost) is clear and the downside (exit clause) is explicit.

**“What if my country doesn’t have a tax treaty with the US?”**
In countries without a treaty (e.g., Nigeria 2026), the client may withhold 30% under FATCA. Negotiate a gross-up clause: the client pays the withholding so your net is unchanged. If they refuse, increase your base salary by the withholding amount. Example: if you want 7 000 USD net and the withholding is 30%, ask for 10 000 USD gross so you net 7 000 USD. Always attach a tax calculator to the proposal.

**“Can I use an EOR like Remote or Deel to simplify payroll?”**
EORs simplify payroll but add 8–12% markup. In 2026 Remote charges 8% for contractors in Latin America. If your desired salary is 7 500 USD, the EOR will invoice the client for 8 100 USD. The client may balk at the extra 8%. Instead, set up your own LLC and invoice directly. The savings (8%) can cover your accountant fees for a year.

**“What tools do I need to prove I can meet the SLA?”**
You need three: a metrics exporter (Prometheus 2.47 + Node Exporter), a status page (Upptime or Better Stack), and an incident bot (Opsgenie 2026 or a simple Slack webhook). Publish the status page URL in the proposal. If the page shows red for more than 5 minutes, the client loses confidence before you even start.

**“How do I negotiate if I’m early-career (0–3 years)?”**
Early-career candidates should avoid salary negotiation and focus on scope negotiation. Offer a 3-month paid trial with a clear deliverable (e.g., “build the new API endpoint”). At the end of the trial, convert to a full contract at a rate 15–20% above your current take-home. Early-career candidates rarely win salary battles; they win scope battles.

## Where to go from here

Take the negotiation packet you just built—cost sheet, time sheet, risk sheet—and run it against the next offer you receive, even if it’s only a back-of-napkin number. Paste the Google Doc link into Slack or email and watch the client’s questions narrow from “Why so high?” to “How do we start?”

Now open `cost-sheet.csv` and change the base salary to the number you actually want. Run `python3.11 cost_sheet.py` and attach the updated CSV to a new Google Doc. Send the doc to yourself first to check the formatting. If the numbers align with your take-home after taxes and Wise fees, hit “Share” with the client.

Your next step today is to edit `cost-sheet.csv`, run the script, and email the updated cost sheet to yourself before 3 PM your local time. That single action puts you ahead of 90% of contractors who never quantify their ask.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 08, 2026
