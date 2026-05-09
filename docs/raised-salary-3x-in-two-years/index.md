# Raised salary 3x in two years

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

In March 2022 I was billing $2,800 per month for a single client in Nairobi. By June 2024 my highest invoice read $8,400 to a US-based SaaS. Between those dates I didn’t win 10 new clients; I raised rates with the same three accounts and added two more at $5,200 each. The jump wasn’t magic—it was a deliberate sequence of role changes, pricing experiments, and infrastructure bets that most freelancers skip because the steps feel too small or too scary. Below is the exact playbook I followed, the tools I used, the numbers that moved, and the one mistake I made that cost me $1,100 in back pay.

## The situation (what we were trying to solve)

In early 2022 my revenue was flat at $2,800 a month for four years. That number was already 30 % above Kenya’s median salary for software engineers, so I wasn’t desperate. What bothered me was the ceiling: every time I tried to raise rates, the client would sigh and say, “We love your work, but the budget is fixed.” I reacted by working longer hours, cutting margins on AWS bills, and even building a small product on the side that earned $400 a month—useful pocket money but not the lever I needed.

The real problem wasn’t money; it was leverage. I was a one-person shop with one skill set (backend APIs) sold to one geography (East Africa). Switching clients risked months of pipeline building, and switching stack risked missing deadlines. I needed to change the unit of value, not just the price tag.

I set three targets for the next 12 months: raise the average invoice to $5,000, reduce client concentration below 40 %, and cut the time I spent on support to under 15 % of my hours.

**Summary:** I started with a ceiling on both price and risk. The goal was not to work harder, but to move from selling hours to selling outcomes.

## What we tried first and why it didn’t work

My first attempt was a classic freelancer trap: niche down. I picked “Django + PostgreSQL performance audits” and advertised on Twitter. Within two weeks I had 23 inquiries, all asking for the same 15-minute consult priced at $120. I did the math: 23 × $120 = $2,760 revenue but 23 × 0.25 hours = 5.75 billable hours—less than a single $2,800 client. Worse, the niche attracted tire-kickers; half the calls ended with “Can you also look at my React app?”

Next I tried retainers: $3,500 for 20 hours a month, no rollover. Two clients signed on, but both micromanaged every hour. After three months I had invoiced $7,000 but spent 45 hours of unpaid support—$156 per hour to me, negative margin.

The third try was product-led services. I built a tiny CLI that auto-optimizes Django migrations and sold “$999 per repo” audits. The tool worked, but the sales cycle took 6 weeks and the support queue grew to 12 open tickets. I realized I had just moved from one bottleneck (time) to another (context switching).

**Summary:** Each experiment increased complexity without increasing leverage. I was still trading hours for dollars, just in smaller chunks.

## The approach that worked

In June 2022 I decided to stop selling hours and start selling outcomes. The pivot had three legs:

1. Role shift: from individual contributor to fractional CTO. I marketed myself as “design and implement the data layer for your next feature” instead of “I’ll code your API.”
2. Geography shift: target US-based SaaS companies with Series A funding, where budgets are large and decision cycles are short.
3. Stack shift: move from monolithic Django on AWS EC2 to a microservice stack on Fly.io with Neon for serverless Postgres. This let me promise 99.9 % uptime and 50 ms P99 latency, two metrics US engineering managers care about.

I picked three anchor customers: a health-tech in Austin, a fintech in Denver, and a mar-tech in NYC. Each had raised $8–12 M, so they could afford $5k–8k a month for a part-time architect. I pitched a 3-month engagement with a fixed scope and a shared Slack channel—no hourly billing, no surprise invoices.

**Summary:** I stopped trying to be cheaper and started being more valuable. The fractional CTO framing gave me permission to price like a team, not an individual.

## Implementation details

Step 1: Website and messaging
I rebuilt my site in Next.js with the headline “Fractional CTO | Data layer design for Series A SaaS.” I added a single pricing page with three tiers: $4,500, $6,200, and $8,000 a month. Each tier capped hours at 40, 50, and 60 respectively. The copy avoided “I will code” and focused on “I will guarantee your next feature ships on time.”

Step 2: Tech stack demo
I recorded a 90-second Loom showing a failing Django endpoint, then I opened VS Code and rewrote it in FastAPI with async endpoints against Neon Postgres. I measured latency with `vegeta` and showed 48 ms P99 vs 210 ms before. I uploaded the repo to GitHub and tagged every commit with a timestamp. The demo cost me $12 in Neon credits but convinced three prospects in one week.

Step 3: Contracts and SLA
I used the Togai template for fractional CTOs, adding two clauses: (1) if uptime drops below 99.9 % in any 30-day window, I issue a 10 % credit; (2) if the feature misses the promised launch date by more than 7 days, I extend the engagement by one week at no charge. I signed NDAs with a DocuSign envelope and received countersigned offers within 48 hours.

Step 4: Tooling for remote work
- **Task tracking**: Linear with a custom template called “Fractional CTO” that auto-creates epics for each feature and subtasks for every PR.
- **Monitoring**: Grafana Cloud for Prometheus metrics, alerting on latency > 50 ms or error rate > 0.3 %.
- **Shared docs**: Notion database with runbooks for every service, so I can hand off to a junior engineer if a client hires one.

I kept my Nairobi office as my primary timezone (UTC+3) but scheduled overlap windows with US clients at 2 pm–5 pm EST, which is 9 pm–12 am in Nairobi. I used a YubiKey for SSH and a hardware wallet for crypto payments to keep latency low and security high.

**Summary:** The stack wasn’t about the shiniest tech; it was about predictable outcomes and seamless handoffs.

## Results — the numbers before and after

Baseline (March 2022):
- Monthly revenue: $2,800
- Client count: 1
- Support hours: 8 h / week
- Latency P99 for primary API: 210 ms
- AWS bill: $189

After 24 months (June 2024):
- Monthly revenue: $8,400 (highest invoice), average $6,100 across five clients
- Client concentration: 32 % (well below the 40 % target)
- Support hours: 3 h / week (under the 15 % target)
- Latency P99: 48 ms (a 77 % reduction)
- AWS bill: $42 (I moved to Fly.io and Neon, cutting EC2 spend by 78 %)

I measured these numbers using a simple Python script that queries Stripe for invoices, Linear for cycle time, and Grafana for latency. The script runs every Sunday at 8 am Nairobi time and emails me a CSV. In the first six months the script saved me 2.3 hours a week chasing spreadsheets.

One surprise: the health-tech client in Austin paid early every month, but the fintech in Denver was late twice. I added an ACH debit mandate in month 6, cutting late payments from 25 % to 0 %. The debit cost me $0.50 per transaction, but the time saved was worth 10×.

**Summary:** In two years I tripled revenue, cut support time by 62 %, and slashed latency by 77 %. The stack change delivered more than the role change.

## What we’d do differently

1. We would have started with one anchor client instead of three. In month 2 the Denver client needed an emergency migration from RDS to Neon, and I spent 18 hours firefighting. If I had focused on Austin alone, I could have finished the migration in week 1 and avoided the fire.
2. We would have priced the first engagement at $5,500 instead of $6,200. The Denver client haggled down to $5,200, and I left $1,100 on the table. In hindsight, a lower first price builds trust and gives room to raise later.
3. We would have automated the Grafana alert setup. I spent 5 hours manually wiring every service to Prometheus. A Terraform module would have saved those hours and prevented the outage in month 4 when I forgot to alert on memory.

**Summary:** Focus beats ambition, and automation beats manual work.

## The broader lesson

The move from local salary to global rate wasn’t about becoming more expensive; it was about becoming more replaceable. When you design systems that can be handed off, documented, and monitored, you stop being a bottleneck and start being a multiplier. The pricing power comes from the guarantee, not the hours. This principle holds whether you’re a freelancer, a startup co-founder, or an engineering manager: if your work requires you personally to keep running, you cap your own ceiling. If your work can run without you, the market rewards you for the outcome.

**Summary:** Leverage is the inverse of dependency. The more your clients can run without you, the more they will pay for you to set it up.

## How to apply this to your situation

1. Pick one metric your ideal client cares about (latency, uptime, MTTR, NPS). Measure it today with a free tool (Pingdom, UptimeRobot, Sentry). Blind spots kill deals.
2. Reframe your offering from “I will code” to “I will guarantee X by date Y.” Use a fixed-scope contract with an SLA clause. Show the before-and-after numbers in your pitch deck—clients pay for proof, not promises.
3. Automate the boring parts first. If you’re still manually deploying or monitoring, stop and write Terraform or Ansible. The time saved compounds every month.
4. Start with one anchor client in a geography that can afford your rate. Use their success story to sign the next two.


| Your current role | New framing | Typical rate jump | Risk level |
|-------------------|-------------|-------------------|------------|
| Freelance backend | Fractional CTO | 2x–4x | Medium |
| Staff engineer at local startup | Tech lead for remote team | 1.5x–2.5x | Low |
| Solo founder of micro-SaaS | Fractional product lead | 3x–5x | High |

**Summary:** Pick one role shift that matches your current stack, then price the outcome, not the hour.

## Resources that helped

- **Contract template**: Togai’s fractional CTO agreement (MIT license, GitHub). I forked it and added SLA clauses.
- **Monitoring stack**: Grafana Cloud + Prometheus + Alertmanager. The free tier covers up to 500 metrics.
- **Serverless Postgres**: Neon’s free tier gives 3 projects, 0.5 GB storage, and 10 million row reads per month—enough to test before paying.
- **Pricing playbook**: Brennan Dunn’s “Double Your Freelancing Rate” course. I paid $299 and it saved me $1,100 in back-and-forth negotiations.
- **Demo tooling**: Loom for async walkthroughs, Vegeta for load testing, and a $10 Raspberry Pi 4 to run the demo locally for clients in low-bandwidth regions.

**Summary:** The tools weren’t the star; they removed friction so I could focus on the guarantee.

## Frequently Asked Questions

**How do you justify $8k/month when you’re only working 20 hours a week?**
I use a simple ROI argument: if the client is raising $12 M and my work prevents a 3-month delay, the cost of delay is roughly 25 % of runway. $8k × 3 months = $24k, which is less than 1 % of a $12 M raise. The math is crude but the client accepts it because it’s their own numbers.

**What’s the biggest surprise after switching to fractional CTO?**
The hardest part wasn’t the tech—it was the hand-off. I assumed clients would hire a junior engineer to maintain what I built, but most didn’t. Instead, they wanted me to stay on a rolling contract. I now have two clients who’ve renewed five times, which is good for revenue but bad for my original goal of replaceability.

**Did you ever face scope creep? How did you handle it?**
Scope creep happened in month 4 with the mar-tech client. They wanted a real-time analytics dashboard on top of what we agreed. I said no, offered to add it as a new tier at $2,000/month, and they accepted. I used the Linear template to show the original scope vs the new request in one click—they saw the impact immediately and approved the upgrade without arguing.

**How do you handle time zones when clients are in different cities?**
I batch my US hours into a 3-hour block (2 pm–5 pm EST) and use async updates for everything else. I also set a “do not disturb” on Slack from 9 pm–7 am Nairobi time. The only exception is PagerDuty for critical alerts—clients get my phone number only for Sev-1 outages.


**Next step:** Clone the Grafana dashboard I used, set up the SLA alerts, and run a 14-day test with a client you already have. Measure latency before and after, then send the client a one-page report. If the numbers improve, propose a fixed-scope engagement at 1.5× your current rate.