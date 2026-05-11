# Land $120k remote when you're in a $30k city

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

I moved from Medellín to a remote job last year and still landed a US-level salary. Here’s exactly how I did it—without burning bridges with clients or HR.

I spent the first six months telling clients my rates in Colombian pesos and watching them cut the number in half. Then I tried framing my salary in US dollars and suddenly budgets opened up. The delta between "$1.2M COP/hour" and "$30/hour" isn’t just math—it’s perception.

That’s when I started treating salary negotiation like a product pitch. Clients care about outcomes, not where the work happens. So I built a package: my rate, my timezone coverage, my guarantees, and my tools. The result? Two clients increased my rate by 40% within 90 days because I treated negotiation as a deliverable, not a conversation.

This guide shows you how to package your remote value so clients see dollar signs instead of pesos. I’ll cover the tools I used, the scripts that saved me hours, and the exact email templates that converted "we’ll circle back" to "when can you start?"

## Why I wrote this (the problem I kept hitting)

Early in my freelance career, I charged $15/hour and justified it with "I’m in a low-cost country." That worked for a while until I took on a US client who paid $25/hour but expected 9-to-5 availability. I burned out trying to match their schedule and eventually quit the contract.

Then I realized: the problem wasn’t my location—it was how I framed my value. I wasn’t selling hours; I was selling delivery speed, timezone overlap, and risk reduction. Once I switched from "I’m cheap" to "I solve your delivery risk," my rates tripled in six months.

I kept hitting the same wall with clients: they’d ask for my rate, I’d quote in local currency, and they’d come back with a counter that assumed I’d work for free. I tried quoting in USD but still got pushback: "We budgeted $X for this role" or "Can you do it for less?"

The breakthrough came when I stopped defending my rate and started presenting a package. I bundled my rate with guarantees: I’d deliver in their timezone, I’d use their tools, I’d cover on-call for the first 30 days. Clients didn’t negotiate the package—they negotiated the delivery timeline.

This isn’t about exploiting location arbitrage. It’s about aligning incentives so both sides win. My clients get predictable delivery and I get paid what I’m worth. The key is treating your salary like a product: package it, test it, and iterate based on feedback.

If you’re charging what you think you’re worth instead of what the market will bear, you’re leaving money on the table. And if you’re using local currency to set expectations, you’re accidentally training clients to undervalue you.

## Prerequisites and what you'll build

You don’t need a fancy website or a polished portfolio to negotiate like this. You need three things: a way to quantify your value, a package that clients can evaluate, and a fallback that protects your income if the deal falls through.

I’ll show you how to build a negotiation package that includes:

- A rate card with tiers based on deliverables, not hours
- A timezone coverage guarantee that reduces risk for your client
- A 30-day onboarding package that includes tool setup and documentation
- A fallback clause that lets you walk away without burning bridges

This isn’t about tricking clients—it’s about aligning expectations. If you can’t deliver what you promise, the rate doesn’t matter.

You’ll need:

- A spreadsheet to model your costs and desired margin (I used Google Sheets with a 30% buffer for taxes and 20% for emergencies)
- A calculator for converting between local currency and USD (I used XE’s API in a Python script so I could update rates daily)
- A tool to track your time and deliverables (I switched from Toggl to Clockify when I realized Toggl’s reports weren’t granular enough for client billing)
- A contract template that includes scope, payment terms, and a fallback clause (I started with Bonsai’s template and customized it for remote work)

The goal isn’t to create a rigid package. It’s to give yourself a starting point that you can adjust based on client feedback. If a client pushes back on the 30-day onboarding, you can offer a 14-day version—but only if you’re confident you can deliver the same quality.

This approach worked for me when I moved from $30/hour to $85/hour in 12 months. It’s not magic—it’s packaging what you already do into something clients can evaluate.

## Step 1 — set up the environment

Before you negotiate, you need to know what you’re worth. I spent weeks guessing my rate based on what other freelancers in Medellín charged. Then I realized I was comparing myself to the wrong benchmark: I’m not competing with local freelancers—I’m competing with US-based contractors who deliver the same outcomes.

So I built a simple system to track my actual costs and desired margin. Here’s how:

1. **Calculate your bare minimum**
   - Rent: $400/month in Medellín
   - Food: $300/month
   - Healthcare: $150/month (I pay for private insurance to cover US visits)
   - Internet: $50/month
   - Software/tools: $100/month
   - Taxes and emergencies: 30% buffer on top of everything
   
   Total: ~$1,100/month minimum. That’s $13,200/year before I can even think about profit.

2. **Add your desired salary**
   I wanted to make $80,000/year after taxes. That means my revenue needs to cover $93,200 ($80k + $13.2k).

3. **Account for client acquisition**
   I lose 1-2 clients per year due to market changes or scope creep. So I add a 20% buffer: $93,200 * 1.2 = $111,840/year. That’s my revenue target.

4. **Convert to hourly rate**
   I work 40 hours/week * 50 weeks/year = 2000 billable hours. $111,840 / 2000 = $55.92/hour.

   But I don’t bill hourly anymore. I bill by deliverables. So I convert that to a weekly rate: $111,840 / 50 weeks = $2,237/week.

5. **Create a rate card**
   I offer three tiers based on deliverables:
   - Tier 1: MVP in 4 weeks ($2,500/week)
   - Tier 2: Full feature set in 8 weeks ($3,000/week)
   - Tier 3: Ongoing maintenance with 24-hour SLA ($3,500/week)

   I made this table to show clients the tradeoffs:

   | Tier | Deliverable | Timeline | Weekly Rate | Client Benefit |
   |------|-------------|----------|-------------|----------------|
   | 1 | MVP with core features | 4 weeks | $2,500 | Fastest time to market |
   | 2 | Full feature set with tests | 8 weeks | $3,000 | Complete product |
   | 3 | Ongoing maintenance with SLA | Monthly | $3,500 | Predictable costs |

   The numbers aren’t arbitrary. Tier 1 gives me 25 hours/week to handle unexpected delays. Tier 2 adds 10 hours for QA and documentation. Tier 3 includes an on-call rotation with a 2-hour response time.

   I tested this with my first US client. They wanted an MVP in 6 weeks for $1,800/week. I countered with Tier 1 at $2,500/week and offered to deliver in 5 weeks. They accepted. My margin was tight, but it proved the model worked.

   **Gotcha**: When I first presented this table, one client asked if I could do Tier 1 for $2,000/week. I said no and walked away. They came back with $2,500/week. Never undervalue your tier structure—it’s your anchor.

6. **Set up a rate conversion tool**
   I wrote a Python script that pulls the current USD/COP exchange rate from XE’s API and converts my rates automatically. This prevents me from accidentally quoting outdated numbers.

   ```python
   import requests
   
   def get_exchange_rate():
       url = "https://xecdapi.xe.com/v1/convert_from.json/"
       params = {
           "from": "USD",
           "to": "COP",
           "amount": 1
       }
       headers = {
           "X-XE-API-User-Agent": "my-app",
           "X-XE-API-Key": "your-api-key"
       }
       response = requests.get(url, params=params, headers=headers)
       return response.json()["to"][0]["mid"]
   
   def convert_to_local(usd_rate):
       cop_rate = get_exchange_rate()
       return usd_rate * cop_rate
   
   # Example usage
   usd_rate = 3000  # $3,000/week
   cop_rate = convert_to_local(usd_rate)
   print(f"$3,000/week = {cop_rate:,.0f} COP/week")
   ```

   I run this script every Monday and update my quotes. Clients appreciate the transparency.

   **Why this matters**: If you’re still quoting in local currency, clients will automatically halve your rate. USD signals professionalism and sets the right expectation.

## Step 2 — core implementation

With your rate card and conversion tool ready, it’s time to package your value. I learned the hard way that clients don’t care about your hourly rate—they care about delivery risk. So I built a package that answers three questions:

1. What will you deliver?
2. How long will it take?
3. What happens if something goes wrong?

Here’s the structure I use in every client proposal:

### The 3-part proposal

**Part 1: The outcome**
- A one-sentence summary of what the client gets
- Example: "A fully-tested e-commerce API that handles 100 concurrent users with <500ms response time"

**Part 2: The timeline**
- A Gantt-style breakdown with milestones
- Example:
  - Week 1-2: Core architecture + database schema
  - Week 3-4: API endpoints with unit tests
  - Week 5-6: Integration tests + load testing
  - Week 7-8: Documentation + deployment

**Part 3: The guarantee**
- A fallback clause that protects both sides
- Example: "If the API fails to meet the 500ms SLA in production for 30 consecutive days, I’ll refund 20% of the project cost and extend support by 2 weeks at no charge."

I used to think guarantees were risky. Then I had a client whose AWS bill exploded because of a misconfigured cache. I fixed it in 2 hours and absorbed the cost. That client renewed for 3 more months and referred two others. The guarantee wasn’t a cost—it was an investment in trust.

### The negotiation script

I wrote a Python script that generates personalized proposals based on client input. It pulls from a template library and inserts the client’s specific requirements. This saves me hours and ensures consistency.

```python
from jinja2 import Template

# Template for a 4-week MVP proposal
template = Template("""
**Proposal for {{ client_name }}**

**Outcome**: {{ outcome }}

**Deliverables**:
- {{ deliverables|join("\n- ") }}

**Timeline**:
- Week 1-2: {{ milestone_1 }}
- Week 3-4: {{ milestone_2 }}

**Guarantee**: {{ guarantee }}

**Rate**: {{ rate }} per week

**Payment Terms**:
- 50% upfront
- 50% on delivery
""")

# Example usage
client_name = "Acme Corp"
outcome = "A fully-tested API for user authentication with JWT"
deliverables = [
    "JWT authentication endpoints",
    "Database schema for users",
    "Unit tests covering 80% of code",
    "Postman collection for API endpoints"
]
milestone_1 = "Core architecture and database setup"
milestone_2 = "API endpoints with tests and documentation"
guarantee = "If uptime <99.5% in first 30 days, 15% refund and 2 weeks extra support"
rate = "$2,500/week"

proposal = template.render(
    client_name=client_name,
    outcome=outcome,
    deliverables=deliverables,
    milestone_1=milestone_1,
    milestone_2=milestone_2,
    guarantee=guarantee,
    rate=rate
)

print(proposal)
```

This script ensures I never forget a deliverable or milestone. It also forces me to think through the project before I quote a rate.

### The email sequence

I tested three approaches to sending proposals:

1. **The direct ask**: "Here’s my proposal. Let me know what you think." → 30% response rate
2. **The question**: "Does this timeline work for your team?" → 50% response rate
3. **The urgency**: "I can start next Monday if we agree on the scope by Wednesday." → 75% response rate

The third approach worked best because it created artificial scarcity. Clients don’t want to miss out on a slot.

Here’s the email template I use:

```
Subject: Proposal for [Project Name] – 4-week delivery

Hi [Client Name],

Thanks for the call yesterday. I’ve put together a proposal based on our discussion:

[Proposal Body]

I can start next Monday if we finalize the scope by Wednesday. Does that timeline work for your team?

Let me know if you’d like to adjust any part of the proposal.

Best,
Kubai
```

**Why this works**: The deadline is arbitrary, but it signals professionalism. Clients appreciate clear timelines because it reduces their own risk.

### Handling pushback

When clients push back on rate, I don’t defend my price. I reframe the conversation around scope or timeline.

- **Pushback**: "Can you do it for $2,000/week?"
- **Response**: "I can deliver the MVP in 3 weeks for $2,000/week, but the API won’t include load testing or documentation. That means you’ll handle deployment and bug fixes. Does that work for your team?"

This forces the client to choose between speed and quality. Most clients opt for the original scope because they know the hidden costs of cutting corners.

**Gotcha**: One client accepted a lower rate but then asked for 24-hour support. I said no and walked away. They came back with the original rate. Never let scope creep without adjusting compensation.

## Step 3 — handle edge cases and errors

The biggest mistake I made was assuming clients would stick to the scope. I had a client who kept adding "small" features: a settings endpoint, a logging middleware, a rate limiter. By week 6, I was at 40 hours of unpaid work.

That’s when I implemented a change request process. Now every new feature triggers a formal request with a timeline and cost impact.

### The change request system

I use a simple Notion database to track change requests:

| Request ID | Date | Description | Client Impact | My Impact | Status |
|------------|------|-------------|---------------|-----------|--------|
| CR-001 | 2024-05-15 | Add rate limiting to API | +2 days | +$500 | Approved |
| CR-002 | 2024-05-20 | Add user profile endpoint | +1 day | +$250 | Pending |

When a client asks for a change, I send this:

```
Subject: Change Request for [Project Name] – Rate Limiting

Hi [Client Name],

Thanks for the update. Adding rate limiting will require:

- 2 additional days of work
- $500 adjustment to the project cost

Here’s the breakdown:
- Core logic: 1 day
- Tests: 0.5 days
- Documentation: 0.5 days

Let me know if you’d like to proceed or adjust the scope.

Best,
Kubai
```

This prevents scope creep from eroding my margin. Clients appreciate the transparency, and I avoid burnout.

### The fallback clause

I include this in every contract:

> "If the project scope changes by more than 20% from the original proposal, we’ll renegotiate the timeline and cost. If we can’t agree on new terms, either party can terminate the contract with 7 days’ notice."

This gives me an out if a client keeps adding work without adjusting compensation. I’ve used it twice—both times the client agreed to renegotiate rather than lose the project.

### Handling payment issues

I use Stripe for invoicing because it supports global payouts and handles currency conversion automatically. But Stripe’s fees add up: 2.9% + $0.30 per transaction.

For projects over $10,000, I negotiate a 50% upfront payment to cover initial costs. For smaller projects, I use Wise (formerly TransferWise) to reduce fees to ~0.5%.

**Gotcha**: One client paid via PayPal and I lost 4% to fees plus a $25 withdrawal fee. Never accept PayPal for large invoices.

### The escalation path

If a client is late on payment, I follow this sequence:

1. **Day 1-3 late**: Friendly reminder + updated invoice
2. **Day 4-7 late**: Suspend work until payment is received
3. **Day 8+ late**: Terminate contract with 7 days’ notice

I’ve only had to escalate twice. Both times the client paid within 24 hours of the suspension notice.

## Step 4 — add observability and tests

You can’t negotiate effectively if you don’t know what’s working. I track three metrics for every client:

1. **Response time**: How long it takes me to reply to emails or Slack messages
2. **Delivery velocity**: How many story points I complete per week
3. **Client satisfaction**: Net Promoter Score (NPS) from quarterly check-ins

I built a simple dashboard in Google Data Studio that pulls from my time tracker (Clockify) and my CRM (Notion). Here’s the query I use to calculate delivery velocity:

```sql
SELECT
    week,
    SUM(story_points) as completed_points,
    COUNT(*) as tickets_closed,
    SUM(story_points) / COUNT(*) as velocity_per_ticket
FROM tickets
WHERE client_id = 'acme-corp'
    AND status = 'Closed'
    AND week >= DATE_SUB(CURRENT_DATE(), INTERVAL 8 WEEK)
GROUP BY week
ORDER BY week
```

This lets me show clients concrete data about my productivity. For example, I can say: "I closed 12 tickets last week for a total of 24 story points. That’s 40% faster than the industry average for a solo developer."

### The client report

Every two weeks, I send a report that includes:

- Work completed
- Upcoming milestones
- Any blockers
- My NPS score (measured via a Typeform survey)

Here’s the template:

```
Subject: Bi-weekly Update for [Project Name] – Week 6-7

Hi [Client Name],

Here’s what I’ve delivered in the last two weeks:

**Completed**:
- User authentication API (fully tested)
- Database migration scripts
- Load testing report (1000 concurrent users, 200ms p95)

**In Progress**:
- Payment integration (ETA: next week)

**Blockers**:
- None

**My NPS**: 9/10 (up from 7 last quarter)

Let me know if you’d like to adjust priorities or add new features.

Best,
Kubai
```

This keeps clients engaged and justifies my rate. When I showed one client that my velocity increased by 30% over three months, they agreed to a 15% rate increase without negotiation.

### Automating the reports

I wrote a Python script that pulls data from Clockify and Notion, generates a report, and emails it to the client. The script runs every Friday at 5 PM.

```python
import requests
from datetime import datetime, timedelta

# Clockify API setup
clockify_api_key = "your-api-key"
workspace_id = "your-workspace-id"
project_id = "acme-corp-project"

# Get time entries for the last two weeks
def get_time_entries():
    url = f"https://api.clockify.me/api/v1/workspaces/{workspace_id}/projects/{project_id}/time-entries"
    headers = {"X-Api-Key": clockify_api_key}
    params = {
        "start": (datetime.now() - timedelta(weeks=2)).strftime("%Y-%m-%d"),
        "end": datetime.now().strftime("%Y-%m-%d")
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# Get completed tickets from Notion
def get_completed_tickets():
    # Notion API call here
    pass

# Generate report
completed_tickets = get_completed_tickets()
time_entries = get_time_entries()

report = f"""
Subject: Bi-weekly Update for Acme Corp – Week {datetime.now().strftime('%V')}

Hi Acme Team,

**Completed**:
{'\n'.join([f'- {ticket["title"]} (Story points: {ticket["points"]})' for ticket in completed_tickets])}

**Time Spent**:
- Total hours: {sum(entry['timeSpent'] for entry in time_entries) / 3600:.1f}
- Billable hours: {sum(entry['timeSpent'] for entry in time_entries if entry['isBillable']) / 3600:.1f}

**Next Steps**:
- Payment integration (ETA: next week)

Best,
Kubai
"""

print(report)
```

This automation saves me 3 hours per month and makes me look like a pro.

**Gotcha**: The first time I sent an automated report, the client replied: "Did a robot write this?" I laughed and said yes. They loved the consistency.

## Real results from running this

I started this system in January 2024. Here’s what changed in six months:

- **Rate increase**: From $30/hour to $85/hour (a 183% jump)
- **Client retention**: 85% of clients renewed for a second project (up from 50%)
- **Project velocity**: Average delivery time dropped from 10 weeks to 6 weeks
- **NPS score**: Increased from 6/10 to 9/10

The biggest surprise was how clients reacted to the guarantees. I thought they’d see it as a risk, but most clients appreciated the transparency. One client said: "This is the first proposal that actually reduced my risk, not increased it."

Here’s a breakdown of my actual revenue by project type:

| Project Type | Rate/Week | Projects/Year | Annual Revenue |
|--------------|-----------|---------------|----------------|
| MVP | $2,500 | 4 | $50,000 |
| Full Feature Set | $3,000 | 3 | $45,000 |
| Ongoing Maintenance | $3,500 | 2 | $35,000 |
| **Total** | | | **$130,000** |

After taxes, expenses, and client acquisition costs, I net about $85,000/year. That’s 3x what I made when I was quoting in local currency.

The most valuable lesson wasn’t the money—it was the confidence. I used to dread negotiations because I felt like I was asking for too much. Now I walk into calls knowing exactly what I’ll deliver and why it’s worth the rate.

I also discovered that clients in higher-cost countries don’t necessarily want the cheapest option. They want the most reliable one. When I framed my rate as "I’ll deliver in your timezone with a 99.5% uptime guarantee," they stopped haggling and started asking about my process.

**Mistake I made**: I once took a project at a 20% discount because the client was a startup. Six months in, they pivoted and the project was canceled. I lost $12,000 in unpaid work. Now I require 50% upfront for new clients.

**Lesson**: Always get paid for the work you do, not the work you hope to do.

## Common questions and variations

### What if my client insists on hourly billing?

I used to charge hourly because it felt fair. Then I had a client who logged 47 hours in one week and expected a discount because "the API was easy." I switched to weekly rates after that.

If a client insists on hourly billing, I offer a capped rate: "I’ll work up to 30 hours/week for $X. If it takes longer, we’ll renegotiate." This prevents runaway hours while giving the client flexibility.

### How do I handle currency fluctuations?

I use Stripe’s automatic conversion for invoicing, but I also include a clause in my contract: "Rates are fixed in USD for the duration of the project. If the exchange rate fluctuates by more than 10% after project start, we’ll adjust the rate proportionally."

This protects me from sudden peso devaluations and gives the client predictability.

### What if I’m just starting out and don’t have leverage?

When I first started freelancing, I took a project at $15/hour. I used it as a case study to build my portfolio and then raised my rates for the next client. The key is to use early projects as stepping stones, not anchors.

I also joined a mastermind group with other freelancers in Latin America. We shared rate sheets and negotiation tactics. Having data from peers gave me confidence to quote higher rates.

### How do I say no without burning bridges?

I used to say yes to every project to build my portfolio. Then I realized that saying yes to the wrong project costs me more in stress and missed opportunities.

Now I use this script:

```
"I appreciate the opportunity, but I’m at capacity right now. I’d love to work together in the future—can we revisit this in [X weeks]?"
```

This keeps the door open while protecting my time. Most clients respect the honesty.

## Where to go from here

Pick one client you’re currently working with and rewrite your proposal using the structure in Step 2. Send it to them this week. Don’t wait for the perfect moment—do it now.

If you’re just starting out, use the rate card template to calculate your minimum viable rate.