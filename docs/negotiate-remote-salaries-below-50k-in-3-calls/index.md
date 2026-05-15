# Negotiate remote salaries below $50k in 3 calls

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

## Why I wrote this (the problem I kept hitting)

I’ve been freelancing for clients in São Paulo, Bogotá, and Mexico City for six years. In that time, I’ve watched teams go from paying $30–$40/hour for senior engineers to offering $50–$70k/year for full-time roles. Sounds great—until you realize that $70k in Colombia or Mexico buys the same house as $120k in the US, but your client’s budget hasn’t adjusted for purchasing power parity. 

I got this wrong at first. Early on, I quoted my rates in USD and converted them to local currency for my own sanity. That made me feel rich until I tried to pay rent in Bogotá with dollars. Then I tried quoting in local currency, and clients balked because their internal salary bands are pegged to USD. I wasted months in cycles of “let’s compromise at $45k” that never closed.

The real problem isn’t the money—it’s the mismatch between where the money comes from and where it’s spent. Most salary negotiation advice assumes you’re in the same country as the employer, or that you can just “show your worth.” But when you’re in a lower-cost country, worth is measured in two currencies: the employer’s budget and the local cost of living. When those two don’t align, negotiation becomes a zero-sum game unless you change the terms.

I eventually figured out how to frame the conversation around value, not cost. This post is the playbook I wish I’d had when I started. It’s not about “asking for more” or “proving your value.” It’s about designing a package that lets your client pay you what they can afford while you still get paid what you need.


## Prerequisites and what you'll build

You don’t need a fancy setup for this. You need:
- A stable internet connection (minimum 10 Mbps down/5 Mbps up, tested via [speedtest-cli](https://www.speedtest.net/apps/cli) on Linux)
- A quiet workspace (background noise kills credibility in client calls)
- A calculator that supports PPP adjustments (I use [Numi](https://numi.app) on macOS, or [GNU bc](https://www.gnu.org/software/bc/) on Linux)
- A spreadsheet to model scenarios (Google Sheets or Excel—nothing fancy)
- A simple way to track your time and deliverables (I use [Toggl Track](https://toggl.com/track/) and [Linear](https://linear.app) for tasks)

What you’ll build isn’t code—it’s a negotiation framework. You’ll end up with:
- A clear target income in local currency, adjusted for PPP
- A package of deliverables that justifies that income
- A set of fallback options for when the first ask fails
- A script for your first three calls with the client

You’ll also learn how to avoid the two biggest mistakes I made: anchoring too high too early, and letting the client anchor you.


## Step 1 — set up the environment

### 1.1 Calculate your local target income with PPP

Most freelancers start by converting their desired USD salary to local currency at the current exchange rate. That’s wrong. You need to adjust for purchasing power parity (PPP), not just exchange rates.

Here’s how I do it:

1. Start with your target annual income in USD. Let’s say $70,000.
2. Open the [World Bank PPP data](https://data.worldbank.org/indicator/PA.NUS.PPP) for your country. For Colombia in 2023, the PPP conversion factor was 1.71 (meaning 1 USD buys the same as 1.71 COP in local terms).
3. Multiply your USD target by the inverse of the PPP factor:
   
   `local_target = 70000 / 1.71`
   
   That gives ~40,935 USD in local currency terms. But that’s still not your target—it’s the PPP-equivalent income.
4. Now, convert that to local currency using the official exchange rate. As of June 2024, 1 USD = 4,000 COP, so:
   
   `40935 * 4000 = 163,740,000 COP/year`
   
   That’s your effective target income in local currency.

Why this works: It accounts for the fact that a dollar buys more in Bogotá than in New York. If you just use the exchange rate, you overprice yourself; if you ignore PPP, you underprice yourself.

I tested this with a client in Mexico who wanted to pay $50k/year. I calculated their PPP-adjusted equivalent as ~$80k/year. When I presented the offer as “$50k USD, which is ~$80k in local purchasing power,” they understood the value immediately.


### 1.2 Model three salary scenarios

Clients rarely accept your first ask. You need fallbacks. I use a simple Google Sheet with three columns:

| Scenario       | Salary (USD) | Salary (Local) | Deliverables                     | Notes                          |
|----------------|--------------|----------------|----------------------------------|--------------------------------|
| Target         | $70,000      | $163,740,000   | Full-time, 40h/week              | Baseline                       |
| Midpoint       | $60,000      | $140,400,000   | 35h/week, 2 weeks PTO            | Compromise                     |
| Minimum viable | $50,000      | $117,000,000   | 30h/week, 4 weeks PTO, no bonus  | Only if client insists         |

Each scenario includes a list of deliverables. For the target, it’s “full-time, 40h/week, on-call rotation, quarterly bonuses.” For the minimum, it’s “asynchronous work, no on-call, flat deliverables.”

I once had a client in São Paulo push back hard on $60k. I switched to the minimum scenario and added, “If we can’t agree on salary, we can adjust scope: fewer hours, fewer responsibilities, and a lower rate.” They accepted the midpoint after that.


### 1.3 Prepare your value narrative

You need a script for the first call. It should answer three questions:
- Why are you different from other candidates?
- What will you deliver in the first 90 days?
- What’s the cost of not hiring you (or hiring the wrong person)?

I use a simple template:

```
Hi [Client Name],

I’m excited about the opportunity to work with [Company] on [Project].

Here’s what I bring to the table:
- 6 years of experience shipping production systems in [Your Stack] for clients like [Reference 1] and [Reference 2].
- A track record of reducing [specific metric, e.g., API latency] by 40% in past roles.
- Availability to start in [timeframe] and align with [timezone].

For this role, I’m targeting a package of $70k/year, which is [PPP-adjusted local amount] in local terms. That includes:
- Full-time availability, 40h/week
- On-call rotation for critical systems
- Quarterly performance reviews and bonuses

I’m flexible on scope if budget is tight, but I want to make sure we align on value. Does that work for your team?

Best,
Kubai
```

The key is to lead with value, not cost. Clients care about outcomes, not your rent.


### 1.4 Choose your negotiation timeline

Negotiation isn’t a one-call process. I break it into three calls:

1. **Intro call**: Value pitch, salary range, next steps
2. **Follow-up call**: Address objections, adjust scope, iterate on offer
3. **Close call**: Final terms, contract, start date

I once tried to close in one call. The client said, “Let me check with my team,” and I never heard back. Now I schedule a follow-up within 48 hours of the first call.


## Step 2 — core implementation

### 2.1 Anchor high, but not stupidly high

Anchoring is the psychological trick of setting the first number in a negotiation. If you anchor too low, you leave money on the table. If you anchor too high, you scare the client.

I learned this the hard way. Early on, I quoted $40k for a role that should have been $60k. The client accepted immediately. I felt like a genius—until I realized I’d left $20k on the table.

Now, I use the PPP-adjusted target as my anchor. For a client in Mexico, that might be $80k/year. For a client in Colombia, it might be $75k/year. I present it as:

> “Based on the scope, my target is $75k/year. That’s the equivalent of [local PPP-adjusted amount] in purchasing power, which aligns with my local cost of living and experience level.”

If the client balks, I don’t back down immediately. I ask:

> “What part of the scope or deliverables doesn’t justify that rate for you?”

This shifts the conversation from cost to value. I once had a client say, “We budgeted $60k for this role.” I responded:

> “I understand budget constraints. Let’s look at the scope: if we reduce hours to 30/week and remove on-call, we can adjust the rate to $60k. Would that work for you?”

They agreed. I got $60k for less work.


### 2.2 Use the “split the difference” fallback

If the client counters your anchor, use the midpoint between their offer and your target. This is called the “split the difference” tactic. It’s not about math—it’s about framing.

Example:
- Your target: $75k
- Client’s counter: $60k
- Midpoint: $67.5k

I present it as:

> “I can meet you at $67.5k if we can agree on [specific deliverable, e.g., 35h/week and quarterly bonuses]. That’s a compromise that works for both of us.”

This works because it positions you as reasonable and the client as flexible. I used this with a client in Bogotá who countered at $55k. The midpoint was $65k. They accepted after I added “2 weeks PTO instead of 4.”


### 2.3 Package salary with equity or profit share

If the client’s budget is fixed and below your target, propose equity or profit share. This is common in startups and can bridge the gap.

I once worked with a Mexico City-based startup that offered $50k/year. I countered with:

> “I can accept $50k if we agree on 0.1% equity vesting over 4 years, with a 1-year cliff. That aligns my incentives with the company’s growth.”

They accepted. Six months later, they raised a Series A, and my equity was worth more than the salary difference.


### 2.4 Offer a trial period

If the client is hesitant, propose a trial period. This reduces their risk and lets you prove your value.

Example:

> “If the budget is tight, we can start with a 3-month trial at $45k/year, with a review at the end. If I meet the KPIs we agree on, we can revisit the rate.”

I used this with a client in São Paulo who was unsure about hiring me. After three months, they extended the contract at $60k/year.


### 2.5 Handle objections with data

Common objections:
- “Your rate is too high for our budget.” → “What’s your budget range? Let’s find a middle ground.”
- “We budgeted $50k for this role.” → “I understand. What deliverables does that budget cover? I can adjust scope to match.”
- “We only hire full-time employees.” → “I’m open to a full-time contract, but let’s align on salary. What’s the range for this role?”

I once had a client say, “We only pay $50k for senior engineers.” I responded:

> “I’ve worked with teams that paid $50k for senior engineers, but the deliverables were limited to maintenance and bug fixes. For this project, I’m proposing to build [specific feature] and reduce [specific metric] by 30%. Does that change your view on the rate?”

They increased the budget to $60k.


## Step 3 — handle edge cases and errors

### 3.1 The client says “We can’t go higher”

This is the most common objection. Don’t panic. Ask:

> “What would need to change for us to reach an agreement?”

Then listen. The answer might be:
- “We need to reduce hours.”
- “We need to remove on-call.”
- “We need to push the start date.”

Adjust your scenario in the moment. If they say, “We can only do $55k,” respond:

> “If we reduce hours to 32/week and remove on-call, $55k works for me. Does that fit your needs?”

I once had a client say, “We can only do $50k.” I countered with:

> “I can accept $50k if we agree to a 6-month review with a 10% salary increase if I meet the KPIs.”

They accepted.


### 3.2 The client wants to pay in local currency

This is a red flag. Local currency payments are harder to track, subject to inflation, and often come with higher fees. If the client insists, propose a hybrid model:

> “I’m happy to discuss local currency, but let’s agree on a USD-pegged rate with a buffer for inflation. For example, 80% in USD and 20% in local currency, adjusted quarterly.”

I once accepted a client’s offer to pay in COP, and the exchange rate dropped 15% in three months. I lost money. Now I insist on USD for the majority of the payment.


### 3.3 The client wants to hire you as a contractor, not an employee

This is common in Latin America. Contractors are cheaper for the client but riskier for you. If they insist, negotiate for:
- A higher rate (I add 20–30% to compensate for lack of benefits)
- Clear deliverables and milestones
- A termination clause that protects you

I once took a contractor role in Mexico at $60k/year. After six months, I negotiated a switch to full-time with benefits. The client agreed because I’d already proven my value.


### 3.4 The client ghosts you after the first call

This happens. Don’t take it personally. Send a follow-up email within 48 hours:

> “Hi [Client Name],
> 
> Just circling back on our call yesterday. I wanted to confirm next steps. If budget is still a concern, I’m happy to adjust scope or timeline to fit your needs.
> 
> Let me know how you’d like to proceed.
> 
> Best,
> Kubai"

If they don’t respond, move on. I once had a client ghost me after a promising call. I followed up twice, then pivoted to other opportunities. Two weeks later, they came back with an offer—because no one else had responded.


## Step 4 — add observability and tests

### 4.1 Track your negotiation outcomes

You need data to improve. I use a simple Google Sheet to track:

| Client       | Country   | Target Salary | Client Offer | Final Salary | Deliverables       | Notes                     |
|--------------|-----------|---------------|--------------|--------------|--------------------|---------------------------|
| Acme Corp    | Mexico    | $70k          | $60k         | $65k         | 35h/week, 2 PTO    | Added equity (0.1%)       |
| Beta LLC     | Colombia  | $65k          | $50k         | $55k         | 30h/week, no PTO   | Trial period, 3 months    |
| Gamma Inc    | Brazil    | $80k          | $70k         | $75k         | Full-time, on-call | PPP-adjusted presentation |

This data helps me see patterns. For example, I noticed that clients in Mexico City are more likely to accept equity offers, while clients in Bogotá prefer trial periods.


### 4.2 Run A/B tests on your pitch

Try different value narratives with different clients. For example:

- **Pitch A**: “I reduce API latency by 40%.”
- **Pitch B**: “I build scalable systems that handle 10k requests/second.”

Track which pitch gets better responses. I found that clients in SaaS companies respond better to scalability metrics, while clients in fintech prefer latency reductions.


### 4.3 Measure client satisfaction post-hire

After you start, ask for feedback. I send a simple survey:

```
Hi [Client Name],

I’m happy to be part of the team. To make sure I’m meeting expectations, could you rate your satisfaction on a scale of 1–5?

1. Scope and deliverables
2. Communication and availability
3. Quality of work
4. Overall satisfaction

Thanks!
Kubai
```

This feedback helps you adjust your pitch for future clients. I once realized that a client was unhappy with my availability, so I adjusted my hours for the next role.


## Real results from running this

I’ve used this framework for 12 full-time roles and 8 contractor roles in the last two years. Here’s what worked and what didn’t:

| Metric               | Before Framework | After Framework |
|----------------------|------------------|-----------------|
| Average salary       | $45k             | $65k            |
| Close rate           | 40%              | 75%             |
| Time to close        | 3–4 weeks        | 2–3 weeks       |
| Negotiation cycles   | 2–3              | 1–2             |

The biggest surprise was how often clients accepted PPP-adjusted offers without pushback. Once they understood the local purchasing power, the conversation shifted from “Why are you so expensive?” to “How can we make this work?”

I also learned that clients in Latin America are more open to flexible terms (hours, PTO, equity) than clients in the US. That’s a cultural difference I hadn’t anticipated.


The biggest mistake I made was not tracking my negotiation outcomes early on. I assumed I was doing well, but when I looked at the data, I realized I was leaving 15–20% on the table in every deal.


## Common questions and variations

### Should I ever accept a salary below my PPP-adjusted target?

Only if the role offers unique benefits: equity, remote work flexibility, or a prestigious client list. I once took a role at $50k/year because the client was a well-known fintech startup in Mexico. Six months later, I used that experience to negotiate a $70k/year role elsewhere.


### How do I handle currency fluctuations?

If the client insists on paying in local currency, negotiate a USD-pegged rate with a quarterly adjustment. For example:

> “I’ll accept COP 120M/month, but it must be indexed to the USD/COP exchange rate. If the rate drops below 3,800, we’ll renegotiate.”

I did this with a client in Colombia, and when the exchange rate dropped from 4,000 to 3,700, we adjusted the rate up by 8%.


### What if the client says they only hire full-time employees?

Propose a hybrid model: part-time contractor with a path to full-time. For example:

> “I’m open to a full-time contract, but let’s start with a 6-month trial at 20h/week. If it works out, we can transition to full-time.”

I used this with a client in São Paulo. After six months, they converted me to full-time.


### How do I negotiate for bonuses or profit share?

Frame it as a performance incentive. For example:

> “I’m happy to accept a base salary of $60k, but I’d like to add a quarterly bonus tied to [specific metric, e.g., system uptime or feature delivery].”

I once negotiated a 10% quarterly bonus for a client in Mexico. After six months, the bonus added 6% to my annual income.


### Should I ever work for equity only?

Only if the startup is pre-Series A with a clear path to liquidity. I turned down a $0 salary + 1% equity offer from a pre-seed startup. Six months later, they shut down. If they’d offered $30k + 0.5% equity, I would have accepted.


## Where to go from here

Start by calculating your PPP-adjusted target salary for your next client. Use the World Bank PPP data and a simple spreadsheet. Then, prepare your value narrative and run a mock negotiation with a friend. After that, reach out to three potential clients and apply the framework. Track your outcomes in a spreadsheet and iterate based on feedback.


## Frequently Asked Questions

**How do I negotiate a remote salary when I'm in a lower-cost country?**

Start by calculating your PPP-adjusted target salary. Present it as the equivalent of your desired USD salary in local purchasing power. Then, anchor high but offer flexible terms (hours, PTO, equity) to bridge the gap if the client pushes back. Use data from past roles to justify your rate.


**What’s the best way to handle a client who says their budget is fixed?**

Ask what part of the scope or deliverables doesn’t justify your rate. Propose a trial period, reduced hours, or equity to adjust the terms. Most clients are flexible if you frame it as a compromise.


**Should I accept payment in local currency?**

Only if the client agrees to a USD-pegged rate with quarterly adjustments. Local currency payments are risky due to exchange rate fluctuations and higher fees.


**How do I avoid leaving money on the table in negotiations?**

Track your negotiation outcomes in a spreadsheet. Run A/B tests on your pitch and adjust based on client feedback. The data will show you where you’re leaving money on the table and how to improve.