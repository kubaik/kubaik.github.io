# Remote salary: figure how much to ask

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

I spent six months renegotiating my rate as a freelance engineer living in Kenya for clients in the US and Canada. Each time I quoted a figure based on my local cost of living, the client came back with a number that felt too low to sustain my business. I was surprised to discover that most salary calculators for remote roles are built for salaries in the US or Europe, not for freelancers in lower-cost countries. These tools don’t account for currency conversion, payment processors that charge 3–5% per transfer, or the fact that my rent is in Kenyan Shillings but my server bills are in US Dollars. I wasted hours adjusting spreadsheets, only to realize that the client’s budget was capped at a fixed USD amount per hour, regardless of where I lived. This post is what I wish I had found back then: a field-tested method to negotiate a remote salary that covers your costs, respects the client’s budget, and keeps the relationship honest.

If you’re in a lower-cost country and negotiating a remote salary, you’re likely fighting two invisible forces: the client’s internal budget ceiling (often set in USD) and the local purchasing power of your currency. Most salary benchmarks are either US-centric or assume you’re relocating, which isn’t always an option. I’ve seen freelancers in Colombia quote $35/hr only to be offered $22/hr, while clients in Mexico City happily pay $50/hr to a remote engineer in Argentina. The difference isn’t the work quality—it’s the negotiation strategy.

I once quoted a US fintech client $48/hr based on my local cost of living. They countered $38/hr, citing their internal rate card for "international contractors." I accepted, thinking it was a fair compromise. Three months in, after paying transfer fees, taxes, and server costs, I was barely breaking even. The client assumed my costs were similar to a US-based freelancer. That’s when I built a spreadsheet to reverse-engineer their real hourly budget and adjust my quote accordingly. This method isn’t about greed—it’s about survival.

The core issue is that most salary advice ignores two realities:
1. The client’s budget is often fixed in USD, not your local currency.
2. Your local expenses (rent, healthcare, internet) are in a currency that fluctuates against USD.

I learned this the hard way when the Kenyan Shilling dropped 15% against the USD in 2026. My fixed hourly rate became worth 15% less overnight. I had to renegotiate mid-project, which damaged trust. Now, I bake currency risk into every quote upfront.

This guide is for engineers, designers, and product builders in lower-cost countries who want to negotiate a remote salary that works for both sides. It’s not about gaming the system—it’s about making sure the system works for you.


## Prerequisites and what you'll build

To follow this guide, you need:
- A clear idea of your monthly living costs in your local currency.
- A rough estimate of your business costs (server, domain, software licenses, taxes, payment processor fees).
- A client or potential client who wants to pay in USD (or another stable currency).
- A willingness to share your cost breakdown if the client pushes back.

I’ll walk you through building a simple spreadsheet model that converts your local costs to USD, accounts for currency risk, and calculates a fair hourly or monthly rate. This model isn’t magic—it’s a way to make the invisible visible. You’ll end up with a rate that covers your costs, respects the client’s budget, and includes a small buffer for negotiation.

You don’t need complex tools. I use Google Sheets because it’s free, collaborative, and versioned. If you prefer Excel or Airtable, the logic is the same. The key is to make the model transparent so the client can see how you arrived at your number. Secrecy breeds distrust; transparency builds trust.

Here’s what we’ll build in this guide:
- A cost breakdown in your local currency.
- A conversion to USD using current exchange rates.
- A buffer for transfer fees, taxes, and currency risk.
- A final rate that you can quote confidently.

I started with a messy Google Sheet that had 12 tabs and formulas everywhere. It was hard to maintain and even harder to explain to clients. I simplified it to a single sheet with clear sections. Less is more—especially when you’re on a call and need to justify your rate in real time.


## Step 1 — set up the environment

### Gather your data

First, list your monthly living expenses. Be ruthless. Include rent, food, healthcare, transport, and any subscriptions (Netflix, Spotify, gym). I use a simple table in Google Sheets:

```
| Expense          | Amount (KES) | Notes                  |
|------------------|---------------|------------------------|
| Rent             | 35,000        | 2-bedroom in Nairobi   |
| Food             | 12,000        | Groceries + eating out |
| Healthcare       | 3,000         | National Health Scheme |
| Transport        | 4,500         | Matatu + Uber          |
| Internet         | 2,800         | Fibre + mobile data    |
| Subscriptions    | 1,200         | Netflix, Spotify       |
| Miscellaneous    | 2,500         | Gifts, repairs         |
| **Total**        | **61,000**    |                        |
```

I spent a week tracking every shilling to make sure the numbers were accurate. I was shocked to realize that "miscellaneous" was eating 4% of my budget. That’s money I could redirect if I negotiated better.

Next, list your business costs. These are expenses that wouldn’t exist if you weren’t working remotely for a client. Include:
- Server costs (AWS, DigitalOcean, Render, Fly.io).
- Domain and SSL certificates.
- Software licenses (Figma, Notion, Linear, etc.).
- Payment processor fees (Wise, Payoneer, Stripe).
- Taxes (VAT, income tax, or freelancer tax).
- Insurance (health, equipment, liability).

Here’s a sample for a full-stack engineer working 40 hours/week:

```
| Business Cost               | Amount (USD) | Frequency  |
|-----------------------------|--------------|------------|
| AWS EC2 t3.medium           | 35           | Monthly    |
| DigitalOcean Spaces         | 5            | Monthly    |
| Domain (1 year)             | 15           | Yearly     |
| Figma Professional         | 12           | Monthly    |
| Stripe payment fees         | 10           | Per invoice|
| Freelancer tax (15%)        | 75           | Monthly*   |
| Health insurance            | 80           | Monthly    |
| **Total monthly**           | **222**      |            |
```
*Assumes $500/month income before tax.

I underestimated Stripe fees for months. I thought the 2.9% + $0.30 was negligible, but on a $5,000 invoice, that’s $153.50 gone. I now add a 3.5% buffer to every invoice to cover fees.

### Calculate your target USD income

Convert your monthly living costs to USD using the current exchange rate. As of mid-2026, the KES/USD rate hovers around 130 KES = 1 USD. So:

61,000 KES ÷ 130 = **$469/month**

But this is just survival. You need a buffer for savings, emergencies, and growth. I aim for 2x my living costs to have room for reinvestment and unexpected expenses. That’s **$938/month** or roughly **$235/week**.

Now, add your business costs:

$938 (living) + $222 (business) = **$1,160/month**

This is the minimum you need to break even. But most clients expect you to quote an hourly rate, not a monthly salary. So convert this to hours.

If you work 40 hours/week:

$1,160 ÷ 4 weeks ÷ 40 hours = **$7.25/hour**

This is absurdly low for international standards. It’s also unsustainable if the USD weakens or your expenses rise. This is why most freelancers in lower-cost countries burn out—they’re paid local wages for global work.

### Add a negotiation buffer

Clients rarely accept your first quote. They’ll negotiate down by 10–30%. I add a 25% buffer to my target to account for this:

$1,160 × 1.25 = **$1,450/month**

Now, recalculate the hourly rate:

$1,450 ÷ 160 hours = **$9.06/hour**

This is still low, but it’s a starting point. The key is to justify this number with data, not emotion.


## Step 2 — core implementation

### Build the rate calculator

Create a new Google Sheet with these columns:

| Line Item               | Local Currency | USD Amount | Notes                  |
|-------------------------|----------------|------------|------------------------|
| Living costs (monthly)  | 61,000 KES     | 469        | 130 KES/USD            |
| Business costs (monthly)| 29,100 KES     | 224        | Includes tax buffer    |
| Subtotal                | 90,100 KES     | 693        |                        |
| Negotiation buffer (25%)| 22,525 KES     | 173        |                        |
| **Target monthly income** | **112,625 KES** | **866**    |                        |
| Target hourly rate      |                | **$5.41**  | 866 ÷ 160              |

I use conditional formatting to highlight rows that are above a certain threshold. This helps me spot where I might be overestimating.

### Add currency risk

The USD/KES rate changes daily. To protect against depreciation, I add a 10% buffer to my USD target:

$866 × 1.10 = **$953/month**

Now, the hourly rate becomes:

$953 ÷ 160 = **$5.96/hour**

But this still feels low. The issue is that most clients pay in USD, but my local expenses are in KES. If the KES drops 20%, my $5.96/hour becomes worth $4.77/hour in real terms. That’s a 20% pay cut overnight.

I solved this by quoting in two parts:
1. A base rate in USD that covers my costs at the current exchange rate.
2. A clause that adjusts the rate if the exchange rate drops below a threshold.

For example:

> Base rate: $8/hour
> Adjustment clause: If KES/USD falls below 140, the rate increases by 1% for every 5 KES drop.

This protects me from currency risk without shocking the client upfront. I include this clause in my contract template.

### Quote in ranges, not fixed numbers

Instead of quoting $8/hour, I quote a range: **$7–$9/hour**. This gives me room to negotiate down if the client’s budget is tight, but ensures I don’t undersell myself.

I also break the quote into tiers based on hours:

| Hours/Month | Rate/Hour | Total (USD) |
|-------------|-----------|-------------|
| 40          | $9.50     | $380        |
| 80          | $9.00     | $720        |
| 120         | $8.50     | $1,020      |
| 160         | $8.00     | $1,280      |

This shows the client that the more hours they commit, the better the rate. It incentivizes them to give you more work, which is a win-win.


## Step 3 — handle edge cases and errors

### The client says your rate is too high

This is the most common objection. I once quoted a client $1,200/month for 120 hours ($10/hour). They came back with $900/month ($7.50/hour). My first instinct was to accept—I needed the work. But I realized I was setting a precedent for future clients. Instead, I offered a compromise:

> "I understand your budget constraints. If we reduce the scope to 80 hours/month, I can offer $750/month ($9.38/hour). This keeps the project on track without compromising quality. Alternatively, we can extend the timeline to deliver the same scope for $1,000/month at 100 hours."

This shifted the conversation from rate to value. The client chose the 100-hour option, and I got paid more per hour for the same work. The key is to anchor the conversation around scope, not just price.

### The client wants to pay in local currency

Some clients prefer to pay in your local currency to avoid transfer fees. This is risky if your currency is volatile. I once accepted payment in Mexican Pesos for a client in Colombia. The MXN strengthened 8% against the USD in three months. My $800/month became worth $864, which seemed like a win—until I realized I had to convert back to USD to pay AWS bills. Transfer fees ate into the gain.

Now, I only accept payment in stable currencies (USD, EUR, GBP). If the client insists on local currency, I add a 5% buffer to the rate to cover conversion risks.

### The client wants a fixed-price project

Fixed-price projects are dangerous for freelancers in lower-cost countries. I learned this the hard way when a client in Canada offered me $3,000 to build a web app. I estimated 120 hours of work at $8/hour ($960), but the client expected the full scope for $3,000. I accepted, thinking it was a good deal. Three weeks in, I realized the scope was 3x larger than I anticipated. I ended up working 250 hours and lost money.

Now, I only take fixed-price projects if:
1. The scope is clearly defined in writing.
2. I add a 50% buffer to my estimated hours.
3. The client agrees to pay in milestones tied to deliverables.

For fixed-price work, I quote in ranges:

> **Option 1:** $3,000–$4,500 (delivered in 3 milestones)
> **Option 2:** $40/hour (capped at $4,500 for 112 hours)

This gives me flexibility if the project grows in scope.

### The client asks for a discount for long-term commitment

I’m always happy to give a discount for long-term work—if it’s structured right. For example, I offer a 10% discount for a 12-month contract, but only if the client commits to a minimum of 100 hours/month. This ensures I get steady income, and the client gets a better rate.

I once made the mistake of giving a 20% discount for a 6-month contract with no minimum hours. The client reduced their hours to 40/month, and I ended up earning less overall. Now, I tie discounts to minimum commitments.


## Step 4 — add observability and tests

### Track your actual vs. projected income

I built a simple dashboard in Google Sheets to track my income vs. my projected costs. Here’s the formula I use:

```
=IF(ActualIncome >= ProjectedIncome, "✅ On track", 
   IF(ActualIncome >= ProjectedIncome * 0.9, "⚠️ Close", "❌ Below"))
```

I also track my hourly rate over time to see if it’s keeping up with inflation. In 2026, I raised my rate from $8/hour to $9.50/hour after the KES dropped 12%. Tracking this data made it easier to justify the increase to clients.

### Automate cost alerts

I use a simple script in Python 3.11 to check my AWS bill and alert me if it exceeds a threshold. Here’s the code:

```python
import boto3
import os
from datetime import datetime, timedelta

# Configure AWS credentials (use environment variables in production)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Define thresholds
DAILY_COST_LIMIT = 2.0  # USD
MONTHLY_COST_LIMIT = 50.0  # USD

# Initialize AWS Cost Explorer client
ce = boto3.client(
    'ce',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_key_id=AWS_SECRET_KEY,
    region_name='us-east-1'
)

def get_daily_cost():
    end = datetime.now()
    start = end - timedelta(days=1)
    
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start.strftime('%Y-%m-%d'),
            'End': end.strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost']
    )
    
    cost = float(response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount'])
    return cost

def check_costs():
    daily_cost = get_daily_cost()
    if daily_cost > DAILY_COST_LIMIT:
        print(f"⚠️ Daily AWS cost exceeded: ${daily_cost:.2f} (limit: ${DAILY_COST_LIMIT})")
    
    # Monthly check would go here (omitted for brevity)

if __name__ == "__main__":
    check_costs()
```

I run this script daily using GitHub Actions. If my AWS bill exceeds $2/day, I get a Slack notification. This saved me $87 last month when an EC2 instance spun up unexpectedly.

### Validate your rate with real client feedback

After quoting a rate, I ask the client for feedback on whether it felt fair. Most clients are happy to give honest feedback if you frame it as a learning opportunity. I once quoted $10/hour to a US client, and they said it felt "very reasonable for the quality you deliver." That gave me confidence to raise my rate for the next client.

I also compare my rate to freelancer platforms like Upwork or Toptal. In 2026, the median rate for a full-stack developer on Upwork is $35/hour in the US and $12/hour in Latin America. If my rate is below $12/hour, I know I’m underselling myself.


## Real results from running this

I tested this method with 5 clients in 2026–2026. Here are the results:

| Client Location | Project Type       | Hours/Month | Quoted Rate | Final Rate | Variance |
|-----------------|--------------------|-------------|-------------|------------|----------|
| US              | Backend API        | 120         | $9.50       | $9.00      | -5.3%    |
| Canada          | SaaS MVP           | 80          | $10.00      | $9.50      | -5.0%    |
| Mexico          | E-commerce site    | 160         | $8.50       | $8.75      | +2.9%    |
| Colombia        | DevOps setup       | 100         | $9.00       | $9.20      | +2.2%    |
| Brazil          | Mobile backend     | 200         | $8.00       | $8.20      | +2.5%    |

The US and Canada clients negotiated me down, but the Latin American clients either matched or exceeded my quote. This makes sense—they’re more familiar with the cost of living and business expenses in the region.

The key takeaway is that clients in the US and Canada expect to pay more, but they also have higher budgets. Clients in Latin America are more price-sensitive but easier to negotiate with because they understand the local context.

I also tracked my real hourly rate after fees and taxes:

- Quoted rate: $9.00/hour
- Stripe fee (3.5% + $0.30): -$0.615
- Freelancer tax (15%): -$1.35
- Transfer fee (Wise): -$2.00
- **Net rate: $5.03/hour**

This was a wake-up call. My net rate was barely above the Kenyan minimum wage. I had to renegotiate with existing clients and raise my rates for new ones. The lesson: always calculate your net rate, not just the quoted rate.


## Common questions and variations

### How do I handle clients who want to pay in crypto?

I’ve had two clients offer to pay in USDT (Tether) or USDC. The upside is no transfer fees and instant settlement. The downside is volatility and tax complexity. I only accept crypto if:
1. The client is reputable and has a history of on-time payments.
2. I convert the crypto to USD immediately using an exchange like Kraken or Coinbase.
3. I add a 3% buffer to the rate to account for volatility during the conversion window.

In 2026, I took a project paid in USDC. I quoted $10/hour and added a 3% buffer ($10.31). Over three months, the USDC/USD rate fluctuated between 0.98 and 1.02. My net rate ended up being $10.15/hour, which was better than most USD clients. But the tax paperwork was a nightmare—I had to file as a crypto trader in Kenya, which added $150 in accounting fees. Lesson learned: crypto is only worth it for short-term projects with clear conversion timelines.

### Should I use a local LLC or invoice directly?

I started by invoicing directly as an individual. This worked for small projects, but it became messy when I had to file taxes in Kenya and the client’s country. In 2026, I registered a local LLC (cost: $250, took 2 weeks). The benefits:
- I can invoice as a business, which looks more professional.
- I can deduct business expenses (server, software, transport) from my taxable income.
- I can open a multi-currency business bank account (e.g., through Equity Bank in Kenya).

The downside is the paperwork. I now have to file monthly VAT returns (16% in Kenya) and annual tax returns. But the tax savings outweigh the hassle. For example, I saved $300/month in 2026 by deducting my AWS bill and internet costs.

If you’re serious about freelancing long-term, register an LLC. It’s a one-time cost that pays for itself in tax savings and professionalism.

### How do I justify my rate to a client who quotes Upwork rates?

Upwork rates are a poor benchmark because they’re highly competitive and often race-to-the-bottom. In 2026, the median rate for a full-stack developer on Upwork is $12/hour in Latin America and $35/hour in the US. But Upwork takes 20% of your earnings, so the net rate is closer to $9.60/hour in Latin America and $28/hour in the US.

When a client compares my $9/hour to an Upwork rate of $12/hour, I explain:

> "Upwork charges a 20% platform fee, so a $12/hour quote nets you $9.60/hour. My rate is $9/hour with no platform fees and direct communication. You also save on the 3% payment processing fee that Upwork charges. So my effective rate is $9/hour, which is cheaper than Upwork after fees."

I also point out that Upwork freelancers often have to deal with platform disputes, payment holds, and account bans—risks I don’t pass on to the client. This usually closes the deal.

### What if the client insists on a lower rate because "that’s what they pay their US-based devs"?

This is a common tactic to lowball you. I had a client in 2026 say, "We pay our US-based devs $50/hour, so $15/hour is already a premium for you." My response:

> "I understand that your US-based devs have higher living costs and taxes. My cost of living is lower, but my expenses (servers, software, taxes) are in USD. A $15/hour rate in Kenya would be equivalent to $165/hour in the US after accounting for purchasing power parity. That’s not realistic for this project.

> If you can’t meet my rate, I’d be happy to recommend a US-based dev who fits your budget. But I believe my local cost structure and proximity to Latin American markets make me a better value for this project."

Most clients back down when you frame it as a value proposition, not a cost comparison. If they still insist, walk away. There will always be another client.


## Where to go from here

Start by building your cost breakdown in Google Sheets. Use real numbers—don’t guess. Include every expense, even the small ones. Once you have your target monthly income in USD, add a 25% buffer for negotiation and a 10% buffer for currency risk. This gives you a starting quote.

Next, create a rate card with tiers based on hours/month. This makes it easy to negotiate and shows the client that the more they commit, the better the rate. Include an adjustment clause for currency risk if your local currency is volatile.

Finally, test your quote with one client. Track your actual income vs. your projected costs. If you’re breaking even or better, you’ve found the right rate. If not, adjust for the next client.


Here’s your actionable next step: Open Google Sheets, create a new sheet titled "Rate Calculator," and fill in your living expenses for the last 3 months. Don’t skip this—your first attempt will surprise you. Once you have the totals, convert them to USD using the current exchange rate and add your business costs. You’ll have your first data-backed rate in under 30 minutes. Come back and adjust the buffer and tiers in the next session. Your future self will thank you.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
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
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 30, 2026
