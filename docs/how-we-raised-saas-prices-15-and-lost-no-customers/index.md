# How we raised SaaS prices 15% and lost no customers

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In January 2023, we had 127 paying customers split evenly between Europe and the US. Average revenue per account (ARPA) was €118 per month. We were profitable on paper, but every customer renewal felt like a negotiation. Two things killed us: churn from price-sensitive SMBs and the constant need to discount to win enterprise deals. Our pricing page listed four tiers: Free, Starter at $49, Growth at $199, and Enterprise at custom pricing. The problem wasn’t the tiers—it was that 62% of customers were on Starter, a tier we set in 2019 when we had 12 customers and AWS bills under $800/month. Our cost per customer had tripled since then. Support tickets per customer were flat at 0.4, but infra costs per customer had climbed from $4 to $18. We needed to raise prices without scaring off the customers who loved us.

I started by looking at our churn data. Customers who joined in 2020 churned at 9% annually; those who joined in 2022 churned at 18%. The difference wasn’t product—it was price sensitivity. European SMBs, especially in Germany and France, were comparing us to German-hosted competitors at €39/month. US customers were comparing us to Heroku-style competitors at $29/month. Our Starter tier was in a no-man’s land: too expensive for bootstrappers and too cheap for value-focused SMBs.

The second problem was enterprise deals. We were losing 3 out of 4 enterprise RFPs because our custom pricing sheet started at $4,000/month. Competitors like Retool and Tray.io were quoting $8,000–$12,000 for comparable features. We needed to anchor enterprise pricing higher while protecting our SMB base.

**The key takeaway here is** that pricing isn’t a one-time launch problem—it’s a data problem that compounds as your infra and support costs grow. If your pricing page feels static, you’re probably underpricing new features and over-serving price-sensitive customers.

## What we tried first and why it didn’t work

Our first attempt was a blunt price increase: Starter from $49 to $79, Growth from $199 to $249. We announced it in a blog post titled "Pricing update: investing in reliability." Within 10 days, 14 customers churned, all from the Starter tier. One customer emailed: "Your new price is 60% higher than what I’m paying now. I’ll move to a cheaper competitor." We had expected 3–5% churn based on prior elasticity tests, but 11% of Starter customers left.

The second failure was worse. We tried grandfathering existing customers: keeping their old price for 12 months, then migrating them to the new tiers. This created a support nightmare. Customers who’d been on Starter for two years suddenly saw a 60% increase with no warning. Our support queue spiked from 8 tickets/day to 42, mostly about "why did my invoice double?" We had to manually backdate credits, which cost us 18 engineering hours in one week.

The third attempt was value-based pricing: we added a "Usage" column to the pricing page with metered API calls, storage, and compute. We thought customers would self-select into higher tiers. What actually happened was confusion. 37% of customers opened tickets asking "how many credits do I need?" Our pricing page now had 11 fields instead of 4. Conversion from free to paid dropped 23% because the cognitive load was too high.

**The key takeaway here is** that pricing changes without customer empathy backfire spectacularly. Grandfathering creates operational debt, blunt increases kill trust, and usage-based pricing confuses non-technical buyers. We needed a surgical approach that respected existing relationships while reflecting our cost reality.

## The approach that worked

We settled on a two-track strategy: a quiet increase for new customers and a grandfathered path for existing ones. For new customers, we raised Starter to $59 and Growth to $249. We kept Enterprise at $4,000 but added a "Scale" tier at $1,200 for mid-market customers. The trick was making the increase feel inevitable without making it personal.

We communicated the change via in-app banners and email, not blog posts. The banner read: "We’ve increased prices to fund infrastructure improvements that keep your data fast and reliable. Existing customers: your current plan won’t change for 12 months." The email included a one-click opt-out for free users and a discount code for existing paid users who wanted to upgrade early.

For enterprise prospects, we introduced a 30-day trial of the Scale tier. We saw a 40% lift in enterprise trial signups because the barrier to entry dropped from "$4k/year" to "free for 30 days." We also added a usage calculator: customers could input their monthly API calls and compute costs before signing up. This reduced pre-sales support tickets by 67%.

The pricing page itself was stripped back to three simple columns: Starter ($59), Growth ($249), Scale ($1,200), Enterprise (custom). No usage fields, no credits. The calculator lived on a separate page linked from the pricing page.

I got this wrong at first by assuming our customers were rational actors. They’re not—they’re emotional. A $10 increase felt like betrayal to a bootstrapper who’d been on Starter since 2020, but the same $10 felt like fair value to a new customer evaluating alternatives. The key was decoupling the emotional reaction from the rational justification.

**The key takeaway here is** that pricing changes work when they feel fair to both sides. New customers accept increases as the cost of doing business; existing customers accept them when they’re framed as a time-limited grace period, not a punishment. The calculator reduced cognitive load, and the trial removed friction—two principles I should have followed from day one.

## Implementation details

We built the pricing change in three sprints over six weeks. Sprint 1 focused on the pricing page redesign. We used Webflow for the marketing site, so we built a new pricing template with conditional logic: logged-out users saw the new tiers; logged-in users saw their existing tier with a 12-month countdown to change.

Sprint 2 was the billing migration. We used Stripe’s pricing table API to dynamically render the new prices while keeping the old prices in the database. We added a `price_override` field to the customer record so existing customers could keep their old price until renewal. The code snippet below shows how we handled this in Python:

```python
import stripe
from django.db import models

class Customer(models.Model):
    stripe_customer_id = models.CharField(max_length=255)
    price_override = models.CharField(max_length=50, null=True)
    override_expires = models.DateField(null=True)


def get_stripe_price(customer):
    if customer.price_override and customer.override_expires > timezone.now().date():
        return customer.price_override
    return "price_new_starter"  # default to new price
```

Sprint 3 was the calculator. We built a React component that pulled real-time pricing from our API. The endpoint returned a JSON object with tiers and usage limits:

```javascript
// GET /api/v1/pricing
{
  "tiers": [
    {
      "name": "Starter",
      "price": 59,
      "api_calls": 50000,
      "storage_gb": 50,
      "compute_hours": 50
    },
    {
      "name": "Growth",
      "price": 249,
      "api_calls": 500000,
      "storage_gb": 250,
      "compute_hours": 200
    }
  ]
}
```

We deployed the calculator on a subdomain (pricing.ourproduct.com) to avoid bloating the main marketing site. The calculator used debounced input to avoid excessive API calls—only sending a request after 500ms of inactivity. This kept the page lightweight even for users on 2G connections.

We also added a feature flag to control the rollout. Only 20% of new visitors saw the new prices for the first week. After analyzing conversion rates, we ramped to 100%. This let us catch edge cases like customers on legacy Stripe plans whose invoices would break if we changed their price object mid-cycle.

**The key takeaway here is** that pricing changes are infrastructure changes. They require the same rigor as database migrations: backups, rollback plans, and feature flags. The calculator wasn’t just a marketing tool—it was a support saver. It reduced the number of tickets asking "how many API calls do I get?" from 12/day to 2/day.

## Results — the numbers before and after

We launched the new pricing on March 15, 2023. By June 15, we had these results:

| Metric | Before (Jan–Mar 2023) | After (Mar–Jun 2023) |
|---|---|---|
| New customer ARPA | $118 | $147 |
| New customer conversion (free → paid) | 4.2% | 3.8% |
| Existing customer churn | 11% (Starter) | 4% (Starter) |
| Support tickets about pricing | 42/week | 8/week |
| Enterprise trial signups | 12/month | 17/month |
| Gross revenue | $14,800 | $18,200 |

The most surprising result was the stability of free-to-paid conversion. We expected a drop because of the price increase, but it only fell 0.4%. The calculator helped—users who tried it were 2x more likely to upgrade. Free users who used the calculator converted at 7.1%, vs 3.2% for those who didn’t.

Existing customer churn dropped from 11% to 4% because we grandfathered them. Only 3 customers out of 79 grandfathered Starter users churned—all due to unrelated business closures. The 12-month grace period gave them time to budget for the increase, and the in-app banner made the change feel less abrupt.

Our infra costs per customer fell from $18 to $14 because the new revenue let us invest in multi-tenant optimizations. We consolidated 8 PostgreSQL databases into 2, reducing idle compute from 40% to 12%. The savings offset 60% of the price increase for us, while customers got faster queries and lower latency.

**The key takeaway here is** that a 15% price increase can be a net win when it’s paired with operational improvements. The churn reduction alone justified the change—we saved $8,400/year in customer acquisition costs by keeping 7 more customers than we would have without grandfathering. The calculator was the unsung hero, turning a support sink into a revenue driver.

## What we’d do differently

If we did this again, we’d start with segmentation. We assumed all Starter customers were bootstrappers, but 34% were actually small agencies using our product as a white-label tool. They resold our API to their clients, so a $10 increase per seat multiplied across their customer base. Next time, we’d offer a Partner tier at $99 with usage-based overages, not a flat increase.

We’d also automate the grandfathering logic. Right now, the `price_override` and `override_expires` fields are updated manually via SQL scripts. One customer slipped through the cracks because their anniversary date was miscalculated—we had to issue a $120 credit. A cron job that runs monthly would prevent this.

The calculator was a success, but it added complexity. We built it in React, which required a separate deployment pipeline. Next time, we’d use a no-code tool like PricingBot or ProfitWell’s pricing page generator to avoid the dev overhead. The trade-off wasn’t worth the engineering hours—our calculator took 40 hours to build and maintain.

Finally, we’d test the messaging more rigorously. The banner said "infrastructure improvements," but we didn’t quantify what that meant. Customers wanted specifics: "Your data will be 30% faster and 99.9% more reliable." Next time, we’d include a latency benchmark or uptime SLA to justify the increase.

**The key takeaway here is** that pricing changes reveal hidden assumptions about your customer base. Segment before you raise prices, automate the grandfathering logic, and avoid over-engineering the calculator. Simplicity wins when the goal is trust, not features.

## The broader lesson

Pricing isn’t about numbers—it’s about narratives. The customers who left after our first increase weren’t rejecting the $30 bump; they were rejecting the story we told about why it happened. We said "we need to invest in reliability," but they heard "we’re nickel-and-diming you." The difference between a 15% price increase that works and one that backfires is the story you attach to it.

This principle applies beyond SaaS. I’ve seen hardware companies fail when they raised prices without explaining the bill of materials increase. I’ve seen agencies fail when they raised rates without showing the new designer credentials. The mechanism—banner, email, calculator—matters less than the narrative.

Another lesson: pricing is a proxy for value. When we added the calculator, we weren’t just reducing support tickets—we were giving customers a way to quantify the value they got from our product. That’s why conversion didn’t drop: the calculator turned an emotional decision ("is this worth it?") into a rational one ("this is worth 147 bucks").

**The broader lesson is** that pricing is storytelling with numbers. The best pricing pages don’t just list tiers—they tell a story about why the price is fair. The best price increases don’t just raise numbers—they raise the customer’s confidence in your ability to deliver value.

## How to apply this to your situation

Start by segmenting your customers. Use RFM analysis (recency, frequency, monetary) to identify your most profitable cohorts. If 60% of your revenue comes from 20% of customers, those are the ones you protect with grandfathering. If 40% of your churn comes from a single pricing tier, that tier is where you experiment with new pricing.

Next, pick a mechanism that feels fair. For bootstrapped customers, a usage-based tier works. For enterprise prospects, a trial or pilot works. For mid-market, a freemium-to-paid funnel works. The mechanism should match the customer’s buying style, not your internal preference.

Then, automate the grandfathering. Use your billing provider’s metadata fields to store override prices and expiry dates. Schedule a monthly job to clean up expired overrides—this prevents the SQL-script mess we created.

Finally, measure everything. Track new customer ARPA, churn by cohort, and support tickets about pricing. If churn spikes 5% after a change, roll it back immediately. If conversion drops 2%, investigate before assuming it’s the price.

**To apply this today, run this SQL query on your billing database:**

```sql
SELECT 
    pricing_tier,
    COUNT(*) as customers,
    AVG(revenue_monthly) as arpa,
    AVG(churned_in_last_90_days) as churn_rate
FROM customers 
WHERE active = true
GROUP BY pricing_tier;
```

This query will tell you which tiers are profitable and which are churn risks. If a tier has high churn and low ARPA, it’s a candidate for a price increase—or a complete redesign.

## Resources that helped

1. **ProfitWell’s 2023 Pricing Page Teardown** – Showed us how Slack and Notion structure their pricing pages to reduce cognitive load. We copied their three-column layout almost verbatim.
2. **Stripe’s pricing table API docs** – The conditional logic examples saved us 12 engineering hours. We used their `lookup_key` parameter to map old prices to new ones without breaking invoices.
3. **PricingBot (by Baremetrics)** – We considered it for the calculator, but their React component was too heavy for our needs. Still, their pricing page generator is excellent for non-technical teams.
4. **"Don’t Make Me Think" by Steve Krug** – Reminded us that pricing pages are UX, not marketing. Every extra field costs conversions.
5. **ProfitWell’s podcast episode "The Psychology of Pricing"** – Changed how we framed the increase. We stopped saying "we need to raise prices" and started saying "we’re investing in reliability to keep your data fast."

## Frequently Asked Questions

**How do I announce a price increase without losing customers?**

Announce it via in-app banners and emails, not blog posts. Frame it as an investment in reliability or features, not a cost-cutting measure. Offer a grace period for existing customers—12 months is standard. For example, say: "We’ve increased prices to fund infrastructure improvements. Your current plan won’t change for 12 months."

**What’s the difference between value-based pricing and cost-based pricing?**

Cost-based pricing starts with your costs and adds a margin. Value-based pricing starts with the customer’s perceived value and sets a price they’re willing to pay. We used cost-based pricing initially (Starter at $49 because our costs were $4/month), which left money on the table. We switched to value-based pricing by anchoring Growth at $249, which matched competitors like Retool.

**Why does my pricing page have a 17% lower conversion after the change?**

Check if the new page has more cognitive load: extra fields, usage calculators, or unclear tiers. We saw a 23% drop when we added a calculator to the main page. Move the calculator to a separate page and link it from the pricing page to reduce friction.

**How do I grandfather customers without breaking my billing system?**

Use your billing provider’s metadata fields to store override prices and expiry dates. For example, in Stripe, set `metadata['price_override'] = 'price_new_starter'` and `metadata['override_expires'] = '2024-03-15'`. Then, modify your invoicing logic to check these fields before generating an invoice.

## Implementation checklist

This checklist will take you from idea to launch in four weeks. Each item takes 1–3 days depending on your stack.

- [ ] Run RFM analysis on your customer base (SQL query above).
- [ ] Pick a tier to increase and a grace period length (12 months is standard).
- [ ] Design a new pricing page with no more than three tiers and one call-to-action.
- [ ] Build a calculator on a separate page with real-time pricing via API.
- [ ] Set up feature flags to roll out the new page to 20% of visitors first.
- [ ] Add metadata fields to your billing database for grandfathering.
- [ ] Write the in-app banner and email copy using the "investment" framing.
- [ ] Schedule a monthly job to clean up expired overrides.
- [ ] Measure new customer ARPA, churn by cohort, and support tickets.
- [ ] Roll back if churn spikes >5% or conversion drops >3%.

Follow this checklist, and you’ll raise prices without losing customers—or your sanity.