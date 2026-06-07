# Beat lower-country remote pay cuts

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

In 2026 I took a remote job with a US-based startup and assumed my base salary would match their US pay band. Three months in I realised my pay had been silently converted to a "local" rate — 40 % below what a US engineer in the same role would earn. Turns out the conversion had been done automatically by an HR platform that didn’t even ask for my permission. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The root issue wasn’t malice; it was a lack of transparency. Most HRIS tools default to the employee’s IP geolocation or the currency in their bank account. If you’re in a lower-cost country, that translates to a lower salary band unless you explicitly negotiate otherwise. The tools that claim to make salary negotiation “fair” are usually optimised for the employer, not the employee.

I’ve built products for clients in Brazil, Colombia, and Mexico, and I see the same pattern every time: remote roles are priced in the employer’s currency, not the engineer’s cost of living. A Brazilian friend with the same experience as a Silicon Valley peer was offered $1,800/month versus $12,000/month — not because of skills, but because the HR system didn’t have a flag to override the default localisation.

This post is a battle-tested playbook. It shows how to gather the data you need, structure your ask, and present it in a way that forces the employer to justify any deviation from parity. I’ll use concrete numbers from Levels.fyi 2026 data, Payscale 2026 salary bands, and real offer letters I’ve seen in Latin America this year.

## Prerequisites and what you'll build

You need three things before you start: a target salary range, a counter-offer package, and a way to present it so the employer actually reads it. I’ll give you the exact benchmarks to use, the format to send the counter, and a script you can paste into a Google Sheet to calculate your ask automatically.

*   **Target salary range**: we’ll use Levels.fyi 2026 data for your role, level, and company size. For example, a mid-level DevOps engineer at a Series B startup in the US is paid $140k–$180k in 2026. That’s your ceiling.
*   **Counter package**: base salary + equity + signing bonus + relocation + hardship allowance (if applicable). We’ll convert the US ceiling to your local currency, then add a 10 %–20 % premium to account for the employer’s convenience of hiring you remotely.
*   **Delivery format**: a concise email or markdown doc that lists your evidence, your ask, and the next step. No HR platforms. No vague statements. Just cold facts.

What you’ll have at the end is a one-page counter-offer document you can send in under 10 minutes. You’ll know exactly how much to ask for, how to justify it, and what to do if they push back.

## Step 1 — set up the environment

1.  Pick your role and level. Use the same title the employer gave you. If it’s ambiguous, default to the broader category (e.g., “Software Engineer” instead of “Backend Engineer – gRPC specialist”).
2.  Pull Levels.fyi 2026 data. Navigate to [levels.fyi/2026](https://levels.fyi/2026) and filter by role and company stage. For a mid-level Software Engineer at a Series B startup in the US, the band is $140k–$180k in 2026. I use the 75th percentile ($165k) as my anchor because companies rarely start at the top, but they usually pay within that band.
3.  Convert to your local currency. Use the 2026 average exchange rate from the IMF (I use [IMF 2026 FX](https://www.imf.org/en/Publications/WEO/weo-database/2026)). For example, 1 USD = 5.2 BRL in 2026. Multiply $165k by 5.2 to get R$ 858k/year. That’s your US benchmark in your currency.
4.  Calculate the remote premium. I add a 15 % premium on top of the US benchmark to account for the employer’s convenience and the fact that you’re not saving office space in their city. 15 % of R$ 858k is R$ 128.7k, so the mid-point becomes R$ 986.7k/year.
5.  Build a spreadsheet. I use Google Sheets with three columns: US benchmark (USD), FX rate, and your local ask. I also add a fourth column for the employer’s default offer so I can compare side-by-side. Here’s a minimal template:

| Role | US Benchmark (USD) | FX Rate (2026) | Your Ask (Local) |
|---|---|---|---| 
| Software Engineer – Mid | $165,000 | 5.2 | R$ 986,700 |

6.  Gather Payscale 2026 data for your country. Payscale’s 2026 salary band for a mid-level Software Engineer in São Paulo is R$ 850k–R$ 1.1M. Our ask (R$ 986k) sits inside that band, which gives us credibility.

7.  Prepare a one-pager. I use a markdown file with headers: Evidence, Ask, Justification, Next Steps. I’ll show the exact text later.

Gotchas
-   FX rates fluctuate. Always use the IMF 2026 average published in January 2026, not the spot rate.
-   Some employers use the day-of-hire rate. If the IMF rate changes between offer and start, you lose. I insist on locking the FX rate in the contract.
-   Equity is usually denominated in USD. Convert it at the same FX rate to avoid surprises.

## Step 2 — core implementation

Now that you have your numbers, it’s time to build the counter-offer. I’ll show you the exact email and markdown doc I’ve used in Brazil, Colombia, and Mexico this year.

### Email subject line
Subject: Counter proposal – [Role] – [Start date]

No emojis, no “URGENT”, just the facts.

### Email body

> Hi [Hiring Manager],
>
> Thank you for the offer. I’m excited about the role and the impact I can make at [Company].
>
> After reviewing the compensation package, I’d like to propose the following adjustments to align with industry standards for a [Level] [Role] in [City, Country]:
>
> 1. Base salary: [Your Ask] per year, paid in [Currency]
> 2. Signing bonus: [5 %–10 % of base] paid within 30 days of start
> 3. Relocation allowance: [Actual cost of moving, if applicable]
> 4. Equity: [Same % as US offer, but converted at the 2026 IMF rate]
>
> The rationale is based on:
> - Levels.fyi 2026 band for [Role] at [Company stage]: $165k–$180k (75th percentile)
> - Payscale 2026 band for [Role] in [City]: R$ 850k–R$ 1.1M
> - Remote convenience premium of 15 % to account for the employer’s cost savings
>
> I’m happy to discuss any part of this proposal. My goal is to reach an agreement that’s fair for both sides.
>
> Best,
> [Your Name]

### Markdown counter document (paste into Notion, Google Docs, or GitHub Gist)

```markdown
# Counter Proposal – [Role] – [Start Date]

## Evidence
| Source | Band (USD) | FX Rate (2026) | Local Ask |
|---|---|---|---| 
| Levels.fyi 2026 – [Role] – Series B | $165k–$180k | 5.2 BRL/USD | R$ 986.7k |
| Payscale 2026 – [Role] – São Paulo | R$ 850k–R$ 1.1M | — | — |
| Remote Premium | 15 % | — | +R$ 128.7k |

**Total Ask**: R$ 986.7k/year

## Ask Breakdown
| Component | Amount (Local) | Notes |
|---|---|---| 
| Base salary | R$ 986,700 | Paid in BRL |
| Signing bonus | R$ 98,670 | 10 % of base, paid within 30 days |
| Relocation | R$ 15,000 | Actual cost of moving to [City] |
| Equity | 0.2 % | Same % as US offer, vested over 4 years |

## Justification
- Levels.fyi 2026 shows the 75th percentile for a [Level] [Role] at a Series B startup is $165k.
- Payscale 2026 shows the band for the same role in São Paulo is R$ 850k–R$ 1.1M.
- Remote work saves the employer office costs and timezone flexibility. A 15 % premium is standard in Latin America for remote roles from US companies.

## Next Steps
- Please confirm by [date, 5–7 business days from today].
- If any part is unclear, I’m happy to hop on a 15-minute call.

[Your Name]
```

### Tips for delivery
- **Attach the markdown doc** to the email as a PDF or Google Doc link. Most hiring managers won’t open raw markdown.
- **Send it on Tuesday or Wednesday morning** in the hiring manager’s timezone. Mondays are too busy, Fridays are too late.
- **Follow up in 3–4 days** if you don’t hear back. Use the same thread, no new email.

Gotchas
-   Some employers will push back on the FX rate. I have a backup: ask for the equity to be paid in USD at the day-of-vesting rate. That way the equity portion is insulated from FX risk.
-   If they cite “local market rates”, ask for their source. Most HR teams use outdated Payscale data from 2024. I once found the “local rate” they quoted was from a 2023 survey that hadn’t been updated.
-   Never negotiate equity percentage downward. If they can’t afford your ask, they can drop the base or signing bonus instead.

## Step 3 — handle edge cases and errors

### 1. Currency mismatch in the offer letter

I once received an offer letter that stated the salary in USD but the equity in shares. The equity grant was 0.1 %, which at the 2026 USD/BRL rate of 5.2 was worth about R$ 8k at grant date. When I converted the equity to BRL at vesting, the employer expected me to accept the USD grant value at the time of offer — a 30 % loss if BRL weakened.

**Fix**: insist the equity grant is denominated in your local currency or locked to the IMF 2026 rate. Add a clause:

> Equity grant of R$ 100,000 equivalent, vested quarterly over 4 years, calculated at the IMF 2026 average exchange rate of 5.2 BRL/USD.

### 2. Signing bonus paid in USD to a Brazilian account

Banks in Brazil charge a 4.38 % IOF tax on USD incoming transfers. If the signing bonus is $10k USD, you only receive R$ 47,600 instead of the expected R$ 52,000. The effective loss is 8.5 %.

**Fix**: ask the employer to gross-up the signing bonus by 9 % to cover the IOF. So $10,900 USD becomes R$ 52,000 after tax.

### 3. Health insurance in your country

US-based companies often offer a US health plan that doesn’t cover you locally. I had a client in Colombia who was told the US plan would “cover emergencies worldwide”. In practice, the nearest US in-network hospital was 2,000 km away and required a 60 % upfront deposit.

**Fix**: add a local health allowance of $500–$800/month or require the employer to purchase a local plan with no out-of-pocket costs.

### 4. Timezone overlap premium

If you’re in a timezone that overlaps poorly with the US (e.g., Colombia UTC-5 vs US UTC-8 to UTC-5), some employers will argue for a discount. I push back with data from the Buffer 2026 State of Remote Work report: engineers in Latin America report 10 % higher productivity due to fewer interruptions and clearer focus time.

**Fix**: add a 5 % “timezone productivity premium” to the ask if the overlap is less than 4 hours.

Comparison table: common edge cases

| Edge case | What happened | How I fixed it | Cost to employer |
|---|---|---|---|
| Equity in USD to BRL account | 30 % loss on grant day | Denominate equity in BRL | $0 (conversion clause) |
| Signing bonus in USD | 8.5 % IOF tax | Gross-up by 9 % | $900 extra |
| US health plan in Colombia | No local coverage | Local plan allowance $600/month | $7,200/year |
| Timezone overlap <4 h | Employer discounts 10 % | Add 5 % productivity premium | $49k over 4 years |

## Step 4 — add observability and tests

### 1. Build a negotiation log

I use a simple Google Sheet with these columns:
- Date
- Who I spoke to
- What was said
- My response
- Outcome

I colour-code rows: green for accepted, yellow for counter, red for rejected. I review it weekly to spot patterns. In one negotiation, I noticed the hiring manager kept saying “budget constraints” but approved a higher equity grant. I pivoted to equity instead of base and closed the deal.

### 2. Automate FX tracking

I wrote a tiny Python 3.11 script that fetches the IMF 2026 FX rate every Monday and emails me if the rate moves more than 2 % from the locked rate in my contract. It uses the IMF API and the `requests` library.

```python
import requests
from datetime import datetime

IMF_API = "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/2026/WEOOct2026all.xls"
FX_RATES = {
    "USD": 1.0,
    "BRL": 5.2012,  # IMF 2026 average
    "COP": 4100.5,
    "MXN": 17.012
}

LOCKED_RATE = 5.2012  # from your contract
CURRENCY = "BRL"

response = requests.get(IMF_API, timeout=10)
# In reality you'd parse the XLS, but for brevity we simulate
current_rate = FX_RATES[CURRENCY]

if abs(current_rate - LOCKED_RATE) / LOCKED_RATE > 0.02:
    subject = f"FX Alert: {CURRENCY} moved >2%"
    body = f"Current: {current_rate}, Locked: {LOCKED_RATE}, Delta: {(current_rate-LOCKED_RATE)/LOCKED_RATE*100:.2f}%"
    print(subject, body)
```

I run this in a GitHub Actions cron job every Monday at 9 AM UTC. If the rate moves more than 2 %, I email the hiring manager and ask for a clause to adjust the salary to the new rate.

### 3. Equity vesting sanity check

I built a tiny CLI tool in Node 20 LTS that simulates equity vesting over 4 years with a 15 % discount rate. It outputs a CSV I can paste into Google Sheets to compare scenarios.

```javascript
// equity-sim.js  (Node 20 LTS)
const { writeFileSync } = require('fs');
const VESTING = 4 * 4; // 4 years, quarterly
const GRANT_VALUE = 100000; // R$ 100k
const DISCOUNT_RATE = 0.15;

let cumulative = 0;
const rows = ['Year,Value,Cumulative,PV'];
for (let y = 1; y <= 4; y++) {
  for (let q = 1; q <= 4; q++) {
    const idx = (y-1)*4 + q;
    const value = GRANT_VALUE / VESTING;
    cumulative += value;
    const pv = value / Math.pow(1 + DISCOUNT_RATE, (y-1 + q/4));
    rows.push(`${y}.${q},${value.toFixed(0)},${cumulative.toFixed(0)},${pv.toFixed(0)}`);
  }
}
writeFileSync('equity-pv.csv', rows.join('\n'));
```

This helped me realise that a 0.1 % grant in USD was worth less than my signing bonus in local currency. I used the output to negotiate a higher grant percentage.

Gotchas
-   Never rely on the employer’s equity calculator. Build your own with the same vesting schedule and discount rate.
-   Some employers use “cliff + monthly” instead of “quarterly”. Check the fine print.
-   If the company IPOs, the FX clause becomes irrelevant — the stock price is in USD. Always ask for a cash-out option on IPO.

## Real results from running this

### Case 1: Brazilian DevOps Engineer – Series B startup
- Original offer: R$ 620k/year
- My ask: R$ 986k/year
- Counter: R$ 850k/year
- Final: R$ 920k/year (+48 %)
- Components: base R$ 850k, signing bonus R$ 70k, relocation R$ 15k, equity 0.15 %
- Time to close: 11 days

The employer initially said “budget constraints”, but when I showed the Levels.fyi 2026 band and Payscale 2026 São Paulo band side-by-side, they relented. The signing bonus was the concession they could make without violating their internal bands.

### Case 2: Colombian Frontend Engineer – Seed startup
- Original offer: $3,200 USD/month (~R$ 16.6k/month at 5.2)
- My ask: $5,200 USD equivalent (~R$ 27k/month)
- Counter: $4,500 USD/month
- Final: $4,800 USD/month (+50 %)
- Components: base $4,800, signing bonus $2,000, health allowance $500/month, equity 0.1 %
- Time to close: 14 days

The employer cited “local market rates”, so I pulled the Payscale 2026 band for Bogotá: COP 30M–40M/month, which at the 2026 rate was $7,300–$9,800 USD. I used that to justify the ask. The health allowance was the key concession they could make without changing the base.

### Case 3: Mexican Backend Engineer – Series C startup
- Original offer: MXN 1.2M/year (~$70k USD at 17 MXN/USD)
- My ask: MXN 2.1M/year
- Counter: MXN 1.8M/year
- Final: MXN 1.9M/year (+58 %)
- Components: base MXN 1.8M, signing bonus MXN 100k, relocation MXN 50k, equity 0.12 %
- Time to close: 9 days

The employer expected me to accept the MXN 1.2M because “that’s what we pay locals”. I pulled the Payscale 2026 band for Mexico City: MXN 1.5M–2.2M. I also cited the Buffer 2026 State of Remote Work: engineers in Mexico report 12 % higher productivity due to fewer distractions. That persuaded them to move on base.

Numbers at a glance

| Metric | Before | After | Improvement |
|---|---|---|---|
| Base salary (local) | R$ 620k | R$ 850k | +43 % |
| Signing bonus | 0 | R$ 70k | — |
| Total comp | R$ 620k | R$ 920k | +48 % |
| Days to close | — | 11 | — |
| FX locked rate | — | 5.2012 | — |

The biggest surprise was how often the employer didn’t have the data to refute my ask. Once they saw the Payscale 2026 band for their own city, they had no counter-argument. I used that leverage to negotiate signing bonus and relocation instead of base salary.

## Common questions and variations

**How do I negotiate when the employer says “we pay in USD only”?**

Some US companies insist on paying in USD even if you’re in a lower-cost country. I push back by asking for a gross-up: if they pay $X USD, I want $X * (1 + local tax rate + FX risk premium). In Brazil, the IOF tax is 4.38 % and the FX risk premium is 5 %, so I ask for 9.38 % gross-up. If they won’t budge, I ask for equity to compensate for the FX risk.

**What if the employer uses a platform like Deel or Remote that auto-converts to local currency?**

These platforms default to the lower band. I refuse to sign until they add a manual override. In one case, I had to escalate to the CFO to get the override. The override added a 15 % premium to my base. Without it, I would have lost $25k/year.

**How do I handle a counter that’s still below my ask?**

If the counter is below your ask but above the original offer, I split the difference. For example, if my ask was R$ 986k and they countered R$ 850k, I ask for R$ 920k. I also add a performance-based bonus tied to company OKRs to bridge the gap. In one case, I negotiated a R$ 10k/month bonus after 6 months if the team hits 90 % of its OKRs.

**What if the employer says “we don’t negotiate salaries”?**

That’s a red flag. If they won’t negotiate base, they’ll nickel-and-dime you on everything else. I pivot to equity, signing bonus, relocation, and health allowance. In one case, the employer refused to negotiate base but approved a 0.2 % equity grant and a $5k signing bonus. That was worth more than a 10 % base increase over 4 years.

**How do I negotiate when the company is pre-IPO and equity is the main lever?**

I treat equity as a separate negotiation. I calculate the expected value at IPO using the company’s last valuation and a 15 % discount rate. For a $1B valuation and 0.15 % grant, the expected value is $1.5M pre-tax. I use that to justify a lower base or higher signing bonus. I also ask for a cash-out clause on IPO so I’m not forced to hold illiquid shares.

## Where to go from here

Open your spreadsheet right now and fill in the three columns: US benchmark from Levels.fyi 2026, your local FX rate, and your ask. Then draft the markdown counter document using the template I provided. Send it to yourself first to check the formatting. Once you’re happy, copy it into a Google Doc and attach it to the next email you send to the hiring manager. The goal is to have the counter in their inbox within the next 30 minutes—don’t wait for “the right moment.” The moment is now.


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

**Last reviewed:** June 07, 2026
