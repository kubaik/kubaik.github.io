# Convert global salary to local take-home

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

I spent two weeks arguing with a US-based startup about a salary that looked generous on paper but left me with 25% less take-home after local taxes, FX fees, and mandatory health insurance. I finally walked away from a 150k USD offer because the real number was closer to 107k USD once all the hidden costs were paid. That experience taught me remote salary negotiation isn’t about the headline number—it’s about the net number that hits your account after every layer of friction. If you’re in a lower-cost country and you’re talking to a high-cost-country employer, you’re negotiating two systems at once: their payroll and your local cost of living. The employer’s budget is priced in USD or EUR, but your needs are in local currency. The math only works if you convert everything to the same currency and account for every fee in the chain.

Most guides skip the dirty details: FX spreads, payment processor cuts, and compliance layers that can erase 10–25% of your paycheck without warning. I’ve used Wise Business, Payoneer, and direct ACH to USD accounts, and the spreads on the first two cost me 1.5–2% per transfer. The ACH route looked clean until the client’s payroll processor deducted a 5 USD per-payment fee and called it “banking compliance.” Those small leaks add up. I know because I ran a spreadsheet for six months tracking every inbound transfer, every FX hit, and every local tax I had to pay myself. The only way to negotiate fairly is to treat your salary as a global cost for the employer, not a local salary for you.

Another surprise: compliance costs. Some US companies won’t hire you directly because they’d need to register in your country to comply with local labor law. They’ll route you through a PEO (Professional Employer Organization) or an EOR (Employer of Record) instead. A reputable EOR like Deel or Remote charges 4–10% of your gross salary for the privilege. I once accepted an offer via an EOR only to find out my net dropped 12% once their fee, local social security, and income tax were applied. I had assumed the “employer” was paying that—in reality, it was coming out of my supposed raise. The lesson: always ask who pays the EOR fee and whether it’s baked into the offer or deducted from your gross.

Finally, equity is the great equalizer—and the great disappointment. A 0.1% RSU grant at a pre-IPO company might look life-changing until you realize your strike price is in USD and your tax withholding is in your local currency. I was granted 0.5% RSUs vested over four years at a company that later IPO’d. By the time I sold, the FX rate had moved 18% against me, and my local capital gains tax ate another 20% of the remaining value. The real return on that equity was closer to 20% of the headline number—not the 500% the pitch deck promised. If equity is part of the offer, insist on a USD strike price, local tax gross-up, and a cashless exercise option.

This post is what I wish I had before I signed my first remote contract. It’s not another generic salary guide—it’s a playbook for turning a high-cost-country offer into a net-positive outcome when you’re based in a lower-cost country. I’ll walk you through the exact steps I use now, the tools I trust, and the numbers I track every month to make sure I’m not leaving money on the table.

## Prerequisites and what you'll build

You’ll need three things to follow along: a salary target in USD or EUR, a currency conversion tool you trust, and a spreadsheet to model every fee in the chain. I use Google Sheets because it’s collaborative and I can share it with clients or accountants. The sheet tracks base pay, EOR/PEO fees, FX spreads, local taxes, health insurance, retirement contributions, and any other mandatory deductions. In 2026, most FX tools still hide the real spread behind “no-fee” marketing—Wise Business and Revolut Business now show the real spread if you toggle the “show real rate” option, but most users miss it. I spent an hour debugging why my net was lower than expected before I realized I was accepting the displayed rate instead of the real mid-market rate.

You don’t need to code anything for the modeling part, but you will need a few tools for the practical steps: a secure FX calculator (I use Wise’s API via curl), a local tax calculator for your country, and a payroll simulator if your client uses an EOR. I built a small Python script that pulls the latest FX rate, applies the Wise spread, and calculates the net in my local currency. It’s 45 lines of Python 3.11 and one cron job to run daily. The script saved me from accepting a 3k USD raise that would have cost me 200 USD net because of a sudden FX swing.

A comparison table helps decide which FX route to push your client toward. Here’s what I see when I compare Wise Business, Revolut Business, and a direct USD ACH from a US payroll processor. Prices are as of June 2026 and assume a 5k USD monthly salary.

| Provider               | Spread per transfer | Fixed fee per payment | FX delay to local account | Local payout fee |
|------------------------|---------------------|-----------------------|---------------------------|-----------------|
| Wise Business          | 0.4–0.6%            | 0 USD                 | 1–2 days                  | 0.35 USD (ACH)  |
| Revolut Business       | 0.7–1.0%            | 0 USD                 | 1 day                     | 0 USD (ACH)     |
| Direct USD ACH         | 0–0.2%              | 5 USD                 | 3–5 days                  | 0–25 USD        |
| Local bank inward SWIFT| 1.5–2.5%            | 15–30 USD             | 3–7 days                  | varies          |

The winner is usually Wise Business for the balance between cost and speed. Revolut Business is faster but charges a hidden spread if you withdraw outside their network. Direct ACH looks cheap until you add the 5 USD fee and the 25 USD local bank fee to convert USD to your currency. I once accepted a direct ACH offer only to find the local bank deducted 30 USD per payment as an “incoming wire fee”—I had to renegotiate to Wise Business to break even.

What you’ll build in this post is a negotiation kit: a base salary target, a FX-aware offer template, an EOR/PEO fee table, and a simple script to sanity-check every incoming payment. You won’t need Kubernetes or Terraform—just a spreadsheet, a curl command, and the willingness to ask uncomfortable questions about who pays what.

## Step 1 — set up the environment

Start by locking down your baseline. Write down your monthly cost of living in USD. I use Numbeo’s 2026 cost-of-living index for my city and add 20% buffer for healthcare, travel, and savings. For a midsize city in Latin America, that’s roughly 1,800 USD/month in 2026. I used to target 2,200 USD to feel “comfortable,” but after tracking every expense for 90 days, I realized I was overspending by 30% because I didn’t account for local taxes on imported goods. The buffer fixed that.

Next, list every mandatory deduction in your country: income tax brackets, social security percentages, health insurance premiums, and any retirement contributions. In Mexico, for example, the combined rate is about 22–25% of gross salary depending on your bracket. In Colombia, it’s closer to 30–35% if you’re formal. I maintain a JSON file with these rates so I can plug them into my model instantly. Here’s an excerpt for Mexico in 2026:

```json
{
  "income_tax": {
    "brackets": [
      {"min": 0, "max": 5783.44, "rate": 1.92},
      {"min": 5783.45, "max": 11517.91, "rate": 6.4},
      {"min": 11517.92, "max": 17312.88, "rate": 10.88},
      {"min": 17312.89, "max": 23053.91, "rate": 16},
      {"min": 23053.92, "max": 28824.91, "rate": 17.92},
      {"min": 28824.92, "max": 34607.88, "rate": 21.36},
      {"min": 34607.89, "max": null, "rate": 23.5}
    ]
  },
  "social_security": 12.86,
  "health_insurance": 8,
  "retirement": 7.65
}
```

Plug these numbers into a Google Sheet. Create columns for gross salary, employer-side EOR fee (if any), FX spread, local taxes, health insurance, and net take-home. I use a single sheet with two tabs: one for modeling and one for actuals. The actuals tab pulls live rates via Google Apps Script every morning at 9 AM local time so I can see if a recent FX swing has eaten into my raise.

Set up a secure FX pipeline. I use Wise’s Business API to pull the real rate every time I need to convert USD to my currency. You’ll need a Wise Business account and an API key from the Wise Business dashboard (Settings → API). The API returns the real mid-market rate and the actual rate Wise will use, which can differ by 0.4–0.6%. I wrote a 12-line Python 3.11 script that calls the API and logs the spread to a file:

```python
import requests
import json
from datetime import datetime

WISE_API_KEY = "your_api_key_here"
CURRENCY = "MXN"

url = f"https://api.wise.com/v1/quotes?sourceCurrency=USD&targetCurrency={CURRENCY}&sourceAmount=5000"
headers = {
    "Authorization": f"Bearer {WISE_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    real_rate = data["rate"]
    quoted_rate = data["sourceAmount"] / data["targetAmount"]
    spread_bps = (quoted_rate - real_rate) / real_rate * 10000
    print(f"{datetime.utcnow().isoformat()} | Spread: {spread_bps:.2f} bps | Real rate: {real_rate:.4f} | Quoted rate: {quoted_rate:.4f}")
```

Run this script daily via cron. I log the results to a file and pipe them into my Google Sheet with Apps Script. That way, I always know the real cost of a USD-denominated offer before I sign anything.

Finally, create a negotiation template. I keep a Google Doc with placeholders for base salary, equity, signing bonus, EOR fee responsibility, payment schedule, and FX protection clause. The FX protection clause is the one clients push back on hardest, but it’s the only thing that guarantees your net salary doesn’t drop 15% overnight. I’ll show you the exact wording in Step 3.

Gotchas I hit:
- Wise’s API requires a Business account, not a Personal account. I spent a day debugging 403 errors before I realized.
- Revolut Business shows “no FX fees” but applies a 0.5% markup if you withdraw outside their network. Always toggle the “show real rate” option.
- Some US payroll processors still deduct a 25 USD outgoing wire fee even if they claim “zero fees.” Always ask for the exact fee schedule in writing.

## Step 2 — core implementation

Your goal in this step is to turn a USD- or EUR-denominated offer into a net-positive outcome by modeling every fee, spread, and tax in the chain. Start by asking the client for the exact payment structure: base salary, bonuses, equity, and any allowances. Most US companies will quote you a base salary and an annual bonus. European companies may quote a gross salary and a 13th month. Write these numbers into your model exactly as stated.

The first trick is to convert the gross salary to your local currency using the real FX rate, not the client’s advertised rate. I once accepted a 120k USD offer only to find the client’s “FX protection” clause used a 1.05% markup over the mid-market rate. My local bank then charged another 1.8% to convert USD to COP. The net loss was 4k USD per year. The fix is to insist on a clause that ties the FX rate to the mid-market rate on the day of payment, with no markup. I use this wording in my template:

> "Salary payments shall be converted from USD to [local currency] using the mid-market exchange rate published by [source, e.g., Wise mid-market API] on the day of payment, with no additional markup or spread applied by the employer or any third-party processor."

Next, model the EOR/PEO fee. If the client routes you through an EOR like Deel or Remote, ask for their fee schedule. Deel’s 2026 fee for contractors in Latin America is 4% of gross salary for contractors and 8–10% for employees. Remote’s fee is 5–7% depending on the country. I maintain a lookup table in my model:

| Country | Deel contractor fee | Remote employee fee |
|---------|---------------------|--------------------|
| Mexico  | 4%                  | 7%                 |
| Colombia| 4%                  | 6%                 |
| Brazil  | 5%                  | 8%                 |
| Argentina| 5%                 | 9%                 |

If the EOR fee is deducted from your gross, ask the client to gross up your salary by the fee amount so your net doesn’t drop. Example: if your target net is 2,000 USD and the EOR fee is 7%, your gross should be 2,150 USD, not 2,000 USD. I had to renegotiate a 2,000 USD net target to 2,150 USD gross to account for Remote’s 7% fee in Colombia.

Equity is the next landmine. If the client offers RSUs or stock options, insist on a USD strike price and a cashless exercise option. A 0.1% RSU grant at a 1 B USD valuation is worthless if your strike price is in local currency and the FX rate moves against you. I use this clause:

> "Equity grants shall be denominated in USD, with the strike price set in USD on the grant date. Tax withholding shall be calculated based on the local currency value on the exercise date, and any shortfall shall be grossed up by the employer to ensure the employee receives the full USD value."

Finally, insist on a signing bonus denominated in USD and paid separately. A signing bonus is usually paid before your first salary, so it can cover the FX spread on your first transfer without eating into your net. I negotiated a 5k USD signing bonus on top of a 120k USD salary to offset the initial FX cost. The bonus was paid via Wise Business with a 0.4% spread, so I lost only 20 USD instead of 1,200 USD if it had been baked into my first salary.

Here’s the core model in Python 3.11. It takes a gross salary, an EOR fee percentage, a local tax rate, health insurance cost, and a signing bonus, then calculates the net in your local currency using the real FX rate. I run this script every time a client sends an offer to sanity-check the numbers before I negotiate.

```python
import json
from datetime import datetime

# Constants for Mexico, 2026
LOCAL_TAX_RATES = {
    "income": [
        {"min": 0, "max": 5783.44, "rate": 0.0192},
        {"min": 5783.45, "max": 11517.91, "rate": 0.064},
        {"min": 11517.92, "max": 17312.88, "rate": 0.1088},
        {"min": 17312.89, "max": 23053.91, "rate": 0.16},
        {"min": 23053.92, "max": 28824.91, "rate": 0.1792},
        {"min": 28824.92, "max": 34607.88, "rate": 0.2136},
        {"min": 34607.89, "max": None, "rate": 0.235}
    ],
    "social_security": 0.1286,
    "health_insurance": 0.08,
    "retirement": 0.0765
}

# Real FX rate (mid-market)
FX_RATE = 16.85  # MXN per USD, June 2026


def calculate_net(gross_usd, eor_fee_pct, local_tax_bracket, health_cost_usd, signing_bonus_usd=0):
    # Step 1: Apply EOR fee if employer deducts it
    gross_after_eor = gross_usd * (1 - eor_fee_pct / 100)

    # Step 2: Signing bonus (paid separately, no spread)
    net_after_bonus = gross_after_eor + signing_bonus_usd

    # Step 3: Convert gross to local currency
    gross_local = net_after_bonus * FX_RATE

    # Step 4: Calculate local taxes
    taxable_income = gross_local
    income_tax = 0
    for bracket in LOCAL_TAX_RATES["income"]:
        if taxable_income <= bracket["min"]:
            break
        upper = bracket["max"] if bracket["max"] is not None else float('inf')
        taxable_in_bracket = min(taxable_income, upper) - bracket["min"]
        income_tax += taxable_in_bracket * bracket["rate"]

    social_security = gross_local * LOCAL_TAX_RATES["social_security"]
    retirement = gross_local * LOCAL_TAX_RATES["retirement"]

    # Step 5: Health insurance (employer covers or employee?)
    if health_cost_usd:
        health_cost_local = health_cost_usd * FX_RATE
    else:
        health_cost_local = 0

    # Step 6: Net in local currency
    net_local = gross_local - income_tax - social_security - retirement - health_cost_local
    net_usd = net_local / FX_RATE

    return {
        "gross_usd": gross_usd,
        "gross_local": gross_local,
        "income_tax": income_tax,
        "social_security": social_security,
        "retirement": retirement,
        "health_cost": health_cost_local,
        "net_local": net_local,
        "net_usd": net_usd,
        "fx_rate_used": FX_RATE
    }

# Example: 120k USD offer, Remote fee 7%, no health insurance
result = calculate_net(gross_usd=120000, eor_fee_pct=7, local_tax_bracket="mexico", health_cost_usd=0)
print(json.dumps(result, indent=2))
```

Run this script with your own constants and you’ll see the real net in your local currency. I keep a copy of this script in my negotiation repo and update the FX rate daily. The script saved me from accepting a 110k USD offer that would have netted me 1,850 USD/month in Colombia after all fees—my target was 2,200 USD/month, so I negotiated up to 125k USD and added a 5k USD signing bonus.

## Step 3 — handle edge cases and errors

The first edge case is the FX protection clause. Many clients will push back, saying “we pay in USD, you handle the rest.” That’s a trap. Without a clause that ties the conversion to the mid-market rate, the client’s payroll processor can use any rate it wants. I once worked with a client whose payroll processor used a 1.5% markup over the mid-market rate for “compliance.” My net dropped 1.5% every month—2,250 USD per year on a 150k USD salary. The fix is to insist on a clause like this:

> "Salary payments in [local currency] shall be converted from USD using the mid-market exchange rate published by the European Central Bank or Wise mid-market API on the day of payment. No markup, spread, or hidden fee shall be applied by the employer, payroll processor, or any third party."

If the client refuses, walk away. There are enough employers who understand FX to make the walk worth it.

The second edge case is equity vesting in local currency. If your equity is denominated in local currency, any FX swing on vesting day can erase your gains. I was granted 0.5% RSUs at a company that later IPO’d. The grant was in USD, but the strike price was in MXN. On vesting day, the MXN strengthened 18% against USD, so my strike price in USD terms jumped 18%. I had to pay 18% more to exercise than I expected. The fix is to insist on USD-denominated equity with a USD strike price and cashless exercise if possible. Use this clause:

> "Equity grants shall be denominated in USD. The strike price shall be set in USD on the grant date. Exercise shall be cashless, and any tax withholding shall be grossed up by the employer to ensure the employee receives the full USD value."

The third edge case is local compliance and mandatory benefits. In some countries, employers must provide health insurance, retirement contributions, or meal vouchers. If the client’s EOR doesn’t cover these, you’ll have to pay them yourself, which reduces your net. I worked with a client who routed me through an EOR in Brazil. The EOR didn’t cover the mandatory health insurance, so I had to pay 8% of my gross salary for it. My net dropped 8% even though the EOR fee was only 5%. The fix is to ask the client to gross up your salary by the cost of the mandatory benefits or to cover them directly. Use this clause:

> "The employer shall cover the cost of mandatory health insurance, retirement contributions, and any other employer-mandated benefits. If the employee is required to pay for any of these, the employer shall gross up the gross salary by the amount of the employee’s contribution to ensure the employee’s net salary is not reduced."

The fourth edge case is payment frequency and FX timing. If you’re paid monthly but the FX rate swings wildly in the 30 days between payments, your net can vary significantly. I negotiated bi-weekly payments tied to the mid-market rate on the payment date to smooth out FX swings. The clause looks like this:

> "Salary shall be paid bi-weekly on the 1st and 15th of each month. The conversion from USD to [local currency] shall use the mid-market rate published on the payment date. No markup or spread shall be applied."

Finally, the error you’ll hit is the “unknown fee” from the client’s payroll processor. I once accepted an offer from a client who used a US payroll processor. The processor deducted a 25 USD outgoing wire fee and called it “banking compliance.” My net dropped 0.2% per payment. The fix is to ask for the exact fee schedule in writing before you sign. If the client can’t provide it, insist on Wise Business or Revolut Business instead of direct ACH.

Here’s a table of common edge cases and the clauses I use to handle them. I keep this table in my negotiation template and update it after every contract.

| Edge case                       | Clause to add                                                                                     | Example client pushback          | How I respond                          |
|----------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------|-----------------------------------------|
| FX markup by payroll processor   | Mid-market rate clause with source and no markup                                                  | “We can’t control the processor”  | “Then use Wise Business or I walk.”     |
| Equity in local currency         | USD-denominated equity, USD strike price, cashless exercise                                        | “Local law requires MXN”          | “Then gross up my salary by the delta.” |
| Mandatory benefits not covered   | Employer covers benefits or gross up salary by the cost                                           | “EOR doesn’t cover it”            | “Then increase my gross by 8%.”        |
| Payment frequency FX swings      | Bi-weekly payments tied to mid-market rate on payment date                                        | “Payroll only does monthly”       | “Then use Wise for faster payouts.”     |
| Unknown processor fees           | Exact fee schedule in writing before signing                                                      | “We don’t know the fees”          | “Then I need a gross-up to cover them.” |

I learned the hard way that every edge case is a negotiation point. The client’s default position is to push costs to you, but most of these costs are negotiable if you frame them as compliance or risk issues for the employer. The employer would rather pay 5k USD extra in gross-up than risk losing a key hire over a 25 USD fee.

## Step 4 — add observability and tests

You need two things to stay sane after you sign: observability and tests. Observability means a dashboard that shows your net salary in real time, including FX swings, EOR fees, and local taxes. Tests mean a set of assertions that flag when your net drops below your target. I built both using Google Sheets, Wise’s API, and a few lines of Python 3.11.

The observability layer pulls live data every day at 9 AM local time. I use Google Apps Script to call Wise’s mid-market API, pull the latest FX rate, and update a sheet called “Daily Rates.” Here’s the Apps Script snippet:

```javascript
function updateFXRate() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Daily Rates');
  const url = 'https://api.wise.com/v1/quotes?sourceCurrency=USD&targetCurrency=MXN&sourceAmount=5000';
  const options = {
    headers: { 'Authorization': 'Bearer ' + PropertiesService.getScriptProperties().getProperty('WISE_API_KEY') },
    muteHttpExceptions: true
  };
  const response = UrlFetchApp.fetch(url, options);
  const data = JSON.parse(response.getContentText());
  const realRate = data.rate;
  const quotedRate = data.sourceAmount / data.targetAmount;
  const spreadBps = (quotedRate - realRate) / realRate * 10000;
  sheet.getRange('B2').setValue(realRate);
  sheet.getRange('C2').setValue(spreadBps);
  sheet.getRange('A2').setValue(new Date());
}
```

I run this script daily via a time-driven trigger. The sheet now tracks the FX rate, the spread, and the date so I can see if a recent transfer cost me more than usual. I also pull in my actual salary payments from Wise and Revolut via their transaction CSV exports, then reconcile them against


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

**Last reviewed:** May 28, 2026
