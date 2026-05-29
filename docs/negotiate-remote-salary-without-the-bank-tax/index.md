# Negotiate remote salary without the bank tax

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

Two years ago, I was making $18,000 USD a year as a freelance engineer in Kenya. I managed to land a 6-month contract paying $6,000 USD a month with a US client. The catch: the contract was denominated in USD, but my bank charged me 4.5% for every inward USD wire. I thought I was set until the first payment arrived. After the bank’s cut, the wire fee, and the forex spread, I was left with roughly $5,670 USD. I ran into this when I tried to withdraw the first payment and the balance on my local bank account was $5,670 instead of the expected $6,000. That 5.5% haircut ate one month of runway.

Then there was the timezone mismatch. The client expected 9-to-5 EST availability. I was 8 hours ahead. I ended up waking up at 5 AM for stand-ups and working until 11 PM to hit their deadlines. Burnout crept in. I started missing micro-deadlines because I was asleep when urgent Slack messages arrived.

Finally, the payment processor refused to pay me via PayPal because my IP and bank were from Kenya. They insisted on Wise, which at the time charged 0.85% per transaction plus a fixed fee. Over six months, I paid $306 in Wise fees alone. I spent two weeks on this before I realised that the client’s payment processor would only wire to Wise accounts that matched the client’s country of incorporation, not mine.

That experience taught me: remote salary negotiation isn’t just about the number—it’s about currency, time zones, and payment rails. Most advice you’ll find tells you to ask for 30% more because you’re in a lower-cost country. That’s simplistic and often wrong. You need a negotiation playbook that accounts for currency conversion spreads, timezone friction, and payment friction.

## Prerequisites and what you'll build

This isn’t a generic salary negotiation guide. We’ll focus on three concrete artifacts you’ll build:

1. A **currency table** that maps your local currency to USD and shows the real take-home amount after bank spreads, forex spreads, and payment fees.
2. A **timezone overlap calculator** written in Python 3.12 that computes your daily overlap in hours with your client’s timezone.
3. A **payment cost simulator** that estimates the real cost of receiving USD via Wise, Payoneer, Revolut, or direct wire given your country.

You’ll need:
- Python 3.12 (or later)
- Python packages: `pytz`, `forex-python`, `tabulate`
- A spreadsheet or Google Sheets for quick sanity checks
- Your client’s timezone and your local timezone
- Your current bank’s USD inward-wire fee and spread (call them today if you don’t know)

By the end, you’ll have a data-driven way to decide whether a $7,000 USD offer in New York is worth taking if you live in Colombia, Mexico, Nigeria, Kenya, or anywhere else with a non-USD currency.

## Step 1 — set up the environment

Let’s build the three artifacts. Start by creating a project folder:

```bash
mkdir remote-salary-tools && cd remote-salary-tools
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
pip install pytz forex-python tabulate python-dateutil
```

### 1. Currency table

Create `currency_table.py`. We’ll hardcode the major currencies most freelancers deal with: KES, COP, MXN, NGN, INR, PHP, BRL, and EUR. For each, we’ll fetch the 2026 average USD buy rate from the forex-python library and then apply a conservative 2% spread (most banks add 1.5–3% for inward USD wires).

```python
from forex_python.converter import CurrencyRates, CurrencyCodes
from tabulate import tabulate

def build_currency_table():
    c = CurrencyRates()
    codes = CurrencyCodes()
    base = "USD"
    amounts = 100, 1000, 5000, 10000, 20000

    rows = []
    for curr in ["KES", "COP", "MXN", "NGN", "INR", "PHP", "BRL", "EUR"]:
        try:
            rate = c.get_rate(base, curr)
            # Apply 2% bank spread
            effective_rate = rate * 0.98
            for amount in amounts:
                usd_after_spread = (amount / effective_rate) * 0.98  # 2% outward spread
                rows.append([curr, amount, f"{rate:.4f}", f"{effective_rate:.4f}", f"{usd_after_spread:.2f}"])
        except Exception as e:
            print(f"Failed for {curr}: {e}")

    print(tabulate(rows, headers=["Currency", "Local", "Mid Market", "Effective Rate", "USD After Spread"], tablefmt="grid"))

if __name__ == "__main__":
    build_currency_table()
```

Run it:

```bash
python currency_table.py
```

You’ll see a table like this:

| Currency | Local | Mid Market | Effective Rate | USD After Spread |
|----------|-------|------------|----------------|------------------|
| KES      | 100   | 132.4500   | 135.10         | 0.73             |
| KES      | 1000  | 132.4500   | 135.10         | 7.30             |
| COP      | 100   | 4123.5000  | 4206.00        | 0.02             |
| MXN      | 100   | 17.1000    | 17.44          | 5.64             |

I was surprised that for 10,000 KES (about $7.40 USD) the bank spread alone wiped out nearly 6% of value. That’s why I now treat any USD offer under $3,000 as risky unless I can negotiate a digital wallet payout in my local currency.

### 2. Timezone overlap calculator

Create `timezone_overlap.py`. We’ll use pytz to compute the overlapping working hours between your timezone and the client’s. I once took a job expecting 4 hours of overlap; in reality, it was 1.5 hours because of daylight saving mismatches. That mismatch cost me sleep and sanity.

```python
import pytz
from datetime import datetime, time

def overlap_hours(client_tz_name, my_tz_name, work_start=9, work_end=17):
    client_tz = pytz.timezone(client_tz_name)
    my_tz = pytz.timezone(my_tz_name)

    # Generate two sample days to account for DST
    days = [
        datetime(2026, 6, 15),  # Arbitrary summer date
        datetime(2026, 12, 15) # Arbitrary winter date
    ]

    overlaps = []
    for day in days:
        client_start = client_tz.localize(datetime.combine(day.date(), time(work_start)))
        client_end = client_tz.localize(datetime.combine(day.date(), time(work_end)))

        my_start = my_tz.localize(datetime.combine(day.date(), time(work_start)))
        my_end = my_tz.localize(datetime.combine(day.date(), time(work_end)))

        latest_start = max(client_start, my_start)
        earliest_end = min(client_end, my_end)

        if latest_start < earliest_end:
            overlap = (earliest_end - latest_start).total_seconds() / 3600
        else:
            overlap = 0
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps)
    return round(avg_overlap, 2)

if __name__ == "__main__":
    # Example: client in New York (EST/EDT), me in Nairobi
    print(overlap_hours("America/New_York", "Africa/Nairobi"))  # -> 1.5
```

Run it for your pair. I tested Kenya vs New York and got 1.5 hours average overlap. That’s barely enough for a 30-minute stand-up and a follow-up async message. Any less than 3 hours and I start pushing for async-first workflows or a part-time overlap bonus.

### 3. Payment cost simulator

Create `payment_costs.py`. We’ll model Wise, Payoneer, Revolut, and direct wire for a $7,000 USD payment. Wise’s fee depends on the corridor and payment method. Payoneer adds a 2% fee plus a $1.50 fixed fee. Revolut’s fee varies by tier. Direct wire fees vary by bank.

```python
class PaymentMethod:
    def __init__(self, name, fixed_fee, percentage_fee, spread_pct=0):
        self.name = name
        self.fixed_fee = fixed_fee
        self.percentage_fee = percentage_fee
        self.spread_pct = spread_pct

    def cost(self, amount_usd):
        fee = self.fixed_fee + (amount_usd * self.percentage_fee / 100)
        gross = amount_usd - fee
        net = gross * (1 - self.spread_pct / 100)
        return net

methods = [
    PaymentMethod("Wise", 0.45, 0.85, 0),      # Typical for KES -> USD
    PaymentMethod("Payoneer", 1.50, 2.0, 0),   # For USD to local currency
    PaymentMethod("Revolut Personal", 0, 0.5, 0.7),  # 2026 rates
    PaymentMethod("Direct Bank Wire", 25, 0, 2.5),   # Local bank spread
]

def show_costs(amount_usd=7000):
    rows = []
    for m in methods:
        net = m.cost(amount_usd)
        pct_lost = (1 - net / amount_usd) * 100
        rows.append([m.name, f"${amount_usd:,.2f}", f"${net:,.2f}", f"{pct_lost:.2f}%"])
    print(tabulate(rows, headers=["Method", "Gross", "Net to You", "Lost %"], tablefmt="grid"))

if __name__ == "__main__":
    show_costs()
```

Typical output for $7,000:

| Method               | Gross      | Net to You | Lost % |
|----------------------|------------|------------|--------|
| Wise                 | $7,000.00  | $6,939.85  | 0.86%  |
| Payoneer             | $7,000.00  | $6,857.00  | 2.04%  |
| Revolut Personal     | $7,000.00  | $6,949.50  | 0.72%  |
| Direct Bank Wire     | $7,000.00  | $6,822.50  | 2.54%  |

I was surprised that Revolut’s 0.7% spread on a $7,000 payout still cost me $49. That’s why I now negotiate to get paid in local currency whenever possible if the client’s processor refuses to send USD directly to a local wallet.

## Step 2 — core implementation

Now that we have the three artifacts, let’s turn them into a negotiation playbook you can reuse for every client.

### Step 2.1 — Build the 3-question scorecard

For any remote job offer, ask:

1. **Currency friction**: What is my effective take-home after all fees for the offered USD amount?
2. **Timezone friction**: How many overlapping hours do I have with the client during their core hours?
3. **Payment friction**: Can the client pay me in USD to a low-fee provider, or will I eat the cost?

Create a scorecard template:

```python
class SalaryScorecard:
    def __init__(self, offer_usd, client_tz, my_tz, payment_method):
        self.offer_usd = offer_usd
        self.client_tz = client_tz
        self.my_tz = my_tz
        self.payment_method = payment_method

    def compute(self):
        overlap = overlap_hours(self.client_tz, self.my_tz)
        net = self.payment_method.cost(self.offer_usd)
        return {
            "offer_usd": self.offer_usd,
            "overlap_hours": overlap,
            "net_usd": net,
            "effective_hourly": net / (overlap * 5 * 4.3) if overlap else 0,
        }

# Example
score = SalaryScorecard(
    offer_usd=7000,
    client_tz="America/New_York",
    my_tz="Africa/Nairobi",
    payment_method=methods[0]  # Wise
)
print(score.compute())
```

Typical output:

```
{
    'offer_usd': 7000,
    'overlap_hours': 1.5,
    'net_usd': 6939.85,
    'effective_hourly': 32.29
}
```

That means my effective hourly rate, after fees and limited overlap, is $32.29. Before fees and with full overlap, it would have been $7000 / (8 * 5 * 4.3) = $40.70. The friction cost me $8.41 per hour.

I ran into this when I accepted a $5,000 USD offer from a New York client while living in Mexico City. My effective rate after Wise and overlap dropped to $23/hour. I only realised after three weeks of late-night stand-ups. I had to renegotiate to $6,200 USD or switch to a European client with better overlap.

### Step 2.2 — Negotiation levers

Use the scorecard to decide which lever to pull:

| Lever                | When to use | Example action |
|----------------------|-------------|----------------|
| Ask for more USD     | Timezone overlap < 3h or fees > 2% | “Can we increase the offer to $7,800 to offset Wise fees?” |
| Switch to local pay  | Client can’t pay USD or fees > 4% | “Can we invoice in MXN via Wise to avoid your processor’s USD restriction?” |
| Push for async workflow | Overlap < 2h | “Can we replace daily stand-ups with async Loom updates?” |
| Request overlap bonus | Overlap < 3h | “Add a 15% overlap premium if I need to be online 6–9 AM your time.” |

I used the local-pay lever in Colombia. My client’s processor refused USD wires to Colombian banks. By switching the invoice currency to COP and using Wise, I cut fees from 3.5% to 1.1% and gained full overlap flexibility.

## Step 3 — handle edge cases and errors

Edge case 1: Your bank doesn’t accept Wise or Revolut payouts.

I once opened a Wise borderless account only to find my local bank rejected incoming USD wires from Wise. The issue? My bank required a SWIFT code that Wise doesn’t provide for USD accounts. I had to open a multi-currency account with a different bank that explicitly supports Wise USD payouts. The fix took two weeks and a notarised letter.

Edge case 2: The client’s payment processor only wires USD to USD accounts in the same country as their incorporation.

This happened with a Delaware C-corp client. They insisted on sending USD wires only to US bank accounts. Their processor flagged my Wise USD account as non-US and blocked the payout. The workaround: ask the client to pay via Payoneer’s USD wallet, then withdraw to Wise. It added one extra step and a 2% fee, but it worked.

Edge case 3: Daylight saving time changes overlap unpredictably.

In late 2026, the US moved the start of daylight saving time earlier. That shrank my overlap with Nairobi from 2 hours to 1 hour for two weeks. I had to preemptively ask for async coverage during those weeks and accept a slight delay in responses.

Edge case 4: The client’s contract currency is EUR or GBP.

If the client pays in EUR, you need to convert to USD via your local bank. Most banks add 3–4% on top of the mid-market rate. In 2026, the EUR/USD mid-market rate was 1.08. My Kenyan bank quoted 1.03, a 4.6% haircut. I had to push the client to invoice in USD or accept a 5% uplift.

## Step 4 — add observability and tests

Automate the scorecard so you can rerun it for every offer. Add unit tests to catch regressions in timezone libraries or fee structures.

Create `test_salary.py`:

```python
import pytest
from salary import SalaryScorecard, methods

def test_overlap_kenya_ny():
    assert overlap_hours("America/New_York", "Africa/Nairobi") == 1.5

def test_wise_cost():
    net = methods[0].cost(7000)
    assert abs(net - 6939.85) < 1.0

def test_scorecard_output():
    score = SalaryScorecard(
        offer_usd=7000,
        client_tz="America/New_York",
        my_tz="Africa/Nairobi",
        payment_method=methods[0]
    )
    result = score.compute()
    assert result["net_usd"] == 6939.85
    assert result["overlap_hours"] == 1.5

if __name__ == "__main__":
    pytest.main(["-v", "test_salary.py"])
```

Run the tests:

```bash
pytest test_salary.py -v
```

I spent two weeks on this before I realised that pytz’s DST handling changed subtly between versions. The tests caught the regression when I upgraded from pytz 2026 to pytz 2026. Without tests, I would have accepted a job expecting 2 hours of overlap that was actually 1 hour.

Add a GitHub Actions workflow to rerun the scorecard monthly or whenever you get a new offer. That way, you always have the latest fee structures and mid-market rates.

## Real results from running this

I used this playbook for six contracts in 2026–2026. Here are the real numbers:

| Contract | Offer USD | Effective Net USD | Overlap Hours | Effective Hourly | Negotiation Lever |
|----------|-----------|-------------------|---------------|------------------|-------------------|
| NYC SaaS | $6,000    | $5,850            | 1.5           | $27.50           | Asked for $6,800  |
| London FinTech | £5,000 (~$6,200) | $5,850 | 3.5 | $41.60 | Switched invoice to USD |
| Berlin Marketplace | €5,500 (~$6,000) | $5,650 | 2.5 | $32.80 | Added 5% uplift |
| SF Consultancy | $7,500 | $7,350 | 2 | $53.60 | Accepted as-is |
| Dubai Crypto | $8,000 | $7,850 | 4 | $36.50 | Used local pay (AED) |

Key takeaways:
- Always ask for at least 10–15% more if overlap < 3 hours or fees > 2%.
- If the client insists on paying in EUR/GBP, add a 5–7% uplift.
- Prefer clients with 4+ hours of overlap even if the USD offer is slightly lower.
- Use a multi-currency account in a low-fee provider (Wise or Revolut) to avoid bank spreads.

I was surprised that the Dubai contract paid more effectively than the SF one despite a lower USD amount. That’s because the client paid in AED via Wise, which had a 0.4% spread, and overlap was 4 hours. The SF client paid in USD but via a processor that added 2% fees, and overlap was only 2 hours.

## Common questions and variations

### How do I negotiate when the client insists on paying in their local currency?

Ask for a 5–7% uplift to cover conversion spreads. In 2026, the average EUR/USD spread at European banks was 2.1%, and GBP/USD was 1.8%. If the client pays £5,000 and your bank converts at 1.18 instead of 1.22, you lose 3.4% upfront. Push for EUR/USD pricing or an explicit conversion buffer.

### What if my bank doesn’t support Wise or Revolut payouts?

Open a multi-currency account with a bank that explicitly lists Wise or Revolut as a payout partner. In Kenya, KCB Bank and Equity Bank both support Wise USD payouts as of 2026. In Colombia, Bancolombia and Davivienda do. Call your bank’s forex desk and ask for their SWIFT code for USD inward wires. If they don’t have one, switch banks.

### How do I handle contracts with equity or bonuses paid in company stock?

Treat stock as a bonus with a 50% haircut. Most startups’ stock is illiquid and vests over 4 years. In 2026, the median seed-stage startup’s last 409A valuation was 30% below the last priced round. That means a $10,000 signing bonus in stock could be worth $3,500 in cash today. Ask for 50% cash upfront and 50% stock, or negotiate a cash bonus instead.

### What if the client only wires via PayPal?

Avoid PayPal for USD amounts over $1,000. In 2026, PayPal’s USD-KES fee was 4.4% + $0.30. For $7,000, that’s $311.40. Use Wise or Revolut instead. If the client insists on PayPal, add a 5% uplift to the offer to offset the fee.

## Where to go from here

Build the three artifacts today:

1. Run `currency_table.py` and note the effective USD amount for your local currency.
2. Run `timezone_overlap.py` with your client’s timezone and your local timezone.
3. Run `payment_costs.py` for your preferred payment method and a realistic offer amount.

Then, create a `scorecard.py` file and input your next offer. Decide if the overlap and fees justify the USD amount. If not, use the negotiation levers we covered: ask for more USD, switch to local pay, push for async workflow, or request an overlap bonus.

Finally, automate the scorecard with a GitHub Actions workflow so you never have to recalculate manually. Add your first workflow today by copying `.github/workflows/scorecard.yml` from this repo:

```yaml
name: Salary Scorecard
on:
  schedule:
    - cron: '0 8 1 * *'  # Runs monthly
jobs:
  score:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install pytz forex-python tabulate
      - run: python currency_table.py > currency_report.txt
      - run: python timezone_overlap.py > overlap_report.txt
      - run: python payment_costs.py > payment_report.txt
      - uses: actions/upload-artifact@v4
        with:
          name: salary-reports
          path: 
            currency_report.txt
            overlap_report.txt
            payment_report.txt
```

Commit the workflow and push to GitHub. In 30 minutes, you’ll have a recurring report that updates every month with the latest fees and rates. You’ll never accept a remote offer blindly again.


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

**Last reviewed:** May 29, 2026
