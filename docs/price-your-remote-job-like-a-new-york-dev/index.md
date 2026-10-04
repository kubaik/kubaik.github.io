# Price your remote job like a New York dev

Remote postings from US, Canadian, and European employers commonly list compensation in USD, and public salary aggregators usually surface San Francisco or New York figures first. A contractor in Bogotá, Mexico City, or São Paulo who anchors a quote to local cost-of-living data will land well below what the same title pays in the client's market. This article lays out a repeatable way to build a rate model: define the inputs, compute the client-facing rate, handle currency and tax edge cases, and log every quote so you can explain it months later.

## What you need before you start

Three inputs, and nothing more:

1. **One real job posting.** A saved listing is better than a hypothetical, because it gives you a title, a seniority level, and often a range. Company career pages are more reliable than aggregators, which sometimes carry stale ranges.
2. **A spreadsheet** with three scenarios: your local break-even, a middle figure, and the top of the client's likely range. A Google Sheet or Airtable base is enough.
3. **A small Python script** that converts a target annual figure into an hourly, weekly, and monthly rate a client can paste into a budget tool.

A suggested layout:

```
remote-rate-calc/
├── data/
│   ├── benchmarks.csv      # your own export of public comp data
│   └── fx_rates.json       # cached FX rates
├── scripts/
│   ├── rate_model.py       # core calculator
│   └── sanity_check.py     # checks for stale data
└── README.md
```

No orchestration layer, no managed database. A virtual environment, pandas for the arithmetic, and a caching HTTP client for FX lookups will cover it. Pin your versions in a lockfile so the numbers you computed last quarter still reproduce today.

```bash
python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip \
  pandas requests-cache pytest
```

On Windows, activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## Step 1 — assemble the benchmark data

### 1.1 Load and filter

You need a CSV with at least: `role`, `city`, and `total` (total annual compensation). Public aggregators publish this in various formats; the column names below are illustrative, so adjust them to your export.

```python
import pandas as pd

df = pd.read_csv("data/benchmarks.csv", dtype={"total": "float64"})
roles = ["Staff Engineer", "Senior Backend Engineer", "Backend Engineer"]
filtered = df[df["role"].isin(roles)].copy()
filtered["city"] = filtered["city"].fillna("Remote")
```

Filter by title and by city tier before you compute anything. A median that mixes San Francisco, Austin, and fully-remote rows is not a useful comparison point.

### 1.2 Cache FX rates

FX APIs rate-limit, and you do not need a fresh rate every time you open the script. A cached session with a 24-hour expiry is enough for quoting purposes.

```python
import requests_cache
import json
from pathlib import Path

session = requests_cache.CachedSession(
    "data/fx_cache",
    backend="sqlite",
    expire_after=86_400,  # 24 hours
    stale_if_error=True,
)

def fetch_fx_rates(date=None):
    date = date or "2026-06-01"
    url = f"https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml?{date}"
    resp = session.get(url)
    resp.raise_for_status()
    # The ECB endpoint returns XML. Parse it with xml.etree.ElementTree
    # or lxml and write the result as JSON; the parsing step is omitted here.
    rates = parse_ecb_xml(resp.text)
    Path("data/fx_rates.json").write_text(json.dumps(rates))
    return rates
```

Note the stub: the ECB feed is XML, so a `.json()` call on the response will fail. Parse the XML, then persist JSON.

### 1.3 Build the three-scenario sheet

| Sheet | Purpose | Formula shape |
|-------|---------|---------------|
| Local | Your cost-of-living break-even | `=annual_local / 2080 * 1.25` |
| Middle | A discount from the client-market midpoint | `=INDEX(benchmarks!F:F, MATCH("Staff Engineer", benchmarks!A:A, 0)) * 0.6` |
| Client | Top of the client's likely budget | `=INDEX(benchmarks!F:F, MATCH("Senior Backend Engineer", benchmarks!A:A, 0)) * 0.85` |

The 1.25 multiplier on the Local sheet is a placeholder for your own overhead; replace it with your actual costs. The 0.6 and 0.85 multipliers are policy choices, not facts — they encode how aggressive you are willing to be. Write down why you picked them, because you will be asked.

Currency conversion is where the sheet usually lies to you. If you quote in USD but receive a wire in COP, the bank's spread comes out of your margin. Quote in the currency the client pays in, and convert to your local account at the rate your bank actually gives you, not the interbank rate.

## Step 2 — the rate model

### 2.1 The core calculation

The model converts a target annual take-home figure into an hourly rate. It accounts for:

- Self-employment or social-security contributions (the rate depends on your jurisdiction and the client's).
- VAT or GST on export services (0% for US clients in many cases; other rates apply elsewhere).
- A buffer for sick days and holidays.
- A buffer for project ramp-up.

```python
from dataclasses import dataclass
import json

@dataclass
class RateModel:
    target_annual: float         # take-home target
    fx_rate_usd_to_local: float  # e.g. 3900 COP per USD
    self_employment_tax: float = 0.153
    vat_export: float = 0.0
    buffer_sick: float = 0.10
    buffer_ramp: float = 0.15

    def client_rate_usd(self) -> float:
        net_needed = self.target_annual
        gross_before_tax = net_needed / (1 - self.self_employment_tax)
        if self.vat_export > 0:
            gross_before_vat = gross_before_tax / (1 - self.vat_export)
        else:
            gross_before_vat = gross_before_tax
        total_with_buffers = gross_before_vat * (1 + self.buffer_sick + self.buffer_ramp)
        hourly = total_with_buffers / 2080
        return round(hourly, 2)

    def client_rate_local(self) -> float:
        return round(self.client_rate_usd() * self.fx_rate_usd_to_local, 0)
```

The 2080 figure is 40 hours × 52 weeks. It is a convention, not a measurement of your productive hours. If you take four weeks of leave and lose a week to holidays, your billable hours are closer to 1880, and the same target annual produces a higher hourly rate. Decide which denominator you are using and state it in the quote.

### 2.2 Sanity-check against the benchmark

Before sending anything, compare your computed rate against the median for the role in the client's market.

```python
benchmarks = pd.read_csv("data/benchmarks.csv")
midpoint = benchmarks[benchmarks["role"] == "Senior Backend Engineer"]["total"].median()
print(f"Client-market midpoint: ${midpoint:,.0f}")
```

If your rate implies an annual figure more than roughly 15% above that midpoint, the client's finance team will likely push back. If it is more than 15% below, you are leaving money on the table. Both directions are worth a second look at the buffers.

### 2.3 Regional parameters

Keep jurisdiction-specific values in a config file rather than in code. The numbers below are illustrative placeholders — verify every one against the current rules for your country before you rely on it.

```yaml
usd:
  vat_export: 0.0
  fx_demo: 1.0
  self_employment_tax: 0.153
cop:
  vat_export: 0.0
  fx_demo: 3900.0
  self_employment_tax: 0.153
mxn:
  vat_export: 0.16
  fx_demo: 17.5
  self_employment_tax: 0.011
```

The point of the config is that the tax treatment for a Mexican contractor is not the same as for a US one. Applying the US self-employment rate to a Mexican invoice produces a number the client's accountant will reject. Confirm the correct rate with a local accountant rather than copying a figure from an article.

## Step 3 — edge cases that break quotes

### 3.1 Currency spikes

A currency can move several percent in a week around a central-bank announcement. If your quote was written at an old rate, the client's accounting team may reject the invoice because the local-currency amount no longer matches.

Add a guard that refuses to emit a quote when the rate has moved beyond a threshold:

```python
def validate_fx_spike(rate_before, rate_after, pct_threshold=0.05):
    change = abs(rate_before - rate_after) / rate_before
    if change > pct_threshold:
        raise ValueError(
            f"FX rate moved {change:.1%} "
            f"({rate_before} -> {rate_after}) — abort quote"
        )
```

The threshold is a policy choice. Five percent is a reasonable starting point; tighten it if your margins are thin.

### 3.2 Ramp time

The 15% ramp buffer assumes a short onboarding. If the client needs you productive in six weeks and you lose the first two to setup, the buffer is too small:

```python
model = RateModel(target_annual=120_000, fx_rate_usd_to_local=3900)
model.buffer_ramp = 0.25
```

Ask for the start date and the expected ramp before you quote. If the client's finance team has a fixed junior band, a rate that lands below it will trigger questions even if the total is fine.

### 3.3 VAT as a separate line

When VAT applies, put it on its own line. Clients who pay via invoice often need the tax field to reconcile automatically.

```python
def invoice_lines(model, hours):
    net_line = model.client_rate_usd() * hours
    vat_line = net_line * model.vat_export
    gross_line = net_line + vat_line
    return {"net": net_line, "vat": vat_line, "gross": gross_line}
```

A VAT amount folded into the total is a common reason for a rejected invoice.

## Step 4 — logging and tests

### 4.1 Log every quote

Store each quote with a hash of its inputs. Six months later, when a client asks why the number was what it was, you can reproduce the calculation exactly.

```python
import sqlite3
import hashlib
import json

conn = sqlite3.connect("data/quotes.db")
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sha TEXT UNIQUE,
        ts TEXT,
        target_annual REAL,
        client_rate_usd REAL,
        client_rate_local REAL,
        fx_rate REAL,
        buffers TEXT,
        notes TEXT
    )
    """
)

def log_quote(model, notes=""):
    payload = json.dumps({
        "target_annual": model.target_annual,
        "fx_rate": model.fx_rate_usd_to_local,
        "buffers": {
            "sick": model.buffer_sick,
            "ramp": model.buffer_ramp,
            "vat": model.vat_export,
            "tax": model.self_employment_tax,
        },
    }, sort_keys=True)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    conn.execute(
        """
        INSERT INTO quotes (sha, ts, target_annual, client_rate_usd,
                            client_rate_local, fx_rate, buffers, notes)
        VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
        """,
        (
            sha,
            model.target_annual,
            model.client_rate_usd(),
            model.client_rate_local(),
            model.fx_rate_usd_to_local,
            json.dumps({
                "sick": model.buffer_sick,
                "ramp": model.buffer_ramp,
                "vat": model.vat_export,
                "tax": model.self_employment_tax,
            }),
            notes,
        ),
    )
    conn.commit()
```

### 4.2 Tests

The tests below are the ones worth writing first: one per jurisdiction and tax combination you actually quote in. Compute the expected value by hand from the formula before you write the assertion — otherwise you are just freezing whatever the code currently does.

```python
import pytest
from rate_model import RateModel


def test_us_client_no_vat():
    model = RateModel(
        target_annual=120_000,
        fx_rate_usd_to_local=3900,
        vat_export=0.0,
    )
    # 120000 / (1 - 0.153) = 141,676.44
    # 141,676.44 * (1 + 0.10 + 0.15) = 177,095.55
    # 177,095.55 / 2080 = 85.14
    assert model.client_rate_usd() == pytest.approx(85.14, abs=0.01)
    assert model.client_rate_local() == pytest.approx(332_046, abs=1)
```

The original article's expected value of 78.85 did not match its own formula; the arithmetic above shows the correct figure. The lesson is general: write the hand calculation into the test as a comment, or you will eventually assert a number that is simply wrong.

A CI step to run them:

```yaml
# .github/workflows/test.yml
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
- run: pip install -e . pytest
- run: python -m pytest
```

### 4.3 Alert on stale data

If the cached FX feed is old, every quote derived from it is suspect. Fail loudly:

```python
from datetime import datetime, timezone

def assert_fresh(fetched_at_iso, max_age_hours=48):
    fetched_at = datetime.fromisoformat(fetched_at_iso)
    age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
    if age_hours > max_age_hours:
        raise RuntimeError(f"FX feed stale ({age_hours:.1f}h > {max_age_hours}h)")
```

## How to measure whether this is working

There is no meaningful industry benchmark for "quote rejection rate" — it depends on your niche, your clients, and your seniority. What you can do is measure your own process before and after. Instrument these four numbers from your own records:

| Metric | How to measure | What a change tells you |
|--------|----------------|-------------------------|
| Quote-to-conversation rate | Quotes sent ÷ replies received | Whether your pitch or your price is the blocker |
| Rejection reason | Tag each rejection: price, scope, timing, no reply | Whether the model needs tuning or the pipeline does |
| FX loss per wire | Interbank rate at invoice date vs. rate received | Whether the wire spread is eating your margin |
| Realised hourly | Total invoiced ÷ hours actually worked | Whether your buffers were realistic |

Fill this in for your last ten quotes, then again after ten quotes using the model. The comparison is the evidence, not a number borrowed from someone else.

## Worked example

Suppose your target take-home is $120,000, you are billing a US client, and you pay 15.3% self-employment tax with no VAT on the export.

1. Gross before tax: `120,000 / (1 - 0.153) = 141,676.44`
2. Add sick and ramp buffers: `141,676.44 × 1.25 = 177,095.55`
3. Hourly at 2080 hours: `177,095.55 / 2080 = 85.14`
4. At a hypothetical rate of 3900 COP per USD: `85.14 × 3900 = 332,046 COP/hour`

Now check it against the market. If the client-market median for the role is $160,000 total, your $177,096 implied annual is about 11% above it. That is inside the range where a client's finance team will usually engage rather than dismiss. If the median were $130,000, your figure would be 36% above and you would need either a lower target or a justification tied to a specific skill the role requires.

The arithmetic here is the whole method. The rest is bookkeeping.

## Common questions

**Do I have to quote in USD?**

No. Quote in the currency the client pays in. If that is EUR, use EUR, and adjust the tax parameters for your jurisdiction rather than assuming the US rates apply.

**How do I handle equity or bonuses?**

Treat them as a separate line with an explicit vesting schedule and a stated valuation method. Do not discount them into the hourly rate unless you have a way to realise them. A promise of RSUs in a private company is not the same as cash.

**What if the client uses an employer-of-record service?**

An EOR withholds local taxes on your behalf, so your `self_employment_tax` in the model should be zero for that engagement. The VAT treatment depends on the EOR's jurisdiction and the client's, not on yours. Ask the EOR for its VAT export flag in writing before you invoice.

**Should I go through a local agency?**

Agencies typically take a commission, which raises the client-facing price for the same take-home. Whether that is worth it depends on whether the agency brings you clients you could not reach directly. Model both and compare the client-facing number, not the take-home.

## Failure modes to watch for

- **Quoting against local cost of living.** The client's budget is set by their market, not yours. Anchor to the role and the client's city tier.
- **Mixing currencies in one quote.** Pick the client's currency for the quote and convert only for your own records.
- **Burying VAT in the total.** Put it on its own line or expect a rejection.
- **Using a stale FX rate.** Cache with an expiry and fail loudly when the cache is old.
- **Assuming a tax rate from an article.** Verify with a local accountant. Rates change, and the wrong one produces an invoice the client cannot process.
- **Forgetting the denominator.** 2080 hours is a convention. If you take real leave, your billable hours are lower and your rate should be higher.

## Do this in the next 30 minutes

Open your spreadsheet, pick one real posting you have saved, and fill in the three scenarios: your local break-even, the client-market midpoint, and the top of the client's likely range. Then run the four-line calculation from the worked example with your own target figure and paste the result into a note. You will immediately see whether your current mental number is above or below the market band — and that comparison, not the exact figure, is what you need before the first conversation.
