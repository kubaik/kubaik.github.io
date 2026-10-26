# Price your remote salary: 4-step script

Most remote-salary advice stops at "use a cost-of-living multiplier." That leaves the actual negotiation without a number. This article builds one: a worksheet that turns a local net salary band into a single USD gross ask you can defend in writing, plus the failure modes that quietly corrupt the result.

The method has four steps: gather inputs, compute the local target, convert to gross and then to USD, and add checks so you notice when a constant goes stale. Everything below is arithmetic on stated assumptions. Substitute your own figures; none of the numbers here are measurements of anyone's outcomes.

## What the worksheet produces

The output is one of three shapes:

1. A single target salary, e.g. 95,000 USD gross.
2. A range with a stated spread, e.g. 92,000–106,000 USD gross.
3. A cost-of-living-indexed range that re-derives itself if you change city.

A single number is easier to defend than a range, because a range invites the counterparty to anchor at the bottom. Compute the range internally, then present the midpoint unless asked.

## Step 1 — Gather and verify the inputs

You need six inputs. Each has a specific failure mode, listed alongside.

| Input | Where it comes from | Typical failure mode |
|---|---|---|
| Local net salary band (median, p75) | Local job boards, recruiter conversations, published national statistics | Mixing net and gross figures from different sources |
| Cost-of-living index for your city | A public cost-of-living index, or your own basket | Using a single headline index instead of rent-heavy and rent-light profiles |
| Effective local income tax rate | The current national tax schedule | Using last year's brackets after a reform |
| USD/local spot rate | A central bank or market rate feed | Using an informal rate for a formal contract, or vice versa |
| FX volatility estimate | Rolling standard deviation of the pair over several years | Copying a figure from an article instead of recomputing it |
| Target savings rate | Your own budget | Setting it from aspiration rather than last year's actuals |

Two of these deserve more than a table row.

**Cost-of-living profiles.** A headline index blends rent, groceries, transport and services. Rent is the line item that varies most between a downtown two-bedroom and a shared flat on the edge of the city. If your index has a rent-heavy variant, use it; if not, build two baskets yourself and compute both. A single headline number can move the final ask by several thousand dollars a year, which is enough to lose a negotiation or underprice yourself.

**FX volatility.** Do not copy a volatility figure. Compute it: download the daily USD/local series for the last five years from your central bank, take log returns, and compute the annualised standard deviation. In a spreadsheet:

```
daily_return = LN(rate_today / rate_yesterday)
annualised_vol = STDEV(daily_return_range) * SQRT(252)
```

252 is the conventional number of trading days per year. The result is the one-sigma annual move. Most people use it directly as a buffer, which is a choice, not a law — a one-sigma buffer covers roughly a two-in-three outcome. If you want more coverage, use 1.5x or 2x the computed figure and say so explicitly in your notes.

**Verify the tax schedule.** Tax brackets change. Re-download the current schedule each January, and re-derive the effective rate for your target net rather than reusing a percentage. The worked example below shows why.

## Step 2 — Compute the local net target

Start from the p75 local net figure, not the median. The median describes the middle of the market; p75 describes what a strong candidate in that market can already earn, which is the relevant floor for a remote role priced against a foreign employer.

```
local_net_target = local_p75_net * (1 + desired_savings_pct) * rent_heavy_col_multiplier
```

Worked example. Suppose the p75 net for a senior backend engineer in your city is 3,100 units of local currency per month, your target savings rate is 25%, and your rent-heavy cost-of-living multiplier is 1.35.

```
local_net_target = 3,100 * 1.25 * 1.35
                 = 3,875 * 1.35
                 = 5,231.25
```

Note what the multiplier is doing. It is not adjusting for the fact that you live in a cheaper city; the p75 band already reflects that. It is adjusting for the fact that a remote employer paying a developed-market rate is competing for your time against local employers, and your reservation price should reflect your actual cost structure, not the local average.

If you skip the cost-of-living multiplier entirely, you are pricing your labour at local market rates while selling it into a foreign market. That is the single most common way remote workers underprice themselves.

## Step 3 — Gross up, then convert

### Gross-up

Local salary bands are usually quoted net. Offers are usually made gross. Convert:

```
local_gross = local_net_target / (1 - effective_tax_rate)
```

The effective tax rate is not the marginal rate. It is total tax divided by gross income, which for most progressive schedules is meaningfully lower than the top bracket. Compute it by running your target net through the actual bracket table:

1. Lay out the brackets: threshold, marginal rate.
2. Guess a gross figure.
3. Apply the brackets to that gross figure to get tax.
4. Subtract tax from gross to get net.
5. Compare to your target net. Adjust the guess and repeat.

Two or three iterations converge. A spreadsheet `VLOOKUP` against a bracket table will not do this correctly on its own, because progressive brackets are cumulative, not a single lookup.

Worked example. Suppose the effective tax rate at your target is 28%.

```
local_gross = 5,231.25 / (1 - 0.28)
            = 5,231.25 / 0.72
            = 7,265.63
```

If a reform moves the effective rate to 32%:

```
local_gross = 5,231.25 / (1 - 0.32)
            = 5,231.25 / 0.68
            = 7,693.01
```

The gross figure rises by about 6% from a four-point change in the effective rate. This is why the tax schedule has to be re-verified rather than assumed.

### Convert to USD with a buffer

```
usd_ask = (local_gross / usd_local_spot) * (1 + fx_buffer)
```

Worked example, with a spot of 4,200 local units per USD and a computed annualised volatility of 12%, used directly as the buffer:

```
usd_ask = (7,693.01 / 4,200) * 1.12
        = 1.8317 * 1.12
        = 2.0515
```

That is roughly 2,052 USD per month, or about 24,600 USD per year on a twelve-month basis. If the local p75 was a monthly net figure, keep every intermediate step monthly and only annualise at the end, otherwise you will silently mix units.

### The FX buffer is not optional

A contract signed at one spot rate and paid at another exposes you to the full move. If the local currency weakens 15% against the dollar over a year, an un-buffered contract loses 15% of its real value with no renegotiation trigger. The buffer converts that risk into a slightly higher headline number, which is easier to negotiate once than to renegotiate later.

Sanity check for the buffer: ask what happens at the 5th percentile of the historical move, not the average. If a 15% adverse move would put you below your local net target, the buffer is too small.

## Step 4 — Add checks so stale data is visible

A worksheet is only as good as its constants. Build three checks and run them every time you touch the sheet.

**Freshness checks.** Each constant carries a date. Flag any constant older than 90 days, and any tax schedule not from the current year.

**Recomputation check.** For one reference case, compute the answer by hand and assert the sheet matches. If they diverge, a formula has been edited.

**Range check.** Divide your USD ask by the U.S. median gross for the same role and check the ratio lands where you expect. A ratio far below your target band usually means a unit error (monthly vs annual, net vs gross) rather than a genuinely low number.

In Python, these become ordinary tests:

```python
import pytest
from calculator import compute_target_range

def test_reference_case():
    result = compute_target_range(
        local_p75_net=3100,
        savings=0.25,
        col_multiplier=1.35,
        effective_tax_rate=0.28,
        spot=4200,
        fx_buffer=0.12,
    )
    assert result["local_net_target"] == pytest.approx(5231.25, rel=0.001)
    assert result["local_gross"] == pytest.approx(7265.63, rel=0.001)
    assert result["usd_monthly_ask"] == pytest.approx(1937.5, rel=0.01)

def test_tax_schedule_is_current_year():
    from calculator import load_tax_schedule
    schedule = load_tax_schedule()
    assert schedule["year"] == 2026
```

The first test pins the arithmetic. The second pins the data. Together they catch the two failure modes that actually occur: a formula edit and a stale constant.

For the spreadsheet version, put the same three checks in a separate tab: a cell comparing the sheet's output to a hard-coded expected value, a cell showing the age in days of each constant, and a conditional format that turns red past the threshold.

## Failure modes worth naming

**Mixing net and gross.** The most common error by a wide margin. Job boards quote net, offers quote gross, and some sources do not say which. Label every number in your sheet with its basis.

**Using the marginal tax rate as the effective rate.** It overstates tax and understates the gross you need. Compute the effective rate from the bracket table.

**Treating the cost-of-living index as a salary index.** They are different things. A city can be 30% cheaper than another and still pay 40% less in nominal terms. The index adjusts your cost base, not the market rate for your skills.

**Ignoring the payment currency.** If the client pays in USD to a foreign entity, you carry the FX risk. If they pay in local currency through a local entity, you do not — but you also lose access to the foreign market rate. Set the buffer to zero in the second case and price off the local gross.

**Contractor versus employee.** If you invoice as an independent contractor, you absorb employer-side contributions that an employee would not. In the U.S., self-employment tax is 15.3% on net earnings, and you may also owe local VAT on services. Gross up:

```
contractor_ask = employee_ask / (1 - self_employment_rate - vat_rate)
```

With a 15.3% self-employment rate and 19% VAT, the divisor is 0.657, so the contractor ask is roughly 1.52x the employee ask. If a client offers a contractor rate less than about 1.35x the equivalent employee rate, the difference is being paid out of your pocket.

**Equity treated as cash.** Vesting schedules, cliffs and liquidity events mean equity is not salary. Value only the portion that vests within twelve months of signing and treat the rest as zero for negotiation purposes.

**Inflation not captured by the index.** A cost-of-living index is a snapshot. In a high-inflation environment it goes stale within months. Recompute quarterly, and consider indexing the contract itself if the client will agree to it.

## A decision checklist before you send a number

- Every constant has a date and a source.
- Net and gross are labelled on every figure.
- The effective tax rate was derived from the current bracket table, not assumed.
- The FX buffer was computed from a historical series, not copied.
- The cost-of-living profile matches your actual housing situation.
- The contractor gross-up is applied if you are invoicing.
- The reference-case test passes.
- The USD ask is written as a single number with the currency and the period stated.

## FAQ

**Should I disclose my local salary band?**
Disclosing a local band hands the counterparty an anchor that is usually below the foreign market rate. Compute your ask from the band, then present only the ask. If pressed for a basis, describe the method — cost-of-living parity plus a stated savings rate — rather than the underlying local figure.

**What if the client pays in a third currency?**
Price in the currency the client will actually send. Convert your local target into that currency using the relevant pair, and compute the buffer from that pair's historical volatility, not from USD/local.

**How often should the worksheet be rebuilt?**
Re-verify the tax schedule annually, the FX volatility quarterly, and the local salary bands every six months. Rebuild the whole sheet before any negotiation rather than reusing a number from a previous round.

**Does a lower local cost of living justify a lower ask?**
It justifies a lower reservation price, which is different. Your reservation price is the minimum you will accept. Your ask should be anchored to the value of the role to the employer, tempered by your reservation price. Confusing the two is what produces offers at the local market rate for remote work.

## Do this in the next 30 minutes

Open a blank spreadsheet and create six rows: local p75 net, cost-of-living multiplier, target savings rate, effective tax rate, spot rate, FX buffer. Fill in the first three from your own situation. Then compute the effective tax rate by running your target net through the current bracket table by hand, and compute the FX buffer by downloading five years of daily rates and taking the annualised standard deviation of log returns. Convert the result to a single monthly USD figure. That number, with its inputs visible, is the counter-offer you can defend.
