# 2026 tech salaries: where cuts stopped, where raises

## The core claim, stated plainly

Salary bands are not a single market. They are a function of four variables that move independently: company funding stage and runway, whether the role touches revenue-generating or regulated code, whether cash is being traded for equity, and the tax treatment of a cross-border employment relationship. When people say "salaries fell" or "salaries rose" in 2026, they are usually describing one of these variables and generalising it to all four.

The practical consequence: two engineers with the same title, same years of experience and same city can be 40% apart in total compensation, and neither is being underpaid relative to their own company's constraints. Judging an offer by title and city alone will mislead you.

This article gives you a model for decomposing an offer, a worked comparison with the arithmetic shown, the failure modes that make equity and cross-border pay worse than they look, and a checklist you can run against a real offer letter.

## Why the "geography only" model fails

The common mental model is a single axis: Nairobi pays X, London pays Y, San Francisco pays Z. That model was always approximate, but it breaks badly when funding conditions diverge between companies in the same city.

A more useful decomposition treats a company as a set of concentric rings around its value engine:

- **Innermost ring — the product.** If the code is the differentiator (a payments ledger, a lending decision engine, custody infrastructure), the company cannot easily replace the people who own it. Compensation for those roles tends to be sticky or rising.
- **Next ring — the customer segment.** Selling to enterprises and to other technical buyers generally supports higher pay than selling to consumers, because contract values and margins are higher.
- **Outer ring — the funding story.** A company with a long runway can hire at its planned band. A company with a short runway is either freezing, cutting, or converting cash into equity.

The outer ring dominates in a tight market. A well-funded company and a thinly-funded company in the same city, hiring for the same role, will produce very different offers, and the difference is not a negotiation tactic — it reflects what each can actually commit to.

## A worked example: two senior backend offers

The following figures are **illustrative**, chosen to make the arithmetic visible. Substitute your own numbers.

**Offer A — Series B fintech, closed a round roughly 18 months before the offer, remote-first.**

- Base cash: $85,000
- Signing bonus: $12,000, paid in two tranches (30 days and 180 days)
- Equity: 3,000 options at a $0.25 strike, current 409A valuation $0.32, 1× acceleration after 12 months
- Remote stipend: $1,200/year, paid as a taxable allowance

First-year cash: $85,000 + $12,000 + $1,200 = **$98,200**.

The equity's paper spread is 3,000 × ($0.32 − $0.25) = **$2,100**. That is the intrinsic value if you could sell today, which you cannot. Everything beyond that is a bet on a future valuation and a future liquidity event.

**Offer B — Series C+ company in the same niche, remote-first, longer runway.**

- Base cash: $95,000
- Signing bonus: $15,000
- Equity: 2,500 options at a $0.50 strike, 409A at $0.55, 3× acceleration after 12 months
- Remote stipend: $2,400/year
- Learning budget: $3,000/year; conference travel: $2,000/year

First-year cash: $95,000 + $15,000 + $2,400 = **$112,400**.

Paper spread on equity: 2,500 × ($0.55 − $0.50) = **$1,250**.

The cash delta is $112,400 − $98,200 = **$14,200**, or about 14.5% of the lower offer. The equity spread is actually *smaller* on Offer B in absolute terms, which is the counter-intuitive part: a later-stage company often has a narrower gap between strike and fair market value because the strike was set at a higher, more recent valuation.

That means the "equity is worth more at the later-stage company" intuition is only true if you weight the probability of a liquidity event heavily. On intrinsic value alone, the earlier-stage grant looks better. On cash, the later-stage offer wins clearly.

### How to measure the equity component honestly

There is no published dataset that will tell you what a specific private company's options are worth. You have to build the estimate yourself, and label every input as an assumption:

1. **Probability of a liquidity event within your vesting horizon.** This is the input people guess most freely and it dominates the result. Anchor it to something observable: has the company had a secondary tender offer? Are later-stage investors marking up the position? How many months of runway does the last disclosed round buy at current burn?
2. **Expected exit valuation relative to the current 409A.** Use the last primary round's post-money valuation as a ceiling unless you have specific evidence for more.
3. **Your ownership percentage, fully diluted.** Ask for the fully diluted share count, not just your share number. A grant of 3,000 shares means nothing without the denominator.
4. **Tax at exercise or settlement.** For options, the spread at exercise can be taxable even without a sale in some jurisdictions. Model this as a cash outflow, not a footnote.

Multiply probability × expected value × your diluted ownership, then discount for illiquidity. If the result is a small fraction of your cash component, treat the equity as a lottery ticket and negotiate on cash.

## Failure modes that make offers worse than they look

**Stale strike prices.** A grant priced at the last round's valuation looks attractive until the next 409A comes in lower. If the fair market value drops below your strike, your options are underwater before they vest, and the "discount" you negotiated evaporates. The mitigation is contractual, not analytical: ask whether the company will reprice or regrant if a future 409A falls below your strike. Many will not commit to this in writing, and that refusal is itself information.

**Acceleration clauses that don't trigger.** Single-trigger acceleration (vesting accelerates on a change of control alone) is meaningfully different from double-trigger (acceleration requires both a change of control and your termination). A "2× acceleration" headline is worth little if the trigger conditions are unlikely to be met in the scenario where you'd actually need it.

**Location-adjustment matrices.** Remote-first companies frequently peg a location to a percentile of a comparable market rather than paying a single global rate. The published band may be wide precisely so that the actual offer can land anywhere inside it. If a band is quoted as a range, assume the offer is at the lower end unless you have a competing offer.

**Cross-border payroll defaults.** When a company in one country employs someone in another through a payroll provider, the default configuration is often the one that minimises the employer's cost, not the employee's. Withholding treatment, social contributions and treaty positions vary by country and by provider. The only reliable approach is to model your own after-tax number for each offer and confirm it with a qualified tax advisor in your country of residence. Do not rely on a provider's marketing page for your personal tax position.

**Contractor-versus-employee arithmetic.** A contractor rate looks higher per hour but carries costs the headline hides: you absorb employer-side contributions, you have no paid leave, and the engagement can end without notice. When comparing, convert both to an annualised all-in cost including unpaid time between contracts, then compare. The comparison often flips depending on how many weeks per year you can actually bill.

## A decision checklist for a real offer

Run these in order. Stop when an offer fails a step that matters to you.

1. **Cash first.** What is guaranteed first-year cash (base + signing + allowances)? Ignore equity for this step.
2. **Runway.** How many months does the last round fund at current burn? Ask directly; a refusal to answer is a signal.
3. **Equity denominator.** What is the fully diluted share count, and what percentage is your grant?
4. **Strike versus current 409A.** What is the spread today, and when was the 409A last set?
5. **Trigger conditions.** Is acceleration single- or double-trigger, and what multiple applies?
6. **Liquidity history.** Has the company ever run a secondary or tender offer? If yes, at what price relative to the then-current 409A?
7. **Tax.** Model your after-tax number under the actual withholding regime, not the headline rate.
8. **Reversibility.** If the equity is worth zero, is the cash still competitive for your market? If not, the offer is an equity bet dressed as a salary.

## Comparing offers across tax regimes

The comparison people get wrong most often is between a high-cash, low-tax jurisdiction and a lower-cash, higher-tax one. The error is comparing gross numbers.

Work through it as a sequence of explicit steps, using illustrative figures:

1. **Start with gross cash.** Offer X: $85,000. Offer Y: $110,000.
2. **Apply the actual withholding regime.** If Offer X is subject to a 30% effective rate, net cash is $85,000 × 0.70 = $59,500. If Offer Y is subject to 0%, net cash is $110,000.
3. **Subtract the cost-of-living delta.** If the higher-cash location costs an additional $30,000/year in housing and related expenses, the effective advantage narrows to $110,000 − $59,500 − $30,000 = $20,500.
4. **Add back the value of benefits you would otherwise buy.** If Offer X includes $5,000/year of learning and travel budgets you would otherwise fund yourself, the gap narrows further.
5. **Then, and only then, add a discounted equity estimate.**

The point of the sequence is that equity should be the last line, not the first. When it is added last, it is much harder for a large but improbable number to dominate the decision.

A further complication: tax residence is not the same as physical location, and treaty positions between countries determine whether income is taxed once, twice, or with a credit. This is genuinely jurisdiction-specific and changes with legislation. Treat any general statement about "0% tax" as a starting question for an advisor, not an answer.

## What the market data can and cannot tell you

Published compensation datasets are useful for establishing the shape of a distribution and almost useless for pricing an individual offer. They are self-reported, skewed toward large employers and toward people who are motivated to report, and they rarely capture equity terms in enough detail to be comparable.

The honest use of such data is to answer "is my cash component within a plausible range for this role and market?" It cannot answer "is this equity worth anything?" and it cannot answer "is this offer good for me?" Those require the company-specific inputs in the checklist above.

If you want to build your own picture rather than rely on someone else's dataset, the measurement is straightforward: record the guaranteed cash, the fully diluted ownership percentage, the strike, the current 409A, the acceleration terms and the runway for every offer you receive or hear about from a trusted peer. After a handful of data points you will have a small, biased, but *first-hand* dataset that is more relevant to your decisions than any aggregate.

## FAQ

**Why do some roles stay flat while others rise in the same market?**

Roles tied to revenue-generating or regulated systems are harder to substitute and tend to hold their bands. Roles that are seen as cost centres are the first to be frozen or converted to contract. The title matters less than whether the work sits close to the money.

**Is a signing bonus a good substitute for base?**

It is a one-time payment, so it does not compound into future raises, equity refreshes or severance calculations. A $15,000 signing bonus and a $15,000 base increase are not equivalent even in year one, and the gap widens every year after. Treat signing bonuses as bridging cash, not as compensation.

**How do I evaluate options when the company won't share the fully diluted count?**

You cannot evaluate them, and that is the answer. Ask in writing. If the company declines to provide the denominator, price the equity at zero for decision purposes and negotiate on cash.

**Does a lower strike always mean a better grant?**

No. A lower strike usually reflects an earlier, lower valuation. What matters is the spread between strike and current fair market value, your ownership percentage after dilution, and the probability of a liquidity event. A low strike on a large grant in a company with no exit path is worth less than a modest grant in a company with a clear path.

**Should I take the higher gross offer?**

Only after you have computed after-tax cash, subtracted the cost-of-living delta, and priced the equity at something you can defend. Gross numbers are the beginning of the comparison, not the end of it.

## Do this in the next 30 minutes

Take any offer you are currently considering — or the one you accepted most recently — and write down five numbers in a single line: guaranteed first-year cash, fully diluted ownership percentage, strike price, current 409A, and months of runway funded by the last round. If you cannot fill in all five, you have identified the exact questions to send the company today, in writing, before you make any further decision.
===END===
