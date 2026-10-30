# Double the price, double the revenue

Most pricing advice for developer tools is written for companies that sell to engineering managers with departmental budgets. Indie tools and small SDKs usually sell to solo developers and small teams paying out of pocket. The two audiences behave differently, and a pricing table copied from a large SaaS company often misreads the second group entirely.

This article walks through a four-tier pricing model built around user maturity and willingness to pay, then shows how to enforce it in code with a usage counter, feature gates, and a plan guard. It also covers how to measure whether a pricing change actually helped, because the honest answer to "should I raise prices?" is always "instrument it and find out."

## What this model assumes

Before pricing anything, the tool needs evidence of product-market fit. Useful signals include:

- A growing set of stars or installs, and more importantly, repeat weekly usage.
- A clear "aha" moment where a user hits a task the tool solves and returns to it without being prompted.
- At least a handful of users who would notice if the tool disappeared tomorrow.

If those signals are absent, pricing work is premature. A pricing page cannot manufacture demand; it can only capture demand that already exists. The failure mode to watch for is spending weeks on a pricing table while the underlying product still has no retention.

The model below assumes a CLI, SDK, or library with a recurring usage pattern — formatting, linting, code generation, data sync, or similar. It maps four tiers to user maturity:

- **Free** — evaluation, open-source contributors, hobby use.
- **Starter** — solo developers and micro-teams.
- **Growth** — teams of roughly 2–10 engineers.
- **Scale** — funded startups and larger organizations.

Each tier carries a seat limit, a usage allowance, and a feature gate. The enforcement mechanism is deliberately simple: a plan value read from an environment variable or config file, checked against a counter. No multi-tenant billing system, no ORM, no database joins on the hot path.

## Choosing a billing provider

The provider handles checkout, tax, and subscription state. What matters for this model is whether it supports seat-based pricing and metered or usage-based billing, because the tiers above combine both.

When comparing providers, check these properties rather than brand names:

- **Seat-based pricing support** — can a subscription have a quantity that changes mid-cycle, with proration?
- **Usage metering** — can you report usage events and bill on them, or do you only need the provider to track plan state?
- **Tax handling** — does the provider act as merchant of record and handle VAT/GST, or do you?
- **Local payment methods** — card fees in some regions are materially higher than local rails, and offering local options can change conversion.
- **Webhook reliability** — plan changes must reach your application; check retry behavior and idempotency.

A practical split: let the billing provider own money movement and subscription state, and let your application own the fast path — the per-command usage check. The application should never call the billing API synchronously on a user's command. It reads a cached plan value and a local counter.

The cost structure is usually a percentage plus a fixed fee per transaction, with the percentage varying by payment method and region. Compute your own effective rate from your actual mix of transactions rather than assuming a headline number.

## Step 1 — set up the usage counter

The counter needs to answer one question quickly: how many times has this user or team used the tool this billing period? Two storage options cover most cases.

**Redis** is the right choice when many processes or machines share the counter and you want atomic increments. A single small instance handles a large number of counters comfortably, since each counter is a few bytes.

**SQLite in WAL mode** is the right choice when the tool runs locally and the counter can be reconciled periodically. It avoids a network hop entirely and has no per-month floor cost beyond disk.

Install Redis on a Debian-based system:

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

Do not expose Redis to the public internet. Bind it to localhost or a private interface and restrict access at the firewall. An open Redis port is one of the most common ways small services get compromised.

A Lua script keeps the increment and read atomic, so two concurrent invocations cannot both read a stale count:

```lua
-- incr_monthly.lua
-- KEYS[1] = user or team id
-- KEYS[2] = plan name
-- ARGV[1] = period key, e.g. "2026-03"
local user_id = KEYS[1]
local plan = KEYS[2]
local period = ARGV[1]
local month_key = "usage:" .. period

local count = redis.call("HINCRBY", month_key, user_id, 1)
return { count, plan }
```

Note the period is passed in as an argument rather than computed inside the script. Redis's Lua sandbox does not provide a reliable wall-clock date, and relying on the server's clock couples your billing period to server configuration. Let the caller decide the period. Invoke it like this:

```bash
redis-cli --eval incr_monthly.lua user123 growth , 2026-03
```

The comma separates keys from arguments in `redis-cli --eval`. Getting that separator wrong is a common source of confusing errors.

## Step 2 — enforce the plan in the CLI

The plan guard runs before the tool's real work. It reads the plan, increments the counter, and compares the result against the tier's allowance.

```javascript
// plan-guard.js
import { execFileSync } from 'node:child_process';

const LIMITS = {
  free: 1000,
  starter: 5000,
  growth: 50000,
  scale: 500000,
};

const UPGRADE_URL = 'https://example.com/upgrade';

export function checkUsage({ userId, plan, period }) {
  const limit = LIMITS[plan];
  if (limit === undefined) {
    throw new Error(`Unknown plan: ${plan}`);
  }

  let count;
  try {
    const out = execFileSync(
      'redis-cli',
      ['--eval', 'incr_monthly.lua', userId, plan, ',', period],
      { encoding: 'utf8', timeout: 200 }
    ).trim();
    count = Number(out.split(',')[0]);
  } catch (err) {
    // Fail open, but record it. See the failure-mode section below.
    return { allowed: true, degraded: true, count: null, limit };
  }

  if (count > limit) {
    return {
      allowed: false,
      degraded: false,
      count,
      limit,
      message:
        `Usage limit reached: ${count}/${limit} on the ${plan} plan.\n` +
        `Upgrade for a higher allowance: ${UPGRADE_URL}?plan=starter&source=cli`,
    };
  }

  return { allowed: true, degraded: false, count, limit };
}
```

Two details matter here. First, `execFileSync` with an argument array avoids shell interpolation, so a user ID containing spaces or shell metacharacters cannot break the command. Second, the failure path is explicit: if the counter is unreachable, the guard fails open and marks the result as degraded. Blocking a paying user because your counter is down is worse than letting a few extra calls through.

The caller then decides what to do:

```javascript
const result = checkUsage({ userId, plan, period });

if (!result.allowed) {
  console.error(result.message);
  process.exit(1);
}
if (result.degraded) {
  console.error('Usage service unavailable; running without plan enforcement.');
}
```

Feature gating is separate from usage gating and is simpler still. A plan-to-features map, loaded once at startup, is enough:

```yaml
# plans.yml
plans:
  free:
    seats: 1
    monthly_calls: 1000
    features: [basic_formatting]
  starter:
    seats: 5
    monthly_calls: 5000
    features: [basic_formatting, custom_rules]
  growth:
    seats: 20
    monthly_calls: 50000
    features: [basic_formatting, custom_rules, team_sharing]
  scale:
    seats: 100
    monthly_calls: 500000
    features: [all_features]
```

The CLI checks `features.includes('custom_rules')` before enabling a code path. No network call, no database lookup.

## Step 3 — handle the failure modes

Pricing enforcement introduces new ways for a tool to break. Each one deserves an explicit decision.

**The counter is unreachable.** Failing closed (blocking the user) turns a Redis outage into a product outage. Failing open (allowing the call) risks a small amount of unbilled usage. For most developer tools, failing open is correct, provided the degraded state is logged so you can reconcile later. A circuit breaker that stops retrying a dead dependency is worth adding once the guard is on the hot path:

```javascript
import CircuitBreaker from 'opossum';

const breaker = new CircuitBreaker(
  async () => execFileSync('redis-cli', ['--eval', 'incr_monthly.lua', userId, plan, ',', period], { encoding: 'utf8' }),
  { timeout: 50, errorThresholdPercentage: 50, resetTimeout: 30000 }
);

breaker.fallback(() => null); // null signals "degraded, allow"
```

Check the library's current documentation for option names and behavior; circuit breaker APIs change between major versions.

**One user, many machines.** A seat is a person, but a person may install the tool on a laptop, a desktop, and a CI runner. Hashing a machine fingerprint and counting each as a seat punishes legitimate use. A better approach is to count distinct authenticated identities, and to treat CI systems as a separate, explicitly allowed class. If you must limit machines, do it as a documented policy with a clear error message, not as a silent failure.

**Plan changes mid-period.** Proration belongs to the billing provider. The application only needs to know the current plan and the current period's usage. When a plan changes, either keep the same counter and apply the new limit, or reset the counter and note the reset in the provider's metadata. Pick one and document it, because users will notice inconsistent behavior.

**Clock and timezone drift.** Billing periods must be defined in a single timezone, usually UTC, and the period key must be computed from that definition. Computing the month from the client's local clock produces off-by-one-day bugs at month boundaries.

**Error messages nobody reads.** A generic "limit reached" message wastes the moment when a user is most willing to pay. Include the actual numbers and a direct upgrade path:

```
Usage limit reached: 1001/1000 on the free plan.
Upgrade for a higher allowance: https://example.com/upgrade?plan=starter&source=cli
```

## Step 4 — measure whether it worked

A pricing change is a hypothesis. Treat it like one.

**What to instrument.** At minimum, record: plan per active user, usage count per period, the ratio of usage to limit, and every upgrade event with the time since the user last hit a limit. The last one is the most informative — it tells you how much friction precedes a purchase.

**How to measure latency impact.** If the plan guard adds a network hop, measure it. `hyperfine` is a convenient tool for comparing two command variants:

```bash
hyperfine --warmup 10 \
  'formatter --no-plan-check file.txt' \
  'formatter file.txt'
```

Run it on the same machine, ideally the slowest machine your users have. Compare the means and, more importantly, the tail — a guard that is fast on average but occasionally blocks for 200ms is worse than one that is consistently slow.

**How to measure conversion.** A pricing experiment needs a denominator. Count visitors to the pricing page, count checkout starts, and count completed subscriptions. Report conversion as a rate with its sample size, not as a bare percentage. A "10% lift" on 40 visitors is noise.

**How to run the test.** Split traffic between the current page and a variant, keep the variant live for a fixed period decided in advance, and avoid changing anything else during the window. If the tool is deployed on a platform with built-in A/B routing, use it. Otherwise, serve the variant from a second URL and route a fraction of traffic to it at the edge.

**A worked example of the arithmetic.** Suppose a pricing page receives 1,000 visitors over a two-week test. The control page converts 2.0% (20 subscriptions) and the variant converts 2.6% (26 subscriptions). The difference is 6 subscriptions. With samples this small, that difference is well within normal variation — a two-proportion test would not come close to significance. To detect a lift of that size with confidence, you would need roughly an order of magnitude more visitors. This is the single most common mistake in pricing experiments: declaring a winner on a sample that cannot support the claim.

## A decision checklist before you ship

- Does the free tier let a new user reach the "aha" moment? If not, the limit is too tight.
- Does every tier have a stated limit in the same units the user thinks in (calls, projects, seats)?
- Is the upgrade path reachable from the exact moment the limit is hit?
- Does the guard fail open, and is the degraded state logged?
- Is the billing period defined in one timezone and computed the same way everywhere?
- Can you answer "how many users hit their limit last month?" from your metrics?
- Have you decided in advance how long the pricing test runs and what result would change your mind?

## An FAQ worth answering

**What if the tool is a library rather than a CLI?**
Gate at import time using the same plan map. Read the plan from an environment variable, load the feature list, and raise a clear error when a gated feature is used. Avoid raising on import itself — that breaks tooling and test suites. Raise at the call site, with a message that names the feature and the upgrade URL.

**Should there be an enterprise tier without a self-serve checkout?**
Often yes. A high-priced tier with a contact form filters out casual inquiries and surfaces real intent. Route submissions to a shared inbox and track response time as a metric. The form is not a sales team; it is a qualification step.

**Do annual plans help?**
They help mature tools with sticky, recurring usage, because they reduce churn and improve cash flow. For early tools, annual plans can lock in a price before you understand your own costs, and the churn at renewal can be higher than monthly churn. If you offer annual, offer it after you have several months of retention data.

**What about regional payment methods?**
Card fees and failure rates vary by region. Offering local payment rails can improve conversion in markets where card penetration or card success rates are low. Check your provider's supported methods and the effective fee for each before assuming it is worth the integration.

## Do this in the next 30 minutes

Open your pricing page source, pick the tier directly above your most common paid tier, and raise its price by 50%. Deploy it as a variant behind a 50/50 split, and add one event to your analytics that fires when a user hits a usage limit. You do not need a conclusion today — you need a denominator. The number that matters is not the conversion rate; it is the conversion rate alongside the sample size it was computed from.
