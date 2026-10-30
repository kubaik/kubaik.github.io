# AI in local currency: costs that surprise…

## Why "run inference near your users" is an incomplete rule

The common starting question is: "Can we run this model in the region closest to our users to keep latency low?" That is a reasonable first instinct, but it treats a multi-variable optimisation problem as a single-variable one. The chain that actually determines your bill and your compliance posture looks more like:

latency → region → hardware availability → currency → FX exposure → data residency → audit → total cost

Teams that stop at "region = latency" often discover the rest of the chain months later, when usage grows, when a quota request stalls, or when finance asks which line item the GPU spend belongs to. This article works through the failure modes of the simple rule, proposes a cost model that separates variable from fixed costs, and gives a decision procedure you can apply to your own workload.

A note on numbers: every figure below is either a documented default/limit, arithmetic shown from stated assumptions, or explicitly labelled illustrative. Where a real benchmark would help, the article explains how to produce it yourself rather than quoting one.

## What the simple rule gets wrong

### Capacity is not fungible across regions

GPU capacity in one region cannot be moved to another with a configuration change. Instance families, accelerator generations, and per-account quotas differ by region and by availability zone. A typical failure mode: a team provisions capacity in a region where quota was easy to obtain, then demand shifts to a different region. The options are to pay on-demand rates in the new region, or to let requests queue or fail.

The documented behaviour to check before committing: per-region service quotas for the specific accelerator instance family you need, and the support process and typical turnaround for a quota increase. Quota increases are not instantaneous and are not guaranteed. If your capacity plan assumes a quota you do not yet hold, you are carrying an unhedged risk.

### Data residency and audit often force duplication

Residency rules generally constrain where data may be stored and, depending on the jurisdiction, where it may be processed. Audit requirements add a second obligation: retaining records of what was processed, when, and by which model version. Neither obligation scales down when usage is low. If a regulation requires raw inputs to remain in-country, you need storage and a logging path there regardless of whether you run one request a day or a million.

The consequence is that compliance cost is largely fixed per region, while compute cost is variable per request. Treating them as one number hides the trade-off that matters.

### Local currency billing is not local currency cost control

Cloud providers publish list prices in local currencies, but the conversion is applied according to the provider's own rates at invoice time, not rates you lock in at purchase. If your compute is priced in one currency and you bill customers in another, you carry FX exposure on the compute line. The size of that exposure depends on the currency pair and the period; it is not a fixed percentage you can assume. Measure it by comparing your internal budget rate against the invoiced rate for several consecutive months — that gives you a realised variance distribution rather than a guess.

### Reserved capacity is a bet on demand

Reserved or committed-use discounts trade flexibility for a lower effective hourly rate. The arithmetic is straightforward: if a one-year commitment reduces the effective rate by some percentage, you break even only if you actually consume enough hours. If adoption plateaus, you keep paying for capacity you are not using. Resale markets exist for some commitment types, but the price you receive depends on remaining term and current demand, and there is no guarantee it recovers your cost.

The practical discipline: before committing, model the downside case where usage is flat or declining, not just the growth case.

## A cost model that separates variable from fixed

Instead of starting from latency, start from a cost-per-request model plus a fixed per-region term:

```
total_monthly_cost ≈ (cost_per_request × request_volume) + Σ (fixed_cost_per_active_region)
```

`cost_per_request` is driven by things you can influence:

- hardware efficiency (accelerator generation, memory bandwidth, whether the workload is compute- or memory-bound)
- model optimisation (quantisation, distillation, smaller architectures where accuracy permits)
- batching (larger batches amortise kernel launch and memory transfer overhead)
- cold-start avoidance (provisioned concurrency or warm pools, which trade money for latency)

`fixed_cost_per_active_region` is driven by constraints you mostly cannot influence away:

- residency obligations that require storage or processing in a jurisdiction
- audit retention (log volume × retention period × storage price)
- region-specific certifications or contractual requirements
- the operational cost of running and monitoring an additional stack

The model explains the counterintuitive result directly: adding a region adds a fixed term, so it only pays off if the variable savings (or the latency or compliance benefit) exceed that fixed term at your actual volume.

### A worked break-even example

Suppose, for illustration, that cross-region inference costs $0.002 per request and in-region inference costs $0.0025 per request because of hardware availability or pricing differences. The per-request penalty for running in-region is $0.0005.

If the fixed cost of maintaining the in-region stack (compliance logging, audit storage, extra monitoring) is $800 per month, the break-even volume is:

```
$800 / $0.0005 per request = 1,600,000 requests per month
```

Below roughly 1.6 million requests per month, the in-region option costs more in total, even though it is better on latency. Above it, the in-region option wins on total cost. These are illustrative figures; substitute your own measured per-request costs and your own fixed-cost estimate. The point is the shape of the calculation, not the specific numbers.

Note what the model does *not* say: it does not say latency is unimportant. It says latency must be priced. If a latency improvement is worth more than the cost difference to your business, run in-region and account for it as a deliberate purchase, not as an assumed saving.

## How to measure your own numbers

You cannot substitute someone else's benchmark for your own workload. Instrument the following:

1. **Per-request cost.** Record GPU/accelerator hours consumed per request (or per batch), multiply by the effective hourly rate for that instance type in that region, and divide by requests served. Track this per model version, because optimisation changes it.
2. **Latency distribution.** Measure p50, p95, and p99 end-to-end latency, not just model inference time. Network transit between regions is part of the user-visible number. Compare the same workload deployed in two candidate regions with a synthetic or replayed traffic mix.
3. **Error and retry rate.** Cross-region calls add failure modes: timeouts, partial responses, and retries that multiply cost. Measure retries as a separate line, because a retry is a second full request.
4. **Fixed per-region cost.** Sum audit log storage, residency-mandated storage, monitoring, and the engineering time attributable to maintaining the extra stack. Engineering time is a real cost even though it does not appear on a cloud invoice.
5. **FX realised variance.** For each month, compare the budget rate you used with the rate actually applied on the invoice. After several months you have a distribution, which is more useful than a single assumed percentage.

A simple harness: deploy the same model and configuration to two regions, replay a fixed request set against both, and log per-request latency, accelerator utilisation, and cost. Repeat at low and high concurrency. The difference between the two regions at high concurrency is usually larger than at low concurrency, because queueing and throttling effects appear under load.

## Failure modes to design against

**Quota exhaustion under load.** Autoscaling cannot exceed your quota. If your scaling policy assumes headroom you do not have, the failure appears as latency spikes and errors, not as a clean error message. Pre-request quota above your projected peak, and alert on utilisation approaching the quota.

**Silent fallback to slower hardware.** Some serving stacks fall back to CPU or a smaller accelerator when the preferred device is unavailable. CPU inference can be an order of magnitude slower and, depending on instance pricing, more expensive per request. Alert on which device actually served each request, not just on whether the request succeeded.

**Artifact drift across regions.** Multiple regions mean multiple copies of model artifacts and configuration. A rollback or a partial deploy can leave one region on a different model version. Validate a checksum or version identifier at startup and refuse to serve if it does not match the expected value. This is a small amount of code that prevents a class of hard-to-diagnose incidents.

**Retry storms.** A cross-region call that times out and is retried can double or triple load on the remote region precisely when it is already struggling. Use bounded retries with jittered backoff, and circuit-break rather than retrying indefinitely.

**Compliance drift.** A routing change made for cost reasons can move data across a boundary without anyone noticing. Encode residency constraints in infrastructure policy or network controls, not in a document that people are expected to remember.

## When running in the user's region is the right call

The simple rule is not wrong; it is incomplete. It is correct when one of the following holds:

- **The application is genuinely interactive.** Live captioning, real-time translation during a call, or conversational interfaces have latency budgets where inter-region round-trip time is a meaningful fraction of the total. Measure the round-trip time between candidate regions and your user population before assuming this.
- **Residency rules require processing in-country, not just storage.** Some jurisdictions constrain processing, not only storage. Where that is the case, the decision is made for you; the engineering task is to make the in-region stack as efficient as possible.
- **Contractual SLAs commit you to a latency or availability figure.** If enterprise contracts specify latency, the cost of the extra region is a cost of meeting the contract, and should be priced into the contract.
- **Currency volatility on your cost side exceeds the compute savings.** If your revenue currency is volatile relative to your cost currency, matching them can reduce variance even at a higher expected cost. Variance reduction has value when you have budget commitments.

## A decision checklist

Work through these in order. Stop at the first one that determines the answer.

1. Does a regulation require processing in a specific jurisdiction? If yes, that region is mandatory. Optimise within it.
2. Does a contract specify a latency or availability figure you cannot meet cross-region? If yes, deploy to meet it and price the cost into the contract.
3. What is your measured per-request cost difference between candidate regions, at realistic concurrency? If you have not measured it, measure it before deciding.
4. What is the fixed monthly cost of each additional region, including audit storage, residency storage, monitoring, and engineering time?
5. Divide fixed cost by per-request difference to get the break-even volume. Compare against your actual and projected volume.
6. If volume is below break-even and no constraint forces the region, run in the cheaper region and treat the latency difference as a measured, accepted trade-off.
7. If you operate multiple regions, enforce residency and version consistency in infrastructure policy, and alert on device type, retry rate, and quota utilisation.

## Comparison of the main approaches

| Approach | Latency | Residency risk | FX exposure | Cost predictability | Operational overhead |
|---|---|---|---|---|---|
| Single region, user's region | Lowest | Lowest | Depends on currency match | Medium | Lowest |
| Single region, cheapest compliant | Highest | Medium | Depends on currency match | Highest | Lowest |
| Multi-region, per jurisdiction | Lowest per region | Lowest | Per-region | Lowest | Highest |
| Primary plus overflow region | Medium | Medium | Mixed | Medium | Medium |

The table is qualitative on purpose. The magnitudes depend on your measured per-request costs, your fixed per-region costs, and your volume, which is exactly why the checklist above asks you to measure rather than assume.

## Common objections

**"Running inference outside the user's jurisdiction violates residency rules."**

Residency rules vary, and some distinguish storage from processing. Some do not. The only safe approach is to read the specific regulation or contract that applies to you and, where the interpretation matters, get it confirmed by the people accountable for compliance. Do not generalise from one jurisdiction's rules to another's. Where the rule permits storage in-jurisdiction with processing elsewhere, the engineering pattern is to keep raw inputs and outputs in the required region and treat the compute region as transient, with encryption in transit and no plaintext persistence outside the boundary. Where the rule constrains processing, that pattern is not available and the compute must be in-region.

**"Cross-region latency will ruin the experience."**

This depends entirely on the interaction model. For asynchronous workloads — upload a file, receive a transcript later — a few hundred milliseconds of added network time is usually imperceptible relative to the total wait. For synchronous, turn-taking interactions, it is not. Categorise your features by interaction model, measure the actual added latency for each, and route accordingly. A feature flag that selects the region per request lets you change the split without a redeploy.

**"Reserved instances always save money."**

They save money if you consume the committed hours. They cost money if you do not. Model the flat-usage and declining-usage cases before committing, and check the terms for the commitment types you are considering, including whether resale is possible and on what terms.

## A minimal routing sketch

The following is a starting point for encoding the decision as code. Replace the placeholder costs and latencies with measured values.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Workload:
    country: str
    latency_sensitive: bool
    processing_must_be_local: bool
    monthly_requests: int

@dataclass
class RegionOption:
    name: str
    cost_per_request: float      # measured, in your budget currency
    fixed_monthly_cost: float    # audit + residency + monitoring, budget currency
    added_latency_ms: int        # measured, relative to user's region

def choose_region(
    workload: Workload,
    candidates: list[RegionOption],
) -> RegionOption:
    # Constraint 1: processing must be local -> no choice.
    if workload.processing_must_be_local:
        local = [r for r in candidates if r.name.startswith(workload.country.lower())]
        if not local:
            raise ValueError(f"no local region configured for {workload.country}")
        return min(local, key=lambda r: r.cost_per_request)

    # Constraint 2: latency-sensitive -> prefer lowest added latency.
    if workload.latency_sensitive:
        return min(candidates, key=lambda r: r.added_latency_ms)

    # Otherwise: minimise total cost at this volume.
    def total(r: RegionOption) -> float:
        return r.cost_per_request * workload.monthly_requests + r.fixed_monthly_cost

    return min(candidates, key=total)

# Illustrative inputs only - substitute measured values.
workload = Workload(
    country="de",
    latency_sensitive=False,
    processing_must_be_local=False,
    monthly_requests=500_000,
)
candidates = [
    RegionOption("eu-central-1", cost_per_request=0.0025, fixed_monthly_cost=800, added_latency_ms=20),
    RegionOption("us-east-1",    cost_per_request=0.0020, fixed_monthly_cost=0,   added_latency_ms=110),
]
print(choose_region(workload, candidates))
```

For the illustrative inputs above, the totals are:

```
eu-central-1: 0.0025 * 500,000 + 800 = 1,250 + 800 = 2,050
us-east-1:    0.0020 * 500,000 + 0   = 1,000
```

So the cross-region option is cheaper at this volume by the model's own arithmetic. Change the volume to 2,000,000 requests and the totals become 5,800 versus 4,000; the gap narrows proportionally but the cross-region option still wins here because the per-request difference dominates. The break-even is where the two expressions are equal:

```
0.0025v + 800 = 0.0020v
0.0005v = 800
v = 1,600,000
```

Below 1,600,000 requests per month the cross-region option is cheaper; above it the in-region option is. That is the same break-even derived earlier, now produced by the code rather than asserted.

## FAQ

**Doesn't GDPR require EU data to be processed in the EU?**

GDPR governs the processing of personal data of people in the EU and imposes conditions on transfers outside the EU; it does not, by itself, specify that a GPU must be physically located in the EU. Whether a particular architecture complies depends on the specific processing, the transfer mechanism used, and the contractual and organisational safeguards in place. Treat this as a question for your data protection officer or legal counsel, not as a general engineering rule.

**How do I compare per-request cost fairly across regions?**

Hold the model, batch size, and concurrency constant, replay the same request set, and measure accelerator hours consumed per request alongside end-to-end latency. Convert to a single budget currency using a rate you state explicitly, and record the rate so you can recompute later if it changes.

**What if my volume is seasonal?**

Size your committed capacity for the trough, not the peak, and cover the peak with on-demand or spot capacity. Spot capacity can be reclaimed, so design the serving path to tolerate interruption: drain in-flight requests, retry elsewhere, and never let a single spot instance be the only path for a request class.

**Should I run the same model in every region?**

Only if residency or latency requires it. Running multiple copies multiplies the artifact-management and version-consistency surface. If you do, validate a version identifier at startup in each region and alert on mismatch.

**How do I keep the decision from drifting?**

Encode residency constraints as infrastructure policy rather than documentation, alert on the device type that actually serves requests, and review the break-even calculation when either your volume or your measured per-request cost changes materially.

## Action for the next 30 minutes

Pick one production inference endpoint, query your billing or cost-explorer data for its accelerator hours over the last full month, and divide by the number of requests served in the same period to get a measured cost per request. Write that number down next to the effective hourly rate and the region it ran in. That single figure is the input every calculation in this article depends on, and most teams have never computed it.
