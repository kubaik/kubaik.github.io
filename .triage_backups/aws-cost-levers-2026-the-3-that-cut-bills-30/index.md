# AWS Cost Levers: Savings Plans vs. Compute Optimizer

## The mistake this article addresses

A common failure mode in AWS cost work looks like this: a team adopts a tool that produces recommendations, treats the recommendation list as the deliverable, and never verifies that spend actually fell. The recommendation engine is free, the dashboard is green, and the bill is unchanged. The opposite mistake is just as common: a team buys a commitment to get a discount, then refactors the workload two months later and discovers the commitment no longer matches what it runs.

Both failures share a root cause. Commitments and recommendation engines are different kinds of levers, and they are usually discussed as if they were alternatives. One is a pricing instrument that changes the rate you pay for compute you already run. The other is an analysis tool that changes what you run. They address different halves of the bill and they fail in different ways.

This article separates the two, describes what each actually does, shows where each breaks, and ends with a decision framework and a concrete first step.

## What Savings Plans actually are

A Savings Plan is a commitment to a consistent amount of compute usage, expressed in dollars per hour, for a one-year or three-year term. In exchange, AWS applies a discounted rate to eligible usage. The commitment is a floor on your spending, not a ceiling: usage above the committed amount is billed at on-demand rates.

The important structural properties:

- **The commitment is hourly and dollar-denominated.** You are not reserving a specific instance type, family, or region. You are reserving a rate of spend.
- **Discounts apply automatically to eligible usage**, with the deepest discount applied to the usage that matches best. You do not assign a plan to a workload.
- **Eligible services include EC2, Fargate, and Lambda**, so a mixed fleet can still consume the commitment. Exact eligibility and discount rates vary by plan type and region and are documented in the AWS Savings Plans user guide; check the current figures rather than trusting a number from an article.
- **Unused commitment is still billed.** If you commit to $3/hour and run $2/hour of eligible usage, you pay for $3/hour.

The last point is the entire risk model. A Savings Plan is a bet that your eligible baseline usage will stay at or above the committed level for the term. If the bet is right, the discount is real and requires no engineering work. If the bet is wrong, you pay for capacity you did not use.

### Where Savings Plans break

The failure mode is not "the discount didn't apply." It is a mismatch between the commitment and the workload's shape.

Consider a service that runs 10 vCPUs overnight and 50 vCPUs during business hours. The daily average may look stable, but the commitment is evaluated hourly. Committing to the daily average means under-committing during peak hours (paying on-demand for the expensive hours) and over-committing overnight (paying for unused commitment). The right commitment level depends on the distribution of hourly spend, not the monthly total.

A second failure mode is drift. A team commits against a workload, then migrates that workload to a different service, moves it to a region with different pricing, or replaces it with a managed offering that is not eligible. The commitment survives; the usage that justified it does not. This is why a commitment decision should be revisited whenever the architecture changes, and why long terms deserve more scrutiny than short ones.

## What Compute Optimizer and Cost Anomaly Detection actually are

These are two separate services with different jobs.

**Compute Optimizer** analyzes historical utilization and produces recommendations: instance types and sizes, whether a workload would run more efficiently on a different architecture, and similar findings. It is an analysis service. It does not change anything. Recommendations are based on observed history, which means they lag changes in workload behavior by the observation window, and they are only as good as the metrics they consume.

**Cost Anomaly Detection** monitors spend and alerts when it deviates from an expected pattern. It is a detection service. It does not prevent the anomaly and it does not explain it. It tells you that something changed, and you investigate.

Both are useful and neither reduces a bill by itself. The reduction comes from the change you make after reading them.

### Where these tools break

The characteristic failure is the unactioned report. A recommendation list is generated, reviewed once, and then ignored. Because the tools do not apply changes, a recommendation that is never implemented has exactly the same financial effect as one that was never generated.

The second failure is trust erosion through noise. Anomaly detection will flag planned changes — a load test, a traffic spike, a batch job that ran twice — as anomalies. Each false alarm costs investigation time, and enough of them cause the team to stop reading the alerts. A detection system nobody reads is worse than none, because it creates the impression of monitoring.

The third failure is staleness. A right-sizing recommendation derived from a month of low utilization is wrong if the workload's traffic pattern changed last week. Acting on stale recommendations can mean provisioning too little capacity, which trades a cost problem for a reliability problem.

## Comparing the two levers

The comparison that matters is not "which saves more" but "which problem does each solve."

| Property | Savings Plans | Compute Optimizer + Cost Anomaly Detection |
|---|---|---|
| What it changes | The rate paid for existing usage | What you run |
| Primary effect | Immediate, mechanical discount | Recommendations requiring engineering work |
| Effort to adopt | Low: choose a commitment amount | Moderate: enable, review, act |
| Effort to sustain | Low, but revisit after architecture changes | Recurring review cadence |
| Main risk | Over-commitment; commitment/workload drift | Unactioned reports; alert fatigue; stale advice |
| Failure mode | Paying for unused commitment | Paying the same bill with a nicer dashboard |
| When it stops helping | When usage falls below the commitment | When nobody implements the findings |

Two things follow from this table. First, the levers are complementary: a commitment reduces the rate on the baseline you intend to keep, and right-sizing reduces the baseline itself. Second, the risk profiles are opposite. Savings Plans fail by paying for something you don't use; the analysis tools fail by producing something you don't use. Both are execution problems, not tooling problems.

## A worked example, with the arithmetic shown

The numbers below are illustrative, chosen to make the reasoning visible. Substitute your own figures.

Suppose a service's eligible compute spend over the last 30 days is $9,000, and you want to size a one-year commitment. The naive approach is to divide by hours:

```
$9,000 / 30 days / 24 hours = $12.50 per hour
```

Committing $12.50/hour means matching the average exactly. Half your hours will be above it and half below. During the above-average hours you pay on-demand for the excess; during the below-average hours you pay for commitment you didn't consume.

A more conservative approach is to find the level that a high fraction of hours exceed. If hourly spend is at or above $10.00 for 90% of hours, committing $10.00/hour means:

- 90% of hours: the commitment is fully consumed, and the discount applies to $10.00/hour of usage.
- 10% of hours: the commitment is not fully consumed, and the shortfall is billed anyway.

The tradeoff is explicit. A lower commitment leaves more usage on-demand but reduces the risk of paying for nothing. A higher commitment captures more discount but increases the shortfall risk. There is no universally correct percentile; the choice depends on how confident you are that the baseline will persist for the term.

Now add the second lever. Suppose Compute Optimizer indicates the fleet is running at low average CPU and a smaller instance type would serve the same traffic. If that change reduces eligible spend from $9,000 to $6,000 per month, then a commitment sized against the old $9,000 baseline becomes over-commitment the moment the change ships. This is the drift failure mode in concrete form, and it is why the order of operations matters: **right-size first, then commit against the smaller baseline.** Committing first and optimizing second means paying a discount on capacity you no longer need.

The reverse order is defensible only when the optimization is uncertain or far in the future. A commitment delivers a known discount immediately; a right-sizing recommendation delivers an unknown discount after engineering work. If the work might not happen this quarter, committing against the current baseline is reasonable — provided you revisit the commitment when the work does happen.

## How to measure this for your own account

Do not trust any discount figure from an article, including this one. Measure it.

**For commitment sizing**, use Cost Explorer filtered to the eligible services (EC2, Fargate, Lambda) over a representative window of at least 30 days, ideally including a full monthly cycle. Export hourly or daily granularity rather than relying on the monthly total. From that series, compute the percentiles you care about — the 50th, 75th, and 90th — and compare each against the discount you would receive. The commitment level is a judgment call between captured discount and shortfall risk, and the percentile series is the input to that judgment.

**For recommendations**, the measurement is whether spend changed after implementation. Record the recommendation, the date it was implemented, and the eligible spend in the weeks before and after. A recommendation that was implemented and produced no measurable change is a signal that the recommendation model does not fit your workload. A recommendation that was never implemented is a process problem, not a tooling problem, and no amount of additional tooling will fix it.

**For anomaly detection**, track two rates over time: how many alerts corresponded to a real, unintended change, and how many did not. If the second rate is high enough that the team stops reading alerts, the detection configuration needs tuning — thresholds, or scoping to the services where anomalies are actually actionable.

## A decision checklist

Work through these in order. The order matters because the second question can invalidate the answer to the first.

1. **Is there a stable eligible baseline?** If eligible hourly spend has a floor that you expect to persist for the term, a commitment is a candidate. If the workload is genuinely spiky with no floor, a commitment mostly converts variable cost into fixed cost.
2. **Is a right-sizing or architecture change likely within the term?** If yes, either defer the commitment, shorten the term, or size the commitment against the post-change baseline. Committing against capacity you plan to remove is the most avoidable version of the over-commitment failure.
3. **What percentile of hourly spend are you willing to commit to?** Lower percentile, lower risk, less discount. Write down the number and the reasoning, so that the decision can be reviewed later rather than re-litigated.
4. **Who owns the recommendation review?** If the answer is "nobody in particular," the analysis tools will produce unactioned reports. Name the owner and the cadence, or skip the tooling.
5. **What triggers a commitment review?** Define it now: a migration, a service replacement, a region change, or a sustained drop in eligible spend. Without a trigger, the review never happens.

## Common questions

**Do Savings Plans cover Graviton instances?** Eligibility depends on the plan type and the instance family, and the discount is applied automatically to eligible usage. Rather than relying on a general claim, check eligibility for the specific families in your account against the current AWS documentation, since coverage has expanded over time and varies by plan type.

**Can Savings Plans be combined with Spot?** Yes. Spot capacity is priced separately, and a Savings Plan commitment applies to the eligible on-demand-rate usage you consume. The practical consequence is that a commitment sized against a baseline you intend to cover with Spot may go unused.

**Do these tools apply changes automatically?** Compute Optimizer produces recommendations; it does not modify resources. Cost Anomaly Detection alerts; it does not remediate. Any cost reduction from these services comes from work a human or an automation you wrote performs in response.

**When is the effort not worth it?** When the eligible spend is small enough that the absolute saving is trivial relative to the time spent reviewing reports and managing a commitment. The threshold is a judgment about your team's time, not a fixed dollar figure.

## Take this action in the next 30 minutes

Open Cost Explorer, filter to EC2, Fargate, and Lambda, and export the last 30 days at hourly granularity. Sort the hourly values and read off the 50th, 75th, and 90th percentiles. Write those three numbers down next to the discount you would receive at each level. You now have the actual input to a commitment decision for your account — not an estimate from someone else's workload — and you can see immediately whether a stable baseline exists or whether the hourly spend is too variable to commit against.
