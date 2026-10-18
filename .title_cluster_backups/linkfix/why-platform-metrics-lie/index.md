# Why platform metrics lie

## Why aggregate platform metrics fail as evidence of value

Platform teams are often asked to justify their existence with a chart. The chart usually shows a p99 latency line that has barely moved for months, or a request-per-second count that climbs in lockstep with traffic. Those numbers are technically correct, but they rarely answer the question product owners actually ask: *what did the platform do for us this quarter?*

The failure mode is structural, not political. A single global percentile collapses regional variance into one number. A request counter tracks demand, not reliability. A shared cloud bill mixes platform and product workloads, so no dollar figure can be attributed to a platform change. The result is a wall of dashboards that looks sophisticated and informs almost nothing.

This article walks through the specific ways aggregate metrics mislead, then describes a composite metric — a **contribution index** — that combines latency, error rate, and cost share per region. It includes the formula, the instrumentation, the failure modes, and a checklist for deciding whether a composite metric is the right tool for a given platform.

## The three constraints that break single-number metrics

A typical platform organization spans several regions. Suppose, illustratively, four: US-East, EU-West, AP-SouthEast, and West Africa. The platform team owns a shared authentication service, a rate-limiting gateway, and a metrics collector.

Two constraints tend to emerge immediately:

1. **Regional latency variance.** Users in a distant region might see 250 ms p99 while the nearest region sees 80 ms. A single global p99 — say 120 ms — is arithmetically defensible and operationally useless, because it hides the outlier that users actually feel.
2. **Cost attribution granularity.** A shared cloud bill mixes platform and product workloads. Without per-workload usage data, no dollar figure can be assigned to a platform improvement, so cost arguments collapse into assertions.

A third constraint usually follows:

3. **Failure-mode visibility.** Error rates and latency are often plotted on separate panels. Correlation between a latency spike and a timeout surge is then invisible to anyone who is not staring at both charts simultaneously.

Each of these is a *constraint* in the sense that it is a specific, named shortcoming of the current measurement approach. Naming the constraint before designing the metric is what keeps the metric from becoming another vanity chart.

## What a naive regional breakdown gets wrong

The obvious first fix is to publish the existing metrics broken down by region. Add panels for `gateway_latency_seconds{region="us-east-1"}`, `gateway_latency_seconds{region="af-south-1"}`, and so on, plus a cost allocation table driven by resource tags.

This fails for three predictable reasons:

- **Over-splitting.** Each panel shows a single number with no context. An engineer sees 250 ms for one region, dismisses it as "network lag," and stops looking.
- **No composite signal.** Product owners cannot see the relationship between latency spikes and error rates when the two live on separate panels.
- **Manual tagging overhead.** Tag-based cost allocation requires every team to tag every resource correctly. In practice, a meaningful fraction of tags are missing or wrong, and the resulting split is misleading in a way that is hard to detect.

The output is a dashboard that adds noise instead of clarity. Stakeholders then ask for a single trustworthy number — which is a reasonable request, and one that a composite metric can answer if it is designed carefully.

## Designing a contribution index

A contribution index (CI) is a weighted composite that satisfies the three constraints above. The design has three parts.

### Weighting by active users

Compute a weighted p99 rather than a flat global p99. Weight each region by active user count so that a small user base cannot dominate the global view. The weighted percentile is not the same as the percentile of the pooled sample; it is the value below which 99% of *user-weighted* observations fall.

### Allocating cost by usage, not tags

Replace tag-based allocation with usage-based allocation. Pull CPU-seconds and network-bytes per region from the cloud provider's cost and usage reports, filtered by the usage types that belong to the platform's services. Derive a dollar figure from the published on-demand price list. This removes the tagging dependency entirely.

### Embedding failure modes as a penalty

Include the HTTP 5xx rate as a penalty factor inside the index rather than as a separate panel. A latency improvement accompanied by an error-rate regression should not read as a win.

### A reference implementation

The following Python computes a weighted p99 and a contribution index. It is deliberately small so that the arithmetic is auditable.

```python
import numpy as np

def weighted_p99(latencies, weights):
    """Return the 99th percentile of latencies weighted by `weights`.

    latencies and weights are 1-D arrays of the same length.
    """
    latencies = np.asarray(latencies, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(latencies)
    sorted_lat = latencies[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    cutoff = 0.99 * cum_w[-1]
    idx = np.searchsorted(cum_w, cutoff)
    return float(sorted_lat[min(idx, len(sorted_lat) - 1)])

def contribution_index(p99_ms, error_rate, cost_usd,
                       weight=1.0,
                       latency_ceiling_ms=200.0,
                       cost_ceiling_usd=5000.0):
    """Composite score in [0, 100]. Higher is better.

    All ceilings are illustrative; set them from your own SLOs and budget.
    """
    latency_score = max(0.0, latency_ceiling_ms - p99_ms) / latency_ceiling_ms
    error_score = max(0.0, 1.0 - error_rate)
    cost_score = max(0.0, 1.0 - cost_usd / cost_ceiling_usd)
    raw = (latency_score * 0.5 + error_score * 0.3 + cost_score * 0.2) * weight
    return round(raw * 100.0, 2)
```

Two properties matter here. First, every ceiling and weight is an explicit parameter, so the assumptions are visible in the code rather than buried in a dashboard query. Second, the function is pure: given the same inputs it returns the same output, which makes it testable.

### Worked example

Suppose a region reports p99 = 250 ms, error rate = 0.03, and cost = 4,000 USD, with weight = 1.0. Using the ceilings above:

- latency_score = (200 − 250) / 200 = −0.25, clamped to 0.0
- error_score = 1.0 − 0.03 = 0.97
- cost_score = 1.0 − 4000 / 5000 = 0.20
- raw = (0.0 × 0.5) + (0.97 × 0.3) + (0.20 × 0.2) = 0.291 + 0.04 = 0.331
- CI = 33.1

Now suppose the same region improves p99 to 150 ms while cost rises to 4,500 USD and error rate stays at 0.03:

- latency_score = (200 − 150) / 200 = 0.25
- error_score = 0.97
- cost_score = 1.0 − 4500 / 5000 = 0.10
- raw = (0.25 × 0.5) + (0.97 × 0.3) + (0.10 × 0.2) = 0.125 + 0.291 + 0.02 = 0.436
- CI = 43.6

The index rises even though cost rose, because the latency gain outweighed the cost penalty under these weights. Change the cost weight to 0.4 and the latency weight to 0.3 and the result flips. That sensitivity is the point: the weights encode a business judgement, and they should be argued about explicitly rather than hidden.

## Instrumentation: what to record and how to verify it

A composite metric is only as good as the data underneath it. The following instrumentation is sufficient for the design above.

- **Latency histograms per region.** Export a histogram (not a summary) from the gateway so that percentiles can be recomputed server-side over arbitrary windows. Scrape every 15 seconds with a Prometheus-compatible collector.
- **Error counters per region.** A counter for HTTP 5xx responses, labelled by route and region, so that error rate can be computed as a ratio against total requests.
- **Usage-based cost inputs.** CPU-seconds and network-bytes per region from the provider's cost and usage report, plus the published on-demand unit prices. Record the price list version alongside the derived figure so that a price change does not silently move the index.
- **Active user counts per region.** From authentication or session logs, aggregated to the same window as the latency data.

To verify the weighted p99, cross-check it against a manual calculation on the same latency series. Sort the observations, cumulate the weights, and find the value at 99% of total weight. The two numbers should agree to within floating-point tolerance. If they do not, the most common cause is a mismatch between the window used for the histogram and the window used for the weights.

To verify the cost component, compare the derived per-region figure against the provider's own cost report for the same period. A persistent discrepancy usually means a usage type is being double-counted or missed.

## Failure modes of composite metrics

Composite metrics have their own failure modes, and they are worth stating plainly.

- **Gaming.** Once a number is published, it becomes a target. A team can raise the index by cutting cost (for example, by reducing instance size) while degrading latency, if the weights allow it. Mitigation: publish the component scores alongside the aggregate so that a shift in one component is visible.
- **Weight staleness.** Weights chosen for one traffic distribution become wrong as traffic shifts. A region that was 5% of users and is now 30% should not carry the old weight. Mitigation: recompute weights on a fixed cadence and record when they last changed.
- **Opacity.** A single number invites the question "why did it drop?" If the answer is not immediately available, trust erodes. Mitigation: every alert should name the offending region and the component that moved.
- **False precision.** A score of 73.5 implies more resolution than the underlying data supports. Rounding to the nearest whole number is usually more honest.
- **Correlation with the wrong thing.** If the index is tuned until it matches a business metric, it stops being a measurement and becomes a fit. Mitigation: define the formula before looking at the outcome it is supposed to predict.

## Decision checklist: is a composite metric right for you?

Not every platform needs one. Use the following checklist before building.

- Do you have more than one region, tier, or tenant with materially different performance characteristics? If not, a single percentile may be sufficient.
- Can you attribute cost to the platform's services using usage data rather than tags? If not, fix attribution first; a composite that includes a wrong cost term is worse than no cost term.
- Do you have a named owner for each component (latency, errors, cost)? If not, the composite will diffuse responsibility rather than clarify it.
- Can you state the business judgement behind each weight in one sentence? If not, the weights are arbitrary and will be argued about indefinitely.
- Will you commit to publishing component scores alongside the aggregate? If not, expect gaming.
- Do you have a cadence for revisiting weights? If not, the index will drift out of relevance.

If most answers are yes, a composite metric is worth building. If several are no, the honest move is to improve the underlying measurements first.

## How to measure whether the index is working

The index itself is not evidence that the platform is valuable. To evaluate it, instrument the following:

- **Time-to-answer.** How long does it take a product owner to get a satisfactory answer to "what changed?" Measure this before and after by timing a sample of real questions.
- **Alert precision.** Of the alerts fired on the index, what fraction corresponded to a real incident that someone acted on? Track this over a quarter.
- **Component attribution.** When the index moves, can the on-call engineer name the component and region within one minute? If not, the alert is under-specified.
- **Weight stability.** How often do the weights change, and does each change correspond to a documented traffic shift?

None of these require a benchmark table; they require a log of questions, alerts, and outcomes.

## Common questions

**How can regions be weighted without accurate user counts?**
Use request volume per region from load balancer logs as a proxy. Even a rough distribution is a better fairness baseline than a flat average, provided the proxy is documented.

**Why does the index drop when latency improves?**
If cost rises faster than latency falls, the cost penalty can outweigh the latency gain under the chosen weights. The index is designed to surface trade-offs, not to reward single-metric wins.

**What alert threshold is reasonable?**
Start from the observed distribution of the index over a baseline period. A drop of several points sustained across two evaluation windows is a common starting point. Tune against historical variance rather than adopting a number from elsewhere.

**When should weights be recomputed?**
On a fixed cadence, and immediately after a major traffic shift such as a new market launch. Record the previous weights so that a change can be attributed.

**How should the index be exposed to non-technical stakeholders?**
Pair the aggregate with the component breakdown and a one-line explanation of the largest mover. A number without a cause invites the wrong conversation.

**What if the platform services are serverless?**
Replace CPU-seconds with invocation duration and invocation count for cost attribution. The structure of the formula is unchanged; only the cost inputs differ.

**Why not publish raw latency and error numbers instead?**
Raw numbers are noisy and require context to interpret. A composite condenses several dimensions into one comparable figure, at the cost of transparency — which is why the component scores should always be published alongside it.

**How long does implementation take?**
It depends almost entirely on whether usage-based cost data is already available. If it is, the metric itself is a few hundred lines of code. If it is not, attribution is the project.

## The one action to take now

Open the dashboard your platform team shows to stakeholders and pick the single metric that most often prompts the question "but what about region X?" Write down, in one sentence, the constraint that metric fails to capture. That sentence is the specification for the first component of your composite metric — and it takes less than thirty minutes to produce.
