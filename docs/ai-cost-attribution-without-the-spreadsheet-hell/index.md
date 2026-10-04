# AI cost attribution without the spreadsheet hell

## The gap between the demo and the invoice

AI features are cheap to prototype and expensive to operate. A prompt change, a new retry loop, a stale vector index, or one customer pasting a 40-page document into a chat box can move the monthly bill by an order of magnitude. The engineering team sees latency and error rates. Finance sees a single line item. Nobody sees which feature, which customer, or which model produced the spend.

Cost attribution is the layer that closes that gap. It is not a cost-cutting project. It is a visibility project: the output is a number a product manager can defend and act on, such as "this feature costs roughly two cents per active user per month, so we can either raise the price or cut usage by a fifth and hold margin."

Two failure modes dominate:

- A firehose of raw logs that nobody can query in time to answer a question.
- A spreadsheet that is two weeks stale by the time it reaches the person asking.

Both are symptoms of the same mistake: capturing data without deciding what question it must answer. Attribution only works when the dimensions you record match the decisions you intend to make.

## Decide the questions before you pick a tool

Every attribution system answers some subset of these:

1. Which feature costs the most in total?
2. Which customer or tenant costs the most, and is that reflected in their contract?
3. Which model or provider drives the spend, and would a cheaper model change the answer?
4. Is the cost per request trending up because prompts got longer, because retries increased, or because the mix of traffic shifted?
5. Can a spike be traced to a specific deploy, prompt template, or customer action?

If you cannot name the decision that a dimension will inform, do not record it. Each dimension multiplies the number of distinct time series you store, and cardinality is the most common reason attribution systems fall over.

### The cardinality budget

A metric with dimensions is stored as one time series per unique combination of dimension values. If you tag requests with user ID, feature, model, region, and status, and you have 50,000 monthly users, 12 features, 6 models, 3 regions, and 4 statuses, the theoretical maximum is 50,000 × 12 × 6 × 3 × 4 = 43.2 million series. In practice the observed combinations are far fewer, because most users touch one or two features, but the ceiling is what kills the backend.

The fix is to split dimensions by purpose:

- **Aggregatable dimensions** (feature, model, region, status) go on metrics. These have bounded value sets and are safe to group by.
- **Identity dimensions** (user ID, tenant ID, session ID, request ID) go on traces or event records, not on metrics. You query them individually when investigating, and you never group a metric by them.

A useful rule: if a dimension can take more than a few hundred distinct values in a month, it does not belong on a metric.

## What to instrument

The minimum useful record for a single AI call contains:

- A trace or request ID, so the call can be joined to logs and support tickets.
- The tenant or user identifier, propagated from the edge.
- The feature or feature-flag value that selected this code path.
- The model identifier and provider.
- Input tokens and output tokens, counted separately, because they are usually priced differently.
- Cached-token counts, if the provider bills them at a discount.
- Retry count and final status.
- Wall-clock latency.

Token counts must come from the provider response, not from a local tokenizer estimate, whenever the provider returns them. Local estimates drift from the billed count, and the drift is systematic rather than random, so it does not average out.

### Propagating identity through the request chain

The most common silent failure is losing the user ID somewhere between the edge and the inference call. Symptoms: a large share of usage attributed to a single `unknown` or `default` bucket, or attribution that matches the total bill but not any individual customer.

Propagate identity as a request header or context value set at the edge, and read it at every hop. In an AWS API Gateway to Lambda setup, a mapping template can inject the header into the event so the function does not depend on the client sending it. In a service mesh, the same value travels as a header between services.

Handle the disconnect case explicitly. If a client drops the connection mid-stream, the provider may still bill for the tokens generated. Record the identity at the start of the call, not at the end, so a cancelled request still carries its owner. If the identity genuinely is absent, write an explicit sentinel value rather than an empty string, because empty strings often get dropped by serializers and silently become `null`.

## Sampling: how to measure the error instead of guessing

At high volume, recording every call is wasteful. Sampling is the standard answer, but the error it introduces has to be measured, not assumed.

The procedure:

1. Pick a sampling rate, for example 10%.
2. For a fixed window, record both the sampled attribution and the true total from the provider's own usage report for the same window.
3. Compute the relative error per feature and per tenant: `(sampled_estimate − true_total) / true_total`.
4. Repeat across at least one full traffic cycle, including a peak.

Two properties matter. First, error shrinks as the square root of the sample count, so low-traffic features and small tenants have the largest relative error. Second, if you sample at the request level but a small number of requests dominate the cost (long documents, large outputs), the estimate will be noisier than the request count suggests. Sampling by cost rather than by request, where the platform allows it, reduces this.

Do not sample the identity of the request. Sample whether you record the full record; keep the total counter exact by incrementing a cheap counter for every call. That way the denominator is always correct and only the breakdown is estimated.

## Comparing architectural options

The table below compares categories of approach, not specific vendors. Pricing and feature details change; the structural trade-offs do not.

| Approach | Granularity | Freshness | Operational cost | Main risk |
|---|---|---|---|---|
| Provider-side usage export | Per API key, per day | Hours to a day | None beyond the provider bill | Cannot separate features sharing one key |
| Cloud billing tags | Per resource, per day | Roughly a day | None | Tag propagation gaps; no per-request detail |
| Metrics pipeline (counters plus a time-series store) | Per feature, model, region | Seconds to minutes | Storage and cardinality | Cardinality explosion |
| Distributed tracing plus an analytical store | Per request | Seconds to minutes | Storage, plus pipeline maintenance | Schema churn; backfill pain |
| Managed APM or observability platform | Per request | Seconds | Per-host or per-ingest fees | Ingest cost can exceed the AI spend it tracks |

A workable pattern for most teams is a hybrid: exact counters on bounded dimensions for dashboards, plus sampled traces for investigation. The counters answer "how much and where." The traces answer "why this request."

### Provider-side exports as a baseline

Every major model provider exposes usage data somewhere: a usage API, a billing export, or a dashboard. Use it as the ground truth for reconciliation, not as the attribution system. It typically aggregates per API key, which means all features sharing a key collapse into one number. The value of provider exports is that they tell you whether your instrumentation is wrong. If your internal total and the provider total diverge by more than a few percent over a week, you have a bug, and finding it is worth more than any dashboard.

### Cloud billing tags as a coarse fallback

Resource tags let you slice a cloud bill without writing instrumentation. The limits are structural: tags attach to resources or invocations, not to individual requests, and billing data refreshes on a delay measured in hours to a day. That makes tags suitable for a monthly reconciliation and unsuitable for debugging a spike that started twenty minutes ago. A frequent failure is a missing tag on one code path, which quietly moves its cost into an untagged bucket rather than raising an error.

### Metrics pipelines

A counters-based pipeline is the cheapest way to get fresh, low-cardinality attribution. The design constraint is the cardinality budget described earlier. Watch for:

- Unbounded label values (raw user IDs, URLs, prompt text).
- High-churn labels where values change every deploy.
- Buffered agents that drop data under load. Dropped metrics look like reduced usage, not like an error, which is the worst possible failure mode for a billing-adjacent system.

Instrument the pipeline itself: emit a counter for records received and a counter for records dropped, and alert when the ratio moves.

### Tracing plus an analytical store

Tracing gives per-request detail and is the right tool for investigation. The costs are pipeline maintenance and schema evolution. A schema that changes three times in a quarter makes historical backfill expensive, because old records no longer fit the new shape. Mitigate by versioning the schema explicitly and keeping the raw payload alongside the parsed columns for a retention window.

### Managed observability platforms

Managed platforms trade money for time. The relevant question is not the sticker price but the ratio of ingest cost to the AI spend being tracked. If the attribution layer costs more than a few percent of the spend it measures, it is not paying for itself. Model the ingest volume before committing: calls per day times average record size times the per-gigabyte rate, then compare that to the monthly model bill.

## Worked example: from tokens to a per-feature number

The following is illustrative arithmetic with stated assumptions, not a benchmark. Suppose a service handles 1,000,000 AI calls per day. The traffic mix is 60% chat completions, 25% vector searches, and 15% embedding calls. Assume the chat calls average 1,200 input tokens and 300 output tokens. Assume the embedding calls average 800 input tokens and no output tokens. Assume the provider prices input tokens at $2.50 per million and output tokens at $10.00 per million, and that embedding input tokens are priced at $0.13 per million.

Chat calls per day: 1,000,000 × 0.60 = 600,000.
Chat input tokens per day: 600,000 × 1,200 = 720,000,000.
Chat output tokens per day: 600,000 × 300 = 180,000,000.
Chat input cost per day: 720,000,000 ÷ 1,000,000 × $2.50 = $1,800.
Chat output cost per day: 180,000,000 ÷ 1,000,000 × $10.00 = $1,800.
Chat total per day: $3,600.

Embedding calls per day: 1,000,000 × 0.15 = 150,000.
Embedding input tokens per day: 150,000 × 800 = 120,000,000.
Embedding cost per day: 120,000,000 ÷ 1,000,000 × $0.13 = $15.60.

Vector search cost depends on the index host, not on tokens, so it is attributed by a different mechanism: allocate the index's hourly cost across the queries it served in that hour.

The point of the exercise is not the totals. It is that the totals are only defensible if the token counts come from provider responses and the prices come from a single, versioned configuration. Hard-coding prices in application code guarantees that a price change silently corrupts historical comparisons.

### Computing cost in code

```python
# Prices are per million tokens. Keep this table in configuration, not in code,
# and record the table version alongside each cost so historical values stay reproducible.
MODEL_PRICING = {
    "chat-large": {"input": 2.50, "output": 10.00},
    "embedding-small": {"input": 0.13, "output": 0.0},
}

def calculate_cost(tokens_input: int, tokens_output: int, model: str) -> float:
    price = MODEL_PRICING.get(model)
    if price is None:
        # Fail loudly. A silent zero here is a billing bug that hides itself.
        raise KeyError(f"no pricing configured for model {model!r}")
    cost = (tokens_input / 1_000_000) * price["input"]
    cost += (tokens_output / 1_000_000) * price["output"]
    return cost
```

Two details matter more than the arithmetic. First, an unknown model must raise, not return zero, because a zero-cost bucket is invisible on a dashboard. Second, the price table needs a version identifier stored with each computed cost, so that when prices change you can still explain last quarter's numbers.

## Failure modes and how to detect them

**Identity loss.** A growing share of usage lands in an `unknown` bucket. Detect by alerting on the ratio of unattributed cost to total cost. Investigate by sampling the requests in that bucket and inspecting their headers.

**Cardinality explosion.** Query latency and storage grow faster than traffic. Detect by tracking the number of distinct series in your store and alerting on its growth rate, not its absolute value. The usual cause is a label whose values are unbounded.

**Silent metric drops.** Attribution totals fall while provider usage rises. Detect by reconciling the internal total against the provider's usage export on a daily schedule and alerting on divergence beyond a threshold you choose.

**Double counting.** Retries are recorded as separate billable calls when the provider deduplicated them, or the reverse. Detect by comparing retry counts in your telemetry against the provider's request counts.

**Price drift.** Costs look wrong after a provider price change. Detect by versioning the price table and reviewing it on a schedule rather than on incident.

**Clock skew across regions.** Time-bucketed totals do not reconcile across regions. Detect by comparing per-region sums against the global sum for the same window.

## Choosing based on your situation

| Situation | Reasonable starting point | Why |
|---|---|---|
| Already running a metrics stack | Add bounded-dimension counters to the existing pipeline | Lowest marginal cost; no new vendor |
| No observability stack yet | Managed platform with per-request tracing | Fastest path to a first report; revisit if ingest cost grows |
| Cloud-only, small spend | Provider usage export plus cloud billing tags | No new code; accept daily granularity |
| Multiple features sharing one API key | Add per-feature counters immediately | Provider exports cannot separate them |
| Strict data residency requirements | Self-hosted metrics plus analytical store | Keeps request records inside the boundary |
| Very high volume | Exact counters plus sampled traces | Keeps the denominator exact, the breakdown estimated |

The exception to all of this is a spend small enough that a monthly invoice is self-explanatory. Below a few hundred dollars a month, a tagged provider export is usually sufficient, and building a pipeline is premature.

## Frequently asked questions

**How do I add instrumentation without slowing down the API?**

Use asynchronous, batched export rather than synchronous per-request calls to a backend. In Node.js, the OpenTelemetry tracing SDK exports spans through a batch span processor, which queues spans and flushes them on an interval, so the request path only pays the cost of creating the span. In Python, the equivalent is a batch span processor rather than a simple one, which exports on the calling thread. Measure the actual overhead rather than trusting a figure: run a load test with instrumentation disabled, then enabled, and compare the 95th and 99th percentiles of latency. Overhead that appears in the median but not the tail usually means a synchronous export somewhere in the path.

**What is the easiest way to get per-user attribution without changing the AI code?**

Set the identity at the edge and read it from the request context at the inference call. In AWS, an API Gateway mapping template can inject a header into the Lambda event so the function does not rely on the client. If the AI call happens inside a library you do not control, wrap the call site and attach the identity to the surrounding span. The failure to watch for is cancellation: record identity before the call starts, so a disconnected client still leaves an attributable record.

**How do I reconcile my numbers with the provider's invoice?**

Export the provider's usage data on a schedule and compare it to your internal totals for the same window, grouped by whatever dimension both sides share, usually API key or day. Expect small differences from timing at window boundaries and from retries. Investigate anything larger than a threshold you set deliberately. The reconciliation job is the single most valuable piece of the system, because it is the only thing that detects silent under-counting.

**When should I sample?**

When the volume makes full-fidelity recording expensive relative to the spend being tracked, and when per-feature accuracy tolerances are loose enough to accept sampling error. Keep exact counters for totals, sample only the detailed records, and measure the error against the provider's ground truth before trusting the estimates.

## One thing to do in the next 30 minutes

Pick one AI feature in production and add three counters to it: total calls, total input tokens, and total output tokens, each tagged only with the feature name and the model name. Do not add user IDs yet. Deploy it, and let it run for a day. Tomorrow, compare the token totals against the provider's usage export for the same period. If they match within a few percent, your token accounting is sound and you can safely add dimensions. If they do not, you have found the bug that would have made every downstream dashboard wrong, and you found it before building anything on top of it.
