# Serverless Cold Starts: Cost Levers and How to Measure Them

## What a cold start actually costs you

A cold start is the extra time a serverless platform spends before your handler runs: allocating a sandbox, booting the runtime, and executing your initialization code (module imports, SDK clients, config loading, connection setup). For a small Node or Python function this is often tens of milliseconds. For a JVM function with heavy class loading it can approach a second. The exact number is workload-specific and must be measured, not assumed.

The cost angle is the part teams underrate. When a function scales from zero, the platform bills for the initialization time as well as the handler time, at the same per-millisecond rate. A function that cold-starts on a meaningful fraction of invocations pays for work that produces no user-visible response. At the same time, the bursts that trigger cold starts are usually the ones closest to your latency SLO, so the failure mode is both a bill and a p99 regression.

This article covers three levers that are commonly combined:

1. **Idle/keep-alive behavior** — how long the platform keeps an execution environment around between invocations.
2. **Provisioned concurrency** — pre-allocated environments you pay for whether or not they serve traffic.
3. **Architecture and memory configuration** — ARM64 vs x86_64 and the memory setting that determines CPU share.

None of these is free. The goal is to pick the combination that minimizes cost per successful request at your target latency, not to minimize cold-start count in isolation.

## Why the obvious metrics mislead

Most teams start by optimizing raw latency. That is the wrong target for three reasons.

**First, cold starts are bursty.** A function can be 99% warm by invocation count and still cold-start on the first request of every traffic spike. Those are exactly the requests your users notice. An average-latency dashboard hides this; a p99 or a cold-start-ratio metric does not.

**Second, provisioned concurrency is a retainer, not a guarantee.** It keeps N environments initialized and ready. You pay for those N for as long as the configuration is active, including periods when no request arrives. If your traffic is spiky — a daily batch, a campaign, a scheduled job — a flat provisioned-concurrency setting means paying peak prices during the trough.

**Third, architecture choice interacts with everything else.** ARM64 (Graviton) runtimes are generally cheaper per unit of compute than x86_64 and often faster for interpreted and JIT-compiled workloads. But native dependencies (image libraries, crypto bindings, some ML runtimes) may need ARM builds, and if your CI produces x86 artifacts by default you can silently negate the benefit.

The metric that ties these together is **cost per request at a latency target**. Instrument that first.

## A mental model: the taxi fleet

Think of execution environments as taxis, the runtime as the driver, and each request as a passenger.

- **Park the taxi** (scale to zero): cost is zero, but the next passenger waits for a car to arrive.
- **Keep the engine running in the driveway** (warm idle): a small ongoing cost, and the next passenger leaves immediately — until the platform decides to reclaim the car.
- **Pre-book a taxi for a shift** (provisioned concurrency): a high retainer per car, guaranteed availability for the shift window.

Cold starts happen when a passenger arrives and no car is parked, idling, or pre-booked. The tuning problem is deciding how many cars to pre-book and for how long, given that pre-booking is the most expensive option per unit time.

Two consequences follow. Pre-booking should cover only the windows where the cost of a cold start exceeds the retainer — typically short, predictable peaks. And the retainer should be sized to the concurrency you actually need at the peak, not to your total traffic.

## Worked example (illustrative)

The numbers below are illustrative, chosen to show the arithmetic. Substitute your own measurements.

Assume a function with these characteristics:

- Average handler duration: 200 ms
- Average initialization (cold start) duration: 700 ms
- Invocations per day: 2,400
- Cold-start ratio: 18% → 432 cold starts/day

Billed compute per day, ignoring the free tier:

- Warm invocations: 2,400 − 432 = 1,968 × 200 ms = 393,600 ms
- Cold invocations: 432 × (700 + 200) ms = 388,800 ms
- Total: 782,400 ms/day

At a hypothetical $0.0000166667 per GB-second (the published x86 rate for a 1 GB function) and 1 GB of memory, that is roughly 782.4 GB-s × $0.0000166667 ≈ $0.013/day in compute. Requests are billed separately. The point is not the absolute figure — it is that **cold starts are roughly half the billed compute here**, because 432 initializations of 700 ms cost about as much as 1,968 warm invocations of 200 ms.

Now reduce the cold-start ratio from 18% to 3% (72 cold starts/day) by whatever combination of levers works:

- Warm: 2,328 × 200 ms = 465,600 ms
- Cold: 72 × 900 ms = 64,800 ms
- Total: 530,400 ms/day

That is a 32% reduction in billed compute. Whether it is worth it depends on what you paid to get there: if provisioned concurrency costs more than the compute you saved, the change is a net loss.

### How to measure this yourself

1. **Get the cold-start ratio.** Enable Lambda Insights or parse the `InitDuration` field from your function's CloudWatch Logs. `InitDuration` is only present on cold starts, so its presence is your signal. Count invocations with and without it.
2. **Get the latency split.** Compare `Duration` on warm vs cold invocations. Use p50 and p99, not the mean.
3. **Get the bill.** Use the Cost Explorer or a tagged cost allocation. Attribute the function's compute line to the cold-start ratio to get an upper bound on what initialization costs you.
4. **Model the change before shipping it.** Recompute the arithmetic above with your measured durations and your proposed cold-start ratio. If the projected saving is smaller than the provisioned-concurrency retainer, do not make the change.

## The three levers, in order of effort

### 1. Reduce initialization work

This is the cheapest lever because it costs no ongoing money. The goal is to make the cold path shorter.

- Move SDK client construction and config parsing out of the handler into module scope where the platform can reuse it across invocations.
- Avoid connecting to databases or caches during initialization unless the connection is genuinely needed on every invocation; lazy-connect on first use instead.
- For Node, bundle to a single file to cut filesystem and module-resolution overhead. For Python, prefer layers with pre-built wheels so imports do not compile at runtime.
- For JVM functions, reduce classpath scanning and defer framework startup where the framework supports it.

Measure the effect by comparing the `InitDuration` distribution before and after. A change that does not move the p50 of `InitDuration` is not doing anything.

### 2. Tune the keep-alive window

Platforms keep an execution environment alive for a period after an invocation to serve the next one. The exact behavior and configurability vary by provider and runtime; check your platform's documentation for the current default and whether it is adjustable.

The trade-off: a longer keep-alive window means fewer cold starts but more billed idle time if the platform bills it. Shorter means the opposite. The right value depends on your inter-arrival time distribution, not on your average traffic. If 90% of your gaps between requests are under 20 seconds, a 30-second window covers most of them; if gaps are usually minutes, no reasonable window will help and you should look at provisioned concurrency or snapshot-based initialization instead.

To measure: log the timestamp of each invocation, compute the gap to the previous invocation, and plot the distribution. Set the keep-alive window at a percentile that matches your latency tolerance.

### 3. Provisioned concurrency, scoped to peak windows

Provisioned concurrency pre-initializes a fixed number of environments. You pay for them continuously while the configuration is active. The correct pattern for spiky traffic is to raise the setting shortly before a known peak and lower it after — driven by a scheduler, not by hand.

Sizing: set provisioned concurrency to the concurrency you expect at the peak, not to your average. Concurrency for a synchronous function is approximately `requests per second × average duration in seconds`. If you expect 50 requests/second at 200 ms each, you need about 10 concurrent environments.

Scheduling: use a scheduler to invoke an administrative function that calls the concurrency API. The scheduler invocation cost is negligible compared to the concurrency retainer, but confirm current pricing rather than assuming.

Snapshot-based initialization (for runtimes that support it) reduces the cold-start penalty by restoring a pre-initialized snapshot instead of running initialization from scratch. Where it is available, it changes the arithmetic: a shorter cold start means a lower cold-start ratio for the same keep-alive settings, and less need for provisioned concurrency.

## A configuration sketch

The following Terraform illustrates the shape of the setup — an ARM64 function with a versioned deployment, a provisioned-concurrency configuration, and a scheduled scale-up event. Verify every argument against the current provider documentation before applying; provider schemas change.

```hcl
resource "aws_lambda_function" "api" {
  function_name = "marketing-api"
  runtime       = "nodejs20.x"
  handler       = "index.handler"
  memory_size   = 512
  timeout       = 10
  architectures = ["arm64"]

  # Publish a version so provisioned concurrency can target it.
  publish = true
}

resource "aws_lambda_provisioned_concurrency_config" "peak_window" {
  function_name                     = aws_lambda_function.api.function_name
  provisioned_concurrent_executions = 10
  qualifier                         = aws_lambda_function.api.version
}

resource "aws_cloudwatch_event_rule" "scale_up" {
  name                = "scale-up-marketing-api"
  schedule_expression = "cron(0 8 * * ? *)"
}

resource "aws_cloudwatch_event_target" "scale_up_target" {
  rule      = aws_cloudwatch_event_rule.scale_up.name
  target_id = "scale-up"
  arn       = aws_lambda_function.api.arn
  input = jsonencode({
    action      = "update"
    concurrency = 50
  })
}
```

Two things to note. First, `aws_lambda_provisioned_concurrency_config` targets a version or alias qualifier, so the function must publish versions. Second, the EventBridge target here points at the function itself; in practice you would point it at a small administrative function that calls the concurrency API, because the payload alone does not change concurrency. The snippet shows the wiring shape, not a complete working deployment.

## Common misconceptions

**"Provisioned concurrency eliminates cold starts."** It eliminates them for the pre-initialized environments. If a burst exceeds the provisioned count, the excess requests cold-start anyway. And if the provisioned environments are recycled between peaks, the first request after a gap can still pay initialization. Size provisioned concurrency to peak concurrency and keep the configuration active across the whole peak window.

**"More memory always means faster cold starts."** Memory determines CPU share on most platforms, so more memory can shorten CPU-bound initialization. But initialization is often dominated by I/O, class loading, or single-threaded setup that does not scale with CPU share. The only reliable answer comes from running a memory sweep against your own function and recording `InitDuration` at each setting.

**"ARM64 is always better."** It is usually cheaper per unit of compute and competitive on speed, but native dependencies may lack ARM builds, and some workloads are genuinely faster on x86_64. The migration cost is real: CI pipelines, container base images, and any pre-compiled layers need attention.

**"Snapshot-based initialization works for every runtime."** Availability is runtime-specific and changes over time. Check the current support matrix for your runtime rather than assuming parity across languages.

## Decision checklist

Before changing anything, answer these:

- What is my current cold-start ratio, measured from `InitDuration` presence in logs?
- What is the p50 and p99 of `InitDuration`, and of warm `Duration`?
- What fraction of my monthly compute bill is attributable to initialization?
- What is my inter-arrival time distribution, and what percentile does my keep-alive window cover?
- What concurrency do I need at peak, and how long does the peak last?
- What does provisioned concurrency cost for that window, versus the compute I expect to save?
- Do all my dependencies have ARM64 builds, and does my CI produce them?

If the projected saving from a change is smaller than its ongoing cost, stop. If the cold path is dominated by initialization work you can move or eliminate, do that first — it is free.

## FAQ

**How do I know if cold starts are costing me money?**
Count invocations that include an `InitDuration` field in CloudWatch Logs, divide by total invocations to get the ratio, and multiply the function's compute cost by that ratio. That gives an upper bound on initialization cost, since cold invocations also run the handler.

**Can I use snapshot-based initialization with Node.js or Python?**
Support is runtime-specific. Check the current support matrix before designing around it. For non-JVM runtimes, the practical levers are usually smaller bundles, fewer dependencies loaded at startup, and ARM64.

**What is the trade-off between a long keep-alive window and provisioned concurrency?**
A keep-alive window is opportunistic — it may or may not survive until the next request, and it costs nothing extra on platforms that do not bill idle time. Provisioned concurrency is a paid guarantee. Use the former for short, frequent gaps and the latter for predictable peaks.

**Do I need to rebuild all dependencies for ARM64?**
Only the native ones. Pure JavaScript and pure Python packages are architecture-independent. Anything with compiled extensions — image processing, crypto bindings, some ML runtimes — needs an ARM build. A multi-architecture container build is the usual approach.

**How much does ARM64 actually save?**
Compute pricing for ARM64 is lower than x86_64 on the major platforms. The saving is a function of your memory and duration settings, so compute it from your own bill rather than applying a rule of thumb.

## Do this in the next 30 minutes

Open the CloudWatch Logs for one production function and run a query that counts invocations with and without the `InitDuration` field over the last 24 hours. That single number — your cold-start ratio — tells you whether any of the levers above are worth your time. If the ratio is under a few percent, close the tab. If it is high, export the `InitDuration` distribution next and start with the free lever: moving initialization work out of the handler.
