# 5 notification systems ranked: push, email, SMS

## The problem this guide addresses

A notification pipeline looks trivial on a whiteboard: accept an event, look up the user's preferences, render a template, hand the payload to a provider. It stops being trivial the moment you add a second channel. Email retries are measured in minutes and hours; SMS retries are measured in seconds; push tokens expire silently; chat channels have provider-specific template approval processes that can take days. Each channel has its own failure taxonomy, its own rate limits, and its own definition of "delivered."

The result is a familiar failure mode. A team starts with one provider SDK wired directly into application code. A second channel arrives and gets its own SDK. A third channel arrives and someone writes a router. Within a year the router is the most business-critical and least tested part of the codebase, and nobody can answer the question "did this user get their alert?" without opening four dashboards.

This guide is about the decision layer above the providers: what to build, what to buy, and what to measure before committing.

## The four channels are not interchangeable

Before comparing tools, it helps to be precise about what each channel actually guarantees. Most bad architecture decisions come from assuming one channel's semantics apply to another.

| Channel | Typical delivery semantics | Failure modes that matter | Cost driver |
|---|---|---|---|
| Push | Best-effort, no delivery receipt | Token invalidation, OS-level throttling, silent drops | Effectively free at the provider level |
| Email | Queued, asynchronous, retried by the provider | Bounce types (hard vs soft), spam classification, domain reputation | Per-message, plus dedicated IP costs at volume |
| SMS | Queued, carrier-dependent | Carrier filtering, sender reputation, per-country regulation | Per-segment, varies widely by destination |
| Chat / messaging apps | Template pre-approval, per-session rules | Template rejection, opt-in compliance, provider API churn | Per-conversation, often with a service window |

Two consequences follow immediately. First, a unified API is valuable not because it saves typing but because it normalizes four different retry and deduplication models. Second, any tool that claims to handle all four "natively" deserves scrutiny: chat channels in particular usually route through a partner integration, and that integration is where the operational surprises live.

## Evaluation gates: define them before you look at features

Feature lists are marketing artifacts. Gates are engineering artifacts. Four gates cover most of the risk:

**1. Latency budget.** Define what "latency" means before measuring it. The useful definition is time from event ingestion (when your system accepts the notification request) to the provider's first accepted response. That excludes downstream carrier delivery, which you generally cannot control or measure directly. A reasonable starting budget for transactional alerts is a median under one second and a p95 under a few seconds. Marketing or digest traffic can be far slower.

**2. Reliability under sustained load.** A 99.9% success rate means roughly 43 minutes of failure per month. Decide whether that is acceptable for the channel in question; a password reset and a weekly digest do not deserve the same target. Measure success rate as (accepted by provider) / (attempted), and track it per channel, because a single blended number hides the channel that is actually broken.

**3. Cost per delivered message.** Compute this from your own traffic mix, not from a pricing page. The formula is:

```
cost_per_delivered =
    (provider_fee_per_attempt
     + infrastructure_cost_per_attempt
     + engineering_amortized_cost_per_attempt)
    / (success_rate)
```

The success-rate divisor matters. A provider that is 2% cheaper per attempt but 3% less reliable is more expensive per delivered message, before you count the support tickets.

**4. Exit cost.** Ask what happens when you leave. If templates live in a proprietary format, if preference data lives in a vendor database, and if your application code imports a vendor SDK directly, the migration cost is real and belongs in the decision.

## How to measure each gate

The gates are only useful if you can produce numbers. The instrumentation is straightforward and worth building once, because it also becomes your production observability.

**Instrument these, per channel and per provider:**

- A histogram of end-to-end latency from ingestion to provider acceptance.
- A counter of attempts, split by outcome: accepted, rejected-permanent, rejected-retryable, timed out.
- A counter of retries, with the backoff delay as an attribute.
- A gauge of queue depth and a counter of messages dropped due to queue overflow.
- A counter of duplicate suppressions (how many sends were prevented by your idempotency key).

**Then run a load test that resembles production.** Synthetic load should replay a realistic mix: mostly push, a minority of email, a small fraction of SMS. A useful pattern is a soak test at your expected peak rate for at least 24 hours, because rate-limit and reputation problems often appear only after sustained volume. Compare p50, p95, and p99 latency, and compare success rate per channel. If a provider degrades only at p99, that is still a real problem: p99 is where your worst user experience lives.

**Watch for these specific signals:**

- Retry storms: retry count rising faster than attempt count, usually caused by retrying on non-retryable errors.
- Backoff collapse: retries firing immediately instead of with exponential delay, which turns a transient provider blip into a self-inflicted outage.
- Queue saturation: ingestion rate exceeding drain rate, which converts a latency problem into a data-loss problem once the queue is full.
- Silent channel failure: one channel's success rate dropping while the blended metric stays flat because the other channels are healthy.

## Architecture patterns that survive contact with production

### Pattern 1: one ingestion point, per-channel workers

Accept every notification through a single endpoint that validates the request, resolves user preferences, and writes to a durable queue. Separate workers consume per channel. This gives you one place to enforce idempotency and one place to record intent, while letting each channel's worker use retry semantics appropriate to that channel.

### Pattern 2: idempotency keys at the boundary

Every notification request should carry a client-generated idempotency key. The ingestion layer stores the key with a TTL and rejects duplicates. This is the single highest-value piece of notification infrastructure you can build, because duplicate sends are the failure users notice most and the one most likely to trigger opt-outs. Note that provider-side deduplication (for example, a collapse key on push) only covers the window between your send and the device; it does not protect you from sending twice in the first place.

### Pattern 3: per-channel retry policy, not a global one

A global retry policy is always wrong for at least one channel. Email soft bounces should be retried over hours; SMS retryable errors should be retried over seconds; push token errors are permanent and should never be retried. Encode the policy per channel and classify errors explicitly rather than treating every non-2xx response the same way.

### Pattern 4: quarantine bad destinations

Bounced addresses, invalid push tokens, and opted-out numbers should be moved to a suppression list automatically. Continuing to send to them damages sender reputation, which is a shared resource: a high bounce rate on one campaign can degrade deliverability for every subsequent message from the same domain or number pool.

## Build versus buy: a decision checklist

Work through these questions in order. The first "yes" usually determines the answer.

1. **Do you send fewer than roughly 100,000 messages a month across all channels?** A managed API is almost always cheaper than the engineering time to build and operate your own router. The crossover point depends on your loaded engineering cost, not on message volume alone.
2. **Do you have a hard data-residency or self-hosting requirement?** If so, an open-source, self-hostable notification layer is the realistic option, and you should budget for operating it: upgrades, schema migrations, and provider module maintenance.
3. **Are you already committed to one cloud provider's ecosystem?** A cloud-native multi-channel service reduces integration work but increases exit cost. Confirm that the channels you need are generally available, not in preview or beta, before you depend on them.
4. **Is your product primarily push-based?** A push-specialized vendor will often have better device coverage and delivery tooling than a general-purpose platform, at the cost of needing separate handling for email and SMS.
5. **Do you need workflow features (delays, digests, per-user throttling, preference UIs) more than raw throughput?** Workflow-first platforms trade some control for a much shorter path to a good user experience.
6. **Is this a script, a cron job, or an internal alert rather than a user-facing system?** A CLI-style library that fans out to many services is the right tool. Do not build a pipeline for a cron alert.

## A worked cost comparison

The following is an illustrative calculation, not a benchmark. Substitute your own numbers.

Assume a product sending 5 million notifications a month: 4.5M push, 400K email, 90K SMS, 10K chat messages.

```
Push:   4,500,000 x $0.0000  = $0.00
Email:    400,000 x $0.0004  = $160.00
SMS:       90,000 x $0.0075  = $675.00
Chat:      10,000 x $0.0100  = $100.00
Provider subtotal             = $935.00

Managed platform fee (illustrative, $0.0005/msg on all traffic)
  5,000,000 x $0.0005         = $2,500.00

Total managed                 = $3,435.00
```

Now compare a self-hosted router on two application servers plus a managed Postgres instance, at an illustrative $400/month, with the same provider costs:

```
Provider subtotal             = $935.00
Infrastructure                = $400.00
Total self-hosted             = $1,335.00
```

The self-hosted option looks cheaper by roughly $2,100/month. Before concluding anything, add the costs the table omits: engineering time to build the router (typically several weeks), ongoing maintenance, on-call for a system that now pages you at 3 a.m., and the cost of the incidents you will have. If one engineer spends four hours a month on it at a fully loaded $100/hour, that is $400/month — which closes most of the gap. The decision turns on your loaded engineering cost and how much you value not owning the pipeline, not on the raw message arithmetic.

## Failure modes to design against

**Retrying non-retryable errors.** A malformed recipient address will never succeed. Retrying it wastes quota and can trip provider abuse detection. Classify errors at the provider boundary and drop permanent failures immediately.

**Unbounded queues.** A queue that accepts everything and drops nothing will eventually drop everything when it fills. Set a maximum depth, decide in advance what happens when it is reached, and alert on depth rather than on the drop.

**Preference races.** A user opts out while a message is in flight. Without a check at send time, the message goes out anyway. Re-read suppression state in the worker, not only at ingestion.

**Template drift.** If templates live in the vendor dashboard and in your repository, they will diverge. Pick one source of truth and treat the other as a build artifact.

**Channel sprawl.** Every new channel adds a provider integration, a retry policy, and a compliance surface. Add channels deliberately, with an owner, or the pipeline becomes unmaintainable.

## FAQ

**Should I use one provider for everything or separate providers per channel?**
Separate providers usually win on quality, because email deliverability, SMS routing, and push delivery are genuinely different specialties. A unified API layer on top lets you keep separate providers without exposing that complexity to application code.

**How do I deduplicate push notifications?**
Use the platform's collapse mechanism: a collapse key on Android-style push and a collapse identifier on Apple push. These replace an older notification for the same key rather than stacking them. They do not replace server-side idempotency, which prevents the duplicate send in the first place.

**How do I protect SMS sender reputation?**
Use dedicated sender identities per use case, warm new numbers gradually rather than blasting from day one, and suppress opt-outs and hard failures immediately. Monitor your bounce and complaint rates and treat any sustained rise as an incident, because carrier filtering is applied to the sender, not the individual message.

**How should I test chat-channel templates before launch?**
Use the provider's sandbox environment for integration testing, and submit production templates early, since approval review is a manual process with a turnaround measured in days. Include your privacy policy and opt-in language in the submission to reduce back-and-forth.

**What is the cheapest way to send transactional email at volume?**
A large cloud email service with per-message pricing is usually the cheapest at low volume; dedicated IPs become worthwhile only once your volume is high enough to warm them properly. Warming is a rate-limited process measured in weeks, and sending too fast from a new IP is the most common way teams damage their own deliverability.

**Do I need OpenTelemetry, or is logging enough?**
Logging tells you what happened to individual messages. Metrics tell you whether the system is healthy right now. You need both, but if you can only add one, add histograms of latency and counters of outcomes per channel — those are what you will look at during an incident.

## Choosing in practice

There is no universally best notification system, and any ranking that claims otherwise is describing one team's constraints. The decision reduces to four questions: how many channels do you need today, how much of the pipeline do you want to own, what is your loaded engineering cost, and how expensive would it be to leave.

Teams commonly end up with a hybrid: a managed API for the channels that are hard to operate well (email deliverability, SMS routing) and a self-hosted path for push and in-app, where the provider is free and the semantics are simple. That is a defensible architecture as long as the routing logic lives in your code, not in a vendor's dashboard.

## Your next 30 minutes

Pick your highest-volume channel and add three metrics to it: a latency histogram from ingestion to provider acceptance, a counter of send outcomes split by retryable and permanent, and a counter of duplicate suppressions. Run one hour of production-shaped traffic against it and record p50, p95, and success rate. Those three numbers are the baseline you will compare every future provider and architecture decision against — and they cost you less than an afternoon.
