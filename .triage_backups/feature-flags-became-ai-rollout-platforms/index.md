# Feature flags became AI rollout platforms

A default flag configuration is fine right up until it isn't. Flag problems tend to surface mid-migration, when there is no time to solve them properly. This article covers the root cause rather than the symptom: AI rollout is not a bigger canary deploy, and flag systems built for booleans strain under it.

## Why this comparison exists

Feature flags began as a way to toggle behavior without a deploy. They are now used as the control plane for model rollouts, prompt changes, retrieval index swaps, and A/B tests on generated output. The category moved quickly enough that vendor marketing pages often still describe older problems while the SDKs ship newer primitives: percentage rollouts scoped to user attributes, model-variant targeting, guardrail hooks, and evaluation pipelines that gate a release on a metric instead of a human.

The practical problem is that teams commonly choose tools on the wrong axis. They compare "does it have flags" when every serious product does. The real question is whether the flag evaluation path can carry AI-specific payloads: model ID, prompt version, temperature, retrieval config, and a fallback policy for when the LLM provider returns a 429. A flag system that only knows boolean and string variations forces that payload into a JSON string, and once it is a string you lose per-field targeting, per-field audit, and per-field rollback.

What trips people up is that AI rollout is not a scaled-up canary deploy. It is a different failure mode: the artifact being shipped changes at runtime, the cost per request is variable, and the same input can produce different output. That difference is what the rest of this article is about.

## Evaluation criteria

Six dimensions matter, weighted toward what actually breaks in production.

1. **Evaluation model.** Can a flag return a structured config rather than a scalar? The target shape is something like `{model: "…", temperature: 0.2, prompt_version: 17}` as a first-class variation, with targeting on user tier and region. If structured values are not supported, everything downstream — audit, per-field rollback, validation — has to be rebuilt outside the flag system.

2. **Guardrail integration.** Does the SDK expose a pre-request hook for a classifier or regex check, and a post-request hook for output validation? Without them, safety logic lives outside the flag system and the audit trail is split across two places.

3. **Evaluation and metrics.** Can the platform consume an eval score — offline or online — and automatically roll back a variant that regresses below a threshold over a rolling window? This is the capability that separates a rollout platform from a flag tool.

4. **Latency.** Local evaluation matters. An SDK that performs a network round trip per evaluation adds tens of milliseconds per call. In a deployment where the nearest edge point of presence is far from your users, that cost is paid on every request. In-process evaluation against a cached ruleset is the property to look for.

5. **Pricing model.** Per-seat pricing breaks when evaluation volume scales with traffic rather than headcount. Per-request pricing breaks when a single page render evaluates dozens of flags. Look for a model that survives both patterns, or be explicit about which one you will hit.

6. **Operational cost in constrained environments.** Offline mode, small binary size, and no hard dependency on a control plane in a single region. If users are on slow connections and deploy windows are tight, a flag system that requires a persistent socket to the control plane is a liability.

A common trap is evaluating tools on the demo dashboard. The dashboard is always fine. What pages someone at 2am is the SDK's behavior when the flag service is unreachable — whether it serves the last known good ruleset or throws and takes down the request path. Weight that heavily.

## How to measure these properties yourself

None of the criteria above require trusting a vendor page. Each can be tested in an afternoon.

**Fallback behavior.** Point the SDK at an unreachable host (a blackholed IP or a stopped local container), then evaluate a flag. Record whether the call returns the last cached value, returns the supplied default, or raises. Repeat after clearing the local cache to see the cold-start path. This is the single highest-value test.

**Evaluation latency.** Instrument the evaluation call itself, not the whole request. Record a histogram of evaluation durations over a few thousand calls, then compare in-process evaluation against a forced remote evaluation. Compare p50 and p99, not the mean.

**Structured variation support.** Define a variation with a nested object containing at least three fields, then attempt to target on one of those fields in the platform's UI or API. If targeting is only possible on the whole blob, the platform does not support structured targeting regardless of what the SDK types say.

**Metric gating.** Configure a synthetic metric that you can drive below a threshold on demand, attach it to a variation, and observe whether the rollout pauses without human action. Measure the lag between the metric crossing the threshold and the rollout stopping — that lag is the real safety window.

**Cost at your traffic.** Take your actual daily evaluation count, multiply by the vendor's per-evaluation rate, and compare against the per-seat tier you would need. Do the same arithmetic for the self-hosted option: instance cost plus the engineering hours to run it.

## The platforms, ranked by fit for AI rollout

### 1. LaunchDarkly (with AI Configs)

**What it does:** A feature flag platform that also offers AI Configs — structured variations carrying model, prompt, and parameter sets, with targeting and a metrics pipeline that can gate a rollout on an evaluation metric.

**Concrete strength:** The evaluation pipeline is the most complete among the options discussed here. A metric can be attached to a variation, and an automatic rollback can be configured when that metric drops more than a set threshold over a rolling window. That closes the loop between shipping a prompt and detecting that the prompt is worse.

**Concrete weakness:** Cost and complexity at the low end. Per-seat pricing plus per-request evaluation costs are hard to justify for a very small team. The SDK footprint is also heavier than the pure open-source options, typically involving a background streaming connection unless a polling fallback is configured.

**Best for:** Teams with a dedicated platform engineer and a real experimentation budget that need audit trails and automated rollback.

### 2. Flagsmith (self-hosted)

**What it does:** Open-source feature flag platform that can run on your own infrastructure, with edge-side evaluation and an API that supports structured flag values.

**Concrete strength:** Self-hosting is a genuine option rather than a checkbox. The API and frontend can run on modest hardware with all evaluation data in your own Postgres database. For teams with data-residency requirements, that matters more than any dashboard feature.

**Concrete weakness:** AI-specific tooling is thinner. There is no built-in eval-metric gating; that has to be built by reading flag state and writing back to a kill-switch flag. Fine with an engineer available, painful without one.

**Best for:** Teams that must keep data in-country or on-premises and are willing to write a small amount of glue code.

### 3. Unleash (self-hosted or cloud)

**What it does:** Open-source flag platform with a gradual-rollout model, strategy constraints, and SDKs for most languages.

**Concrete strength:** The strategy model is expressive without being a full programming language. Constraints like region, plan, and a percentage rollout compose cleanly, which maps well onto AI rollout by region and tier.

**Concrete weakness:** Structured variations exist, but the surrounding tooling assumes scalar values. Encoding a model config typically means storing JSON in a string variation and parsing it client-side, which loses per-field targeting and audit.

**Best for:** Teams already running Unleash for classic flags who want to extend it to model rollouts without adding a second vendor.

### 4. Statsig (experimentation-first)

**What it does:** A platform built around experiments and metrics first, with flags as a delivery mechanism, plus statistical tooling and a warehouse-native model.

**Concrete strength:** The experimentation statistics are the strongest of the group. Sequential testing, variance reduction techniques such as CUPED, and multiple-comparison handling are built in, which matters when many prompt variants run at once.

**Concrete weakness:** The flag evaluation path is more network-dependent than the open-source options, and free-tier evaluation limits are easy to hit if flags are evaluated on every request. For high-traffic applications the per-evaluation cost compounds.

**Best for:** Teams whose primary question is "which variant is actually better" rather than "how do I ship this safely."

### 5. PostHog (flags plus product analytics)

**What it does:** Product analytics with a feature flag system, session replay, and LLM observability features.

**Concrete strength:** The flag system is coupled to analytics, so targeting can use any event property already tracked. That is useful for AI rollout — target users who hit a specific error, or who belong to a specific cohort, without exporting data.

**Concrete weakness:** It is a broad product, and the flag SDK is not the fastest. Local evaluation exists, but ruleset sync is heavier than a dedicated flag tool's. If flags are on the critical path, you pay for a lot of analytics you may not use.

**Best for:** Teams already using PostHog for analytics who want flags without adding a vendor.

### 6. OpenFeature (the standard, not a vendor)

**What it does:** A CNCF specification and set of SDKs that abstract flag providers behind a common API.

**Concrete strength:** It decouples application code from the vendor. A call such as `client.getObjectValue("ai-config", defaultConfig)` stays the same when the provider package changes. For AI rollout that means moving from a self-hosted provider to a managed one without touching application code.

**Concrete weakness:** It is a spec, not a platform. A provider is still required, and AI-specific primitives such as eval gating and guardrail hooks are not in the spec — they are provider-specific extensions. Adopting OpenFeature does not solve the hard problem; it makes the easy part portable.

**Best for:** Teams that expect to change providers and want to avoid a rewrite.

### 7. GrowthBook (self-hosted, warehouse-native)

**What it does:** Open-source experimentation platform that reads metrics from your data warehouse and can be self-hosted.

**Concrete strength:** The warehouse-native model means experiment metrics live where the data already is. For AI evaluation, a quality score can be computed in the warehouse and read directly, avoiding a data export.

**Concrete weakness:** The flag evaluation path is designed around experiments rather than high-frequency runtime config. If a flag is evaluated on every LLM call, the overhead is noticeable compared with a dedicated flag SDK.

**Best for:** Data teams with a warehouse already in place who want experimentation without a new data pipeline.

## The top pick, and the reasoning

LaunchDarkly with AI Configs ranks first, but not for the reason the marketing suggests. It ranks first because automatic rollback on an evaluation metric is the only feature in this list that closes the loop between "a new prompt shipped" and "the new prompt is worse." Every other option requires a human to notice and act.

The scenario is worth spelling out. A team ships prompt version 18 to 100% of traffic. The new prompt is more concise, which users like, but it omits a required citation in a small share of responses. Without metric gating, that ships and stays shipped until someone files a bug. With a metric gate on a groundedness score, the rollout pauses automatically when the score drops below the threshold over a rolling window. The size of the affected share does not change the mechanism — what changes is whether detection takes minutes or weeks.

That is the difference between a flag system and an AI rollout platform. A flag system lets you turn it off. A rollout platform turns it off for you.

The cost is real. For a small team with a large monthly evaluation count, the bill is meaningfully higher than a self-hosted instance on a modest VM. But the failure mode being insured against — a silent quality regression on a model rollout — erodes user trust, and trust is harder to restore than a VM is to provision.

## Honorable mentions

**OpenFeature with a managed provider.** If the evaluation engine is desirable but vendor lock-in is not, an OpenFeature provider gives a portable API over it. The tradeoff is that structured AI configs are exposed through provider-specific extensions, so portability is partial.

**Unleash's gradual rollout for model migration.** Migrating from one model to another with a clean percentage rollout and a kill switch is something Unleash does well and cheaply. It is not an experimentation platform, but it is a solid rollout mechanism.

**PostHog's LLM observability.** Seeing the prompt, the response, and the flag state in one trace is genuinely useful for debugging. It is not a replacement for a flag platform, but it pairs well with one.

**Rolling your own config service.** A small Go or Node service serving a JSON ruleset, cached in Redis with a short TTL and evaluated in-process, is a legitimate design. The trap is that targeting, audit, and a UI eventually become requirements, and the result is a worse version of a flag platform. Do this only if flag needs are genuinely simple and stable.

## Failure modes that justify dropping an option

These are patterns worth testing for explicitly, because each one has ended an evaluation.

**A pure analytics platform with flags bolted on.** The flag SDK was an afterthought, and the evaluation path made a network call that added tens of milliseconds per evaluation. For a single LLM call that already takes hundreds of milliseconds, that is survivable; for a page that evaluates a dozen flags, it is not.

**A per-seat flag tool with no structured variations.** The pricing looked attractive until model config was being encoded as a JSON string and parsed in five places. Every new field meant a change in five files. That cost shows up as maintenance, not as an invoice.

**A self-hosted platform with a heavy control plane.** An API and worker needing multiple gigabytes of RAM to run comfortably, more than the application it serves, is backwards for a constrained deployment.

**A vendor whose SDK throws when the control plane is unreachable.** This is the failure mode weighted most heavily above. If the flag service is down and the SDK throws, the request path goes down with it. The correct behavior is to serve the last known good ruleset. Documentation mentioning a fallback is not enough — the SDK default is what matters, and it should be verified by test.

## Choosing based on your situation

| Situation | Recommended option | Why |
|---|---|---|
| Small team, managed cloud acceptable, eval gating required | LaunchDarkly AI Configs | Automatic rollback on metric regression is the differentiator |
| Data residency required, self-hosted | Flagsmith or Unleash | Runs on your own infrastructure, database-backed |
| Already using PostHog for analytics | PostHog flags | No new vendor, targeting on existing event properties |
| Experimentation is the primary goal | Statsig | Strongest statistical tooling for many-variant tests |
| Warehouse-native, data team in place | GrowthBook | Metrics computed where the data already lives |
| Provider portability required | OpenFeature plus a provider | Swap providers without rewriting application code |
| Very simple, stable flag needs | Self-hosted config service plus a cache | Cheapest, but you own the maintenance |

The deciding question is not "which has the most features." It is "what happens when the flag service is unreachable, and what happens when a model rollout regresses quality." Answer those two and the choice usually makes itself.

## A worked evaluation path

The following illustrates the shape of the integration using OpenFeature with an in-memory provider. The provider is a stand-in; the structure — a structured variation, a safe fallback, and a targeting context — is the part that generalizes.

```python
from openfeature import api
from openfeature.provider.in_memory_provider import InMemoryProvider

# Provider configured with a structured AI config variation
api.set_provider(InMemoryProvider({
    "ai-config": {
        "variations": {
            "default": {
                "model": "claude-sonnet-4",
                "temperature": 0.2,
                "prompt_version": 17,
                "max_tokens": 1024,
            },
            "canary": {
                "model": "claude-sonnet-4",
                "temperature": 0.0,
                "prompt_version": 18,
                "max_tokens": 1024,
            },
        },
        "defaultVariant": "default",
    }
}))

client = api.get_client()
config = client.get_object_value(
    "ai-config",
    {"model": "claude-sonnet-4", "temperature": 0.2, "prompt_version": 17},
    evaluation_context=api.EvaluationContext(
        targeting_key=user_id,
        attributes={"region": "NG", "plan": "trial"},
    ),
)
```

The important detail is that the fallback value is a safe default, not an exception. If the provider is unreachable, the SDK returns the fallback and the request proceeds. That is the behavior to want in a constrained deployment, and it is worth verifying rather than assuming.

On the guardrail side, the pattern is a pre-request check and a post-request validation. The exact hook names differ by SDK; the two-phase shape does not.

```javascript
// Node 20 LTS, using a flag SDK with a hook API
client.on('before-evaluation', async (context) => {
  const config = context.flagValue;
  if (config.prompt_version < 10) {
    throw new Error('Prompt version below minimum supported: ' + config.prompt_version);
  }
  return context;
});

client.on('after-evaluation', async (context) => {
  const score = await runGroundednessCheck(context.response);
  if (score < 0.7) {
    await metrics.record('groundedness_below_threshold', {
      variant: context.variant,
      score,
    });
  }
  return context;
});
```

The flag platform owns the rollout decision; application code owns the quality signal. The platform's job is to act on that signal without a human in the loop.

Note the asymmetry between the two hooks. The pre-request hook can fail closed, because a prompt version below the supported minimum is a configuration error. The post-request hook should fail open and record, because a groundedness checker that itself errors must not take down the response path. Getting that asymmetry backwards turns a safety mechanism into an availability risk.

## Frequently asked questions

**What is the difference between a feature flag and an AI rollout platform?**

A feature flag returns a value that changes application behavior, usually a boolean or a string. An AI rollout platform returns a structured configuration — model, prompt version, parameters — and can act on evaluation metrics to pause or roll back a rollout automatically. The flag is the mechanism; the rollout platform is the control loop around it.

**How do I roll out a new LLM prompt to 10% of users?**

Define the prompt as a variation in the flag platform, set a percentage rollout targeting 10% of users, and attach a metric that measures quality. Evaluate the flag on each request and pass the resulting prompt version to the LLM call. The key detail is making the fallback a known-good prompt version rather than an exception, so an unreachable flag service does not break the request path.

**Do I need a separate tool for AI experimentation and feature flags?**

Usually not, but it depends on whether statistical rigor is required. Comparing two prompt variants and wanting a confidence interval calls for a platform with proper experimentation statistics. Shipping a config change with a kill switch is served by a flag platform. Many teams start with flags and add experimentation tooling as the number of variants grows.

**What happens if the feature flag service goes down during an AI rollout?**

It depends entirely on the SDK's fallback behavior. The correct behavior is to serve the last known good ruleset from a local cache and proceed with the fallback value. The failure mode to avoid is an SDK that throws when the control plane is unreachable, which takes down the request path. Check this explicitly before adopting a vendor — it is the single most important operational property of a flag SDK.

**How many flags should be evaluated per request?**

There is no universal number, but the cost is linear in evaluations, so the practical answer is to evaluate once per request and pass the result down. Evaluating the same flag in several layers of the call stack multiplies both latency and cost without changing the outcome, and it makes the audit trail harder to read.

## Final recommendation

Choose a platform with AI Configs if budget and a platform engineer are available, because automatic rollback on an evaluation metric is the feature that turns a flag system into a rollout platform. Choose Flagsmith or Unleash if data residency or self-hosting is a hard requirement — some glue code will be needed, but the whole path stays under your control. Choose OpenFeature if keeping the option to move matters more than any single provider's primitives.

The action to take in the next 30 minutes: open the flag SDK configuration, find the fallback value for the AI config flag, and confirm it is a known-good default rather than an exception. If it throws, change it to return the last known good config. That single change is the difference between a flag service outage being a non-event and being an incident.
