# AI features as tech debt or golden paths

## The conventional playbook and where it stops being true

The standard advice for shipping an AI feature is well known: put the model call behind a thin wrapper service, cache aggressively, gate rollout with a feature flag, and ship. Each piece of that advice is individually reasonable. Together they encode assumptions that quietly stop holding once traffic, networks, or model behavior change.

The assumptions are worth naming explicitly, because each one is a future incident:

- **Assumption 1: errors are transient.** The wrapper retries on failure. But an overloaded upstream returning 502s for minutes is not transient, and retrying makes the overload worse.
- **Assumption 2: the cache will be warm.** A cold cache means every request is a model call, which is exactly when the model is most likely to be rate-limited.
- **Assumption 3: the model's output contract is stable.** Providers change response envelopes, add fields, or wrap payloads. A pass-through wrapper forwards the change straight to clients.
- **Assumption 4: the feature flag is a kill switch.** A flag controls whether traffic is routed, not whether the downstream dependency is healthy. Flipping it off does not drain in-flight requests or protect the model.

None of these assumptions are about AI specifically. They are about distributed systems. The reason AI features surface them so sharply is that the model call is slow, expensive, non-deterministic, and owned by someone else.

## A worked failure: cache stampede on a cold key

Consider a summarization endpoint. The wrapper checks Redis for a key derived from the prompt. On a miss, it calls the model and writes the result back. Now trace what happens when a popular prompt expires while traffic is high.

Assume the following, all illustrative round numbers chosen for arithmetic clarity:

- 200 requests per second arrive for the same prompt.
- The cache entry expires at t=0.
- Each model call takes 2 seconds.
- The wrapper has no in-flight deduplication.

Every request in that 2-second window sees a miss. That is 200 requests/second × 2 seconds = 400 concurrent model calls for one logical answer. If the provider's per-key rate limit is, say, 20 requests per second, roughly 380 of those calls are rejected, and the wrapper retries them, which adds more load. The cache never gets a chance to fill because the write happens after the call returns, and the calls are failing.

This is a classic thundering herd. It has nothing to do with AI, but AI makes it worse because the model call is slow (widening the window) and metered (so the stampede costs money, not just latency).

The fix is not a bigger cache. It is single-flight: the first request for a key acquires a short lock and performs the model call; concurrent requests for the same key wait on that result or fall back immediately. In Redis this is commonly done with a `SET key value NX PX <ttl>` lock plus a publish/subscribe or polling wait, or with a Lua script that makes the check-and-set atomic.

### How to measure whether you have this problem

Instrument three counters per cache key namespace:

1. `cache_lookup_total{result="hit|miss"}` — a plain hit/miss counter.
2. `model_calls_total` — incremented once per outbound model request, including retries.
3. `inflight_deduped_total` — incremented when a request joined an existing in-flight call instead of starting a new one.

Then compare `model_calls_total` against `cache_lookup_total{result="miss"}` over a one-minute window. If model calls substantially exceed misses, you are stampeding or retrying, and the two are hard to tell apart without the third counter. A load test that replays the same prompt at high concurrency against a cold cache will reproduce this in seconds; a production dashboard that only tracks average latency will not show it, because the failures are absorbed by retries until they are not.

## Why "swap the model later" is harder than it sounds

Wrapping the model behind an interface is good practice, but the interface is rarely the hard part. The hard part is the response contract.

A wrapper that forwards the provider's JSON to clients has made the provider's schema part of your public API. When the provider changes the envelope — for example, moving from a top-level `summary` field to a nested `result.summary` — every client breaks at once. The wrapper did its job (it routed the call) and still caused an outage, because it never validated the shape of what came back.

The defensive pattern is to treat the model response as untrusted input:

- Parse it against an explicit schema at the wrapper boundary.
- On schema failure, do not forward. Log the raw payload, return a typed error, and serve a fallback.
- Version your own response contract independently of the model version, so clients depend on your schema, not the provider's.

This also makes model swaps cheap in the way the original advice intended. If your wrapper validates against your schema, a provider change becomes a mapping change in one place, and a contract violation becomes a logged error rather than a client-side crash.

A concrete check: for every field your clients read, there should be a test that feeds the wrapper a malformed or reshaped payload and asserts that the wrapper returns your error type rather than the raw body. If that test does not exist, you do not yet have a model-agnostic wrapper — you have a proxy.

## Feature flags are routing, not resilience

A feature flag is a boolean (or a percentage) that decides whether a code path runs. It is not a circuit breaker, and it is not a health check.

The failure mode is easy to describe and common: the flag is on, the model is degraded, and turning the flag off only stops new requests from entering the path. Requests already in flight still complete (or time out), and any code that calls the model outside the flag's scope keeps calling it. Meanwhile, the flag's own evaluation depends on a remote service or a synced config file, which is itself a dependency that can lag or fail.

What actually protects the feature is a circuit breaker around the model call:

- Track failures (timeouts, 5xx, schema violations) in a rolling window.
- When the failure rate crosses a threshold within that window, open the circuit and fail fast for a cooldown period.
- After the cooldown, allow a limited number of trial requests (half-open). If they succeed, close the circuit; if not, reopen it.

Feature flags and circuit breakers are complementary. The flag decides who gets the feature; the breaker decides whether the feature can currently serve anyone. Conflating them is what produces the "we turned it off but it was still down" incident.

## A state-machine framing

A more honest model for an AI feature is a state machine whose transitions can fail independently:

`idle → fetching → validating → serving (cache or model) → returning`

Each transition needs its own timeout, its own error type, and its own fallback:

- **fetching → validating:** timeout budget, retry policy with jitter, and a hard cap on total attempts.
- **validating:** schema check; on failure, log and route to fallback rather than returning raw output.
- **serving:** single-flight cache fill; serve stale-but-valid entries if the model is unavailable.
- **returning:** always return something well-formed — a cached result, a precomputed result, or an explicit "unavailable" response the UI can render.

The value of writing this down is that it forces you to decide what happens at each edge. Most wrapper implementations only define the happy path and let exceptions propagate, which is why the failure modes above all look like "the feature is down" from the outside.

## Choosing between a thin wrapper and a guarded pipeline

The wrapper approach is not wrong. It is the right starting point when its assumptions hold. The table below is a decision aid, not a benchmark; fill it in for your own context and count the left column.

| Factor | Guarded pipeline | Thin wrapper |
|---|---|---|
| Client network | Mobile, intermittent | Stable, wired or office Wi-Fi |
| Traffic shape | Spiky, bursty | Steady |
| Model/contract churn | Frequent | Rare |
| Blast radius of a bad response | Consumer-facing | Internal tool |
| Team capacity for on-call | Has rotation | No rotation |
| Cost sensitivity | Metered, high volume | Low volume |

A rough rule: if three or more rows land in the left column, the wrapper's assumptions are already broken for you, and the guarded pipeline is the honest starting point. If most rows land on the right, ship the wrapper — but add the schema validation and the circuit breaker, because those are cheap and they are the two pieces that fail most expensively.

Note what is *not* in the table: language or framework. The state machine can be written in any language with async support. The choice of Rust, Go, or TypeScript matters far less than whether single-flight, schema validation, and the breaker exist at all.

## Objections worth taking seriously

**"This is over-engineering for an early product."**

Partly true. But two of the pieces are nearly free: schema validation at the boundary, and a circuit breaker that returns a fallback. Single-flight is a few lines around your cache write. What is genuinely expensive is the full state machine with per-transition observability, and that can wait until you have traffic that justifies it. The mistake is shipping with *none* of them and discovering the need during an incident.

**"Feature flags are required for safe rollouts."**

They are useful, but a flag is not a rollback mechanism for a degraded dependency. If your deploy pipeline can ship a previous artifact quickly, and your routing can be changed without a remote service, you have rollback without the flag's latency and its own failure modes. Decide based on how fast you can actually revert, not on the flag's existence.

**"Caching is always a win."**

Caching trades freshness and memory for latency and cost. It is a win when the cached value is stable and the key space is bounded. It is a liability when keys are high-cardinality (slightly different phrasings of the same question), when values go stale quickly, and when a cold cache triggers the stampede described above. Cache the things you can key reliably; for everything else, consider serving a precomputed fallback instead of caching arbitrary model output.

**"Rust is too hard for this."**

The language is not the point, and the state machine does not need to be large. What matters is that transitions are explicit and logged. A smaller implementation in a language your team already operates is usually better than a larger one in a language they do not.

## A checklist before you ship

- [ ] Every model response is parsed against an explicit schema; malformed output returns a typed error, never the raw payload.
- [ ] Cache writes use single-flight so concurrent misses produce one model call, not N.
- [ ] TTLs are deliberate per key class, and stale-but-valid entries can be served when the model is unhealthy.
- [ ] A circuit breaker wraps the model call, with a defined failure threshold, cooldown, and half-open trial.
- [ ] There is a fallback path that returns a well-formed response when the model is unavailable.
- [ ] Metrics exist for cache hit/miss, model calls (including retries), deduplicated in-flight calls, and breaker state.
- [ ] A test feeds the wrapper a reshaped payload and asserts your error type is returned.

## FAQ

**How do I handle model drift without forcing clients to update?**

Version your own response contract, not the model. Clients depend on your schema and version tag; when the upstream shape changes, you update the mapping in one place and bump your version. Clients that pin a version keep working until they choose to move.

**What should I use instead of a remote feature-flag service?**

Anything that ships with your deployment artifact: a config file, an environment variable, or a build-time constant. The tradeoff is that changing it requires a deploy. That is acceptable if your deploys are fast and reversible; it is not acceptable if they are not.

**Why do exact-match caches perform poorly for conversational prompts?**

Because users express the same intent many ways. Slight variations in phrasing, spelling, or dialect produce different keys and therefore different cache entries, so the hit rate stays low while memory usage grows. Either normalize keys aggressively, or accept the low hit rate and rely on fallbacks rather than caching.

**What numbers should I watch to know if the feature is production-ready?**

Track cache hit rate, p99 end-to-end latency, model call failure rate, and fallback rate. Set thresholds based on your own latency budget and error tolerance rather than copying figures from an article — the right values depend on your users and your SLA. The point is to have the four series on a dashboard before launch, not after.

## Do this in the next 30 minutes

Open your AI wrapper's request path and find the cache write. Check whether two concurrent requests for the same missing key can both trigger a model call. If they can, add single-flight: a short-lived lock (`SET ... NX PX`) around the fetch-and-write, with concurrent requests either waiting briefly or returning your fallback response. Then add one counter for deduplicated calls and confirm, under a concurrent replay of the same prompt against a cold cache, that the model is called once.
