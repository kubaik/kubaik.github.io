# Merge three payment APIs into one system

## The conventional wisdom (and why it's incomplete)

The standard playbook says: build one adapter per country. A mobile money provider needs its SDK, one card processor handles Nigeria, another covers Ghana. Wrap each in a clean interface like `PaymentProcessor.sendPayment()`, add feature flags, and call it done. That is the path of least resistance, and it is what most teams do first.

The conventional view tends to ignore three costs that only show up after launch:

1. **Latency tax**: Every hop between your service and a provider adds round-trip time. Multiple providers mean multiple connection pools, multiple timeout policies, and multiple retry strategies competing for the same runtime resources.
2. **Maintenance sprawl**: When a provider changes its webhook signature scheme, every service that imports the adapter must be updated and redeployed. The change is small; the coordination is not.
3. **Regulatory drift**: When a central bank changes KYC or BVN rules, the required validation lands in whichever adapter serves that country. If the adapters share a library, the change can block unrelated providers.

The adapter pattern solves a code-organization problem. It does not solve a systems problem. Those are different problems, and conflating them is the root of most payment integration pain.

## What actually happens when you follow the standard advice

Teams commonly ship three adapters in two weeks using a mainstream Node.js LTS runtime and a general-purpose HTTP client. After launch, the first surprise is latency variance. Mobile money APIs are often fast during business hours and slower during evening peaks when consumer top-up traffic spikes. Sandbox endpoints can return 503s under concurrency with messages like `"Too many concurrent requests"`. Webhook retry policies are provider-specific and usually non-negotiable: for example, three retries with exponential backoff starting at one second.

Those behaviors collide in production. A p95 that looks fine at 1,000 TPS can degrade sharply at 2,500 TPS, not because any single provider got slower, but because your own retry and connection behavior amplifies their variance.

Then the bills arrive. Each adapter tends to open its own connection pool to its provider. Three adapters under load can mean three times the concurrent connections, three times the egress from duplicated payload logging, and three separate retry storms. Finance sees a payment-service cost increase and blames the payment service, not the architecture.

Finally, the alerts pile up. Three providers typically produce at least three classes of alert:

- Rate limiting: HTTP 429 on a payments endpoint
- Timeouts: request timeouts on a push or STK endpoint
- Signature failures: webhook signature validation failed

Each of those needs its own runbook, its own dashboards, and its own on-call context. The operational load scales with provider count, not transaction count.

A common failure mode: the p99 is acceptable with one provider, then jumps once a second and third provider are added. The bottleneck is often not CPU or memory but the separate connection pools competing for the same heap and event loop.

## A different mental model

Treat payment providers as **fallible, high-latency subsystems** that need circuit breakers, bulkheads, and caching, not just clean interfaces. Instead of one adapter per country, build one **gateway** that handles all providers under a single domain model. The gateway becomes the single source of truth for payment state, retries, and observability.

The gateway pattern gives three concrete wins:

1. **Connection pooling**: One shared pool of HTTP connections across providers reduces memory pressure and prevents each provider's traffic from starving the others. The exact p99 improvement depends on your workload; measure it rather than assume it.
2. **Priority routing**: Low-value transactions can go to the cheapest provider; high-value transactions can go to the most reliable. This is a policy decision expressed in one place.
3. **Unified observability**: A single `/health` endpoint aggregates provider health. You stop guessing which provider is slow and start seeing it in one dashboard.

The common mistake is to treat the gateway like a façade. It is not. It is a **state machine** that owns payment state, not just a router. When a webhook arrives, the gateway updates the payment record and broadcasts the event. That reduces race conditions and eliminates duplicate webhook handling.

A useful side effect: ledger writes get faster because the gateway can batch and order them, even though external provider latency is unchanged.

## The core design: one gateway, three responsibilities

A minimal gateway has three endpoints:

- `POST /payments` (create)
- `GET /payments/{id}` (read)
- `POST /webhooks/provider/{name}` (ingest)

It does not need a full ledger or complex retry logic initially. It needs a single connection pool, a circuit breaker per provider, and a normalized internal payment ID. That is a few hundred lines in Go or Rust.

The gateway has three responsibilities that adapters usually scatter:

1. **Normalize the request**: map your domain model to each provider's payload format.
2. **Normalize the response**: map provider-specific statuses into one internal state machine (`pending`, `authorized`, `captured`, `failed`, `refunded`).
3. **Normalize the identity**: map provider-specific idempotency keys and webhook event IDs into one internal ID.

If you get the third one wrong, the first two do not matter.

## Idempotency: the part that bites hardest

Provider idempotency keys differ in format. Some are long UUID-style strings; others are short numeric strings. If your ledger's duplicate check uses the raw provider key, two different providers can produce keys that look distinct to your system but refer to the same logical payment attempt after a retry.

A robust approach is a normalization table:

```sql
CREATE TABLE provider_key_normalization (
    provider       TEXT NOT NULL,
    raw_key        TEXT NOT NULL,
    normalized_id  UUID NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (provider, raw_key)
);

CREATE INDEX ON provider_key_normalization (normalized_id);
```

The gateway then:

1. Receives a provider-specific key.
2. Looks it up in the normalization table.
3. If missing, generates a UUID and stores it.
4. Uses the UUID for all downstream operations.

The lookup adds a small amount of latency (typically low single-digit milliseconds on an indexed table), but it prevents duplicate payments that would otherwise require manual reconciliation.

A query to detect collisions in existing data:

```sql
SELECT
    provider,
    COUNT(*) AS total_events,
    COUNT(DISTINCT normalized_id) AS unique_payments
FROM payment_events
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY provider
HAVING COUNT(*) > COUNT(DISTINCT normalized_id);
```

Any row returned here means at least one provider key is being reused across events that your system treated as separate payments.

## Webhook signature versioning

Providers change webhook signature formats. A common transition is from a single header value:

```
X-Provider-Signature: sha256=abc123...
```

to a timestamped, multi-version format:

```
X-Provider-Signature: t=1712345678,v1=abc123...,v2=def456...
```

If each adapter has its own validation logic, the change must be applied everywhere. A gateway centralizes it with a versioned parser:

```go
func validateProviderSignature(signature string, payload []byte, secret []byte) error {
    parts := strings.Split(signature, ",")
    if len(parts) == 1 {
        // Legacy single-value format
        return legacyValidate(signature, payload, secret)
    }
    // Timestamped, multi-version format
    return multiVersionValidate(parts, payload, secret)
}
```

Add a gauge that records which signature version each provider is currently using:

```go
var webhookSignatureVersion = prometheus.NewGaugeVec(
    prometheus.GaugeOpts{
        Name: "webhook_signature_version",
        Help: "Detected webhook signature version per provider",
    },
    []string{"provider"},
)
```

When a provider silently upgrades, the gauge changes and an alert fires before signature validation starts failing in production.

## Handling provider-specific latency and retries

Provider latency is not constant. Regional load balancers, evening peaks, and internal incidents all produce jitter. A naive fixed retry policy (for example, three retries with exponential backoff) can make an outage worse by adding load to a struggling provider.

A more robust approach is to make retry policy a function of recent observed latency:

1. Track a per-provider latency histogram, updated on a short interval.
2. If recent latency exceeds a threshold, reduce retry count and increase timeout.
3. If latency exceeds a higher threshold, fail over to the next-best provider immediately.

A simplified implementation:

```rust
async fn route_payment(provider: &str, tx: &Transaction) -> Result<PaymentId, PaymentError> {
    let latency_ms = PROVIDER_LATENCY.get(provider).unwrap().load(Ordering::Relaxed);
    let max_retries = if latency_ms > 1000 { 1 } else { 3 };
    let timeout = if latency_ms > 500 {
        Duration::from_secs(3)
    } else {
        Duration::from_secs(1)
    };

    let mut retries = 0;
    loop {
        match call_provider(provider, tx, timeout).await {
            Ok(id) => return Ok(id),
            Err(e) if retries >= max_retries => return Err(e),
            Err(_) => {
                retries += 1;
                let backoff = Duration::from_millis(100 * 2u64.pow(retries as u32));
                tokio::time::sleep(backoff).await;
            }
        }
    }
}
```

The important design point is that retry policy is data-driven, not hard-coded. A provider that is healthy gets aggressive retries; a provider that is struggling gets fast failover.

## Circuit breakers and bulkheads

A gateway without circuit breakers is just a router with extra steps. The circuit breaker pattern prevents a slow provider from consuming all available connections.

A minimal state machine:

- **Closed**: requests pass through. Count failures.
- **Open**: requests fail fast without calling the provider. After a cooldown, move to half-open.
- **Half-open**: allow a limited number of probe requests. If they succeed, close the circuit; if they fail, reopen.

Bulkheads complement this by isolating resources per provider. Instead of one shared pool of 50 connections, allocate a maximum per provider so that one provider's retry storm cannot starve the others.

```yaml
providers:
  provider_a:
    max_connections: 20
    timeout_ms: 1000
    circuit_breaker:
      failure_threshold: 5
      cooldown_ms: 10000
  provider_b:
    max_connections: 15
    timeout_ms: 1500
    circuit_breaker:
      failure_threshold: 3
      cooldown_ms: 5000
  provider_c:
    max_connections: 15
    timeout_ms: 1200
    circuit_breaker:
      failure_threshold: 5
      cooldown_ms: 10000
```

The sum of `max_connections` should be less than your runtime's connection budget. If it is not, the bulkheads are decorative.

## Observability: what to instrument

The gateway's value depends on being able to answer three questions quickly:

1. Which provider is slow right now?
2. Which provider is failing right now?
3. Are we retrying more than usual?

The minimum instrumentation:

- A histogram of request duration per provider, with labels for endpoint and outcome.
- A counter of requests by provider and status class.
- A counter of retries per provider.
- A gauge of circuit breaker state per provider (0 = closed, 1 = half-open, 2 = open).
- A gauge of webhook signature version per provider.

A PromQL query for per-provider p99 latency:

```promql
histogram_quantile(
  0.99,
  sum by (le, provider) (rate(payment_duration_seconds_bucket[5m]))
)
```

A query for provider error rate:

```promql
sum by (provider) (rate(http_requests_total{status=~"5.."}[1m]))
/
sum by (provider) (rate(http_requests_total[1m]))
```

An alert rule that fires when a provider's 5xx rate exceeds a threshold:

```yaml
- alert: ProviderHighErrorRate
  expr: |
    sum by (provider) (rate(http_requests_total{status=~"5.."}[1m]))
    /
    sum by (provider) (rate(http_requests_total[1m])) > 0.05
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Provider {{ $labels.provider }} 5xx rate above 5% for 2 minutes"
```

The goal is not to have the most dashboards. The goal is to have one dashboard that answers the three questions above without switching contexts.

## How to decide which approach fits your situation

A decision checklist, in order of importance:

1. **Provider count**: One provider and no near-term plan to add another? An adapter is fine. Two or more providers with shared business logic? A gateway pays for itself.
2. **Transaction value**: High average transaction value makes provider downtime expensive. If a single provider outage can block revenue, centralize failover.
3. **Audit requirements**: If regulators or finance require one ledger for all providers, a gateway makes that ledger possible. Adapters make it a reporting problem.
4. **Team size**: A gateway is a system to operate. If there is no one to operate it, an adapter is the honest choice.
5. **Operational load**: Track how many engineer-hours per month go to provider-specific issues. If that number is growing faster than transaction volume, the adapter pattern is not scaling.

A useful heuristic: if provider-specific bugs consume more than a few engineer-days per quarter, the coordination cost of adapters is probably exceeding the build cost of a gateway. Measure it rather than guess.

## The cases where the adapter pattern is right

The adapter pattern still wins in two situations:

1. **Single-country, single-provider products**: If you are launching in one market with one provider and expect to stay there for the foreseeable future, the complexity tax of a gateway outweighs the benefit.
2. **Heavy ERP or legacy coupling**: If an existing enterprise system already has a native connector for one provider, inserting a gateway layer can break workflows that are expensive to change. Keep the adapter for that provider.

A hybrid is also legitimate: keep an adapter for a provider with deep legacy coupling, and build a gateway for the rest. The pain is localized, and the gateway handles the providers that actually need coordination.

## Objections and responses

**Objection: "A gateway is a single point of failure."**

A gateway is a single logical component, but it can be deployed as multiple replicas behind a load balancer. The relevant question is whether the gateway's failure modes are better understood than three independent adapters' failure modes. With circuit breakers and bulkheads, a gateway isolates provider failures rather than propagating them. Without them, a gateway is indeed a single point of failure — and so are the adapters.

**Objection: "The gateway adds latency because of indirection."**

Internal routing adds a small amount of latency, typically low single-digit milliseconds. External provider latency is usually hundreds of milliseconds to seconds. The net latency effect depends on whether connection reuse and bulkheads save more than the routing adds. Measure both.

**Objection: "Upgrading providers becomes harder."**

Upgrades become easier because the change lives in one place. A webhook signature change is one parser update, not a multi-service rollout. The trade-off is that a bug in the gateway affects all providers, which is why the gateway needs strong test coverage and staged rollouts.

**Objection: "We'll lose provider-specific features."**

Keep provider-specific features behind feature flags in the gateway. If one provider offers a special discount or payout API, expose it through a feature-flagged endpoint. You do not lose features; you centralize them.

## Instrumenting connection pools: a 30-minute exercise

If you are already running multiple adapters and want to know whether connection pooling is a problem, start by measuring it.

For a Node.js service, expose pool metrics and query them:

```bash
curl -s http://localhost:4000/metrics | grep -E 'pool_(size|active|idle)'
```

For a Go service using `database/sql`:

```go
db.Stats() // returns OpenConnections, InUse, Idle, WaitCount, WaitDuration
```

For a Rust service using a connection pool library, expose the pool's size and wait time as gauges.

What to look for:

- `WaitCount` or equivalent increasing over time means requests are waiting for connections.
- `InUse` consistently at the pool maximum means the pool is saturated.
- `Idle` near zero during traffic means there is no headroom for bursts.

If each provider has its own pool and each pool is saturated, the total connection count is likely higher than necessary. Consolidating pools behind a gateway, or at minimum sharing a pool with per-provider bulkhead limits, is the first structural change to try.

## A worked example: routing policy

Suppose you have three providers with different cost and reliability profiles. You want to route based on transaction value and observed health.

Define a policy:

```
if provider_health < 0.95:
    exclude provider
if transaction_value < 10:
    prefer cheapest provider
elif transaction_value < 100:
    prefer provider with lowest recent p99
else:
    prefer provider with highest uptime
```

This is a decision function, not a hard-coded route. It can be expressed in a small config file:

```yaml
routing:
  health_threshold: 0.95
  tiers:
    - max_value: 10
      strategy: lowest_cost
    - max_value: 100
      strategy: lowest_p99
    - max_value: null
      strategy: highest_uptime
  fallback: next_available
```

The gateway evaluates this on each request. The evaluation is cheap; the value is that the policy is visible and testable in one place.

## Testing the gateway

A gateway has more failure modes than an adapter, so testing matters more.

- **Unit tests** for payload normalization and response mapping.
- **Contract tests** for each provider's request and response schema, run against sandbox environments.
- **Chaos tests** that inject latency and errors into one provider and verify that the gateway fails over correctly.
- **Replay tests** that feed recorded webhooks (including old signature versions) through the gateway and verify idempotency.

The replay test is the one most teams skip and most regret skipping. Webhook signature changes and duplicate deliveries are the two most common production surprises, and both are reproducible offline.

## Summary

The adapter pattern is a code-organization pattern. The gateway pattern is a systems pattern. They solve different problems, and the choice between them depends on provider count, transaction value, audit requirements, and team capacity.

If you are building across multiple markets with multiple providers, a gateway with a single connection pool, circuit breakers, bulkheads, normalized idempotency keys, and unified observability is usually the more predictable system to operate. Start minimal: three endpoints, one pool, one breaker per provider. Add complexity when measurements justify it.

The most valuable first step is not to rewrite anything. It is to measure the connection pools and provider error rates you already have.

## Do this in the next 30 minutes

Pick one provider in your current integration and answer three questions with data:

1. What is its p99 request duration over the last 24 hours?
2. What is its 5xx rate over the last 24 hours?
3. How many connections does your service hold to it at peak?

If you do not have those numbers, add the instrumentation first. A single histogram and a single counter per provider is enough to start. Everything else in this article depends on being able to see those three numbers.
