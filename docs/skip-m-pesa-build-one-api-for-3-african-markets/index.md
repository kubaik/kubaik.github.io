# Skip M-Pesa: build one API for 3 African markets

Serving Kenya, Nigeria and Ghana usually starts as three integrations: M-Pesa in Kenya, a card-and-transfer processor in Nigeria, and MTN Mobile Money in Ghana. That means three API contracts, three sandbox lifecycles, three support channels, and three upgrade calendars. The setup works until the maintenance load catches up with the team.

A common failure mode is not the first integration but the fourth change to it. A provider adds a field, changes a response shape, or tightens validation, and suddenly a code path that has been stable for months breaks in production. Multiply that by three providers and the integration layer becomes the most expensive part of the product to own.

This article walks through a mental model that reduces that load: treat a payments orchestrator as the single API your application talks to, and keep provider-specific behaviour behind thin, versioned adapters. It also covers when the conventional advice — integrate directly with each provider — is genuinely correct, and how to decide.

## The conventional wisdom, and where it runs out

The standard advice is to integrate with each local provider directly. That is three APIs, three UAT cycles, three support matrices, three upgrade timelines. The approach is not wrong; it is incomplete. It assumes the integration layer is the only complexity, when in practice the business logic layer also starts branching on country codes for tax, currency formatting, error messages and refund rules. Every new product feature then has three code paths.

The second piece of conventional wisdom is to abstract the providers behind a single interface: `ProviderA.charge()`, `ProviderB.reverse()`, `ProviderC.refund()`. This feels DRY and looks clean on a diagram. It backfires when each provider adds custom fields the abstraction did not anticipate. A typical example: a reversal response includes a `reason_code` field that is not in the published spec, so the generic interface either drops it or has to grow a provider-specific escape hatch. The abstraction becomes leaky, and teams end up refactoring it after the fact.

There is also a compliance tail. Each country has its own regulator, sandbox and KYC rules. Treating them as three separate problems produces three separate compliance binders. Treating them as one problem produces a single binder with per-country sections, which is usually easier to audit.

## What actually goes wrong with three direct integrations

The problems are rarely in the happy path. They show up in the edges.

**Settlement files arrive in different formats and at different times.** A common pattern is one provider delivering CSV dumps overnight, another sending JSON over SFTP, and a third exposing XML over HTTPS with a delay tolerance. The reconciliation job has to merge three time zones, three encodings and three rate schemas while still closing the books by the morning. If any file is late or malformed, the finance team is chasing a cash position that does not tie out.

**Idempotency guarantees differ and are often under-documented.** A provider may publish that idempotency keys are valid for 24 hours, then in practice expire them sooner under load. This is the kind of detail that surfaces during a traffic spike, when a retry loop starts failing and the failure looks like a race condition in your own code. The root cause is a provider-side limit that was documented in a changelog or an issue tracker, not in the main API reference.

**Support becomes a dashboard problem.** When a customer says "I paid but your system shows pending", an agent has to check three dashboards. If the agent picks the wrong one, the customer gets told the money is already refunded when it is still in transit. A well-designed abstraction can hide the information the agent needs most. A poorly designed one hides it by default.

**Costs compound in ways that are easy to miss.** Three sandbox accounts, three production key sets, three webhook URL registrations, three sets of compliance artefacts. Add the engineering time to instrument each provider's metrics and the on-call rotation that now has three separate health checks.

## A different mental model: one orchestrator, three adapters

Instead of integrating with each provider directly, integrate with a single payments orchestrator that already supports the markets you need. Treat the orchestrator as the canonical source of truth for status, refunds and webhooks. Keep provider-specific quirks behind adapter layers that are versioned and tested separately. That shift turns "three integrations" into "one integration plus three adapters".

The orchestrator approach works because it centralises the reconciliation problem. A single webhook endpoint receives status updates from all providers, and the orchestrator's reconciliation engine merges settlement files into one ledger. Idempotency still has to be handled, but against a single API contract rather than three. The adapter layer translates the orchestrator's generic status codes into provider-specific ones, so your business logic does not branch by country.

A useful discipline: decide which layer owns which fact.

- The orchestrator owns payment status, refund state, and the canonical ledger.
- The adapter owns provider-specific field mapping, validation quirks, and any provider-imposed cooldowns.
- Your application owns business rules: pricing, tax, entitlements, and what the customer sees.

If a fact is owned by two layers, you will eventually have two versions of it and no way to tell which is right.

## Worked example: a refund across three providers

Consider a refund flow. The user asks for a refund of 5,000 KES on a payment made yesterday.

1. Your application calls the orchestrator's refund endpoint with the orchestrator's payment ID and the amount.
2. The orchestrator routes the refund to the correct provider based on the original payment.
3. The provider processes the refund and returns a status. The orchestrator normalises that status and emits a webhook.
4. Your adapter receives the webhook, maps the orchestrator status to whatever your internal model expects, and writes it to your ledger.

Now add a provider-specific rule. Suppose one provider enforces a cooldown on refunds — say, refunds cannot be issued until 24 hours after settlement. The orchestrator exposes a single refund endpoint and does not know about that cooldown. If the adapter does not implement the cooldown check, the refund request will be rejected by the provider, and depending on how the rejection is surfaced, the failure may look like a generic error rather than a rule violation.

The correct handling is to put the cooldown in the adapter, not in the application. The adapter knows the provider, so it can reject the request early with a clear error, and the application can present a sensible message to the user. Putting the cooldown in the application means every caller has to remember it, and eventually one will not.

The same pattern applies to field validation. If a provider requires a field that only exists in live (for example, a customer phone number in a specific format), the adapter should enforce it. The application should not know about provider-specific field names.

## Where the conventional wisdom is right

Direct integration is the correct choice in several cases, and it is worth being explicit about them.

**Offline or low-connectivity environments.** A point-of-sale terminal that must work without reliable internet cannot depend on an orchestrator's webhook delivery. It needs to talk to the provider's USSD or STK flow directly.

**Latency-critical paths.** Some flows, such as pre-authorisation or risk checks, have tight latency budgets. If a provider's risk API is synchronous and fast, routing it through an orchestrator adds a hop that may not be worth it. The orchestrator should handle the final charge; the pre-authorisation step can go direct.

**Regulatory requirements for raw provider data.** Some regulators require settlement reports to be generated from the provider's own ledger, not an intermediary's. If an auditor insists on seeing raw MIS reports from the provider, you need direct access to the provider's SFTP or API endpoints. No orchestrator can substitute for that.

**Low volume.** At very low transaction volumes, the overhead of building and maintaining adapters may outweigh the benefits. The orchestrator's pricing model — often a percentage plus a fixed fee per transaction — can be more expensive than direct integrations when volume is small. The break-even point depends on your team size and the fixed costs of running three integrations; it is worth calculating explicitly rather than assuming.

## How to decide: a checklist

Rather than a scored matrix with invented weights, use a checklist. Answer each question honestly.

1. **Do you need to support offline payments or USSD/STK flows?** If yes, you need at least one direct integration for that flow.
2. **Does any flow have a hard latency SLA under roughly one second end-to-end?** If yes, route that flow directly and keep the rest on the orchestrator.
3. **Does your regulator require raw settlement data from the provider?** If yes, keep a direct integration for compliance reporting.
4. **Is your team smaller than three engineers dedicated to payments?** If yes, start with the orchestrator. Three direct integrations will consume more capacity than you have.
5. **Are you launching a new product in the next quarter?** If yes, start with the orchestrator. You can add direct integrations later for specific flows.
6. **Are you extending an existing product that already has three direct integrations?** Migrate incrementally. Move one flow at a time and keep the direct integrations until the orchestrator covers the majority of use cases.
7. **Do you have a clear owner for adapter maintenance?** If not, either assign one or accept that adapters will rot.

If you answer yes to questions 1, 2 or 3, plan for a hybrid: orchestrator for most traffic, direct integrations for the specific flows that require them. If you answer yes to 4 or 5, start with the orchestrator and revisit later.

## Instrumentation: what to measure and how

Before writing the first adapter, decide what you will measure. The goal is to be able to answer three questions quickly: is reconciliation healthy, are adapters failing, and are idempotency keys being reused incorrectly.

A reasonable starting set of metrics:

- `payments_reconciliation_errors_total{provider}` — a counter of reconciliation errors by provider.
- `payments_adapter_idempotency_failures_total{provider}` — a counter of idempotency key failures by adapter.
- `payments_reconciliation_latency_seconds{provider}` — a histogram of reconciliation latency by provider.

To measure reconciliation latency, record the time from when the settlement file is received to when the ledger is updated and the balance is verified. Compare P50, P95 and P99 across providers. If one provider's P99 is an order of magnitude worse than the others, that is a signal to investigate before it becomes an incident.

To measure idempotency failures, count every time a request is rejected because the key was already used or has expired. A rising count in one adapter usually means the key generation logic is wrong for that provider.

To measure reconciliation errors, count every time the ledger does not balance after processing a settlement file. Alert on any non-zero value; reconciliation errors are the kind of thing that should be investigated immediately, not trended.

Prometheus is a common choice for these metrics, but the exact tooling does not matter. What matters is that the metrics exist before the first incident, not after.

## A concrete next step

Open your payments metrics file — for example `metrics/payments.go` — and add three Prometheus collectors:

```go
var (
    reconciliationErrors = promauto.NewCounterVec(prometheus.CounterOpts{
        Name: "payments_reconciliation_errors_total",
        Help: "Total reconciliation errors by provider",
    }, []string{"provider"})
    adapterIdempotencyFailures = promauto.NewCounterVec(prometheus.CounterOpts{
        Name: "payments_adapter_idempotency_failures_total",
        Help: "Total idempotency key failures by adapter",
    }, []string{"provider"})
    reconciliationLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
        Name:    "payments_reconciliation_latency_seconds",
        Help:    "Reconciliation latency in seconds",
        Buckets: prometheus.ExponentialBuckets(0.1, 1.5, 10),
    }, []string{"provider"})
)
```

Commit the file and push it. You will have real data on reconciliation errors within 24 hours of the first settlement file being processed. That data is the foundation for every decision that follows.

## Failure modes to plan for

A few failure modes recur often enough to be worth naming.

**Silent sandbox/live divergence.** Sandboxes are often more permissive than live endpoints. A field that is optional in sandbox may be mandatory in live. The fix is to test against a live-like sandbox tier if one exists, and to run a small set of smoke tests against production before enabling a new provider for all traffic.

**Idempotency key exhaustion.** If your keys are generated with a time component, make sure the time window matches the provider's actual key lifetime, not the documented one. A conservative approach is to use a composite key that includes the provider name and a timestamp truncated to a window shorter than the documented lifetime. That way, retries within the window are idempotent, and retries across windows are rejected by the provider's own anti-duplication logic.

**Reconciliation drift.** If your ledger and the provider's settlement file disagree, the cause is usually one of: a missing transaction, a duplicate transaction, a currency conversion applied in one place but not the other, or a timing difference across time zones. Build a reconciliation test that replays a month of settlement files and asserts the ledger balances to zero. Run it nightly.

**Adapter rot.** Providers change their APIs. If your adapters are not versioned and tested, the change will surface in production. Reserve engineering capacity for adapter maintenance — a common rule of thumb is around 10% of the team's time, but the right number depends on how many providers you integrate with and how often they change.

## Summary

If you are building a payment system that spans Kenya, Nigeria and Ghana, start with a single payments orchestrator and thin adapters. Treat the orchestrator as the canonical source of truth for status, refunds and webhooks. Keep provider-specific quirks behind versioned adapter layers. This reduces the maintenance tax of three separate integrations and gives you a single reconciliation pipeline.

Instrument before you build. Add Prometheus metrics for reconciliation latency, adapter errors and idempotency failures. Run adapter tests against a live-like sandbox tier before every release. Reserve capacity for adapter churn.

If your product is offline, latency-critical, or subject to strict compliance audits, you may still need direct integrations — but only for those specific flows. Keep the rest of the traffic on the orchestrator. The hybrid approach is usually the one that scales without turning the codebase into a collection of country-specific branches.

## FAQ

**How do I handle sandbox versus live differences for a mobile money provider?**

Sandboxes are often permissive by design, returning success for requests that would fail in production. Test against a live-like sandbox tier if the provider offers one, and run a small smoke test against production before enabling a new provider for all traffic. Treat any field that is required in production but optional in sandbox as a bug in your test coverage, not a quirk to work around.

**What is the fastest way to reconcile three providers into one ledger?**

Use the orchestrator's reconciliation engine as the source of truth, then pull each provider's settlement files once per day and merge them into a single ledger. The webhook gives you real-time status; the settlement files give you final amounts. Make the reconciliation job idempotent so that a duplicated file still results in a balanced ledger.

**How do I avoid idempotency key exhaustion at high volume?**

Use a composite key that includes the provider name and a timestamp truncated to a window shorter than the provider's documented key lifetime. If the provider documents a 24-hour lifetime but expires keys sooner under load, a shorter window gives you a safety margin. Retries within the window are idempotent; retries across windows are rejected by the provider's anti-duplication logic.

**Why does a sandbox accept card numbers that production rejects?**

Sandboxes are often configured to accept happy-path inputs so developers can test flows without real card data. Production enforces validation rules that the sandbox does not. Use a sandbox tier that mirrors production validation if one is available, and never assume that a successful sandbox test implies a successful production test.

**Do I still need direct integrations if I use an orchestrator?**

Only for specific flows. Offline payments, latency-critical pre-authorisation, and regulatory requirements for raw provider data are the three common reasons to keep a direct integration. Everything else can go through the orchestrator.

## Take action in the next 30 minutes

Open your payments metrics file and add the three Prometheus collectors shown above. Commit and push. If you do not yet have a payments metrics file, create one and wire it into your application's metrics endpoint. You will have real data on reconciliation errors and idempotency failures within 24 hours — and that data is what turns the decision between direct integrations and an orchestrator from an argument into an engineering question.
