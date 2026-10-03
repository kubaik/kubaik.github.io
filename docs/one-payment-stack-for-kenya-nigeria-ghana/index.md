# One payment stack for Kenya, Nigeria, Ghana

## Why three integrations look cheaper than they are

The default plan for Kenya, Nigeria and Ghana is three integrations: M-Pesa for Kenya, a card/bank provider such as Flutterwave or Paystack for Nigeria, and MTN Mobile Money or Vodafone Cash for Ghana. Each gets its own repository, webhook handler, reconciliation job, alerting rules and support runbook. On a whiteboard this looks like clean separation of concerns. In production it usually means the same business problem — take money, confirm money, refund money — is solved three times with three sets of bugs.

The failure mode is rarely the happy path. It is the overlapping path. A user starts an M-Pesa STK push, the prompt times out, and they immediately retry with a card. If the two flows live in separate stacks with separate idempotency keys, both can succeed and the customer is charged twice. The refund is fast, but the support ticket, the chargeback risk and the trust damage are not. A typical failure mode is not "provider is down"; it is "two providers both said yes."

The conventional advice also assumes each country has one dominant provider and that users stay with it for the whole session. Real behaviour is messier: users switch rails because of network failures, insufficient float, or a balance check that fails mid-flow. An architecture that treats each rail as a separate product cannot see that a single user is doing one thing — trying to pay.

Finally, provider APIs are not stable. Outages, deprecations and contract changes happen on the provider's schedule. Three stacks mean three places to absorb a breaking change, three alerting pipelines to tune, and three on-call surfaces. The cost is not the code you write; it is the coordination you inherit.

## What actually breaks under the three-stack model

**Duplicate charges across rails.** Without a shared idempotency key and a single attempt record, retries and user-driven fallbacks can produce two successful captures. The fix is architectural, not a patch: one `PaymentAttempt` row per user intent, with provider attempts as children.

**Divergent retry policies.** Each stack tends to get tuned independently, often from folklore rather than measurement. One stack retries three times, another five, another none. The result is inconsistent latency and inconsistent failure rates that are hard to compare because the denominators differ.

**Reconciliation tax.** Every provider names the same concepts differently. A phone number may arrive as `MSISDN`, `customer.phone` or `subscriberId`. A reference may be `tx_ref`, `flwRef` or a provider transaction ID. When a weekend batch has to be matched across three schemas, the work is field mapping, duplicate detection and agreeing on what "success" means. That is engineering time spent on translation, not on product.

**Compliance drift.** Data-residency, encryption-at-rest and audit-log requirements differ by jurisdiction and are enforced differently by each regulator. Three pipelines mean three chances for a misconfiguration. A single misconfigured storage rule can put sensitive data somewhere it should not be for hours before anyone notices, and the remediation cost is measured in regulatory exposure, not just engineering hours.

**Blast radius and observability.** With three stacks, one provider's incident affects only its own stack — that part is genuinely good. But it also means three dashboards, three SLOs and three alert thresholds. Correlated failures (for example, a shared upstream network issue) are invisible because nothing joins the data.

## A different mental model: one domain, many adapters

Instead of three stacks, define one domain language and let adapters translate. The core objects are small:

- `PaymentAttempt` — one row per user intent to pay. Owns the idempotency key, amount, currency, customer and chosen provider.
- `PaymentConfirmation` — the provider's authoritative result, normalized.
- `RefundRequest` — a request against a confirmation, with its own idempotency key.

The abstraction exposes a narrow interface. Everything provider-specific lives behind it.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PaymentResult:
    status: str                 # "succeeded" | "failed" | "pending"
    reference: str              # canonical reference for this attempt
    provider: str               # "mpesa_ke", "flutterwave_ng", ...
    provider_fee_minor: int     # fee in minor units of `currency`
    currency: str
    raw_response: dict          # full provider payload, unmodified

class PaymentGateway:
    async def attempt(
        self,
        amount_minor: int,
        currency: str,
        customer_id: str,
        idempotency_key: str,
        provider_hint: Optional[str] = None,
    ) -> PaymentResult:
        """Attempt a payment. Must be idempotent on idempotency_key."""
        ...
```

Two details matter more than they look. First, `idempotency_key` is a required argument, not an afterthought — it is what prevents the duplicate-charge failure mode. Second, `raw_response` is carried through unmodified so provider-specific fields remain available for refunds and debugging. The abstraction normalizes; it does not erase.

Each provider becomes an adapter implementing the same contract:

```python
class ProviderAdapter:
    name: str
    country: str
    currency: str

    async def charge(self, attempt) -> PaymentResult: ...
    async def refund(self, confirmation, amount_minor: int, idempotency_key: str) -> PaymentResult: ...
    def can_handle(self, error: Exception) -> bool: ...
    def validate_compliance(self, attempt) -> None: ...
```

Adding a new rail is one new adapter, not a new stack.

## Fallback as a first-class concern

Treat providers as fallbacks, not as fixed primaries. The user's preferred rail is tried first; on a retryable failure, the abstraction selects the next rail from a score table ordered by measured latency, error rate and cost. The score table should be derived from your own telemetry, not from published marketing numbers.

A useful shape for the table:

| Field | Meaning | Source |
|---|---|---|
| provider | adapter name | config |
| country | ISO country code | config |
| p95_latency_ms | 95th percentile charge latency | your metrics |
| error_rate | failures / attempts over window | your metrics |
| fee_bps | effective fee in basis points | provider contract |
| concurrency_limit | max in-flight requests | provider contract |
| circuit_state | closed / open / half-open | circuit breaker |

The exact numbers will differ per merchant, per corridor and per time of day, so do not copy a table — generate one. A window of 5–15 minutes is usually short enough to react to an incident and long enough to avoid flapping. Demote a provider when its error rate crosses a threshold you have chosen, and re-admit it gradually via a half-open state.

Two rules keep fallback safe:

1. **Fallback only on retryable errors.** A timeout or a 5xx is retryable. A declined card or insufficient balance is not; retrying those just generates noise and can look like fraud.
2. **Never fall back across currencies silently.** If the user intended to pay in KES, do not quietly charge a card in NGN. Convert explicitly at the boundary and record both the original and settled amounts on the attempt.

## Measuring instead of guessing

Any claim about failure rates, latency or savings has to come from your own instrumentation. The measurement plan is simple enough to run in an afternoon.

**Instrument the attempt lifecycle.** Emit a structured event for every state transition: `attempt_created`, `charge_sent`, `charge_response`, `confirmation_received`, `refund_requested`, `refund_settled`. Include `attempt_id`, `provider`, `country`, `currency`, `latency_ms`, `error_class` and `idempotency_key`.

**Compute the rates you actually care about.**

- Attempt failure rate = failed attempts / total attempts, per provider and per country.
- Duplicate rate = attempts with more than one successful confirmation / total attempts. This should be zero; if it is not, your idempotency is broken.
- Fallback rate = attempts that succeeded on a non-preferred provider / total attempts.
- Reconciliation break rate = transactions that fail automated matching / total transactions.

**Watch the tail, not the mean.** A provider with a 400 ms median and a 9 s p99 will hurt you at peak. Track p50, p95 and p99 separately.

**Run a synthetic probe.** A scheduled job that creates a small real or sandbox payment against each provider on a fixed cadence gives you an independent signal that does not depend on customer traffic. Compare the synthetic result against your production metrics; divergence usually means the probe is hitting a different path than real users.

**Alert on error budget, not on individual errors.** Define an SLO per provider (for example, 99% of charges resolve within 30 seconds) and page only when the burn rate over a window exceeds your threshold.

## Reconciliation and the canonical schema

Pick one canonical schema and map every provider into it. The choice matters less than the discipline of having exactly one. A minimal canonical transaction record:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CanonicalTransaction:
    attempt_id: str
    provider: str
    provider_reference: str
    amount_minor: int
    currency: str
    status: str                 # "succeeded" | "failed" | "pending" | "refunded"
    customer_phone_e164: str
    created_at: datetime
    settled_at: datetime | None
    idempotency_key: str
    raw_response: dict
```

Map provider fields into this shape in the adapter, not in the reconciliation job. Then reconciliation is a single join on `attempt_id` and `provider_reference`, and the break rate becomes a number you can trend.

For each provider, document the mapping explicitly in code and in a test fixture. When a provider renames a field, the fixture fails and you know immediately which adapter broke. That is the whole point of the pattern: provider changes land in one file.

## Compliance as an adapter responsibility

Regulatory requirements differ by country and change over time. Rather than building a separate pipeline per country, put a `validate_compliance` step in each adapter that runs in the same transaction as the charge.

Typical checks include:

- Data residency: ensure personally identifiable information is written to storage in the permitted region.
- Encryption: ensure sensitive fields are encrypted at rest with the required key management.
- Audit logging: emit an immutable audit record in the format the regulator expects.
- Transaction limits: reject attempts that exceed the customer's verification tier.

Because the check runs before the request leaves your system, an invalid attempt never reaches the provider and never gets persisted in a non-compliant form. When a rule changes, you change one adapter and its tests.

Keep the rules in configuration where possible, and version them. A compliance rule that lives only in a developer's memory is a future incident.

## A worked sizing example (illustrative)

The following numbers are illustrative, chosen to show the arithmetic rather than to predict any particular team's outcome. Substitute your own.

Assume a team spends 40 engineer-hours per month maintaining each of three stacks: dependency upgrades, provider contract changes, reconciliation fixes and on-call follow-ups. That is 120 engineer-hours per month. At a fully loaded cost of $100 per hour, the maintenance cost is:

```
120 hours/month × $100/hour = $12,000/month
```

Assume a unified abstraction costs 300 engineer-hours to build and 20 engineer-hours per month to maintain:

```
Build:       300 hours × $100/hour = $30,000 one-time
Maintenance:  20 hours/month × $100/hour = $2,000/month
```

The monthly saving is $12,000 − $2,000 = $10,000. Payback period:

```
$30,000 / $10,000 per month = 3 months
```

The result is sensitive to the maintenance estimate, which is the number teams most often get wrong. Measure it before committing: count the hours your team actually spent on provider-related work over the last two months, then use that figure. If the real number is 15 engineer-hours per stack per month, the saving is only $2,500 per month and payback stretches to a year. The abstraction is worth building when the measured maintenance burden is high and traffic spans more than one country.

## When separate stacks are the right call

The abstraction is not universally correct. Three cases justify keeping stacks apart:

**Very low volume in a country.** If a corridor processes a handful of transactions per day, the maintenance cost of an adapter, its tests and its monitoring can exceed the value. Use the provider's hosted checkout page and skip the integration until volume justifies it.

**A mandated in-country switch with no standard API.** Some jurisdictions require routing through a specific domestic switch. If that switch exposes a proprietary interface, isolate it in its own adapter and keep the rest of the system on the common contract. Do not contort the abstraction to fit one provider.

**A legacy system with no tests.** Rewriting three untested stacks into an abstraction is a large, risky change. Migrate incrementally: build the abstraction alongside the existing code, route one country through it, verify for a full billing cycle, then move the next. Each migration step should be independently reversible.

## Decision checklist

Use this before writing adapter code.

- Does more than one country contribute meaningful transaction volume? If not, defer.
- Have you measured the current maintenance cost per stack over at least two months?
- Do you have a canonical schema with a written mapping for every provider you support today?
- Is `idempotency_key` required on every charge and refund call?
- Can you classify every provider error as retryable or terminal, with tests?
- Is there a single place where a provider contract change must be absorbed?
- Do you have per-provider p50/p95/p99 latency and error-rate metrics?
- Is there a synthetic probe independent of customer traffic?
- Are compliance checks executed before the request leaves your system?
- Can you roll a single country back to its previous path without a deploy?

If several answers are "no", fix those before building the abstraction. The abstraction amplifies whatever discipline you already have; it does not create it.

## FAQ

**Should we just use one global provider instead?**
If a single provider covers every corridor you need, at acceptable cost and with the features you require, that is simpler than building an abstraction. Verify coverage per country rather than assuming it, and keep a fallback path for the corridors where coverage is partial or the provider has a history of outages.

**How do we handle currency conversion?**
Convert explicitly at the boundary and store both the original and settled amounts on the attempt. Use a rate source you can audit, record the rate and timestamp used, and reconcile against it. Never let a fallback silently change the currency the customer agreed to pay in.

**What happens when a provider changes its API contract?**
The change should be absorbed by one adapter and caught by its test fixtures. If a contract change requires edits in more than one place, the abstraction boundary is in the wrong position.

**How do we debug a failed payment?**
Give every attempt a `trace_id` that spans the adapter call, the provider response and the confirmation. Expose an internal endpoint that returns the attempt record, the adapter logs, the score-table entry at the time of the attempt and the circuit-breaker state. That single view is usually enough to distinguish a provider failure from a bug in your own fallback logic.

**Won't retrying across providers hit rate limits?**
It can, so bound it. Apply exponential backoff with jitter per provider, enforce a per-provider concurrency limit, and stop retrying when the circuit is open. Track rate-limit responses as a distinct error class so you can tune the limit from data rather than guesswork.

## Do this in the next 30 minutes

Pick one provider you already integrate with and add three structured log fields to its charge path: `attempt_id`, `idempotency_key` and `latency_ms`. Then query the last 24 hours and compute the failure rate and the p95 latency for that provider alone. That single number tells you whether your current stack is healthy, and it is the first column of the score table you will need if you ever build the abstraction.
