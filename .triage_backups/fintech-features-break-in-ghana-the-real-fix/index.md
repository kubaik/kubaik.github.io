# Fintech features break in Ghana: the real fix

## Why the same code passes in one market and fails in another

A payment or refund endpoint can behave correctly against a Nigerian sandbox and fail intermittently against a Ghanaian production endpoint with no code change. The symptoms are usually vague: a generic `upstream request timeout`, a `400 Bad Request` with no field-level detail, or a `5xx` that appears only during certain hours. Because the failure is intermittent and market-specific, teams often start by re-reading their own code, which is usually correct. The problem is not the code path; it is an assumption baked into the code that happens to hold in one market and not another.

These assumptions are rarely written down. They live in a global timeout constant, a validation regex, a retry schedule, or a cron expression. Each one encodes a fact about a specific market's telecoms, banking, or regulatory environment, and each one silently becomes wrong the moment the integration is pointed at a different country.

A useful framing is to treat "market" as a first-class input to your application, in the same way you treat currency or locale. Any value that varies by market — timeout, rate limit, identifier format, retry delay, minimum amount — should be selected by that input rather than hardcoded. The rest of this article covers the categories where this matters most, how to diagnose each one with real instrumentation, and how to verify a fix.

## The four categories of hidden regional dependency

| Category | Typical symptom | Why it escapes local testing |
|---|---|---|
| Time zone and DST handling | Cron job or scheduled retry fires at the wrong local hour | Developer machines and CI run in UTC, so offset bugs are invisible |
| Provider timeout mismatch | Intermittent `upstream request timeout` in one market only | A single global timeout is tuned to the fastest sandbox |
| Regulatory identifier formats | `400 Bad Request` or `INVALID_REFERENCE` with no field detail | Mocks use plausible-looking IDs that satisfy the regex but not the real format |
| Currency minor units and rounding | `AMOUNT_TOO_SMALL` or a zeroed amount after rounding | Unit tests use whole numbers, so sub-unit truncation never triggers |

The common thread is that none of these produce a stack trace pointing at the real cause. They surface as generic upstream errors, which is why the first diagnostic step is always to make the market, carrier, and timestamp visible in your logs and metrics.

## Diagnosing a provider timeout mismatch

**Symptom pattern:** a feature works in sandbox and in one production market but times out in another, with a generic upstream timeout in the logs.

The most common cause is a single global timeout tuned to the fastest provider you tested against. Sandbox environments are frequently faster and more forgiving than production, and different providers enforce different server-side limits. A client timeout shorter than the provider's own processing window produces a client-side abort that looks like a network failure.

The fix is a per-market timeout map plus an explicit abort controller, so the timeout is a deliberate configuration value rather than a library default:

```javascript
// markets.js
const MARKET_TIMEOUTS_MS = {
  NG: 10_000,
  KE: 12_000,
  GH: 15_000,
  SN: 14_000,
};

// refund.js
async function requestRefund(payload, market) {
  const timeout = MARKET_TIMEOUTS_MS[market] || 12_000;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(providerUrl(market), {
      method: 'POST',
      signal: controller.signal,
      headers: { 'X-Market': market },
    });
    clearTimeout(id);
    return res;
  } catch (err) {
    if (err.name === 'AbortError') {
      log.error('timeout', { market, timeout });
      throw new Error(`REFUND_TIMEOUT_MARKET_${market}`);
    }
    throw err;
  }
}
```

Note that the timeout values above are illustrative placeholders. The correct values come from measurement, not from copying a table out of an article.

### How to measure the right timeout per market

Do not guess. Instrument the provider call and record the duration distribution per market:

1. Emit a histogram (or at minimum p50/p95/p99) of provider response time, tagged by market and endpoint.
2. Run the measurement in the environment that matters — production or a staging environment that mirrors production network paths and provider behaviour.
3. Set the client timeout above the p99.9 of successful calls for that market, with headroom for provider-side queueing during peak hours.
4. Re-measure after any provider or infrastructure change.

A practical way to get the distribution without waiting for organic traffic is to replay recorded production requests against staging:

```bash
# Replay a captured request set and print a latency summary
vegeta attack -rate 100/60s -duration 5m -targets refund-targets.txt | vegeta report -type=hdrplot
```

The `hdrplot` output gives a high-dynamic-range histogram you can read a p99.9 from. Compare that number across markets before choosing timeout values. If a market's p99.9 sits close to the provider's documented processing limit, the correct fix may be an asynchronous flow (submit, then poll for status) rather than a longer synchronous timeout.

## Diagnosing identifier format mismatches

**Symptom pattern:** validation passes locally, but the provider rejects the request with a generic `400 Bad Request` or `INVALID_REFERENCE`, and the error body carries no field name or expected format.

National identifier formats are not uniform across markets. Length, character set, and separator conventions differ, and some markets use several identifier types (national ID, social security number, tax ID) with different rules. A single regex tuned to one market will accept inputs that are structurally valid for another but semantically wrong.

The trap is compounded by mocks. A test fixture like `NIN1234567890` satisfies an alphanumeric regex but tells you nothing about whether a real Senegalese identifier would pass. When a real identifier is longer or numeric-only, the regex either rejects it (a loud failure, which is fine) or accepts it and defers the failure to the bank (a quiet failure, which is expensive).

The fix is to make validation market-aware and to log the market alongside the rejection:

```javascript
// validators.js
const ID_FORMATS = {
  NG: /^[A-Z0-9]{11}$/i,
  GH: /^[A-Z0-9]{11}(-[A-Z0-9]{4})?$/i,
  SN: /^\d{13}$/,
  KE: /^[A-Z0-9]{11}$/i,
};

export function validateNationalId(id, market) {
  const format = ID_FORMATS[market];
  if (!format) return false;
  if (!format.test(id)) return false;
  return true;
}
```

The patterns shown are illustrative shapes, not authoritative specifications. Confirm the exact rules for each market against the provider's current integration documentation before shipping, and version the map so a format change is a reviewable diff.

### How to measure identifier rejection rates

Track two separate metrics, because they fail at different layers:

- `id_validation_failure_total`, tagged by market — rejections caught by your own validator. A spike here means your regex is too strict for real users.
- `provider_rejection_total{reason="INVALID_REFERENCE"}`, tagged by market — rejections that passed your validator and failed at the provider. Any sustained non-zero rate here means your validator is too permissive.

The second metric is the important one. It is the count of requests you forwarded that the provider refused on format grounds, and it should be near zero. If it is not, capture the rejected identifier's shape (length, character classes, position of separators — never log the full identifier) and compare it against your regex.

## Diagnosing rate limiting and retry behaviour

**Symptom pattern:** failures cluster at particular hours, or appear only under load, and the logged error is a timeout or connection reset rather than an explicit `429`.

Providers commonly enforce per-carrier or per-account request ceilings, and sandboxes often do not enforce them at all. When a limit is exceeded, the provider may return `429 Too Many Requests`, but it may also close the connection or return a `5xx`. That means your retry logic can be reacting to a rate limit it never recognises as one, and retrying harder makes the problem worse.

The fix has two parts: a per-carrier token bucket that keeps you under the limit, and exponential backoff with jitter that respects `Retry-After` when the provider sends it.

```python
# rate_limiter.py
import time

CARRIER_LIMITS_PER_MINUTE = {
    'mtn-gh': 500,
    'airtel-gh': 400,
    'safaricom-ke': 600,
    'glo-ng': 300,
    'orange-sn': 450,
}

class TokenBucket:
    def __init__(self, rate_per_minute, burst=None):
        self.rate = rate_per_minute / 60.0
        self.capacity = burst or rate_per_minute
        self.tokens = self.capacity
        self.updated = time.monotonic()

    def consume(self, n=1):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

buckets = {}

def check_rate_limit(carrier: str) -> bool:
    limit = CARRIER_LIMITS_PER_MINUTE.get(carrier)
    if limit is None:
        return True  # unknown carrier: fail open, but alert on this
    bucket = buckets.setdefault(carrier, TokenBucket(limit))
    return bucket.consume(1)
```

The limits above are illustrative. Real ceilings are per-provider, sometimes per-account, and often undocumented until you hit them.

### How to measure the real ceiling

You cannot read most of these limits from documentation, so measure them deliberately in a non-production environment:

1. Send a controlled ramp of requests to the provider's sandbox or a test account, increasing rate in steps.
2. Record the rate at which the first `429` (or connection reset) appears, and the exact response headers.
3. Record whether the provider sends `Retry-After`, and in what unit.
4. Confirm the observed ceiling against your own traffic: plot `requests_per_minute` and `provider_rejection_total{reason="rate_limited"}` on the same axis and look for the knee.

Once you know the ceiling, size the token bucket below it, and log every local rate-limit rejection so you can distinguish "we throttled ourselves" from "the provider throttled us".

## A worked example: tracing one failing refund

Suppose a refund endpoint shows a 3% failure rate in one market and 0% elsewhere. Here is a reasoning sequence that narrows the cause without guesswork.

1. **Split the metric by market.** If the failure is confined to one market, the bug is in a market-dependent value, not in shared logic.
2. **Check the error class.** `AbortError` points at a client timeout. A `400` with `INVALID_REFERENCE` points at validation. A `429` or connection reset points at rate limiting.
3. **Compare the configured value to the measured distribution.** If the failing market's p99.9 provider latency exceeds the configured timeout, the timeout is the cause. If it does not, look elsewhere.
4. **Replay a failing request with the market header set.** If it succeeds on replay, the failure is load- or timing-dependent, which points at rate limiting or provider-side queueing rather than a deterministic validation bug.
5. **Check the time-of-day distribution.** Failures concentrated in two daily windows strongly suggest a rate limit tied to peak traffic.

Each step produces evidence that either confirms or eliminates a category. The value of the sequence is that it avoids the common failure mode of "add more retries" before the cause is known — which, for a rate-limit problem, makes the failure rate worse.

## Verifying a fix

Verification has to reproduce the conditions that caused the failure, which means per-market load with realistic timing, not a single happy-path test.

```javascript
import http from 'k6/http';
import { check } from 'k6';

const MARKETS = ['NG', 'KE', 'GH', 'SN'];
const TIMEOUTS = { NG: 10000, KE: 12000, GH: 15000, SN: 14000 };

export default function () {
  const market = MARKETS[Math.floor(Math.random() * MARKETS.length)];
  const timeout = TIMEOUTS[market];
  const res = http.post(`https://staging.example.com/refunds`, JSON.stringify({ market }), {
    timeout: timeout,
    tags: { market },
  });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'timeout respected': (r) => r.timings.duration <= timeout,
  });
}
```

Run it with `k6 run --vus 50 --duration 5m refunds.js` and inspect the per-market error rate in the summary. Tagging by market is what makes the result actionable; an aggregate error rate hides exactly the signal you need.

For validation, a table-driven test that covers each market's real format is worth more than a handful of hand-written cases:

```python
# test_id_formats.py
import pytest
from validators import validate_national_id

@pytest.mark.parametrize('market,id_value,valid', [
    ('NG', 'AB123456789', True),
    ('NG', 'ab123456789', True),
    ('NG', '123456789', False),
    ('GH', 'AB123456789', True),
    ('GH', 'AB123456789-1234', True),
    ('SN', '1234567890123', True),
    ('SN', '123456789012', False),
])
def test_national_id(market, id_value, valid):
    assert validate_national_id(id_value, market) == valid
```

Run with `pytest test_id_formats.py -v`. Every one of these fixtures should be derived from the provider's documented format, and the test file should be updated whenever a provider changes its rules.

For rate limiting, replay a burst above the measured ceiling and confirm the client throttles before the provider does:

```bash
vegeta attack -rate 600/60s -duration 2m -targets refund-targets.txt | vegeta report
```

The expected outcome is that your local limiter rejects the excess and the provider never returns `429`. If the provider still returns `429`, your ceiling estimate is too high.

Finally, instrument the outcomes you care about and alert on them per market:

- provider call duration (p50/p95/p99), tagged by market and endpoint
- provider rejection count, tagged by market and reason
- local rate-limit rejections, tagged by carrier
- identifier validation failures, tagged by market

An alert on provider rejections for a single market will catch a format regression within hours. An alert on local rate-limit rejections will catch a mis-sized bucket before it becomes a provider-side incident.

## A decision checklist for new market integrations

Before enabling a new market, confirm each of the following has a measured value rather than a copied default:

- Client timeout per market, set above the measured p99.9 for that market
- Provider rate ceiling per carrier, measured, with a token bucket sized below it
- Identifier formats per market, taken from current provider documentation and covered by table-driven tests
- Currency minor units and rounding mode, with amounts handled as integers in minor units or as `Decimal`, never as floats
- Cron and scheduled jobs stored in UTC and converted at runtime, with DST rules verified per market
- Market, carrier, and timestamp present in every log line and metric tag for provider calls
- Sandbox-versus-production differences documented in the repository and reviewed when they change

## Common follow-up questions

**Why does the sandbox behave differently from production?**

Sandboxes typically skip rate limiting, use faster or more lenient processing, and may not enforce the same identifier validation. Treat a sandbox pass as evidence that the request shape is accepted, not that the integration is production-ready.

**Should the timeout be longer or should the flow be asynchronous?**

If the measured p99.9 for a market is close to the provider's documented maximum processing time, a longer synchronous timeout only moves the failure. An asynchronous submit-and-poll flow is usually the correct design for slow operations such as refunds.

**How do you avoid hardcoding limits that change?**

Keep market-dependent values in configuration, version them in the repository, and review changes as diffs. A value in a config file with a comment explaining how it was measured is reviewable; a constant buried in a function is not.

**What should be logged when a provider rejects a request?**

The market, carrier, endpoint, request identifier, and the provider's error code. Never log full national identifiers, account numbers, or other personal data — log the shape (length, character classes) instead, which is enough to diagnose a format mismatch.

## Do this in the next 30 minutes

Open the file that defines your provider client timeout and replace the single global constant with a map keyed by market. Then add a histogram metric for provider call duration tagged by market, and deploy it. You will not yet know the right timeout values, but you will have started collecting the data that tells you what they are — which is the only reliable way to set them.
===END===
