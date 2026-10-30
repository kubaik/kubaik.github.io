# Currency switch fails in Ghana: one config file to rule…

A recurring failure mode in multi-country payment backends is a currency switch that silently resolves to the wrong currency. The UI renders Ghanaian cedi, the log line says the service fell back to Kenyan shillings, and no exception is thrown. The symptom is confusing because nothing looks broken: the request succeeds, a price is returned, and only the currency is wrong.

This article covers the underlying cause, a layered fix, and how to verify the fix before it reaches production. The examples use Node.js and Python, but the pattern applies to any stack.

## The failure mode

A typical log line looks like this:

```
2026-05-15T09:32:37.441Z ERROR currency_service: using fallback currency KES for request 8f3a1…
```

The request came from a Ghanaian user, the app displayed GHS, and the backend treated it as KES. The message does not say "wrong currency" — it reports a fallback, which reads like expected behavior.

Two things make this hard to diagnose:

1. **The fallback is silent.** No exception, no 4xx, no alert. Downstream pricing, ledger entries, and receipts are all internally consistent — just denominated in the wrong currency.
2. **The code path is identical across countries.** The same service that works for Kenyan traffic fails for Ghanaian traffic because the currency code arrives through a different channel.

Teams commonly assume the bug is in the frontend locale detection, add the new currency code to a supported list, and redeploy. That works in a local environment where the currency source is hard-coded, then fails again in staging because the actual traffic never populated that field.

## Why the currency code gets lost

Mobile money and card APIs do not share a single convention for transmitting the currency code. A non-exhaustive set of patterns seen in the wild:

- The code arrives in a request body field such as `currencyCode`.
- The code arrives in a custom header such as `X-Currency` or `X-Currency-Code`.
- The code arrives as a query parameter such as `currency`.
- The code arrives in a body field named `currency_code`.

A first-pass Express middleware often looks like this:

```javascript
// typical first-pass implementation
app.use((req, res, next) => {
  req.currency = req.query.currency || req.body.currency || 'KES';
  next();
});
```

This has two defects. First, the default is a specific currency, so any request that does not match the two checked locations is silently mislabeled. Second, adding more sources as `else if` branches produces a combinatorial mess as providers multiply.

Global payment gateways add a second layer of friction: many assume a single settlement currency per account, so per-request currency switching has to be handled in your own layer before the gateway call. Teams that skip that layer end up forking codebases per country, which doubles review load and blocks cross-border features.

## Fix 1: extract from all sources in a deterministic order

Replace the assumption that the currency is always in the same place with an ordered list of candidate locations. The first valid match wins.

```javascript
// currencyExtractor.js
const currencyPriority = [
  'headers.x-currency',
  'headers.x-currency-code',
  'query.currency',
  'body.currency',
  'body.currencyCode',
  'headers.currency'
];

function getNested(obj, path) {
  return path.split('.').reduce((acc, part) => acc?.[part], obj);
}

function extractCurrency(req) {
  for (const path of currencyPriority) {
    const value = getNested(req, path);
    if (typeof value === 'string' && /^[A-Z]{3}$/.test(value)) {
      return value;
    }
  }
  return 'KES'; // fallback only if nothing is found
}
```

Two details matter. The regex `/^[A-Z]{3}$/` rejects lowercase, whitespace-padded, and malformed values, so `ghs` and `GHS ` do not leak into pricing logic. And the priority order is data, not control flow, so adding a provider means adding one string to an array rather than adding a branch.

Note that the fallback here is still dangerous. A safer production version throws instead of defaulting; see Fix 2.

## Fix 2: validate at extraction time, not in the business logic

A second common failure is validation that runs too late. A Senegalese user sends XOF in `body.currency_code`. The extractor returns it correctly. Then the service layer rejects it because the allow-list was written when only four currencies were supported:

```javascript
// typical late validation
const ALLOWED_CURRENCIES = new Set(['GHS', 'NGN', 'KES', 'UGX']);

function createPrice(req) {
  if (!ALLOWED_CURRENCIES.has(req.currency)) {
    throw new Error(`Unsupported currency ${req.currency}`);
  }
  // build price object...
}
```

By the time this throws, the request has already spent time in the service layer, a spinner has been shown, and the client receives a 400 after a multi-second wait:

```json
{
  "error": "Unsupported currency XOF"
}
```

Move validation into the extractor so failure is fast and the error is precise:

```javascript
const ALLOWED_CURRENCIES = new Set(['GHS', 'NGN', 'KES', 'UGX', 'XOF', 'ZAR']);

function extractCurrency(req) {
  for (const path of currencyPriority) {
    const value = getNested(req, path);
    if (typeof value !== 'string') continue;
    const normalized = value.toUpperCase();
    if (ALLOWED_CURRENCIES.has(normalized)) {
      return normalized;
    }
  }
  throw new Error('No valid currency found in request');
}
```

The latency difference is the point: rejection now happens in middleware, before any pricing or rendering work. The exact saving depends on where the old validation sat, but the failure moves from "after the expensive path" to "before it."

One caution: if the allow-list is shared between services, keep it in one place. A stale copy in a downstream service reintroduces the same bug.

## Fix 3: normalize casing per provider and environment

Some gateways are case-sensitive about the currency code, and the expected case can differ between sandbox and production for the same provider. A request that passes in one environment fails in the other with an error like:

```json
{
  "error": "Invalid currency code. Expected lowercase 3-letter code."
}
```

Handle this with a provider-aware normalizer rather than a country-aware one. Country is the wrong axis because the same provider is used across environments and, increasingly, across countries.

```python
# currency_provider_adapter.py
from enum import Enum

class Provider(Enum):
    MTN_MOMO_SANDBOX = "mtn_momo_sandbox"
    MTN_MOMO_PROD = "mtn_momo_prod"
    FLUTTERWAVE = "flutterwave"
    ORANGE_MONEY = "orange_money"

CURRENCY_CASE_RULES = {
    Provider.MTN_MOMO_SANDBOX: "lower",
    Provider.MTN_MOMO_PROD: "upper",
    Provider.FLUTTERWAVE: "upper",
    Provider.ORANGE_MONEY: "upper",
}

def normalize_currency(currency: str, provider: Provider) -> str:
    rule = CURRENCY_CASE_RULES.get(provider, "upper")
    return currency.lower() if rule == "lower" else currency.upper()
```

A detector supplies the provider:

```python
# provider_detector.py
def detect_provider(req) -> Provider:
    host = req.headers.get("host", "")
    if "mtn-momo-sandbox" in host:
        return Provider.MTN_MOMO_SANDBOX
    if "flutterwave" in host:
        return Provider.FLUTTERWAVE
    if "orange-money" in host:
        return Provider.ORANGE_MONEY
    return Provider.MTN_MOMO_PROD  # default
```

Host-based detection is fragile if traffic is proxied. Prefer an explicit provider identifier set by your gateway routing layer over string matching on the host.

The failure mode when this step is skipped is predictable: someone hard-codes casing by country, then a new environment for the same provider breaks the assumption.

## The two-layer pipeline

Putting the fixes together, every request should pass through:

1. **Extraction** — read from the ordered candidate list, normalize to uppercase, reject anything not matching `^[A-Z]{3}$`.
2. **Validation** — check membership in a single shared allow-list; fail fast with a clear error.
3. **Normalization** — convert to the casing the target provider expects, based on provider and environment, immediately before the outbound call.

Extraction and validation belong in middleware. Normalization belongs in the outbound adapter. Keeping them separate means the internal representation of a currency is always canonical uppercase ISO 4217, and only the wire format varies.

## How to verify the fix

A passing unit test on the extractor is necessary but not sufficient. What matters is that real-shaped traffic from each provider produces the right currency end to end. Two things to instrument:

- The resolved currency on every request, tagged by provider.
- The count of requests that hit the fallback path or threw, tagged by provider.

Both should be exported as counters. A non-zero fallback counter after a deploy is the signal that a provider is sending the currency somewhere the extractor does not look.

For replay, a load-testing tool that can issue requests with custom headers, query strings, and bodies works. A minimal k6 script:

```javascript
// test_currency_switch.js
import http from 'k6/http';
import { check } from 'k6';

const providers = [
  { name: 'M-Pesa', host: 'api.example.com', currency: 'KES', header: 'X-Currency-Code: KES', body: { currencyCode: 'KES' } },
  { name: 'Flutterwave', host: 'api.example.com', currency: 'NGN', query: 'currency=NGN' },
  { name: 'MTN MoMo', host: 'api.example.com', currency: 'GHS', header: 'X-Currency: GHS' },
  { name: 'Orange Money', host: 'api.example.com', currency: 'XOF', body: { currency_code: 'XOF' } }
];

export default function () {
  providers.forEach(provider => {
    const url = `https://${provider.host}/v1/price`;
    const params = {
      headers: provider.header ? { 'X-Currency': provider.currency } : {},
      qs: provider.query ? { currency: provider.currency } : {},
      body: provider.body ? JSON.stringify(provider.body) : null,
      tags: { provider: provider.name }
    };

    const res = http.get(url, params);

    check(res, {
      [`${provider.name} returns correct currency`]: (r) =>
        r.json().currency === provider.currency,
      [`${provider.name} latency < 200ms`]: (r) =>
        r.timings.duration < 200
    });
  });
}
```

Run it against a staging environment that mirrors production routing:

```bash
k6 run --vus 20 --duration 60s test_currency_switch.js
```

Interpret the results by provider tag, not by aggregate. An aggregate pass rate of 99% can hide a single provider failing 100% of the time if it is a small share of traffic. The 200ms threshold is illustrative; set it from your own baseline rather than copying it.

For the fallback counter, the check to run is: after replaying one request per provider, does any provider increment the fallback counter? If yes, the extractor is missing a location for that provider.

## Preventing recurrence

The extraction logic should live in one versioned package that every service imports, not in copy-pasted middleware. The package should export a single entry point and own the candidate list, the allow-list, and the provider casing rules.

```javascript
// index.js
const { extractAndNormalizeCurrency } = require('@fintech/currency-extractor');

module.exports = (req, res, next) => {
  try {
    req.currency = extractAndNormalizeCurrency(req);
    next();
  } catch (err) {
    next(err);
  }
};
```

Note the absence of a fallback currency in the catch block. Defaulting to a currency on an extraction failure is how the original bug reached production; failing the request is the correct behavior.

Pin the version explicitly:

```json
{
  "dependencies": {
    "@fintech/currency-extractor": "2.1.0"
  }
}
```

Run the provider replay test on every commit that touches the package:

```yaml
# .github/workflows/test_currency.yml
name: Test currency extraction
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
      - run: npx k6 run tests/test_currency_switch.js
```

The value of centralizing is not the code itself — it is that the candidate list and allow-list have exactly one definition. When a new provider is onboarded, one array entry and one casing rule cover every service.

## Adjacent failures worth checking

These are related but distinct problems that surface in the same systems.

**Webhook signature mismatch.** The webhook handler extracts the currency from the body while the signature was generated over the header. The HMAC fails even though the payload is genuine. Fix: normalize the currency to canonical form before signature verification, and verify against the raw bytes the sender signed.

**Index selectivity collapse.** After enabling multi-currency, a `prices` table indexed on `(product_id, currency)` can become unattractive to the planner if the same product has many currency rows. A partial index for the currencies that dominate query traffic can help:

```sql
CREATE INDEX idx_prices_product_currency ON prices (product_id, currency)
WHERE currency IN ('NGN', 'GHS', 'KES');
```

Measure before adding this. Check the query plan with `EXPLAIN ANALYZE` on the actual workload; a partial index that does not match the query predicate will not be used.

**UI/API currency divergence.** The UI reads the currency from `navigator.language` instead of the API response, so it renders one currency while the API charges in another. Fix: make the API response the single source of truth, and if the currency is communicated via a header, expose it to the browser:

```http
Access-Control-Expose-Headers: X-Currency
```

**Rate limiting by the wrong key.** A rate limiter keyed on client IP misbehaves when traffic is routed through shared CDN edges. Keying on the currency header is one option, but it changes the semantics of the limit — you are now limiting per currency rather than per client. Decide deliberately which axis you want to throttle on.

## Escalation when the fixes do not resolve it

If extraction, validation, and normalization are all correct and one country still fails, the likely cause is a gateway-side discrepancy between sandbox and production. The path forward:

1. Check the gateway's API changelog for the period since your last successful integration test. Sandbox APIs change without always being backward compatible.
2. Replay the exact failing request with the gateway's own curl examples against both sandbox and production. Compare header casing and body field names byte for byte.
3. Open a support ticket including the sanitized payload, the exact error, the reproducing curl command, and the environment.
4. If the failure is blocking, route that country's traffic to a fallback gateway behind a feature flag. Keep the flag scoped to the country and provider so it can be reverted without touching other traffic.

Document the fallback so on-call engineers can disable the primary gateway quickly. The value of the runbook is the time-to-revert, not the elegance of the flag.

## FAQ

**Why does the backend default to one currency when the user is elsewhere?**
Because the middleware that sets the currency has a hard-coded default and only checks one or two locations. If the user's provider sends the code in a header the middleware does not read, the default wins silently.

**How should sandbox versus production differences be handled?**
With a provider-and-environment adapter that normalizes casing at the outbound boundary. Do not branch on country, because the same provider spans environments and countries.

**What is the fastest way to catch a currency regression before merge?**
Replay one request per provider in CI and assert on the returned currency, plus a counter that increments whenever the fallback path is taken. The counter is what catches a new provider sending the code somewhere unexpected.

**Is a country-specific fork ever justified?**
Rarely. A shared extraction and normalization package with per-provider rules covers the same ground without duplicating review load or blocking cross-border features.

## One thing to do in the next 30 minutes

Open the middleware that sets the request currency and check whether it has a hard-coded default. If it does, replace the default with a thrown error and add the header candidates to the priority list. Then send a request that carries the currency only in a header:

```bash
curl -H "X-Currency: GHS" http://localhost:3000/v1/price
```

If the response contains `"currency":"GHS"`, the header path works. If it fails loudly instead of returning a price, the default is gone and the next missing source will surface as an error rather than a wrong charge.
