# Defending APIs Against Amplification, Stuffing and Smuggling

Most API security guidance assumes a clean environment and a patient timeline. Production gives you neither. The patterns that cause the most damage in real deployments are rarely exotic exploits; they are cheap requests that become expensive work on the backend. This article covers three such patterns, why perimeter rules alone miss them, and how to build an API layer that stays cheap under adversarial load.

## The failure mode: cheap to send, expensive to process

Classic API hardening focuses on injection, broken authentication and data exposure. Those still matter. But a large share of operational pain comes from attacks that never try to break encryption or forge a token. They only need to make your system do more work than the request cost the attacker.

Three patterns dominate this category:

- **Cache-stampede amplification.** An attacker requests keys that are not in cache, or requests the same uncached key from many sources at once. Each miss triggers backend compute and database reads. On endpoints with heavy enrichment logic, one client can fan out into hundreds of concurrent backend operations.
- **Credential stuffing against APIs.** Leaked credential lists are replayed against login and token endpoints at scale. APIs are attractive targets because they lack the browser-side friction (CAPTCHAs, device fingerprinting, JS challenges) that web login pages often have.
- **Request smuggling via HTTP/2 pseudo-headers.** HTTP/2 header compression and the `:method`, `:path`, `:scheme` and `:authority` pseudo-headers create room for requests that some WAF rule sets do not normalise correctly, letting crafted requests reach internal routes.

None of these necessarily trips a default WAF ruleset or a per-IP rate limit. A typical failure mode is a stack that is well defended against malformed input but has no defence against well-formed input sent at volume.

## Why adding infrastructure first usually fails

The intuitive response is to add a WAF with a broad managed ruleset, a per-IP rate limit, and a CDN cache in front of the API. Each of these helps, and each has a characteristic failure mode when used alone.

**Cache stampedes get worse, not better.** Consider an endpoint like `/products?id=...` that reads from a database, performs a short enrichment step, and writes the result to a cache with a short TTL. If many clients request the same uncached key within the same second, every one of them misses. The cache does not protect the backend during the miss window; it concentrates the load into it. Retrying mobile clients on unreliable networks make this worse, because retries arrive in bursts.

**WAF false positives hit legitimate users on shared infrastructure.** Managed rulesets that flag header manipulation will also flag carrier-grade NAT ranges, corporate proxies and browser retry behaviour. The result is legitimate 429s and 403s concentrated in exactly the regions where connectivity is already poor. Tuning rules to reduce false positives tends to reduce true positives at the same time.

**Cost scales with rejected traffic, not just accepted traffic.** Every request that reaches the WAF, the gateway and a compute function has a cost, even if it is ultimately rejected. If the rejection happens late in the path, the attacker's amplification is also your cost amplification.

A useful rule of thumb: if a defence does not make the *rejection* cheap, it does not solve the amplification problem.

## Three principles that actually reduce blast radius

Rather than blocking attacks at the perimeter, make the API itself resilient to being called badly. Three principles cover most of the ground.

1. **Request de-amplification.** Prevent one request from turning into many backend operations. Deduplicate, coalesce and cache aggressively.
2. **Stateless admission control.** Decide whether a request is allowed before it reaches compute-heavy code, using information the edge already has.
3. **Circuit breakers on upstream calls.** Prevent one slow dependency from cascading into a full outage.

A workable layering is:

- **Edge admission** — run pre-authentication and schema checks in a CDN compute layer (for example, CloudFront Functions or Lambda@Edge) before the request reaches the API gateway.
- **Request normalisation** — in the gateway or a thin middleware, deduplicate identical payloads and enforce strict header and body validation.
- **Backend resilience** — wrap upstream calls in circuit breakers, bounded retries and trace sampling.

The rest of this article walks through each layer with working code and the trade-offs that matter.

## Layer 1: Edge admission

The goal of edge admission is to reject obviously invalid requests before they consume gateway or compute capacity. A minimal implementation checks three things: a well-formed authorization header, an allowlisted `Content-Type`, and a body that matches a strict schema.

The example below uses a schema validation library for Node.js. Any equivalent library in the same category works; the pattern matters more than the specific package.

```javascript
// lambda-at-edge/admission.js
import { ZodError, z } from 'zod';

const OrderSchema = z.object({
  user_id: z.string().uuid(),
  items: z.array(
    z.object({
      product_id: z.string().uuid(),
      quantity: z.number().int().positive(),
    })
  ),
  idempotency_key: z.string().uuid(),
});

export const handler = async (event) => {
  const request = event.Records[0].cf.request;

  // 1. Authorization header shape check.
  // Note: this validates shape only. Signature verification belongs
  // where you have the key material and clock, not at the edge.
  const auth = request.headers['authorization']?.[0]?.value;
  if (!auth || !auth.startsWith('Bearer ')) {
    return {
      status: '401',
      statusDescription: 'Unauthorized',
      body: 'Missing or invalid Authorization header',
    };
  }

  // 2. Content-Type allowlist.
  const contentType = request.headers['content-type']?.[0]?.value;
  if (contentType !== 'application/json') {
    return {
      status: '400',
      statusDescription: 'Bad Request',
      body: 'Content-Type must be application/json',
    };
  }

  // 3. Body validation. Reject before the request reaches the gateway.
  let parsed;
  try {
    parsed = JSON.parse(request.body);
  } catch {
    return {
      status: '400',
      statusDescription: 'Bad Request',
      body: 'Malformed JSON',
    };
  }

  try {
    OrderSchema.parse(parsed);
  } catch (error) {
    if (error instanceof ZodError) {
      return {
        status: '400',
        statusDescription: 'Bad Request',
        body: JSON.stringify({ error: 'schema_validation_failed' }),
      };
    }
    return {
      status: '500',
      statusDescription: 'Internal Server Error',
      body: 'validation_error',
    };
  }

  return request;
};
```

Three implementation notes that are easy to get wrong:

- **Do not attempt full JWT verification at the edge unless you have a stable key source.** Key rotation and clock skew are easier to handle in one place. Shape validation at the edge plus signature verification at the gateway is usually the right split.
- **Return generic error bodies.** Detailed validation errors are useful to attackers probing your schema. Log the detail; return a code.
- **Measure the added latency, do not assume it.** Instrument the edge function with a histogram of its own execution time, and compare p50 and p99 against the same route with the function disabled. A few milliseconds at the edge is often cheaper than the compute it prevents, but the only way to know for your workload is to measure both sides.

## Layer 2: Request normalisation and idempotency

The second layer sits between the edge and your business logic. It does two things: it rejects requests that carry HTTP/2 pseudo-headers, and it deduplicates identical mutations using an idempotency key.

Pseudo-headers (`:method`, `:path`, `:scheme`, `:authority`) are part of the HTTP/2 wire format and are consumed by the protocol implementation. They should never appear as ordinary request headers. If your gateway or middleware sees them in the header collection it passes to your application, that is a sign of misconfiguration or smuggling, and the request should be rejected rather than forwarded.

Idempotency keys are the higher-leverage change. Make them required on every mutation endpoint, not optional. A replay of an old payload then becomes a cheap cache hit instead of a second write.

```python
# api-gateway/middleware.py
import os
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError

app = FastAPI()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis.internal"),
    port=6379,
    decode_responses=True,
    socket_timeout=5,
)

class OrderRequest(BaseModel):
    user_id: str
    items: list[dict]
    idempotency_key: str

@app.post("/orders")
async def create_order(request: Request):
    try:
        payload = await request.json()
        order = OrderRequest(**payload)
    except (ValueError, ValidationError):
        raise HTTPException(status_code=400, detail="invalid_payload")

    # SET NX EX is atomic: only the first caller wins.
    # If the key already exists, this is a replay.
    acquired = await redis_client.set(
        f"idem:{order.idempotency_key}",
        "1",
        nx=True,
        ex=5,
    )
    if not acquired:
        return {"status": "duplicate", "idempotency_key": order.idempotency_key}

    # Business logic here. On failure, delete the key so the client
    # can legitimately retry.
    return {"status": "accepted", "idempotency_key": order.idempotency_key}
```

The important detail is `SET ... NX EX`, which is atomic. A naive `EXISTS` followed by `SET` has a race window in which two concurrent requests both observe the key as absent and both proceed — exactly the behaviour idempotency keys are supposed to prevent.

**Choosing the TTL.** The TTL should be longer than the longest realistic client retry window and shorter than the period over which a client might legitimately want to repeat a logically distinct request. Five seconds covers immediate retries from a flaky connection. If your clients retry with exponential backoff over minutes, a TTL of 5 seconds is too short; measure your client retry distribution and set the TTL above the 99th percentile.

**Failure mode: Redis unavailable.** Two options: fail open (process the request, log the miss) or fail closed (reject). Failing open risks duplicate writes; failing closed risks an outage that is worse than duplicates. For most order or payment flows, failing open with a loud alert is the safer default, because the duplicate is recoverable and the outage is not.

**Memory.** A UUID string is 36 bytes. At 1,000,000 keys per day with a 5-second TTL, the steady-state key count is bounded by `1,000,000 / 86,400 * 5 ≈ 58` keys, not 1,000,000. The figure that matters is the *concurrent* key count, which is tiny. Compute it as `requests_per_second * ttl_seconds` and add overhead for the Redis key prefix and object metadata.

## Layer 3: Backend resilience

The third layer protects against slow dependencies. The pattern is a circuit breaker around each upstream call, with bounded retries and a retry budget.

```python
# lambdas/order_service.py
import asyncio
from pybreaker import CircuitBreaker

order_breaker = CircuitBreaker(fail_max=5, reset_timeout=30)

@order_breaker
async def get_product(ddb, product_id: str):
    response = await ddb.get_item(
        TableName="products",
        Key={"id": {"S": product_id}},
    )
    return response.get("Item")

async def create_order(ddb, order):
    try:
        product = await get_product(ddb, order.items[0]["product_id"])
    except Exception:
        if order_breaker.current_state == "open":
            raise RuntimeError("order_service_unavailable")
        raise
    # ... rest of the order flow
```

Notes on the pattern:

- **`fail_max=5` and `reset_timeout=30` are starting points, not defaults.** The right values depend on your dependency's failure distribution. Instrument both the open and half-open transitions; a breaker that never opens is not protecting anything, and one that flaps is causing its own outage.
- **Bound retries explicitly.** Three attempts with exponential backoff capped at a couple of seconds is a common shape. Unbounded retries under load are a self-inflicted amplification attack.
- **Sample traces deliberately.** Trace every error and a low percentage of successes. Tracing every request during an attack is itself expensive and produces data you will not read.

## How to measure whether this is working

Any claim about latency or cost reduction depends entirely on your workload. The honest approach is to instrument the specific counters that would show the effect, and compare before and after on the same route.

Instrument these:

- **Edge rejection rate**, split by reason (auth shape, content type, schema, pseudo-header). A sudden shift in the mix is an early warning of a new attack pattern.
- **Cache miss fan-out**, defined as the number of backend operations triggered per cache miss on a given key. This is the metric that reveals cache stampedes.
- **Duplicate request ratio**, defined as the fraction of mutation requests that hit an existing idempotency key. Before enabling idempotency, this ratio is invisible; after enabling it, it tells you how much replay traffic you had.
- **Circuit breaker state transitions** and the time spent open.
- **Cost per thousand requests**, computed from your provider's billing dimensions, not from an aggregate monthly bill.

Then run the comparison:

```bash
# Example: compare p95 latency for one route before and after enabling
# edge admission. Run the same load profile against both revisions.
hey -n 20000 -c 200 -m POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"user_id":"...","items":[],"idempotency_key":"..."}' \
  https://api.example.com/orders
```

Any percentage improvement you report should be derived from these measurements on your own traffic. Numbers from someone else's deployment do not transfer, because the ratio of attack traffic to legitimate traffic, the cache hit rate and the cost model all differ.

## A decision checklist before you build this

Not every API needs all three layers. Use the following to decide where to start.

| Signal | What it suggests |
|---|---|
| Duplicate mutations appear in logs | Add idempotency keys first; this is the cheapest high-impact change |
| p99 latency spikes correlate with cache misses | Investigate cache stampede; consider request coalescing and longer TTLs on hot keys |
| Login endpoints see high request volume from many IPs | Credential stuffing; per-IP rate limits will not help, per-account lockout and proof-of-work will |
| Header anomalies appear in gateway logs | Enable strict pseudo-header handling and reject malformed requests at the gateway |
| One upstream dependency causes cascading failures | Add a circuit breaker before adding more capacity |

If none of these signals appear in your telemetry, the first action is not to add a layer — it is to add the instrumentation that would reveal them.

## FAQ

**Why not rely on WAF rate limiting instead of idempotency keys?**
Per-IP rate limiting shifts a distributed attack to other IPs rather than stopping it. Idempotency keys operate per logical request, so a replay is rejected regardless of source. The two are complementary: rate limiting reduces volume, idempotency reduces the cost of each accepted request.

**What TTL should idempotency keys use?**
Longer than the 99th percentile of your client retry interval, and short enough that a deliberate repeat request is not mistaken for a retry. Measure your clients; do not copy a number.

**Should the middleware fail open or closed if the deduplication store is unavailable?**
For most write paths, fail open with an alert. A duplicate write is usually recoverable; a total write outage is not. The exception is flows where a duplicate has financial or safety consequences, where fail closed is the correct trade-off.

**Is edge admission worth it for a low-traffic API?**
The value is proportional to how expensive a rejected request is on your backend. If rejection is already cheap, edge admission adds complexity for little gain. Start with schema validation at the gateway and add edge admission only if measurements show rejected requests are consuming meaningful compute.

**Does schema validation at the edge replace validation in the service?**
No. Treat edge validation as a load-shedding filter, not a security boundary. The service must still validate, because the edge can be bypassed by internal callers, direct gateway access or misconfiguration.

## One action for the next 30 minutes

Pick a single mutation endpoint and send it a request with no idempotency key:

```bash
curl -i -X POST https://api.example.com/orders \
  -H "Content-Type: application/json" \
  -d '{"user_id":"123e4567-e89b-12d3-a456-426614174000","items":[]}'
```

If the response is a success rather than a 400, that endpoint will accept replays. Add a required idempotency key to it, store the key with an atomic `SET ... NX EX`, and log the duplicate ratio for a week. That single measurement will tell you whether the rest of this article is worth building.
