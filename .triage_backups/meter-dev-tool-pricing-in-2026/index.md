# Meter dev-tool pricing in 2026

Flat-rate pricing for a developer tool is easy to set up and hard to keep honest: one heavy CI user can consume more support time and compute than a hundred casual ones, while both pay the same. Usage metering fixes the alignment problem, but it moves complexity into the billing path, where bugs are expensive. This article walks through a working metered-billing service — Stripe Checkout and subscriptions, Redis counters, idempotent usage events — and then through the edge cases that break metered systems in practice.

## Prerequisites and what you'll build

You need two things:

- A CLI or web service you intend to monetize. The examples use Python 3.11 and FastAPI; the same design works for Node.js, Go, or a VS Code extension that reports events to a backend.
- A Stripe account with test mode enabled.

You will build a **usage-metered pricing service** that:

1. Defines three tiers (free, usage, enterprise).
2. Records API calls or CLI invocations per customer.
3. Creates a Stripe subscription and handles proration on plan changes.
4. Exposes a usage endpoint so a client can show a live meter.

A CLI is a useful reference because its events arrive from many machines, over unreliable networks, often from CI jobs that retry. That is exactly the environment where naive metering double-counts.

## Step 1 — set up the environment

Create a directory and install dependencies:

```bash
mkdir devtool-pricing && cd devtool-pricing
python -m venv .venv && source .venv/bin/activate  # or `.\.venv\Scripts\activate` on Windows
pip install fastapi uvicorn stripe redis python-dotenv
echo "FASTAPI_ENV=dev" > .env
```

FastAPI is a reasonable default here because it gives async endpoints and generated OpenAPI docs without extra wiring. Pin your exact versions in `requirements.txt` once you have a working set — the Stripe Python SDK's API surface changes across major versions, so treat any upgrade as a migration.

Run Redis locally for counters and rate limiting:

```bash
docker run -d --name redis-dev -p 6379:6379 redis:7.2-alpine redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

Redis is used here for atomic increments (`INCR`) and cheap TTLs. Note the `allkeys-lru` policy above: it is fine for a development counter store, but it means keys can be evicted under memory pressure. For production billing counters you want a policy that never evicts metering data — persistence and eviction behavior are billing correctness concerns, not just caching concerns.

Create `main.py` with the scaffolding:

```python
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import stripe
import redis.asyncio as redis
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
redis_client = redis.from_url("redis://localhost:6379")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Run it with hot reload:

```bash
uvicorn main:app --reload --port 8000
```

Without `--reload`, the server still starts but code changes are not picked up until you restart it. This is a common source of "my fix didn't work" confusion during local development.

## Step 2 — record usage and check the subscription

The core is a `POST /usage` endpoint that increments a counter and reports the current count against the customer's plan limit. Add it:

```python
@app.post("/usage")
async def log_usage(
    stripe_customer_id: str = Header(...),
    tool_event: str = "api_call",
):
    key = f"usage:{stripe_customer_id}:{tool_event}"
    count = await redis_client.incr(key)

    subs = await stripe.Subscription.search_async(
        query=f"customer:'{stripe_customer_id}' AND status:'active'"
    )
    if not subs.data:
        raise HTTPException(status_code=402, detail="No active subscription")

    subscription = subs.data[0]
    usage_limit = subscription.metadata.get("usage_limit", "1000")

    return JSONResponse({
        "usage": count,
        "limit": int(usage_limit),
        "percentage": min(100, (count / int(usage_limit)) * 100),
    })
```

Design points worth keeping:

- Usage is stored per customer **and** per event type, so a CLI can report `cli_run` while an API reports `api_call` without the counters colliding.
- The subscription lookup happens on every request. That is a correctness choice, not a performance one: it means a canceled subscription stops being served immediately. If the lookup becomes a latency problem, cache it with a short TTL and accept a small window of stale authorization — but decide that deliberately.
- The limit is read from subscription metadata rather than hardcoded, so a plan change does not require a deploy.

The Redis `INCR` is atomic, but the sequence "increment, then check subscription" is not transactional. A request that arrives between cancellation and the next check will still be counted. For most dev tools this is acceptable; for strict entitlements, check entitlement first and increment second, and accept that you may undercount instead of overcount.

## Step 3 — make usage events idempotent

This is the failure mode that matters most. A CI job that retries after a network timeout will re-send the same event, and a naive `INCR` double-bills the customer. The fix is to make the event carry a client-generated idempotency key and to reject replays:

```python
import hashlib

@app.post("/usage")
async def log_usage(
    stripe_customer_id: str = Header(...),
    tool_event: str = "api_call",
    idempotency_key: str = Header(None),
):
    if idempotency_key:
        hashed = hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]
        # SET NX returns True only if the key did not exist.
        first_seen = await redis_client.set(
            f"idemp:{hashed}", "1", nx=True, ex=86400
        )
        if not first_seen:
            raise HTTPException(status_code=409, detail="Duplicate request")

    key = f"usage:{stripe_customer_id}:{tool_event}"
    count = await redis_client.incr(key)
    # ... subscription check and response as above
```

Two details matter. First, the key is namespaced by a hash so that arbitrary client input cannot collide with or overwrite other Redis keys. Second, the TTL must be longer than the client's maximum retry window. If a CI job retries for an hour, a 24-hour TTL is safe; if it can retry for two days, it is not.

A second edge case is **plan downgrades mid-cycle**. A customer on the usage tier who downgrades to free should keep access until the end of the paid period. Stripe handles this if the subscription update is sent with `proration_behavior="create_prorations"`; the customer is credited for unused time and the downgrade takes effect at period end. Do not implement proration yourself — the arithmetic around partial periods and timezones is a reliable source of disputes.

A third is **zero-dollar subscriptions**. A customer who cancels a paid plan may still be entitled to the free tier. If the entitlement check treats "no active paid subscription" as "no access," those users are blocked incorrectly. Check the plan the customer is entitled to, not just whether a paid subscription exists.

A minimal test to pin the behavior:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_usage_increments():
    resp = client.post(
        "/usage",
        headers={"stripe-customer-id": "cus_test", "idempotency-key": "a1b2c3"},
    )
    body = resp.json()
    assert body["usage"] == 1
    assert body["limit"] == 500

def test_duplicate_is_rejected():
    headers = {"stripe-customer-id": "cus_test", "idempotency-key": "dup-1"}
    client.post("/usage", headers=headers)
    second = client.post("/usage", headers=headers)
    assert second.status_code == 409
```

Run with:

```bash
pip install pytest pytest-asyncio
export STRIPE_SECRET_KEY=sk_test_...
pytest -v
```

## Step 4 — observability and load testing

Metering systems fail quietly. A counter that stops incrementing produces a smaller invoice, not an error page, so the signals you need are different from a typical web service.

Instrument these:

1. **Usage events per customer per event type.** A counter labeled by customer and event type lets you see a customer whose events stopped arriving — usually a broken client, occasionally a billing bug.
2. **Duplicate-rejection rate.** A spike in 409s means clients are retrying, which means something upstream is failing. It is an early warning, not just a metric.
3. **Redis latency and error rate.** Every metering request depends on Redis. If it degrades, billing degrades.

```python
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

USAGE_COUNTER = Counter(
    "devtool_usage_total", "Total usage events", ["customer", "event"]
)
DUPLICATE_COUNTER = Counter(
    "devtool_usage_duplicate_total", "Rejected duplicate events", ["customer"]
)

@app.get("/metrics")
async def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

A retry wrapper around Redis calls is worth adding, but be careful about what you retry. Retrying an `INCR` that may have succeeded is exactly the double-counting problem you solved with idempotency keys, so the retry must be on the read path or be itself idempotent.

To measure the system rather than guess at it, run a load test and compare two numbers: the number of requests your client sent and the counter value your service recorded. They should match exactly once duplicates are accounted for. A simple k6 script:

```javascript
import http from 'k6/http';
export const options = { vus: 50, duration: '5m' };
export default function () {
  http.post('http://localhost:8000/usage', JSON.stringify({}), {
    headers: {
      'stripe-customer-id': 'cus_test',
      'idempotency-key': `k6-${__VU}-${__ITER}`,
    },
  });
}
```

Use a unique idempotency key per iteration, as above, so the test measures throughput rather than the duplicate-rejection path. Then reconcile: `k6` reports total requests, and your Redis counter for `cus_test` should equal that number. Any gap is a bug in the metering path, and it is much cheaper to find it here than in a customer's invoice.

## Choosing what to meter

The unit you meter determines how customers perceive the price. A few options and their trade-offs:

| Metered unit | Fits | Watch out for |
|---|---|---|
| API calls | Hosted services, gateways | Retries inflate counts unless idempotent |
| CLI invocations | Local-first tools | Offline runs need local buffering and later sync |
| Files or lines processed | Linters, formatters, scanners | Large files make cost unpredictable per run |
| Seats plus usage | Team products | Two meters to explain; keep the split simple |
| Compute seconds | Build and CI tools | Requires accurate, tamper-resistant measurement |

The general rule: meter something the customer can predict before they run the command. "You will be billed per API call" is predictable. "You will be billed per unit of internal work" is not, and unpredictable bills generate support tickets regardless of whether the average is lower.

## Migrating from flat-rate pricing

Moving existing customers from flat-rate to metered pricing is where most of the risk lives, because existing customers did not opt into the new model.

A workable pattern is a **grandfathered allowance**: give each existing customer a usage allowance equal to what their current flat fee would buy under the new rates, and hold that allowance for a fixed period. For example, if the new rate is $0.002 per call and a customer pays $49/month, their grandfathered allowance is 24,500 calls per month:

```
$49 / $0.002 per call = 24,500 calls
```

Show the customer their usage against that allowance before the grandfathering ends, so the first metered invoice is not a surprise. The metric to watch during migration is not revenue — it is support tickets per customer, because a spike there means the new bill is not understood.

## Common questions

**What if my tool is CPU-bound rather than API-bound?**

Meter something the user controls: CLI invocations, files processed, or repository size. For a local tool, the client must buffer events offline and sync later, which makes idempotency keys mandatory rather than optional.

**Can I mix seat-based and usage-based pricing?**

Yes. A common split is a seat fee covering fixed costs (hosting, support) plus usage covering variable costs. Keep the explanation to one sentence per meter; two meters with unclear boundaries is worse than one imperfect meter.

**How do I handle tax and currency?**

Use your payment provider's tax calculation rather than implementing rates yourself. Tax rules change and vary by jurisdiction; this is a case where the boring, provider-managed path is the correct one.

**How do I know if my metering is correct?**

Reconcile continuously. For each customer, compare the counter your service holds against the sum of events your client reports sending. A persistent gap means lost or duplicated events. Run this reconciliation on a schedule, not only when a customer complains.

## What to do in the next 30 minutes

Pick one endpoint or CLI command in your tool and add a single counter that records how many times it is invoked per customer, with a client-supplied idempotency key. Do not add billing yet — just record and reconcile. Run your own client against it for a day, then compare the client's event count to the server's counter. If those two numbers match, you have the foundation for metered pricing; if they do not, you have found the bug that would otherwise have appeared on a customer's invoice.
