# AI payments break in M-Pesa first

AI features that touch payments fail in predictable ways, and the failures usually have nothing to do with the model. The pattern repeats across providers: a webhook arrives twice, a retry policy changes without notice, a local cache drifts from the provider's ledger, and a discount or credit gets applied more than once. This article covers what comes after the happy path — the invariants a payment provider does not give you, and how to build them yourself.

## The gap between what the docs say and what production needs

Payment provider documentation describes the successful call. It shows the webhook payload shape, the charge endpoint, the signature header. What it typically does not specify is the failure envelope: how often a webhook is redelivered, whether retries preserve ordering, how idempotency keys are compared, and what happens to in-flight requests during a provider-side incident.

Teams that treat payment integration as glue code end up with race conditions in feature store writes, duplicated events in analytics, and silent drift when a provider changes a retry policy. This is not technical debt you pay down later; it is a gap between what the provider guarantees and what the user experiences.

A common trap: an AI feature suggests a discount when a payment fails, and the discount is applied twice because the webhook arrived twice. The provider's docs may never mention duplicate delivery. The background worker assumes idempotency. The result is an over-credited user and a support ticket that will not reproduce in staging.

The core problem is not model accuracy. It is that the payment layer does not hand you the invariants — exactly-once processing, ordered delivery, stable error semantics — that a reliable system needs. Until you internalize that, every AI feature you ship on top of it is fragile by design.

## What is actually happening under the hood

You are not calling one API. You are stitching together multiple consistency models into one user-facing flow. Mobile-money providers often confirm through an eventually consistent channel (an SMS or callback). Card processors commonly use an idempotent charge-and-refund model. Aggregators frequently sit on top of external PSPs with their own two-phase semantics. Your AI feature reads state from all of them, writes back changes, and must present one source of truth.

The first cut most teams make is to treat each provider as a stateless function: call the API, record the response, done. That works for a demo and fails the moment you need to retry, reconcile, or audit. The second cut is a local cache of provider state, which goes stale the moment a retry policy changes or a webhook arrives out of order.

The system that survives production treats the provider as an unreliable event stream, not a reliable RPC. Concretely, you need to:

- deduplicate events arriving at the webhook endpoint
- reconcile provider state against your local feature store
- handle idempotency failures without corrupting balances
- surface provider-specific errors in a vocabulary your AI layer can act on

A frequent failure mode is conflating the provider's transaction ID with your internal event ID. Transaction IDs differ in format and comparison semantics across providers — some are numeric, some alphanumeric and case-sensitive, some UUIDs. If you store them all as raw strings without normalization, deduplication misses duplicates and reconciliation drifts.

Another trap is assuming webhook order matches transaction order. In practice webhooks arrive out of order, duplicated, or delayed. The system must tolerate all three without corrupting a balance or issuing an over-refund.

Timeouts and rate limits also differ per provider. A sandbox may hang for tens of seconds on a declined card; another may reject a reused idempotency key after a small number of attempts; another may return HTTP 500 for specific test card numbers. If retry logic does not back off with jitter, you can saturate your own retry queue and trigger provider-side throttling.

Finally, the AI layer needs a consistent error vocabulary. If every provider error collapses to a generic "failed" label, the model will keep suggesting retries for irrecoverable declines. Normalizing errors is what lets it choose between retry, discount, and cancellation.

## Step-by-step implementation

The following is a minimal but production-oriented implementation that handles three providers in one codebase. It uses Python 3.11, FastAPI, Redis for deduplication, and SQLAlchemy 2.0 for the feature store. Pin versions to whatever your environment supports; the patterns matter more than the exact releases.

### 1. Normalize provider responses

Each provider returns transaction data in a different shape. Normalize into a common event format before writing to the feature store.

```python
# providers/schema.py
from pydantic import BaseModel, Field
from typing import Optional

class PaymentEvent(BaseModel):
    provider: str  # "mpesa", "paystack", "flutterwave"
    tx_id: str  # provider-specific transaction ID
    amount: int  # amount in smallest currency unit (cents/centavos/kobo)
    status: str  # "pending", "success", "failed", "reversed"
    timestamp: int  # Unix epoch in seconds
    raw: dict = Field(default_factory=dict)  # provider-specific extras

class MpesaEvent(PaymentEvent):
    provider: str = "mpesa"
    tx_id: str = Field(..., alias="TransactionId")
    status: str = Field(..., alias="ResultDesc")
    amount: int = Field(..., alias="TransAmount")
    timestamp: int = Field(..., alias="Timestamp")

class PaystackEvent(PaymentEvent):
    provider: str = "paystack"
    tx_id: str = Field(..., alias="transaction_id")
    status: str = Field(..., alias="status")
    amount: int = Field(..., alias="amount")
    timestamp: int = Field(..., alias="paid_at")

class FlutterwaveEvent(PaymentEvent):
    provider: str = "flutterwave"
    tx_id: str = Field(..., alias="tx_ref")
    status: str = Field(..., alias="status")
    amount: int = Field(..., alias="amount")
    timestamp: int = Field(..., alias="created_at")
```

### 2. Deduplicate webhooks with Redis

Use a Redis sorted set keyed by a hash of provider plus normalized transaction ID. The hash sidesteps case-sensitivity differences between providers. Trim old entries to bound memory.

```python
# services/dedup.py
import hashlib
import redis.asyncio as redis
from providers.schema import PaymentEvent

def normalize_tx_id(provider: str, tx_id: str) -> str:
    # Normalize case for providers whose IDs are effectively case-insensitive.
    # Keep the raw value for providers where case is significant.
    if provider in ("paystack",):
        return tx_id.lower()
    return tx_id

async def dedupe_event(event: PaymentEvent, redis_client: redis.Redis) -> bool:
    normalized = normalize_tx_id(event.provider, event.tx_id)
    key = f"dedupe:{event.provider}:{normalized}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    inserted = await redis_client.zadd(
        "dedupe_set",
        {digest: event.timestamp},
        nx=True,
    )
    # Only keep events from the last 24h to bound memory usage
    await redis_client.zremrangebyscore(
        "dedupe_set",
        0,
        event.timestamp - 86400,
    )
    return bool(inserted)
```

### 3. Reconcile provider state with the local feature store

The feature store tracks derived state such as discount eligibility. Do not overwrite the user's balance from webhook data alone; update derived state and reconcile balances through a separate, auditable path.

```python
# models/feature_store.py
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class UserBalance(Base):
    __tablename__ = "user_balances"
    user_id = Column(String(36), primary_key=True)
    balance = Column(Integer, default=0)  # in smallest unit
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

class DiscountEligibility(Base):
    __tablename__ = "discount_eligibility"
    user_id = Column(String(36), primary_key=True)
    eligible = Column(Integer, default=0)  # 0 or 1
    last_evaluated = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

```python
# services/reconcile.py
from sqlalchemy import update, func
from sqlalchemy.ext.asyncio import AsyncSession
from models.feature_store import DiscountEligibility
from providers.schema import PaymentEvent

async def reconcile_user(user_id: str, event: PaymentEvent, session: AsyncSession):
    # Only update derived state, never the user balance.
    if event.status == "success":
        stmt = (
            update(DiscountEligibility)
            .where(DiscountEligibility.user_id == user_id)
            .values(eligible=1, last_evaluated=func.now())
        )
        await session.execute(stmt)
    elif event.status in ("failed", "reversed"):
        stmt = (
            update(DiscountEligibility)
            .where(DiscountEligibility.user_id == user_id)
            .values(eligible=0, last_evaluated=func.now())
        )
        await session.execute(stmt)
```

### 4. Handle provider-specific retries and backoff

Providers differ in rate limits and timeout behavior. Use exponential backoff with jitter and cap retries at a value you can justify from the provider's documented limit. The numbers below are illustrative starting points — measure your own provider behavior and adjust.

```python
# services/retry.py
import asyncio
import random
from typing import Callable, Any

# Illustrative caps. Derive these from your provider's documented rate
# limits and observed behavior, not from a blog post.
PROVIDER_RETRIES = {
    "mpesa": 3,
    "paystack": 5,
    "flutterwave": 4,
}

PROVIDER_BACKOFF = {
    "mpesa": [1, 2, 4],
    "paystack": [1, 2, 4, 8, 16],
    "flutterwave": [1, 2, 4, 8],
}

async def with_retry(provider: str, fn: Callable[..., Any], *args, **kwargs) -> Any:
    attempts = PROVIDER_RETRIES[provider]
    backoff = PROVIDER_BACKOFF[provider]
    for attempt in range(attempts):
        try:
            return await fn(*args, **kwargs)
        except Exception:
            if attempt == attempts - 1:
                raise
            delay = backoff[attempt] + random.uniform(0, 0.5)
            await asyncio.sleep(delay)
```

### 5. Map provider errors to AI-compatible labels

The AI layer needs to know why a payment failed to decide between retry, discount, and cancellation. Normalize errors into a small vocabulary.

```python
# providers/errors.py
from typing import Dict, Optional

ERROR_MAPPING: Dict[str, Dict[str, str]] = {
    "mpesa": {
        "insufficient funds": "insufficient_funds",
        "invalid amount": "invalid_amount",
        "user cancelled": "user_cancelled",
        "timeout": "timeout",
    },
    "paystack": {
        "card_declined": "card_declined",
        "insufficient_funds": "insufficient_funds",
        "invalid_cvc": "invalid_cvc",
        "expired_card": "expired_card",
    },
    "flutterwave": {
        "failed": "generic_failure",
        "cancelled": "user_cancelled",
        "timeout": "timeout",
    },
}

def normalize_error(provider: str, raw_error: str) -> Optional[str]:
    mapping = ERROR_MAPPING.get(provider, {})
    lowered = raw_error.lower()
    for key, label in mapping.items():
        if key in lowered:
            return label
    return "generic_failure"
```

### 6. Put it together in a FastAPI endpoint

```python
# main.py
from fastapi import FastAPI
from providers.schema import PaymentEvent
from services.dedup import dedupe_event
from services.reconcile import reconcile_user
from providers.errors import normalize_error
import redis.asyncio as redis
import sqlalchemy.ext.asyncio as sa

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379, db=0)
async_engine = sa.create_async_engine("postgresql+asyncpg://user:pass@localhost/db")

@app.post("/webhook/{provider}")
async def webhook(provider: str, event: PaymentEvent):
    # 1. Deduplicate
    if not await dedupe_event(event, redis_client):
        return {"status": "duplicate"}

    # 2. Normalize error for the AI layer
    error_label = normalize_error(provider, event.raw.get("error", ""))

    # 3. Reconcile user state
    async with async_engine.begin() as session:
        await reconcile_user(event.user_id, event, session)

    # 4. Trigger AI feature using error_label
    # ... your AI logic here ...

    return {"status": "processed"}
```

Note that the endpoint above still needs a `user_id` on the event, which providers do not always supply directly. In practice you resolve it from a mapping table keyed by provider transaction reference, and you must handle the case where the mapping is not yet present (webhook arrives before your charge call returns). A robust handler either parks the event in a pending table or re-fetches the transaction from the provider before reconciling.

## How to measure the failure modes yourself

Any benchmark table copied from someone else's environment is worse than useless — your provider mix, traffic shape, and retry policy will differ. Measure these instead:

- **Duplicate webhook rate.** Log every inbound webhook with provider, normalized transaction ID, and receipt timestamp. Count distinct transaction IDs versus total deliveries per 24 hours. The ratio is your duplicate rate. Instrument this before you add deduplication so you have a baseline.
- **Out-of-order arrivals.** For each transaction, record the provider's event timestamp and your receipt timestamp. Sort by provider timestamp and count inversions in your receipt order. Inversions indicate reordering.
- **Idempotency key collisions.** Log every outbound charge with its idempotency key and the provider's returned transaction ID. Group by normalized key; any group with more than one transaction ID is a collision.
- **Reconciliation drift.** Run a job that pulls the provider's transaction list for the last 24 hours and diffs it against your feature store. Count rows present on one side only. This is your drift metric.
- **Retry amplification.** Count outbound requests per provider per minute and compare against your normal baseline. A spike without a matching spike in user-initiated charges indicates retry amplification.

For each metric, the command depends on your stack. With Redis-backed deduplication, `redis-cli ZCARD dedupe_set` gives the current window size; comparing that against your inbound webhook counter tells you how much deduplication is actually firing. With PostgreSQL, a query grouping events by `provider, tx_id` and filtering `HAVING COUNT(*) > 1` gives duplicate counts directly.

## Failure modes worth designing for

1. **Sandbox idempotency is not production idempotency.** Sandboxes frequently accept the same idempotency key twice and return different transaction IDs. Never treat sandbox behavior as evidence about production semantics.
2. **Webhook signature drift.** Some providers include a timestamp in the signature. If your server clock drifts beyond the tolerance, the signature check fails and the event is silently dropped. Validate the timestamp explicitly and log drift events so you notice.
3. **Test-card blacklists change.** Sandbox card blacklists are updated without notice. Automated test suites that rely on specific card numbers will fail intermittently. Cache the blacklist or use provider-documented test cards and expect churn.
4. **Undocumented rate limits.** A provider may document a limit but not the response headers that tell you when the limit resets. Implement a local limiter that respects the documented limit and treats HTTP 429 as a signal to back off, not to retry immediately.
5. **Drift after provider outages.** During a provider incident, events do not arrive. Your feature store drifts. Implement a backfill job that re-processes events from the provider's audit log or transaction list after an outage.
6. **Currency conversion drift in AI suggestions.** If the AI suggests a discount in one currency against a balance in another, a stale rate produces suggestions that are visibly wrong. Cache rates with a short TTL and serve stale rates only during outages, with a flag.
7. **Client-side retry duplication.** A user refreshing after a failed payment can trigger a second charge. Deduplicate on the client using the idempotency key, not just on the server.
8. **Idempotency key case sensitivity.** Some providers compare idempotency keys case-sensitively. If your client generates lowercase UUIDs and another code path generates mixed case, you get duplicate charges. Normalize the key before sending.

## When this approach is the wrong choice

This pattern — deduplication, reconciliation, error normalization — is overkill if you integrate with a single provider and your AI feature never writes back to a balance. If you call a charge endpoint and show a success message, a simple retry loop with exponential backoff is enough.

It is also the wrong choice if your P99 budget is very tight. Deduplication and reconciliation add latency, and provider-side variance often dominates. If you need sub-100ms end-to-end, use a serverless handler with a local cache and accept weaker reliability guarantees, or move reconciliation off the request path entirely.

Finally, it is the wrong choice if your team has no relational database experience. The reconciliation logic is the kind of code where a single bug corrupts balances. If your team is more comfortable with a document store, consider a simpler pattern that tracks only the latest provider event and treats the provider as the source of truth for balances.

## FAQ

**Why not use the provider's official SDK for retries?**
Official SDKs often do not expose the retry policy or backoff behavior, or they pick defaults that are aggressive for production. If the SDK retries at fixed short intervals without jitter, it can amplify load during an incident. Wrap the SDK call in your own retry function so you control backoff and jitter.

**How do I handle sandbox versus production differences without duplicating code?**
Switch endpoints and credentials via environment variables, but keep the same retry, deduplication, and reconciliation code. If you find yourself duplicating retry logic per environment, extract it into a shared function.

**What if the AI model needs the raw error message?**
Normalize the error for the model's input, but store the raw error in your event log. The model gets a consistent vocabulary; support and debugging keep the full context.

**How do I test duplicate webhook handling without spamming the provider?**
Use a local mock server or a provider sandbox that allows duplicate events. Send two identical events with the same transaction ID within the retry window and verify the second is dropped. For providers that support idempotency keys, send the same key twice and verify only one charge exists.

**What is the smallest viable system that still handles these failure modes?**
One PostgreSQL table for events, a unique constraint on `(provider, normalized_tx_id)`, and a cron job that reconciles every few minutes. Skip Redis; the unique constraint drops duplicates. This handles duplicate events and reconciliation drift with far less code.

**How do I handle currency conversion without a single external rate API?**
Use a local cache of rates from a source you trust, and prefer a central bank reference rate for the currency pair. Cache with a short TTL and serve stale rates only during outages, with a flag on the response so downstream logic knows.

## What to do next

In the next 30 minutes, add one measurement: log every inbound webhook with `provider`, normalized `tx_id`, provider timestamp, and receipt timestamp to a table or log sink. Do not add deduplication yet. Run it for a day, then query for duplicate `(provider, tx_id)` pairs and for timestamp inversions. That gives you the real duplicate and reordering rates for your traffic, which is the only baseline worth designing against.
