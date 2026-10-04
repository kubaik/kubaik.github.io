# Designing Reliable Webhook Pipelines for African Payment APIs

## Why provider documentation is not a reliability contract

Payment provider documentation describes what an API returns under normal conditions. It rarely describes what happens when a webhook arrives twice, when a retry window closes, or when a signature timestamp drifts. Teams integrating M-Pesa, Paystack or Flutterwave commonly discover these behaviours only after real traffic arrives.

A typical failure mode looks like this: an application acknowledges a webhook within the provider's timeout, but the handler does more work than the timeout allows. If the fraud check, database write and downstream notification together take longer than the provider's patience, the provider treats the delivery as failed and retries. The retry may arrive while the first attempt is still running, producing duplicate writes unless the handler is idempotent.

The practical conclusion is to treat provider callbacks as an unreliable message queue rather than as fire-and-forget HTTP requests. Once callbacks may arrive late, out of order, duplicated or never, the architecture changes. The provider is no longer the source of truth for delivery; the application's own durable log is.

## A reference architecture for payment webhooks

The pattern below decouples ingestion from processing. It has four layers:

1. **Ingress service.** A lightweight HTTP service that accepts webhooks from all providers. It validates signatures, records the raw payload with a unique event ID, and appends a message to a durable log. It does not run inference or perform long database work.
2. **Durable log.** A message queue or stream with consumer groups. Each provider gets its own stream key, for example `m_pesa_c2b`, `paystack_webhook`, `flutterwave_event`. Ordering within a stream is preserved; consumer groups allow multiple workers to share the load without losing messages.
3. **Processing workers.** Services that pull messages, run fraud scoring or other business logic, and update payment state. The critical property is idempotency: the same message may be delivered more than once, and processing it repeatedly must not create duplicate effects.
4. **Reconciliation scheduler.** A periodic job that finds events older than a threshold and requeues them. This catches messages lost during a broker restart or a worker crash.

The trade-off is added latency between webhook receipt and final state. For payment flows where a delay of a few minutes is acceptable, this is usually the right trade. For flows that require synchronous confirmation, it is not.

## Ingress service

The ingress service should do the minimum possible work before acknowledging the provider. Signature validation, payload parsing and enqueueing are the only responsibilities.

```python
# main.py
from fastapi import FastAPI, Request, HTTPException
import redis.asyncio as redis
import json
import hashlib
import hmac
from datetime import datetime, timezone

app = FastAPI()

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

PROVIDER_SECRETS = {
    "m_pesa": "your_m_pesa_passkey",
    "paystack": "your_paystack_secret",
    "flutterwave": "your_flutterwave_secret",
}

@app.post("/webhook/{provider}")
async def receive_webhook(provider: str, request: Request):
    if provider not in PROVIDER_SECRETS:
        raise HTTPException(status_code=400, detail="Unknown provider")

    body = await request.body()
    signature = request.headers.get("X-{}-Signature".format(provider.title()))

    expected = hmac.new(
        PROVIDER_SECRETS[provider].encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    if not signature or not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_id = f"{provider}:{payload.get('id', datetime.now(timezone.utc).isoformat())}"
    stream_key = f"{provider}_webhooks"

    message = {
        "event_id": event_id,
        "provider": provider,
        "payload": payload,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }

    await redis_client.xadd(
        stream_key,
        {"data": json.dumps(message)},
        maxlen=10000,
        approximate=True
    )

    return {"status": "enqueued", "event_id": event_id}
```

Design notes:

- A single generic route handles all providers. Provider-specific parsing belongs in the worker, not in the request path.
- `hmac.compare_digest` avoids timing side channels when comparing signatures.
- `XADD` with `maxlen` caps stream memory. The exact memory footprint depends on payload size and the trim policy; measure it rather than assuming a figure.
- The event ID is prefixed with the provider name to avoid collisions between providers.
- The handler returns as soon as the message is enqueued. No inference, no external calls.

## Worker service

The worker pulls from the stream, runs the fraud model, and writes to Postgres. Two properties matter: idempotency and failure isolation.

```python
# worker.py
import asyncio
import json
import logging
from datetime import datetime, timezone
import redis.asyncio as redis
import psycopg
from psycopg_pool import AsyncConnectionPool
from pybreaker import CircuitBreaker
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL = joblib.load("/app/fraud_model.joblib")
BREAKER = CircuitBreaker(fail_max=5, reset_timeout=30)

pg_pool = AsyncConnectionPool(
    conninfo="postgresql://user:pass@localhost:5432/payments",
    min_size=2,
    max_size=10,
    max_waiting=10,
    timeout=5,
)

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def predict_fraud(payload: dict) -> float:
    features = [
        float(payload.get("amount", 0)),
        float(payload.get("customer_age", 30)),
        1 if payload.get("is_first_transaction", False) else 0,
        float(payload.get("hour_of_day", 12)),
    ]
    return float(MODEL.predict_proba([features])[0][1])

async def process_stream(provider: str, consumer_name: str):
    while True:
        try:
            messages = await redis_client.xreadgroup(
                f"{provider}_consumers",
                consumer_name,
                {f"{provider}_webhooks": ">"},
                count=10,
                block=5000,
            )

            if not messages:
                continue

            for stream, message_id, data in messages[0][1]:
                payload = json.loads(data["data"])
                event_id = payload["event_id"]

                try:
                    with BREAKER:
                        risk_score = await asyncio.to_thread(
                            predict_fraud, payload["payload"]
                        )

                    async with pg_pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                """
                                INSERT INTO payment_events
                                (event_id, provider, payload, risk_score, processed_at)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (event_id) DO NOTHING
                                """,
                                (
                                    event_id,
                                    provider,
                                    json.dumps(payload["payload"]),
                                    risk_score,
                                    datetime.now(timezone.utc),
                                ),
                            )

                    await redis_client.xack(
                        f"{provider}_webhooks",
                        f"{provider}_consumers",
                        message_id,
                    )

                except Exception as e:
                    logging.error(f"Failed to process {event_id}: {e}")

        except Exception as e:
            logging.error(f"Stream consumer {consumer_name} crashed: {e}")
            await asyncio.sleep(5)

async def main():
    consumers = [
        asyncio.create_task(process_stream("m_pesa", "mpesa_worker_1")),
        asyncio.create_task(process_stream("paystack", "paystack_worker_1")),
        asyncio.create_task(process_stream("flutterwave", "flutterwave_worker_1")),
    ]
    await asyncio.gather(*consumers)

if __name__ == "__main__":
    asyncio.run(main())
```

Design notes:

- `XREADGROUP` with `>` delivers new, never-delivered messages to the consumer group.
- `XACK` is called only after the database write succeeds. If the worker crashes before acknowledging, the message remains pending and can be claimed by another consumer.
- The `ON CONFLICT (event_id) DO NOTHING` clause makes the database write idempotent. A duplicate delivery produces no duplicate row.
- The circuit breaker stops traffic to the model service after repeated failures. It does not solve the underlying problem; it prevents a slow or failing dependency from consuming all worker capacity.
- Model inference runs in a thread so the event loop is not blocked. For CPU-bound models, a separate process pool is usually a better choice.
- The `except Exception` block logs and continues. The message stays pending, which is the intended retry mechanism. A production system should also track delivery attempts and move poison messages to a dead-letter stream after a bounded number of retries.

## Reconciliation scheduler

A periodic job scans recent stream entries and moves anything older than the threshold to a dead-letter stream for reprocessing.

```python
# scheduler.py
import asyncio
import json
from datetime import datetime, timedelta, timezone
import redis.asyncio as redis

redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

async def find_late_events():
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

    for provider in ["m_pesa", "paystack", "flutterwave"]:
        stream_key = f"{provider}_webhooks"
        messages = await redis_client.xrevrange(stream_key, count=100)

        for message_id, data in messages:
            payload = json.loads(data["data"])
            received_at = datetime.fromisoformat(payload["received_at"])

            if received_at < cutoff:
                await redis_client.xadd(
                    f"{provider}_dlq",
                    {"data": data["data"]},
                    maxlen=5000,
                    approximate=True
                )
                await redis_client.xack(
                    stream_key, f"{provider}_consumers", message_id
                )
                print(f"Requeued late event {message_id} from {provider}")

if __name__ == "__main__":
    asyncio.run(find_late_events())
```

This is deliberately crude. It scans only the last 100 entries per stream, so it will miss older pending messages in a large backlog. A more robust version uses `XPENDING` to inspect the pending entries list directly, and a bounded retry counter per message.

## Measuring the pipeline instead of asserting numbers

Latency and requeue rates are properties of a specific deployment, provider behaviour and traffic shape. They should be measured, not copied from an article.

To measure webhook latency, record two timestamps per event: the provider's own event timestamp from the payload, and the ingress receipt time. The difference is the delivery latency. Store both in the event row and compute percentiles with a query over a rolling window.

To measure worker latency, record the ingress receipt time and the time the database write commits. The difference is the processing latency. Persisting both timestamps makes the distribution queryable without adding a metrics dependency.

To measure requeue rate, count messages moved to the dead-letter stream per provider per day, divided by total events for that provider.

To measure reconciliation rate, compare the set of provider-side transaction IDs for a period against the set of `event_id` values in the payment table. The gap is the reconciliation failure set.

A useful load test is to replay a captured day of webhook payloads at a controlled rate against a staging deployment and watch the pending entries list on each stream. If pending entries grow without bound, workers are the bottleneck. If they stay flat but worker latency rises, the database or model service is the bottleneck.

## Failure modes and how to handle them

### Duplicate deliveries

Providers may retry a delivery even after a successful response, for example if the acknowledgement was lost in transit. Idempotency at the database layer is the only reliable defence. A unique constraint on `event_id` with `ON CONFLICT DO NOTHING` is sufficient for simple cases. For side effects outside the database, such as sending an SMS, use an outbox table that is written in the same transaction as the payment update, and a separate process that drains the outbox exactly once.

### Out-of-order arrivals

A payment may be captured before it is authorised, or a refund may arrive before the original charge. Processing order matters for state transitions. One approach is to store the provider's event timestamp and reject transitions that move state backwards. Another is to buffer events per transaction ID for a short window and apply them in timestamp order.

### Signature validation failures

Signature schemes that include a timestamp will reject requests when the server clock drifts. Ensure NTP is running and monitor clock offset. For providers that use a separate webhook secret from the API secret, keep the two in distinct configuration keys to avoid accidental reuse.

### Provider retry windows

Providers retry on their own schedule, which is usually documented but not always honoured exactly. Do not rely on the provider's retry to recover from a sustained outage. If the worker pool is down for longer than the provider's retry window, the events are lost from the provider's perspective. The reconciliation scheduler and a periodic pull of provider transaction history are the safety net.

### Database connection exhaustion

A connection pool with a fixed maximum will reject new connections once saturated. Monitor pool wait time and queue depth. When the pool is exhausted, the correct response is usually to slow down the consumer, not to increase the pool size indefinitely. Increasing the pool shifts the bottleneck to the database server.

### Model service latency and cold starts

If the fraud model runs as a separate service, its cold start and tail latency become part of the webhook processing path. Options include keeping a warm replica, using a provisioned concurrency mechanism if the platform offers one, or batching predictions. Each has a cost. The right choice depends on the acceptable p99 for the payment flow.

### Redis memory growth

Streams accumulate entries until trimmed. `MAXLEN` with `approximate=True` is cheap but not exact; memory can grow above the nominal cap. Monitor `used_memory` and the stream length. If the broker restarts with AOF persistence, recovery time scales with the size of the append-only file.

### Model drift

A fraud model trained on historical data will drift as user behaviour changes. Track the distribution of risk scores over time and the rate of manual overrides. A rising override rate is an early signal that retraining is needed. The retraining pipeline should be separate from the serving path.

## When this architecture is the wrong choice

The queue-based pattern is not universal. It is a poor fit when:

- **Sub-second synchronous confirmation is required.** The queue adds latency between receipt and final state. If the user must see a confirmed result in the same request, process synchronously and use the queue only for post-processing.
- **The provider offers a pull-based reconciliation API.** If the provider exposes a transaction history endpoint, a periodic pull may be simpler and more reliable than relying on webhooks. Webhooks become an optimisation, not the source of truth.
- **Regulatory rules require immediate settlement.** Some instant payment schemes mandate synchronous confirmation. In that case the webhook is a notification, not the mechanism of record.
- **The event volume is very low.** For a few hundred events per day, a single-process handler with a database unique constraint may be sufficient. Adding a broker increases operational surface for little benefit.
- **The team cannot operate a broker.** A managed queue or a database-backed job table may be a better fit than self-hosted Redis or Kafka if there is no operational capacity to run it.

## A decision checklist

Before adopting this pattern, answer these questions:

1. What is the maximum acceptable delay between webhook receipt and final payment state?
2. Does the provider expose a transaction history endpoint that can be polled for reconciliation?
3. What is the provider's documented retry schedule, and what happens after the final retry?
4. Can every downstream side effect be made idempotent, or is an outbox required?
5. What is the p99 latency of the slowest step in the processing path, and does it fit within the provider's acknowledgement timeout?
6. How will duplicate and out-of-order events be detected in production, not just in tests?
7. What is the plan when the broker itself is unavailable?
8. Who is paged when the dead-letter stream grows?

## Next 30 minutes

Open the webhook handler for one payment provider and add two columns to the event table: `provider_event_timestamp` and `ingress_received_at`. Deploy the change and start recording both values. Within a day, query the p99 of `ingress_received_at - provider_event_timestamp` per provider. That single number tells you whether the current architecture is absorbing provider latency or merely hiding it until the next traffic spike.
