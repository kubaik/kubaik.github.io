# AI features: M-Pesa, Paystack, Flutterwave traps

Payment provider documentation is written for the happy path: a successful charge, a webhook that fires once, and a JSON body containing `status: "success"`. It rarely covers a mobile money transfer that sits in `pending` for six hours, a webhook that arrives twice with different payloads, or an API that returns HTTP 200 with an error message embedded in the body. AI features built on payment data — fraud scoring, retry logic, support triage — inherit all of that unreliability.

Most AI payment integrations assume the payment layer is a clean event source. It is not. M-Pesa, Paystack and Flutterwave each have distinct failure modes that do not resemble a standard HTTP error. M-Pesa STK Push returns `ResponseCode: "0"` for a request that was accepted but not yet processed, with the real result arriving later via callback. Paystack emits `charge.success` and `transfer.success` webhooks with entirely different schemas. Flutterwave's v3 API returns `status: "success"` inside a 200 response even when the transaction failed, with the real status buried in `data.status`.

If an AI feature makes decisions from those signals — "should this payment be retried?", "is this customer likely to churn?", "should this transaction be flagged?" — the mess has to be normalized before the model sees it. The normalization layer, not the model, is where most of the engineering effort goes.

## The shape of the problem

At the core this is an event normalization pipeline. Each provider emits events in its own format, with its own timing semantics and its own definition of "done." The job is to map those into a single internal event schema so downstream features never branch on provider.

The first thing to understand is that these providers differ in *when* they consider a transaction final, not just in payload shape. M-Pesa's C2B and STK Push flows are asynchronous by design: the initial API call returns a `CheckoutRequestID`, and the final result arrives via callback to a URL registered in advance. That callback can arrive in five seconds or five minutes. Paystack is mostly synchronous for card charges but asynchronous for transfers and recurring billing. Flutterwave is a mix: card charges return a `tx_ref` that must be verified with a second API call, while bank transfers use webhooks.

So an AI feature cannot react to a single event. It needs a state machine that tracks each transaction across multiple events, with timeouts and reconciliation. A common trap is treating the webhook as the source of truth. Webhooks get lost, duplicated and delivered out of order. A fraud model that only sees webhooks will miss transactions where the webhook never arrived but the payment succeeded — or worse, count a duplicate webhook as a separate transaction.

The second layer is idempotency. Each provider has a different stable identifier: M-Pesa uses `CheckoutRequestID` and `MerchantRequestID`, Paystack uses `reference`, Flutterwave uses `tx_ref`. These need to map into a single `transaction_id` usable as a primary key. Without that mapping, a model double-counts transactions and produces garbage predictions.

The third layer is error semantics. A 200 response does not mean success. A 400 does not always mean failure: M-Pesa returns `errorCode: "500.001.1001"` for a duplicate request, which is actually a success if the original went through. Paystack returns `status: false` with a `message` field for validation errors, but `status: true` with `data.status: "failed"` for a declined card. Flutterwave returns `status: "error"` for both validation errors and system failures, with the distinction buried in `data.status`.

The normalized `outcome` field should therefore have four values: `succeeded`, `failed`, `pending` and `unknown`. The `unknown` state matters. It is what to use when the provider's response is ambiguous or a callback is still outstanding. Defaulting ambiguous responses to `failed` causes retry logic to re-attempt payments that already succeeded.

## A minimal normalization layer

The following example uses Python with FastAPI and Pydantic. It handles the three providers' webhook formats and produces a unified event.

First, the normalized event schema:

```python
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum

class Outcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    UNKNOWN = "unknown"

class NormalizedEvent(BaseModel):
    transaction_id: str
    provider: str
    amount_minor: int          # smallest currency unit, never a float
    currency: str
    outcome: Outcome
    raw_payload: dict
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provider_timestamp: datetime | None = None
```

Note that `amount_minor` is an integer. Storing money as a float invites reconciliation mismatches; store the smallest unit (kobo, cents) and convert only for display.

Now the provider-specific parsers. Each takes a raw webhook body and returns a `NormalizedEvent`, or raises `ValueError` if the payload is unrecognizable.

```python
def parse_mpesa(payload: dict) -> NormalizedEvent:
    # M-Pesa C2B / STK callback format
    stk_callback = payload.get("Body", {}).get("stkCallback", {})
    if not stk_callback:
        raise ValueError("Not an M-Pesa STK callback")

    result_code = stk_callback.get("ResultCode")
    checkout_id = stk_callback.get("CheckoutRequestID")

    # ResultCode 0 = success, 1032 = cancelled by user,
    # 1037 = timeout (may still settle later), others = failure
    if result_code == 0:
        outcome = Outcome.SUCCEEDED
    elif result_code == 1032:
        outcome = Outcome.FAILED
    elif result_code == 1037:
        outcome = Outcome.PENDING
    else:
        outcome = Outcome.UNKNOWN

    metadata = stk_callback.get("CallbackMetadata", {}).get("Item", [])
    amount = next(
        (item["Value"] for item in metadata if item["Name"] == "Amount"), 0
    )

    return NormalizedEvent(
        transaction_id=checkout_id,
        provider="mpesa",
        amount_minor=int(round(float(amount) * 100)),
        currency="KES",
        outcome=outcome,
        raw_payload=payload,
    )

def parse_paystack(payload: dict) -> NormalizedEvent:
    event = payload.get("event")
    data = payload.get("data", {})

    if event == "charge.success":
        outcome = Outcome.SUCCEEDED
    elif event == "charge.failed":
        outcome = Outcome.FAILED
    else:
        outcome = Outcome.UNKNOWN

    return NormalizedEvent(
        transaction_id=data.get("reference", ""),
        provider="paystack",
        amount_minor=data.get("amount", 0),  # Paystack already uses kobo
        currency=data.get("currency", "NGN"),
        outcome=outcome,
        raw_payload=payload,
    )

def parse_flutterwave(payload: dict) -> NormalizedEvent:
    data = payload.get("data", {})
    status = data.get("status")

    if status == "successful":
        outcome = Outcome.SUCCEEDED
    elif status == "failed":
        outcome = Outcome.FAILED
    elif status == "pending":
        outcome = Outcome.PENDING
    else:
        outcome = Outcome.UNKNOWN

    return NormalizedEvent(
        transaction_id=data.get("tx_ref", ""),
        provider="flutterwave",
        amount_minor=int(round(float(data.get("amount", 0)) * 100)),
        currency=data.get("currency", "NGN"),
        outcome=outcome,
        raw_payload=payload,
    )
```

This is the minimum viable normalizer. It does not handle idempotency or deduplication. For that, a store such as Redis can track `transaction_id` values and reject duplicates within a window. A 24-hour TTL on the idempotency key is a reasonable starting point, since many providers retry webhooks for up to 24 hours.

## Measuring the cost instead of guessing it

Latency and reconciliation load are the two numbers worth knowing, and both are measurable on your own stack rather than borrowed from someone else's.

**Normalization latency.** Instrument the handler with a timer that starts when the HTTP request is accepted and stops when the `NormalizedEvent` is persisted. Emit that duration as a histogram (for example, a Prometheus histogram or a StatsD timing metric) tagged by provider. Then compare the p50 and p99 against your provider's webhook timeout. If p99 approaches the timeout, the handler is doing too much synchronously and the AI inference should move to a queue.

**Idempotency store latency.** Time the store lookup separately from the parse step. The difference between the two histograms tells you whether the bottleneck is parsing or I/O, which determines whether to optimize code or move the store closer to the application.

**Reconciliation volume.** Count rows in `pending` state older than your threshold, per provider, per hour. That count, multiplied by your polling interval, gives the API call rate your reconciliation job will generate. Compare it against the provider's documented rate limit before shipping. If the projected rate exceeds the limit, batch requests or lengthen the interval.

**Duplicate rate.** Log every webhook delivery with its provider-supplied delivery ID and its transaction ID. The ratio of distinct transaction IDs to total deliveries over a day is the duplicate rate. This is the single most useful number for deciding how aggressively to deduplicate.

None of these require a benchmark table from someone else's system. They require a timer, a counter and a dashboard.

## Failure modes worth designing around

**Duplicate webhooks with different payloads.** A provider may send a success webhook, and if the endpoint times out, retry with the same transaction reference but a new delivery ID. If the idempotency key is the delivery ID, the same transaction is processed twice. Key on `reference` or `tx_ref`, never the delivery ID.

**Timezone drift.** Provider timestamps may be expressed in East Africa Time, West Africa Time or UTC depending on the integration. If an AI feature derives time-series features such as "transaction velocity in the last hour," mixing timezones produces nonsense. Convert to UTC on ingest and store the offset separately if it matters.

**Currency unit confusion.** Paystack expresses amounts in kobo, Flutterwave in the major unit, M-Pesa in KES. Without normalization, a 10,000 NGN transaction becomes 1,000,000 minor units and a fraud model flags it as anomalous. Normalize to the smallest unit at the boundary.

**Callback URL registration.** M-Pesa requires a publicly accessible HTTPS callback URL registered with the provider, and the registration is cached. Changing tunnels or hosts during development means re-registering, and a URL that works in staging can fail in production behind firewall rules. Verify the registered URL from outside the network before relying on it.

**Ambiguous error codes.** A timeout code such as M-Pesa's 1037 means the transaction may still settle. Treating it as a failure and triggering a retry can double-charge the customer. Mark it `pending` and reconcile.

**Unverified webhook signatures.** Paystack and Flutterwave sign their webhooks with a shared secret. Without verification, anyone who knows the endpoint URL can inject fake events, corrupting both the ledger and any training data derived from it. Verify with a constant-time comparison such as `hmac.compare_digest` before parsing.

**Unredacted payload storage.** Payment payloads contain phone numbers, email addresses and sometimes partial card data. Writing them verbatim to a shared logging system creates a compliance exposure. Redact known sensitive fields before the payload leaves the handler.

## Choosing between real-time normalization and batch

The normalization layer is not always the right answer. A short decision checklist:

- **One provider, low volume.** If there is a single provider and transaction volume is low, a webhook handler with a `switch` on event type is sufficient. The complexity of a full normalization layer only pays off with multiple providers or meaningful volume.
- **Batch-oriented AI features.** A monthly churn prediction does not need real-time normalization. Pull data from each provider's API on a schedule and join it in the warehouse. This is simpler and more reliable than normalizing in real time.
- **Prototype or MVP.** Skip reconciliation. Accept that some transactions will be missed and focus on validating the AI feature itself. Add reconciliation when there are real users whose money is at stake.
- **Provider offers a unified API.** A payment orchestrator that already normalizes across providers may remove weeks of work. The trade-off is cost, margin and loss of provider-specific control.
- **Regulatory constraints.** If transaction data must remain in a particular jurisdiction, a third-party orchestrator may not be an option, and the normalization layer has to be built in-house.

## Testing and tooling

For the normalization layer, FastAPI with Pydantic gives automatic request validation and generated OpenAPI docs, which helps when inspecting unfamiliar webhook payloads. A Node.js stack with Express works similarly, but raw body parsing must be preserved for signature verification — a JSON body parser that consumes the stream first will break the HMAC check.

For idempotency and state tracking, Redis with `SET key value NX EX 86400` atomically sets an idempotency key with a 24-hour expiry. If persistence matters more than throughput, a relational database with a unique index on `transaction_id` works, at the cost of higher write latency.

For reconciliation, an HTTP client and a scheduler are usually enough. A full workflow engine is rarely justified unless the flow involves genuinely multi-step sagas across systems.

For the AI feature itself, be aware that a hosted model API can add hundreds of milliseconds to seconds of latency. If that exceeds the provider's webhook timeout, score asynchronously and store the result for later retrieval rather than blocking the handler.

To test webhook handling locally, expose the development server through a tunnel and register that URL with the provider. M-Pesa requires re-registration whenever the URL changes; Paystack and Flutterwave offer dashboard or CLI tools for sending test events.

## FAQ

**How should callbacks that never arrive be handled?**
With a reconciliation job that polls the provider's transaction status endpoint. For M-Pesa, the Transaction Status API accepts the `CheckoutRequestID`. Run the job on a schedule against any transaction in `pending` older than a threshold. If the status API also fails, mark the transaction `unknown` and alert a human rather than guessing.

**Why do duplicate webhooks arrive at all?**
Providers retry when the endpoint does not return a success response quickly enough, and network issues on their side can also produce duplicates. The fix is an idempotent handler that checks the transaction reference against a store before processing.

**How should Flutterwave's status field be normalized?**
Read `data.status`, not the top-level `status`, which may be `success` even for failed transactions. Map `successful` to `succeeded`, `failed` to `failed`, `pending` to `pending`, and everything else to `unknown`.

**Is it safe to run AI inference inside the webhook handler?**
Only if the p99 inference latency is comfortably below the provider's timeout. Otherwise enqueue the event, return success immediately, and score asynchronously.

## Do this in the next 30 minutes

Open the webhook handler and find where the transaction ID is derived. If it uses the webhook delivery ID rather than the provider's `reference`, `tx_ref` or `CheckoutRequestID`, that is the bug that double-processes transactions. Change it to the provider's stable identifier, then add a `SET NX EX 86400` check against Redis before processing. That is a small change that removes an entire class of duplicate-event bugs from everything downstream, including any model trained on the resulting data.
