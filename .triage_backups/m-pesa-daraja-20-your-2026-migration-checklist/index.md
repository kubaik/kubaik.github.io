# M-Pesa Daraja 2.0: your 2026 migration checklist

API version migrations are rarely a version bump. They change authentication, request schemas, callback shapes and error semantics all at once, and the failure modes only appear under production traffic. This article is a migration checklist for an M-Pesa Daraja-style integration, written so that it applies whether the breaking change you are facing is a documented major version, a silent gateway upgrade, or a deprecation deadline you have been ignoring.

It assumes you already integrate with the Daraja APIs, know what a Lipa Na M-Pesa Online (STK push) request looks like, and have a sandbox account on the Safaricom Developer Portal. If you are starting from scratch, register there first and obtain a Consumer Key and Consumer Secret before continuing.

## Why version migrations break in production

The recurring failure mode is not that the new endpoint is hard to call. It is that the old one keeps working in sandbox long after it stops working in production, or vice versa. A few patterns show up repeatedly:

- **Mocked headers.** Sandbox test suites often mock request headers, so a newly required header is never exercised. The call passes locally and returns a precondition error against the live gateway.
- **Silent schema changes.** A field is renamed, re-cased, or moved one level deeper in the JSON response. Naive parsers that index directly into the payload raise `KeyError` or return `None` at exactly the wrong moment.
- **Deadline clustering.** Deprecation dates push many teams into the same final week. Support queues lengthen, and incident response capacity drops precisely when it is needed most.
- **Callback drift.** The request succeeds, the money moves, and the callback handler silently drops the notification because the payload shape changed.

The checklist below is ordered so that each step makes the next one testable. Nothing here depends on a specific SDK version; it depends on reading the current portal documentation and verifying behaviour against the sandbox.

## Prerequisites and what you will build

You will need:

- A Unix shell (Linux or macOS; Windows via WSL2 is fine)
- Python 3.11 or Node 20 LTS
- Redis 7.2 for caching access tokens (optional but recommended)
- A sandbox app registered on the Safaricom portal
- A public HTTPS tunnel such as ngrok to expose your webhook — localhost callbacks will not be delivered

The result is a minimal but production-shaped integration that:

- Authenticates against the OAuth token endpoint and caches the token
- Issues an STK push request
- Parses the callback payload defensively
- Sends an idempotency key with every payment attempt
- Handles success, timeout and user cancellation with structured logging

## Step 1 — pin your environment

Create a directory and install pinned packages:

```bash
git init mpesa-migration
cd mpesa-migration
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install requests==2.31.0 redis==7.2.0 python-dotenv==1.0.0 tenacity==8.2.3
```

The Node equivalent:

```bash
npm init -y
npm install axios@1.6.2 redis@7.2.0 dotenv@16.3.1 p-retry@6.1.0
touch .env
```

Create a `.env` file. Values come from your portal app; the shortcode below is the standard sandbox test shortcode used in Safaricom's public examples:

```
MPESA_CONSUMER_KEY=your_consumer_key
MPESA_CONSUMER_SECRET=your_consumer_secret
MPESA_PASSKEY=your_passkey
MPESA_BUSINESS_SHORT_CODE=174379
MPESA_CALLBACK_URL=https://your-public-url.com/mpesa/callback
REDIS_URL=redis://localhost:6379/0
```

Start Redis locally if you want token caching:

```bash
docker run --name redis-mpesa -p 6379:6379 -d redis:7.2-alpine
```

Before writing any code, do one thing manually: call the OAuth endpoint with `curl` and print the raw response body. Do not assume the token is at the top level. Response envelopes change between environments, and one extra nesting level is enough to break a parser that indexes directly.

```bash
curl -s -X POST "https://sandbox.safaricom.co.ke/oauth/v1/generate?grant_type=client_credentials" \
  -u "$MPESA_CONSUMER_KEY:$MPESA_CONSUMER_SECRET" | python -m json.tool
```

Whatever structure that command prints is the structure your client must handle. Treat the code below as a template and adjust the accessor to match what you actually observe.

## Step 2 — core implementation

Save this as `mpesa.py`. Note that the token accessor is written defensively: it accepts either a top-level `access_token` or one nested under `body`, because both shapes have appeared in Daraja responses across environments.

```python
import os
import uuid
import hashlib
import logging
import requests
import redis
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("mpesa")

class DarajaClient:
    OAUTH_URL = "https://sandbox.safaricom.co.ke/oauth/v1/generate"
    STK_URL = "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest"

    def __init__(self):
        self.consumer_key = os.environ["MPESA_CONSUMER_KEY"]
        self.consumer_secret = os.environ["MPESA_CONSUMER_SECRET"]
        self.passkey = os.environ["MPESA_PASSKEY"]
        self.shortcode = os.environ["MPESA_BUSINESS_SHORT_CODE"]
        self.callback = os.environ["MPESA_CALLBACK_URL"]
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def get_token(self):
        cache_key = "daraja_token"
        cached = self.redis.get(cache_key)
        if cached:
            return cached.decode()

        resp = requests.post(
            self.OAUTH_URL,
            params={"grant_type": "client_credentials"},
            auth=(self.consumer_key, self.consumer_secret),
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Accept either envelope shape; log which one you got so you notice drift.
        token = payload.get("access_token") or payload.get("body", {}).get("access_token")
        if not token:
            raise RuntimeError(f"no access_token in response keys={list(payload)}")
        self.redis.setex(cache_key, 3500, token)  # token lifetime minus a buffer
        return token

    def lipa_stk(self, phone: str, amount: int, reference: str, idempotency_key: str):
        token = self.get_token()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        password = self._generate_password(timestamp)

        payload = {
            "BusinessShortCode": self.shortcode,
            "Password": password,
            "Timestamp": timestamp,
            "TransactionType": "CustomerPayBillOnline",
            "Amount": amount,
            "PartyA": phone,
            "PartyB": self.shortcode,
            "PhoneNumber": phone,
            "CallBackURL": self.callback,
            "AccountReference": reference,
            "TransactionDesc": "Payment for goods",
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Idempotency-Key": idempotency_key,
        }

        resp = requests.post(self.STK_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _generate_password(self, timestamp):
        raw = f"{self.shortcode}{self.passkey}{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()
```

Three details are worth calling out because they are the ones that change between versions:

- **Timestamp format.** The STK push password is derived from `BusinessShortCode + PassKey + Timestamp`, where the timestamp is a 14-digit `YYYYMMDDHHMMSS` string. If your timestamp helper produces a different width or includes separators, the derived password will not match and the request will be rejected. Assert the length before sending.
- **Password derivation.** Older integrations used Base64 of the concatenated string. Newer ones use a SHA-256 hex digest. These are not interchangeable — check the current documentation for your account and verify against the sandbox before switching.
- **Idempotency key.** Send a fresh UUID per logical payment attempt, not per HTTP attempt. Retries of the same logical payment must reuse the same key; distinct payments must not.

The Node equivalent, saved as `mpesa.js`:

```javascript
import axios from "axios";
import crypto from "crypto";
import dotenv from "dotenv";
import { createClient } from "redis";

dotenv.config();

const client = createClient({ url: process.env.REDIS_URL });
await client.connect();

class DarajaClient {
  OAUTH_URL = "https://sandbox.safaricom.co.ke/oauth/v1/generate";
  STK_URL = "https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest";

  async getToken() {
    const key = "daraja_token";
    const cached = await client.get(key);
    if (cached) return cached;

    const auth = Buffer.from(
      `${process.env.MPESA_CONSUMER_KEY}:${process.env.MPESA_CONSUMER_SECRET}`
    ).toString("base64");

    const res = await axios.post(
      this.OAUTH_URL,
      null,
      {
        params: { grant_type: "client_credentials" },
        headers: { Authorization: `Basic ${auth}` },
        timeout: 10_000,
      }
    );
    const payload = res.data;
    const token = payload.access_token ?? payload.body?.access_token;
    if (!token) throw new Error(`no access_token in keys=${Object.keys(payload)}`);
    await client.setEx(key, 3500, token);
    return token;
  }

  async lipaStk(phone, amount, reference, idempotencyKey) {
    const token = await this.getToken();
    const timestamp = new Date()
      .toISOString()
      .replace(/[-:TZ.]/g, "")
      .slice(0, 14);
    const password = this.generatePassword(timestamp);

    const payload = {
      BusinessShortCode: process.env.MPESA_BUSINESS_SHORT_CODE,
      Password: password,
      Timestamp: timestamp,
      TransactionType: "CustomerPayBillOnline",
      Amount: amount,
      PartyA: phone,
      PartyB: process.env.MPESA_BUSINESS_SHORT_CODE,
      PhoneNumber: phone,
      CallBackURL: process.env.MPESA_CALLBACK_URL,
      AccountReference: reference,
      TransactionDesc: "Payment for goods",
    };

    const headers = {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-Idempotency-Key": idempotencyKey,
    };

    const res = await axios.post(this.STK_URL, payload, { headers, timeout: 10_000 });
    return res.data;
  }

  generatePassword(timestamp) {
    const raw = `${process.env.MPESA_BUSINESS_SHORT_CODE}${process.env.MPESA_PASSKEY}${timestamp}`;
    return crypto.createHash("sha256").update(raw).digest("hex");
  }
}

const daraja = new DarajaClient();
daraja
  .lipaStk("+254712345678", 100, "INV-12345", crypto.randomUUID())
  .then(console.log)
  .catch(console.error);
```

## Step 3 — handle errors and retries

The most useful thing you can do here is not memorise a table of error codes. It is to log the full response body on any non-2xx status, including the status code and any correlation identifier the gateway returns. Error taxonomies change between versions; the habit of capturing the raw body does not.

That said, a few classes of failure are stable enough to plan for:

| Class | Typical cause | Correct response |
|---|---|---|
| 400 validation | Malformed timestamp, wrong password derivation, bad payload shape | Do not retry. Fix the request. Log the full body. |
| 401 / 403 | Expired or wrong credentials, wrong environment | Refresh the token; if it persists, check the portal app. Do not retry in a tight loop. |
| 409 / duplicate | Same idempotency key submitted twice | Treat as success if the original succeeded. Query the transaction status before assuming failure. |
| 429 | Rate limiting | Retry with exponential backoff and jitter. Cap the total attempts. |
| 5xx / timeout | Gateway or network failure | Retry with the same idempotency key so the payment is not duplicated. |

The critical rule: **retry with the same idempotency key, never a new one.** Generating a fresh UUID inside the retry wrapper is the single most common way to double-charge a customer.

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

class TransientError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=5),
    retry=retry_if_exception_type(TransientError),
)
def lipa_stk_retry(self, phone, amount, reference, idempotency_key):
    try:
        return self.lipa_stk(phone, amount, reference, idempotency_key)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status is not None and 500 <= status < 600:
            raise TransientError(str(exc)) from exc
        raise  # 4xx: do not retry
```

The Node equivalent with `p-retry`:

```javascript
import retry from "p-retry";

async function lipaStkRetry(phone, amount, reference, idempotencyKey) {
  return retry(
    async () => {
      try {
        return await daraja.lipaStk(phone, amount, reference, idempotencyKey);
      } catch (err) {
        const status = err.response?.status;
        if (status >= 500) throw err; // retryable
        throw new retry.AbortError(err.message); // 4xx: stop
      }
    },
    { retries: 3, minTimeout: 1000, maxTimeout: 5000 }
  );
}
```

## Step 4 — parse the callback defensively

Callback payload shapes are the most common source of silent data loss during a migration. The request succeeds, the customer is charged, and your handler throws on a missing key and returns a 500 — which may cause the gateway to retry, or may simply leave your system out of sync.

Write the handler so that an unexpected shape produces an alert rather than an exception. Log the raw body first, then parse.

```javascript
import express from "express";

const app = express();

app.post("/mpesa/callback", express.json(), (req, res) => {
  const raw = JSON.stringify(req.body);
  console.log("mpesa.callback.raw", raw);

  const callback = req.body?.Body?.stkCallback ?? req.body?.stkCallback;
  if (!callback) {
    console.error("mpesa.callback.unrecognised_shape", raw);
    // Acknowledge anyway: a 5xx here causes gateway retries, not a fix.
    return res.status(200).send("Accepted");
  }

  const { ResultCode, ResultDesc, CheckoutRequestID, CallbackMetadata } = callback;

  if (ResultCode === 0) {
    const items = CallbackMetadata?.Item ?? [];
    const receipt = items.find((i) => i.Name === "MpesaReceiptNumber")?.Value;
    console.log("mpesa.callback.success", { CheckoutRequestID, receipt });
  } else {
    // ResultCode 1032 is a user cancellation; 1037 is a timeout.
    console.log("mpesa.callback.failure", { CheckoutRequestID, ResultCode, ResultDesc });
  }

  res.status(200).send("Accepted");
});
```

Two operational rules matter more than the parsing itself:

1. **Always return 200 once you have accepted the body.** If your business logic needs to fail, fail asynchronously and alert. A non-200 response invites retries you probably do not want.
2. **Persist the raw payload before parsing.** When a shape changes, the stored payload is the only way to reconstruct what happened.

## Step 5 — observability and tests

Instrument four things. Each maps to a failure mode above:

- **Token fetch count.** A counter incremented on every cache miss. If this climbs during steady traffic, your cache TTL is wrong or Redis is unreachable.
- **Non-2xx responses by status class.** Split 4xx from 5xx. A spike in 4xx means a schema or credential problem; a spike in 5xx means upstream trouble.
- **Idempotency key reuse within a window.** Increment a counter when the same key is seen twice inside, say, five minutes. This catches double-submission races in your own checkout flow.
- **Callback parse failures.** Increment when the defensive branch above fires. This is your earliest signal that a payload shape changed.

A minimal OpenTelemetry span around the token fetch:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def get_token(self):
    with tracer.start_as_current_span("mpesa.get_token") as span:
        span.set_attribute("component", "mpesa")
        # ... existing implementation
```

For tests, mock the HTTP layer and assert on the *shape* you expect, so that a change in the real response fails a test rather than a production deploy:

```python
from unittest.mock import patch, MagicMock
from mpesa import DarajaClient

def test_token_top_level():
    client = DarajaClient()
    resp = MagicMock()
    resp.json.return_value = {"access_token": "abc"}
    with patch("requests.post", return_value=resp):
        assert client.get_token() == "abc"

def test_token_nested():
    client = DarajaClient()
    resp = MagicMock()
    resp.json.return_value = {"body": {"access_token": "abc"}}
    with patch("requests.post", return_value=resp):
        assert client.get_token() == "abc"

def test_timestamp_is_fourteen_digits():
    client = DarajaClient()
    ts = "20260101120000"
    assert len(ts) == 14
    assert client._generate_password(ts) == client._generate_password(ts)
```

## Measuring the migration instead of guessing

Any claim about latency improvement, failure-rate reduction or cost saving should come from your own instrumentation, not from a blog post. Here is how to produce the numbers honestly.

**Latency.** Wrap every outbound HTTP call in a timer and export a histogram. Compare the p50 and p95 of the STK push call before and after the migration, over the same traffic mix. If you add token caching at the same time, measure the token fetch separately so you can attribute the change.

**Failure rate.** Count STK push attempts and non-zero `ResponseCode` values. Report the ratio over a fixed window, and segment by error class. A drop from one rate to another is only meaningful if the traffic mix and the upstream gateway are the same.

**Callback processing time.** Time the handler from request receipt to acknowledgement. If your handler does synchronous database work before responding, that is the number to reduce, and it is usually reducible without touching the payment API at all.

**Cost.** Only compute this from your own billing data. Count callbacks, count retries, and multiply by your actual per-message rate. Do not accept a percentage figure from an article — including this one.

## A rollout checklist

Run the migration in this order. Each step is a gate.

1. **Confirm the current contract.** Call the OAuth and STK endpoints from `curl` against the sandbox and save the raw responses.
2. **Diff against your client.** For every field your code reads, confirm the path still exists. Any field you read with a direct index is a risk.
3. **Add the defensive accessors.** Token and callback parsing should tolerate both old and new shapes during the transition.
4. **Add idempotency keys.** Before any retry logic exists, make sure a stable key is generated once per logical payment.
5. **Add retry logic with the same key.** Verify with a test that a simulated 500 does not produce two payments.
6. **Instrument the four metrics.** Deploy them before the migration so you have a baseline.
7. **Canary.** Route a small percentage of traffic to the new path and watch the 4xx rate. Roll forward only when it is stable.
8. **Retire the old path.** Remove the dual-shape accessors once the old endpoint is decommissioned, so the code does not accumulate dead branches.

## FAQ

**Should I cache the access token?**
Yes, if you make more than a handful of calls per minute. The token has a finite lifetime; cache it for slightly less than that and refresh on expiry. Handle a 401 by invalidating the cache and retrying once.

**What if the sandbox and production return different response shapes?**
This happens. The defensive accessor pattern in Step 2 exists for exactly this reason. Log which shape you received at least once per environment so the difference is visible rather than mysterious.

**Do I need a public HTTPS URL for callbacks?**
Yes. The gateway must be able to reach your callback endpoint over the public internet. A tunnel such as ngrok works for development; production needs a real hostname with a valid certificate.

**How do I test a user cancellation?**
The sandbox supports simulating failure result codes. Assert that your handler distinguishes a cancellation from a timeout and that neither is treated as success.

**Can I keep the old endpoints running in parallel?**
During a canary period, yes. Keep the two code paths separate and route by configuration rather than by branching inside a single function, so retiring one is a deletion rather than a refactor.

## Action for the next 30 minutes

Open your integration's HTTP client code and find every place it reads a field from an M-Pesa response by direct index — `response["access_token"]`, `payload["Body"]["stkCallback"]`, and so on. For each one, replace it with a defensive accessor that logs the full response body when the expected path is missing, and add a test that asserts the accessor handles both the old and new shapes. That single change converts your next schema surprise from a production incident into a log line.
