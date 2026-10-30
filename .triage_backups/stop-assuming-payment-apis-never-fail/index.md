# Stop assuming payment APIs never fail

The default configuration is fine right up until it isn't. Traditional observability was never the hard part; knowing when a dependency is about to fail was. This article walks through the tradeoffs of putting an AI recommendation call and a third-party payment call on the same request path, and what to do when the payment provider stops answering the way you expect.

## The problem: two unreliable dependencies on one request path

When a recommendation engine is stitched into a checkout flow that uses M-Pesa, Paystack, or Flutterwave, the first thing that looks impressive is the instant personalization. The second thing that trips most teams is the hidden latency and error surface that comes from the payment gateway. A payment provider that returns a timeout on a meaningful fraction of requests can cascade into a large drop in conversion, because the surrounding service aborts the request early rather than degrading gracefully.

The assumption that trips people up is simple: that a payment provider will always answer within its advertised SLA. It won't, and the failure is rarely clean. A gateway can accept your request, time out on the response, and still settle the charge. That asymmetry — an ambiguous outcome rather than a clear error — is what makes payment integrations harder than ordinary HTTP dependencies.

This article covers three things: how to structure the call path so a slow gateway doesn't take down checkout, how to handle provider-specific error semantics without losing the information you need to decide whether to retry, and how to reconcile the transactions that end up in an unknown state.

## Prerequisites and what you'll build

The example builds a small FastAPI (Python 3.11) service that:

1. Receives a user ID and a purchase amount.
2. Calls a model hosted behind a managed inference endpoint (SageMaker Runtime is used here as the concrete example).
3. Sends the payment request to the selected gateway (M-Pesa, Paystack, or Flutterwave).
4. Returns a JSON payload containing the recommendation, the payment status, and a fallback flag.

The service runs on AWS Lambda (arm64) behind API Gateway, uses Redis 7.2 for a short-lived fallback cache, and uses `httpx` for async HTTP calls. A circuit breaker from `pybreaker` and exponential backoff via `tenacity` wrap the gateway call.

Latency budgets matter more than any single number. A cold start for a Python Lambda is typically in the low hundreds of milliseconds; an inference call to a hosted endpoint is typically a few hundred milliseconds; a payment call is the least predictable of the three and can range from roughly 150 ms to several seconds depending on the provider and the region. Treat all three as estimates to be measured in your own environment, not as guarantees.

## Step 1 — set up the environment

**Create a virtual environment** with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Install the required libraries.** Pin versions to avoid surprise upgrades:

```bash
pip install fastapi uvicorn[standard] httpx boto3 redis pybreaker tenacity
```

Pin exact versions in your own `requirements.txt`; the package names above are stable, but the versions you should pin depend on your deployment date and your security policy.

**Provision AWS resources** using the AWS CDK. The minimal stack includes:

- A Lambda function (`ai_checkout_handler`) with 256 MB memory and a 5 s timeout.
- An ElastiCache Redis cluster in the same VPC.
- An IAM role that grants `sagemaker:InvokeEndpoint` on the specific endpoint ARN you deploy.

```typescript
// cdk-stack.ts
import * as cdk from 'aws-cdk-lib';
import { Runtime } from 'aws-cdk-lib/aws-lambda';
import { Function } from 'aws-cdk-lib/aws-lambda-nodejs';
const app = new cdk.App();
const stack = new cdk.Stack(app, 'AiCheckoutStack');
new Function(stack, 'AiCheckoutHandler', {
  runtime: Runtime.PYTHON_3_11,
  handler: 'handler.main',
  memorySize: 256,
  timeout: cdk.Duration.seconds(5),
});
```

**Configure environment variables** for the three gateways. Use placeholder values and never commit real keys:

- `MPESA_KEY`, `MPESA_SECRET`
- `PAYSTACK_SECRET`
- `FLUTTERWAVE_SECRET`

**Deploy** the CDK stack:

```bash
cdk deploy --require-approval never
```

## Step 2 — core implementation

The core logic lives in `handler.py`. Each gateway call is wrapped in a retry decorator that respects the provider's documented rate limits (M-Pesa, for example, documents a per-consumer-key request ceiling). Backoff starts at 500 ms and doubles up to 4 s. If all attempts fail, the circuit breaker opens for 30 s and returns a cached fallback response.

```python
# handler.py
import os, json, time
from fastapi import FastAPI, HTTPException
import httpx, redis, boto3
from pybreaker import CircuitBreaker, CircuitBreakerError
from tenacity import retry, stop_after_attempt, wait_exponential

app = FastAPI()

# Redis client for idempotent retry cache
redis_client = redis.StrictRedis(host=os.getenv('REDIS_HOST'), port=6379, db=0)

# SageMaker client
sm_client = boto3.client('sagemaker-runtime', region_name='eu-west-1')

# Circuit breakers per provider
mpesa_cb = CircuitBreaker(fail_max=5, reset_timeout=30)
paystack_cb = CircuitBreaker(fail_max=5, reset_timeout=30)
flutterwave_cb = CircuitBreaker(fail_max=5, reset_timeout=30)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
def call_gateway(url: str, payload: dict, headers: dict):
    response = httpx.post(url, json=payload, headers=headers, timeout=5)
    response.raise_for_status()
    return response.json()

def invoke_ai(user_id: str, amount: float):
    payload = {"user_id": user_id, "amount": amount}
    resp = sm_client.invoke_endpoint(
        EndpointName='ai-recommender',
        Body=json.dumps(payload).encode('utf-8'),
        ContentType='application/json',
        Accept='application/json'
    )
    return json.loads(resp['Body'].read())

@app.post('/checkout')
async def checkout(user_id: str, amount: float, provider: str):
    # 1. AI recommendation
    ai_result = invoke_ai(user_id, amount)

    # 2. Choose provider URL and secret
    if provider == 'mpesa':
        url = 'https://sandbox.safaricom.co.ke/mpesa/v1/transactions'
        secret = os.getenv('MPESA_SECRET')
        cb = mpesa_cb
    elif provider == 'paystack':
        url = 'https://api.paystack.co/transaction/initialize'
        secret = os.getenv('PAYSTACK_SECRET')
        cb = paystack_cb
    elif provider == 'flutterwave':
        url = 'https://api.flutterwave.com/v3/payments'
        secret = os.getenv('FLUTTERWAVE_SECRET')
        cb = flutterwave_cb
    else:
        raise HTTPException(status_code=400, detail='Unsupported provider')

    headers = {'Authorization': f'Bearer {secret}', 'Content-Type': 'application/json'}
    payload = {'amount': amount, 'email': ai_result['email']}

    try:
        # 3. Protected call with circuit breaker
        with cb:
            payment_resp = call_gateway(url, payload, headers)
        fallback = False
    except (httpx.HTTPError, CircuitBreakerError) as exc:
        # 4. Fallback path - store intent for later reconciliation
        redis_client.setex(f'fallback:{user_id}:{provider}', 300, json.dumps(payload))
        payment_resp = {'status': 'pending', 'reason': str(exc)}
        fallback = True

    return {
        'ai': ai_result,
        'payment': payment_resp,
        'fallback': fallback
    }
```

**Why this structure helps.** The `retry` decorator absorbs transient network glitches without exhausting the Lambda's timeout. The circuit breaker stops a failing provider from consuming the entire request budget on every call, which is the common failure mode during a regional outage. The Redis fallback preserves the intent to pay so it can be reconciled once the provider recovers.

**Where this structure is weak.** The retry decorator wraps a non-idempotent POST. If the first attempt actually reached the provider and only the response was lost, a retry can create a second charge. That is why the idempotency key in Step 3 is not optional.

## Step 3 — handle edge cases and errors

### 1. Provider-specific error payloads

M-Pesa returns JSON with `errorCode` and `errorMessage`. Paystack uses HTTP 402 for insufficient funds and returns a `status` boolean with a `message`. Flutterwave nests the error under `data.status`. A common gotcha is treating any non-2xx as a generic `HTTPError`; you lose the granular code that tells you whether to retry or to abort. Map it explicitly:

```python
def map_error(provider: str, resp_json: dict):
    if provider == 'mpesa':
        if resp_json.get('errorCode') == '500.001.1001':
            return 'temporary_unavailable'
    elif provider == 'paystack':
        if resp_json.get('status') is False and resp_json.get('message') == 'Insufficient funds':
            return 'permanent_failure'
    elif provider == 'flutterwave':
        if resp_json.get('status') == 'error' and resp_json.get('message') == 'Network error':
            return 'temporary_unavailable'
    return 'unknown'
```

The exact error codes above are illustrative of the shape of each provider's response; verify the current codes against each provider's API documentation before relying on them for retry decisions.

### 2. Idempotency keys

All three gateways support an idempotency header. Without one, a retry after a timeout risks a double charge. The pattern is to derive a deterministic key from the transaction intent — for example, a SHA-256 of `user_id + provider + a client-generated request ID` — store it in Redis with a TTL, and send it as the idempotency header. The critical detail is that the key must be generated **before** the first attempt and reused on every retry; generating it inside the retry loop defeats the purpose.

### 3. Clock skew between Lambda and provider clocks

Some providers validate request timestamps within a short window. Lambda's clock is generally synchronized, but a function running in a custom VPC behind a NAT gateway can drift. The defensive fix is a small buffer:

```python
timestamp = int(time.time()) - 2
```

This is a subtle gotcha that only shows up under load, when the function is under CPU pressure and the time between constructing the payload and signing it grows.

### 4. Rate-limit back-pressure

Exceeding a provider's request ceiling returns HTTP 429. Exponential backoff alone is not enough here: if every caller retries at the same backoff schedule, you get a synchronized retry storm that keeps you pinned at the limit. Add client-side throttling with a token bucket so the call rate is capped before it reaches the gateway. The tradeoff is that a token bucket adds queueing latency under load — you are trading a small, predictable delay for a much larger, unpredictable one.

### 5. The ambiguous outcome

The hardest case is neither success nor failure: the provider accepted the request but the response never arrived. Your service must not report success, and it must not silently drop the intent. The fallback cache plus a reconciliation job (see the final section) is what makes this survivable.

## Step 4 — add observability and tests

### Metrics

Publish three custom metrics, and instrument them with a timer rather than a fixed value:

1. `PaymentSuccess` — incremented on a confirmed charge.
2. `PaymentFallback` — incremented when a fallback entry is written.
3. `PaymentLatencyMs` — the elapsed time of the HTTP call, recorded as a histogram.

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

def emit_metric(name, value, unit='Count'):
    cloudwatch.put_metric_data(
        Namespace='AiCheckout',
        MetricData=[{'MetricName': name, 'Value': value, 'Unit': unit}]
    )
```

### How to measure this yourself

Rather than trusting any published benchmark, run your own load test and record the following. Use a load generator such as `k6` or `vegeta` against a staging deployment, with each gateway's sandbox endpoint mocked or pointed at a test account:

- **Success rate** — successful charges divided by total attempts.
- **Fallback rate** — fallback writes divided by total attempts.
- **Latency distribution** — p50, p95, p99 of the payment call alone, excluding AI inference, so you can attribute the tail correctly.
- **Error mix** — count by mapped error class (`temporary_unavailable`, `permanent_failure`, `unknown`).

Run the test twice: once with client-side throttling enabled and once disabled. The difference in fallback rate and p99 latency tells you whether the token bucket is earning its keep in your environment. Do not copy a fallback rate from someone else's blog post; the number depends on your provider, region, and traffic shape.

### Unit tests with pytest

Mock the three providers with `respx` to return deterministic payloads:

```python
# test_handler.py
import pytest, httpx, json
from fastapi.testclient import TestClient
from handler import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_gateways(respx_mock):
    # M-Pesa success
    respx_mock.post('https://sandbox.safaricom.co.ke/mpesa/v1/transactions').mock(
        return_value=httpx.Response(200, json={'ResponseCode': '0', 'ResponseDesc': 'Success'})
    )
    # Paystack failure
    respx_mock.post('https://api.paystack.co/transaction/initialize').mock(
        return_value=httpx.Response(402, json={'status': False, 'message': 'Insufficient funds'})
    )
    # Flutterwave timeout
    respx_mock.post('https://api.flutterwave.com/v3/payments').mock(
        side_effect=httpx.ConnectTimeout('timeout')
    )

def test_successful_mpesa():
    resp = client.post('/checkout', json={'user_id': 'u123', 'amount': 1500, 'provider': 'mpesa'})
    data = resp.json()
    assert resp.status_code == 200
    assert data['fallback'] is False
    assert data['payment']['ResponseCode'] == '0'

def test_paystack_fallback():
    resp = client.post('/checkout', json={'user_id': 'u124', 'amount': 2000, 'provider': 'paystack'})
    data = resp.json()
    assert data['fallback'] is True
    assert data['payment']['status'] == 'pending'
```

`pytest -q` should report all tests passing. The exact count and runtime depend on your suite.

### Structured logging

Structured logs make the "temporary_unavailable" pattern visible in CloudWatch Logs Insights:

```python
import structlog
log = structlog.get_logger()
log.info('payment_attempt', provider=provider, user=user_id, latency_ms=latency)
```

A typical query:

```
fields @timestamp, @message
| filter @message like /temporary_unavailable/
| stats count() by provider, bin(5m)
```

## Common failure modes and variations

### Variation: batch AI calls

If you need to score a cart of 10 items, batch the payload to the inference endpoint. Endpoints typically accept requests up to a documented size limit, and latency grows with batch size. Adjust the Lambda memory upward if CPU-bound preprocessing becomes the bottleneck.

### Variation: orchestration with Step Functions

For high-value transactions, a Step Functions state machine that separates inference, payment, and reconciliation into distinct Lambda tasks gives you visual retry policies and an audit trail. The tradeoff is added per-transition overhead and a more complex deployment surface.

### Variation: server-side idempotency

Some teams store a transaction hash in DynamoDB with a conditional write (`ConditionExpression = attribute_not_exists(pk)`). This removes the Redis dependency but adds write latency per attempt and requires careful handling of the conditional-check failure path.

## Frequently Asked Questions

**How can I test payment provider failures locally?**

Use `respx` (or `nock` for Node) to mock HTTP responses. Simulate timeouts with `httpx.ConnectTimeout` and rate-limit errors with a 429 payload. Keep mock definitions in a `tests/mocks.py` file and import them in your pytest fixtures.

**Why does a Lambda sometimes exceed its configured timeout even with retries?**

Retries multiply the request budget. With a 500 ms initial backoff doubling to 4 s, three attempts can consume several seconds of wall clock before the payment call even returns. Add the inference call and you can exceed a 5 s timeout. The fix is a hard per-attempt timeout on the HTTP client (for example, `httpx.Timeout(3.0)`) plus a circuit breaker that opens before the budget is exhausted, so the request fails fast and predictably rather than being killed mid-flight.

**What is the safest way to store the fallback payload?**

Redis with a short TTL is cheap and fast but volatile. For regulatory compliance, also write the payload to object storage with server-side encryption and a lifecycle rule that expires it after the retention period your jurisdiction requires.

**When should I move from Lambda to a container service for this workload?**

If a large share of invocations are hitting the timeout limit, or you need more memory than the Lambda maximum for large inference batches, a container service gives you predictable CPU and memory without cold starts. The tradeoff is a higher baseline cost and more operational surface area.

## Where to go from here

You now have a checkout endpoint that tolerates the most common payment gateway failure modes and, more importantly, knows when it does not know the outcome. The next step is to close the loop: automate reconciliation of the fallback entries.

**Actionable next step (30 minutes):** write a reconciliation script that scans the `fallback:*` keys, re-issues each payment with the same idempotency key, and records the final status. Run it locally against a Redis instance seeded with one fake fallback key to confirm the retry path works end to end.

```python
# reconcile.py
import os, json, redis, httpx
r = redis.StrictRedis(host=os.getenv('REDIS_HOST'), port=6379, db=0)
for key in r.scan_iter('fallback:*'):
    payload = json.loads(r.get(key))
    # pick provider from key name
    provider = key.decode().split(':')[2]
    # reuse the same call_gateway logic from handler.py
    # (omitted for brevity)
    print(f'Retrying {provider} for {payload}')
```

The skeleton above is deliberately incomplete: the real value is in reusing the same idempotency key and error-mapping logic from `handler.py`, so that a reconciled retry behaves identically to a first attempt.
