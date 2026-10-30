# Why do production agents still need humans?

Production agents — LLM-driven chat bots, automated fraud detectors, inventory-balancing microservices — are increasingly the front line for businesses operating in regions with volatile networks and fast-moving regulation. The promise is familiar: 24/7 availability, elastic scaling, sub-second responses measured in a lab. In the field, the same agents meet flaky mobile links, payment-rail quirks, and rules that change without warning. The hidden assumption that trips teams up is that a model can operate safely without a human safety net. This article is about where that assumption fails and how to build the boundary properly.

## The error and why it is confusing

When an agent misclassifies a transaction or drops a user request, the logs often show something generic:

```
Error: Unexpected token in JSON payload (code 500)
```

The natural reading is "parsing bug" or "malformed request" or "transient outage." The symptom is clean; the root cause is not. It can be a corrupted SMS gateway payload on a low-bandwidth link, a business rule that only a human can resolve, or a stale compliance threshold. The same message appears in a fully automated pipeline and in a hybrid pipeline that already has a manual review step. Nothing in the string tells you whether the failure is technical, business-logical, or regulatory.

The practical consequence: teams treat the symptom as a code defect and loop on retries. Retries consume Lambda time, push latency past the timeout, and never resolve the underlying condition, because the underlying condition is not a transient fault.

## What is actually causing it

The recurring root cause is the absence of a well-defined human-in-the-loop (HITL) boundary. Three forces make that boundary necessary:

1. **Network volatility.** On congested mobile links, request timeouts are routine. The timeout itself is not the problem. The problem is that the business rule requires manual verification of the user's identity before proceeding, and the agent has no way to express "I cannot decide this."
2. **Payment-rail idiosyncrasies.** Rails return non-standard error codes for states like "insufficient balance after pending settlement." These codes are often undocumented in the public SDK, so the agent classifies them as generic failures. A human reviewer can check the merchant ledger and approve or reject.
3. **Regulatory flux.** AML thresholds and similar limits change, and they are frequently cached. If the cache is stale, the agent either blocks a legitimate transaction or lets a risky one through. Only a human can override the stale value inside the short window before the cache refreshes.

These forces produce the same surface error, but the fix lives outside the code path that raised it. A proper HITL design isolates the decision point, surfaces the exact failure reason, and routes it to a human operator with enough context to act.

## Fix 1 — the most common cause: blind retries on business errors

**Symptom pattern:** the agent repeatedly retries a failing API call, logs the generic parse error, and the error rate spikes for a sustained window.

**Root cause:** retry logic that is blind to business context. A typical implementation:

```javascript
const axios = require('axios');
const axiosRetry = require('axios-retry');
axiosRetry(axios, { retries: 5, retryDelay: axiosRetry.exponentialDelay });

async function chargeCustomer(payload) {
  return await axios.post('https://api.flutterwave.com/v3/charges', payload);
}
```

When the gateway returns a business-specific code, the retry loop treats it as transient, inflating latency and exhausting the function timeout. The fix is to make the retry policy aware of the error code and hand off to a human when the code is business-specific.

**Steps:**
1. Extend the error handler to inspect `error.response.data.code`.
2. If the code is in a known business-critical list, publish a message to a review queue instead of retrying.
3. Attach the original payload and a correlation ID for traceability.

```python
import json, boto3
sqs = boto3.client('sqs', region_name='us-east-1')
QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/123456789012/human-review-queue'

def handle_error(response):
    code = response.get('code')
    if code in {'MPAY-302', 'FLW-401'}:
        message = {
            'correlation_id': response.get('request_id'),
            'payload': response.get('original_payload')
        }
        sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message))
        return {'status': 'handed_off'}
    else:
        raise RuntimeError('Unexpected error')
```

Routing the error to a human removes the wasteful retry loop and converts an unresolvable failure into a queued decision.

## Fix 2 — the less obvious cause: stale configuration cache

**Symptom pattern:** after a deployment, the agent flags legitimate transactions as fraud, generating a surge of alerts. Logs show no stack trace, only the generic parse error.

**Root cause:** the cache holding AML thresholds is stale. Many teams read it once:

```bash
redis-cli -h redis-prod.example.com GET aml_threshold
```

If the refresh job fails — a missed cron run, a power outage, a network partition — the agent keeps using the old threshold. The mismatch triggers false positives that surface downstream as parsing errors, because a service expecting a numeric field receives a string.

**Fix:** a cache-with-fallback pattern. Read from the cache, but if the value is older than a chosen staleness window, fall back to a durable store. Instrument the cache with a TTL that forces refresh.

A comparison of the two operating modes, with figures labelled illustrative:

| Aspect | Fully automated | Human-in-the-loop |
|--------|----------------|-------------------|
| Latency (illustrative) | ~150 ms | ~200 ms including queue and review |
| Failure rate (illustrative) | ~2.4% | ~0.6% |
| Compliance risk | Higher | Lower |

**Implementation (Node.js):**

```javascript
const Redis = require('ioredis');
const { DynamoDBClient, GetItemCommand } = require('@aws-sdk/client-dynamodb');
const redis = new Redis({ host: 'redis-prod.example.com', port: 6379 });
const ddb = new DynamoDBClient({ region: 'us-east-1' });

async function getAmlThreshold() {
  const cached = await redis.get('aml_threshold');
  const ttl = await redis.ttl('aml_threshold');
  if (cached && ttl > 300) return Number(cached);
  const cmd = new GetItemCommand({ TableName: 'Config', Key: { name: { S: 'aml_threshold' } } });
  const { Item } = await ddb.send(cmd);
  const fresh = Number(Item.value.N);
  await redis.set('aml_threshold', fresh, 'EX', 3600);
  return fresh;
}
```

The TTL guard prevents the agent from acting on stale data, and the fallback guarantees correctness even when the refresh job is missed.

## Fix 3 — the environment-specific cause: intermediaries rewriting the stream

**Symptom pattern:** agents in a specific region start throwing parse errors after a mobile carrier or CDN changes its compression or header behavior. The error appears only on a particular network path.

**Root cause:** a proxy injects a non-standard header or re-encodes the body, and some HTTP clients misread the result — reading the full stream, stripping headers after the fact, or leaving stray bytes in front of the JSON.

**Fix:** switch to a client configuration that disables automatic decompression and tolerates leading bytes.

```python
import requests

session = requests.Session()
# Disable automatic gzip handling
session.headers.update({'Accept-Encoding': 'identity'})

def fetch_payload(url):
    resp = session.get(url, timeout=5)
    resp.raise_for_status()
    # Use a safe json loader that tolerates stray bytes
    return resp.content.lstrip(b'\xef\xbb\xbf').decode('utf-8')
```

This is environment-specific, and that is the point: a universal "no-human" stance fails when network stacks differ, because the failure mode itself differs by path.

## How to verify the fix worked

Do not trust a single dashboard. Instrument the boundary itself and compare before and after on the same traffic.

1. **Metric collection.** Emit custom metrics for `HumanReviewHandOffs`, `CacheStaleHits`, and `ProxyErrorRate`. Alarm when any exceeds a threshold you have chosen relative to total requests.
2. **Trace correlation.** Attach the correlation ID from the queue message to the downstream invocation, and confirm the trace shows a human-review segment. Without this, hand-offs are invisible.
3. **Split traffic.** Deploy the updated retry logic to a fraction of traffic using a function alias or feature flag. Compare error rates between control and variant over the same window.
4. **Load test.** Simulate concurrent users on a throttled network profile and record latency. The number that matters is the p95, not the mean.

**How to measure the specific claims in this article.** Any figure you see quoted for HITL benefits should be reproducible:

- **Retry waste.** Count invocations whose only outcome was a retry of a business-code error. Multiply by the measured average duration and your per-GB-second price. This gives the cost of the loop you removed.
- **Hand-off latency.** Instrument the time between the queue message being written and the reviewer action being recorded. That interval, not the reviewer's click, is what your SLA must absorb.
- **Error-rate change.** Compare the count of the specific error class per thousand requests, before and after, on the same traffic mix. A change in traffic mix will otherwise masquerade as a fix.
- **Stale-cache incidents.** Log every fallback read, then count how many would have used a value older than your staleness window. This is the number that justifies the fallback path.

If those four numbers move in the expected direction, the boundary is working. If only the headline error rate moves, you have probably changed what you log, not what happens.

## How to prevent this from happening again

Prevention starts with **policy as code**. Define a schema listing the error codes that require human review, store it in version control, and load it at startup.

```json
{
  "humanReviewCodes": ["MPAY-302", "FLW-401", "PAYSTACK-999"]
}
```

Combine this with CI checks that reject new agent code which hard-codes a retry-only path for those codes. Then schedule a health check that verifies:

- The cache TTL for the threshold key is above your staleness window.
- The review queue depth is below a threshold you can act on.
- The set of proxy headers observed in production matches the allowed set.

Finally, enforce a runbook that requires sign-off before any change to the HITL schema, so compliance stays in the loop rather than being notified afterward.

## A worked example: deciding where the boundary goes

Suppose a payment agent handles 100,000 requests per day. Of those, 3% fail with a business-specific code. Two designs are on the table.

**Design A — retry everything.** Each business-code failure is retried three times before giving up. That is 100,000 × 0.03 × 3 = 9,000 extra invocations per day. Each invocation costs compute time you would otherwise not spend, and each one delays the user's eventual failure message.

**Design B — route business codes to review.** The same 3,000 requests per day go to a queue. A reviewer handles each in roughly 20 seconds of active work. That is 3,000 × 20 = 60,000 seconds, or about 16.7 hours of reviewer time per day. If that exceeds your staffing, the boundary is drawn too wide: narrow the code list to only the codes that genuinely require judgment, and let the rest fail fast with a clear user-facing message.

The arithmetic is the point. HITL is not free, and the review load scales linearly with traffic. A boundary that works at 10,000 requests per day can collapse at 100,000. Compute the reviewer hours before you ship, not after.

## Failure modes of the boundary itself

A HITL design can fail in ways that are worse than no boundary at all:

- **Queue as a black hole.** Messages are written but nobody owns the queue. Add an alert on queue age, not just depth.
- **Missing context.** A reviewer who sees only a correlation ID cannot decide anything. Ship the payload, the attempted action, and the reason for the hand-off in the same message.
- **Duplicate side effects.** If the agent retries after handing off, the human may approve a transaction that already succeeded. Make the hand-off terminal for that request.
- **Silent fallback.** If the cache fallback fails and the agent proceeds with a default threshold, you have replaced a visible error with an invisible one. Fail closed and alert.
- **Boundary creep.** Every new error code added to the review list increases reviewer load permanently. Review the list on a schedule and remove codes that no longer need judgment.

## When none of this works: escalation path

1. Open a ticket with the logs for the offending request, the correlation ID, and the queue message if one exists.
2. Tag both the on-call engineer and the compliance owner. These are different people with different incentives; do not merge them.
3. Run a diagnostic script that gathers a short network trace, a cache TTL dump, and the function's environment configuration.
4. If the issue persists past your defined investigation window, escalate to the architecture group with the diagnostic bundle attached.

The escalation path exists so that no silent failure stays in production longer than the window your SLA allows.

## Frequently asked questions

**How do I decide which error codes need human review?**
Identify codes that map to business-critical decisions: regulatory blocks, high-value payments, ambiguous fraud signals. Store them in version-controlled schema and review the list on a schedule with compliance.

**Why can't I rely solely on automated retries?**
Retries consume compute and cannot resolve errors that require contextual judgment, such as missing customer documents or outdated AML thresholds.

**What is the minimal latency impact of adding a human step?**
It depends entirely on your queue and staffing. Measure the interval between message write and reviewer action; that is the number to put in the SLA, not an assumed constant.

**When should I fall back to a different payment rail?**
Define a threshold in advance — for example, repeated business-specific errors from one rail inside a short window — and log every fallback for audit.

## The next 30 minutes

Open your agent's error handler and add one branch: if the response code is in a small, explicit set of business-critical codes, publish a message containing the correlation ID, the original payload, and the reason to a queue you already own. Do not change the retry logic yet, and do not add a dashboard. Just make the hand-off visible in production. Once you can see how often it fires, you have the data to decide how wide the boundary should be.
