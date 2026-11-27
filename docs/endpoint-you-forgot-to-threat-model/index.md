# endpoint you forgot to threat model

A new API endpoint usually starts as a route, a handler, and a happy-path test. It ships. Weeks later, someone notices that the endpoint accepts an `X-Forwarded-For` header without validation, or that it returns a stack trace on malformed JSON, or that its rate limit is per-IP but the client is behind a shared NAT. None of these are exotic vulnerabilities. They are the predictable result of treating threat modeling as a pre-launch ceremony rather than a design activity.

The real problem is not that teams ignore security. It is that threat modeling happens too late to change the design, so it becomes a list of tickets that get deprioritized. This post walks through a threat-modeling pass on a single endpoint — a webhook receiver that accepts JSON payloads from a third-party payment processor — and shows how the model changes the implementation before a line of production code is written.

## The endpoint: a webhook receiver that trusts too much

Consider a common shape. A SaaS product lets customers connect a payment provider. The provider sends events to `POST /webhooks/payments`. The handler needs to:

1. Verify the request came from the provider.
2. Parse the JSON body.
3. Look up the customer by an identifier in the payload.
4. Update subscription state.
5. Return `200 OK` quickly so the provider does not retry.

A typical first implementation looks like this:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/webhooks/payments")
def handle_payment_webhook():
    payload = request.get_json(force=True)
    customer_id = payload["data"]["customer"]
    event_type = payload["type"]
    if event_type == "invoice.paid":
        mark_invoice_paid(customer_id, payload["data"]["invoice_id"])
    return jsonify({"received": True})
```

This code has at least five problems that a threat model would surface before deployment:

- No signature verification. Anyone who knows the URL can send a fake `invoice.paid` event.
- `force=True` accepts any content type, including form-encoded data, and bypasses the `Content-Type` check.
- No limit on body size. A large payload can exhaust memory or disk.
- No idempotency. A retried event can double-apply a state change.
- No timeout or async boundary. If `mark_invoice_paid` is slow, the provider times out and retries, amplifying load.

None of these are theoretical. They are the default behavior of a handler written without a threat model. The fix is not to add a WAF and hope. It is to enumerate the trust boundaries and the assets, then decide which controls belong in the endpoint itself.

## Trust boundaries: where the attacker actually enters

Threat modeling frameworks like STRIDE give you a vocabulary, but the useful output is a diagram with lines. For a webhook receiver, the boundaries are:

- **The public internet to your load balancer.** The attacker controls the request method, headers, body, and timing.
- **The load balancer to your application.** If you terminate TLS at the edge and forward plain HTTP, the application cannot distinguish a request from the edge from a direct request unless you enforce network policy.
- **Your application to the database.** The handler writes state based on unverified input.
- **Your application to downstream services.** A slow database or a synchronous email send turns a webhook into a denial-of-service vector.

A common mistake is to model only the first boundary. Teams add a signature check and consider the endpoint secure, while leaving the application-to-database boundary unprotected against replay and the application-to-downstream boundary unprotected against retry storms.

A second mistake is to model the endpoint in isolation. The webhook receiver shares a database connection pool with the rest of the application. If the receiver holds connections while waiting on a slow downstream call, it starves the rest of the system. The threat is not just data corruption; it is availability of unrelated features.

## The mechanism at fault: implicit trust in the request body

The core mechanism behind most webhook vulnerabilities is implicit trust in the request body. The handler parses JSON and uses fields directly as identifiers, amounts, and state transitions. There is no schema, no signature, and no separation between "the provider said this" and "we believe it."

A concrete failure mode: an attacker sends a valid-looking payload with `"type": "invoice.paid"` and a `customer_id` belonging to another tenant. If the handler looks up the customer by that ID and marks the invoice paid without checking that the invoice belongs to that customer, the attacker has performed a cross-tenant state change. The signature check would prevent this if it were present, but signature checks are often added after the first incident, not before.

Another failure mode is replay. Even with a signature, a valid signed payload can be replayed if the handler does not track event IDs. The provider may sign each event, but the signature does not include a nonce or timestamp that the receiver validates. An attacker who captures one signed request can resend it indefinitely, and the handler will apply the state change each time.

A third failure mode is resource exhaustion through parsing. JSON parsers differ in how they handle deeply nested structures. A payload with thousands of nested arrays can cause a stack overflow or excessive CPU in some parsers. The mitigation is a depth limit and a body size limit, not a larger timeout.

## Designing controls into the handler

The fix is to make the trust boundary explicit in code. That means four controls, in order:

1. **Authenticate the sender** with a shared secret and an HMAC over the raw body, compared in constant time.
2. **Validate the envelope** — content type, body size, JSON depth — before parsing.
3. **Deduplicate** by event ID using a unique constraint in the database, not an in-memory set.
4. **Decouple** the state change from the response with a queue or a transactional outbox.

Here is a minimal implementation that applies all four. It uses Flask and Redis for deduplication, but the pattern applies to any stack.

```python
import hashlib
import hmac
import json
import os
import time
from flask import Flask, request, jsonify, abort
import redis

app = Flask(__name__)
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"].encode()
MAX_BODY_BYTES = 64 * 1024
MAX_JSON_DEPTH = 8


def verify_signature(raw_body: bytes, signature_header: str) -> bool:
    if not signature_header:
        return False
    expected = hmac.new(WEBHOOK_SECRET, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def json_depth(obj, current=0):
    if isinstance(obj, dict):
        return max((json_depth(v, current + 1) for v in obj.values()), default=current)
    if isinstance(obj, list):
        return max((json_depth(v, current + 1) for v in obj), default=current)
    return current


@app.post("/webhooks/payments")
def handle_payment_webhook():
    raw_body = request.get_data(cache=True)
    if len(raw_body) > MAX_BODY_BYTES:
        abort(413)
    signature = request.headers.get("X-Signature", "")
    if not verify_signature(raw_body, signature):
        abort(401)
    if request.content_type != "application/json":
        abort(415)
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        abort(400)
    if json_depth(payload) > MAX_JSON_DEPTH:
        abort(400)
    event_id = payload.get("id")
    if not event_id:
        abort(400)
    # Deduplicate for 24 hours; provider retries within that window.
    if not r.set(f"webhook:{event_id}", "1", nx=True, ex=86400):
        return jsonify({"received": True, "duplicate": True})
    enqueue_event(payload)
    return jsonify({"received": True})
```

This handler is not complete — `enqueue_event` must be implemented, and the deduplication window must match the provider's retry policy — but it demonstrates the shape. The signature check uses `hmac.compare_digest` to avoid timing leaks. The body size check happens before parsing. The depth check happens after parsing but before any business logic. The deduplication uses Redis `SET NX` with a TTL, which is atomic across processes.

The important design decision is that the handler does not perform the state change synchronously. It enqueues the event and returns. That decoupling means a slow database or a downstream outage does not cause the provider to time out and retry, which would amplify load. The queue consumer can apply the state change with its own retry and dead-letter policy.

## What to monitor after the endpoint ships

Threat modeling does not end at deployment. The controls you added produce signals that tell you whether they are working. The metrics worth alerting on are:

- **Signature failures per minute.** A sudden increase means either a misconfigured client or an attacker probing the endpoint. Both warrant investigation.
- **Duplicate event rate.** If the deduplication key is wrong or the TTL is too short, duplicates will be processed. A rising duplicate rate after a provider change is a signal to re-check the event ID field.
- **Queue depth and consumer lag.** If the consumer falls behind, the provider may retry events that are already queued. The deduplication window must cover the maximum lag.
- **4xx rate by status code.** A spike in 413s means clients are sending larger payloads than expected; a spike in 401s means signature verification is failing.

These are not vanity metrics. Each one maps to a control you added. If you cannot tell from your dashboards whether the signature check is rejecting traffic, you have not finished the threat model.

## The broader lesson: threat modeling is a design review, not a security audit

The pattern above generalizes. A threat model for any new endpoint should answer four questions before the first commit:

1. **Who can call this endpoint, and how do we know?** Authentication is not a middleware checkbox; it is a property of the endpoint's contract.
2. **What can a caller control, and what do we do with it?** Every field in the request body is attacker-controlled until validated. That includes headers, query parameters, and the order of JSON keys.
3. **What happens if this endpoint is called twice, ten times, or a thousand times in a second?** Idempotency and rate limiting are design decisions, not operational afterthoughts.
4. **What does this endpoint do when a downstream dependency is slow or down?** The answer determines whether the endpoint is a load amplifier.

A useful exercise is to write the threat model as a table with columns for asset, entry point, threat, control, and detection. The table is short — five to ten rows for a typical endpoint — and it forces the team to name the detection signal for each control. If a control has no detection, it is not a control; it is a hope.

| Asset | Entry point | Threat | Control | Detection |
|---|---|---|---|---|
| Subscription state | Request body `customer_id` | Cross-tenant modification | Verify signature; look up tenant from event, not body | Signature failures; 4xx rate |
| Invoice records | Request body `invoice_id` | Replay of valid event | Deduplicate by event ID with TTL | Duplicate rate |
| Application availability | Request body size | Memory exhaustion | Body size limit; JSON depth limit | 413 rate |
| Downstream services | Synchronous handler | Retry storm | Enqueue and return | Queue depth; consumer lag |

The table is not exhaustive, but it is enough to drive the implementation. The controls are specific, and the detection column tells you what to instrument.

## Frequently Asked Questions

**How do I threat model an API endpoint without a security team?**

Start with the trust boundary diagram and the four questions above. You do not need a formal methodology to identify that a webhook receiver trusts its request body. Write the table, assign each control to a code change, and add the detection signal to your existing dashboards. The exercise takes an hour and surfaces design changes that are cheaper to make before the endpoint ships.

**What is the difference between authentication and signature verification for webhooks?**

Authentication establishes who is calling; signature verification establishes that the payload was not modified in transit and came from someone who knows the shared secret. For webhooks, the signature is the authentication mechanism because the caller is a server, not a user. An API key in a header is weaker because it does not bind the key to the body, so a captured key can be reused with a different payload.

**Should webhook handlers do the work synchronously or enqueue it?**

Enqueue it. The provider's timeout is typically short — often a few seconds — and any synchronous work that exceeds it causes a retry. Retries amplify load and can cause duplicate state changes if the handler is not idempotent. Enqueuing decouples the response time from the work time and gives you a place to apply retries and dead-lettering on your own terms.

**How long should the deduplication window be?**

It must be at least as long as the provider's maximum retry window plus your maximum consumer lag. If the provider retries for 24 hours and your consumer can lag by an hour, a 24-hour TTL is too short. Check the provider's documentation for the retry schedule and set the TTL with margin. The cost of a longer TTL is storage; the cost of a shorter one is duplicate state changes.

## The next 30 minutes: write the trust boundary table for your endpoint

Open the handler for the endpoint you are about to ship. Write a four-column table on a piece of paper or in a scratch file: asset, entry point, threat, control. Fill in one row for every field in the request body that influences state, every header you read, and every downstream call you make synchronously. If a row has no control, that is a design change, not a ticket. If a control has no detection signal, add the metric before you deploy. The table is the threat model; the code follows from it.

## Official documentation

- [Redis documentation](https://redis.io/docs/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
