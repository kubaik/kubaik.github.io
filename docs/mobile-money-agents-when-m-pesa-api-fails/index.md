# Mobile money agents: when M-Pesa API fails

hidden failure is easy to demo and hard to keep honest at scale. It's the kind of problem that's easy to reproduce and hard to explain. This is the version of the write-up that includes the part that broke.

## The error and why it's confusing

You've built an agent that accepts payments in local mobile money — M-Pesa, MTN MoMo, Airtel Money, or one of the dozens of other rails. The integration works in staging. In production, a subset of users pay successfully but never get their agent credits. Or worse: the agent credits them twice. The logs show a timeout, then a success, then a callback that arrives 45 seconds later. The user's phone shows "Payment received" but your database says "pending."

The confusing part is that the error isn't an error at all. The mobile money provider's API returns `HTTP 200 OK` with a body that says `{"status": "pending", "transactionId": "..."}`. Your code treats that as success. Then the final status comes asynchronously — sometimes as a webhook, sometimes as a polling response, sometimes never. The user's payment provider has already debited their wallet. Your agent has not delivered. The user is now in a WhatsApp group asking why they were charged twice.

This is not a bug in your code. It's a fundamental mismatch between the synchronous request-response model most developers build for and the store-and-forward, eventually-consistent reality of mobile money rails. The mobile money network is not a credit card processor. It's a distributed system with intermittent connectivity, human agents who float cash, and settlement windows measured in hours, not milliseconds. The part that trips people up is treating a `pending` response as a terminal state, and that's what this post actually covers.

## What's actually causing it (the real reason, not the surface symptom)

The surface symptom is a timeout or a `pending` status. The real reason is that [mobile money APIs](/mobile-money-agent-fees-why-your-bot-keeps-failing/) are designed for **asynchronous settlement**. When a user initiates a payment, the request goes through several hops:

1. Your server → mobile money aggregator API (e.g., Safaricom Daraja, Flutterwave, Paystack)
2. Aggregator → mobile network operator (MNO) core
3. MNO core → USSD/SIM toolkit session on the user's phone
4. User enters PIN → MNO validates → debits wallet
5. MNO sends confirmation back up the chain

Steps 3–5 can take anywhere from 5 seconds to 2 minutes. The MNO's API gateway will often return a `pending` or `accepted` response immediately after step 1, then deliver the final result via a callback URL or a separate polling endpoint. If your server doesn't handle that callback correctly — or if the callback never arrives because the MNO's retry logic gives up after 3 attempts over 30 seconds — you're left with an orphaned transaction.

A common failure mode: the callback URL is behind an HTTPS endpoint that uses a self-signed certificate or an expired Let's Encrypt cert. The MNO's callback dispatcher fails TLS verification and silently drops the payload. No error reaches your logs. The user's money is gone. Your agent is idle.

Another typical case: you're using a serverless function (AWS Lambda, Cloudflare Workers) with a 10-second timeout. The MNO's callback arrives 15 seconds after the initial request. Lambda has already terminated. The callback hits a cold start and times out again. You see `Task timed out after 10.00 seconds` in CloudWatch, but the transaction remains unresolved.

The third common cause is idempotency. Mobile money networks are notorious for sending duplicate callbacks. If your handler isn't idempotent, you'll credit the user twice for one payment. Then you'll try to reverse the second credit, which triggers a refund flow that takes 3–5 business days. The user sees a debit, a credit, and another debit. They lose trust.

## Fix 1 — the most common cause

**Symptom pattern:** Your logs show `HTTP 200` with `{"status": "pending"}`. No callback ever arrives. The transaction sits in `pending` forever. Users complain they paid but got nothing.

**Cause:** You're treating the initial API response as the final state. You're not polling, and your callback endpoint is either unreachable or not configured.

**Fix:** Implement a two-phase transaction model. Phase 1: accept the payment request, store it as `pending` with the provider's transaction ID. Phase 2: poll the provider's status endpoint every 5 seconds for up to 2 minutes, and also expose a webhook endpoint. Whichever resolves first wins; the other becomes a no-op.

Here's a Python example using `httpx` 0.27 and FastAPI 0.115:

```python
import asyncio
import httpx
from fastapi import FastAPI, Request, BackgroundTasks

app = FastAPI()

async def poll_transaction(tx_id: str, provider_url: str, api_key: str):
    """Poll every 5s for up to 2 minutes."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(24):  # 24 * 5s = 120s
            await asyncio.sleep(5)
            resp = await client.get(
                f"{provider_url}/status/{tx_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            data = resp.json()
            if data["status"] in ("success", "failed"):
                await finalize_transaction(tx_id, data["status"])
                return
    # After 2 minutes, mark as timed out and trigger manual review
    await finalize_transaction(tx_id, "timeout")

@app.post("/initiate-payment")
async def initiate_payment(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    # Call provider to initiate payment
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.provider.com/payments",
            json={"amount": body["amount"], "phone": body["phone"]},
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
    tx = resp.json()
    # Store pending transaction in DB
    await db.execute(
        "INSERT INTO transactions (tx_id, status, amount) VALUES ($1, $2, $3)",
        tx["transactionId"], "pending", body["amount"]
    )
    # Start polling in background
    background_tasks.add_task(
        poll_transaction, tx["transactionId"], "https://api.provider.com", API_KEY
    )
    return {"status": "pending", "tx_id": tx["transactionId"]}

@app.post("/webhook/mobile-money")
async def webhook(request: Request):
    payload = await request.json()
    tx_id = payload["transactionId"]
    status = payload["status"]
    # Idempotent update: only update if still pending
    result = await db.execute(
        "UPDATE transactions SET status = $1 WHERE tx_id = $2 AND status = 'pending'",
        status, tx_id
    )
    if result.rowcount == 0:
        # Already finalized by polling; ignore duplicate
        return {"status": "ignored"}
    return {"status": "ok"}
```

This costs roughly 24 HTTP requests per transaction. At typical provider rates (e.g., $0.002 per status check), that's $0.048 per payment. For a $5 payment, that's ~1% overhead. Acceptable for reliability. If you're processing thousands of payments per hour, batch your polling or use a job queue like Celery 5.4 with Redis 7.2 as the broker.

## Fix 2 — the less obvious cause

**Symptom pattern:** Callbacks arrive but are duplicates. Your database shows two credits for one payment. Users report double charges. Your refund process is manual and slow.

**Cause:** Mobile money networks retry callbacks aggressively. If your endpoint returns a non-2xx status or takes longer than 5 seconds to respond, the provider will retry up to 3 times. If your handler isn't idempotent, each retry creates a new credit.

**Fix:** Make every callback handler idempotent using a unique constraint on the provider's transaction ID. Use a database-level unique index and an `INSERT ... ON CONFLICT DO NOTHING` pattern. Also, respond to callbacks as fast as possible — acknowledge first, process later.

Here's a Node.js 20 LTS example using Express 4.19 and PostgreSQL 16:

```javascript
const express = require('express');
const { Pool } = require('pg');
const app = express();
app.use(express.json());

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

app.post('/webhook/mobile-money', async (req, res) => {
  const { transactionId, status, amount, phone } = req.body;
  // Acknowledge immediately to avoid retries
  res.status(200).json({ received: true });

  // Process asynchronously
  try {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      // Insert with unique constraint on transactionId
      const insertResult = await client.query(
        `INSERT INTO mobile_money_transactions (tx_id, status, amount, phone)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (tx_id) DO NOTHING`,
        [transactionId, status, amount, phone]
      );
      if (insertResult.rowCount === 0) {
        // Duplicate callback, ignore
        await client.query('COMMIT');
        return;
      }
      if (status === 'success') {
        await client.query(
          'UPDATE users SET credits = credits + $1 WHERE phone = $2',
          [amount, phone]
        );
      }
      await client.query('COMMIT');
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  } catch (err) {
    console.error('Webhook processing failed:', err);
    // Don't re-throw; we already sent 200
  }
});
```

This pattern handles up to 3 retries without double-crediting. The unique constraint on `tx_id` is the key. Without it, you're relying on application logic that can race under load.

## Fix 3 — the environment-specific cause

**Symptom pattern:** Works in staging, fails in production. Callbacks never arrive. Your provider dashboard shows "callback failed" or "delivery error." You're running behind a load balancer, API gateway, or CDN.

**Cause:** Mobile money providers often have strict requirements for callback URLs: HTTPS with a valid certificate, no redirects, no authentication headers, and a response within 5 seconds. If you're terminating TLS at an AWS Application Load Balancer (ALB) and your backend expects HTTP, the provider's callback dispatcher might be hitting an HTTP endpoint that redirects to HTTPS. Or your AWS WAF is blocking the provider's IP range.

**Fix:** Verify your callback endpoint is publicly reachable and returns 200 within 5 seconds. Use `curl` from an external network to test. Check that your ALB listener rules don't redirect. If you're using Cloudflare, ensure the provider's IPs are allowlisted in your firewall rules. Also, disable any request body parsing middleware that might reject the provider's content type.

A common trap: you're using AWS Lambda behind API Gateway with a custom domain. API Gateway's default timeout is 29 seconds, but Lambda's is 3 seconds by default. If your handler takes 4 seconds, Lambda times out and API Gateway returns 502. The provider sees a failure and retries. You see `Execution failed due to configuration error: Malformed Lambda proxy response` in CloudWatch. Fix: increase Lambda timeout to 10 seconds and ensure your handler returns a valid response object.

| Provider | Callback timeout | Retry policy | IP allowlist required? |
|----------|------------------|--------------|------------------------|
| Safaricom Daraja | 5 seconds | 3 retries over 30s | Yes (provided in docs) |
| MTN MoMo | 10 seconds | 5 retries over 60s | No, but recommended |
| Airtel Money | 5 seconds | 3 retries over 45s | Yes |
| Flutterwave | 10 seconds | 4 retries over 120s | No |

## How to verify the fix worked

You need three checks: callback delivery, idempotency, and end-to-end latency.

1. **Callback delivery:** Set up a simple logging endpoint that records every incoming request with headers and body. Trigger a test payment from a sandbox phone number. Confirm the callback arrives within 10 seconds. If it doesn't, check your provider's dashboard for delivery errors.

2. **Idempotency:** Send the same callback payload twice manually using `curl`. Confirm your database only has one credit. Check that the second request returns 200 but doesn't modify state.

3. **End-to-end latency:** Measure the time from payment initiation to user credit. For mobile money, typical p50 is 8–15 seconds, p95 is 45–90 seconds. If your p95 is over 2 minutes, you're likely missing callbacks and relying solely on polling. Add more polling workers or investigate callback delivery.

A simple verification script using `curl` and `jq`:

```bash
# Simulate duplicate callback
for i in 1 2; do
  curl -X POST https://your-app.com/webhook/mobile-money \
    -H "Content-Type: application/json" \
    -d '{"transactionId":"test-123","status":"success","amount":100,"phone":"+254700000000"}' \
    -w "\nHTTP %{http_code}\n"
done

# Check database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM mobile_money_transactions WHERE tx_id = 'test-123';"
# Should return 1
```

## How to prevent this from happening again

Prevention is about designing for the failure modes you now know exist. Three practices make the biggest difference:

**1. Always assume callbacks will fail.** Poll as a fallback. Set a maximum polling duration (e.g., 2 minutes) and a maximum number of attempts (e.g., 24). After that, move the transaction to a `needs_review` state and alert your team. Never leave a transaction in `pending` indefinitely.

**2. Use a dead-letter queue for unresolved transactions.** If a transaction times out, push it to an SQS queue or a database table for manual reconciliation. Your support team can then check the provider's dashboard and manually credit or refund. This costs human time but prevents user churn.

**3. Monitor callback success rate as a first-class metric.** Track the percentage of callbacks that arrive within 30 seconds. A healthy rate is above 95%. If it drops below 90%, investigate immediately — it usually means your endpoint is misconfigured or the provider is having an outage. Set up a CloudWatch alarm or a Prometheus alert on this metric.

Also, consider using a payment aggregator that handles these edge cases for you. Services like Flutterwave, Paystack, and Chipper Cash abstract away some of the provider-specific quirks. They cost 1–3% per transaction but can save you weeks of engineering time. For a small team, that tradeoff is often worth it.

## Related errors you might hit next

Once you fix the callback issue, you'll likely encounter these related problems:

- **"Duplicate transaction detected"** — your idempotency key is working, but the provider is sending a new transaction ID for the same payment. This happens when the user retries after a timeout. Fix: deduplicate by phone number + amount + time window (e.g., 5 minutes).

- **"Insufficient balance"** — the user's mobile money wallet doesn't have enough funds. The provider returns this synchronously, but sometimes it arrives as a callback after you've already shown a success message. Fix: don't show success until the final status is confirmed.

- **"Transaction reversed"** — the provider reverses a payment after settlement, often due to fraud checks or user complaints. Your agent has already delivered the service. Fix: implement a reversal handling flow that can claw back credits or suspend the user's account.

- **"Callback signature verification failed"** — you're validating the provider's signature but using the wrong secret or algorithm. Fix: check the provider's docs for the exact HMAC algorithm (usually SHA-256) and ensure you're using the raw request body, not the parsed JSON.

- **"Rate limit exceeded"** — you're polling too aggressively. Most providers limit status checks to 10 per minute per transaction. Fix: back off exponentially. Start with 5-second intervals, then 10, 20, 40.

## When none of these work: escalation path

If you've implemented polling, idempotent callbacks, and verified your endpoint is reachable, but transactions still fail, you've likely hit a provider-specific issue. Here's your escalation path:

1. **Check the provider's status page.** Safaricom, MTN, and Airtel all have public status dashboards. If there's an outage, wait it out and communicate with your users.

2. **Open a support ticket with the provider.** Include the transaction ID, timestamp, and the exact error from your logs. Providers usually respond within 24–48 hours for developer accounts. For enterprise accounts, you may have a dedicated Slack channel.

3. **Test with a different provider.** If you're using a single aggregator, try a direct integration with the MNO. Sometimes the aggregator's own infrastructure is the bottleneck.

4. **Implement a fallback payment method.** Allow users to pay via bank transfer, card, or cash agent. This reduces dependency on a single rail and gives users an alternative when mobile money fails.

5. **Consider a hybrid approach.** Use mobile money for small transactions (< $10) and cards for larger ones. Cards have lower failure rates for high-value payments, though they come with higher fees and chargeback risk.

**Your next step:** Open your production database and run a query to count transactions stuck in `pending` for more than 10 minutes. If that number is greater than zero, you have orphaned payments. Pick one of those transaction IDs and trace it through your logs and the provider's dashboard. That single investigation will tell you whether your problem is callback delivery, idempotency, or provider outage — and that determines which fix to apply first.

## Frequently Asked Questions

**Why does my mobile money integration work in sandbox but fail in production?**

Sandbox environments often simulate instant success and don't enforce callback timeouts or IP allowlists. Production providers have stricter requirements: HTTPS with valid certificates, no redirects, and response times under 5 seconds. Your sandbox callback URL might be HTTP, which works in testing but fails in production. Always test with a staging environment that mirrors production network conditions.

**How do I handle duplicate callbacks from M-Pesa?**

Use a unique constraint on the provider's transaction ID in your database. When a callback arrives, attempt an `INSERT ... ON CONFLICT DO NOTHING`. If the insert affects zero rows, it's a duplicate — acknowledge it with a 200 response but don't process it again. This is the only reliable way to handle duplicates, because application-level checks can race under concurrent load.

**What's the typical latency for mobile money payments?**

For successful payments, p50 latency is 8–15 seconds, p95 is 45–90 seconds. Failures can take longer because the provider retries callbacks multiple times. If your p95 exceeds 2 minutes, you're likely missing callbacks and relying solely on polling. Monitor both callback arrival time and total resolution time to catch this early.

**Should I use a payment aggregator or integrate directly with MNOs?**

Aggregators like Flutterwave and Paystack cost 1–3% per transaction but handle provider-specific quirks, retries, and reconciliation. Direct integration with Safaricom Daraja or MTN MoMo avoids those fees but requires you to build and maintain the reliability logic yourself. For teams with fewer than 5 engineers, aggregators are usually worth the cost. For high-volume merchants, direct integration can save significant fees.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
