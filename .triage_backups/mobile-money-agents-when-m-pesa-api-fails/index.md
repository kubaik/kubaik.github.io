# Mobile money agents: when M-Pesa API fails

Mobile money integrations have a property that makes them hard to test honestly: the happy path works in staging, and the failure path only appears in production, under load, with real money. A payment can return `HTTP 200` with a `pending` status, the user's wallet gets debited, and the agent never delivers. This article covers why that happens and what to build so it stops happening.

## The error and why it's confusing

You've built an agent that accepts payments in local mobile money — M-Pesa, MTN MoMo, Airtel Money, or one of the dozens of other rails. The integration works in staging. In production, a subset of users pay successfully but never get their agent credits. Or worse: the agent credits them twice. The logs show a timeout, then a success, then a callback that arrives 45 seconds later. The user's phone shows "Payment received" but your database says "pending."

The confusing part is that the error isn't an error at all. The mobile money provider's API returns `HTTP 200 OK` with a body that says `{"status": "pending", "transactionId": "..."}`. Your code treats that as success. Then the final status comes asynchronously — sometimes as a webhook, sometimes as a polling response, sometimes never. The user's payment provider has already debited their wallet. Your agent has not delivered.

This is not a bug in your code. It's a fundamental mismatch between the synchronous request-response model most developers build for and the store-and-forward, eventually-consistent reality of mobile money rails. The mobile money network is not a credit card processor. It's a distributed system with intermittent connectivity, human agents who float cash, and settlement windows measured in hours, not milliseconds. The part that trips people up is treating a `pending` response as a terminal state.

## What's actually causing it (the real reason, not the surface symptom)

The surface symptom is a timeout or a `pending` status. The real reason is that mobile money APIs are designed for **asynchronous settlement**. When a user initiates a payment, the request goes through several hops:

1. Your server → mobile money aggregator API (e.g., Safaricom Daraja, Flutterwave, Paystack)
2. Aggregator → mobile network operator (MNO) core
3. MNO core → USSD/SIM toolkit session on the user's phone
4. User enters PIN → MNO validates → debits wallet
5. MNO sends confirmation back up the chain

Steps 3–5 can take anywhere from 5 seconds to 2 minutes. The MNO's API gateway will often return a `pending` or `accepted` response immediately after step 1, then deliver the final result via a callback URL or a separate polling endpoint. If your server doesn't handle that callback correctly — or if the callback never arrives because the MNO's retry logic gives up after a fixed number of attempts — you're left with an orphaned transaction.

A common failure mode: the callback URL is behind an HTTPS endpoint that uses a self-signed certificate or an expired Let's Encrypt cert. The MNO's callback dispatcher fails TLS verification and silently drops the payload. No error reaches your logs. The user's money is gone. Your agent is idle.

Another typical case: you're using a serverless function (AWS Lambda, Cloudflare Workers) with a short timeout. The MNO's callback arrives after the function has already terminated. The callback hits a cold start and times out again. You see `Task timed out after 10.00 seconds` in CloudWatch, but the transaction remains unresolved.

The third common cause is idempotency. Mobile money networks are notorious for sending duplicate callbacks. If your handler isn't idempotent, you'll credit the user twice for one payment. Then you'll try to reverse the second credit, which triggers a refund flow that takes days. The user sees a debit, a credit, and another debit. They lose trust.

## Fix 1 — the most common cause

**Symptom pattern:** Your logs show `HTTP 200` with `{"status": "pending"}`. No callback ever arrives. The transaction sits in `pending` forever. Users complain they paid but got nothing.

**Cause:** You're treating the initial API response as the final state. You're not polling, and your callback endpoint is either unreachable or not configured.

**Fix:** Implement a two-phase transaction model. Phase 1: accept the payment request, store it as `pending` with the provider's transaction ID. Phase 2: poll the provider's status endpoint every 5 seconds for up to 2 minutes, and also expose a webhook endpoint. Whichever resolves first wins; the other becomes a no-op.

Here's a Python example using `httpx` and FastAPI:

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

This costs up to 24 HTTP requests per transaction. The exact per-request price depends on the provider's status-endpoint pricing, which is usually zero or negligible; check the provider's rate card before assuming a cost. If you're processing thousands of payments per hour, batch your polling or move it to a job queue rather than a per-request background task.

## Fix 2 — the less obvious cause

**Symptom pattern:** Callbacks arrive but are duplicates. Your database shows two credits for one payment. Users report double charges. Your refund process is manual and slow.

**Cause:** Mobile money networks retry callbacks aggressively. If your endpoint returns a non-2xx status or takes longer than the provider's timeout to respond, the provider will retry. If your handler isn't idempotent, each retry creates a new credit.

**Fix:** Make every callback handler idempotent using a unique constraint on the provider's transaction ID. Use a database-level unique index and an `INSERT ... ON CONFLICT DO NOTHING` pattern. Also, respond to callbacks as fast as possible — acknowledge first, process later.

Here's a Node.js example using Express and PostgreSQL:

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

The unique constraint on `tx_id` is the key. Without it, you're relying on application logic that can race under load. Note that acknowledging before processing means a crash between the `200` and the database write leaves the transaction unresolved — which is why the polling fallback from Fix 1 still matters.

## Fix 3 — the environment-specific cause

**Symptom pattern:** Works in staging, fails in production. Callbacks never arrive. Your provider dashboard shows "callback failed" or "delivery error." You're running behind a load balancer, API gateway, or CDN.

**Cause:** Mobile money providers often have strict requirements for callback URLs: HTTPS with a valid certificate, no redirects, no authentication headers, and a response within the provider's timeout window. If you're terminating TLS at an AWS Application Load Balancer (ALB) and your backend expects HTTP, the provider's callback dispatcher might be hitting an HTTP endpoint that redirects to HTTPS. Or your AWS WAF is blocking the provider's IP range.

**Fix:** Verify your callback endpoint is publicly reachable and returns 200 within the provider's timeout. Use `curl` from an external network to test. Check that your ALB listener rules don't redirect. If you're using Cloudflare, ensure the provider's IPs are allowlisted in your firewall rules. Also, disable any request body parsing middleware that might reject the provider's content type.

A common trap: you're using AWS Lambda behind API Gateway with a custom domain. API Gateway's default integration timeout is 29 seconds, but Lambda's default timeout is 3 seconds. If your handler takes 4 seconds, Lambda times out and API Gateway returns 502. The provider sees a failure and retries. You see `Execution failed due to configuration error: Malformed Lambda proxy response` in CloudWatch. Fix: increase the Lambda timeout and ensure your handler returns a valid response object.

Callback timeout and retry behavior varies by provider and can change without notice. Rather than hardcoding assumptions, read the current values from each provider's developer documentation and record them in a config file your code reads at startup. A comparison table of specific timeouts and retry counts is only useful if it's regenerated from those docs; treat any static table as stale.

## How to verify the fix worked

You need three checks: callback delivery, idempotency, and end-to-end latency.

1. **Callback delivery:** Set up a logging endpoint that records every incoming request with headers and body. Trigger a test payment from a sandbox phone number. Confirm the callback arrives within the provider's documented window. If it doesn't, check your provider's dashboard for delivery errors.

2. **Idempotency:** Send the same callback payload twice manually using `curl`. Confirm your database only has one credit. Check that the second request returns 200 but doesn't modify state.

3. **End-to-end latency:** Measure the time from payment initiation to user credit. Instrument two timestamps — payment initiated and transaction finalized — and compute the delta per transaction. Report p50 and p95. If p95 exceeds your provider's documented maximum settlement time by a wide margin, you're likely missing callbacks and relying solely on polling. Add more polling workers or investigate callback delivery.

A simple verification script using `curl` and `psql`:

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

**2. Use a dead-letter queue for unresolved transactions.** If a transaction times out, push it to a queue or a database table for manual reconciliation. Your support team can then check the provider's dashboard and manually credit or refund. This costs human time but prevents user churn.

**3. Monitor callback success rate as a first-class metric.** Track the percentage of callbacks that arrive within the provider's window. Establish a baseline from your own traffic, then alert when the rate drops meaningfully below it — a sudden drop usually means your endpoint is misconfigured or the provider is having an outage.

Also, consider using a payment aggregator that handles these edge cases for you. Aggregators abstract away some of the provider-specific quirks. They charge a per-transaction fee, typically a percentage, but can save weeks of engineering time. For a small team, that tradeoff is often worth it.

## Related errors you might hit next

Once you fix the callback issue, you'll likely encounter these related problems:

- **"Duplicate transaction detected"** — your idempotency key is working, but the provider is sending a new transaction ID for the same payment. This happens when the user retries after a timeout. Fix: deduplicate by phone number + amount + time window (e.g., 5 minutes).

- **"Insufficient balance"** — the user's mobile money wallet doesn't have enough funds. The provider returns this synchronously, but sometimes it arrives as a callback after you've already shown a success message. Fix: don't show success until the final status is confirmed.

- **"Transaction reversed"** — the provider reverses a payment after settlement, often due to fraud checks or user complaints. Your agent has already delivered the service. Fix: implement a reversal handling flow that can claw back credits or suspend the user's account.

- **"Callback signature verification failed"** — you're validating the provider's signature but using the wrong secret or algorithm. Fix: check the provider's docs for the exact HMAC algorithm and ensure you're using the raw request body, not the parsed JSON.

- **"Rate limit exceeded"** — you're polling too aggressively. Providers typically limit status checks per transaction per minute. Fix: back off exponentially. Start with 5-second intervals, then 10, 20, 40.

## When none of these work: escalation path

If you've implemented polling, idempotent callbacks, and verified your endpoint is reachable, but transactions still fail, you've likely hit a provider-specific issue. Here's your escalation path:

1. **Check the provider's status page.** Safaricom, MTN, and Airtel all have public status dashboards. If there's an outage, wait it out and communicate with your users.

2. **Open a support ticket with the provider.** Include the transaction ID, timestamp, and the exact error from your logs. Response times vary by account tier.

3. **Test with a different provider.** If you're using a single aggregator, try a direct integration with the MNO. Sometimes the aggregator's own infrastructure is the bottleneck.

4. **Implement a fallback payment method.** Allow users to pay via bank transfer, card, or cash agent. This reduces dependency on a single rail and gives users an alternative when mobile money fails.

5. **Consider a hybrid approach.** Use mobile money for small transactions and cards for larger ones. Cards have different failure characteristics for high-value payments, though they come with higher fees and chargeback risk.

## Decision checklist before you ship

- [ ] Initial API response is stored as `pending`, never as final.
- [ ] A polling loop with a bounded attempt count runs for every initiated transaction.
- [ ] The webhook handler is idempotent via a database unique constraint on the provider transaction ID.
- [ ] The webhook acknowledges before processing, and the polling fallback covers the gap if processing crashes.
- [ ] The callback URL is HTTPS, has a valid certificate, does not redirect, and responds within the provider's timeout.
- [ ] Lambda/function timeouts exceed the provider's callback timeout.
- [ ] A `needs_review` state and dead-letter path exist for transactions that never resolve.
- [ ] Callback arrival rate and end-to-end latency are instrumented and alerted on.

**Your next step:** Open your production database and run a query to count transactions stuck in `pending` for more than 10 minutes. If that number is greater than zero, you have orphaned payments. Pick one of those transaction IDs and trace it through your logs and the provider's dashboard. That single investigation will tell you whether your problem is callback delivery, idempotency, or provider outage — and that determines which fix to apply first.
