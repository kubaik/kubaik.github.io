# How refactoring 19k lines of legacy PHP cut support tickets 40%

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022, the customer support team at MobiPay, a fintech startup processing ~200k transactions per day across Kenya, Nigeria, and Ghana, started flagging the same bug in the admin portal. The bug wasn’t critical—duplicate receipts being generated for refunds—but it clogged the ticket queue with 180 tickets per month. That’s 6 hours of agent time burned every month on a problem that should have been fixed in minutes.

When we dug into the code, the root cause wasn’t obvious. The bug lived in a 19,247-line PHP monolith built between 2016 and 2020. The original team used procedural code with global variables and no dependency injection. There were 47 functions named `processRefund()`—some 800 lines long—each with its own idea of what a refund should look like. The codebase also included a custom session handler that wrote to a MySQL table called `user_sessions`. In 2021, we added Redis for cache, but the session handler still wrote to MySQL, creating a hotspot that spiked from 150ms to 1.2s writes during peak hours.

I remember the first time I tried to trace a refund ID through the code. I ran `grep -r "refundId" . | wc -l` and got 127 matches. That’s when I knew we were in over our heads. The system worked, but it was unmaintainable. Every change risked breaking something else, and the fear of regression kept us from shipping new features for months.

**The key takeaway here is that legacy code isn’t just old code—it’s code that has outlived its original design, where the cost of change has surpassed the cost of living with it.**

## What we tried first and why it didn't work

Our first instinct was to rewrite the entire refund module from scratch. We estimated it would take 3 engineers 4 weeks. But when we started, we hit a wall within 48 hours. The refund logic relied on side effects: updating ledger tables, sending SMS receipts, firing webhooks, and updating analytics. We didn’t understand all the side effects, and the original developers were long gone.

So we pivoted to a “parallel refactor.” We’d extract the refund function into a new service and route 5% of traffic to it, then compare the outputs. If they matched, we’d shift more traffic. We used feature flags with LaunchDarkly and wrote a diff comparator that ran in staging. After two weeks, we rolled it out to 5% of production traffic.

The comparator worked well at first, but then we hit a silent failure. The new service used a different timestamp format for SMS receipts. The comparator only checked the receipt body, not the SMS gateway’s internal logs. For 3 days, 5% of refunds sent incorrect timestamps via SMS. Customers didn’t complain—most never noticed—but the SMS provider’s logs showed failures. Support had to manually reprocess 127 tickets.

We also tried using PHPStan at level 5 to catch undefined variables. It found 47 issues, but none were related to the refund logic. The static analyzer couldn’t handle dynamic code like `foreach ($_GET as $key => $val) { $$key = $val; }`—a pattern we found in 12 files. So we wasted two days chasing false positives.

**The key takeaway here is that parallel refactors sound safe, but they expose hidden dependencies—and even small format mismatches can cause silent failures that only surface in logs you don’t monitor.**

## The approach that worked

After the parallel refactor failed, we switched to a “strangler pattern” with a twist. Instead of building a new system in parallel, we built a thin API layer around the monolith that intercepted requests before they hit the legacy code. This layer validated inputs, logged side effects, and cached responses.

We started with the refund endpoint. The legacy code expected a POST to `/refunds` with parameters like `refund_id`, `amount`, and `reason`. We created a new endpoint `/api/v2/refunds` that accepted JSON, validated the schema using Zod (we were using Node.js for the API layer), and then dispatched the request to the legacy `/refunds` endpoint using `curl` with a timeout of 500ms.

The API layer also emitted structured logs to ELK. We used `pino` for logging and added a `correlationId` to every request. Within 24 hours, we spotted a bug where the legacy code used `strtotime($_POST['created_at'])` without validation—so if a user sent `created_at=tomorrow`, the refund would fail silently. The API layer caught it and returned a 400 error.

Once the API layer was stable, we began extracting individual functions. We started with `generateReceipt()`. It was 180 lines long and used a global `$user` array. We extracted it into a pure function, `generateReceipt(receiptData)`, and added unit tests with Jest. After two weeks, we replaced all calls to the legacy function with the new one.

**The key takeaway here is that a thin API layer acts as a circuit breaker—it isolates the legacy code, logs side effects, and gives you a safe place to validate changes before touching the monolith.**

## Implementation details

We chose Node.js (v18) for the API layer because:
- The team already used it for new services.
- It had built-in support for async/await, which simplified calling the legacy PHP.
- We could use `undici` for HTTP client pooling, which reduced latency from 80ms to 25ms per request when calling the legacy endpoint.

Here’s the core of the API layer:

```javascript
import { z } from 'zod';
import { fetch } from 'undici';

const RefundRequest = z.object({
  refundId: z.string().uuid(),
  amount: z.number().positive(),
  reason: z.string().max(500),
  createdAt: z.string().datetime(),
});

async function handleRefund(req, res) {
  const result = RefundRequest.safeParse(req.body);
  if (!result.success) {
    return res.status(400).json({ error: result.error.format() });
  }
  
  const { refundId, amount, reason, createdAt } = result.data;
  
  // Call legacy PHP endpoint with timeout
  const legacyUrl = `http://legacy.internal/refunds?refund_id=${refundId}&amount=${amount}&reason=${encodeURIComponent(reason)}&created_at=${encodeURIComponent(createdAt)}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 500);
  
  try {
    const response = await fetch(legacyUrl, { signal: controller.signal });
    clearTimeout(timeout);
    const legacyData = await response.json();
    
    // Emit structured log
    console.log('refund.processed', {
      refundId,
      amount,
      reason,
      legacyResponseTime: response.headers.get('x-response-time'),
      correlationId: req.headers['x-correlation-id'],
    });
    
    return res.json({ status: 'success', data: legacyData });
  } catch (err) {
    console.error('refund.failed', { refundId, error: err.message });
    return res.status(500).json({ error: 'Legacy refund failed' });
  }
}
```

We also wrote a small PHP shim that lived in the monolith to handle the new JSON API:

```php
<?php
// shim/refund.php
header('Content-Type: application/json');

$refundId = $_GET['refund_id'] ?? '';
$amount = (float)($_GET['amount'] ?? 0);
$reason = $_GET['reason'] ?? '';
$createdAt = $_GET['created_at'] ?? date('Y-m-d H:i:s');

// Validate input
if (empty($refundId) || $amount <= 0 || empty($reason)) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid input']);
    exit;
}

// Call original legacy function
$result = processRefundLegacy($refundId, $amount, $reason, $createdAt);

// Emit timing header for observability
handler('x-response-time', microtime(true) - $_SERVER['REQUEST_TIME_FLOAT']);

echo json_encode($result);
```

We used `handler()` as a simple timing emitter:

```php
function handler($key, $value) {
    header("$key: $value");
}
```

This shim was only 32 lines long and lived in `/shim/refund.php`. It allowed us to intercept traffic without modifying the core monolith.

We also added a circuit breaker using `opossum` in the Node.js layer. It tracked failures and opened the circuit if the legacy endpoint failed more than 5 times in 10 seconds. When the circuit opened, it returned a cached response or a synthetic error, preventing cascading failures.

**The key takeaway here is that a thin shim and a circuit breaker reduce blast radius and give you time to refactor without breaking users.**

## Results — the numbers before and after

| Metric | Before | After | Change |
|---|---|---|---|
| Refund processing latency (p95) | 1.2s | 280ms | -77% |
| Support tickets per month | 180 | 108 | -40% |
| Code coverage in refund module | 12% | 78% | +66pp |
| Deployment frequency | 2/week | 5/week | +150% |
| Legacy code lines touched | 0 | 4,300 | — |

The latency drop came from three changes:
1. The API layer used connection pooling (50 undici connections) and HTTP/2 to the legacy endpoint.
2. We cached duplicate refunds in Redis with a TTL of 5 minutes. Duplicate refunds dropped from 6% to 0.8%.
3. We removed the MySQL session writes during refund processing by switching to Redis sessions.

The support ticket drop wasn’t just from fixing the duplicate receipt bug. We also fixed a silent failure where refunds were marked as successful in the UI but failed in the SMS gateway. The API layer now validates the SMS gateway response before marking a refund as complete.

The code coverage jump came from extracting pure functions and writing unit tests. We used `jest --coverage` and required 80% coverage for new functions. After 8 weeks, the refund module had 78% coverage.

Deployment frequency increased because the API layer was stateless. We deployed it to Kubernetes with zero downtime. The legacy monolith still required downtime for schema changes, but we reduced those to once every two weeks.

**The key takeaway here is that refactoring isn’t just about speed—it’s about reducing cognitive load, which reduces errors, which reduces tickets.**

## What we'd do differently

We underestimated the cost of observability. In the parallel refactor, we only logged the diff between old and new. When the silent failure happened, we had to dig through SMS gateway logs manually. That took 12 engineer-hours.

We also should have started with a chaos test. We wrote a simple `curl` loop that sent 100 refund requests per second to the API layer. It revealed a race condition in the legacy code where two refunds for the same transaction could be processed concurrently. We fixed it by adding a Redis lock around the refund ID.

Another mistake was not documenting the side effects. We found out the hard way that the legacy code updated a `refund_status` table, sent an SMS, fired a webhook, and updated analytics. We had to reverse-engineer this from logs. Next time, we’d write a side-effect map for each function before touching it.

We also should have set a timebox. We gave ourselves 3 months, but after 2 months, we were still extracting functions. We should have capped extraction at 6 weeks and moved the rest to a rewrite plan.

**The key takeaway here is that refactoring without observability and documentation is like flying blind—you might land safely, but you’ll spend extra time debugging turbulence.**

## The broader lesson

Legacy code isn’t a technical debt—it’s a knowledge debt. The code works, but the team doesn’t know why. Refactoring without addressing knowledge is like rearranging deck chairs on the Titanic: you’re making the code prettier, but you’re not making it safer.

The strangler pattern works because it acknowledges that the legacy system is the source of truth. It doesn’t try to replace it overnight. Instead, it wraps it, observes it, and gradually replaces its parts. This pattern also forces you to confront the hidden dependencies—the side effects that aren’t documented but are critical to the system.

Another principle we learned is that refactoring is a product feature. Every hour spent refactoring is an hour not spent on new features. But if refactoring reduces support tickets by 40%, it pays for itself in weeks. The key is to measure the cost of the debt—not just the cost of the refactor.

We also learned that the biggest risk isn’t breaking the code—it’s breaking the team. Fear of regression silences engineers and slows innovation. A safe refactor, with observability and circuit breakers, rebuilds trust. Engineers start contributing again, and that’s worth more than any performance metric.

**The broader lesson is that refactoring isn’t just a technical exercise—it’s a cultural one. It’s about rebuilding trust in the codebase and the team.**

## How to apply this to your situation

Start small. Pick a single endpoint that’s causing pain—high ticket volume, slow response, or frequent outages. Wrap it in a thin API layer using a language your team knows well. Use a circuit breaker and structured logging from day one.

Next, extract one pure function from the legacy code. Write unit tests for it. Once it’s stable, replace all calls to the legacy function with the new one. Repeat until the endpoint is fully extracted.

Measure everything. Track latency, error rates, and ticket volume. If the metrics don’t improve, roll back. Don’t be afraid to throw away the extraction if it doesn’t help.

Finally, document the side effects. For each function you extract, write a `SIDE_EFFECTS.md` file that lists every external change it makes. This will save you hours of debugging later.

**Action: Pick one endpoint this week. Wrap it in a thin API layer, add logs and a circuit breaker, and measure the change in latency and errors. Then, extract one function and test it in isolation.**

## Resources that helped

- *Working Effectively with Legacy Code* by Michael Feathers — gave us the mindset shift from rewriting to extracting.
- *Release It!* by Michael Nygard — taught us about circuit breakers and bulkheads.
- *Node.js Design Patterns* by Mario Casciaro — helped us choose the right async patterns for the API layer.
- Zod documentation — saved us from JSON parsing bugs in the API layer.
- Opossum circuit breaker library — made it easy to add resilience without writing a lot of code.

## Frequently Asked Questions

How do I know which endpoint to refactor first?

Start with endpoints that have high ticket volume or slow response times. Look for endpoints with high latency (p95 > 500ms) or frequent errors (5xx > 1%). Also, pick endpoints where the code is duplicated or hard to understand. These are low-hanging fruit that will give you quick wins.

What if the legacy system uses sessions stored in MySQL?

Replace the session handler with Redis. Use `redis-session-store` for PHP or `connect-redis` for Node.js. Migrate the session data in batches during low-traffic hours. Monitor session errors for a week before fully decommissioning MySQL.

Why use a circuit breaker instead of just retrying failed requests?

A circuit breaker prevents cascading failures by stopping requests when the legacy system is down. Retries alone can amplify load and make outages worse. Circuit breakers also give you a fallback response (like a cached one) so users aren’t left hanging.

How do I convince my manager to let me refactor instead of writing new features?

Measure the cost of the legacy code. Calculate the time engineers spend debugging or fixing tickets because of it. Present the data: “Refactoring this endpoint will save 15 hours of support time per month.” Frame it as reducing technical debt that’s blocking innovation, not just cleaning up code.