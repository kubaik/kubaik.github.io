# Portfolio that hires: show constraints, not code

Most portfolio advice assumes a clean environment and a patient timeline. Production gives you neither. A portfolio that gets read by a hiring manager is not the one with the most polished UI; it is the one where the constraints are visible and the trade-offs are written down.

## What a portfolio is actually being judged on

A reviewer scanning a portfolio is trying to answer three questions, usually in under two minutes:

1. Can this person ship something that survives a bad network?
2. Can this person reason about latency, cost and failure under load?
3. Can this person write code and documentation that a team of twenty can maintain without a long onboarding?

A CRUD app with a green Lighthouse score answers none of them. Lighthouse runs on a fast, stable connection and rewards optimizations that often do not matter on mobile networks. A page can score 98 and still freeze for twelve seconds when the connection drops, because Lighthouse never drops the connection.

The fix is not more features. It is making the constraints explicit and measurable, so a reviewer can see the reasoning without having to run the code.

## The failure modes that show up first

### The template stack that passes every linter

The standard Next.js plus a hosted backend template deploys in minutes, looks polished, and passes lint and type checks. A typical failure mode appears when the connection is throttled: the UI blocks on a request that never resolves, and reconnection logic either does not exist or assumes a clean disconnect.

Two things usually cause this:

- A client-side timeout that is longer than the user's patience, so the UI waits instead of degrading.
- Reconnect logic with no backoff, which turns a brief drop into a retry storm that makes recovery slower.

Neither is visible in a Lighthouse report. Both are visible in a throttled load test.

### Serverless cold starts against a tight API budget

A serverless API behind a managed gateway is a reasonable default, but cold starts commonly land in the hundreds of milliseconds. If the API budget is 200 ms p95, the first request after an idle period will blow it.

Provisioned concurrency removes most of the cold-start penalty, but it also changes the cost model from pay-per-request to pay-for-idle-time. That is a legitimate trade-off, but it has to be stated as one. A portfolio that shows the cost of both configurations, and explains which one was chosen and why, is more convincing than one that quietly pays for always-on capacity.

### The AI-generated project description

A generic "AI-powered" project description reads the same as a hundred others. The problem is not that the project is bad; it is that the description contains no evidence of judgement. There is no constraint, no measurement, and no rejected alternative.

The remedy is the same in all three cases: instrument the system, publish the numbers, and write down what you chose not to do.

## The approach: a portfolio as an instrumented product

Build the portfolio as a small production system with four locked constraints:

- **Mobile-first latency budget.** Every endpoint responds within 200 ms p95 under a synthetic 3G profile. Page loads within 400 ms p95.
- **Intermittent-connection tolerance.** Simulate a 5-second drop every 30 seconds and verify the UI recovers within 2 seconds.
- **Cost ceiling.** The whole stack runs on a small shared-CPU instance plus a single cache node, with the monthly cost stated in the README.
- **Maintainability.** A README of roughly 200 lines that explains every caching strategy, database index and retry policy, plus a short post-mortem.

The post-mortem is the part reviewers tend to value most, because it is the only artifact that shows reasoning rather than output. Keep it to about 500 words and cover three things: latency spikes observed, cache misses observed, cost changes observed — each with the fix that was applied.

## Choosing the stack

| Layer | Category | Notes |
|---|---|---|
| Frontend | React framework with server rendering | SSR for API-backed pages; avoid static export if data must be fresh |
| Hosting | Small shared-CPU container host | Scale-to-zero preview environments for CI |
| Cache | Redis-compatible key-value store | Set an explicit eviction policy and maxmemory |
| Database | Managed PostgreSQL | One small instance; add indexes only where a query plan justifies them |
| Load testing | k6 or an equivalent scriptable load tester | Runs in CI on every pull request |
| CI/CD | Hosted CI runner | Runs load tests and unit tests; blocks merge on threshold breach |

Pin versions in the README rather than in the article, because versions move and the reasoning does not.

## Measuring the 3G profile

The critical piece is the throttled load test. The example below uses k6 with a constant-VU scenario and two thresholds: one for page loads and one for API calls.

```javascript
import { check } from 'k6';
import http from 'k6/http';

export const options = {
  scenarios: {
    mobile_profile: {
      executor: 'constant-vus',
      vus: 20,
      duration: '2m',
      tags: { scenario: '3g_mobile' },
    },
  },
  thresholds: {
    // Page loads: 400 ms p95. API calls: 200 ms p95.
    'http_req_duration{scenario:3g_mobile}': ['p(95)<400'],
    'http_req_duration{scenario:3g_mobile,kind:api}': ['p(95)<200'],
  },
};

export default function () {
  const res = http.get('https://example.com/api/projects', {
    tags: { kind: 'api' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 200 ms': (r) => r.timings.duration < 200,
  });
}
```

Two notes on correctness. First, a scenario-level `thresholds` block inside `scenarios` is not how k6 applies thresholds; thresholds belong at the top level of `options`, which is where they are above. Second, k6 does not throttle the network by itself. To emulate a 3G profile you either run k6 behind a network emulator such as `tc netem` on the load generator, or drive the test through a proxy that applies latency and bandwidth limits. The RTT, bandwidth and jitter values belong in the README next to the test, so a reviewer can see the assumptions.

Run this in CI on every pull request. If p95 exceeds the budget, the job fails and the change is blocked until it is fixed.

## Worked example: setting a retry policy

Suppose the observed distribution of request durations on a throttled connection looks like this (illustrative figures, chosen to make the arithmetic visible):

- 50th percentile: 180 ms
- 90th percentile: 420 ms
- 99th percentile: 2.4 s
- Occasional stalls: 6–8 s

A naive retry policy — retry every 250 ms until success — will, in the stall case, issue roughly 24–32 requests over 6–8 seconds. Each of those is a fresh request that competes for the same congested link, which makes recovery slower, not faster.

A better policy: exponential backoff with full jitter, capped, plus a fast-fail.

- Base delay 150 ms, doubling: 150, 300, 600, 1200, 2400 ms.
- Cap at 3 s.
- Full jitter: sleep `random(0, delay)` rather than `delay`.
- Fast-fail: if two consecutive attempts exceed 2 s, stop retrying and surface a degraded state to the UI.

The fast-fail is what keeps the UI responsive. The user sees a cached or partial view within about 2 seconds instead of a spinner for 8.

The same reasoning applies on the server side. A heartbeat that expects a response every second, and forces a reconnect after two missed beats, detects a half-open socket in about 2 seconds rather than waiting for a TCP timeout that can take much longer. This is a small amount of code and it is the difference between a UI that recovers quickly and one that appears hung.

## Failure-mode analysis: cache stampedes

Caching metadata in Redis with an LRU eviction policy works until a traffic spike hits the same uncached endpoint. When many requests miss simultaneously, they all go to the origin, the origin slows down, and the cache fills with short-lived entries that evict the keys you actually wanted to keep.

Three mitigations, in increasing order of complexity:

1. **Short TTLs on volatile keys.** If cached responses have a 30-second TTL, a spike evicts keys that were going to expire anyway rather than hot keys with long TTLs. This pairs well with a `volatile-ttl` eviction policy.
2. **Single-flight.** On a cache miss, only one request per key goes to the origin; the rest wait on the same promise. This is a few lines in most languages and removes the thundering herd entirely.
3. **Local in-memory cache in front of Redis.** A small process-local cache with a very short TTL absorbs repeated reads of the same key within a single instance. The trade-off is staleness and per-instance memory; it is worth it only for read-heavy keys where a few seconds of staleness is acceptable.

The post-mortem should say which of these was chosen and why the others were rejected.

## Handling real payment and messaging APIs

A portfolio that accepts payments or sends messages has to deal with third-party APIs that are themselves unreliable. Two patterns matter.

**Retry with idempotency.** Any write operation that can be retried must carry an idempotency key so a duplicate request does not create a duplicate charge or message. The example below uses a retry-capable HTTP client in Go.

```go
package payments

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/hashicorp/go-retryablehttp"
)

type ChargeRequest struct {
	PhoneNumber    string `json:"phone_number"`
	Amount         string `json:"amount"`
	IdempotencyKey string `json:"idempotency_key"`
}

func (c *Client) Charge(ctx context.Context, req ChargeRequest) (string, error) {
	const url = "https://api.example.com/v1/charges"

	retryClient := retryablehttp.NewClient()
	retryClient.RetryMax = 3
	retryClient.RetryWaitMin = 100 * time.Millisecond
	retryClient.RetryWaitMax = 2 * time.Second
	retryClient.CheckRetry = func(ctx context.Context, resp *http.Response, err error) (bool, error) {
		if resp != nil && (resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests) {
			return true, nil
		}
		return retryablehttp.DefaultRetryPolicy(ctx, resp, err)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("encode charge request: %w", err)
	}

	httpReq, err := retryablehttp.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build charge request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Idempotency-Key", req.IdempotencyKey)

	resp, err := retryClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("charge failed after retries: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("charge failed: %s", resp.Status)
	}

	var result struct {
		ChargeID string `json:"charge_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode charge response: %w", err)
	}

	return result.ChargeID, nil
}
```

Two details matter here. The idempotency key is generated by the caller and must be stable across retries — a timestamp generated inside the retry loop would defeat the purpose. And the retry policy retries on 5xx and 429 but not on 4xx, because a malformed request will fail the same way every time.

**Network-aware payment method selection.** On a slow connection, a card form that requires several round trips is worse than a redirect-based flow. A simple heuristic: measure the connection's effective RTT before rendering the payment UI, and prefer the flow with fewer round trips when RTT is high.

```javascript
async function choosePaymentFlow(amount, email) {
  const rtt = await measureRtt(); // e.g. median of three small HEAD requests
  const slowNetwork = rtt > 200;

  return {
    amount,
    email,
    currency: 'KES',
    // Fewer round trips on a slow link; richer form on a fast one.
    method: slowNetwork ? 'redirect' : 'inline',
  };
}
```

**Edge caching for read paths.** A key-value store at the edge is a good fit for read-heavy metrics that change slowly. The pattern is a cache-aside read with an explicit fallback to origin.

```typescript
export interface Env {
  PORTFOLIO_KV: KVNamespace;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === '/api/metrics') {
      try {
        const cached = await env.PORTFOLIO_KV.get('metrics', { type: 'json' });
        if (cached) {
          return new Response(JSON.stringify(cached), {
            headers: { 'Content-Type': 'application/json' },
          });
        }
      } catch {
        // Fall through to origin on cache error.
      }
    }

    return fetch(request);
  },
};
```

The catch block is deliberate: a cache outage should degrade to origin reads, not to a 500.

## What to measure, and how

Do not publish benchmark tables copied from a blog post. Publish how to reproduce your own. For each metric below, the instrumentation is the deliverable.

| Claim | What to instrument | How to compare |
|---|---|---|
| API latency under load | Request duration histogram, tagged by endpoint | Run the load test before and after the change; compare p50, p95, p99 |
| Cold-start penalty | Time from request received to first byte, for the first request after idle | Issue one request after a 10-minute idle period, repeat 20 times |
| Cache effectiveness | Hit ratio and eviction count from the cache server's own metrics | Compare hit ratio across two eviction policies under the same load |
| Reconnect time | Time from connection drop to first successful request after reconnect | Throttle the connection in the load test and record the gap |
| Cost | Provider's billing page, itemised | State the monthly figure and the assumptions behind it |

A reviewer who can run your load test and reproduce your numbers is far more convinced than one who reads a table.

## A decision checklist

Before publishing, check each of these:

- [ ] Every endpoint has a stated latency budget, and the budget is enforced in CI.
- [ ] The load test runs against a throttled profile, not a local loopback.
- [ ] Retry logic uses backoff with jitter and has a fast-fail path.
- [ ] Write operations carry idempotency keys.
- [ ] The cache has an explicit eviction policy and a stated TTL rationale.
- [ ] The README states the monthly cost and the assumptions behind it.
- [ ] The post-mortem names at least one rejected alternative and why it was rejected.
- [ ] No claim in the README is unsupported by a number you can reproduce.

## FAQ

**Why not just use a Lighthouse score?**
Lighthouse measures a fast, stable connection. It does not drop the connection, does not simulate jitter, and does not exercise cold starts. It is a useful signal for front-end rendering, but it is not a substitute for a throttled load test with enforced thresholds.

**How do I keep the infrastructure cost low while running load tests?**
Separate the load-test target from production. Run the load test in CI against a preview environment that scales to zero when idle, and keep production on a small shared-CPU plan with a single cache node. Avoid always-on replicas and provisioned capacity unless the portfolio is genuinely receiving traffic; those are the line items that change the bill.

**What should the post-mortem contain?**
Roughly 500 words covering latency spikes observed, cache misses observed, and cost changes observed, each with the fix applied. Name the specific trade-off — for example, "switched eviction policy from LRU to volatile-TTL to stop evicting hot keys" — rather than restating the architecture.

**How do I handle intermittent connectivity without building a full offline-first app?**
Start with three cheap mechanisms: exponential backoff with jitter on every client retry, a server-side heartbeat so half-open sockets are detected within a couple of seconds, and a short-TTL cache layer so a brief drop does not trigger a stampede on the origin. These cover most intermittent-connection failures. Reach for a full offline-first sync layer only if the product must accept writes while disconnected.

**Does the portfolio need to be a real product with users?**
No. It needs to be a real system with real constraints and honest measurements. A small app with a documented latency budget, a reproducible load test, and a post-mortem that names rejected alternatives demonstrates more than a larger app with no instrumentation.

## Do this in the next 30 minutes

Pick one endpoint in your portfolio, add a top-level k6 threshold for it (`'http_req_duration{kind:api}': ['p(95)<200']`), run the test once against your deployed environment, and paste the resulting p95 into your README next to the budget. If it fails, you have just found the first entry for your post-mortem.
