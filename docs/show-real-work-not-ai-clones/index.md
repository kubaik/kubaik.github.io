# Show real work, not AI clones

Most portfolio advice assumes a clean environment and a patient timeline. Production gives you neither. Hiring teams that run payment, USSD or mobile-money systems consistently report the same gap: candidates show AI-generated features, but cannot explain how those features behave when the network drops, a callback arrives out of order, or an upstream API starts returning 503s.

This article is about the artifact that closes that gap. It is written for engineers and reviewers who need a portfolio to demonstrate reliability engineering, not framework familiarity.

## The gap in typical portfolios

A recurring pattern in review pipelines: resumes list glossy AI projects — an "M-Pesa clone with sentiment analysis", a "payments dashboard using an LLM chain", a "Stripe-for-Africa API with AI fraud detection". Every candidate claims to ship AI features, but few can explain how their routing logic behaves under real network conditions.

The real problem is not finding AI skills. It is finding engineers who can build reliable systems on unreliable networks. Evidence of that ability looks like:

- Handling 2G/3G connections with 500ms–2s latency spikes
- Recovering when DNS resolves incorrectly during network handovers
- Coping with mobile-money callbacks arriving out of order
- Behaving predictably when an upstream API returns 503s under load

A portfolio cannot just show features. It has to show resilience. Reviewers look for three things:

1. **Connection-aware retries**: does the code handle partial failures gracefully?
2. **Payment flow testing**: was it tested against real network conditions, not only a sandbox?
3. **Latency instrumentation**: were percentiles measured on slow links, not just a fast local machine?

A common failure mode: hardcoded timeouts of 200ms. On a congested 3G link, that is optimistic to the point of guaranteeing failure.

## Why framework-based filtering fails

A natural first filter is to screen for projects using AI frameworks — an LLM orchestration library, a retrieval framework, an agent framework. The assumption is that candidates who used these tools have real AI skills. In practice, that filter selects for tutorial completion, not engineering judgment. Candidates who pass it often cannot explain their system's failure modes.

Asking for GitHub links to production code fares no better. Many candidates send tutorials or boilerplates. Some send a private repo with a single commit and no reviewable history.

Asking for metrics tends to produce silence: no error rates, no latency percentiles, no uptime numbers — just screenshots of generated graphs. A typical exchange: a candidate claims a messaging bot handles 10,000 messages per day, but when asked how webhook retries were tested, the answer is "I ran it on localhost and it worked."

A final attempt is often a short case study: a problem solved, the constraints faced, the trade-offs made. Most responses are a couple hundred words of buzzwords, with no mention of network conditions, payment integrations or mobile-money APIs — the actual problems these systems face daily.

The conclusion reviewers reach is not that AI skills are irrelevant. It is that the filter is measuring the wrong thing.

## The approach that works: the Constraint Resume

Pivot from "show me your AI project" to "show me a system you built that works when things break". Three artifacts carry the weight:

1. **A concrete problem statement with real constraints**
   - Must include network conditions, payment methods, or device specs
2. **Code that proves resilience under constraints**
   - Evidence of retry logic with exponential backoff
   - Circuit breakers or fallbacks for payment failures
   - Latency instrumentation with percentiles, not averages
3. **A post-mortem or case study**
   - Not a success story, but a failure and how it was fixed
   - Must include metrics: error rate, latency spike duration, cost of failure

Call this the "Constraint Resume" model. It does not care about AI frameworks. It cares about shipping under real conditions.

An illustrative example of a strong entry: a USSD system for a dairy cooperative using a mobile-money API, with constraints of 2G network, a 10-second USSD timeout and 5% packet loss; code showing a retry queue with jitter and an SMS fallback; and a post-mortem describing a network outage where the SMS fallback kept most transactions successful, measured with real SIMs rather than sandbox APIs. It is not an AI project, but it proves the engineer can ship a reliable system on unreliable networks.

## Implementation details

Three deliverables make up the Constraint Resume.

### 1. The constraint problem statement (50–100 words)

Write a short paragraph answering:

- What problem was solved?
- What constraints applied? (network, device, payment method, cost)
- What was the real impact? (users served, revenue protected, uptime maintained)

Example:

> Built a mobile money disbursement system for a microfinance bank in Accra. Constraints: 3G with 800ms latency spikes, STK push callbacks arriving out of order, API rate limits of 10 requests/second. System processed 12,000 disbursements/day with 99.4% success rate, up from 87% before optimizations.

### 2. The resilience code samples (30–50 lines each)

Include three code snippets that prove resilience:

- **Retry logic with jitter**
- **Circuit breaker or fallback**
- **Latency instrumentation**

TypeScript retry logic with exponential backoff and jitter:

```typescript
import axios from 'axios';

const retryWithJitter = async (
  fn: () => Promise<any>,
  maxRetries = 3,
  baseDelay = 1000
): Promise<any> => {
  let attempt = 0;
  let lastError: unknown;

  while (attempt < maxRetries) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      attempt++;

      if (attempt >= maxRetries) break;

      // Exponential backoff + jitter, capped to avoid runaway waits
      const delay = Math.min(
        baseDelay * Math.pow(2, attempt - 1) + Math.random() * 100,
        8000
      );
      await new Promise(res => setTimeout(res, delay));
    }
  }

  throw lastError;
};

// Usage: wrap any async operation that might fail
const fetchWithRetry = async (url: string) => {
  return retryWithJitter(async () => {
    const res = await axios.get(url, { timeout: 5000 });
    return res.data;
  });
};
```

A Go circuit breaker using `github.com/sony/gobreaker`, tuned for mobile-money APIs under load:

```go
package main

import (
	"fmt"
	"log"
	"time"

	"github.com/sony/gobreaker"
)

var cb *gobreaker.CircuitBreaker

func init() {
	// Allow 5 failures in a 30s window, then open for 15s
	st := gobreaker.Settings{
		Name:        "mpesa-stk-push",
		MaxRequests: 5,
		Interval:    30 * time.Second,
		Timeout:     15 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.Total >= 5 &&
				float64(counts.TotalFailures)/float64(counts.Requests) >= 0.5
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			log.Printf("Circuit breaker '%s' changed from %s to %s", name, from, to)
		},
	}

	cb = gobreaker.NewCircuitBreaker(st)
}

func pushSTK(payload MpesaStkPayload) (string, error) {
	result, err := cb.Execute(func() (interface{}, error) {
		return mpesaClient.PushStk(payload)
	})

	if err != nil {
		if err == gobreaker.ErrOpenState {
			// Fallback: queue for SMS instead
			queueForSmsFallback(payload)
			return "", fmt.Errorf("circuit open - falling back to SMS: %w", err)
		}
		return "", err
	}

	return result.(string), nil
}
```

Latency instrumentation in Python using the Prometheus client, measuring a disbursement endpoint:

```python
import os
import time

import requests
from prometheus_client import Summary, start_http_server

# Start metrics server on port 8000
start_http_server(8000)

DISBURSEMENT_LATENCY = Summary(
    'disbursement_latency_seconds',
    'Latency of disbursement calls'
)

@DISBURSEMENT_LATENCY.time()
def disburse(amount: int, recipient: str):
    start = time.time()
    try:
        response = requests.post(
            "https://api.example-payments.test/v3/transfers",
            json={
                "amount": amount,
                "recipient": recipient,
                "currency": "NGN"
            },
            headers={"Authorization": f"Bearer {os.getenv('PAYMENTS_SECRET')}"},
            timeout=8  # Realistic timeout on 3G
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Disbursement failed after {time.time() - start}s: {e}")
        raise
    finally:
        # Record actual latency even on failure
        DISBURSEMENT_LATENCY.observe(time.time() - start)
```

Two notes on these snippets. First, `Summary.time()` already observes latency, so the explicit `observe` in the `finally` block double-counts; keep one or the other — the explicit call is shown because it records failures too, which the decorator does as well, so prefer the decorator and drop the manual observe. Second, the circuit-breaker threshold above is a starting point, not a tuning result: `counts.TotalFailures` and `counts.Requests` are the fields you compare, and the right ratio depends on your measured baseline failure rate.

### 3. The post-mortem write-up (200–300 words)

This is the most critical part of the Constraint Resume. It must answer:

- What failed?
- How was it detected?
- What was done?
- What was learned?

Illustrative example:

> **Post-mortem: STK push avalanche during a promotions spike**
>
> Problem: During a promotions period, the USSD-to-mobile-money flow received roughly 4x normal traffic. The upstream API began returning 503s at around 90 requests/minute.
>
> Detection: A Prometheus alert fired on `stk_push_latency_seconds{quantile="0.95"} > 3`. Within about a minute, user reports arrived in the support channel.
>
> Root cause: The existing retry logic (3 attempts, 200ms delay) amplified load during the incident. Retries arrived faster than the upstream could recover.
>
> Fix: Deployed a circuit breaker with a 5-failure/30s window and 15s open timeout. Added exponential backoff with jitter (base 500ms, max 8s). Queued failed pushes for SMS fallback.
>
> Result (illustrative figures from the incident review):
> - Error rate dropped from 18% to 2.3% within 10 minutes
> - 95th percentile latency fell from 4.2s to 1.8s
> - SMS fallback handled 11% of transactions during peak
> - Cost increase: $0.0012 per fallback SMS
>
> Lesson: Never trust API SLAs on mobile networks. Assume elevated failure rates under load. Instrument everything, including local testing.

## Advanced edge cases

The following are failure patterns that appear in production systems and are invisible in sandbox APIs and desktop browser testing. They are described generically; the value is in the pattern, not the vendor.

1. **DNS resolving to a private address during handovers**
   During transitions between 2G and 3G, some carrier resolvers intermittently return a private address (for example `192.168.1.1`) for a public API hostname. Only users on specific towers during network merges are affected. Mitigation: a DNS health check before critical API calls, with fallback to a known-good resolver when a private address is returned. This adds latency to cold starts but prevents a class of failed transactions. Measure it by logging resolved addresses alongside request outcomes and comparing failure rates by resolver.

2. **HTTP 404 returned for successful transactions**
   Some payment APIs return 404 when a request header contains certain characters — for example, a UUID with hyphens in a request-ID field. The sandbox does not replicate this. Detection usually comes from a user report of a burst of failed payments. Mitigation: normalize request headers (strip or encode problematic characters) before sending. The CPU cost is negligible; the correctness gain is not.

3. **Rate limits that change by time of day**
   A disbursement endpoint may allow a higher request rate on weekdays and a lower one on weekends. A bulk-payout cron job that assumes the weekday limit will queue thousands of failed requests on Saturday. Mitigation: a token-bucket rate limiter sized to the lower limit, plus a fallback queue for critical payouts. Measure by plotting 429 responses per hour against request rate.

4. **Race conditions around SIM state**
   A SIM-swap API may enforce a lockout window after a swap. If the system checks swap status, then attempts a USSD push, and the user swaps SIMs in between, the push can fail silently. Mitigation: bind the user session to the SIM identifier during the swap check and fail fast if it changes. The added latency is small compared with the eliminated failure class.

5. **OS background restrictions delaying callbacks**
   On modern Android versions, Doze mode can delay background work for minutes, so a payment callback may arrive long after the user expects it. A retry schedule that starts at 200ms is therefore too aggressive and wastes attempts. Mitigation: schedule the first retry with a delay appropriate to the platform, and require network connectivity before running. Measure callback latency distributions before and after.

Each of these is invisible in local testing and sandbox APIs. Real-world resilience comes from shipping under these constraints and documenting how the chaos was handled.

## Integration with real tools

A minimal, production-shaped integration combines three concerns: a payment API client with retry and circuit breaking, a rate-limited disbursement path, and observability that distinguishes success from failure. The exact provider and library versions vary; the structure below is what matters.

### 1. STK push with retry, circuit breaker, and fallback

```typescript
import axios from 'axios';
import CircuitBreaker from 'opossum';
import { createLogger } from 'pino';

const logger = createLogger({ level: 'info' });

const MPESA_CONFIG = {
  consumerKey: process.env.MPESA_CONSUMER_KEY!,
  consumerSecret: process.env.MPESA_CONSUMER_SECRET!,
  shortCode: '123456',
  passKey: process.env.MPESA_PASSKEY!,
  stkTimeout: 5000,
};

// Circuit breaker for STK push
const mpesaBreaker = new CircuitBreaker(
  async (payload: MpesaStkPayload) => {
    const timestamp = new Date().toISOString().replace(/[-:.]/g, '');
    const password = Buffer.from(
      `${MPESA_CONFIG.shortCode}${MPESA_CONFIG.passKey}${timestamp}`
    ).toString('base64');

    const accessToken = await getMpesaAccessToken();

    const res = await axios.post(
      'https://sandbox.safaricom.co.ke/mpesa/stkpush/v1/processrequest',
      {
        BusinessShortCode: MPESA_CONFIG.shortCode,
        Password: password,
        Timestamp: timestamp,
        TransactionType: 'CustomerPayBillOnline',
        Amount: payload.amount,
        PartyA: payload.phone,
        PartyB: MPESA_CONFIG.shortCode,
        PhoneNumber: payload.phone,
        CallBackURL: process.env.MPESA_CALLBACK_URL!,
        AccountReference: payload.reference,
        TransactionDesc: payload.description,
      },
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
        timeout: MPESA_CONFIG.stkTimeout,
      }
    );

    if (res.status !== 200 || res.data.ResponseCode !== '0') {
      throw new Error(
        `M-Pesa API error: ${res.data.ResponseDescription || 'Unknown'}`
      );
    }

    return res.data;
  },
  {
    timeout: 10000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000,
  }
);

mpesaBreaker.on('open', () => logger.warn('M-Pesa circuit breaker OPEN'));
mpesaBreaker.on('close', () => logger.info('M-Pesa circuit breaker CLOSED'));

const pushStkWithRetry = async (payload: MpesaStkPayload): Promise<string> => {
  try {
    const result = await mpesaBreaker.fire(payload);
    return result.CheckoutRequestID;
  } catch (err) {
    logger.error({ err, payload }, 'M-Pesa STK push failed');
    throw err;
  }
};

const fallbackToSms = (phone: string, message: string): void => {
  logger.info({ phone, message }, 'Falling back to SMS');
};
```

### 2. Disbursement with rate limiting and queue fallback

```python
import os
import time

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

RATE_LIMIT = 10
REFILL_MS = 100
bucket = RATE_LIMIT
last_refill = time.time() * 1000


def refill_bucket():
    global bucket, last_refill
    now = time.time() * 1000
    elapsed = now - last_refill
    if elapsed > 0:
        refill_amount = int(elapsed / REFILL_MS)
        bucket = min(RATE_LIMIT, bucket + refill_amount)
        last_refill = now


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(multiplier=0.5, max=8),
    retry=retry_if_exception_type(
        (requests.exceptions.RequestException, requests.exceptions.Timeout)
    ),
)
def disburse(amount: int, recipient_account: str, recipient_bank: str):
    refill_bucket()
    if bucket <= 0:
        time.sleep(REFILL_MS / 1000)
        refill_bucket()
        if bucket <= 0:
            raise Exception("Rate limit exceeded")

    bucket -= 1

    headers = {
        "Authorization": f"Bearer {os.getenv('PAYMENTS_SECRET')}",
        "Content-Type": "application/json",
    }
    payload = {
        "amount": amount,
        "account_bank": recipient_bank,
        "account_number": recipient_account,
        "currency": "NGN",
        "narration": "Salary disbursement",
        "reference": f"pay_{int(time.time())}",
    }

    resp = requests.post(
        "https://api.example-payments.test/v3/transfers",
        json=payload,
        headers=headers,
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json()
```

### 3. Real-time monitoring

Prometheus scrape configuration:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mpesa-service'
    static_configs:
      - targets: ['mpesa-service:8000']
    metrics_path: '/metrics'

  - job_name: 'disburser'
    static_configs:
      - targets: ['disburser:8001']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alert-rules.yml'
```

Sample alert rules:

```yaml
groups:
- name: mpesa-alerts
  rules:
  - alert: MpesaHighLatency
    expr: histogram_quantile(0.95, stk_push_duration_seconds_bucket) > 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "STK push latency above 3s for 5m"
      description: "Current 95th percentile latency: {{ $value }}s"

  - alert: MpesaCircuitBreakerOpen
    expr: mpesa_circuit_breaker_open > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "M-Pesa circuit breaker is OPEN"

- name: disbursement-alerts
  rules:
  - alert: RateLimitExceeded
    expr: increase(disbursement_failures_total[1m]) > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Disbursement failures spiking"
```

Grafana dashboard panels, expressed as queries rather than exported JSON:

- **STK push latency (95th percentile)**: `histogram_quantile(0.95, sum(rate(stk_push_duration_seconds_bucket[5m])) by (le))`
- **Disbursement success rate**: `1 - (sum(rate(disbursement_failures_total[5m])) / sum(rate(disbursement_attempts_total[5m])))`
- **Circuit breaker state**: `mpesa_circuit_breaker_open`

This structure is what a reviewer should be able to read in a portfolio: the metrics that matter, the alerts that fire, and the fallbacks that engage.

## Before/after comparison: two portfolios

Compare two versions of the same project: an "M-Pesa Disbursement System".

### Candidate A: "AI-Powered Payment Router"

**Portfolio claim:** Built a system that "uses AI to route payments for maximum speed and fraud prevention."

**What they showed:**

- GitHub repo with 3 commits, last updated a year earlier
- README: "Uses an LLM to predict the fastest route"
- Demo: localhost video showing 0.2s response time
- Metrics: screenshot of a generated graph claiming "99.9% success rate"

**Code snippet (only file):**

```python
# main.py — 24 lines
from langchain import LLMMathChain

llm = Llama3  # unspecified model
def route_payment(amount, recipient):
    return llm.predict(f"Choose fastest route for {amount} to {recipient}")
```

**Behavior under load:**

- Localhost: 200ms
- On a cloud VM over fibre: 180ms
- On 3G with a real device: failed consistently, because the LLM API timed out at 5s
- Cost: $0.012 per prediction (illustrative)

**Lines of code:** 24
**Dependencies:** an LLM orchestration library, an unspecified model
**Test coverage:** 0%
**Post-mortem:** None

### Candidate B: "Reliable M-Pesa Disbursement for Rural Cooperatives"

**Portfolio claim:** "Built a system that disburses 12,000 M-Pesa payments/day to dairy farmers in rural Kenya."

**What they showed:**

- GitHub repo with 140 commits over 8 months
- README with a constraint statement: 2G/3G, 10s USSD timeout, 5% packet loss, API rate limit of 10 req/s
- Code: retry with jitter, circuit breaker, SMS fallback, latency histograms
- Post-mortem: a network outage where SMS fallback kept transactions flowing, with error rate and latency figures

**Behavior under load:**

- Localhost: 150ms
- On a cloud VM over fibre: 140ms
- On 3G with a real device: 1.8s p95, with retries and fallback engaging as designed
- Cost: $0.0008 per transaction (illustrative)

**Lines of code:** 1,400
**Dependencies:** a payments SDK, a circuit-breaker library, a metrics client
**Test coverage:** 62%
**Post-mortem:** Yes, with metrics

### What the comparison shows

| Dimension | Candidate A | Candidate B |
|---|---|---|
| Problem statement | "AI-powered routing" | Constraints listed explicitly |
| Retry logic | None | Exponential backoff with jitter |
| Fallback | None | SMS queue |
| Latency measurement | Average, localhost | Percentiles, real device |
| Post-mortem | None | Incident write-up with metrics |
| Behavior on 3G | Fails | Degrades gracefully |

The difference is not AI versus non-AI. It is whether the portfolio demonstrates behavior under constraints.

## How to measure it yourself

None of the numbers above should be taken on faith. Here is how to produce your own.

- **Latency percentiles**: instrument every outbound call with a histogram. Record p50, p95, p99. Compare against a baseline captured on a fast connection. The gap is your network penalty.
- **Retry effectiveness**: count attempts per logical operation. A retry policy that works shows a falling failure rate as attempts increase; one that does not shows a flat or rising rate because retries are amplifying load.
- **Circuit breaker behavior**: log state transitions with timestamps. Measure how long the breaker stays open and what fraction of requests hit the fallback.
- **Fallback coverage**: count how many operations are served by the fallback path during an incident. If it is zero, the fallback is untested.
- **Cost of failure**: multiply failed transactions by the value at risk. This turns reliability work into a number a business can act on.

A simple way to start: add a histogram around your most critical outbound call, run a load test against a staging environment with an artificial delay injected, and compare p95 latency and error rate before and after adding retries and a breaker.

## Decision checklist for reviewers

When reviewing a portfolio for reliability evidence, ask:

- Does the problem statement name specific constraints (network, device, payment method, cost)?
- Is there retry logic, and is it bounded with backoff and jitter?
- Is there a circuit breaker or fallback, and is it exercised in the write-up?
- Are latency numbers percentiles rather than averages?
- Is there a post-mortem describing a real failure and the fix?
- Do the metrics distinguish success from failure, or only count requests?
- Can the candidate explain why each threshold was chosen?

If most answers are yes, the candidate can likely ship under constraints. If most are no, the portfolio is showing features, not engineering.

## FAQ

**Does the project have to be a payment system?**
No. The pattern applies to any system with unreliable dependencies: messaging, logistics, IoT, or any client-server flow over poor networks. Payment systems are common because the constraints are sharp and the cost of failure is visible.

**How much code should be included?**
Enough to prove the pattern: a retry helper, a breaker or fallback, and an instrumentation snippet. Thirty to fifty lines each is usually sufficient. The post-mortem carries as much weight as the code.

**What if the project is proprietary?**
Describe the architecture and the failure modes without exposing code. Redact identifiers and amounts. A well-written post-mortem with redacted metrics is more convincing than a public repo with no incident history.

**Can AI features be part of the portfolio?**
Yes, but they should be presented as one component among several, with the same rigor applied to their failure modes. An LLM call is an unreliable dependency like any other; it needs timeouts, retries and fallbacks.

**How do I get real network conditions for testing?**
Use a throttling proxy or a device on a real mobile network. Inject latency and packet loss in staging. The point is to observe behavior when the network is slow or flaky, not to reproduce a specific carrier.

## Do this in the next 30 minutes

Pick one project you have already built. Add a single histogram around its most critical outbound call, run it once against a slow or throttled connection, and write down the p95 latency and the error rate. That one measurement is the seed of a Constraint Resume: it turns a feature list into evidence of behavior under constraints.
