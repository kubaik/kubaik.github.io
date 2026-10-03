# Pricing SaaS tools by real cost, not seat count

Most pricing tutorials show the happy path: pick a tier name, draw three columns, ship it. This article covers what comes after — the part where the bill and the price list disagree, and the disagreement is expensive.

## Why seat-based pricing breaks

Seat pricing is easy to sell and easy to explain. It also decouples revenue from the thing that actually costs money. A single seat can generate one request per day or two million; the invoice looks the same. That mismatch produces three recurring failure modes.

**Failure mode 1: infra cost is not the price floor.** A team can price an API call above its marginal compute cost and still lose money on it, because the marginal cost of a request is not the marginal cost of a customer interaction. Once a request fails, a support ticket is opened, an engineer investigates, and the cost of that ticket can exceed the revenue from thousands of successful calls. The floor is `infra cost + expected support cost`, not `infra cost`.

**Failure mode 2: seats punish power users.** A seat-based plan silently subsidises light users and taxes heavy ones. When a single account starts running thousands of concurrent jobs, the account's cost curve detaches from its seat count, and the invoice arrives after the damage.

**Failure mode 3: free tiers can be drained in a burst.** A monthly event allowance with no rate limit is a shared resource. One account that consumes the allowance in a single burst leaves every other free user throttled for the rest of the period, and the support queue fills with "the product stopped working" tickets that have nothing to do with a bug.

The rest of this article builds a pricing model that tracks cost, plus the instrumentation needed to defend it.

## Prerequisites and what you'll build

You need a running service you can instrument, a way to replay traffic, and a pricing page you can change without a deploy. If you have no traffic yet, the load generator below stands in for it.

By the end you will have:

* A pricing model tied to measured infra and support cost rather than seat count.
* A local replay environment for testing a proposed tier against recorded traffic.
* Metrics for latency, infra cost per request, and support tickets per 1 000 requests.
* A billing script that simulates the heaviest accounts and shows the break-even point.

Stack used here: Node 20 LTS for the API and billing script, Redis 7.2 for counters and feature flags, Prometheus with Grafana for metrics, Docker Compose for local infrastructure. The AWS SDK is included only to model Lambda and DynamoDB costs if you have no production data yet. Running everything locally costs nothing; the cloud equivalents are within typical free-tier allowances.

## Step 1 — set up the environment

```bash
mkdir pricing-lab && cd pricing-lab
npm init -y
npm install express@4.18.2 redis@4.6.12 prom-client@14.2.0 node-cache@5.1.2 @aws-sdk/client-dynamodb@3.600.0 @aws-sdk/client-sqs@3.600.0
npm install --save-dev jest@29.7.0 @types/jest@29.5.12 @types/node@20.12.7 typescript@5.4.5 ts-jest@29.1.2
```

A minimal API that returns feature flags and counts usage:

```typescript
import express from 'express';
import { createClient } from 'redis';
import promClient from 'prom-client';

const app = express();
const redis = createClient({ url: 'redis://localhost:6379' });

const httpRequestsTotal = new promClient.Counter({
  name: 'http_requests_total',
  help: 'Total HTTP requests',
  labelNames: ['route', 'status'],
});

await redis.connect();

app.get('/api/flags', async (_req, res) => {
  await redis.incr('api_calls');
  httpRequestsTotal.inc({ route: '/api/flags', status: '200' });
  res.json({ beta: (await redis.get('beta')) === '1' });
});

app.get('/health', (_req, res) => {
  httpRequestsTotal.inc({ route: '/health', status: '200' });
  res.send('ok');
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`API listening on ${port}`);
});
```

Local infrastructure with Docker Compose:

```yaml
version: '3.8'
services:
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  prometheus:
    image: prom/prometheus:v2.52.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  grafana:
    image: grafana/grafana:11.1.0
    ports:
      - "3001:3000"
    volumes:
      - grafana_storage:/var/lib/grafana

volumes:
  redis_data:
  grafana_storage:
```

`prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['host.docker.internal:3000']
```

```bash
docker-compose up -d
npm run build && node dist/index.js &
```

On macOS or Windows, `host.docker.internal` resolves to the host from inside a container. On Linux it does not; add an `extra_hosts` entry mapping `host.docker.internal` to `host-gateway`, or point the scrape target at the host's bridge IP.

## Step 2 — build the cost model

The model has two parts: a measured infra cost per unit of work, and a measured support cost per ticket. Both come from your own telemetry, not from a table copied out of someone else's article. The structure below shows the shape; every constant must be replaced with a number you measured.

```typescript
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';

export type Usage = {
  requests: number;
  concurrentJobs: number;
  supportTickets: number;
};

// Replace every constant below with your own measured value.
const LAMBDA_COST_PER_100MS = 0.000012;
const DYNAMO_READ_COST = 0.000025;
const REDIS_COST_PER_MB_HOUR = 0.000008;
const SUPPORT_COST_PER_TICKET = 28;

function infraCost(usage: Usage, hours: number): number {
  const lambdaCost = usage.requests * 0.005 * LAMBDA_COST_PER_100MS * hours;
  const dynamoCost = usage.requests * DYNAMO_READ_COST * hours;
  const redisCost = Math.min(
    usage.concurrentJobs * 2 * REDIS_COST_PER_MB_HOUR * hours,
    0.01
  );
  return lambdaCost + dynamoCost + redisCost;
}

export function suggestedPrice(usage: Usage, hours: number): number {
  const infra = infraCost(usage, hours);
  const support = usage.supportTickets * SUPPORT_COST_PER_TICKET;
  return Math.max(infra + support, 0.05) * 1.25;
}
```

The `0.05` floor prevents a zero invoice for a single low-volume developer. The `1.25` multiplier is a margin assumption covering payment processing and overhead; adjust it to your actual fee structure.

A `/price` endpoint that reads live counters:

```typescript
app.get('/price', async (_req, res) => {
  const requests = await redis.get('api_calls');
  const jobs = (await redis.get('concurrent_jobs')) || '1';
  const tickets = (await redis.get('support_tickets')) || '0';

  const usage: Usage = {
    requests: parseInt(requests || '0', 10),
    concurrentJobs: parseInt(jobs, 10),
    supportTickets: parseInt(tickets, 10),
  };

  const price = suggestedPrice(usage, 24);
  httpRequestsTotal.inc({ route: '/price', status: '200' });
  res.json({
    price,
    currency: 'USD',
    breakdown: {
      infra: infraCost(usage, 24),
      support: usage.supportTickets * SUPPORT_COST_PER_TICKET,
    },
  });
});
```

### How to measure the constants instead of guessing them

Do not trust any cost table, including the one above. Measure:

* **Infra cost per request.** Instrument a counter for total requests and a gauge for cumulative provider spend pulled from the billing API. Divide spend by requests over a fixed window (one week is usually stable enough). Compare week over week to catch drift.
* **Support cost per ticket.** Take total support payroll plus tooling for a period, divide by tickets closed in the same period. If you cannot attribute payroll, use the fully loaded hourly cost of the people who answer tickets multiplied by average handling time.
* **Concurrency cost.** Run a load test at increasing concurrency and record provider spend at each step. The slope of that line is your marginal cost per concurrent job.
* **Latency baseline.** Record P50 and P99 under expected load before you change anything, so you have a reference point when a tier change alters traffic shape.

A useful sanity check: compute infra cost per 1 000 requests and support cost per 1 000 requests separately. If support is the larger number, your pricing problem is a product problem — the fix is fewer tickets, not a higher price.

## Step 3 — handle the edge cases that break pricing

**Cache stampede.** When a feature flag flips for many accounts at once, every request misses the cache and hits Redis simultaneously. Redis CPU climbs, latency follows, and the cost per request rises exactly when volume does. A short-lived in-process cache absorbs the burst:

```typescript
import NodeCache from 'node-cache';
const cache = new NodeCache({ stdTTL: 5 });

app.get('/api/flags', async (_req, res) => {
  const cached = cache.get('beta');
  if (cached !== undefined) {
    httpRequestsTotal.inc({ route: '/api/flags', status: '200' });
    return res.json({ beta: cached === '1' });
  }
  const beta = await redis.get('beta');
  cache.set('beta', beta);
  httpRequestsTotal.inc({ route: '/api/flags', status: '200' });
  res.json({ beta: beta === '1' });
});
```

The trade-off is staleness: with a 5-second TTL, a flag change takes up to 5 seconds to propagate. For feature flags that is usually acceptable; for kill switches it is not, so keep those on a separate uncached path.

**Free-tier exhaustion.** A monthly allowance with no rate limit is a shared resource. Nginx rate limiting enforces a per-client ceiling before the request reaches your application:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
server {
  location / {
    limit_req zone=api_limit burst=30 nodelay;
    proxy_pass http://localhost:3000;
  }
}
```

`rate=10r/s` with `burst=30 nodelay` allows short bursts up to 30 requests while capping sustained traffic at 10 per second per address. Tune both numbers against your measured P99 request rate per account, not against intuition.

**Concurrency spikes.** A single account can enqueue thousands of jobs at once. Decouple request acceptance from execution with a queue so the spike becomes a backlog rather than a provider-limit event:

```typescript
import { SQSClient, SendMessageCommand } from '@aws-sdk/client-sqs';
const sqs = new SQSClient({ region: 'ap-southeast-1' });

app.post('/jobs', async (req, res) => {
  const { userId, job } = req.body;
  await sqs.send(new SendMessageCommand({
    QueueUrl: process.env.JOBS_QUEUE_URL,
    MessageBody: JSON.stringify({ userId, job }),
  }));
  res.send('queued');
});
```

Note the queue URL comes from an environment variable. Hard-coding an account ID and queue name into source is a credential-hygiene problem and a portability problem at the same time.

The pattern across all three: every pricing incident is either a rate spike or a support-ticket spike. Instrument both before publishing a price.

## Step 4 — observability and tests

Alerting rules:

```yaml
groups:
- name: pricing-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.12
    for: 5m
    labels:
      severity: page
    annotations:
      summary: "High latency on {{ $labels.route }}"
  - alert: CostSpike
    expr: increase(infra_cost_total{job="api"}[1h]) > 10
    for: 15m
    labels:
      severity: ticket
    annotations:
      summary: "Cost spike detected"
```

Tests that pin the pricing invariants:

```typescript
import { suggestedPrice } from './pricing';

test('price floor is enforced', () => {
  const price = suggestedPrice({ requests: 1, concurrentJobs: 1, supportTickets: 0 }, 24);
  expect(price).toBeGreaterThanOrEqual(0.05);
});

test('margin is applied on top of infra plus support', () => {
  const usage = { requests: 1000, concurrentJobs: 1, supportTickets: 0 };
  const price = suggestedPrice(usage, 24);
  const infraOnly = 1000 * 0.005 * 0.000012 * 24 + 1000 * 0.000025 * 24;
  expect(price).toBeGreaterThan(infraOnly * 1.25);
});
```

The second test computes its expected floor from the same constants rather than hard-coding a number, so it keeps working when constants change and fails when the margin logic breaks.

```bash
npm test
```

Observability checklist:

* Grafana dashboard with three panels: P99 latency, infra cost per 1 000 requests, support tickets per 1 000 requests.
* One alert when infra cost per 1 000 requests exceeds your measured baseline by a margin you choose, sustained for 15 minutes.
* One alert when support tickets per 1 000 requests exceed the trailing baseline.

## A worked example: finding the break-even account

Suppose the measured constants are: 0.005 seconds of compute per request at $0.000012 per 100 ms, plus $0.000025 per DynamoDB read, plus a $0.05 floor, with a 1.25 margin. Support cost is $28 per ticket, and the observed rate is 0.2 tickets per 1 000 requests.

For 1 000 requests over 24 hours:

1. Compute: 1 000 × 0.005 s = 5 s of compute. At $0.000012 per 100 ms (0.1 s), that is 5 s ÷ 0.1 s = 50 units × $0.000012 = $0.0006.
2. Reads: 1 000 × $0.000025 = $0.025.
3. Support: 0.2 tickets × $28 = $5.60.
4. Total cost: $0.0006 + $0.025 + $5.60 = $5.6256.
5. Price before margin: max($5.6256, $0.05) = $5.6256.
6. Price after margin: $5.6256 × 1.25 = $7.032.

Support is 99.5% of the cost. This is the single most useful output of the model: it tells you that optimising compute is pointless and that reducing ticket volume is the only lever that matters. A pricing change that raises the price without reducing tickets simply moves the loss to the customer's churn decision.

Now run the same arithmetic at 1 000 000 requests with the same ticket rate:

1. Compute: 1 000 000 × 0.005 s = 5 000 s ÷ 0.1 s = 50 000 units × $0.000012 = $0.60.
2. Reads: 1 000 000 × $0.000025 = $25.
3. Support: 200 tickets × $28 = $5 600.
4. Total: $5 625.60. Price after margin: $7 032.

The ratio is unchanged because both components scale linearly with requests. Break-even analysis only becomes interesting when ticket rate is sublinear in volume — which is what a good self-service experience buys you. Measure your ticket rate at two different volume levels before assuming it is constant.

## Decision checklist before publishing a price

* Every constant in the pricing function traces to a measurement with a date and a source.
* The price floor exceeds marginal infra cost plus expected support cost per unit.
* Free tiers have a per-client rate limit, not just a monthly allowance.
* Concurrency is metered separately from request count.
* A replay of one week of production traffic produces a price within your tolerance of the actual bill.
* There is an alert on cost per 1 000 requests and one on tickets per 1 000 requests.
* The pricing page can change without a deploy.

## Common questions

**How do I price serverless workloads with spiky usage?**

Use a two-part tariff: a small fixed monthly fee covering baseline provisioned throughput, plus a variable rate tied to a high percentile of concurrency (P95 or P99) rather than the peak. Percentile-based charging means a single burst does not dominate the invoice. Compute the variable component from your measured marginal cost per concurrent job, not from a published price list.

**What if users are in regions with different infra costs?**

Apply a regional multiplier to the infra component only, and keep the support component flat, since support cost does not vary with the customer's region. Derive each multiplier by measuring the same workload in each region and dividing. Publishing the derivation prevents the "why is my bill three times higher" conversation.

**Should existing customers be grandfathered?**

A bounded grandfathering window — one billing cycle is common — gives customers time to adjust usage and gives you a deadline to plan around. An unbounded window means the old price structure persists forever and every future change has to account for it.

**How do I communicate a pricing change?**

Send the cost breakdown before the invoice. A message that shows the measured infra cost, the support cost, and the resulting price is far more persuasive than a message that shows only the new number. Customers argue with prices; they argue less with arithmetic they can check.

## What to do in the next 30 minutes

Instrument the two numbers the model depends on and compute the ratio between them. Add a counter for total requests and a gauge for cumulative provider spend, let them run for a day, then divide. If support cost per 1 000 requests exceeds infra cost per 1 000 requests, you have found your actual pricing problem, and it is not the price.
