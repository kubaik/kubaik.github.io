# Prevent ownership drift when velocity explodes

High delivery velocity is a good problem until it outruns accountability. Production gives you neither a clean environment nor a patient timeline, and the failure mode is predictable: multiple engineers touch the same service, each assuming someone else has fixed the race condition, tuned the TTL, or noticed the latency regression.

The pattern is common. Teams optimise for velocity — more pull requests, more deploys, more features — but when ownership isn't explicitly tied to deployment boundaries, changes accumulate faster than accountability can keep pace. The hard part isn't writing the code; it's answering "who owns the payment latency spike after the new checkout went live?" Without an answer, that question produces finger-pointing rather than fixes.

This article is not about slowing down. It's about preventing ownership drift when velocity spikes. The running example is a Node API gateway fronting a downstream service, with a Redis cache layer bolted on. A typical failure mode looks like this: a cache added by one team improves median response time considerably, but introduces a periodic p99 spike during cache invalidation. Nobody owns that tail latency, because the cache and the service live in the same repository and neither team has an explicit SLO for p99.

## Prerequisites and what you'll build

You'll need:

- Node 20 LTS (the example targets v20.12.0)
- Redis 7.2 for caching
- A metrics backend (any Prometheus-compatible remote write endpoint, self-hosted or managed)
- A small dev instance — 2 vCPU and 2 GiB is enough to exercise the code paths
- A CI runner (GitHub Actions is used in the workflow below)

By the end you'll have:

1. A minimal Node API gateway that proxies requests to a backend
2. A Redis cache layer with TTL-based invalidation
3. Explicit ownership boundaries expressed as deployment tags and error budgets
4. Metrics that surface ownership drift when error rates cross the p99 budget

The point isn't production polish. It's to make ownership gaps visible before they become incidents. If the setup feels too simple, that's intentional — the trap appears even at this scale.

## Step 1 — set up the environment

Start with a fresh directory:

```bash
mkdir ownership-gateway && cd ownership-gateway
npm init -y
git init
```

Install dependencies:

```bash
npm install express redis@4.6.10 express-rate-limit@6.7.0
```

Create `.env`:

```ini
REDIS_URL=redis://127.0.0.1:6379
PORT=8000
CACHE_TTL=30
CURRENT_TEAM=platform-team-cache
```

Spin up Redis 7.2 in Docker for local testing:

```bash
docker run --name redis-ownership -p 6379:6379 -d redis:7.2-alpine redis-server --save "" --appendonly no
```

Verify Redis is running:

```bash
redis-cli ping
# Expect: PONG
```

Create `src/index.js`:

```javascript
import express from 'express';
import { createClient } from 'redis';
import rateLimit from 'express-rate-limit';

const app = express();
const redis = createClient({ url: process.env.REDIS_URL });

await redis.connect();

const limiter = rateLimit({
  windowMs: 1000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);
app.use(express.json());

app.get('/health', (req, res) => {
  res.status(200).json({ ok: true });
});

app.get('/api/data', async (req, res) => {
  const cacheKey = `data:${req.ip}`;
  const cached = await redis.get(cacheKey);

  if (cached) {
    return res.json({ source: 'cache', data: JSON.parse(cached) });
  }

  // Simulate a downstream service call that can fail
  const data = { id: 1, value: Math.random() };
  await redis.set(cacheKey, JSON.stringify(data), {
    EX: parseInt(process.env.CACHE_TTL, 10),
  });

  res.json({ source: 'service', data });
});

app.listen(process.env.PORT, () => {
  console.log(`Gateway listening on port ${process.env.PORT}`);
});
```

Add a `.gitignore`:

```ini
node_modules/
.env
*.log
.DS_Store
```

Commit the scaffolding:

```bash
git add .
git commit -m "Scaffold Node gateway with Redis cache"
```

Why this shape? It's small enough to deploy in minutes, yet it already contains the seeds of ownership drift:

- The cache and the service share a single Redis connection in the same repository
- No clear owner for cache eviction policy or TTL tuning
- The health endpoint hides latency regressions because it doesn't exercise the cache path

A common mistake is treating the cache as "just a performance tweak" and delegating its tuning to whoever opened the pull request. That leads to TTLs set to five minutes during development being pushed to production where 30 seconds is the real requirement — and cache stampedes plus p99 spikes every time the TTL expires.

## Step 2 — make the boundary explicit

Split the gateway into two logical components: the proxy and a cache module. This mirrors the real situation where two teams work in the same repository but one owns the cache layer and the other owns the proxy logic.

Create `src/cache.js`:

```javascript
import { createClient } from 'redis';

const redis = createClient({ url: process.env.REDIS_URL });

// Explicitly declare cache ownership
const CACHE_OWNER = 'platform-team-cache';
const CACHE_TTL = parseInt(process.env.CACHE_TTL || '30', 10);

// Cache write function — only the cache owner should call this.
// Other services should use get() only.
export async function getCache(key) {
  return redis.get(key);
}

export async function setCache(key, value) {
  if (!key || !value) {
    throw new Error('Invalid cache key or value');
  }
  await redis.set(key, JSON.stringify(value), { EX: CACHE_TTL });
}

export function getOwner() {
  return CACHE_OWNER;
}

export function getTTL() {
  return CACHE_TTL;
}
```

Update `src/index.js` to import and use the cache module:

```javascript
import express from 'express';
import { getCache, setCache, getOwner, getTTL } from './cache.js';

const app = express();

// Remove the old cache logic; replace with:
app.get('/api/data', async (req, res) => {
  const cacheKey = `data:${req.ip}`;
  const cached = await getCache(cacheKey);

  if (cached) {
    return res.json({ source: 'cache', data: JSON.parse(cached) });
  }

  const data = { id: 1, value: Math.random() };
  await setCache(cacheKey, data);

  res.json({ source: 'service', data });
});
```

Add an endpoint that exposes the ownership contract:

```javascript
app.get('/cache/meta', (req, res) => {
  res.json({
    owner: getOwner(),
    ttl_seconds: getTTL(),
  });
});
```

Run the service:

```bash
node src/index.js
```

Hit the endpoints:

```bash
curl -s http://localhost:8000/cache/meta | jq
# {"owner":"platform-team-cache","ttl_seconds":30}

curl -s http://localhost:8000/api/data | jq
# {"source":"service","data":{...}}

curl -s http://localhost:8000/api/data | jq
# {"source":"cache","data":{...}}
```

This is the critical step: moving cache logic into a separate module with a named owner creates a boundary. The proxy team can no longer change cache behaviour without touching the cache owner's code. That small friction is what prevents the "someone else will fix it" mentality.

A recurring gotcha is merging cache logic into a shared `utils` folder without declaring ownership. When a TTL is hard-coded to 60 seconds in dev but the downstream service actually requires 10 seconds, the regression can persist for weeks because no single engineer owns the file. The eventual fix often touches a dozen files, requires a rollback, and lands in the middle of the night.

## Step 3 — handle edge cases and errors

Edge cases that surface ownership drift:

1. Cache stampedes during TTL expiry
2. Redis connection leaks under load
3. Invalid cache keys breaking downstream services
4. Misrouted cache metadata causing silent failures

Add error handling and ownership checks. Update `src/cache.js`:

```javascript
import { createClient } from 'redis';

const redis = createClient({ url: process.env.REDIS_URL });

await redis.connect();

const CACHE_OWNER = 'platform-team-cache';
const CACHE_TTL = parseInt(process.env.CACHE_TTL || '30', 10);

// Guardrail: prevent stampedes
const STAMPEDE_LOCK_TTL = 5; // seconds

// Only allow cache writes from the designated owner
function assertCacheOwner() {
  if (process.env.CURRENT_TEAM !== CACHE_OWNER) {
    throw new Error(`Cache writes restricted to team: ${CACHE_OWNER}`);
  }
}

export async function getCache(key) {
  if (!key) throw new Error('Cache key required');
  return redis.get(key);
}

export async function setCache(key, value) {
  assertCacheOwner();
  if (!key || !value) throw new Error('Invalid cache key or value');
  await redis.set(key, JSON.stringify(value), { EX: CACHE_TTL });
}

export async function stampedeLock(key) {
  assertCacheOwner();
  const lockKey = `lock:${key}`;
  const locked = await redis.set(lockKey, '1', { NX: true, EX: STAMPEDE_LOCK_TTL });
  return locked === 'OK';
}

export async function pingCache() {
  return redis.ping();
}

export function getOwner() {
  return CACHE_OWNER;
}

export function getTTL() {
  return CACHE_TTL;
}
```

Note the two additions that matter for correctness: `redis.connect()` is called in this module, and `pingCache()` is exported so the health endpoint doesn't need a second client. Sharing one client across modules avoids connection leaks, which are a frequent production issue when a module creates its own client on every import.

Update the proxy in `src/index.js` to handle stampedes:

```javascript
import express from 'express';
import {
  getCache,
  setCache,
  stampedeLock,
  pingCache,
  getOwner,
  getTTL,
} from './cache.js';

const app = express();

app.get('/api/data', async (req, res) => {
  const cacheKey = `data:${req.ip}`;
  try {
    const cached = await getCache(cacheKey);
    if (cached) {
      return res.json({ source: 'cache', data: JSON.parse(cached) });
    }

    // Stampede protection
    const locked = await stampedeLock(cacheKey);
    if (!locked) {
      // Another request is regenerating the cache; serve stale if available
      const stale = await getCache(cacheKey);
      if (stale) {
        return res.json({ source: 'cache-stale', data: JSON.parse(stale) });
      }
      return res.status(503).json({ error: 'Service unavailable' });
    }

    const data = { id: 1, value: Math.random() };
    await setCache(cacheKey, data);

    res.json({ source: 'service', data });
  } catch (err) {
    console.error(`Cache error owner=${getOwner()}:`, err.message);
    res.status(500).json({ error: 'Cache unavailable' });
  }
});

app.get('/cache/meta', (req, res) => {
  res.json({ owner: getOwner(), ttl_seconds: getTTL() });
});

app.get('/health/redis', async (req, res) => {
  try {
    const pong = await pingCache();
    res.status(200).json({ redis: 'ok', pong });
  } catch (err) {
    res.status(503).json({ redis: 'down', error: err.message });
  }
});
```

A common misstep is logging errors without tying them to ownership. An error line that reads only `Cache unavailable` gives the on-call engineer nowhere to route the page. Including `owner=platform-team-cache` in the log line — and, better, as a label on the metric — turns an anonymous failure into a routable one. The mechanism is simple: structured logs plus a metric label, not a cultural change.

## Step 4 — add observability and tests

Ownership drift is invisible until you instrument it. Add a Prometheus-compatible metrics endpoint and a small test suite.

Install dependencies:

```bash
npm install prom-client@15.1.0 jest@29.7.0 supertest@6.3.3
```

Create `src/metrics.js`:

```javascript
import prom from 'prom-client';

const register = new prom.Registry();
prom.collectDefaultMetrics({ register });

const httpRequestDurationSeconds = new prom.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5],
});

const cacheErrors = new prom.Counter({
  name: 'cache_errors_total',
  help: 'Total cache errors by type',
  labelNames: ['type', 'owner'],
});

register.registerMetric(httpRequestDurationSeconds);
register.registerMetric(cacheErrors);

export { register, httpRequestDurationSeconds, cacheErrors };
```

Instrument the gateway in `src/index.js`:

```javascript
import { register, httpRequestDurationSeconds } from './metrics.js';

app.use((req, res, next) => {
  const end = httpRequestDurationSeconds.startTimer();
  res.on('finish', () => {
    end({ method: req.method, route: req.path, status_code: res.statusCode });
  });
  next();
});

app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  } catch (err) {
    res.status(500).end(err.message);
  }
});
```

Add a test file `src/index.test.js`:

```javascript
import request from 'supertest';
import app from './index.js';

describe('Gateway', () => {
  it('should return cache metadata with owner', async () => {
    const res = await request(app).get('/cache/meta');
    expect(res.body.owner).toBe('platform-team-cache');
    expect(res.body.ttl_seconds).toBe(30);
  });

  it('should serve stale cache during stampede', async () => {
    // Simulate two parallel requests to the same key
    const [res1, res2] = await Promise.all([
      request(app).get('/api/data'),
      request(app).get('/api/data'),
    ]);
    expect(res1.body.source).toMatch(/cache/);
    expect(res2.body.source).toMatch(/cache-stale|service/);
  });
});
```

Add a CI workflow `.github/workflows/test.yml`:

```yaml
name: Test and metrics
on: [push]
jobs:
  test:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm test
```

Skipping tests for cache behaviour is a frequent oversight. A TTL change from 30 to 60 seconds can trigger a stampede that spikes p99 latency well beyond baseline, and a suite that mocks Redis entirely will never see it. The stampede test above exercises the real code path with two concurrent requests, which is the smallest test that would have caught that class of regression.

## How to measure this on your own service

Rather than trust numbers from someone else's environment, instrument your own. The procedure is short:

1. **Establish a baseline.** Run the gateway without the cache module and record p50, p95 and p99 of `http_request_duration_seconds` from the histogram in `src/metrics.js`. Use a load generator of your choice and hold request rate constant for the duration of the run.
2. **Add the cache, keep ownership implicit.** Deploy the version from Step 1. Compare the same percentiles. This isolates the cache's effect on the median versus the tail.
3. **Add the ownership guardrails.** Deploy the Step 3 version. Compare again. The delta between step 2 and step 3 is the cost of the guardrails — typically a small increase in p99 from the extra Redis round trip for the lock, in exchange for eliminating stampede-driven spikes.
4. **Watch `cache_errors_total`.** Break the cache deliberately (stop Redis, or set `CURRENT_TEAM` to the wrong value) and confirm the error counter increments with the `owner` label populated. If it doesn't, your alerting can't route the page.

The numbers you get will depend on your hardware, network topology and cache hit rate. What matters is the shape: without ownership, tail latency is dominated by whichever unowned component happens to be slow that day; with ownership, tail latency is bounded and attributable.

## Failure modes that survive the guardrails

- **Engineers bypass the write guardrail** by setting `CURRENT_TEAM` in their local environment. This is a culture and access-control issue, not a code issue. The durable fix is to block direct cache writes in production at the infrastructure layer — IAM policy, network policy, or a managed cache that only accepts writes from a designated role — rather than relying on an environment variable.
- **TTL tuning stays manual.** A cache miss that triggers a slow downstream call can produce a p99 regression orders of magnitude larger than the cache itself. Automate the feedback loop: increase TTL while p99 is stable, decrease it when p95 rises, and alert when the two signals disagree.
- **Requirements live outside the code.** A transaction-list endpoint set to a 10-minute TTL is fine until a feature launch tightens the freshness requirement to 30 seconds. If that requirement lives only in a product spec, no engineer will update the TTL. Put the TTL in the endpoint's OpenAPI spec and add a CI check that rejects TTL values above the documented maximum.

## Decision checklist

Before you ship the next feature, confirm each of these has a named owner:

| Item | Question to answer | Where it should be recorded |
|---|---|---|
| Cache TTL | Who decides the value, and what is the maximum allowed? | Service config plus OpenAPI spec |
| Eviction policy | Who owns the eviction strategy and its tuning? | Cache module README |
| Invalidation | Who publishes invalidation events, and who consumes them? | Pub/sub channel documentation |
| SLO | What is the p99 target, and who is paged when it is breached? | Alerting rules with owner label |
| Write access | Who is permitted to write to the cache in production? | IAM or network policy |

If any row is blank, that is the drift you are about to discover the hard way.

## Common questions and variations

**What if the cache and service are in different repositories?**
Pin versions explicitly — `cache-sdk@1.2.3` rather than `latest` — and add a CI check that the proxy never depends on an unreleased cache version. Relying on `latest` produces silent upgrades that break the proxy in ways that are hard to attribute.

**How do you handle cache invalidation across services?**
Use a pub/sub channel. When the data service publishes an invalidation event for a key, every gateway subscribes and drops its local copy. This shifts ownership of invalidation policy to the data service rather than the gateway team. Keep pub/sub for explicit invalidation and let TTL handle the rest; setting TTL very low to compensate for missing invalidation reintroduces stampedes.

**What if you're using a managed cache?**
Managed caches don't remove ownership drift; they relocate it. The same principles apply: declare an owner for TTL policy, key naming and invalidation strategy. Letting each team set its own TTL produces conflicting policies, and the fix is to centralise TTL decisions in a config service owned by the platform team, with a documented deprecation window for changes.

**How do you cost this?**
Work it out from your own numbers. Take your current instance type's hourly on-demand price, multiply by the number of instances, and add your observability spend per metric series. Compare that against the cost of unplanned work: count the hours your team spent on cache-related incidents in the last quarter, multiply by a loaded hourly rate, and divide by three to get a monthly figure. The comparison is only meaningful with your own inputs; published benchmarks from other environments rarely transfer.

## Where to go from here

In the next 30 minutes: open your repository's README and add a single line under an "Ownership" heading naming the cache layer's owner, its TTL, and the endpoint that exposes this contract — for example, `Cache layer: owned by platform-team-cache. TTL: 30s. Stampede protection enabled. See /cache/meta.` If your service already exposes a metrics endpoint, add an `owner` label to the p99 latency histogram so that a breach routes to a team rather than to a channel. Do this before the next feature spike, not after the next incident.
