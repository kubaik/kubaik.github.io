# Ship to Nairobi users: Starlink latency guide 2026

Median latency is a comfortable number to optimise against, and it is the wrong one. On congested mobile networks, the user experience is governed by the tail: the 95th and 99th percentile requests, where a single half-second stall can make a user abandon a form and never return. This article covers how to move cache and prefetch logic to a CDN edge function so that tail latency is absorbed before the user's browser ever waits on origin.

The guidance is written for teams shipping to users on mixed connectivity — fibre, consumer satellite, and congested 4G — where the last mile is the dominant source of variance.

## Why tail latency, not bandwidth, is the design target

Bandwidth on modern mobile networks is usually adequate. What varies is the time it takes for a connection to become usable and for the first byte of a response to arrive. Three mechanisms drive that variance:

**Connection setup.** A TLS handshake plus TCP slow start costs multiple round trips before any payload moves. On a link with 150 ms RTT, that is a fixed tax on every new connection.

**Retransmission and handoff.** Packet loss on a congested cell, or a tower handoff while the user is moving, produces stalls of hundreds of milliseconds. These do not show up in the median at all.

**Render-blocking resources.** An HTML document that blocks on a synchronous API call before painting anything leaves the user staring at a blank screen for the full duration of the slowest dependency.

The practical consequence: a page whose median load is 2 seconds can still have a 95th percentile above 4 seconds, and it is the tail that generates support contacts and abandonment. Optimising the median further yields little; reducing the number of sequential round trips between the browser and origin yields a lot.

The design principle that follows is simple: **move the decision-making that determines what the browser fetches next to a location close to the user**, so the browser can start work in parallel instead of waiting for origin.

## Prerequisites and what you will build

You will need:

- A Node.js 20 LTS backend (Express 4.x is sufficient; any HTTP framework works)
- A CDN distribution that supports edge functions at the viewer-request stage
- A Redis 7.x instance, ideally with automatic failover, for origin-side caching
- A way to test on a real congested mobile connection — a phone on a local 4G network is more honest than a throttled desktop browser

What you will build:

- A small edge function that rewrites HTML responses, injects a prefetch hint for the slowest API route, and sets a short `stale-while-revalidate` window on dynamic responses
- An origin cache with a bounded in-process fallback so that a cache failover does not become an outage
- A measurement loop using `curl` timing output and a load test, so that claims about improvement are backed by numbers you produced yourself

The edge function itself is roughly 50 lines. The interesting part is the failure handling and the measurement.

## Step 1 — set up the origin service

Create a project and install dependencies:

```bash
npm init -y
npm install express@4.19 redis@4.6.12 zod@3.22.4
```

A minimal Express service with a Redis-backed cache:

```javascript
// server.js
import express from 'express';
import { createClient } from 'redis';
import { z } from 'zod';

const app = express();
const port = process.env.PORT || 3000;

const redis = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
  socket: { reconnectStrategy: (retries) => Math.min(retries * 100, 5000) }
});

await redis.connect();

const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number().nonnegative()
});

app.get('/api/products/:id', async (req, res) => {
  const id = req.params.id;
  const cacheKey = `prod:${id}`;
  const cached = await redis.get(cacheKey);

  if (cached) {
    return res.json(JSON.parse(cached));
  }

  const mock = { id, name: 'Sample Product', price: 29.99 };
  await redis.set(cacheKey, JSON.stringify(mock), { EX: 30 });
  res.json(mock);
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
```

Run Redis locally:

```bash
docker run --rm -p 6379:6379 redis:7.2-alpine
```

Start the server, then verify the cache warms:

```bash
curl -s http://localhost:3000/api/products/123 | jq .
```

Expected output:

```json
{
  "id": "123",
  "name": "Sample Product",
  "price": 29.99
}
```

For a managed Redis deployment, the relevant settings are: a subnet group spanning at least two availability zones, a node with enough memory for your working set, cluster mode disabled unless the dataset exceeds roughly 10 GB, and encryption in transit enabled. Cluster mode is a real option for large datasets, but for a cache in the single-digit-gigabyte range it adds client-side complexity — cross-slot command errors are a common first encounter — without a latency benefit.

Record the primary endpoint and export it:

```bash
export REDIS_URL=redis://prod-cache.example.abc123.use1.cache.amazonaws.com:6379
```

## Step 2 — the edge function

The edge function runs at the viewer-request stage. It does two things: it rewrites HTML to include a prefetch hint, and it applies a short cache window to dynamic API responses.

```javascript
// edge-function.js
async function handler(event) {
  const request = event.request;
  const uri = request.uri;

  // Static assets are served from object storage; leave them alone.
  if (uri.startsWith('/static/')) {
    return request;
  }

  // HTML: inject a prefetch hint and set a short shared cache window.
  if (uri.endsWith('.html') || uri === '/') {
    const newHeaders = {
      'content-type': { value: 'text/html; charset=utf-8' },
      'cache-control': { value: 'public, s-maxage=60, stale-while-revalidate=30' }
    };

    const newBody = `
<!doctype html>
<html>
<head>
  <link rel="prefetch" href="/api/products/123" as="fetch" crossorigin>
</head>
<body>
  <h1>Loading...</h1>
  <script>fetch('/api/products/123').then(r=>r.json()).then(d=>console.log(d))</script>
</body>
</html>
    `.trim();

    return {
      statusCode: 200,
      statusDescription: 'OK',
      headers: newHeaders,
      body: newBody
    };
  }

  // Dynamic API: cache at the edge with stale-while-revalidate.
  if (uri.startsWith('/api/')) {
    const cacheKey = `api:${uri}`;
    const cached = await caches.default.match(cacheKey);

    if (cached) {
      return cached;
    }

    const upstreamResp = await fetch(request);
    const clone = upstreamResp.clone();

    event.waitUntil(
      caches.default.put(
        cacheKey,
        new Response(clone.body, {
          headers: { ...clone.headers, 'cache-control': 's-maxage=60, stale-while-revalidate=300' }
        })
      )
    );

    return upstreamResp;
  }

  return request;
}
```

A note on the runtime: edge function runtimes differ in what they permit. Some are deliberately restricted to sub-millisecond CPU budgets and do not expose `fetch` or a cache API at all; others allow network calls and a persistent cache but bill per request at a higher rate and add tens of milliseconds. Check which class your CDN offers before writing the function. If your provider's viewer-request stage is CPU-only, the HTML rewrite still works and the API caching belongs at the origin-shield or origin-cache layer instead.

Deploy and attach the function using your provider's CLI. The shape of the commands is consistent across providers — create, publish, then reference the published ARN in the distribution's viewer-request association:

```bash
aws cloudfront create-function \
  --name html-prefetch-v1 \
  --function-config Comment="Prefetch HTML assets", Runtime=cloudfront-js-2.0 \
  --function-code fileb://edge-function.js

aws cloudfront publish-function --name html-prefetch-v1 --if-match <ETag>
```

## Step 3 — measure before you claim anything

Do not trust a dashboard that reports averages. Measure connection timing directly from a client on the network you care about.

```bash
curl -w "\nLookup: %{time_namelookup}s Connect: %{time_connect}s TLS: %{time_appconnect}s Pretransfer: %{time_pretransfer}s Starttransfer: %{time_starttransfer}s Total: %{time_total}s\n" \
  -o /dev/null -s https://cdn.example.com/
```

Run this 30 times on a real 4G connection and keep the raw output. The fields that matter are `time_starttransfer` (time to first byte of the response body) and `time_total`. Sort the results and look at the 95th percentile, not the mean.

To instrument the same measurement continuously, emit the timing values from your load test rather than from a single shell session. A minimal k6 script:

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 50,
  duration: '5m',
  thresholds: {
    http_req_duration: ['p(95)<800']
  }
};

export default function () {
  const res = http.get('https://cdn.example.com/api/products/123');
  check(res, {
    'status is 200': (r) => r.status === 200
  });
}
```

Run it from a host on the same network path as your users:

```bash
docker run --rm -i grafana/k6 run - < load-test.js
```

The threshold `p(95)<800` is a decision you make, not a fact about the world. Pick it from your product's tolerance for a slow first paint, and treat a breach as a regression to investigate.

### What to instrument

| Layer | Metric | Why it matters |
|---|---|---|
| Edge function | Invocation duration, p99 | Confirms the edge is not becoming the bottleneck |
| Edge cache | Hit ratio on dynamic routes | Directly determines origin round trips |
| Origin cache | Hit ratio, eviction rate | Working set vs. provisioned memory |
| Origin | Error rate, retransmit-adjacent timeouts | Distinguishes cache problems from network problems |
| Client | `time_starttransfer` p95 from a real network | The only number the user experiences |

If you want a single target: the client-side `time_starttransfer` p95 is the one to defend.

## Step 4 — failure modes and their fixes

### Cache failover becomes an outage

A managed cache with automatic failover will have a window — typically seconds to tens of seconds — where reads fail while the primary moves. If your request path treats a cache miss and a cache error identically, every request in that window hits origin simultaneously.

The fix is a bounded in-process fallback: keep the last known value for each key for a short period, and serve it stale rather than failing.

```javascript
// server.js — add after redis.connect()
const fallbackStore = new Map();

app.get('/api/products/:id', async (req, res) => {
  const id = req.params.id;
  const cacheKey = `prod:${id}`;

  try {
    const cached = await redis.get(cacheKey);
    if (cached) return res.json(JSON.parse(cached));
  } catch (e) {
    console.warn('Cache read failed:', e.message);
  }

  // Serve last known value for up to 5 seconds after a cache failure.
  const stale = fallbackStore.get(cacheKey);
  if (stale && (Date.now() - stale.ts) < 5000) {
    return res.json(stale.data);
  }

  const mock = { id, name: 'Sample Product', price: 29.99 };
  fallbackStore.set(cacheKey, { data: mock, ts: Date.now() });

  try {
    await redis.set(cacheKey, JSON.stringify(mock), { EX: 30 });
  } catch (e) {
    console.warn('Cache write failed:', e.message);
  }

  res.json(mock);
});
```

Note that the fallback store is per-process. With N origin instances you get N independent copies, which is fine for a 5-second stale window and wrong for anything longer. Do not let the fallback TTL drift upward to paper over a persistent cache problem; alert on cache errors instead.

To test the failover path, reboot a cache node while a load test is running and watch the origin error rate:

```bash
aws elasticache reboot-cache-cluster \
  --cache-cluster-id prod-cache \
  --cache-node-ids-to-reboot 0001
```

Run the load test in parallel and compare the error rate during the reboot window against the baseline. If the fallback is working, the error rate should stay near baseline; if it spikes, the fallback window is too short or the fallback is not being consulted.

### Prefetch hints that do nothing

A `prefetch` hint is advisory. Browsers may ignore it entirely depending on the `as` value, the `crossorigin` attribute, and whether credentials are involved. A cross-origin prefetch without `crossorigin` will be discarded, and a prefetch for a credentialed endpoint will be dropped unless the fetch is issued with matching credentials mode.

The practical approach is to verify with the browser rather than assume. Open DevTools, filter the network panel to the prefetched URL, and confirm a second request is not issued when the page's own script fetches it. If you see two requests, the prefetch did not populate the cache and is pure overhead.

### Long URLs and cache key explosion

Edge functions have limits on request-line size, and cache keys built from full URLs fragment badly when query strings carry tracking parameters. Two mitigations:

Canonicalise the cache key by stripping known-irrelevant parameters:

```javascript
function canonicalKey(uri) {
  const [path, query] = uri.split('?');
  if (!query) return path;
  const keep = new URLSearchParams(query);
  for (const p of ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid']) {
    keep.delete(p);
  }
  const rest = keep.toString();
  return rest ? `${path}?${rest}` : path;
}
```

Hash genuinely long query strings into a fixed-length key:

```javascript
import { createHash } from 'node:crypto';

const shortId = createHash('sha256')
  .update(req.originalUrl)
  .digest('hex')
  .slice(0, 16);
```

The hash approach collapses distinct inputs into one key, so it is only safe when the cacheable response does not depend on the parameters you hashed away. For anything user-specific, do not hash — scope the key explicitly and keep the TTL short.

### User-specific data leaking across users

Caching per-user responses at a shared edge is a data-leak risk. If you must cache them, include a stable user identifier in the key and keep the TTL short enough that staleness is not a correctness problem:

```javascript
const userKey = `user:${hash(userId)}:cart`;
await redis.set(userKey, JSON.stringify(cart), { EX: 10 });
```

A ten-second TTL on a shopping cart is defensible. A sixty-second TTL is not, because a failover or a delayed invalidation can serve one user's cart to another. When in doubt, do not cache authenticated responses at the edge at all; cache them at the origin instead, where the key space is under your control.

### WebSockets

Viewer-request edge functions generally do not participate in WebSocket upgrades. If your application needs bidirectional streaming, terminate the WebSocket at a regional endpoint and use the edge only for the initial HTTP handshake and static assets.

## A worked decision example

Suppose a team is deciding whether to add an edge cache layer for a dashboard used by customers on mixed connectivity. The reasoning, with numbers labelled as illustrative:

Assume 100,000 page views per day. Assume the current median time to first byte is 1.2 s and the 95th percentile is 3.5 s, measured from a real client on a 4G network. Assume the origin API responds in 80 ms when the cache is warm and 400 ms when it is cold, and that the origin cache hit ratio is 70%.

Each uncached request costs an extra 320 ms of origin time. With a 70% hit ratio, 30,000 requests per day pay that cost. Moving the cache to the edge does not change the origin's cold-path cost, but it removes the round trip from the user's perspective for the 70% that would have hit the origin cache anyway — the user now gets the response from a nearby point of presence instead of a distant origin.

The decision hinges on two questions: what fraction of requests are cacheable at the edge (not just at origin), and whether the edge function's own execution time is small relative to the round trip it saves. If the edge function adds 50 ms and saves 150 ms, the trade is positive. If it adds 50 ms and saves 20 ms, it is not, and the effort belongs in reducing render-blocking resources instead.

There is no universal answer. Measure both numbers on your own network before committing.

## Checklist before shipping

- [ ] Client-side `time_starttransfer` p95 measured from a real congested network, not a throttled desktop
- [ ] Edge function duration p99 instrumented and confirmed below your provider's budget
- [ ] Origin cache failure path tested by rebooting a node under load
- [ ] Fallback stale window bounded and alerted on, not silently extended
- [ ] Prefetch hints verified in a browser network panel, not assumed
- [ ] Cache keys canonicalised; tracking parameters stripped
- [ ] Authenticated responses either not edge-cached or keyed per user with a short TTL
- [ ] Load test threshold chosen from product tolerance, and a breach treated as a regression

## Take this action now

Open your CDN's edge function file — or create one if you do not have one — and change the prefetch hint to point at the single API route your monitoring shows as slowest at the 95th percentile. Then run the `curl` timing command above thirty times from a phone on a congested mobile network, save the output, and compare the `time_starttransfer` p95 against a run with the prefetch hint removed. That comparison, not this article, is the evidence you should act on.
