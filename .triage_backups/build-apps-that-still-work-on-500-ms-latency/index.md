# Build apps that still work on 500 ms latency

## The problem: happy-path tutorials vs. high-latency reality

Most framework tutorials assume a 20–80 ms round trip. When the same code is deployed for users on high-latency links — satellite backhaul, congested mobile networks, or long-haul routes to a distant origin — the failure mode is consistent: pages that render a spinner and never finish, timeouts that appear random, and error rates that spike whenever the link quality dips.

The underlying reasons are mechanical, not mysterious. On a 500 ms RTT link, every round trip costs half a second. A TLS handshake is several round trips. A cache miss that triggers an origin fetch is another. An uncached asset is another. Add them up and a page that felt instant on a 30 ms link becomes unusable.

This article covers the specific techniques that keep a dashboard responsive under a 400–500 ms RTT budget, why each one works, and how to measure whether it's working in your environment.

## Why latency dominates bandwidth on slow links

Bandwidth determines how long a payload takes to transfer once the connection exists. Latency determines how long it takes to establish the connection and negotiate each request. On a link with 500 ms RTT and 1.6 Mbps throughput:

- A full TLS 1.2 handshake with an RSA certificate typically requires two round trips before application data flows — roughly 1 second of wall-clock time on a 500 ms RTT link.
- TLS 1.3 reduces this to one round trip for the initial handshake, and zero round trips on resumption.
- A 1.2 MB image at 1.6 Mbps takes about 6 seconds to transfer, but a 50 KB image takes about 250 ms.

The practical implication: shaving bytes off a large asset helps, but shaving round trips off the connection setup helps more, because round trips are multiplied by the RTT and bytes are divided by the bandwidth. On a 500 ms link, a single saved round trip is worth roughly 500 ms; a saved megabyte is worth roughly 5 seconds at 1.6 Mbps — but only if the megabyte was actually going to be transferred.

This is why certificate algorithm choice, HTTP version, connection reuse, and preloading matter disproportionately on high-latency links.

## Prerequisites and what you'll build

The examples below use a Next.js App Router dashboard, a Redis-compatible cache, an S3-compatible object store for static assets, and a CDN. None of these are mandatory — the techniques transfer to any stack. The specific tools are chosen because they have documented behavior that makes the reasoning concrete.

You'll need:

- Node.js 20 LTS or later
- Docker (for the local Redis instance)
- A CDN account with configurable TLS (Cloudflare, Fastly, or similar)
- An S3-compatible object store for static assets
- Lighthouse 11 or later, and Playwright for synthetic testing

The dashboard itself is intentionally small: a server component that reads a cached page payload, a client component that renders it, and a layout that preloads the critical chunk and the LCP image.

## Step 1 — set up the environment

1. Create the project and install dependencies:

```bash
npx create-next-app@latest latency-dashboard --typescript --app
cd latency-dashboard
npm install redis opossum
```

2. Create `.env.local` with your cache and asset host configuration:

```env
REDIS_URL="redis://localhost:6379/0"
ASSET_HOST="https://assets.yourdomain.com"
```

3. Start a local Redis instance:

```bash
docker run -d --name redis7 \
  -p 6379:6379 \
  redis:7.2
```

Redis 7.2 is a reasonable baseline because it supports the commands used below (`GET`, `SET` with `EX`, and `SET` with `NX` for lock-free stampede protection). If you only need string caching, the vanilla image is sufficient; managed Redis-compatible services work the same way.

4. Start the dev server:

```bash
npm run dev
```

Verify the app responds before adding any of the latency work, so you have a clean baseline to compare against.

## Step 2 — core implementation

The implementation has three layers: a cache that absorbs origin round trips, a layout that preloads critical resources, and a TLS configuration that minimizes handshake round trips.

### Layer 1: edge cache with a Redis-compatible store

Create `lib/cache.ts`:

```typescript
import { createClient } from 'redis';

const client = createClient({
  url: process.env.REDIS_URL,
  socket: {
    reconnectStrategy: (retries) => Math.min(retries * 100, 5000),
  },
});

client.on('error', (err) => console.error('redis error', err));

let connected = false;
export async function getClient() {
  if (!connected) {
    await client.connect();
    connected = true;
  }
  return client;
}

export async function getCachedPage(path: string) {
  const redis = await getClient();
  const hit = await redis.get(path);
  return hit ? JSON.parse(hit) : null;
}

export async function setCachedPage(path: string, data: unknown, ttlSeconds = 300) {
  const redis = await getClient();
  await redis.set(path, JSON.stringify(data), { EX: ttlSeconds });
}
```

A common bug in this pattern is calling `client.connect()` at module load time. On serverless platforms that reuse module instances across invocations, that can throw "already connected" on the second call. Tracking connection state explicitly avoids it.

### Layer 2: preload critical assets

In `app/layout.tsx`, preload the main client chunk and the LCP image so the browser can start fetching them while it parses HTML:

```tsx
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Dashboard',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="preload"
          as="image"
          href={`${process.env.ASSET_HOST}/hero.webp`}
          fetchPriority="high"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
```

The `fetchPriority="high"` attribute is a hint, not a guarantee; it works in Chromium-based browsers and is ignored elsewhere. The measurable effect is that the LCP image request starts earlier in the HTML parse, which on a 500 ms RTT link can save one round trip — roughly 500 ms.

### Layer 3: TLS configuration

TLS handshake cost is dominated by two things: the protocol version and the certificate algorithm. Two changes matter:

1. **Enable TLS 1.3.** It reduces the handshake to one round trip (or zero on resumption) instead of two. Most CDNs and modern web servers enable it by default; verify with:

```bash
openssl s_client -connect yourdomain.com:443 -tls1_3 </dev/null 2>/dev/null | grep "Protocol"
```

If the output shows `TLSv1.3`, it's enabled. If it shows `TLSv1.2`, check your server or CDN configuration.

2. **Use an ECDSA certificate where supported.** An ECDSA P-256 certificate produces smaller signatures than RSA, which reduces the bytes transferred during the handshake. The handshake round-trip count is the same, but the per-round-trip payload is smaller, which matters on constrained links.

To check what algorithm your current certificate uses:

```bash
echo | openssl s_client -connect yourdomain.com:443 2>/dev/null \
  | openssl x509 -noout -text \
  | grep -E "Public Key Algorithm|Signature Algorithm"
```

If the output says `rsaEncryption` and `sha256WithRSAEncryption`, you're serving an RSA certificate. Whether to switch depends on your client mix — see the failure-mode section below.

### Layer 4: static assets on a CDN with a nearby POP

Serve the LCP image and other static assets from a CDN POP that is geographically close to your users. The measurement that matters is not "is it on a CDN" but "what is the RTT from the user to the POP serving the asset."

To measure this from a test location, use `curl`'s timing breakdown:

```bash
curl -w "dns: %{time_namelookup}\nconnect: %{time_connect}\ntls: %{time_appconnect}\nttfb: %{time_starttransfer}\ntotal: %{time_total}\n" \
  -o /dev/null -s https://assets.yourdomain.com/hero.webp
```

The `time_connect` value approximates the TCP RTT to the POP. If it's above ~100 ms, the user is not hitting a nearby POP and you should check your CDN's routing configuration.

## Step 3 — handle edge cases and errors

### Cache stampede on cold start

When the cache restarts, every concurrent request to the same path misses and hits the origin. On a 500 ms RTT link, 100 concurrent misses means 100 origin round trips, which can saturate the origin and cascade into timeouts.

The fix is a lock: the first request to miss acquires a short-lived lock, fetches from origin, and populates the cache; concurrent requests either wait briefly or serve stale data.

```typescript
import { getClient, setCachedPage } from './cache';

export async function getCachedPageWithLock(path: string, fetchFromOrigin: () => Promise<unknown>) {
  const redis = await getClient();

  const hit = await redis.get(path);
  if (hit) return JSON.parse(hit);

  const lockKey = `lock:${path}`;
  const acquired = await redis.set(lockKey, '1', { NX: true, EX: 10 });

  if (!acquired) {
    // Another request is already fetching. Wait briefly, then read the cache.
    await new Promise((r) => setTimeout(r, 100));
    const retry = await redis.get(path);
    if (retry) return JSON.parse(retry);
    // Fall through and fetch ourselves rather than fail.
  }

  const data = await fetchFromOrigin();
  await setCachedPage(path, data, 300);
  await redis.del(lockKey);
  return data;
}
```

The `NX` flag makes the `SET` atomic: only one caller gets the lock. The `EX` expiry ensures a crashed holder doesn't deadlock the path.

### Circuit breaker for flapping links

On links that intermittently spike from 400 ms to over 1 second RTT, a request that would normally succeed can exceed the client's timeout. A circuit breaker stops sending requests to a failing dependency for a cooldown period, which prevents a slow dependency from consuming the entire request budget.

```typescript
import CircuitBreaker from 'opossum';

const breaker = new CircuitBreaker(
  async (path: string) => {
    const res = await fetch(`https://api.yourdomain.com/page/${path}`, {
      signal: AbortSignal.timeout(1200),
    });
    if (!res.ok) throw new Error(`status ${res.status}`);
    return res.json();
  },
  {
    timeout: 1200,
    errorThresholdPercentage: 50,
    resetTimeout: 30000,
  }
);

breaker.fallback(() => ({ stale: true, data: null }));

export async function fetchPage(path: string) {
  return breaker.fire(path);
}
```

The key parameter is `timeout: 1200`. It must be shorter than the overall request budget, so a single slow dependency doesn't consume the entire budget and cause the page to fail. The fallback returns a stale-data marker, which the UI can render as a degraded state rather than an error.

### Legacy client TLS fallback

Some older clients cannot validate ECDSA certificate chains. If you switch to ECDSA and see an increase in handshake failures, you have two options: serve an RSA certificate on a separate port for those clients, or terminate TLS at a CDN that negotiates the algorithm based on the client's capabilities.

If you control the server, a dual-certificate configuration looks like this:

```nginx
server {
  listen 443 ssl;
  http2 on;
  ssl_protocols TLSv1.2 TLSv1.3;
  ssl_certificate     /etc/ssl/fullchain-ecc.pem;
  ssl_certificate_key /etc/ssl/privkey-ecc.pem;
  # Additional RSA chain for clients that cannot negotiate ECDSA
  ssl_certificate     /etc/ssl/fullchain-rsa.pem;
  ssl_certificate_key /etc/ssl/privkey-rsa.pem;
}
```

The server presents both chains and the client picks. This is preferable to a separate port because it requires no client-side logic.

### Timeout budgets

Every layer that can wait needs an explicit timeout, and the timeouts must nest correctly. A typical budget for a 500 ms RTT link:

- Total request budget: 5 seconds
- Origin fetch timeout: 2 seconds
- Cache read timeout: 200 ms
- Circuit breaker timeout: 1.2 seconds (must be less than origin fetch timeout)

If a cache read can hang for 5 seconds, it will consume the entire request budget and the origin fetch will never run. Set cache timeouts aggressively low — a cache that takes more than 200 ms to respond is not providing value on a 500 ms RTT link anyway.

## Step 4 — observability and tests

### Lighthouse CI

Lighthouse's simulated throttling (default: 150 ms RTT, 1.6 Mbps) gives a reproducible baseline, but it does not simulate 500 ms RTT. To approximate a high-latency link, configure the throttling explicitly:

```yaml
- uses: treosh/lighthouse-ci-action@v11
  with:
    urls: |
      https://staging.yourdomain.com/dashboard
    uploadArtifacts: true
    temporaryPublicStorage: true
    configPath: ./lighthouserc.json
```

With `lighthouserc.json`:

```json
{
  "ci": {
    "collect": {
      "settings": {
        "throttling": {
          "rttMs": 500,
          "throughputKbps": 1600,
          "cpuSlowdownMultiplier": 4
        }
      }
    }
  }
}
```

The `rttMs: 500` setting is the important one. Lighthouse's default throttling is optimistic for high-latency targets.

### Real-device synthetic testing

Lighthouse simulates latency in software. To measure the real thing, run a Playwright test from a CI runner in a region with high RTT to your origin:

```typescript
import { test, expect } from '@playwright/test';

test('dashboard loads under 500 ms RTT', async ({ page }) => {
  const client = await page.context().newCDPSession(page);
  await client.send('Network.emulateNetworkConditions', {
    offline: false,
    downloadThroughput: (1600 * 1024) / 8,
    uploadThroughput: (750 * 1024) / 8,
    latency: 500,
  });

  const start = Date.now();
  await page.goto('https://staging.yourdomain.com/dashboard');
  await page.waitForSelector('[data-testid="dashboard-ready"]');
  const duration = Date.now() - start;

  console.log(`time to dashboard-ready: ${duration} ms`);
  expect(duration).toBeLessThan(5000);
});
```

The `data-testid` selector should be on an element that only renders after the dashboard data is available, not on a loading spinner. Otherwise the test measures the wrong thing.

### Alerting on TLS handshake time

Most CDNs expose TLS handshake duration as a metric. A reasonable alert threshold is p95 handshake duration above 200 ms — on a healthy TLS 1.3 connection with a nearby POP, handshakes should be well under that. If the metric rises, the likely causes are:

- A certificate change that broke session resumption
- A CDN POP change that increased RTT
- A protocol downgrade (TLS 1.3 to 1.2)

Each of these is diagnosable from the metric trend, which is why alerting on it is worth the setup cost.

## Failure modes and how to detect them

### Failure mode: certificate chain served out of order

If the server sends the leaf certificate before the intermediate, some clients fail to build a valid chain and fall back or fail outright. This is invisible in browsers that cache the intermediate from a previous connection, but visible on first visits.

Detection: run a chain validation from a clean client:

```bash
openssl s_client -connect yourdomain.com:443 -showcerts </dev/null 2>/dev/null \
  | openssl verify -CAfile /etc/ssl/certs/ca-certificates.crt
```

If the output is not `OK`, the chain is mis-ordered or incomplete.

### Failure mode: cache key collision across tenants

If the cache key is derived only from the URL path and not from the tenant or user, one user's cached page can be served to another. This is a correctness bug, not a performance bug, but it often appears after a caching change and is blamed on the cache being "stale."

Detection: include the tenant ID in the cache key and assert in tests that two different tenants get different payloads.

### Failure mode: preload hint on the wrong resource

`<link rel="preload">` on a resource that isn't used on the page consumes bandwidth without benefit, and on a 500 ms RTT link that bandwidth is precious. A preload for an image that turns out to be below the fold delays the LCP image.

Detection: check Lighthouse's "Preload key requests" and "Avoid chaining critical requests" audits. If a preloaded resource doesn't appear in the LCP critical path, remove the preload.

### Failure mode: circuit breaker fallback masks real errors

A fallback that returns `{ stale: true, data: null }` can hide an origin outage from the UI, which then renders an empty dashboard. Users see a working page with no data and don't report it.

Detection: emit a metric every time the fallback fires, and alert if the fallback rate exceeds a small threshold (for example, 1% of requests over 5 minutes).

## When this approach helps and when it doesn't

| Situation | Does this help? | Why |
|---|---|---|
| 400–500 ms RTT to origin, read-heavy dashboard | Yes, significantly | Round-trip savings dominate |
| 20–50 ms RTT, read-heavy dashboard | Marginally | Round trips are already cheap |
| Write-heavy workload | No | Cache doesn't help; focus on write path |
| Small payloads (< 100 KB) on fast links | No | Bandwidth isn't the bottleneck |
| Large payloads (> 1 MB) on slow links | Yes, but optimize payload size first | Bandwidth and round trips both matter |
| Users behind a flapping link | Yes, with circuit breaker | Prevents one slow dependency from failing the page |

## FAQ

**Why does TLS handshake time matter more than bandwidth on high-latency links?**

A TLS handshake requires round trips before any application data moves. On a 500 ms RTT link, each round trip costs half a second. TLS 1.3 reduces the handshake to one round trip (or zero on resumption), while TLS 1.2 with RSA requires two. That difference is 500 ms of wall-clock time on every new connection, regardless of how much bandwidth is available. Bandwidth only helps once the connection is established.

**Should I use ECDSA or RSA certificates?**

ECDSA P-256 is smaller and faster to verify than RSA 2048, and it's supported by all current browsers and TLS stacks. The exception is very old clients (Android 4.x, IE 11) that cannot validate ECDSA chains. If your traffic includes those clients, serve both certificate types and let the server negotiate. The measurement to run before switching is the handshake failure rate by client version, which most CDNs expose.

**Is Redis the right edge cache, or should I use a KV store?**

Redis is appropriate when you need sub-10 ms p99 reads, structured data types, or atomic operations like the `SET NX` lock shown above. A KV store with eventual consistency is simpler for small immutable values but doesn't support the atomic operations needed for stampede protection. If your payloads are large and read-heavy, Redis is usually the better fit; for small immutable values, either works.

**How do I test high-latency performance without being in the region?**

Use Lighthouse's `throttling.rttMs` setting for a quick simulated baseline, then add a Playwright test using CDP's `Network.emulateNetworkConditions` with `latency: 500`. Run the Playwright test from a CI runner in a region with high RTT to your origin, so the emulated latency is added on top of real network latency. This catches both code regressions and routing regressions.

## One action to take in the next 30 minutes

Run this command against your production origin and read the output:

```bash
curl -w "dns: %{time_namelookup}\nconnect: %{time_connect}\ntls: %{time_appconnect}\nttfb: %{time_starttransfer}\ntotal: %{time_total}\n" \
  -o /dev/null -s https://yourdomain.com
```

If `time_appconnect` minus `time_connect` is above 200 ms, your TLS handshake is costing more than one round trip on a typical connection and is the first thing to investigate. Check whether TLS 1.3 is enabled and whether session resumption is working before changing anything else.
