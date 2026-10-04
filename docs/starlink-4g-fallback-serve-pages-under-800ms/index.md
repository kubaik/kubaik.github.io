# Starlink 4G fallback: serve pages under 800ms

Satellite and rural 4G links change the shape of a performance problem. A Starlink terminal or a new 4G cell can give a village a 30–60 ms RTT link to a nearby ground station, but the last hop to the handset is still a cheap Android phone on a congested carrier network. The radio upgrade does not upgrade the device, the browser, or the TCP stack.

This article is about the serving strategy that survives that mismatch: stream a tiny HTML shell fast, keep the client bundle small enough to download on a 1 Mbps link, and cache at the edge so a slow origin never blocks first paint.

## The failure mode to design against

A common failure mode is treating "4G" as a synonym for "fast". A 4G radio can deliver 1 Mbps or 20 Mbps depending on load, and the client device is often the bottleneck. A 256 MB RAM Android phone running a current Chrome build spends measurable time on JavaScript parse and image decode before it can paint anything useful.

Two measurements make the problem concrete:

- **Effective connection type (ECT)** reported by the browser, via the Network Information API or the `Save-Data` and `Downlink` request headers. These are hints, not guarantees, but they are the only signal the server gets before sending bytes.
- **Time to First Byte (TTFB)** measured at the client, not at the load balancer. A CDN edge that responds in 40 ms is useless if the HTML depends on a 400 ms origin round trip.

The design goal used throughout this article is a TTFB of 800 ms or less on a 1 Mbps downlink with 280 ms RTT and 10% packet loss. That is a deliberately harsh profile; if the page meets it, ordinary 4G will feel fast.

## Prerequisites and target artefacts

The stack assumed here is Node 20 LTS, a current Next.js App Router release, and a Redis-compatible cache. Any streaming SSR framework works; the pattern is what matters.

The build should produce three artefacts:

1. A streamed HTML shell of roughly 1–2 kB that renders a skeleton immediately.
2. Critical CSS inlined in the shell, with the rest deferred.
3. A client bundle under 120 kB gzipped, loaded only after the shell is interactive.

You can reproduce the network conditions locally without a real device. Chrome DevTools supports a custom throttling profile, and `tc netem` on Linux can add latency and loss at the interface level:

```bash
# Add 280ms RTT and 10% loss on eth0 (requires root)
sudo tc qdisc add dev eth0 root netem delay 140ms loss 10%
# Remove it later
sudo tc qdisc del dev eth0 root netem
```

Note that `delay 140ms` produces roughly 280 ms RTT because the delay applies in each direction. Verify with `ping` before trusting any measurement.

## Step 1 — set up the project and the cache

Create a project and install the runtime dependencies:

```bash
npx create-next-app@latest --typescript --eslint --tailwind --src-dir --import-alias '@/*'
cd my-app
npm install redis
```

Create a Redis client module. The important part is not the library but the timeouts: a cache call that hangs for 30 seconds is worse than a cache miss.

```typescript
// src/lib/redis.ts
import { createClient } from 'redis';

const client = createClient({
  socket: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    connectTimeout: 5000,
  },
  password: process.env.REDIS_PASSWORD,
});

client.on('error', (err) => console.error('Redis client error', err));

// Connect once per process. In serverless runtimes, reuse the connection
// across invocations via a module-level singleton.
let connected = false;
export async function getClient() {
  if (!connected) {
    await client.connect();
    connected = true;
  }
  return client;
}

export default client;
```

A Redis-compatible cache should live in the same region as the edge function that reads it. Cross-region cache reads reintroduce exactly the latency the cache was meant to remove.

Next, add a route that classifies the connection from request headers. The `Downlink` and `Save-Data` headers are client hints; they must be requested with `Accept-CH` before the browser sends them.

```typescript
// src/app/api/edge/route.ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const saveData = request.headers.get('Save-Data') === 'on';
  const downlinkHeader = request.headers.get('Downlink');
  const downlink = downlinkHeader ? parseFloat(downlinkHeader) : null;

  let ect = '4g';
  if (saveData) {
    ect = 'slow-2g';
  } else if (downlink !== null && downlink < 1) {
    ect = '2g';
  }

  return NextResponse.json({ ect, downlink, saveData });
}
```

Test it locally:

```bash
curl -H 'Save-Data: on' http://localhost:3000/api/edge
# {"ect":"slow-2g","downlink":null,"saveData":true}
```

Treat `ect` as a hint for choosing compression level and image quality, never as a hard gate. A user on a fast link who has `Save-Data: on` still deserves the full page, just delivered more cheaply.

## Step 2 — stream a small HTML shell

The core idea is that the first byte of HTML should not depend on any data fetch. Render the skeleton, stream it, and let the data arrive behind it.

```typescript
// src/app/page.tsx
import { Suspense } from 'react';
import Script from 'next/script';

export const dynamic = 'force-dynamic';

function CriticalSkeleton() {
  return (
    <main>
      <header>
        <h1>Regional News</h1>
      </header>
      <section aria-busy="true">
        {Array.from({ length: 5 }).map((_, i) => (
          <article key={i} className="skeleton-line" />
        ))}
      </section>
    </main>
  );
}

async function ArticleList() {
  // This fetch happens after the shell has already been flushed.
  const res = await fetch(`${process.env.ORIGIN}/api/data`, {
    cache: 'no-store',
  });
  const data = await res.json();
  return (
    <ul>
      {data.items.map((item: { id: number; title: string }) => (
        <li key={item.id}>{item.title}</li>
      ))}
    </ul>
  );
}

export default function Home() {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta httpEquiv="Accept-CH" content="Downlink, Save-Data" />
        <title>Regional News</title>
        <style
          dangerouslySetInnerHTML={{
            __html: `
              body { margin: 0; font-family: system-ui, sans-serif; background: #fff; color: #111; }
              .skeleton-line { height: 1.25rem; margin: 0.75rem 0; background: #eee; border-radius: 4px; }
            `,
          }}
        />
      </head>
      <body>
        <CriticalSkeleton />
        <Suspense fallback={null}>
          <ArticleList />
        </Suspense>
        <Script
          id="load-client"
          src="/client-bundle.js"
          strategy="lazyOnload"
        />
      </body>
    </html>
  );
}
```

Two details matter for the 800 ms target:

- The skeleton is rendered by the server, so it paints as soon as the first HTML chunk arrives. No JavaScript is required for first paint.
- `ArticleList` is wrapped in `Suspense`, so the shell flushes before the data fetch resolves. The fetch latency is paid after the user already sees something.

Avoid `will-change` and large `transform` layers in the skeleton CSS. On a 256 MB device, compositing layers cost memory that the device does not have.

## Step 3 — keep the client bundle small

The client bundle is the part most likely to blow the budget. A React tree with polyfills for old browsers can easily exceed 300 kB gzipped. The fix is to target a modern baseline and drop the polyfills.

```typescript
// src/client/client.tsx
'use client';
import { useEffect, useState } from 'react';

type Item = { id: number; title: string };

export default function App() {
  const [items, setItems] = useState<Item[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retries = 0;
    const maxRetries = 3;

    const load = async () => {
      try {
        const res = await fetch('/api/data');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!cancelled) setItems(data.items);
      } catch {
        retries += 1;
        if (retries <= maxRetries) {
          const delay = Math.min(1000 * 2 ** retries, 5000);
          setTimeout(load, delay);
        } else if (!cancelled) {
          setItems([]);
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (items === null) return null;

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>{item.title}</li>
      ))}
    </ul>
  );
}
```

Bundle it with esbuild, targeting ES2020 so that no polyfills are emitted for features every current browser supports:

```bash
npx esbuild src/client/client.tsx \
  --bundle \
  --outfile=public/client-bundle.js \
  --target=es2020 \
  --minify \
  --format=esm
```

Then measure the actual transferred size, not the file size on disk. Compression changes the number substantially:

```bash
gzip -c public/client-bundle.js | wc -c
# Also measure brotli, which is what most CDNs will serve
brotli -c public/client-bundle.js | wc -c
```

If the brotli figure is above the budget, the usual culprits are a date library, a state management library, or an icon set imported wholesale. Replace or lazy-load them before touching the server.

## Step 4 — cache and compress at the edge

The API route should read from cache, write back on miss, and stream the response so the first byte is not held back by serialization.

```typescript
// src/app/api/data/route.ts
import { NextResponse } from 'next/server';
import { getClient } from '@/lib/redis';

const CACHE_KEY = 'homepage:data';

async function loadData() {
  const client = await getClient();
  const cached = await client.get(CACHE_KEY);
  if (cached) return { value: cached, hit: true };

  const data = JSON.stringify({
    items: Array.from({ length: 20 }, (_, i) => ({
      id: i,
      title: `Article ${i}`,
    })),
  });
  await client.set(CACHE_KEY, data, { EX: 60 });
  return { value: data, hit: false };
}

export async function GET() {
  const { value, hit } = await loadData();

  return new NextResponse(value, {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300',
      'X-Cache': hit ? 'HIT' : 'MISS',
    },
  });
}
```

Two notes on compression. First, do not set `Content-Encoding: br` manually unless the body is actually brotli-compressed; a mismatched header produces a decode error in the browser. Let the CDN or the framework negotiate compression via `Accept-Encoding`. Second, brotli quality 11 is expensive at request time; most CDNs serve a precompressed quality 11 asset or a dynamic quality 4–5. The difference in size between quality 4 and quality 11 on typical HTML is small, while the CPU difference is large.

To verify the edge cache is working, compare the `X-Cache` header across two requests and check the timing:

```bash
curl -s -o /dev/null -w '%{time_starttransfer}\n' https://your-host/api/data
curl -s -o /dev/null -w '%{time_starttransfer}\n' https://your-host/api/data
```

The second request should be markedly faster. If it is not, the cache key is probably varying on a header or cookie it should not.

## Step 5 — handle the failure cases

Four failure modes account for most of the pain on constrained links.

**The client bundle never arrives.** Add an error handler that offers a retry rather than leaving a blank page:

```typescript
<Script
  id="load-client"
  src="/client-bundle.js"
  strategy="lazyOnload"
  onError={() => {
    const el = document.createElement('div');
    el.innerHTML = '<p>Content loaded slowly. <button type="button">Retry</button></p>';
    el.querySelector('button')?.addEventListener('click', () => window.location.reload());
    document.body.appendChild(el);
  }}
/>
```

**The fetch fails mid-request.** The retry loop in Step 3 handles this with exponential backoff capped at 5 seconds. Cap the retry count; on a 10% loss link, unlimited retries turn a slow page into a hung one.

**The device runs out of memory.** Cap the number of skeleton rows and rendered list items to what fits the viewport. Rendering 20 list items on a 256 MB device to save a scroll is a bad trade.

**The origin or CDN is unreachable.** A service worker can serve a cached shell and a fallback page:

```javascript
// public/sw.js
const CACHE = 'shell-v1';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(['/', '/offline.html']))
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.mode !== 'navigate') return;
  event.respondWith(
    fetch(event.request).catch(() =>
      caches.match('/offline.html').then((cached) => cached || caches.match('/'))
    )
  );
});
```

Register it from the layout, and version the cache name whenever the shell changes so old shells are evicted.

## Step 6 — measure, don't assume

Every claim in this article is a design target, not a measured result. To find out whether your page meets it, instrument these four numbers:

| Metric | How to capture | Target |
|---|---|---|
| TTFB | `PerformanceNavigationTiming.responseStart - requestStart` | ≤ 800 ms |
| HTML shell size | `Content-Length` on the first response | ≤ 2 kB |
| Client bundle size | `brotli -c bundle.js \| wc -c` | ≤ 120 kB |
| Cache hit ratio | `X-Cache: HIT` count / total requests | > 80% |

A small RUM snippet captures the first two without a third-party dependency:

```typescript
// src/app/layout.tsx
'use client';
import { useEffect } from 'react';

export function RUM() {
  useEffect(() => {
    const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (!nav) return;
    const ttfb = nav.responseStart - nav.requestStart;
    navigator.sendBeacon('/api/rum', JSON.stringify({ ttfb, path: location.pathname }));
  }, []);
  return null;
}
```

For lab measurements, Lighthouse with a throttling profile is useful but noisy on low-end devices. Run at least five iterations and compare medians rather than single runs. WebPageTest offers a 4G profile and a low-end device preset; use it when you need a repeatable number.

To measure the effect of a change, hold everything constant except the variable under test. If you reduce the bundle from 200 kB to 100 kB, the expected download time on a 1 Mbps link falls by roughly 800 ms, since 100 kB is about 800 kilobits. That is arithmetic from stated assumptions, not a benchmark result. Whether the real page improves by that much depends on what was blocking on the bundle.

## Common questions

**Does this work without Next.js?**
Yes. The pattern is framework-independent: stream a shell, inline critical CSS, keep the client bundle small, cache at the edge. In Express, that means `res.write()` for the shell, `compression` middleware for the response, and a Redis read before any origin call. The specific APIs differ; the ordering does not.

**Should I detect the connection and serve different bundles?**
Serving two bundles doubles the cache surface and risks serving the wrong one. Prefer one small bundle that works everywhere. Use `Save-Data` to reduce image quality and skip non-essential requests, not to fork the application.

**How do I handle images?**
Serve a tiny placeholder inline and let the real image load lazily. A 1×1 base64 PNG as `blurDataURL` is around 85 bytes and paints immediately:

```typescript
<Image
  src="/hero.jpg"
  alt=""
  width={1200}
  height={675}
  placeholder="blur"
  blurDataURL="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
  sizes="100vw"
/>
```

Set `sizes` accurately so the browser does not fetch a 1200 px image for a 320 px viewport.

**What about third-party scripts?**
Load them after the shell is interactive, and never let one block first paint. If an ad or analytics script is not essential to the first view, defer it and measure its cost separately with the Performance panel's bottom-up view.

## What to do in the next 30 minutes

Open your production site in Chrome DevTools, set a custom throttling profile of 1 Mbps down, 280 ms RTT and 10% packet loss, then reload and record `responseStart - requestStart` for the HTML document. If that number is above 800 ms, the problem is on the server or the edge, not in the bundle, and the fix is to stream the shell before any data fetch. If it is below 800 ms but the page still feels slow, measure the brotli-compressed size of your largest JavaScript file with `brotli -c file.js | wc -c` and compare it against the 120 kB budget.
