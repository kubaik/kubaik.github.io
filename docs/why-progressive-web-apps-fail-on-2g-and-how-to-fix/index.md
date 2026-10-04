# Why progressive web apps fail on 2G and how to fix…

## The assumption that breaks PWAs on slow networks

A PWA is not slow on 2G because it is a PWA. It is slow because the architecture was designed against a fast, reliable link and never revisited. The failure mode is consistent: a large JavaScript bundle, a blocking script in the document head, images served at display resolution, and a client-side render that cannot paint anything until the bundle has downloaded, parsed, and executed.

On a 100 kbps link with packet loss, every one of those decisions compounds. A 2 MB bundle is not "a bit slow" on that link; it is roughly 160 seconds of pure transfer time before a single byte of application logic runs. That number alone explains most abandonment.

The useful mental model is a budget, not a checklist. Fix a target — for example, first contentful paint under 3 seconds on a throttled profile — and then allocate the budget across transport setup, HTML, critical CSS, fonts, images, and JavaScript. Anything that does not fit gets deferred, removed, or moved to a later interaction. The rest of this article works through where that budget actually goes and which changes recover the most of it.

## How to measure before you change anything

Optimization without measurement produces confident guesses. Before touching code, establish a repeatable profile and capture a baseline.

**Choose a throttle profile and write it down.** A defensible starting point for 2G-class testing is 100 kbps downlink, 50 kbps uplink, 300 ms round-trip latency, and a small packet-loss percentage. Record the exact numbers you use, because results are only comparable across runs that share a profile.

**Instrument the right metrics.** At minimum: time to first byte, first contentful paint, largest contentful paint, time to interactive, total transferred bytes, and the number of requests. On the client, capture `PerformanceNavigationTiming` and `PerformanceResourceTiming` entries so you can see which resources dominate the transfer. On the server or CDN, log time to first byte separately from origin processing time — a slow origin and a slow network look identical to the user but have different fixes.

**Run the test against a representative page.** A product detail page with a dozen images, a few API calls, and a form is more informative than a marketing homepage. Test both a cold cache and a warm cache, because the cold-cache path is what new users experience.

**Automate a budget check in CI.** A Lighthouse budget that fails a pull request when total transfer size or time to interactive regresses is the cheapest way to stop slow-network performance from eroding. Set the thresholds from your own baseline rather than copying arbitrary numbers; a regression gate is useful even if the absolute values are not yet where you want them.

Once you have a baseline, change one thing at a time and re-measure. The sections below are ordered roughly by how much of the budget they typically recover.

## The transport layer: connection setup is a real cost

On a high-latency link, connection establishment is not free. A TCP handshake plus a TLS handshake can consume several round trips before the first byte of HTML arrives. At 300 ms round-trip time, three round trips is most of a second spent on nothing but setup.

Two changes address this directly:

- **Terminate TLS close to the user.** Serving from edge locations reduces the round-trip time for the handshake itself. This is a CDN configuration change, not an application change, and it usually pays for itself immediately on slow links.
- **Reduce the number of origins.** Each additional hostname means another DNS lookup, another connection, and another handshake. Consolidating static assets and API traffic onto as few origins as practical removes setup cost that no amount of minification can recover.

HTTP/3 over QUIC is worth enabling where the CDN supports it, because it can reduce head-of-line blocking and improve behavior under loss. The honest caveat: QUIC runs over UDP, and some networks block or throttle UDP. Clients fall back to TCP, so enabling HTTP/3 is safe, but it should be treated as an improvement for the subset of users whose networks support it rather than a guaranteed win.

## Payload: the largest recoverable win

Transfer size dominates every other factor on a constrained link. The most effective changes are unglamorous.

**Compress text with Brotli.** Brotli is widely supported by modern browsers and CDNs, and it typically outperforms gzip on HTML, CSS, and JavaScript. Verify that your CDN is actually applying it — a surprising number of deployments serve uncompressed JSON because the content type was not in the compression list.

**Serve images in modern formats with a fallback.** AVIF and WebP both reduce image weight substantially compared with JPEG at comparable visual quality. The safe pattern is a `<picture>` element with format negotiation handled by the browser:

```html
<picture>
  <source type="image/avif" srcset="product-400.avif 400w, product-800.avif 800w" sizes="(max-width: 600px) 400px, 800px" />
  <source type="image/webp" srcset="product-400.webp 400w, product-800.webp 800w" sizes="(max-width: 600px) 400px, 800px" />
  <img src="product-800.jpg" alt="Product" loading="lazy" decoding="async" width="800" height="600" />
</picture>
```

Two details matter here. First, AVIF decoding is more CPU-intensive than JPEG decoding, and on low-end devices that cost can show up as a slower largest contentful paint even though fewer bytes were transferred. Measure both transfer size and paint timing before declaring victory. Second, always set intrinsic `width` and `height` so the browser can reserve layout space and avoid reflow as images arrive.

**Send only the data the screen needs.** API responses are frequently far larger than the rendered view requires. Field selection, whether through GraphQL or a REST endpoint that accepts a sparse field list, reduces bytes on the wire and parsing cost on the device. This is a backend change with a measurable frontend payoff.

## Rendering: get pixels on screen before JavaScript finishes

Client-side rendering is the single biggest structural obstacle on slow networks. If the initial HTML is an empty shell, the user sees nothing until the framework bundle has been fetched, parsed, executed, and has completed its first data fetch.

Server rendering addresses this directly. The server produces HTML that the browser can paint immediately, and JavaScript arrives later to make the page interactive. Streaming server rendering goes further: the server flushes the document head and above-the-fold markup as soon as they are ready rather than waiting for the slowest data source, so the user sees content while the rest of the page is still being assembled.

The trade-off is real and worth stating plainly. Server rendering moves work to the server, adds a runtime dependency for the initial paint, and makes debugging harder because errors occur in a different environment from the browser. It also does not eliminate JavaScript — it defers it. The correct framing is that server rendering changes *when* the user sees content, not whether the application eventually ships a bundle.

A related pattern is pre-rendering static routes at build time and serving them from the edge. For content that does not vary per user — documentation, catalogs, marketing pages — this is simpler than server rendering and equally effective, because the HTML is already sitting at the edge when the request arrives.

## Caching: make the second visit nearly free

A service worker that pre-caches the application shell and serves stale content while revalidating in the background can make repeat visits dramatically cheaper. The first visit still pays full cost, and the service worker installation itself adds some latency on that first load, so the benefit is concentrated in returning users.

Two rules keep this from becoming a liability:

- **Version your caches and delete old ones on activation.** A cache that is never invalidated will serve stale assets indefinitely.
- **Never cache responses that carry user-specific data unless you have thought carefully about who else might receive them.** A shared cache serving one user's account page to another is a security incident, not a performance bug.

For API responses, edge caching is often a better fit than an application-level cache tier. An in-process or network cache still requires a round trip to reach, and on a 300 ms link that round trip can consume the entire benefit of the cache hit. Caching at the edge removes the round trip as well as the origin work.

## Offline and flaky-connection behavior

On unreliable links, requests fail partway through and users lose work. The fix is to treat the network as optional for anything the user has already typed.

Queue mutations locally and replay them when connectivity returns. Give each queued operation a client-generated identifier so that a retry after an ambiguous failure does not create a duplicate record on the server — the classic case being a form submitted twice because the first response never arrived. Use exponential backoff with jitter for retries, and surface queue state in the interface so users know whether their submission has actually been accepted.

The storage layer for this is a design decision, not a library decision. IndexedDB is the browser primitive; whether you wrap it in a library is a matter of team preference. What matters is that the queue survives a page reload, that replay is idempotent, and that conflicts have a defined resolution rule rather than an implicit "last write wins."

## A worked budget example

The following figures are illustrative, chosen to show the arithmetic rather than to describe a measured system. Suppose the target is first contentful paint under 3 seconds on a 100 kbps link.

At 100 kbps, the practical throughput is roughly 12.5 kilobytes per second, and real-world overhead usually brings this down further — assume 10 kilobytes per second to be conservative. A 3-second budget therefore allows about 30 kilobytes of transfer before paint, minus whatever the handshake consumes.

If the handshake and TLS setup cost two round trips at 300 ms each, that is 600 ms, leaving 2.4 seconds, or roughly 24 kilobytes. Critical CSS for a simple layout might be 10 kilobytes compressed, the initial HTML 8 kilobytes, and a font subset 15 kilobytes — already over budget. The conclusion is not that the target is impossible; it is that the above-the-fold content must be assembled from a very small set of resources, and everything else must be deferred until after first paint.

This is why the ordering in this article matters. Shrinking images helps total transfer, but if the critical path is HTML plus CSS plus a font, image optimization does not move first paint at all. Instrument the critical path specifically, and optimize the resources that actually block rendering.

## Common failure modes

**Optimizing the wrong metric.** Reducing total page weight while time to interactive stays flat usually means the bottleneck is JavaScript execution, not transfer. Profile the main thread before adding more compression.

**Testing on a fast connection with throttling applied only to the network.** CPU throttling matters as much as network throttling on low-end devices. A bundle that downloads quickly can still take seconds to parse and execute on a slow processor.

**Assuming a modern format is a free win.** AVIF saves bytes and costs decode time. On low-end hardware the trade can go the wrong way for large hero images. Measure paint timing, not just transfer size.

**Shipping a performance budget that nobody enforces.** A budget in a document changes nothing. A budget that fails CI changes behavior.

**Treating offline support as a checkbox.** A queue without idempotent replay and a defined conflict rule will eventually duplicate or lose user data, which is worse than failing immediately.

## Choosing what to do first

| Constraint | Highest-leverage change |
|---|---|
| Empty shell until JavaScript loads | Server render or pre-render the initial route |
| Large image payloads | Modern formats with `<picture>` fallback and correct sizing |
| Slow first byte on repeat visits | Edge caching with correct cache keys and invalidation |
| Work lost on connection drops | Local queue with idempotent replay |
| Regressions after launch | Automated budget check in CI |

Start from the top row that matches your observed bottleneck, not from the row that sounds most modern.

## FAQ

**Does server rendering mean abandoning a client framework?**
No. The same components can render to HTML on the server and hydrate in the browser. The decision is about what the first response contains, not about which framework you use.

**Is HTTP/3 worth enabling?**
It is worth enabling because clients fall back to TCP when UDP is unavailable, so the downside is limited. Do not assume it will help every user; some networks block UDP entirely.

**How do I detect a slow connection in JavaScript?**
The Network Information API exposes `navigator.connection.effectiveType`, which reports values such as `slow-2g`, `2g`, `3g`, and `4g`. Support varies, so treat it as a hint for progressive enhancement rather than a hard gate:

```javascript
const connection =
  navigator.connection ||
  navigator.mozConnection ||
  navigator.webkitConnection;

const isSlow = connection
  ? /(^|-)2g$|^3g$/.test(connection.effectiveType)
  : false;
```

**Should I maintain a separate lightweight site for slow connections?**
Separate codebases double maintenance and drift over time. Prefer one codebase that renders a small critical path and defers everything else. If a separate experience is unavoidable, generate it from the same source rather than maintaining it by hand.

## Do this in the next 30 minutes

Open your slowest representative page in a browser with the network throttled to a 2G-class profile, and record the transfer size and request count of every resource that blocks first paint. That list — not the full page weight — is your optimization target. Pick the single largest item on it and either defer it, compress it, or remove it, then re-run the same profile and compare.
