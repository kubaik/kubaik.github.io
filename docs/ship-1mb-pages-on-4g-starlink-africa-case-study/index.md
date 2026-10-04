# Ship 1MB pages on 4G: Starlink Africa case study

Most tutorials demonstrate the happy path: a fast laptop, a fast office connection, a fast origin. Production traffic is not like that. Mobile and satellite-backed links introduce latency asymmetry, jitter and burst loss that a lab benchmark will never show you. This article covers how to serve a large HTML page on a constrained mobile network, how to measure whether your changes actually helped, and which failure modes to plan for.

## Why large pages break on mixed satellite-terrestrial paths

A page that feels instant on office Wi-Fi can feel broken on a mobile link that partly traverses a satellite hop. Four properties of those paths matter:

1. **Latency asymmetry.** Uplink can be several times slower than downlink. Protocols and application patterns that assume roughly symmetric round trips, such as long-polling or chatty request/response loops, degrade badly.
2. **Variable RTT jitter.** Round-trip time can swing by tens of milliseconds within a single TCP flow. Congestion windows oscillate, and throughput becomes unstable even when average bandwidth looks fine.
3. **Burst loss.** Handoffs between beams or between satellite and terrestrial segments can drop a burst of packets over a few hundred milliseconds. That is long enough to stall TLS activity, trigger retransmission timeouts and stall a request that was otherwise progressing.
4. **Cost asymmetry.** Some satellite consumer plans meter or throttle uplink while treating downlink as cheap. For a provider, that means the bytes your users send cost more than the bytes they receive, which changes the economics of retries, telemetry and prefetching.

A typical failure mode: a team optimizes downlink payload size, ships a large win on paper, and still sees stalls because uplink retransmissions and handoff timeouts dominate the user-visible delay. The fix is not one setting. It is compression, transport choice, caching and measurement together.

## What you will build

By the end you will have:

- A Node 20 LTS + Express stack that serves a roughly 1 MB HTML page.
- Brotli compression with a correct gzip fallback and a correct `Vary: Accept-Encoding` header.
- A CDN in front of the origin with edge caching and stale-while-revalidate.
- Instrumentation that tells you transfer time, time to first byte and Core Web Vitals separately.
- A Lighthouse CI check that fails a pull request when a performance budget regresses.

You will need Node 20 LTS, npm, an AWS account with CloudFront, a domain you control, and optionally a CI runner. Exact provider pricing changes frequently, so treat any cost figure below as illustrative and verify current rates before committing.

## Step 1 — set up the environment

Start a fresh repo:

```bash
mkdir 4g-baseline && cd 4g-baseline
git init
node -v  # confirm a Node 20 LTS release
npm init -y
npm i express compression accepts
```

Create `server.js`:

```javascript
import express from 'express';
import compression from 'compression';
import fs from 'fs';

const app = express();

// Middleware order matters: compression must be registered before routes.
app.use(compression({ threshold: 0, filter: () => true }));

app.get('/', (req, res) => {
  const html = fs.readFileSync('index.html', 'utf8');
  res.set('Content-Type', 'text/html');
  res.send(html);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
```

Generate a synthetic page of roughly 1 MB:

```bash
node -e "
const fs = require('fs');
let s = '<html><body>';
for (let i = 0; i < 10000; i++) s += '<p>Lorem ipsum dolor sit amet...</p>';
s += '</body></html>';
fs.writeFileSync('index.html', s);
console.log('Wrote index.html:', fs.statSync('index.html').size, 'bytes');
"
```

The exact byte count depends on your loop content, so read the printed size rather than assuming a number.

Measure the raw transfer with `curl`:

```bash
curl -w "\nDNS: %{time_namelookup}s\nTCP: %{time_connect}s\nTLS: %{time_appconnect}s\nTotal: %{time_total}s\n" \
  --compressed http://localhost:3000/
```

Run it ten times and take the median. On localhost this mostly measures your own machine, so it is useful only as a baseline before you introduce network distance. The `compression` middleware negotiates gzip by default; the next step replaces that with Brotli and an explicit fallback.

## Step 2 — Brotli with a gzip fallback

Replace the default compression with a handler that negotiates the encoding itself:

```javascript
import express from 'express';
import fs from 'fs';
import { createBrotliCompress, createGzip } from 'zlib';
import accepts from 'accepts';

const app = express();

const BROTLI_PARAMS = {
  [zlib.constants.BROTLI_PARAM_QUALITY]: 6,
  [zlib.constants.BROTLI_PARAM_SIZE_HINT]: fs.statSync('index.html').size,
};

function brotliOrGzip(req, res, next) {
  const accept = accepts(req);
  const encoding = accept.type(['br', 'gzip']);

  res.set('Vary', 'Accept-Encoding');

  if (encoding === 'br') {
    res.set('Content-Encoding', 'br');
    fs.createReadStream('index.html')
      .pipe(createBrotliCompress({ params: BROTLI_PARAMS }))
      .pipe(res);
  } else if (encoding === 'gzip') {
    res.set('Content-Encoding', 'gzip');
    fs.createReadStream('index.html').pipe(createGzip()).pipe(res);
  } else {
    fs.createReadStream('index.html').pipe(res);
  }
}

app.get('/', brotliOrGzip);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Listening on ${PORT}`));
```

```bash
npm i accepts
```

Two details are easy to get wrong. First, `accepts` returns `false` when no acceptable type is found, so the fallback branch must handle that case rather than assuming a string. Second, `Vary: Accept-Encoding` must be present on every response that can differ by encoding, or a shared cache will serve the wrong body to the wrong client.

Brotli quality 6 is a common production choice: it captures most of the size reduction while keeping CPU cost reasonable. Do not assume a fixed ratio. Measure it:

```bash
gzip -9 -c index.html | wc -c
brotli -q 6 -c index.html | wc -c
```

If `brotli` is not installed, use the Node API to write both files to disk and compare their sizes. The ratio depends heavily on how repetitive your HTML is. A synthetic page of repeated paragraphs compresses far better than real markup with varied content, so treat any ratio from a synthetic test as an upper bound, not a promise.

Add resource hints to the head of the page:

```html
<head>
  <link rel="preconnect" href="https://cdn.example.com" crossorigin>
  <link rel="dns-prefetch" href="https://cdn.example.com">
</head>
```

Preconnect helps only for origins you actually contact early. Adding hints for origins you do not use costs a connection for nothing.

## Step 3 — Put a CDN in front

A CDN shortens the network path and lets you terminate modern transport closer to the user. With CloudFront, the settings that matter most for this workload are:

- Cache policy that includes `Accept-Encoding` in the cache key.
- A TTL long enough to get cache hits, with stale-while-revalidate so a revalidation does not block the response.
- HTTP/3 enabled so clients that support QUIC can use it.

A minimal Serverless Framework service to deploy the origin:

```yaml
service: 4g-baseline

provider:
  name: aws
  runtime: nodejs20.x
  region: us-east-1
  memorySize: 512
  timeout: 10

functions:
  app:
    handler: handler.handler
    events:
      - http: ANY /
      - http: ANY /{proxy+}

package:
  patterns:
    - '!node_modules/**'
    - 'index.html'
```

```bash
npm i -g serverless
serverless deploy --stage prod
```

Then verify what the edge actually returns:

```bash
curl -sI -H "Accept-Encoding: br" https://your-distribution.example.com/ | grep -i -E 'content-encoding|vary|cache-control'
curl -w "\nTotal: %{time_total}s\nSize: %{size_download} bytes\n" \
  -o /dev/null -H "Accept-Encoding: br" https://your-distribution.example.com/
```

Run the second command repeatedly and record the median. Do not compare a localhost number to a CDN number and call it a win; compare like with like, from the same client location, before and after each change.

## Step 4 — Handle the fallback and error cases

**Unsupported Brotli.** Some clients in the wild still do not advertise `br`. The negotiation above already handles this through `accepts`, which reads the `Accept-Encoding` header. Do not branch on the user agent string. User-agent sniffing is brittle and will misclassify clients, and a client that sends `Accept-Encoding: gzip, deflate` will be served gzip correctly by header negotiation alone.

If a client advertises neither `br` nor `gzip`, serving the identity body is the correct behavior. Returning `406 Not Acceptable` is legal but hostile: the client asked for a resource, not for a specific encoding, and most real clients will retry without the header. Prefer the uncompressed body.

**Burst loss during handoff.** TCP treats loss as congestion and backs off, which is exactly the wrong response to a handoff-induced burst. QUIC, the transport under HTTP/3, handles this better because stream multiplexing means one lost packet does not block unrelated streams, and its loss recovery is more aggressive about distinguishing congestion from random loss. Enable HTTP/3 where your CDN supports it and confirm it with:

```bash
curl -I --http3 https://your-distribution.example.com/
```

If the command fails, the client or the edge does not have HTTP/3 available; that is a capability check, not a performance measurement.

**Uplink cost.** If uplink is metered, reduce what the client sends. Practical options:

- Batch telemetry and send it on a timer instead of per event.
- Avoid client-side polling; prefer server push or long-lived connections that do not retry aggressively.
- Do not prefetch resources the user is unlikely to request.

Each of these reduces uplink bytes and, just as importantly, reduces the number of round trips that can be lost to a handoff.

## Step 5 — Measure the right things

A single "page load time" number hides the parts you need to fix. Instrument these separately:

- **DNS, TCP, TLS and total time** from `curl`'s `-w` output, run repeatedly and reported as a median.
- **Time to first byte** at the CDN edge, which isolates origin and cache behavior from transfer time.
- **Transfer time and decoded size**, to confirm the encoding actually applied.
- **Core Web Vitals** in the field, not just in the lab.

For server-side tracing, OpenTelemetry's Node SDK with auto-instrumentation gives you spans for HTTP handling and outbound calls:

```javascript
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { resourceFromAttributes } from '@opentelemetry/resources';

const sdk = new NodeSDK({
  resource: resourceFromAttributes({ 'service.name': '4g-baseline' }),
  traceExporter: new OTLPTraceExporter({ url: 'http://localhost:4318/v1/traces' }),
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();
```

Import it before your server module so instrumentation patches the modules first:

```javascript
import './tracer.js';
import express from 'express';
```

For lab checks, Lighthouse CI can enforce budgets on every pull request:

```yaml
name: Lighthouse CI
on: [pull_request]
jobs:
  lhci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npx lhci autorun
```

A Lighthouse run executes on a fast CI machine with a simulated throttle. It is a regression detector, not a prediction of your users' experience. Use it to catch a bundle that grew by 200 KB, and use field data to decide whether the change matters on real networks.

## Worked example: where the time actually goes

Assume an illustrative page of 1,000 KB uncompressed, a link with 12 Mbps downlink, and an RTT of 120 ms. These numbers are chosen for arithmetic, not measured from a specific network.

Transfer time for the uncompressed body, ignoring protocol overhead:

```
1,000 KB = 8,000 Kbit
8,000 Kbit / 12 Mbps = 0.667 s
```

Now suppose Brotli reduces the body to 250 KB, a 4:1 ratio:

```
250 KB = 2,000 Kbit
2,000 Kbit / 12 Mbps = 0.167 s
```

The transfer saving is about 0.5 s. But the connection setup still costs roughly two RTTs before any body byte moves: one for TCP and one for TLS. At 120 ms RTT that is about 240 ms, and it is unaffected by compression. This is why compression alone cannot fix a high-latency path, and why connection reuse, preconnect and a CDN edge close to the user matter as much as the byte count.

It also explains why the same optimization can look dramatic on one network and marginal on another. On a 100 Mbps link, saving 750 KB buys about 60 ms. On a 12 Mbps link, it buys about 500 ms. The ratio of the win is a function of the bandwidth you started with.

## Failure modes to plan for

**Compressing already-compressed content.** Applying Brotli to JPEG, PNG, WebP or video wastes CPU and can slightly increase size. Restrict compression to text types.

**Caching the wrong variant.** Omitting `Vary: Accept-Encoding` lets a shared cache store the gzip body and serve it to a client that asked for Brotli, or vice versa. This shows up as intermittent decode errors that are hard to reproduce.

**Buffering the whole response.** Streaming the compressed output avoids holding the full body in memory, which matters when several large responses are in flight.

**Trusting a single measurement.** One `curl` run tells you almost nothing. Report medians over at least ten runs and compare the same client, same location, same time of day.

**Optimizing downlink only.** If handoffs and uplink retries dominate, a smaller body will not fix the stall. Check whether the delay appears before the first byte or during transfer.

## Decision checklist

Before you ship a performance change, confirm:

- [ ] The response sets `Content-Encoding` and `Vary: Accept-Encoding` correctly.
- [ ] The fallback path serves a valid body, not an error, when no supported encoding is advertised.
- [ ] Compression is applied only to compressible content types.
- [ ] The CDN cache key includes the encoding.
- [ ] HTTP/3 is enabled and verified where the CDN supports it.
- [ ] You have before-and-after medians from the same client location.
- [ ] You can separate connection setup time from transfer time in your data.
- [ ] A CI budget fails the build when the page grows past a threshold.
- [ ] Uplink traffic is batched and not retried aggressively.

## FAQ

**Can this be done without a CDN?**
Yes. You lose edge caching and edge-terminated QUIC, so connection setup cost stays on the full path. On a single origin, aggressive compression plus preconnect can still help, but the latency component will dominate.

**Does Brotli always beat gzip?**
For text, usually, but the margin varies by content. For very small responses the framing overhead can make compression a net loss, which is why a size threshold exists in most middleware. Measure on your real assets.

**Should I branch on user agent for encoding?**
No. Negotiate on `Accept-Encoding`. User-agent strings are unreliable, and a header-based negotiation already covers every client that matters.

**What about images?**
Image bytes usually dominate page weight. Serve modern formats with a `<picture>` fallback, set explicit width and height to avoid layout shift, lazy-load below-the-fold images, and give the hero image a high fetch priority. Verify which formats your target browsers support rather than assuming.

**Is HTTP/3 worth enabling?**
It helps most where loss and handoffs are common, because one lost packet does not block unrelated streams. Confirm support at your edge and in your client population before relying on it.

## Do this in the next 30 minutes

Open your production HTML response and check two headers:

```bash
curl -sI -H "Accept-Encoding: br, gzip" https://your-site.example.com/ \
  | grep -i -E 'content-encoding|vary|cache-control'
```

If `Content-Encoding` is missing or `Vary` does not include `Accept-Encoding`, fix that first. Then run ten timed requests before and after the fix and compare the medians from the same machine. That single comparison will tell you more than any benchmark table.
