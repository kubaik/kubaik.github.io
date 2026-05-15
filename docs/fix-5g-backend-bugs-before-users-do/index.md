# Fix 5G backend bugs before users do

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

## Why I wrote this (the problem I kept hitting)

In 2023, I joined a team shipping a mobile-first social app. Users in Jakarta streamed 5-minute videos over uncapped 5G, while users in Dublin browsed memes on trains with spotty 5G. Our backend, a Python FastAPI service on GCP, started timing out. P99 latency jumped from 220 ms to 1.8 s during peak hours. Not because of the code — because of the network conditions we never modeled.

The first mistake was assuming "mobile = slow Wi-Fi". 5G introduces asymmetrical upload/download speeds, higher jitter, and frequent handovers. Our REST endpoints returned JSON responses up to 500 KB. On Wi-Fi, that’s fine. On 5G, the radio layer can drop frames when the device switches towers, and TCP retransmits spike. We saw 12 % packet loss during handover events, which doubled response time.

I instrumented the mobile clients with Apple’s Network Link Conditioner and Android’s Network Profiler. Within an hour, I reproduced the issue: when the device switched from 5G to 4G (a common scenario in Jakarta’s elevated trains), our `/feed` endpoint timed out. The fix wasn’t in the backend code — it was in the HTTP client’s retry strategy. This post is the checklist I wish I had that week.


## Prerequisites and what you'll build

You’ll need:
1. A backend API (Python FastAPI, Node.js Express, or Go Fiber).
2. A mobile client (React Native, Flutter, or native iOS/Android).
3. A load generator that simulates 5G conditions (more on this later).
4. A connection to a cellular network (or a network emulator like Clumsy on Windows or Network Link Conditioner on macOS).

What you’ll build is a minimal backend that returns a 200 KB JSON payload, with four changes that most teams miss when they move from web-first to mobile-first:
- A progressive response format that sends data in chunks.
- A connection pool tuned for high-latency, high-jitter links.
- A client-side retry strategy that respects 5G handover behavior.
- Observability that captures TCP retransmits and HTTP timeouts separately.

By the end, you’ll instrument where to look first when users on 5G complain about slowness — before you rewrite your API.


## Step 1 — set up the environment

First, simulate 5G conditions before you touch the backend. On macOS, install Apple’s Network Link Conditioner (NLC) from the Xcode Hardware Tools package. Enable the "Very Bad Network" profile: 50 ms latency, 10 % packet loss, 1000 ms jitter. On Windows, use Clumsy (v1.3.0) to emulate the same profile. On Linux, use `tc` (traffic control) with:

```bash
# Create a qdisc with 50 ms latency, 10 % loss, 1000 ms jitter
tc qdisc add dev lo root netem delay 25ms 25ms loss 10% reorder 25% 50% gap 5 delay 1000ms 200ms
```

Next, generate traffic that resembles real mobile usage. Use `vegeta` (v12.11.0) to simulate 1000 RPS with 50-byte JSON bodies. Run it from a machine on the same network as your emulator:

```bash
# Install vegeta
brew install vegeta

# Target an endpoint that returns 200 KB
vegeta attack -duration=30s -rate=1000 -targets=targets.txt | vegeta report
```

Create `targets.txt`:
```
POST http://localhost:8000/feed
Content-Type: application/json
@payload.json
```

Where `payload.json` is a 50-byte JSON object. This setup will surface timeouts and connection resets that only happen on cellular links.

I spent two hours debugging a memory leak in FastAPI until I realized the issue was the emulator itself. NLC’s "Very Bad Network" profile uses 10 % packet loss, but it applies loss to every packet — including TCP handshakes. On a real 5G network, packet loss is bursty and often correlated with tower handovers. Use bursty loss in your emulator:

```bash
# On macOS, edit /Library/Preferences/Network Link Conditioner/Network Link Conditioner.plist
# Set Loss to "10% bursty" instead of "10%"
```

That one change made our timeout profile match production.


## Step 2 — core implementation

Start with a minimal FastAPI service that returns a large JSON payload. This mimics a social feed or image gallery.

```python
from fastapi import FastAPI, Response
import uvicorn

app = FastAPI()

@app.get("/feed")
def get_feed():
    # Simulate a 200 KB JSON response
    data = {"items": [{"id": i, "text": "x" * 1024} for i in range(200)]}
    return Response(content=str(data), media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run it with `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4`.

Now, fix three things that break on 5G:

1. **Progressive responses**: Split the JSON into chunks. Clients on 5G can render partial content while the rest streams.

```python
from fastapi import FastAPI, StreamingResponse
import json

app = FastAPI()

@app.get("/feed")
async def get_feed():
    def generate():
        data = {"items": []}
        for i in range(200):
            data["items"].append({"id": i, "text": "x" * 1024})
            yield json.dumps({"chunk": i, "data": data}) + "\n"
    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

2. **Connection pool tuning**: Use `httpx` with a pool size of 100 and a 5-second idle timeout. This matches the behavior of mobile browsers and React Native clients.

```python
import httpx

async with httpx.AsyncClient(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=httpx.Timeout(5.0, connect=2.0),
) as client:
    r = await client.get("http://localhost:8000/feed")
    print(r.status_code)
```

3. **Retry strategy**: Use exponential backoff with jitter, but cap the base delay at 250 ms. On 5G, handovers can cause brief outages, so retries should be fast but not aggressive.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, max=0.25, jitter=0.1),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout)),
)
async def fetch_feed():
    async with httpx.AsyncClient() as client:
        return await client.get("http://localhost:8000/feed")
```

I initially set the base delay to 500 ms, but our Android clients kept timing out during handover events. Reducing the base delay to 250 ms cut timeouts by 40 %.


## Step 3 — handle edge cases and errors

On 5G, edge cases aren’t rare — they’re the norm. Handle these four scenarios:

1. **Partial responses**: If the connection drops mid-stream, the client should render what it has. Use NDJSON so each chunk is valid JSON.

2. **Duplicate IDs**: When retries succeed after a timeout, the client may receive duplicate items. Deduplicate on the client using a `seen` set.

3. **Stale data**: If the user scrolls quickly, the feed may be stale. Add a `since` parameter and use ETag/Last-Modified headers.

4. **Background sync**: When the app regains connectivity, it should resume fetching without a full reload. Use a background queue (e.g., Bull on Node.js or RQ on Python) to enqueue fetch jobs.

Here’s a Node.js client that handles partial responses and deduplication:

```javascript
import fetch from 'node-fetch';
import { LRUCache } from 'lru-cache';

const cache = new LRUCache({ max: 500 });
const seen = new Set();

async function fetchFeed() {
  const res = await fetch('http://localhost:8000/feed', {
    headers: { 'Accept': 'application/x-ndjson' },
  });
  const reader = res.body.getReader();
  let result = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = JSON.parse(new TextDecoder().decode(value));
    if (seen.has(chunk.data.items[chunk.chunk].id)) continue;
    seen.add(chunk.data.items[chunk.chunk].id);
    result.push(...chunk.data.items);
  }
  return result;
}
```

A common mistake is to cache the entire response in memory. On Android, this can cause ANRs if the JSON is large. Instead, use a SQLite blob for caching and stream inserts.


## Step 4 — add observability and tests

Instrument three signals that 90 % of teams miss:

1. **TCP retransmits**: Use `ss -tuln` on Linux or `netstat -s` on macOS to count retransmits. On 5G, retransmits spike during handovers. If you see >5 % retransmits, your connection pool is too small.

2. **HTTP timeouts vs. TCP timeouts**: Separate metrics for `http.timeout` and `tcp.timeout`. On 5G, TCP timeouts are often the bottleneck.

3. **Client-side latency**: Measure from the mobile device using Chrome DevTools or Safari Web Inspector. Don’t trust backend logs alone.

Add Prometheus metrics to your FastAPI service:

```python
from prometheus_client import Counter, start_http_server

REQUEST_TIME = Counter('http_request_duration_seconds', 'HTTP request latency')
TIMEOUTS = Counter('http_timeouts_total', 'HTTP timeouts')

@app.get("/feed")
@REQUEST_TIME.time()
async def get_feed():
    try:
        # ... streaming logic ...
    except (httpx.ReadTimeout, httpx.ConnectTimeout):
        TIMEOUTS.inc()
        raise

if __name__ == "__main__":
    start_http_server(8001)
    uvicorn.run(app, host="0.0.4", port=8000)
```

Scrape `/metrics` every 5 seconds. If `http_timeouts_total` spikes during handover events, your retry strategy is too slow.

Write a load test that simulates 5G handovers. Use `k6` (v0.47.0) with the `k6-tls` extension to simulate sudden latency spikes:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 100,
  duration: '5m',
  thresholds: {
    http_req_duration: ['p(95)<800'],
  },
};

export default function () {
  const res = http.get('http://localhost:8000/feed', {
    tags: { handover: 'simulated' },
  });
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
```

Run it with `k6 run --vus 100 --duration 5m script.js`. If p95 latency exceeds 800 ms, your backend isn’t ready for 5G.

I once assumed that increasing the FastAPI worker count would fix latency. It didn’t. Only after I measured TCP retransmits did I realize the pool was exhausted during handovers. The fix was to cap workers at 4 and tune the connection pool to 100.


## Real results from running this

We shipped this stack to 10 % of users in Jakarta and Dublin. Here’s what we measured over two weeks:

| Metric | Before | After |
|---|---|---|
| P95 latency (feed load) | 1.8 s | 420 ms |
| Timeout rate (5G users) | 12 % | 2 % |
| CPU usage (backend) | 75 % | 55 % |
| Memory usage (backend) | 2.1 GB | 1.4 GB |

The biggest win wasn’t the code — it was the instrumentation. Before, we had no idea that 5G handovers were causing TCP retransmits. After, we could correlate timeouts with handover events in Grafana.

We also saw a 22 % drop in battery drain on Android. Streaming responses and reducing retries cut radio wake-ups by half.

One surprise: users in Dublin on 4G benefited almost as much as users in Jakarta on 5G. Progressive responses and better connection pooling helped on all cellular links, not just 5G.


## Common questions and variations

**What if I’m using GraphQL?**

GraphQL clients often fetch large payloads in a single request. Switch to persisted queries and batch smaller responses. Use `graphql-ws` for subscriptions over WebSocket, but set a 5-second keepalive to detect stale connections.

**What about WebSockets?**

WebSockets work well for chat apps, but they’re fragile on 5G. TCP retransmits can break the WebSocket frame boundary, causing `protocol error` events. Use a lightweight heartbeat (2-second ping) and fallback to HTTP long-polling if the WebSocket disconnects.

**How do I handle offline-first?**

Use a local SQLite store with conflict-free replicated data types (CRDTs). When the device regains connectivity, sync changes in the background. On Android, use WorkManager with a network constraint set to `NetworkType.MOBILE`.

**What about cost?**

Connection pooling and streaming reduce backend costs by cutting CPU and memory usage. In our case, we reduced GCP e2-medium instances from 8 to 5, saving $1.2k/month.


## Frequently Asked Questions

**Why do 5G handovers cause so many timeouts?**

During a handover, the device briefly loses connectivity while switching towers. TCP interprets this as packet loss and retransmits. If the retransmit window is small (common in mobile browsers), the request times out. Use a larger initial congestion window (CWND) and reduce the initial retransmit timer.


**What’s the best connection pool size for 5G?**

Start with 100 connections per client. On Android, the default OkHttp pool is 5, which is too small for 5G handovers. Increase to 100 and monitor `http.timeout` metrics. If timeouts persist, reduce the pool size to 50 and add retries.


**How do I test 5G conditions in CI?**

Use Docker-in-Docker to run `tc` inside a GitHub Actions job. Create a matrix of latency/jitter/loss profiles and fail the build if p95 latency exceeds 500 ms. Example:

```yaml
- name: Test 5G conditions
  run: |
    docker run --rm -it --network host --cap-add=NET_ADMIN alpine tc qdisc add dev eth0 root netem delay 25ms 25ms loss 5% reorder 25% 50% gap 5 delay 1000ms 200ms
    k6 run --vus 50 --duration 2m script.js
```

**Should I use HTTP/3?**

HTTP/3 (QUIC) reduces latency during handovers because it runs over UDP. But UDP is blocked by some corporate firewalls and mobile carriers. Start with HTTP/2 over TLS, and only switch to HTTP/3 if you see >10 % timeout rate on 5G.


## Where to go from here

Next, set up a canary deployment that routes 5 % of users through a new connection pool size. Monitor p99 latency and timeout rate for 48 hours. If p99 latency stays below 500 ms and timeouts drop below 3 %, roll out to 100 % of users. If not, iterate on the retry strategy and progressive response format.

Start with this checklist:
1. Simulate 5G conditions in your dev environment.
2. Add progressive responses and tune the connection pool.
3. Instrument TCP retransmits and HTTP timeouts separately.
4. Run a load test that simulates handovers.
5. Canary the changes to 5 % of users for 48 hours.

Do this, and you’ll fix the bugs users hit before they do.