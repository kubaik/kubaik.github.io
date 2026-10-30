# Serve 4G-as-baseline apps in 2026

Most performance tutorials assume a fast, stable network. That assumption increasingly fails. In many regions the realistic baseline for a mobile user is 4G-class connectivity: a few hundred milliseconds of round-trip time, intermittent loss, and a metered data plan. Satellite links add a second pattern: excellent median latency and throughput off-peak, with congestion, jitter and rain fade during peak hours. An app that feels instant on fibre can feel broken under those conditions, and the difference is rarely the server's raw speed. It is payload size, cache behavior, retry policy and how much work the client does before it can render.

This article covers how to design and adapt a React front end, a Go API and a PostgreSQL backend for that baseline. The techniques are ordinary: budget the payload, cache aggressively at the right layer, compress on the fly, bound concurrency, and measure with a network profile that matches your users. None of them require new infrastructure.

## Define the baseline before you optimise

Write down the network conditions you are designing for. Without that, every performance decision is guesswork. A useful starting template:

| Parameter | 4G-class baseline | Congested satellite peak |
| --- | --- | --- |
| Median RTT | 150–300 ms | 400–600 ms |
| Jitter | 30–80 ms | over 100 ms |
| Packet loss | 1–5 % | 5–15 % |
| Downlink | 2–10 Mbps | highly variable |
| Uplink | 0.5–2 Mbps | 128 kbps–1 Mbps |
| Data | metered | metered |

These are illustrative ranges, not measurements. Replace them with numbers from your own users: server-side timing by region, client-side `PerformanceNavigationTiming`, or a synthetic monitor placed near your audience. The point is to have a written budget you can test against.

From that budget you can derive a payload target with arithmetic rather than intuition. Suppose a user has 2 Mbps effective throughput, which is 2,000,000 bits per second, or 250,000 bytes per second. A 1,200,000-byte bundle takes 1,200,000 / 250,000 = 4.8 seconds just to transfer, before parsing or rendering. Cut the bundle to 300,000 bytes and the same transfer takes 1.2 seconds. That is the whole argument for aggressive payload reduction: on a constrained link, bytes are time.

Set explicit budgets and enforce them in CI:

- Initial JS transferred, compressed: pick a number and fail the build above it.
- Total bytes for the first meaningful render: same.
- API response size for the hot endpoint: same.
- p99 server response time and p99 client-observed latency: alert thresholds.

## Step 1 — set up the environment

A local stack that mirrors production shape is enough to develop against:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: app
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    command: >
      -c shared_preload_libraries=pg_stat_statements,auto_explain
      -c auto_explain.log_min_duration=100
      -c auto_explain.log_analyze=true
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d app"]
      interval: 2s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --save 30 1 --loglevel warning

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
      - PORT=8080
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - VITE_API_ENDPOINT=http://localhost:8080
    depends_on:
      - backend

volumes:
  pg_data:
```

Pin exact image tags in real deployments. Floating tags such as `postgres:16-alpine` will silently change under you.

A multi-stage build keeps the Go image small, which matters when you pull it over a slow link in CI or on a constrained host:

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server main.go

FROM alpine:3.19
WORKDIR /root/
COPY --from=builder /server /usr/local/bin/server
EXPOSE 8080
USER 1000
CMD ["server"]
```

For the front end, split vendor code into stable chunks so that a deploy does not invalidate the whole cache, and keep the compression target honest:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            const lib = id.split('node_modules/')[1].split('/')[0]
            return lib === 'react' || lib === 'react-dom' ? 'vendor' : `lib-${lib}`
          }
        }
      },
      minify: 'terser',
      terserOptions: { compress: { passes: 2 }, mangle: { toplevel: true } }
    },
    target: 'es2022',
    cssCodeSplit: true,
    reportCompressedSize: true
  },
  server: { port: 3000, host: true, hmr: { port: 3000 } }
})
```

A recurring gotcha: build-time compression only covers static assets. Dynamic API responses are compressed at request time, if at all. Check what your server actually sends with `curl -H 'Accept-Encoding: br,gzip' -s -o /dev/null -w '%{size_download} %{content_type}\n' https://your-api/v1/data` and compare it with the uncompressed size. If the ratio is close to 1, nothing is compressing.

## Step 2 — cache with stale-while-revalidate

On a high-latency link, the fastest request is the one you do not make. A short-TTL cache in front of a hot read endpoint absorbs bursts and smooths database load. The classic failure mode is a thundering herd: many clients miss at the same instant and all hit the database. Two mitigations are standard.

First, add jitter to the TTL so entries do not expire in lockstep. Second, use probabilistic early refresh: before an entry expires, each request has a small chance of triggering a background refresh, so the entry is usually refreshed before it goes cold. The refresh probability should rise as the entry ages.

```go
package main

import (
	"context"
	"math/rand"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	baseTTL      = 500 * time.Millisecond
	refreshAfter = 300 * time.Millisecond
	refreshProb  = 0.03
)

// shouldRefresh returns true with a probability that grows as the entry ages
// past refreshAfter. Callers trigger a background refresh when it returns true.
func shouldRefresh(age time.Duration) bool {
	if age < refreshAfter {
		return false
	}
	if age >= baseTTL {
		return true
	}
	// Linear ramp from 0 at refreshAfter to refreshProb at baseTTL.
	frac := float64(age-refreshAfter) / float64(baseTTL-refreshAfter)
	return rand.Float64() < refreshProb*frac
}

type Cache struct {
	rdb *redis.Client
}

func (c *Cache) Get(ctx context.Context, key string) ([]byte, error) {
	return c.rdb.Get(ctx, key).Bytes()
}

func (c *Cache) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	// Jitter the TTL by +/-10% so entries do not expire together.
	jitter := time.Duration(rand.Int63n(int64(ttl / 5)))
	return c.rdb.Set(ctx, key, val, ttl-ttl/10+jitter).Err()
}
```

The documented behavior of `Set` with a duration is that the key is removed after that duration; a zero duration means no expiry. Jittering the TTL is what prevents synchronized expiry, and the ramp is what keeps a hot key warm without every request paying for a refresh.

Two things to be careful about:

- **Stampedes on cold start.** When a key does not exist at all, the probabilistic path does not help. Either accept the first-request cost or add a single-flight lock so only one request populates the key while others wait briefly or serve a stale copy.
- **Cache key shape.** Include the parts of the request that change the response (path, normalized query parameters, tenant, locale). A key that ignores a variant will serve the wrong body, which is worse than a miss.

If you want to know whether this is working, instrument it. Export a cache hit ratio gauge and a counter of background refreshes, and alert when the hit ratio drops or the refresh rate spikes. A hit ratio in the low 70s percent usually means the TTL is too short relative to request rate, or the key space is too fragmented.

## Step 3 — compress and bound concurrency

Compression is the cheapest payload win available. Brotli generally beats gzip on text, but at higher levels it costs meaningful CPU. A practical policy:

1. Compress only responses above a threshold (around 1 kB); below that, the framing overhead can exceed the savings.
2. Use a moderate Brotli level by default and reserve the highest level for precompressed static assets.
3. Negotiate from `Accept-Encoding`. If a client sends only `gzip`, serve gzip rather than nothing.
4. Cap concurrent compression work. Compression that saturates the CPU raises latency for everyone, which is the opposite of the goal.

```go
// compressIfWorthwhile compresses body with Brotli when it is large enough
// and the client accepts it. Returns the body unchanged otherwise.
func compressIfWorthwhile(acceptEncoding string, body []byte, minSize int) ([]byte, string) {
	if len(body) < minSize {
		return body, ""
	}
	if !strings.Contains(acceptEncoding, "br") {
		return body, ""
	}
	var buf bytes.Buffer
	w := brotli.NewWriterLevel(&buf, 6)
	if _, err := w.Write(body); err != nil {
		return body, ""
	}
	if err := w.Close(); err != nil {
		return body, ""
	}
	// Only use the compressed form if it actually helped.
	if buf.Len() >= len(body) {
		return body, ""
	}
	return buf.Bytes(), "br"
}
```

Note the final check: if compression did not shrink the body, send the original. This happens more often than people expect with already-compressed formats such as JPEG or with very small JSON.

Concurrency limits are the other half. A slow client on a lossy link can hold a connection and a worker for a long time. Bound inflight requests per client and set read/write timeouts so a stalled connection is reclaimed rather than accumulating:

```go
srv := &http.Server{
	ReadTimeout:       5 * time.Second,
	ReadHeaderTimeout: 2 * time.Second,
	WriteTimeout:      15 * time.Second,
	IdleTimeout:       60 * time.Second,
}
```

Choose timeouts from your latency budget, not from habit. If p99 target latency is 500 ms, a 15-second write timeout is a safety net against pathological clients, not a normal operating parameter.

## Step 4 — handle loss, retries and connection pooling

Packet loss changes the shape of the problem. The failure mode to avoid is a retry storm: a client times out, retries immediately, and multiplies load on a server that is already struggling. Three rules help.

- **Retry only idempotent requests**, and only on connection errors or timeouts, not on every 5xx.
- **Use exponential backoff with jitter.** Without jitter, retries from many clients align.
- **Give the client a deadline** and surface a fast, useful failure rather than a spinner that never resolves.

For PostgreSQL, connection pooling matters more when latency is high because each connection setup is expensive. Transaction-mode pooling keeps the server-side connection count low, but it has a well-known constraint: prepared statements are not preserved across transactions. If your driver or ORM relies on them, use session mode or disable server-side prepared statements for pooled connections.

```ini
[databases]
app = host=postgres port=5432 dbname=app user=app password=${DB_PASSWORD}

[pgbouncer]
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
server_idle_timeout = 30
```

The trade-off is explicit: transaction mode gives you many client connections over few server connections, at the cost of session state. Session mode preserves state at the cost of more server connections. Pick based on whether your queries depend on session state.

## Step 5 — measure with a realistic profile

Numbers from a fast local network tell you almost nothing about a 4G user. Instrument both sides:

**Server side.** Export request duration as a histogram with path labels, response size as a histogram with a compression label, and a cache hit ratio gauge. Use `pg_stat_statements` and `auto_explain` (with `log_min_duration` set to something meaningful for your budget) to find slow queries.

**Client side.** Use the browser's `PerformanceNavigationTiming` and `PerformanceResourceTiming` entries to get real transfer times, and send them to your backend in batches. Aggregate by connection type and region.

**Synthetic testing.** Run a load test under a throttled profile. The important part is the profile, not the tool: set latency, jitter, loss and bandwidth to your baseline. Assert on the percentiles you care about.

```javascript
import http from 'k6/http'
import { check } from 'k6'

export const options = {
  scenarios: {
    fourG: {
      executor: 'per-vu-iterations',
      vus: 200,
      iterations: 200,
      maxDuration: '10m',
      thresholds: {
        http_req_duration: ['p(99)<500'],
        http_req_failed: ['rate<0.01']
      },
      tags: { profile: '4G' }
    }
  }
}

export default function () {
  const res = http.get('https://api.example.com/v1/data')
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response size < 4 kB': (r) => r.body.length < 4096,
  })
}
```

The thresholds are the point. A load test without thresholds is a graph; a load test with thresholds is a regression gate. Note that the default HTTP transport in most load tools does not negotiate HTTP/2 unless you enable it, so verify what protocol you are actually testing with a packet capture or the tool's own reporting.

To measure the cache specifically, run `MONITOR` in the Redis CLI for a couple of minutes while you drive traffic at the endpoint. Count the `GET` and `SET` commands. If you see far more `SET` commands than expected, your TTL or refresh probability is too aggressive. If you see a burst of `GET` misses at the same instant, your jitter is not working.

## Failure modes to design for

- **Synchronized expiry.** All keys written at deploy time expire together. Mitigation: jittered TTLs.
- **Cold cache after deploy.** A new version changes the key prefix and every request misses. Mitigation: keep the key prefix stable across deploys, or warm the cache before shifting traffic.
- **Retry amplification.** A brief outage triggers client retries that outlast the outage. Mitigation: backoff with jitter and a retry budget.
- **Compression CPU saturation.** High compression levels under load push CPU to 100 % and latency up. Mitigation: moderate levels, a concurrency cap, and a gzip fallback.
- **Slow-client starvation.** A few very slow clients consume all workers. Mitigation: per-client inflight limits and short read timeouts.
- **Stale reads after writes.** A short-TTL cache serves data that is seconds out of date. Mitigation: invalidate explicitly on write, or bypass the cache for the writer's own reads.
- **Metrics that lie.** Averages hide the tail. Mitigation: always alert on p99, never on mean.

## Common questions

**Why not just put a CDN in front of everything?** Edge caches are excellent for static assets and cacheable GETs. Dynamic, per-user responses usually miss at the edge, so the cache has to sit closer to the data, in the same availability zone as the API and database. Use the CDN for what it is good at and a short-TTL application cache for the rest.

**Do service workers help?** They can, for offline and repeat visits, but registration and activation cost time on first load. Register them after the page is interactive, and cache a small, explicit set of assets rather than everything.

**Should I move to HTTP/3?** It reduces head-of-line blocking on lossy networks, which is exactly the 4G failure mode. The trade-offs are CPU cost, operational complexity and library maturity. Measure it against your baseline before committing; the benefit depends on your loss rate.

**Can this be done outside Go?** Yes. The patterns are language-independent: compress on the fly, cache with jittered TTLs, bound concurrency, retry with backoff. Runtime differences show up in memory per request and CPU cost of compression, not in whether the approach works.

**How do I handle variable satellite bandwidth?** Detect slow first requests on the client and downgrade subsequent requests: request a smaller payload variant, prefer a compressed encoding, and defer non-critical data. This is a client-side policy, not a server feature.

## Action for the next 30 minutes

Pick your slowest endpoint and measure it under a realistic profile. Run `curl -w '%{time_total}\n' -o /dev/null -s https://your-api/v1/your-endpoint` ten times and note the spread, not just the average. Then run `MONITOR` in the Redis CLI for two minutes while you drive traffic at that endpoint, and count the `GET` misses and `SET` commands. If misses cluster, add jitter to your TTL and a small probabilistic early refresh, then repeat the measurement. You will have a before-and-after you can trust, and a baseline you can defend.
