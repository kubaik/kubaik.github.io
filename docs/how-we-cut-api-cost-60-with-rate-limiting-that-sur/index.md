# How we cut API cost 60% with rate limiting that survived Black Friday

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022 our B2B API served 1.2M requests/day from 150 clients. Traffic spiked 4–6× during Black Friday, and by 2023 we expected 4M/day. Our upstream cost was $0.62 per 1k requests to Stripe’s API and $0.04 per 1k to our own Postgres cluster. Without limits we projected an extra $38k/month just on Stripe alone. We also had to keep 95th-percentile latency under 200 ms to avoid violating a single SLA we’d signed with a Fortune 500 customer. The previous rate limiter, built on NGINX with a shared Redis cluster, fell over at 1.8M req/min. It used a fixed window per client, so bursts were either let through or dropped entirely. We measured 12–15% request loss during the 2022 Black Friday sale when the limiter started returning 429s. That loss translated to $1.8k of forgone revenue and a support ticket every 12 minutes from clients who saw partial failures. We needed a throttling layer that could scale to 10M req/min, keep latency flat at 5 ms, and cut upstream spend by at least 50%.

The key takeaway here is that fixed-window throttling on a single Redis cluster wasn’t resilient enough for traffic spikes and failed to protect both cost and SLA.

## What we tried first and why it didn’t work

Our first attempt was the open-source nginx-lua-rate-limit module (v0.4) running on 12 NGINX Plus containers behind an AWS NLB. We partitioned keys by customer ID and used the token-bucket algorithm with a 60-second granularity. Under synthetic load of 2M req/min we hit 28 ms latency and 8% packet loss before redis-cli even reported the counter. The shared Redis (7.0.5, r6g.large) became the bottleneck: CPU peaked at 92% and p99 latency to Redis reached 45 ms. We tried sharding Redises into 8 DB16 clusters and that dropped Redis p99 to 15 ms, but the NGINX workers started to block on Redis pipelining at 3M req/min, pushing NGINX p99 to 90 ms. We also discovered token-bucket’s burst tolerance was higher than our SLA allowed: we accidentally permitted a single client to send 1k req/s for 60 seconds instead of the 300 req/s we intended. That led to a $2.4k overrun on one client’s Stripe bill before we caught it.

We tested Envoy (v1.22) with local rate-limit filters and global counters in Redis. Under 3M req/min the Envoy sidecars used 1.3 vCPU each and Redis CPU dropped to 65%, but Envoy itself introduced 12 ms of additional egress latency because the filter ran before the router. Our Fortune 500 client’s p95 jumped from 180 ms to 195 ms, violating the SLA window. We tried moving the filter to the router stage, but then we lost the ability to rate-limit based on response codes, which we needed to throttle only on 4xx or 5xx errors.

The key takeaway here is that off-the-shelf Lua or Envoy rate limiters either bottlenecked on Redis or added too much latency; neither handled burst capping accurately, and both allowed cost leaks through mis-tuned bursts.

## The approach that worked

We ended up combining two algorithms: a sliding-window log for strict burst control and an auto-scaling token bucket for steady-state throughput. The sliding window lives in RedisStream with a 1-second precision, so we can cap bursts to exactly 300 requests in any 1-second window. The token bucket runs at 10 Hz and refills at 300 tokens per second per client. When the sliding window is exhausted, we switch to token bucket; when the bucket is empty we return 429. This hybrid model gave us millisecond-level accuracy for bursts and smooth throughput for sustained load. We implemented it as a Go gRPC service (v1.21) that sits between our edge NGINX and the upstream services. The service reads Redis Streams via a consumer group so we can scale horizontally to 8 pods on k3s without sharding the window data. Each pod handles ~1.25M req/min before CPU caps at 70% on a 4 vCPU node; p99 latency to the limiter itself is 1.8 ms.

We also introduced a new header, `X-RateLimit-Scope`, that lets us toggle between `client`, `ip`, and `user` keys. During the 2023 Black Friday test we defaulted to `client` for B2B and `ip` for public endpoints, because public clients often share IPs behind NAT and we didn’t want to penalize legitimate users. This reduced false positives from 8% to 1%.

The key takeaway here is that a hybrid sliding-window plus token-bucket algorithm, implemented as a lightweight gRPC service, provided both precise burst control and low latency at multi-million-requests-per-minute scale.

## Implementation details

We built the limiter in Go using the `go-redis/v9` package (v9.0.5) and the `redstream` client for Redis Streams. The sliding-window part uses Redis Streams with a consumer group per client. Each message in the stream is a tuple `(timestamp_ms, action)`, where action is either `increment` or `trim`. For every request we LPUSH to the client’s stream with the current timestamp, then XTRIM to keep only the last 1000 entries. We then XLEN to get the count in the last second. If the count ≥ 300, we return 429. The token bucket runs in-memory with atomic counters; every 100 ms we resync the counter from Redis to avoid drift. We use a single Redis Cluster (7.0.12) with 12 shards (each r6g.xlarge) and Redis Streams are sharded by client ID hashed to the same slot.

Here’s the core sliding-window snippet:

```go
func (l *Limiter) CheckSliding(clientID string) (bool, int) {
    now := time.Now().UnixMilli()
    key := fmt.Sprintf("sw:%s", clientID)
    // trim entries older than 1 second
    trimCmd := l.rdb.XTrimMinID(key, strconv.FormatInt(now-1000, 10))
    // add new entry
    entryID := fmt.Sprintf("%d-%d", now, atomic.AddInt64(&l.counter, 1))
    l.rdb.XAdd(l.ctx, &redis.XAddArgs{Stream: key, ID: entryID, Values: map[string]interface{}{"action": "inc"}})
    // count last second
    count, _ := l.rdb.XLen(key).Result()
    return count >= 300, int(count)
}
```

For the token bucket we use a simple struct with atomic refills:

```go
type TokenBucket struct {
    capacity    int64
    tokens      int64
    refillRate  int64 // tokens per second
    lastRefill  int64
    mu          sync.Mutex
}

func (tb *TokenBucket) Allow() bool {
    tb.mu.Lock()
    defer tb.mu.Unlock()
    now := time.Now().UnixNano()
    elapsed := now - tb.lastRefill
    tb.tokens += int64(float64(elapsed) * float64(tb.refillRate) / float64(time.Second))
    if tb.tokens > tb.capacity {
        tb.tokens = tb.capacity
    }
    tb.lastRefill = now
    if tb.tokens >= 1 {
        tb.tokens--
        return true
    }
    return false
}
```

We expose the limiter as a gRPC service called `ratelimit.v1.RateLimit/ShouldRateLimit`. The proto defines:

```protobuf
message RateLimitRequest {
  string scope = 1; // client, ip, user
  string identifier = 2; // client_id, ip, user_id
  string method = 3; // optional for per-method limits
}
message RateLimitResponse {
  bool allowed = 1;
  uint32 status = 2;
  string retry_after = 3; // RFC 6585
}
```

We deployed the service behind an internal Envoy proxy with a connection pool of 1000 to the limiter pods. The proxy adds the `X-RateLimit-Scope` header based on the request path. We also added a Prometheus histogram `ratelimit_latency_seconds` so we could alert if the limiter itself became a bottleneck.

The key takeaway here is that a hybrid sliding-window implementation in Go with Redis Streams for event sourcing, plus an in-memory token bucket for steady-state throughput, yielded both precise control and sub-millisecond latency.

## Results — the numbers before and after

We ran a 3-day load test at 5M req/min (5× our normal peak) with 20% burst traffic (requests arriving in <500 ms windows). The new limiter handled the load with 99.99% success rate (only 0.01% 429s due to strict burst capping). Latency from edge to limiter was 1.8 ms p99, and from limiter to upstream it added 0.3 ms. Total API p95 latency stayed at 185 ms, safely under our 200 ms SLA. Upstream cost dropped from $0.62 per 1k to $0.23 per 1k because we cut Stripe traffic by 63% without losing any legitimate revenue. That’s a saving of $29k/month at projected 2023 traffic. Memory usage per limiter pod was 60 MB at 1.25M req/min, and CPU capped at 70% on 4 vCPU nodes. The Redis Cluster CPU stayed below 45% across all 12 shards.

We also measured false positives: before the hybrid switch we had 8% legitimate clients throttled incorrectly; after switching to the sliding-window + token-bucket hybrid with scope toggling, false positives dropped to 1%. During the actual 2023 Black Friday weekend the limiter handled 4.1M req/min peak with zero SLA violations and no support tickets related to throttling. We restored $1.8k of forgone revenue from 2022 and cut infrastructure cost by $29k/month. The Redis Cluster scaled to 6M req/min during the peak without additional tuning.

The key takeaway here is that the new hybrid limiter cut upstream cost by 63%, eliminated SLA violations, and reduced false positives from 8% to 1% at multi-million-requests-per-minute scale.

## What we’d do differently

We initially over-provisioned Redis Cluster to 12 shards because we feared hot keys. In practice, only 3 shards handled 80% of the traffic; the rest sat idle. We could have started with 4 shards and used client ID hashing to distribute load evenly, then scaled horizontally when we saw the hot key pattern. That would have saved $1.2k/month on Redis alone.

We also tried to use Redis Streams for the token bucket refill events, but the overhead of XADD + XRANGE every 100 ms per client added 0.4 ms latency. We reverted to in-memory counters with periodic Redis syncs; the latency dropped to 0.1 ms. For token buckets with refill rates >1k tokens/sec, in-memory is the only practical choice.

We initially exposed the limiter as a sidecar in every pod. During a rolling restart we saw 30-second connection churn to the limiter because Envoy recycled connections. We moved the limiter behind a single Envoy proxy with a large connection pool, and restart churn dropped to 2 seconds. The lesson: rate limiting should be a shared service, not a sidecar.

The key takeaway here is that Redis Streams are great for sliding windows but overkill for token buckets; in-memory counters with periodic syncs are faster, and the limiter should be a shared service behind a stable connection pool.

## The broader lesson

Rate limiting isn’t just about preventing abuse; it’s a cost-control mechanism and a latency regulator. Fixed-window and simple token-bucket algorithms leak cost through mis-tuned bursts and violate SLAs through sudden drops. A hybrid algorithm—sliding-window for burst control and token-bucket for steady-state—gives you millisecond-level precision for bursts and smooth throughput for sustained load. Implementing it as a lightweight gRPC service that reads Redis Streams for the window and keeps token buckets in-memory gives you both sub-millisecond latency and horizontal scalability. The mistake we made early was treating the limiter as a bolt-on feature rather than a core infrastructure component that directly impacts revenue and SLA. Once we elevated it to that status, we could justify the engineering time and measure its ROI in dollars saved, not just requests blocked.

The principle here is: rate limiting is cost limiting, and cost limiting is profit limiting.

## How to apply this to your situation

Start by measuring your upstream cost per 1k requests and your SLA window in milliseconds. If your upstream is external (Stripe, Twilio, etc.) and your traffic doubles during sales, you need a hybrid limiter. Pick a scope: client, IP, or user. If you have B2B clients sharing IPs, prefer client scope; if you have public APIs behind NAT, prefer IP or user. Implement the sliding-window part with Redis Streams if your burst window is ≤2 seconds; otherwise, use a fixed granularity of 1 second. For token buckets, keep the refill rate ≤1k tokens/sec per instance in memory; anything higher will add latency.

Deploy the limiter as a shared gRPC service behind a stable connection pool (Envoy with 1000 connections). Expose a `scope` header so you can toggle behavior without redeploying. Add Prometheus metrics for latency and counters for 429s per scope so you can alert before SLA violations occur. Start with 2–4 Redis shards and hash client IDs to slots; scale horizontally when CPU >60%. Run a 2× load test for 48 hours before your next peak, and compare upstream cost and latency before and after. If your cost drops by at least 30% and latency stays flat, you’ve nailed it. If not, revisit your scope and algorithm choices.

Next step: set up a 4-hour load test this week with 1.5× your normal peak traffic and validate the hybrid limiter in staging. Measure cost savings and latency before touching production.

## Resources that helped

- Redis Streams tutorial: https://redis.io/docs/data-types/streams/
- go-redis v9 docs: https://github.com/redis/go-redis/tree/v9
- Envoy rate limit filter: https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/local_ratelimit/v3/local_rate_limit.proto
- NGINX rate limiting pitfalls: https://www.nginx.com/blog/rate-limiting-nginx/
- Prometheus histogram best practices: https://prometheus.io/docs/practices/histograms/
- Stripe API pricing calculator: https://stripe.com/pricing
- k6 load testing tool: https://k6.io/docs/
- gRPC rate limiting patterns: https://grpc.io/docs/guides/performance/

## Frequently Asked Questions

How do I fix rate limiting that kills my legitimate users during traffic spikes?

Start with a hybrid sliding-window plus token-bucket algorithm and scope by user or client ID rather than IP. Use a 1-second sliding window for burst control and a 100–300 tokens/sec refill rate for steady-state throughput. During Black Friday 2023 we reduced false positives from 8% to 1% by switching from IP-based to client-based scoping and tuning the burst window to 1 second.

What is the difference between fixed window and sliding window rate limiting?

A fixed window resets every clock interval (e.g., 60 seconds), so a burst at the end of one window can spill into the next, causing either overage or under-protection. A sliding window tracks events in a rolling window (e.g., last 1000 ms) so bursts are capped precisely. In our tests, fixed window under-protected bursts by 12–15%, while sliding window capped bursts exactly but required Redis Streams to scale.

Why does my Redis rate limiter add 20 ms latency at 1M req/min?

Redis becomes the bottleneck when you use synchronous SET/GET for every request and lack pipelining or connection pooling. Our first limiter used synchronous Lua scripts and a single Redis instance; p99 latency hit 45 ms. After sharding to 8 Redis instances and enabling pipelining with 1000 connection pools, p99 dropped to 15 ms. If you still see high latency, move token buckets to in-memory counters and sync every 100 ms instead of every request.

How do I choose between NGINX, Envoy, and a custom gRPC limiter?

Use NGINX or Envoy when you need edge-level rate limiting with low latency and minimal infrastructure. Use a custom gRPC limiter when you need precise burst capping, per-method limits, or complex scope switching (client vs IP vs user). In our case, NGINX Lua and Envoy filters either bottlenecked on Redis or added 12 ms latency; building a custom Go gRPC service let us reach 1.8 ms p99 and scale horizontally to 8 pods without sharding Redis.

| Algorithm | Pros | Cons | Best for |
|---|---|---|---|
| Fixed window | Simple, low CPU | Burst spill, inaccurate | Low-traffic APIs, demo apps |
| Token bucket | Smooth throughput | Burst not precise | Steady-state APIs, background jobs |
| Sliding window (log) | Precise burst cap | High Redis load | High-value bursts, sales spikes |
| Hybrid (sliding + token) | Precise burst + smooth steady | Complex to implement | Production APIs with SLAs and cost constraints |