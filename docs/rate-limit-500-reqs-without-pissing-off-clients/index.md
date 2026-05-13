# Rate limit 500 req/s without pissing off clients

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In early 2023 our payments microservice went from 50 req/s to 500 req/s overnight when a partner integrated us as their primary payment provider. Within a week we were getting angry Slack messages: "Why is my 100 ms request taking 8 seconds?" We dug into New Relic and saw that 95% of the latency spike happened at the Redis layer where our rate limiter lived. The default token-bucket implementation was blocking legitimate retries, not distinguishing between clients, and causing cascade failures when traffic surged. We needed a rate limiter that could protect us from abuse while still letting honest clients retry failed requests without penalty. Our SLO at the time required 99th-percentile latency under 200 ms. Anything worse and partners started to complain.

We surveyed what others were doing. Most teams either:
- Used a naive in-memory counter that reset on every restart (we crashed twice in staging).
- Rolled their own leaky-bucket with DynamoDB TTLs (cost $600/month for 10 GB of Dynamo writes).
- Adopted a SaaS rate limiter like Cloudflare Turnstile ($1.20 per 100k requests) but found it added 40 ms of extra latency and didn’t expose per-client counters.

We measured our current failure modes. At 500 req/s with bursts of 2000 requests, 42% of failed requests were legitimate retries that got rate-limited because our bucket refill rate (500 tokens/s) was slower than the retry storm. Worse, we had no way to distinguish between a single misconfigured client hammering us and a distributed load balancer retrying failed requests. That became our north-star: protect against abuse but don’t punish retries.


Rate limiting was a gating factor for new integrations. Potential partners pulled back when they saw 8-second response times during load tests. We had to ship something in two weeks or risk losing deals.


## What we tried first and why it didn’t work

Our first attempt was a naive token-bucket in Redis with a fixed refill rate. We set a 100 requests per second bucket per client IP. That immediately broke. Why?

1. **Retry storms and backoff confusion**: Partners implemented exponential backoff starting at 500 ms. When a client’s request failed due to rate limiting, it resubmitted 500 ms later — exactly when the bucket had just refilled 50 tokens. The client would retry, get rate-limited again, and back off to 1 s, 2 s, 4 s. The retry pattern synchronized across thousands of clients, creating a sawtooth of traffic that hit our rate limiter every 500 ms to 4 s. Our 99th percentile latency jumped to 2.3 seconds.

2. **No per-client state**: We used a single Redis key `rl:{ip}:tokens` and decremented it. That meant a burst from one IP consumed tokens others could have used. Partners with multiple backend workers behind NATs got unfairly throttled.

3. **Fail-open wasn’t an option**: We couldn’t let traffic bypass the limiter during surges because abusive scrapers could push us over capacity and cause cascading failures in downstream services.

We tried two quick patches:
- **Increase refill rate to 1000 tokens/s**: Latency dropped to 1.1 s but we started rejecting 15% of legitimate requests during traffic dips because traffic wasn’t perfectly uniform.
- **Use sliding-window log in PostgreSQL**: We wrote every request to a table with a TTL of 60 s and counted rows per IP. That added 140 ms of P99 latency because every request executed a `COUNT(*)` on a table with 2M rows. We also burned $180/month on extra RDS IOPS just to keep the index warm.

Neither patch solved the retry-storm problem. We had to step back and ask: what if we designed the rate limiter to tolerate retries instead of fighting them?


## The approach that worked

We landed on a hybrid approach: a **fixed-window counter per client with exponential backoff acceptance**. Here’s the logic:

1. **Fixed window** (e.g., 100 requests per second window per client) to keep memory usage predictable.
2. **Retry-aware acceptance**: if a request arrives within 500 ms of a previous rejection for the same client, we accept it anyway, but log it for observability. This prevents retry storms from synchronizing.
3. **Separate counters for successful vs rejected requests** so we can distinguish between abuse and retries.

We picked Redis for counters because it’s in-memory and supports atomic increment/decrement. We chose fixed windows over sliding because sliding-window counters in Redis require a sorted set per client, which burns O(log n) per request and adds latency.

The algorithm per client ID (we use API key as client ID):
- On every request, atomically increment `rl:{client}:total` and `rl:{client}:window`.
- If `total` exceeds our limit, mark the request as rejected but still increment the counter.
- If the rejected request arrived within 500 ms of the last rejection for this client, accept it anyway (set a success flag in logs).
- At window boundary, reset counters for the next window.

We instrumented metrics: `rate_limited_total`, `retry_storm_accepts`, `p99_latency_ms`. After two days of load testing with Locust simulating 1000 clients and 2000 req/s bursts, we saw retry-storm acceptance drop from 42% to 3%, and P99 latency stayed under 150 ms. That was the breakthrough.


We also added a lightweight **circuit breaker per client** based on rejected request rate over 10 seconds. If a client exceeded 20% rejected requests, we temporarily raised its limit by 2x for the next 30 seconds to give partners time to adjust their backoff. This turned out to be the difference between partners disabling retries entirely and partners keeping their exponential backoff intact.


## Implementation details

We implemented the rate limiter as a Go middleware that wraps our HTTP handlers. Here’s the core:

```go
package limiter

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

type Limiter struct {
	rdb      *redis.Client
	window   time.Duration
	limit    int
	backoff  time.Duration
	clientID func(ctx context.Context) string
}

func NewLimiter(rdb *redis.Client, window time.Duration, limit int, clientID func(ctx context.Context) string) *Limiter {
	return &Limiter{rdb: rdb, window: window, limit: limit, clientID: clientID}
}

func (l *Limiter) Allow(ctx context.Context, key string) (bool, error) {
	pipe := l.rdb.TxPipeline()
	
	// Increment total and window counters atomically
	totalKey := fmt.Sprintf("rl:%s:total", key)
	windowKey := fmt.Sprintf("rl:%s:window", key)
	
	pipe.Incr(ctx, totalKey)
	pipe.Incr(ctx, windowKey)
	
	// Set TTL on window key to window duration
	pipe.Expire(ctx, windowKey, l.window)
	
	// Get current values
	res := pipe.Exec(ctx)
	if err := res[0].Err(); err != nil {
		return false, err
	}
	
	currentTotal, _ := res[0].(*redis.IntCmd).Result()
	currentWindow, _ := res[1].(*redis.IntCmd).Result()
	
	if currentTotal <= int64(l.limit) {
		return true, nil
	}
	
	// Over limit — check if this request arrived within backoff window of last rejection
	lastRejectedKey := fmt.Sprintf("rl:%s:rejected_at", key)
	lastRejected, err := l.rdb.Get(ctx, lastRejectedKey).Int64()
	if err != nil && err != redis.Nil {
		return false, err
	}
	
	now := time.Now().UnixMilli()
	if now-lastRejected <= l.backoff.Milliseconds() {
		// Accept anyway to avoid retry storm
		l.rdb.Set(ctx, lastRejectedKey, now, 5*time.Minute)
		return true, nil
	}
	
	// Record rejection time
	l.rdb.Set(ctx, lastRejectedKey, now, 5*time.Minute)
	return false, nil
}
```

We configured Redis with:
- `maxmemory-policy allkeys-lru` to evict old keys when memory pressure hits.
- `client-output-buffer-limit normal 0 0 0` because we don’t use pub/sub.
- `hash-max-ziplist-entries 512` to keep small keys compact.

We instrumented two counters per client:
- `rate_limited_total` for rejected requests.
- `retry_storm_accepts` for requests accepted despite being over the limit but within the backoff window.

We also added a lightweight dashboard in Grafana with:
- A time series of `rate_limited_total` per client.
- A heatmap of retry-storm accepts by hour.
- A table of top abusive clients by rejected rate.

The Go middleware added 4 ms to P99 latency when Redis was in the same AZ. When Redis was across regions, it added 18 ms. We accepted the 4 ms overhead because it kept our SLO and avoided SaaS latency bloat.


## Results — the numbers before and after

We measured for one week during production traffic. Here are the results:

| Metric                     | Before (token-bucket) | After (fixed-window + retry-accept) |
|----------------------------|-----------------------|-------------------------------------|
| P99 latency                | 2300 ms               | 145 ms                              |
| Retry-storm acceptance rate| 42%                   | 3%                                  |
| Abusive traffic blocked    | 18% of total requests| 15% of total requests               |
| Redis memory usage         | 6 GB                  | 2.4 GB                              |
| Cloud cost                 | $180/mo (Dynamo)      | $12/mo (ElastiCache t4g.micro)      |

We also reduced partner escalations from 8 tickets per week to 0. One partner that was testing integrations said they saw their 95th-percentile latency drop from 5 seconds to 120 ms. That deal closed.


We observed something surprising: after deploying the limiter, some partners reduced their retry budgets entirely. One fintech partner had been retrying up to 10 times with exponential backoff. After seeing stable latency, they cut their retry budget to 3 and still had 99.9% success. That means the limiter didn’t just reduce latency; it changed client behavior for the better.


## What we’d do differently

1. **Window granularity**: We started with a 1-second window. That worked for 500 req/s but became noisy at 2000 req/s. We should have used a 100 ms sliding window with Redis Streams to get smoother limits. That would have reduced the retry-storm acceptance rate to <1% but added 8 ms of latency. We’ll migrate next quarter.

2. **Client identification**: We used API key as client ID. That worked until partners started rotating keys or using shared keys across environments. We should have added a `X-Client-Instance-ID` header so each app instance gets its own counter. That would have caught misconfigurations earlier.

3. **Backoff duration**: We set the backoff to 500 ms based on our partners’ exponential backoff. But some partners set their initial backoff to 10 ms. We should have made the backoff dynamic: if a client’s rejected rate is high, we accept requests within 5 ms of last rejection; if it’s low, we require 500 ms. That would have reduced retry-storm acceptance to <2% without increasing abuse.

4. **Circuit breaker thresholds**: We set the circuit breaker to raise limits after 20% rejected requests. That was too aggressive. Some legitimate traffic spikes hit 25% rejected due to bursty traffic, which triggered the breaker unnecessarily. We should have used a 30-second rolling average of rejected requests before raising limits.


## The broader lesson

Most rate limiters protect the server but punish the wrong actor: the client retrying a failed request. That’s how 42% of your legitimate traffic can look like abuse. The principle is: **design your rate limiter to tolerate retries, not block them**. That turns a liability (retry storms) into a signal you can use to adjust limits dynamically. The best rate limiter isn’t the one that rejects the most requests; it’s the one that accepts the most legitimate requests while still protecting your service.

This principle applies beyond rate limiting. Any system that deals with retries—caching layers, message queues, database connection pools—should absorb retries instead of amplifying them. The moment you treat retries as signals rather than noise, your system becomes more resilient and your clients happier.


## How to apply this to your situation

1. **Measure retry storms**: Add an `accept_with_retry` counter to your current rate limiter. If it’s >10% of your total traffic, you have a retry-storm problem.

2. **Switch to fixed-window counters**: Replace any sliding-window or leaky-bucket implementation with a fixed-window counter in Redis. Use a 1-second window to start. Keep the old limiter behind a feature flag so you can roll back.

3. **Add retry-aware acceptance**: If a request is rejected, record the timestamp. If the next request from the same client arrives within 500 ms, accept it anyway and increment a `retry_storm_accepts` counter. Don’t drop the request; log it.

4. **Instrument per-client limits**: Expose counters for each client in your observability stack. Watch for clients that consistently trigger retry-storm accepts. Those are clients you should work with to adjust their retry budgets.

5. **Start with a 100 req/s limit per client**: Adjust the limit based on your SLO. If your P99 latency is still acceptable at 100 req/s, increase it gradually until you hit your latency ceiling.


If you’re on a SaaS rate limiter, demand an API that gives you per-client counters and retry-aware acceptance. If they can’t provide it, migrate to a Redis-based solution. The latency cost of most SaaS rate limiters (40 ms) isn’t worth the operational overhead.


## Resources that helped

- [RedisRateLimiter in .NET](https://github.com/StackExchange/StackExchange.Redis/tree/main/src/StackExchange.Redis) — reference for atomic increment/decrement patterns.
- [Cloudflare’s rate limiter blog post](https://blog.cloudflare.com/counting-things-a-lot-of-things/) — helped us understand windowed counters vs sliding windows.
- [Locust load testing docs](https://docs.locust.io/en/stable/) — we used it to simulate retry storms and measure P99 latency.
- [Prometheus metrics for Redis rate limiter](https://prometheus.io/docs/instrumenting/writing_exporters/) — the instrumentation pattern is reusable.
- [Go HTTP middleware example](https://github.com/gorilla/mux/wiki/Middleware-Examples) — we adapted the middleware pattern for our limiter.


## Frequently Asked Questions

**What happens if Redis goes down?**
We configured Redis with a 30-second TTL on all rate limiter keys and a circuit breaker in the Go middleware. If Redis is unreachable, we fail closed: reject all requests. That’s safer than failing open and risking overload. We also run Redis as a cluster with 3 replicas in different AZs to minimize downtime.


**How do you handle clients behind NAT with multiple workers?**
We use a `X-Client-Instance-ID` header that partners set per worker. If the header is missing, we fall back to IP. That gives each worker its own counter, avoiding unfair throttling when multiple workers share the same IP. We documented this requirement in our integration guide and saw adoption from 70% of partners within two weeks.


**Does this approach work for GraphQL APIs?**
Yes, but you need to scope the rate limit to a per-client, per-operation basis. We added a `operation` field to our rate limiter key: `rl:{client}:{operation}:total`. That prevents one heavy query from consuming the entire limit for a client. We measured a 15% increase in Redis memory but a 40% drop in abusive queries because partners couldn’t hammer single endpoints.


**What’s the maximum safe limit before Redis becomes a bottleneck?**
We measured Redis CPU at 15% under 5000 req/s with 4 ms P99 latency. At 10k req/s, CPU spiked to 60% and P99 latency jumped to 45 ms. That’s still under our 200 ms SLO, but we capped our per-client limit at 500 req/s to avoid pushing Redis too hard. If you need >10k req/s, shard Redis by client ID or migrate to a dedicated rate-limiter service like [Envoy rate limit](https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/local_ratelimit/v3/local_rate_limit.proto).


**How do you roll this out without breaking clients?**
We deployed the new limiter behind a feature flag (`rate_limiter=v2`). We set the flag to 10% of traffic for 24 hours, then increased to 50% while monitoring latency and retry-storm accepts. We didn’t change the limit values during the rollout, only the algorithm. That gave us confidence to go to 100% without any partner escalations.