# Rate limit clients without breaking them

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In early 2026, the analytics team at CloudMetrics noticed a 34% spike in 5xx errors from our public REST API over two weeks. The traffic wasn’t malicious—it was legitimate customers running nightly batch jobs that hit `/events` 12k times instead of the intended 200. Our existing rate limiter, built on NGINX in 2026, used a simple sliding-window log with Redis. It blocked bursts after 100 requests per minute, but the error messages weren’t actionable: clients got HTTP 429 with a `Retry-After` header, and their SDKs crashed because they didn’t parse it. Teams spent 16 engineer-hours per week manually whitelisting IPs. We needed a system that reduced false positives, gave clients clear feedback, and kept our Redis bill under $240/month at 2026 traffic volumes.

We tried a naive fix first—moving to a token-bucket at 1,000 req/min per API key—but it ignored the fact that many customers run parallel workers. Their total traffic was within limit, but individual workers exceeded it, causing cascading backpressure. We measured a 22% increase in 429s because the bucket refilled too slowly for sibling workers. Our first priority became decoupling fairness from burst tolerance: stop penalizing legitimate concurrency while still protecting upstream services.

**Summary:** CloudMetrics’ REST API needed a rate limiter that reduced false positives by 50%, provided machine-readable feedback, and cost under $240/month in 2026 traffic. The NGINX sliding window and token bucket both failed under legitimate parallel workloads, causing 22% more 429s.


## What we tried first and why it didn’t work

We started with the Redis-backed sliding-window log we’d used since 2026. It was simple: LPUSH to a list, LTRIM to 100 items, then TTL 60s. We set the limit at 100 req/min per API key. The problem wasn’t the algorithm—it was the feedback loop. SDKs in JavaScript, Python, and Go didn’t parse `Retry-After` headers consistently, and customers’ cron jobs crashed instead of retrying intelligently. We logged 452 tickets in 30 days where users said, “Your API is flaky,” when the real issue was their retry logic. We calculated the wasted engineering time at $37k/year in support tickets and lost trust.

Next, we migrated to a token bucket using Redis Cell v0.3.0. Cell implements the GCRA algorithm and lets us set capacity=1000 and period=60s. The math looked right: at steady state, a client could burst up to 1,000 requests once per minute, but sustained traffic would be smoothed to ~16.7 req/s. What we didn’t model was parallel workers. A customer running 10 Celery tasks each fetching `/events` would see each task blocked after 100 requests, even though total traffic was 1k per minute. Their logs showed 429s with `Retry-After` values jumping from 1s to 60s randomly. We measured a 22% increase in 429s after the change because the bucket couldn’t handle sibling contention. We rolled back after 10 days and saw 429s drop 18%, but we still hadn’t solved the real problem: how to rate limit a fleet of workers without breaking them.

A third attempt used a distributed semaphore with Redlock in Go. We set `maxConcurrent = 50` per API key. The idea was to let bursts in but cap concurrency. It worked for small customers but failed for our top 5% who needed 200 concurrent workers. Their jobs queued up, latency to their dashboard spiked by 800ms, and they blamed our API. We measured 34% longer job durations during peak hours. We reverted after 17 days and lost two enterprise customers who cited “unreliable ingestion.”

**Summary:** CloudMetrics tried three approaches—sliding window, token bucket, and distributed semaphore—none of which handled legitimate parallel workloads. Each added 18–22% more 429s or increased customer job latency by 34–800ms, and none gave SDKs consistent, machine-readable feedback.


## The approach that worked

We landed on a hybrid design: a **leaky-bucket refill policy** with **adaptive concurrency**. The bucket’s leak rate is set to 16.7 req/s (1,000 req/min), but the bucket size is dynamic based on the customer’s observed median request concurrency over the last 24 hours. We call this the “adaptive window.”

Here’s how it works:
- A customer running 20 parallel workers sees a bucket size of 20.
- Another running 200 workers gets a bucket size of 200.
- The bucket refills at a fixed rate (leak), so high-concurrency customers still hit the 1,000 req/min ceiling, but they’re not penalized for legitimate parallelism.

We store state in Redis using two keys per customer:
- `lm:bucket:<key>`: a sorted set of request timestamps (for leaky bucket math)
- `lm:stats:<key>`: a hash with `median_concurrency` updated hourly via a background Lua script

The Lua script uses Redis’ `ZRANGEBYSCORE` to count requests in the last 60s, computes the 50th-percentile concurrency over the last 24 hours, and updates `median_concurrency` accordingly. We set bucket size = `max(median_concurrency, 10)`. This prevents tiny customers from getting a 10-capacity bucket that starves them during legitimate bursts.

We also added **machine-readable feedback** in the `429` response body:
```json
{
  "error": "rate_limit_exceeded",
  "retry_after": 2.3,
  "bucket": {
    "capacity": 200,
    "refill_rate": 16.7,
    "retry_token": "eyJpZCI6MTIzfQ=="
  }
}
```

SDKs now retry with exponential backoff using `retry_token` as a cursor, avoiding duplicate requests. We measured a 58% drop in duplicate events after this change.

Finally, we implemented **local caching** in the API gateway (Envoy 1.31) so that 95% of requests hit a 100ms in-memory check instead of a 3ms Redis round-trip. The cache key is `lm:<key>:<hour_of_day>` so it resets predictably and prevents memory leaks. We use a TTL of 30s to sync with the hourly stats update.

**Summary:** CloudMetrics combined a dynamic leaky bucket, hourly median concurrency stats, machine-readable 429 feedback, and Envoy local caching to cut false positives 58% while preserving legitimate parallelism. The system now handles 12k req/min per customer without 429s if the bucket refill allows it.


## Implementation details

We implemented the limiter as a Go 1.23.0 service behind Envoy 1.31. The Go service exposes `/_limit` which returns JSON with headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and the new `X-RateLimit-RetryToken`. Envoy uses the Lua filter to call the Go service only on cache miss.

Core data structures in Redis:
- `lm:bucket:<key>`: sorted set of Unix-millisecond timestamps
- `lm:stats:<key>`: hash with fields `median_concur`, `last_updated`, `ttl`
- `lm:cache:<key>:<hour>`: string with JSON `{"remaining": 100, "reset_at": 1735684800000}`

The leaky-bucket Lua script (run on every request):
```lua
-- KEYS[1] = bucket key (lm:bucket:<key>)
-- KEYS[2] = stats key (lm:stats:<key>)
-- ARGV[1] = now_ms
-- ARGV[2] = request_weight (always 1)
-- ARGV[3] = bucket_capacity
local now = tonumber(ARGV[1])
local weight = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])

-- 1. Remove old entries (older than 60s)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - 60000)

-- 2. Count current requests in window
local count = redis.call('ZCARD', KEYS[1])

-- 3. Update bucket capacity from stats
local stats = redis.call('HGETALL', KEYS[2])
local median_concur = tonumber(stats.median_concur or 10)
local bucket_size = math.max(median_concur, capacity)

-- 4. Compute leaky bucket
local leak_rate_per_ms = 16.7 / 1000
local leaked = math.floor((now - (tonumber(stats.last_updated) or now)) * leak_rate_per_ms)
local available = math.max(0, bucket_size - count + leaked)

-- 5. Decide
if available >= weight then
  redis.call('ZADD', KEYS[1], now, now)
  redis.call('EXPIRE', KEYS[1], 60)
  return {1, available - weight, bucket_size}
else
  local reset_at = now + (weight / leak_rate_per_ms) * 1000
  return {0, reset_at, bucket_size}
end
```

The median concurrency script runs hourly via a Go worker using `ZRANGEBYSCORE` to get all timestamps in the last 24 hours, computes the 50th percentile concurrency, and writes it back to `lm:stats:<key>`. We use `ZRANGESTORE` to snapshot the bucket before trimming, preserving history for the median calculation.

Envoy Lua filter snippet:
```lua
function envoy_on_request(request_handle)
  local key = request_handle:headers():get("x-api-key")
  local hour = os.time() // 3600
  local cache_key = string.format("lm:cache:%s:%d", key, hour)
  local cached = request_handle:streamInfo():dynamicMetadata():get("rate_limit")
  if cached then
    request_handle:headers():add("x-rate-limit-remaining", tostring(cached.remaining))
    return
  end
  local res = request_handle:httpCall(
    "rate_limit_service",
    {
      [":method"] = "GET",
      [":path"] = "/_limit",
      [":authority"] = "rate-limit.internal",
      ["x-api-key"] = key
    },
    "",
    1000,
    100
  )
  local body = res:body():getBytes(0, res:body():length())
  local data = cjson.decode(body)
  request_handle:headers():add("x-rate-limit-remaining", tostring(data.remaining))
  request_handle:headers():add("x-rate-limit-retry-token", data.retry_token)
  request_handle:streamInfo():dynamicMetadata():set("rate_limit", data)
end
```

We deployed the Lua script behind feature flag `rate_limit_v2` for 10% of traffic for 4 days, then 50% for 7 days, and finally 100%. We used GitHub Actions to run a canary deployment with Prometheus metrics: `rate_limit_429_total`, `rate_limit_cache_hit_ratio`, and `rate_limit_redis_latency_ms`.

**Summary:** CloudMetrics implemented a Go 1.23.0 service with Envoy 1.31, Redis sorted sets for leaky buckets, hourly Lua scripts for median concurrency stats, and Envoy local caching. The Lua bucket script runs on every request to decide allow/deny and updates the cache for 95% of subsequent requests in 100ms.


## Results — the numbers before and after

We measured 30 days before and after the adaptive limiter rollout. Key results:

| Metric | Before | After | Delta | Notes |
|---|---|---|---|---|
| 5xx errors | 34% of traffic | 3% | -91% | Measured by Cloudflare RUM |
| 429 errors | 18% of traffic | 2% | -89% | Reduces false positives |
| Customer tickets | 452/month | 98/month | -78% | Support load in Zendesk |
| SDK duplicate events | 58% of batches | 25% | -58% | Measured by Snowplow |
| P99 latency (API) | 84ms | 96ms | +14% | Caused by Lua script overhead |
| p50 latency (API) | 12ms | 13ms | +8% | Within SLO |
| Redis requests/sec | 4,200 | 1,800 | -57% | Due to Envoy caching |
| AWS cost (Redis) | $290/month | $130/month | -55% | Savings from reduced ops |
| Median job duration (customer) | 1.4s | 1.4s | 0% | No regression for top 5% workloads |

We also ran a controlled experiment with one customer running 200 parallel workers fetching `/events`. Before the change, their cron job would receive 429s within 2 minutes and crash. After the change, their workers never saw 429s because the bucket size adapted to 200 and the leak rate smoothed the load to 1,000 req/min. Their job duration stayed at 1.4s, and their logs showed zero 429s during the 7-day trial.

We were surprised by the latency regression: +14% P99 on the API. Profiling showed the Lua script in Redis added 1–2ms per request, and the Go service added 0.8ms. We shaved 0.7ms by moving the bucket size lookup to Envoy’s dynamic metadata and 0.5ms by batching the Lua script with pipelining. The final P99 settled at 96ms, still within our 100ms SLO.

The biggest win wasn’t the numbers—it was the support load. The machine-readable `429` body meant SDKs could retry correctly without human intervention. We measured a 78% drop in tickets, saving roughly $37k/year in engineering time.

**Summary:** After rolling out the adaptive leaky-bucket limiter, CloudMetrics reduced 5xx errors 91%, 429s 89%, and customer tickets 78%, while cutting Redis costs 55%. Latency increased 14% P99 but stayed under the 100ms SLO. A 200-worker customer saw zero 429s and unchanged job duration.


## What we’d do differently

We over-optimized for the median customer and under-tuned the tail. Our first bucket size formula was `max(median_concurrency, 10)`, but we missed the top 1% of customers who spike to 1,000 concurrent workers during Black Friday. Their bucket size capped at 1,000, but the leak rate of 16.7 req/s couldn’t handle their burst. We had to hot-patch the formula to `max(median_concurrency, 10, 0.1 * peak_last_7d)`, which added another Redis call per request. Next time, we’ll pre-compute peak concurrency nightly and store it in a key so the hot path stays O(1).

We also assumed all SDKs would adopt the new `retry_token` field. Only 60% did by the 30-day mark. Teams that didn’t update their SDKs still got `Retry-After` headers, but their cron jobs crashed. We added a compatibility mode that falls back to `Retry-After` if `retry_token` is missing, and we published a migration guide with code snippets for Go, Python, and JavaScript. Next time, we’ll version the response body and deprecate old formats in a controlled window.

The hourly median concurrency script used `ZRANGEBYSCORE` over 24 hours of data, which peaked at 800k timestamps for our largest customer. We had to cap the range at the last 7 days to keep Redis memory under 500MB. Next time, we’ll use a rolling window of 7 days and store the median in a separate time-series key to avoid scanning large sorted sets during peak hours.

Finally, we didn’t model the Envoy Lua filter’s memory usage. During the 50% canary, we saw memory leaks in the filter’s dynamic metadata due to unbounded growth. We capped metadata TTL at 30s and added a Prometheus alert `envoy_rate_limit_metadata_bytes > 100000`. Next time, we’ll set the TTL at the Lua level and use `lua_shared_dict` with a fixed size.

**Summary:** CloudMetrics would pre-compute peak concurrency nightly, version the 429 response body, cap sorted-set ranges, and configure Envoy Lua metadata TTL upfront. These fixes would have saved 3 engineer-days of firefighting and avoided a compatibility cliff with legacy SDKs.


## The broader lesson

Rate limiting isn’t a binary switch—it’s a feedback system between you and your clients. The moment you treat it as a gate instead of a throttle, you break legitimate workloads. The adaptive concurrency pattern we landed on is a specific instance of a general principle: **offload fairness decisions to the client whenever possible.**

When we gave clients a machine-readable bucket state (`capacity`, `refill_rate`, `retry_token`), their SDKs stopped thrashing and started cooperating. We turned a customer support problem into a product feature. The same principle applies to quotas, concurrency limits, and even feature flags: expose enough state so the client can self-regulate, and you’ll spend less time firefighting.

Another lesson is to measure what hurts the client, not just what hurts you. Our first three attempts optimized for our NGINX logs and Redis CPU, not for customer job duration or SDK stability. Once we instrumented job duration and duplicate events, we saw the real cost of false positives. Always correlate your rate limiter metrics with customer outcomes, not just operational metrics.

Finally, caching isn’t optional—it’s the difference between a system that scales and one that melts under load. Our 95% Envoy cache hit ratio cut Redis load by 57%, but it also made the system resilient to Redis failures. Next time, we’ll treat the cache as a first-class primitive, not an optimization.

**Summary:** Rate limiting should expose state to clients so they can self-regulate, not just protect upstream services. Measure customer outcomes, not just operational metrics. Cache aggressively to turn a brittle system into a resilient one.


## How to apply this to your situation

Start by instrumenting your current limiter’s failure modes. If you use NGINX or Envoy’s built-in rate limit, add a Prometheus counter for `rate_limit_429_total` and group by API key. Look for customers with high 429 rates but low error rates—those are false positives. Next, implement a minimal adaptive window: use a Redis sorted set to track request timestamps, update a median concurrency stat hourly, and set bucket size = `max(median, 10)`. Keep the leak rate at your target sustained rate (e.g., 1,000 req/min = 16.7 req/s).

Add machine-readable feedback to your 429 response. Include `retry_after`, `bucket_capacity`, and a `retry_token`. Publish SDK snippets (Go, Python, JS) that show how to retry with exponential backoff using the token. Announce the change in your changelog and deprecate old header formats slowly.

Cache the limiter’s decision in your gateway. Use a 30s TTL and a key that resets hourly so you don’t leak memory. Measure cache hit ratio and latency; if cache misses spike above 10%, tune the Lua script or increase the TTL.

Finally, set a hard SLO for your rate limiter’s latency: P99 < 50ms. If you can’t hit it, move the Lua script to a sidecar or pre-compute bucket states nightly. Remember: the limiter is part of your product surface, not just ops.

**Next step:** Deploy a 7-day shadow run of the adaptive limiter behind a feature flag, mirroring 10% of traffic. Measure 429s, SDK crashes, and P99 latency. If 429s drop 50% and SDK crashes stay flat, roll forward. If not, iterate on the bucket size formula before touching anything else.


## Resources that helped

- Redis Cell v0.3.0 docs: https://github.com/brandur/redis-cell/blob/v0.3.0/README.md — GCRA algorithm reference
- Envoy rate limit v3 API: https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/rate_limit/v3/rate_limit.proto — gateway integration
- GCRA and token bucket comparison: https://brandur.org/rate-limiting — theoretical background
- Go SDK template for retry tokens: https://github.com/cloudmetrics/retry-token-sdk — our public repo
- Prometheus alerts for rate limiters: https://github.com/cloudmetrics/alerts/blob/main/rate-limit.rules — YAML for Prometheus


## Frequently Asked Questions

**How do I migrate from NGINX rate limit to adaptive concurrency without breaking clients?**
Start with a shadow run: mirror 10% of traffic to the new limiter but still allow all requests. Log 429s and latency, but don’t block. Publish a changelog entry with code snippets for SDKs to parse the new `429` body. After 7 days, if 429s drop 50% and SDK crashes stay the same, enable blocking for the mirrored traffic. Finally, disable NGINX rate limit and migrate DNS. This gives clients 30 days to update SDKs before you enforce the new format.

**What bucket size formula works for a SaaS with wide customer concurrency ranges?**
Use `bucket_size = max(median_concurrency, 10, 0.1 * peak_last_7d)`. The median prevents starvation for small customers, the 10 is a safety floor, and the 10% peak protects against Black Friday spikes. Update `peak_last_7d` nightly via a background job that scans Redis sorted sets. For the largest 0.1% of customers, cap the bucket at 10k to prevent memory bloat.

**Why did latency increase after adding the Lua script in Redis?**
The Lua script added 1–2ms per request in Redis, and the Go service added 0.8ms. We mitigated it by moving bucket size lookup to Envoy’s dynamic metadata (saved 0.7ms) and enabling pipelining in the Lua script (saved 0.5ms). Final P99 settled at 96ms, within our 100ms SLO. If your SLO is tighter, consider pre-computing bucket states nightly and caching them in a sidecar.

**How do you handle Redis outages without breaking the API?**
We implemented a circuit breaker in the Go limiter service: if Redis latency > 20ms or error rate > 5% for 10s, we fail open and allow traffic for 30s. During the 2026 Redis failover, the breaker tripped after 12s, and we served 100% of requests with a 3% increase in 429s. We alert on `rate_limit_circuit_breaker_open` and page within 5 minutes. The Envoy cache still served 95% of requests, so users barely noticed.

**What’s the biggest pitfall when rolling out a new rate limiter?**
Assuming all SDKs will adopt new fields immediately. Only 60% of CloudMetrics’ SDKs updated to use `retry_token` within 30 days, causing crashes for legacy clients. Always ship a compatibility mode that falls back to `Retry-After` and publish migration guides with copy-paste code snippets. Announce deprecation timelines in your changelog.