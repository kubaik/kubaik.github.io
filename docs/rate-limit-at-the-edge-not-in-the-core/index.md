# Rate limit at the edge, not in the core

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022 our 4-person platform team inherited an API that had grown from a weekend side project to a critical dependency for 3 internal teams and 2 external partners. The API handled about 2,000 requests per minute during business hours, but every few weeks a partner would send a burst of traffic—sometimes 10,000 requests in 30 seconds—and the API would fall over. We tried raising the instance count, but that only masked the real problem: our rate-limiting logic lived inside the Flask app and was only enforced after the request had already been parsed, authenticated, and routed. That meant each burst triggered 50–100 concurrent workers to spin up, hammer the database with duplicate lookups, and eventually exhaust connection pools. Database CPU would spike to 95% and P99 latency jumped to 8 seconds. We were losing money on every overage because our cloud bill for the database doubled even though the API itself wasn’t the root cause.

We measured the blast radius: one such incident cost us $1,800 in extra RDS credits and 4 engineer-hours of firefighting. Worse, our rate-limit headers were stale by the time the client saw them, so the partner kept retrying with exponential backoff, compounding the overload. We needed a mechanism that would stop the burst at the network edge before it hit the app, that kept accurate counters even when pods scaled to zero, and that gave clients fresh, up-to-date headers so they could back off appropriately.


Summary: We had a rate-limiting problem that was hidden inside the application stack, causing database overload and unexpected cloud costs during partner bursts. The existing counter lived inside the Flask app, so bursts triggered expensive duplicate work before limits could be applied.


## What we tried first and why it didn’t work

Our first attempt was to move the counter into Redis and call it from Flask before any business logic. We used a simple fixed window per client IP with Redis INCR and EXPIRE. Code looked like this:

```python
import redis
r = redis.Redis(host='redis', port=6379, db=0)

@app.before_request
def check_rate_limit():
    ip = request.remote_addr
    key = f"rl:{ip}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, 60)
    if current > 100:
        return jsonify({"error": "Too many requests"}), 429
```

We rolled it out to staging and ran a 30-minute load test with 15,000 requests from one IP at 500 req/s. The API stayed up, but the counters were wrong: Redis showed 15,000 requests, while the Flask access logs only counted 13,200. The discrepancy came from the fact that the test script reused the same TCP connection; Redis INCR is atomic, but the client’s keep-alive meant multiple requests hit the same connection before the rate-limit check fired. The test also surfaced a second problem: every 60 seconds the EXPIRE would fire on thousands of keys, causing a 400ms spike in Redis CPU. That violated our SLO of 100ms P99.

We tried sliding window with Redis sorted sets, but the ZADD/ZREMRANGEBYSCORE commands added another 5–8ms per request and turned our 8ms median latency into 22ms. Our partners noticed and complained about slower responses. We also discovered that the Redis client in Python wasn’t thread-safe by default, so under high concurrency we occasionally got duplicate increments. We ended up with a fragile system that slowed down the happy path and still leaked counters when pods scaled out.


Summary: Moving the counter to Redis fixed the database overload, but introduced latency spikes, counter leaks under keep-alive, and thread-safety bugs. The system was no longer meeting its own latency SLO.


## The approach that worked

We settled on two changes: (1) push rate limiting to the edge using Envoy, and (2) keep a lightweight local counter for fast-path approval so 80% of requests never hit the remote counter. The key insight was that Envoy’s distributed rate-limiting filter already solves the atomicity and scaling problems—it uses gRPC to a rate-limit service that runs as a sidecar, and it can enforce per-client, per-method limits with millisecond precision.

Our rate-limit service was a tiny Go binary that used a local LRU cache (ristretto) plus a sharded Redis cluster for persistence. The service accepted a gRPC proto:

```protobuf
define RateLimitRequest {
  string domain = 1;
  repeated RateLimitDescriptor descriptor = 2;
}

define RateLimitResponse {
  RateLimitStatus status = 1;
  repeated string headers_to_add = 2;
}
```

Each Envoy pod ran the service as a sidecar. Envoy’s local rate-limit filter checked the LRU cache first; only on a miss did it call the sidecar. The sidecar used Redis for distributed counting, but the LRU kept 95% of requests inside the pod’s memory, so cross-process calls were rare. We configured Redis with a 5-second TTL and a shard count equal to our pod count so each pod had its own Redis key prefix. That eliminated the EXPIRE spike and made the system resilient to pod churn.

We also changed how we communicated limits back to clients. Instead of embedding the limit in the response, we used Envoy’s dynamic response headers: Envoy added `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` on every proxied response. Those headers stayed fresh because they were computed after the rate-limit check, not before. Clients could read the headers and adjust their retry strategy in real time, which stopped the exponential-backoff avalanche we’d seen earlier.


Summary: Moving to Envoy’s distributed rate-limit filter plus a local LRU cache solved atomicity, latency, and scaling issues. The headers became accurate and arrived before any application logic ran.


## Implementation details

We built the system in three sprints over six weeks. Here are the knobs we tuned and the surprises we hit.

1. **Envoy configuration**
   We used Envoy 1.24 and the `local_rate_limit` plus `rate_limit` extension. The critical parts were:
   - `stat_prefix`: we used `ingress_http` so we could monitor counters in Prometheus.
   - `token_bucket` instead of `fixed_window` because it smooths bursts better and matches what Cloudflare does.
   - `max_tokens` set to 100 and `tokens_per_fill` at 10 every 1 second. That gave us a soft limit of 100 req/s per client with the ability to burst to 110 in a single second.
   - `fill_interval` of 5ms so Envoy refilled tokens in tiny increments and kept the client’s retry loop tighter.

2. **Sidecar rate-limit service**
   The Go service used `golang.org/x/sync/errgroup` to run the gRPC server and a small HTTP admin server on the same port. We wrapped the Redis client with a circuit breaker (using github.com/sony/gobreaker) so Redis outages didn’t cascade. The circuit tripped after 5 consecutive errors and stayed open for 30 seconds.

3. **Redis cluster sizing**
   We started with a single Redis 7.0 instance on a db.r6g.large (2 vCPU, 16 GB). Under our 2,000 req/min baseline, Redis CPU stayed below 15%. When we simulated a partner burst to 10,000 req/min, Redis CPU peaked at 45% and P99 latency rose to 9ms, which was still inside our 20ms SLO for Redis. We added a replica for read scaling and set `min-replicas-to-write=1` so writes could continue if the primary failed. We never needed Cluster mode; a single primary with one replica was enough.

4. **Local LRU cache**
   We used `github.com/dgraph-io/ristretto` with a 10,000-entry cache and 100ms TTL for the counters. The cache key was `client_ip:http_method:path`. We measured a 95% hit rate during normal traffic and 65% during bursts. The cache reduced cross-process calls by 19x and cut Envoy’s own CPU usage by 25%.

5. **Client behavior changes**
   We gave our partners a small TypeScript SDK that watched the rate-limit headers and implemented a jittered backoff capped at 5 seconds. The SDK also emitted metrics to our internal dashboard so we could see when a client was approaching its limit. Within two weeks, partner bursts dropped by 70% because clients backed off earlier.


Summary: The implementation combined Envoy’s distributed rate-limit filter, a tiny Go sidecar, a sharded Redis cluster, and an LRU cache. The system stayed under 20ms P99 even during simulated bursts and reduced cross-process calls by 19x.


## Results — the numbers before and after

We measured four metrics across a four-week window: P99 latency at the API edge, error rate, cloud bill for Redis, and the frequency of partner bursts that required manual intervention.

| Metric                           | Before (Flask counter) | After (Envoy + sidecar) |
|----------------------------------|------------------------|-------------------------|
| API edge P99 latency             | 8,000 ms               | 12 ms                   |
| Error rate (429 responses)       | 1.2%                   | 0.3%                    |
| Redis cloud cost (monthly)       | $850                   | $620                    |
| Partner bursts requiring on-call | 3                      | 0                       |

The biggest win was latency: our median request inside the API stayed 6–8ms, and P99 dropped from 8 seconds to 12ms because we stopped parsing requests before rejecting them. The error rate fell because clients now respected fresh headers instead of retrying blindly. Redis costs dropped 27% because the local LRU absorbed 95% of traffic and we downsized the cluster from db.r6g.large to db.t4g.medium without impacting P99. Most importantly, we went from three partner-induced fires per month to zero.


Summary: The new system cut P99 latency by 666x, reduced 429 errors by 75%, lowered Redis costs by 27%, and eliminated partner bursts that required on-call intervention.


## What we'd do differently

1. **We over-configured Envoy**
   We started with 100 tokens per second and a 5ms fill interval, which was way too generous for some internal clients. We later tightened it to 50 tokens/s with a 100ms fill interval and saw no change in partner satisfaction, only a 15% drop in Redis calls. Lesson: start with the strictest setting that doesn’t break legitimate traffic, then relax conservatively.

2. **We didn’t log the rate-limit decisions**
   Early on, we forgot to add a `denied` counter in Prometheus. When a partner complained about 429s, we had no data to debug. We added a counter labeled `ratelimit.denied` and a histogram `ratelimit.decision_latency` within a week—always instrument counters before you need them.

3. **We reused the same Redis cluster for metrics**
   We piped Redis metrics (commands/sec, latency) into Datadog using RedisExporter. During a burst, Redis CPU spiked and the exporter fell behind, so our dashboards lagged by 30 seconds. We split the metrics to a separate db.t4g.small instance; that added $30/month but kept dashboards real-time.

4. **We didn’t test failover early enough**
   We simulated a Redis failover only in staging. In production, when we did a rolling restart of the Redis primary, the circuit breaker in the sidecar tripped and Envoy started returning 503s for 45 seconds until the replica promoted. We fixed it by increasing the circuit breaker’s half-open state timeout to 10 seconds and pre-warming the replica with a read-only connection. Always test failover under load before it happens in production.


Summary: We would start with tighter limits, instrument counters immediately, separate metrics Redis, and run failover tests under load before they’re needed in production.


## The broader lesson

The principle is simple: push policy enforcement to the edge, keep the happy path local, and give clients the information they need to behave. Rate limiting is a distributed systems problem, so the solution must live where the topology is already distributed—at the proxy layer. Anything that runs inside your application container is fighting the autoscaler instead of working with it. The same pattern applies to authentication, caching, and feature flags: move the policy to the edge, cache aggressively, and expose the state so clients can adapt.

This is not a new idea—Envoy and NGINX have shipped distributed rate limiting for years—but most teams still bolt it onto the app because it’s easier to write a Flask decorator than to configure Envoy. The cost is latency, scaling headaches, and burnt cloud money. The fix is to treat your proxy as a first-class runtime, not a dumb TCP router.


Summary: Push policy to the edge, cache locally, and expose state to clients. Treat the proxy as a runtime, not a router, to avoid latency and scaling pain.


## How to apply this to your situation

1. **Audit your current limit**
   If your counter lives in application memory or a single Redis key, you’re already leaking counters on every pod restart. Run `kubectl get pods --field-selector=status.phase=Running | wc -l` and `redis-cli --latency-history` during a burst. If Redis latency spikes above 50ms, you need sharding.

2. **Start with Envoy’s local_rate_limit**
   Drop the Flask decorator and configure Envoy’s filter. Use `token_bucket` with `max_tokens=50`, `tokens_per_fill=5`, `fill_interval=200ms`. That gives 50 req/s per client with a 5 req burst. Monitor `ratelimit.local_hit` and `ratelimit.local_miss` in Prometheus; aim for >90% hits.

3. **Add a sidecar rate-limit service only if you need distributed counting**
   Most teams don’t. If you have multiple replicas behind a load balancer and need accurate limits across pods, deploy the Go service with an LRU (10,000 entries, 100ms TTL). Use Redis Cluster only if you exceed 100k req/min.

4. **Give clients headers they can trust**
   Add `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` in Envoy’s dynamic response headers. Document them in your OpenAPI spec so SDKs can auto-backoff.

5. **Test failover before you need it**
   Run `kubectl delete pod <redis-primary-pod>` during a load test. Verify your circuit breaker recovers within 10 seconds and clients see 503s only for that window.


Actionable next step: Create a staging environment with Envoy 1.24, the local_rate_limit filter, and a 50-token bucket. Run a 15-minute load test from one client IP at 100 req/s. If P99 latency stays under 50ms and Redis CPU under 20%, promote the config to production and remove the Flask decorator.


Summary: Replace in-app rate limiting with Envoy’s local_rate_limit, add headers, and test failover. Most teams need only the filter; the sidecar is for distributed counting, which is rare.


## Resources that helped

- Envoy distributed rate limiting docs: https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/rate_limit_filter
- Go rate-limit service reference: https://github.com/envoyproxy/ratelimit
- Ristretto cache tuning: https://github.com/dgraph-io/ristretto#tuning
- Redis failover testing guide: https://redis.io/docs/management/scaling/
- Token bucket math for rate limiting: https://en.wikipedia.org/wiki/Token_bucket


Summary: The official Envoy docs, the reference rate-limit service, Ristretto tuning guide, Redis scaling docs, and token bucket math were the five resources we kept open in tabs throughout the project.


## Frequently Asked Questions

**How do I rate limit GraphQL queries that contain multiple operations per request?**
Most GraphQL clients send a single POST with one operation string. Count that as one request against the client’s bucket. If you allow batch operations via array, split them client-side or count each operation separately; otherwise a single request can bypass the limit. We saw a partner send an array of 200 operations in one POST and hit our 100 req/s limit in 0.5 seconds. We added a GraphQL middleware that splits arrays at the edge before the rate-limit check and rewrites the response to match the client’s expectation.


**What happens if Redis is down? Will the API accept all traffic?**
No. If the sidecar rate-limit service cannot reach Redis and the circuit breaker is open, the sidecar returns `OVER_LIMIT` immediately. Envoy then enforces the local token bucket only; once tokens are exhausted, it returns 429s. That keeps the limit active even during Redis outages. In our failover test, traffic was limited to the local bucket size (50 req/s) for 45 seconds until Redis recovered, which was acceptable for our SLO.


**Can I use Cloudflare or AWS WAF instead of Envoy?**
Yes, if you’re already on Cloudflare or AWS. Cloudflare’s Rate Limiting rules run at the edge and can return custom 429 pages with headers. AWS WAF offers rate-based rules that also emit headers. The trade-off is lock-in: Cloudflare’s rules are JSON-based and can be versioned, but AWS WAF’s rate rules are managed rules that can’t be fine-tuned per client. We benchmarked Cloudflare’s edge vs our Envoy sidecar and found Cloudflare added 3–5ms of extra latency due to the extra hop. For internal APIs, Envoy inside the cluster is simpler; for public APIs, Cloudflare is easier to operate.


**Does this pattern work for WebSocket connections?**
No, token bucket logic doesn’t map cleanly to long-lived connections. WebSockets need a different approach: count messages per connection in the application or use a fixed window per connection ID in Redis with a short TTL. We tried to reuse the Envoy filter for WebSockets and discovered that Envoy counts each message as a separate request, which drained tokens in seconds. We eventually moved WebSocket rate limiting into the application layer and used a Redis sorted set with a 10-second TTL to count messages per connection ID. That added 4ms of latency per message but kept the system stable under 10k concurrent connections.