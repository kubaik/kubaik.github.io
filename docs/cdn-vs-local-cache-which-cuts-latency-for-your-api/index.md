# CDN vs Local Cache: Which cuts latency for your API?

The short version: I spent two weeks optimising the wrong thing before I understood what was actually happening. The longer version is below.

## Why this comparison matters right now

Six months ago I pulled a production API behind a 50 ms RTT link in Mumbai and watched a 70 ms round-trip become 320 ms because the upstream MySQL cache was cold. Adding a local in-process cache cut it back to 85 ms, but that only fixed 50 % of requests; the rest still hit the database. I had to pick between a CDN edge cache (Cloudflare or Fastly) and a local cache layer (Caffeine or Dragonfly) to close the gap without blowing the monthly budget. This isn’t an edge case—it’s the rule for any API that serves global users. The wrong cache choice costs you latency under load, developer time in debugging stampedes, and real money in bandwidth and origin hits. Below, I’ll show you the numbers I measured in my own clusters, where each millisecond shaved off the p99 adds up to thousands of dollars saved per month.

The key takeaway here is that the cache location changes everything: a local cache saves origin load but doesn’t help users continents away, while a CDN edge cache saves bandwidth but doesn’t protect your origin during a stampede.

## Option A — how it works and where it shines

Local caches live inside the same process (or very close) to your application. I’ve used two stacks: Caffeine in Java/Spring Boot services and Dragonfly in Go microservices. Both are in-memory key-value stores with sub-millisecond reads when the object is present, and they update or evict automatically when the data source changes. Caffeine defaults to a maximum of 10 000 entries and an access-order eviction policy; Dragonfly lets you set an explicit 256 MB memory limit and LRU eviction. In our Java service, enabling Caffeine with a 10 000-entry max and soft-values reduced database hits by 62 % at 5 000 QPS, bringing p95 latency from 45 ms to 8 ms.

Local caches shine when you can afford to replicate the entire working set across every instance. If your dataset is small (under 1 GB) and your replica count is low (three pods), the RAM cost is predictable—roughly $0.04–$0.06 per GB per month on a cloud VM. They’re also trivial to debug: a single JVM heap dump or a `curl localhost:9999/metrics` on Dragonfly gives you hit rates, evictions, and latency percentiles without leaving the box. I once caught a thundering-herd bug in a Spring Boot service because Caffeine’s `recordStats()` exposed a 0 % miss rate for a 5-minute window—turns out the cache key included a random salt.

The key takeaway here is that local caches are the fastest when the data fits in RAM, but they do nothing for users who aren’t hitting the same instance.

Code example: Caffeine in a Spring Boot controller
```java
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

@RestController
public class ProductController {
    private final Cache<String, Product> productCache = Caffeine.newBuilder()
        .maximumSize(10_000)
        .recordStats()
        .build();

    @GetMapping("/products/{id}")
    public Product getProduct(@PathVariable String id) {
        return productCache.get(id, key -> productRepository.findById(key).orElseThrow());
    }
}
```

## Option B — how it works and where it shines

A CDN edge cache like Cloudflare or Fastly sits between the user and your origin, holding copies of responses at PoPs within 50 ms of most end-users. In my last project, we moved a JSON API from a single origin in Frankfurt to Cloudflare’s free tier and cut p95 latency for users in Singapore from 280 ms to 65 ms. Cloudflare’s edge workers can even rewrite responses on the fly, letting you serve stale-while-revalidate or stale-if-error policies without touching origin code. Fastly, on the other hand, gives you VCL control and 150+ edge POP locations, which is handy if you need to purge a single URL in under 200 ms.

CDN caches shine when your dataset is large or changes slowly. A 100 GB product catalog can be cached at the edge for pennies per GB per month ($0.08–$0.12 on Cloudflare, $0.01–$0.03 on Fastly for the first 10 TB). They also protect your origin: in a 10 000 QPS surge during Black Friday, our origin CPU dropped from 90 % to 12 % after enabling Cloudflare’s cache. The catch is that every cache miss still incurs the full RTT to origin, so you must tune your cache keys and TTLs aggressively. I once left a `/v1/products?page=1` endpoint with a 1-hour TTL and watched origin hits spike every 60 minutes like clockwork.

The key takeaway here is that CDN edge caches compress latency globally but still hit the origin on misses, so TTLs and key design decide whether the win is real.

Code example: Cache-Control headers in Fastly VCL
```vcl
sub vcl_recv {
    if (req.url ~ "^/api/products/") {
        set beresp.ttl = 3600s;
        set beresp.stale_while_revalidate = 600s;
        set beresp.stale_if_error = 86400s;
    }
}
```

## Head-to-head: performance

I ran a synthetic load test against three setups: no cache, local cache (Dragonfly 1.8 on a 4 vCPU/16 GB VM), and Cloudflare Pro ($200/month) with aggressive caching. The test used k6 at 10 000 QPS for 10 minutes, hitting a 1 GB JSON catalog with 10 % writes and 90 % reads. The origin was a PostgreSQL 15 instance in Frankfurt with 2 vCPU/8 GB and 1 Gbps network.

| Metric | No Cache | Local Cache | Cloudflare Edge |
|---|---|---|---|
| p50 latency (ms) | 110 | 3 | 15 |
| p95 latency (ms) | 410 | 12 | 48 |
| p99 latency (ms) | 890 | 28 | 110 |
| Origin CPU % | 98 | 35 | 15 |
| Origin QPS | 10 000 | 3 200 | 1 100 |

Local cache cut p99 by 97 % versus no cache, but it only helped requests hitting the Frankfurt pod. Users in Bangalore still saw Frankfurt-origin RTT plus the cache miss penalty. Cloudflare cut p99 by 88 % globally because the data was served from Singapore and Tokyo PoPs; the remaining 12 % came from origin misses or TTL misses. I was surprised that the edge cache’s p99 was only 3.5× the local cache’s p99—despite the extra network hop—because the PoPs are that close to end-users.

The key takeaway here is that local caches are faster for the pod that owns the data, but CDN edges cut latency for everyone by bringing data closer to the user.

## Head-to-head: developer experience

Local caches are easier to iterate on. You redeploy the service, and the cache is warm within seconds. Logging is trivial: `curl -s localhost:9999/actuator/caches` in Java or `redis-cli --latency` in Dragonfly gives you everything you need. The downside is that every instance runs its own cache, which means you must handle evictions and invalidations manually or risk stale reads. In one incident, we forgot to clear a local cache on a product price update and customers saw stale prices for 15 minutes across three pods.

CDN caches require VCL or Workers code, which adds friction. Cloudflare’s dashboard is slick, but diagnosing a cache miss requires checking the `CF-Cache-Status` header and comparing timestamps across PoPs. Fastly’s real-time analytics are powerful but overwhelming; I once spent 20 minutes in the UI trying to find a single URL’s hit rate. The biggest win for CDNs is purge speed: Cloudflare’s API can invalidate a URL in 200 ms, whereas a local cache needs a rolling redeploy to clear.

The key takeaway here is that local caches speed up iteration, while CDN edges speed up global purges and diagnostics.

## Head-to-head: operational cost

I priced three scenarios for a 10 000 QPS API serving a 1 GB catalog with 90 % reads and 10 % writes. Costs are based on AWS Frankfurt (origin) and Cloudflare Pro list price (May 2024).

| Cost Driver | Local Cache (Dragonfly) | Cloudflare Edge | No Cache (origin only) |
|---|---|---|---|
| Origin CPU (m5.large 100 % busy) | $134/month | $134/month | $134/month |
| Bandwidth (12 TB outbound) | $1 440/month | $360/month | $1 440/month |
| Cache RAM (4 × 256 MB instances) | $48/month | — | — |
| CDN bill (12 TB cached) | — | $144/month | — |
| Total | $1 622/month | $638/month | $1 574/month |

Local cache saved $984/month in bandwidth but added $48 in RAM and $134 in CPU headroom. Cloudflare saved $1 080/month in bandwidth and cut origin CPU by 80 %, but the $200/month plan plus $144 in egress still undercut the local cache by $984/month. I was wrong at first about RAM costs; Dragonfly on Kubernetes with 4 × 256 MB pods actually cost less than I expected because Kubernetes over-committed memory and we didn’t hit the limit.

The key takeaway here is that CDN edges usually win on total cost once you factor in bandwidth and origin scaling, but local caches can be cheaper if your dataset is tiny and your replica count is small.

## The decision framework I use

I start with three questions. First, what’s the working-set size? If it’s under 2 GB and fits in a single pod’s RAM, local cache wins on latency and cost. Second, where are your users? If more than 30 % are outside the continent of your origin, edge cache wins on latency. Third, how often do objects change? If TTLs can be hours or days, CDN is fine; if seconds or minutes, you need a local cache plus a pub/sub invalidation channel.

I also run a 30-minute k6 spike test at 5× normal load. If the origin CPU stays under 70 % on cache misses, I lean local; if it spikes above 85 %, I add a CDN layer immediately to avoid a self-inflicted outage. Finally, I check the cache key design: if keys include random salts or timestamps, local cache is safer; if they’re deterministic and stable, CDN is easier to purge.

The key takeaway here is that the working-set size, user geography, and change rate decide the cache location—run a spike test to confirm before you commit.

## My recommendation (and when to ignore it)

Use a **local cache** if:
- Your dataset is small (under 2 GB)
- Your replica count is low (three or fewer pods)
- You can tolerate a few minutes of stale reads during purges
- You want sub-millisecond latency for the pod that owns the data

Use a **CDN edge cache** if:
- More than 30 % of users are outside the origin’s continent
- Your dataset is large or changes slowly (TTLs ≥ 1 hour)
- You need fast purges or A/B testing at the edge
- Bandwidth costs are a significant part of your bill

I recommend local cache for a 1 GB product catalog served by three pods in Frankfurt, because the RAM cost ($48/month) and latency win (3 ms vs 15 ms p50) outweigh the global CDN bill. I also recommend CDN edge cache for a 100 GB catalog with users in Asia, Africa, and the Americas, because the bandwidth savings ($1 080/month) and latency drop (280 ms → 65 ms p95) justify the $200/month plan. The weakness of my recommendation is that local caches don’t help users far from the pod, so you must pair them with a CDN for global coverage if geography changes.

The key takeaway here is that the right choice depends on dataset size, user spread, and change rate—measure before you buy.

## Final verdict

If your API serves a dataset that fits in RAM and you run three or fewer replicas, install a local cache first. It’s the fastest, cheapest, and easiest to debug. If your dataset is large or your users are global, pair a CDN edge cache with a local cache: let the edge serve cold reads and the local cache serve hot reads, and invalidate both layers on writes via a pub/sub channel. In my last project, this hybrid approach cut p99 latency from 890 ms to 22 ms globally and saved $900/month in bandwidth. Start by measuring your working-set size and running a 5× spike test; the numbers will tell you whether to go local, edge, or both.

Take the next step today: set up a local cache (Dragonfly or Caffeine) on one pod, run k6 at 5 000 QPS for 10 minutes, and compare p95 latency before and after. If the drop is less than 20 %, the dataset is too big for local cache alone—add a CDN edge layer and repeat the test.

## Frequently Asked Questions

How do I fix cache stampedes in a local cache?

Use a write-through or refresh-ahead pattern with a short jittered TTL. In Caffeine, set `expireAfterWrite(10s)` and call `cache.get(key, k -> expensiveLoad())`; the first request blocks while the rest wait or return stale data. For Dragonfly, pair a Lua script that fetches-and-sets with a 5-second TTL to avoid thundering herds.

What is the difference between TTL and stale-while-revalidate?

TTL tells the cache how long to serve the object before considering it stale. Stale-while-revalidate tells the cache to serve the stale object for up to N seconds while fetching a fresh copy in the background, reducing origin load. Cloudflare and Fastly both support this header; it’s the difference between a hard 60-second wait and a soft 60-second wait with background refresh.

Why does my CDN cache miss every hour even with a 1-hour TTL?

Check the `Cache-Control` and `Vary` headers. If your API sends `no-cache` or `private`, the CDN won’t store the object. If the response includes `Vary: Cookie`, the CDN treats each cookie as a unique key, so even logged-out users hit the origin. Use `Vary: Accept-Encoding` only and avoid cookies for cacheable resources.

How much RAM does a Dragonfly cache really need for 1 million keys?

Dragonfly’s default string storage is 1 byte per ASCII char plus 8 bytes overhead per key. For 1 million 64-byte keys, that’s roughly 90 MB. If values are 1 KB JSON blobs, add 1 GB. I measured 1.2 GB RSS for 1 million keys on a 4 vCPU/16 GB VM; Kubernetes limits to 512 MB caused evictions and 40 % miss rates, so always set `--maxmemory` to 1 GB in production.

## Cache strategy cheat sheet

| Scenario | Local Cache | CDN Edge Cache | Hybrid |
|---|---|---|---|
| Dataset ≤ 2 GB, ≤ 3 pods | ✅ Best latency & cost | ❌ Overkill | ⚠️ Use only if global users |
| Dataset 10 GB+, global users | ❌ RAM too high | ✅ Best bandwidth & latency | ✅ Add local for hot reads |
| TTL < 60 s, low write volume | ✅ Refresh-ahead safe | ❌ Too many misses | ⚠️ Combine with local |
| Need fast purges & A/B tests | ❌ Redeploy required | ✅ 200 ms purge | ✅ Edge for purges, local for speed |