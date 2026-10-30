# Cutting cloud egress costs: architecture that works

Egress charges are one of the least visible line items in a cloud bill, and one of the fastest growing. The reason is rarely a single large download. More often it is the cumulative effect of thousands of small requests crossing a regional or provider boundary: an internal service calling an external API, a log shipper forwarding events to a third-party observability tool, a database replica being queried from the wrong region, or a microservice that quietly talks to a SaaS endpoint hosted on another continent.

This article covers the architecture decisions that reduce egress cost, how to reason about the tradeoffs, and how to measure whether a change actually helped. The focus is on patterns, not vendor-specific claims, because the underlying mechanics are similar across providers: data that stays inside a region or inside a private network path is typically cheaper or free, and data that crosses to the public internet or another region is billed.

## Why egress is an architecture problem, not a traffic problem

A common failure mode looks like this: a workload migrates to a new region to satisfy data residency requirements, traffic volume is unchanged, and the egress bill roughly doubles. The migration itself is not the cause. The cause is that the architecture was already dependent on data leaving the region, and the new topology made that dependency more expensive.

Consider a service that makes 1,000 requests per second to an external API, each carrying a 3 KB JSON payload. That is:

- 1,000 requests/sec x 3 KB = 3,000 KB/sec = 3 MB/sec
- 3 MB/sec x 86,400 sec/day = 259,200 MB/day, about 253 GB/day
- 253 GB/day x 30 days = about 7.6 TB/month

At an illustrative $0.09/GB, that is roughly $684/month for a single integration. Multiply that across a handful of external dependencies and the number grows quickly. The per-request cost is trivially small; the aggregate is not. This is why "optimize the API calls" rarely fixes the problem. The fix is usually to stop the calls from crossing the boundary at all, or to shrink what crosses.

The same arithmetic applies to observability. If every service ships structured logs to a third-party endpoint, the log volume — not the application traffic — becomes the egress driver. Instrumenting egress per service, not per user, is what surfaces this.

## How to evaluate egress reduction options

Before comparing patterns, establish what you are optimizing for. Four dimensions matter:

**Cost.** Compute the reduction in billable bytes, not the reduction in request count. A change that halves requests but doubles payload size saves nothing.

**Latency.** Measure 95th percentile response time, not averages. Averages hide the tail that generates support tickets. Distributed tracing is the right tool; compare the same endpoint before and after the change under comparable load.

**Operational complexity.** Score honestly. Adding a new gateway, a cache layer, or a second region each adds failure modes, on-call surface, and configuration drift. A pattern that saves money but requires a new team to operate it may not be a net win.

**Compliance risk.** If data residency is a requirement, any pattern that moves data across a boundary must be justified. Patterns that keep data in-region are easier to defend in a data protection impact assessment than patterns that route data through a third party.

A weighted score is useful, but the weights are a business decision. A regulated workload may weight compliance above cost; a consumer app may weight latency above both. The point is to make the tradeoff explicit rather than defaulting to the cheapest option.

## Patterns that reduce egress

### 1. Keep internal service-to-service traffic inside the region

The highest-leverage change is usually the simplest: ensure that services calling other services resolve to endpoints in the same region, and that calls to provider-managed services (object storage, queues, key-value stores) go through private network endpoints rather than the public internet.

Private endpoints for managed services typically eliminate public internet egress for that traffic and keep it on the provider's private network. The tradeoff is configuration work: endpoint policies, security groups, and IAM changes. For teams already running microservices, the change is often limited to service discovery configuration and network policy.

**Failure mode to watch:** a service that resolves a regional endpoint at startup but caches a stale DNS answer after a failover. Verify that clients re-resolve endpoints and that health checks cover the private path, not just the public one.

### 2. Cache third-party responses at the edge with explicit TTL and size limits

If an external API returns data that is stable for even a short window, caching it at the edge avoids repeated egress for identical requests. The design decisions that matter are TTL (how long the data is acceptable), cache key (what makes two requests "the same"), and size limit (to prevent a single large response from dominating the cache).

```javascript
// Edge function: set cache TTL by path pattern
function handler(event) {
  const request = event.request;

  // Only cache GET requests
  if (request.method !== 'GET') {
    return request;
  }

  // TTL by path pattern
  const path = request.uri;
  if (path.startsWith('/posts') || path.startsWith('/users')) {
    request.headers['cache-control'] = { value: 'public, max-age=300' };
  } else {
    request.headers['cache-control'] = { value: 'public, max-age=60' };
  }

  return request;
}
```

Note that this example sets the request header so the origin response can be cached; the exact mechanism depends on the CDN. The important part is the TTL policy, which should be derived from the upstream API's documented rate limits and the tolerance for stale data — not guessed.

**Failure mode to watch:** cache stampede. When a popular entry expires, many concurrent requests miss simultaneously and hit the origin. Mitigate with request coalescing or a short jitter added to TTLs. Also be careful never to cache responses containing personal data unless the cache itself is in-region and covered by your data processing agreements.

### 3. Compress payloads before they cross a boundary

JSON compresses well, often by 60–80% for typical API responses, because it is repetitive text. Enabling gzip or brotli on the response path reduces billable bytes directly. The cost is CPU on both ends and a small amount of latency.

This is one of the few changes that is almost always worth doing, because it requires no architectural change — just configuration. Measure the actual compression ratio for your payloads rather than assuming; binary formats and already-compressed content gain little.

### 4. Use private connectivity to SaaS providers where available

Some SaaS providers offer private network connectivity, so traffic between your VPC and the provider does not traverse the public internet. This reduces or eliminates public egress for those integrations and can simplify the compliance story, since the traffic path is documented.

**Failure mode to watch:** not all providers support this, and setup requires coordination. Verify the provider's documented support before designing around it, and confirm the failover path if the private connection degrades.

### 5. Replicate data regionally and read locally

For multi-region applications, the expensive pattern is querying a database in another region. Replicating data so that reads are served locally and only writes cross regions reduces cross-region traffic substantially. Managed replication features for relational and key-value databases make this operationally feasible, but they introduce eventual consistency.

**Failure mode to watch:** conflict resolution. If writes can occur in more than one region, define the merge policy before enabling replication, not after the first conflict. Also measure the replication lag under peak load; a lag that is acceptable at low traffic may violate read-after-write expectations at high traffic.

### 6. Strip unnecessary fields at the edge

If an API response contains fields the client does not need — internal identifiers, large nested objects, full records when a summary suffices — removing them at the edge reduces egress volume. This is essentially payload minimization and it doubles as a privacy control.

**Failure mode to watch:** over-aggressive stripping that breaks clients. Treat the transformation as a versioned interface and test it against real client behavior, not just schema validation.

### 7. Centralize outbound routing to control what leaves

For larger organizations, routing outbound traffic through a controlled egress path makes it possible to observe and, where appropriate, block traffic to high-cost or low-value destinations. This is a governance pattern more than a cost pattern: the savings come from being able to see and act on egress, not from the routing itself.

**Failure mode to watch:** the routing layer becomes a bottleneck and a single point of failure. Size it for peak, and monitor the added hop's latency.

## Patterns that often fail to pay off

**Global acceleration services.** These are designed to reduce latency by routing traffic over provider backbones, not to reduce egress cost. Depending on the topology, they can route traffic through additional regions and increase inter-region transfer. Evaluate them for latency goals, and measure egress impact separately.

**Third-party CDNs with origin shielding.** A CDN can reduce bandwidth cost to end users, but the CDN-to-origin leg is itself billable egress. Whether the net is positive depends on cache hit ratio and the CDN's origin-fetch pricing. Model both legs before migrating.

**Multi-cloud failover.** Cross-cloud egress is typically priced at or above public internet rates, and operating identity, networking, and data consistency across two providers adds substantial overhead. Multi-cloud is justified by specific availability or procurement requirements, not by egress savings.

**Serverless databases with cross-region replication.** Replication is convenient, but the replication traffic itself is billable. For high-write workloads, the sync cost can exceed the savings from serving reads locally. Model the write volume before adopting.

## Choosing a pattern: a decision checklist

Work through these in order. The first applicable answer usually determines the highest-impact change.

1. **Is any internal traffic leaving the region?** If yes, fix that first. It is the cheapest change with the largest effect.
2. **Are you calling external APIs frequently with small payloads?** If yes, evaluate caching and compression before considering network changes.
3. **Are logs or analytics events leaving the region?** If yes, this is often the largest single contributor and the easiest to relocate.
4. **Are database reads crossing regions?** If yes, evaluate regional replicas, and model the replication traffic before committing.
5. **Is a SaaS integration a major contributor?** If yes, check whether the provider offers private connectivity.
6. **Do you have visibility into egress per service?** If no, fix that before changing architecture. You cannot prioritize what you cannot attribute.

## How to measure egress reduction honestly

Any change should be validated with before-and-after measurement under comparable conditions. What to instrument:

- **Billable bytes by service and usage type.** Cloud cost and usage reports can be grouped by service and usage type. Filter for data transfer line items.
- **Request volume and average payload size per integration.** This lets you compute egress per request, which is the metric that reveals small-payload amplification.
- **95th percentile latency for affected endpoints.** Use distributed tracing, and compare the same endpoint under comparable load.
- **Cache hit ratio**, if you added caching. A low hit ratio means the cache is not paying for itself.

A useful first command is a cost-and-usage query grouped by service and usage type, filtered to the data transfer usage types:

```bash
# Top services by data transfer cost over a 30-day window.
# Adjust the dates and region to match your account.
aws ce get-cost-and-usage \
  --time-period Start=2026-06-01,End=2026-07-01 \
  --granularity DAILY \
  --metrics "BlendedCost" "UsageQuantity" \
  --group-by Type=DIMENSION,Key=SERVICE Type=DIMENSION,Key=USAGE_TYPE
```

Review the output for usage types containing "DataTransfer". That tells you which services are responsible and whether the transfer is internet egress, inter-region, or intra-region. Then drill into the top service and correlate its egress with request volume from your application metrics.

For a change to be worth keeping, the reduction should be measurable, sustained across a full billing cycle, and not achieved by degrading latency or reliability beyond what the workload tolerates.

## A worked example

Suppose a service makes 2,000 requests per second to an external API, each with a 4 KB response, and none of it is cached or compressed.

- 2,000 x 4 KB = 8,000 KB/sec = 8 MB/sec
- 8 MB/sec x 86,400 = 691,200 MB/day, about 675 GB/day
- 675 GB/day x 30 = about 20.3 TB/month

At an illustrative $0.09/GB, that is roughly $1,827/month.

Now apply two changes: gzip compression, which for this payload yields a 70% reduction, and edge caching with a 60-second TTL, which serves 80% of requests from cache.

- Compressed payload: 4 KB x 0.30 = 1.2 KB
- Egress after compression: 20.3 TB x 0.30 = 6.09 TB
- Egress after caching: 6.09 TB x 0.20 = 1.22 TB

At the same illustrative rate, that is roughly $110/month — a reduction of about 94% from the original. The exact numbers depend on your payloads, cache hit ratio, and provider pricing; the point is the order of operations. Compression is free to try and always helps. Caching helps proportionally to hit ratio. Neither requires moving the service.

## FAQ

**Why does egress cost more than ingress?**

Ingress is typically free or cheap because providers compete to receive data. Egress is billed because it represents data leaving the provider's network and consuming shared capacity. This asymmetry is standard across major providers, though the specific rates differ. Treat egress as a cost you design around, not one you can negotiate away.

**Is egress pricing going up?**

Rates vary by provider, region, and destination, and they change. Rather than relying on a remembered figure, check your provider's current pricing page and your own cost and usage report. The architecture patterns in this article reduce the billable volume regardless of the rate.

**Can caching violate data protection rules?**

Yes, if the cached data contains personal data and the cache is outside the region or operated by a third party not covered by your agreements. Cache only non-personal, stable responses at the edge, and keep anything sensitive in-region.

**How do I find the top egress sources?**

Use your provider's cost and usage report, grouped by service and usage type, filtered to data transfer line items. Correlate the top services with application-level request and payload metrics. The combination tells you both where the cost is and why.

**Do I need to refactor the whole application?**

No. Start with the highest-volume endpoints and the largest single contributor, which is often logging or analytics. Moving one high-volume integration in-region can produce a measurable change without touching the rest of the system.

## What to do in the next 30 minutes

Open your cloud cost console, filter the last 30 days for data transfer usage types, and write down the top three services by egress cost along with their usage type (internet, inter-region, or intra-region). That single list tells you whether your problem is external APIs, cross-region database traffic, or log shipping — and which pattern from this article to apply first.
