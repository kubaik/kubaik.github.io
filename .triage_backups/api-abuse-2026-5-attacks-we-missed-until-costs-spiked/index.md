# Tiered API abuse defense: match cost to attacker economics

Most API security guidance assumes a clean environment and a patient timeline. Production provides neither. This article covers abuse patterns that show up as cost anomalies rather than as blocked requests, why the obvious first defenses often fail, and how to build a tiered defense whose cost is proportional to each attack's economics.

## The situation

A recognizable scenario: a REST API on AWS Lambda (Python 3.12, FastAPI) costs three times what was budgeted while request volume has barely moved. CloudWatch shows a large share of requests returning 429 before reaching the handler. A CDN and a web application firewall sit in front of the origin, and the bill still climbs.

The useful question is not "which OWASP category applies" but "which abuse patterns are profitable for an attacker against this specific stack." The patterns below are worth modeling because they consume disproportionate backend resources per unit of attacker effort:

1. **Cache-stampede amplification via conditional GETs.** A client sends `If-None-Match` with a stale validator; the origin recomputes an expensive response and returns 304. The attacker pays for one small request; the origin pays for full computation.
2. **JWT `kid` header swap.** Each request carries a different `kid`, forcing a key lookup (often a network call to a JWKS endpoint or cache miss) on every request. Cheap to generate, expensive to serve.
3. **GraphQL depth attacks.** A single deeply nested query can return megabytes of JSON from a small request body.
4. **Distributed credential stuffing.** Many source IPs, low per-IP rate, high aggregate rate. Per-IP limits do not see it.
5. **Cold-start abuse.** Bursts of requests to a cold function drain concurrency and inflate latency for legitimate traffic, sometimes triggering retries that amplify load.

A representative stack for reasoning about this:
- AWS Lambda (Python 3.12, arm64), modest memory allocation
- API Gateway HTTP API
- A CDN in front, with edge compute for token validation
- Redis (ElastiCache) for rate-limit counters
- A managed web application firewall with a managed rule group

## What teams try first, and why it often fails

**Attempt 1: maximum-sensitivity managed rules.**
Turning up a managed rule group across all distributions is the default first move. Two things commonly go wrong. First, false positives on health checks and legitimate clients rise, so the team either whitelists broadly (reopening the hole) or spends time tuning. Second, inspection cost per request rises because the firewall examines more of each request. The fix is not "more sensitivity" but "fewer, targeted rules based on observed traffic."

**Attempt 2: per-IP fixed-window rate limiting.**
A typical decorator looks like this:

```python
from fastapi import Request, HTTPException
from redis.asyncio import Redis

async def rate_limit_ip(request: Request, limit: int = 100, window: int = 60):
    key = f"rl:{request.client.host}"
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, window)
    if current > limit:
        raise HTTPException(status_code=429, detail="Too Many Requests")
```

Two problems. Fixed windows allow a burst at the boundary (a client can send `limit` requests at the end of one window and `limit` again at the start of the next). More importantly, the key is per IP, so distributed attacks pass straight through. Attackers also pivot to a single endpoint that triggers expensive work inside the handler, moving cost from the edge to the database.

**Attempt 3: JWT `kid` validation at the edge.**
Moving validation to edge compute is meant to reject malformed tokens before they reach the origin. What breaks first is the execution timeout. Edge compute has a hard timeout (commonly a few seconds), and a key-lookup function that occasionally exceeds it will time out and retry, amplifying load. The practical fix is a short explicit timeout plus a local LRU cache of recently seen keys.

## A tiered defense

Stop trying to block everything at one layer. Match each defense to the economics of the attack it addresses.

| Tier | Where | Purpose | Typical cost driver |
|------|-------|---------|---------------------|
| Edge | CDN edge compute | Reject malformed tokens and unknown `kid` cheaply | Per-request edge compute time |
| Rate | Firewall + Redis counters | Sliding-window limits, including per-token not just per-IP | Counter storage and lookups |
| Origin | Application middleware | Request-shape analysis, GraphQL depth limits | Lambda duration |
| Data | Database | Query-time guards, statement timeouts | Database CPU and I/O |

### 1. Sliding-window rate limits with Redis

Replace fixed windows with a sliding algorithm such as GCRA (Generic Cell Rate Algorithm) implemented as a Lua script so the read-modify-write is atomic:

```lua
-- GCRA rate limiter for Redis
-- KEYS[1]: rate limit key
-- ARGV[1]: limit (max requests per period)
-- ARGV[2]: period in seconds
-- ARGV[3]: current time in seconds
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local period = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local state = redis.call('HMGET', key, 'tokens', 'last')
local tokens = tonumber(state[1]) or limit
local last = tonumber(state[2]) or now

-- Refill tokens based on elapsed time
local elapsed = math.max(0, now - last)
tokens = math.min(limit, tokens + (elapsed * limit / period))

if tokens < 1 then
  redis.call('HSET', key, 'tokens', tokens, 'last', now)
  redis.call('PEXPIRE', key, period * 1000)
  return {0, math.floor(tokens)}
end

tokens = tokens - 1
redis.call('HSET', key, 'tokens', tokens, 'last', now)
redis.call('PEXPIRE', key, period * 1000)
return {1, math.floor(tokens)}
```

Call it from FastAPI with `redis.eval(GCRA_LUA, 1, key, limit, period, now)`. Key the limiter on a stable identity — an API token or authenticated subject — not just the source IP, so distributed attacks are counted together.

### 2. JWT `kid` handling

Validate `kid` against a JWKS that is refreshed on a fixed schedule and cached at the edge. Include `kid` in the cache key so a valid key is served from cache and an unknown `kid` fails fast. Reject unknown `kid` with 400 before any origin call. Keep the edge validation timeout well under the platform's hard limit so a slow JWKS fetch cannot cascade into retries.

### 3. GraphQL depth limiting

Add a depth analyzer in middleware. The visitor pattern below counts nesting depth and raises before resolvers execute:

```python
from graphql import parse, visit
from graphql.language.visitor import Visitor
from fastapi import HTTPException

class DepthVisitor(Visitor):
    def __init__(self, max_depth=6):
        self.max_depth = max_depth
        self.current_depth = 0

    def enter(self, node, *args, **kwargs):
        if hasattr(node, "selection_set") and node.selection_set:
            self.current_depth += 1
            if self.current_depth > self.max_depth:
                raise HTTPException(status_code=400, detail="Query too deep")

def limit_depth(query: str, max_depth=6):
    ast = parse(query)
    visitor = DepthVisitor(max_depth)
    visit(ast, visitor)
    return visitor.current_depth
```

Note that this counts the maximum depth reached during traversal, not the depth of the final node visited; if you need the true maximum, track `self.max_depth_seen` separately. Depth checks run in low single-digit milliseconds for typical queries and reject oversized queries before any resolver executes. Pair with a response-size cap at the data layer as a backstop.

### 4. Cold-start abuse

If the runtime supports snapshot-based startup (for example, Lambda SnapStart on Java), enabling it reduces cold-start latency substantially. Otherwise, provisioned concurrency on the auth function absorbs bursts without draining the shared pool. Provisioned concurrency is billed continuously, so size it to your baseline and let on-demand handle the rest.

## Implementation details

**Redis sizing.** A single small node becomes a bottleneck as request rate climbs, and failover during a restart makes rate-limit counters unavailable. During that window, traffic reaches the origin unbounded. Run rate-limit state in its own Redis cluster, separate from session cache. Use AOF persistence if you accept the write cost, or accept that counters reset on failover and size the origin to survive a short burst.

**Conditional GET amplification.** A firewall rule can block requests carrying `If-None-Match` when the response would be large. The rule shape is a byte-match on the `if-none-match` header:

```json
{
  "Name": "ConditionalGETAmplification",
  "Priority": 1,
  "Statement": {
    "ByteMatchStatement": {
      "SearchString": "If-None-Match",
      "FieldToMatch": { "SingleHeader": { "Name": "if-none-match" } },
      "TextTransformations": [{ "Priority": 0, "Type": "NONE" }]
    }
  },
  "Action": { "Block": {} },
  "VisibilityConfig": {
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "ConditionalGETAmplification"
  }
}
```

Blocking all conditional GETs will break legitimate caching, so scope the rule to endpoints whose responses are expensive to recompute, and monitor the sampled-request metric before enforcing.

**Database guards.** Add a statement timeout and a maximum response size at the data layer. A connection pooler in transaction mode prevents connection exhaustion under burst. A hard cap on returned payload size catches the cases that slip past the depth limiter.

## How to measure whether any of this is working

Do not trust a single before/after table. Instrument these:

- **Cost per million requests** — divide your monthly bill by requests served, per endpoint if possible. Track the trend, not a single number.
- **P95 and P99 latency** — from your load balancer or API gateway access logs, not from application logs, so you see the full path.
- **429 rate and where it is emitted** — edge, firewall, or application. A rising 429 rate at the edge means your limits are too tight; a rising 429 rate at the application means attackers are reaching the origin.
- **Database CPU** — the signal that tells you abuse has moved past the edge.
- **Cold-start count per hour** — from your platform's metrics, if available.

To measure a specific attack's cost, run a controlled load test against a staging endpoint that mimics production shape. Compare cost per request before and after each defense is enabled. The number that matters is the marginal cost of serving an abusive request, not the total bill.

## A worked example

Suppose a GraphQL endpoint serves a query that returns 2 MB of JSON. At an illustrative compute cost of $0.00002 per request-second and 400 ms of compute per request, each request costs roughly $0.000008 to serve. At 1,000 requests per second sustained for an hour, that is 3.6 million requests, or about $28.80 per hour in compute alone, before database cost. If a depth limit rejects those queries at 2 ms of middleware time instead, the same 3.6 million requests cost about $0.14. The arithmetic is illustrative, but the shape is the point: the ratio between "serve the query" and "reject the query" is what determines whether a defense pays for itself.

## What to do differently

**1. Model attacker economics before choosing a defense.** Build a simple table: attack type, cost to run per million requests, cost to block per million requests, and value if it succeeds. Any row where blocking costs more than running is not worth engineering time. Any row where running is far cheaper than blocking is where you should invest.

**2. Isolate rate-limit state.** Do not share a Redis instance between session cache and rate-limit counters. A failover in the shared instance takes both down.

**3. Validate `kid` at the issuer where possible.** If your identity provider can sign and publish keys in a way that makes unknown `kid` cheap to reject, push validation there rather than paying edge compute for every request.

**4. Use short, explicit edge timeouts.** A timeout that is close to the platform hard limit will cause retries under load. Set it well below the limit and alert on timeout rate.

## FAQ

**What is the smallest change that reduces abuse cost today?**
Enable snapshot-based startup or provisioned concurrency on the auth function. Cold-start abuse is often the cheapest attack to mitigate because the fix is a configuration change, not new code.

**How do I know if `kid` validation is working?**
Watch edge logs for execution errors mentioning JWKS or key lookup. A low, stable rate means the cache is serving; a rising rate means the refresh interval is too long or the cache key is wrong.

**Should I limit GraphQL depth or payload size first?**
Depth first. A shallow query with wide fields can still return a large payload, but depth is the cheapest signal to compute and rejects the most common amplification pattern before resolvers run. Add payload size as a backstop.

**Is a separate Redis cluster for rate limits worth the cost?**
If your API handles more than a few thousand requests per second, yes. Below that, a single node with persistence disabled may be fine, but understand the failover window and size the origin to survive it.

## Next step in the next 30 minutes

Open your rate-limit middleware and change the key from the source IP to the authenticated subject or API token. Run your existing load test for 15 minutes and compare the 429 rate at the edge against the 429 rate at the application. If the application rate drops, distributed abuse was previously invisible to your limiter and the change is worth keeping.
