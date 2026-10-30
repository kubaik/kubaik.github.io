# Agent Systems: Circuit Breakers Are Not Enough

Circuit breakers and bulkheads are the standard resilience playbook, inherited from the microservices era. They work well when failures are binary: a dependency is up or down, fast or slow. Agent systems break that assumption in three specific ways, and a naive port of the microservices playbook leaves teams with a false sense of security and cascades that are harder to diagnose than the original outage.

This article covers the failure modes that circuit breakers miss, what to add instead, and how to measure whether the additions actually help. The code examples use .NET, but the patterns apply to any language with a policy library.

## Why the microservices playbook is incomplete for agents

A classic circuit breaker trips when a downstream call fails or times out repeatedly. That model assumes failures are *resource-shaped*: the dependency is overloaded, unreachable, or slow. When the breaker opens, the system stops hammering a struggling dependency and gives it room to recover. When the dependency recovers, the breaker closes.

Agent systems violate three of the assumptions underneath that model:

1. **Failures can be logical, not resource-shaped.** An agent can return a well-formed, fast, HTTP 200 response that is semantically wrong — a malformed plan, a request routed back to the sender, a "no data" result that is actually a validation error. A circuit breaker that only inspects status codes and latency will never trip on these, and a breaker that trips on *any* error will open on expected outcomes and stay open.
2. **Agents are stateful.** In-process caches, short-term memory, and per-agent context mean that a bulkhead sized for a stateless handler will saturate for reasons that have nothing to do with downstream load. Cache churn, eviction thrash, and lock contention all consume the same thread pool that the bulkhead is trying to protect.
3. **Agents call each other recursively.** A single user request can fan out into a tree of sub-tasks, and the number of concurrent downstream calls can grow multiplicatively with depth. A circuit breaker on the leaf service will trip, but by then the tail latency of the original request has already been destroyed by the fan-out itself.

Each of these has a specific fix. None of them is a circuit breaker.

## Failure mode 1: self-referential feedback loops

Consider two agents, A and B. A receives a request, decides it needs enrichment, and forwards to B. B inspects the payload, decides it is malformed, and forwards it back to A for re-validation. A sees a fast round-trip (a few milliseconds) and, depending on how the breaker is configured, either ignores it or trips immediately.

The pathological case is the second one. If the breaker trips on any error, it opens on a *logical* failure and never cools down, because the underlying services are healthy and there is nothing to recover. Every subsequent external request is rejected instantly. The system looks down while every dependency is up.

The fix is a **semantic circuit breaker**: inspect the error payload, not just the status code, and classify failures into two buckets.

- **Tripping failures**: timeouts, connection errors, 5xx responses, rate-limit responses. These indicate the dependency is unhealthy. Count them toward the breaker threshold.
- **Non-tripping failures**: validation errors, "no data" results, business-rule rejections. These are expected outcomes. Log them, return them to the caller, but do not count them toward the breaker.

In practice this means a predicate on the response body. With Polly, `HandleResult` can inspect the deserialized payload:

```csharp
AsyncCircuitBreakerPolicy<LlmResult> breaker = Policy
    .HandleResult<LlmResult>(r =>
        r.IsTransientFailure ||          // timeout, 5xx, rate limit
        r.StatusCode >= 500)
    .CircuitBreakerAsync(
        handledEventsAllowedBeforeBreaking: 5,
        durationOfBreak: TimeSpan.FromSeconds(30));
```

The predicate is the whole point. A breaker without one is a coin flip on whether a logical error takes down the service.

There is a second-order problem: a feedback loop between two agents can bounce a request back and forth indefinitely without ever producing a tripping failure. Add a **hop counter** to the request envelope and reject any request that exceeds it. This is cheap and catches the loop before it becomes a latency problem.

## Failure mode 2: state-leak bulkhead saturation

Bulkheads isolate a dependency by capping concurrent calls to it. The classic model assumes the handler is stateless: a thread is either waiting on the downstream call or it is not.

Agents are not stateless. A typical agent holds an in-process cache of recent embeddings, tool results, or conversation context. When traffic spikes, the cache eviction policy starts thrashing: every request spends time evicting and re-fetching instead of doing useful work. The bulkhead's thread pool fills with threads that are *not* waiting on the downstream dependency — they are waiting on the cache. The bulkhead opens and rejects traffic even though the downstream service is healthy.

This is a state-leak: mutable state inside the protected region consumes the same capacity that the bulkhead is trying to ration.

The fix is to **externalize mutable state** so the bulkhead protects only compute and I/O, not memory management. Move the cache to a shared store with a TTL:

```csharp
public class CacheService
{
    private readonly IDatabase _db;
    private readonly TimeSpan _ttl;

    public CacheService(IConnectionMultiplexer redis, TimeSpan ttl)
    {
        _db = redis.GetDatabase();
        _ttl = ttl;
    }

    public async Task<string?> GetAsync(string key) =>
        await _db.StringGetAsync(key);

    public async Task SetAsync(string key, string value) =>
        await _db.StringSetAsync(key, value, _ttl);
}
```

Two details matter. First, the TTL must be set on every write, not just at the store level, so that a burst of writes cannot evict unrelated entries. Second, the cache key must be deterministic and stable — if it depends on a hash of an object whose serialization order varies, hit rates collapse and the cache becomes pure overhead.

Once state is externalized, the bulkhead behaves as designed: it caps concurrent calls to the LLM, and cache lookups that hit do not consume a bulkhead slot at all. A cache hit should short-circuit before the policy wrap, not inside it.

## Failure mode 3: recursive dependency explosion

A single agent request can delegate to a sub-agent, which delegates to three helpers, each of which calls a shared service. If depth grows linearly, the number of concurrent leaf calls grows multiplicatively. At depth 1 there is 1 call; depth 2, 3 calls; depth 3, 9; depth 4, 27; depth 5, 81. The arithmetic is exact and the growth is the problem: a shared service sized for a few thousand requests per second can be overwhelmed by a handful of deep requests.

The circuit breaker on the shared service will eventually trip, but it trips *after* the fan-out has already happened. The original request has already spawned its tree, and the tail latency experienced by the user is the sum of the slowest path through that tree, not the latency of any single call.

Two guards are needed, and they are complementary:

1. **A max recursion depth.** Reject any delegation that would exceed a configured depth. This is a hard cap, not a heuristic. It is the only thing that bounds the fan-out.
2. **A fallback that returns partial results.** When the depth cap is hit, return what has been computed so far with an explicit marker that the result is partial. A user who gets a partial plan in 200 ms is better served than one who gets a full plan in 20 seconds or a timeout.

```csharp
public record AgentRequest(string Prompt, int Depth = 0);

public const int MaxDepth = 3;

public async Task<Plan> HandleAsync(AgentRequest request)
{
    if (request.Depth >= MaxDepth)
        return Plan.Partial(request.Prompt, "max recursion depth reached");

    var subRequests = Decompose(request.Prompt)
        .Select(p => new AgentRequest(p, request.Depth + 1));

    var results = await Task.WhenAll(subRequests.Select(HandleAsync));
    return Plan.Merge(results);
}
```

The depth cap is not a substitute for the circuit breaker. It is what keeps the breaker from being the *first* line of defense against a problem that has already propagated.

## Putting the three fixes together

A resilient agent handler applies all three fixes in a specific order. The order matters because each fix removes work from the layers below it.

1. **Check the cache first.** A cache hit never enters the policy wrap. This is the cheapest possible outcome and should be the most common one for repeated prompts.
2. **Enforce the recursion guard.** Reject or truncate before any downstream call is made. This bounds the fan-out at the source.
3. **Apply the semantic circuit breaker around the LLM call.** Count only tripping failures toward the threshold.
4. **Apply the bulkhead around the same call.** With state externalized, the bulkhead now bounds only concurrent LLM calls.

```csharp
app.MapPost("/agent/plan", async (
    [FromBody] string userPrompt,
    CacheService cache,
    IHttpClientFactory httpFactory,
    AsyncBulkheadPolicy<HttpResponseMessage> bulkhead,
    AsyncCircuitBreakerPolicy<HttpResponseMessage> breaker) =>
{
    var cacheKey = $"plan:{StableHash(userPrompt)}";
    var cached = await cache.GetAsync(cacheKey);
    if (cached is not null)
        return Results.Ok(cached);

    var http = httpFactory.CreateClient("llm");
    var policyWrap = Policy.WrapAsync(bulkhead, breaker);

    HttpResponseMessage response;
    try
    {
        response = await policyWrap.ExecuteAsync(
            () => CallLlmAsync(userPrompt, http));
    }
    catch (BrokenCircuitException)
    {
        return Results.StatusCode(503);
    }
    catch (BulkheadRejectedException)
    {
        return Results.StatusCode(429);
    }

    if (!response.IsSuccessStatusCode)
        return Results.StatusCode((int)response.StatusCode);

    var plan = await response.Content.ReadAsStringAsync();
    await cache.SetAsync(cacheKey, plan);
    return Results.Ok(plan);
});
```

Two things to note. First, `StableHash` must be a deterministic hash — `string.GetHashCode()` is randomized per process in modern .NET and will produce cache misses across restarts and across instances. Use a cryptographic hash or a stable non-cryptographic hash such as FNV-1a. Second, the `BrokenCircuitException` and `BulkheadRejectedException` are caught separately so the caller can distinguish "dependency is down" (503) from "we are overloaded" (429). Collapsing them into one status code makes debugging harder.

## How to measure whether any of this helps

The published benchmarks for policy libraries measure the libraries, not your system. The only measurement that matters is the one taken against your own workload. Here is what to instrument and what to compare.

**Instrument these four things:**

- **Circuit breaker state transitions**, with the triggering error class attached. If the breaker opens, you want to know whether it opened on a timeout, a 5xx, or a logical error that slipped through the predicate.
- **Bulkhead queue depth and rejection count.** Queue depth rising toward the cap is an early warning; rejections are the late warning.
- **Cache hit rate and cache latency.** A hit rate below roughly half usually means the key is unstable or the TTL is too short.
- **Tail latency of the original request**, not just the leaf call. This is the number the user experiences, and it is the one that recursive fan-out destroys.

**Then run two configurations against the same traffic:**

- **Baseline:** circuit breaker on status codes only, in-process cache, no recursion cap.
- **Enhanced:** semantic predicate, externalized cache, recursion cap with partial-result fallback.

Compare p50, p95, and p99 latency, breaker trips per hour, bulkhead rejections per hour, and node CPU. Run for long enough to cover a full traffic cycle — at least one weekday and one weekend for a consumer workload, or several business days for an internal tool. A short test will miss the burst patterns that trigger the failures you are trying to fix.

If the enhanced configuration does not reduce breaker trips, the predicate is probably still counting logical errors. If it does not reduce tail latency, the recursion cap is probably set too high to matter. If it does not reduce CPU, the cache is probably not being hit — check the key stability first.

## A decision checklist

Before adopting the standard circuit-breaker-and-bulkhead playbook for an agent system, answer these:

- Does the breaker predicate inspect the response body, or only the status code and latency?
- Is there a hop counter on the request envelope to catch agent-to-agent loops?
- Is any mutable state held inside the bulkhead's protected region?
- Is the cache key deterministic across processes and restarts?
- Is there a hard cap on recursion depth, and does the cap return a partial result rather than an error?
- Are cache hits short-circuited before the policy wrap, so they consume no bulkhead slot?
- Are "dependency down" and "we are overloaded" reported as distinct status codes?
- Is the tail latency of the *original* request measured, not just the leaf call?

A "no" on any of the first five means the system has a failure mode that the circuit breaker will not catch. A "no" on either of the last three means the failure, when it happens, will be harder to diagnose than it needs to be.

## FAQ

**Does a semantic circuit breaker require parsing every response body?**

No. In most systems, the tripping failures (timeouts, connection errors, 5xx) are identifiable before the body is read. Only successful responses need body inspection, and only to distinguish "expected empty result" from "unexpected payload." The cost is one deserialization per successful call, which is usually already happening.

**Can the recursion cap be replaced by a timeout?**

No. A timeout bounds how long a request runs, not how much work it spawns. A deep fan-out can consume downstream capacity for the full duration of the timeout and still return nothing useful. The cap bounds the work; the timeout bounds the wait. They solve different problems.

**Is an external cache always better than an in-process cache?**

No. An in-process cache is faster and has no network hop. It is the right choice when the working set is small, the process is long-lived, and the bulkhead is not the bottleneck. It becomes the wrong choice when cache churn consumes the same capacity the bulkhead is trying to ration. Measure the time spent on cache operations as a fraction of request time; if it is more than a few percent under load, externalize it.

**What if the LLM provider already has its own rate limiting?**

Provider-side rate limiting protects the provider, not your system. It surfaces as 429 responses, which your breaker should count as tripping failures — but by the time you see them, your bulkhead queue is already full. Client-side bulkheading is what keeps you from reaching that point.

## Take action in the next 30 minutes

Open the file that configures your circuit breaker and read the predicate. If it only checks status codes and latency, add a body inspection that classifies at least one logical error as non-tripping. Then add a hop counter to your request envelope and reject requests above a small fixed limit. Those two changes are a few lines each and close the two failure modes that a circuit breaker cannot see.
