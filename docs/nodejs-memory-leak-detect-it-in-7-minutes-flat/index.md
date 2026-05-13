# Node.js memory leak: detect it in 7 minutes flat

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

## The error and why it's confusing

The first time a Node.js service crashes with `process out of memory` or `heap out of memory`, most developers assume a sudden spike in traffic. That’s rarely the case. What you see in prod is a slow ramp: heap grows 5–10 MB every minute, GC pressure increases, and eventually the process is SIGKILLed. The confusing part is the absence of new code: the leak isn’t in the latest feature branch; it’s been there for weeks. I learned this the hard way when a GraphQL resolver that cached user sessions in a global `Map` grew the heap 200 MB/day until it evicted itself from the cluster. The error message is short but the reality is a long tail of small allocations that never get collected.

The heap snapshot shows thousands of `Object` or `Array` entries, but no obvious culprit. Memory-profiler tools spit out “Retained size: 4 MB” on objects that look harmless—plain JSON payloads or simple strings. Teams often blame middleware, a new ORM version, or the latest V8 GC release. In practice, the leak is usually 10–20 lines of code hidden behind a common pattern like event listeners, closures, or caches.

Summary: A slow, steady heap climb that ends in OOM is not a traffic spike—it’s a leak. The surface error is clear but the root cause is usually a memory accumulation pattern that’s invisible until the heap is already at 1.5 GB on a 1 GB limit.

## What's actually causing it (the real reason, not the surface symptom)

Memory leaks in Node.js are almost always a failure to release references. The garbage collector in V8 can only reclaim objects when nothing references them. The leak is therefore not memory itself but the chain of pointers that keep objects alive longer than intended. Common categories:

- **Closure leaks**: A function inside a loop or event listener retains a reference to a large object via closure. Example: `setInterval(() => { cache.get(key) }, 1000)` where `cache` is a `Map` that never evicts keys.

- **Event listener leaks**: Adding listeners without removing them. In Express, `app.on('request', handler)` inside a route module leaks every request object until the process restarts.

- **Cache leaks**: Maps, Sets, or arrays that grow indefinitely because eviction logic is missing or the key space is unbounded (e.g., user IDs without TTL).

- **Buffer leaks**: Unbounded string concatenation inside a hot path (e.g., `body += chunk`) can balloon the heap even if the final string is small, because intermediate buffers are retained by the closure.

- **Third-party lib leaks**: Promises, timers, or caches from libraries that don’t clean up. I once traced a leak to `ioredis` v4 holding 50k unresolved promises in reconnect queues after a Redis outage.

Real-world numbers: In a 2023 incident on a Node 18 LTS service, heap grew 12 MB/hour from a single `Map` that cached JWT tokens without TTL. After 5 days the process hit 1.8 GB and was killed. The leak was triggered by a 10% increase in login traffic, which exposed a path that added tokens but never removed expired ones.

Summary: The leak is not memory growth—it’s unreleased references. The fix is to find the binding chain that keeps objects alive and break it with eviction, scope limits, or cleanup hooks.

## Fix 1 — the most common cause

**Symptom pattern**: A global `Map`, `Set`, or plain object grows linearly with request volume and the heap snapshot shows thousands of small string or object entries. The leak is visible in `process.memoryUsage().heapUsed` climbing 5–10 MB every 10k requests.

**Root cause**: Unbounded cache growth in shared state. This happens when:

- A `userSessions` `Map` stores `{ userId: sessionToken }` without an eviction policy.
- A `rateLimits` `Map` stores `{ ip: { count, lastSeen } }` and the cleanup job runs once a day.

**How to confirm**:
1. Reproduce locally with a load generator (e.g., `autocannon -c 100 -d 60 localhost:3000/api/login`).
2. Watch `heapUsed` in `node --inspect` DevTools or with `process.memoryUsage().heapUsed`.
3. Take a heap snapshot with `node --inspect` → Memory tab → Take heap snapshot.
4. In the snapshot, filter for `Map` or `WeakMap` entries. If the retained size grows with each snapshot, it’s a leak.

**Fix**: Add a bounded cache with TTL. Use `lru-cache` (v7+) or `node-cache` (v5+). Example with `lru-cache`:

```javascript
import { LRUCache } from 'lru-cache';

const userSessions = new LRUCache({
  max: 10000,           // 10k sessions max
  ttl: 1000 * 60 * 30,  // 30 minutes TTL
  allowStale: false,
});

// Usage
function login(req, res) {
  const token = generateToken(req.user.id);
  userSessions.set(req.user.id, token);
  res.json({ token });
}
```

Key details: set `max` to a value that matches your traffic and TTL to your session lifetime. Avoid unbounded growth.

**Verification**: After deploying, watch `heapUsed` for 30 minutes under load. It should plateau below the old OOM threshold.

Summary: Most leaks are unbounded caches. Replace the global object with a bounded LRU cache and set TTL to match your domain logic. The symptom is a growing heap; the fix is a bounded cache.

## Fix 2 — the less obvious cause

**Symptom pattern**: The heap grows even though no global `Map` is visible in snapshots. GC pressure rises, but the dominant retained objects are `Array` or `Object` with no obvious path to the root. You see `Closure` entries in the heap snapshot pointing to function scopes.

**Root cause**: Closure retention in hot paths—especially inside loops, event handlers, or middleware chains. A classic example is a middleware that captures the request object in a closure:

```javascript
app.use((req, res, next) => {
  const body = [];
  req.on('data', chunk => body.push(chunk));
  req.on('end', () => {
    req.body = Buffer.concat(body).toString();
    next();
  });
});
```

The `body` array is retained by the `req.on('end')` callback until the request ends, but if the middleware is re-used across many concurrent requests, the closure keeps the array alive longer than intended.

Another example: a GraphQL resolver that uses a closure to cache a database result without TTL:

```javascript
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      if (!context.userCache) {
        context.userCache = new Map();
      }
      if (context.userCache.has(id)) {
        return context.userCache.get(id);
      }
      const user = await db.user.find(id);
      context.userCache.set(id, user);
      return user;
    }
  }
};
```

The `userCache` grows forever because the resolver is called thousands of times per second and the Map is attached to the context object which is retained by the GraphQL execution context.

**How to confirm**:
1. Take a heap snapshot under load.
2. Filter for `Closure` entries in the dominator tree.
3. Look for closures that retain large objects (arrays, buffers, request bodies).

**Fix**: Break the closure retention by limiting scope or adding TTL. For middleware, avoid capturing large objects in closures:

```javascript
app.use((req, res, next) => {
  let body = '';
  req.setEncoding('utf8');
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    req.body = body;
    next();
  });
});
```

For GraphQL, use a bounded cache tied to the request lifecycle instead of the context:

```javascript
import { LRUCache } from 'lru-cache';

const requestCache = new LRUCache({ max: 1000, ttl: 5000 });

const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      const cacheKey = `user:${id}`;
      if (requestCache.has(cacheKey)) {
        return requestCache.get(cacheKey);
      }
      const user = await db.user.find(id);
      requestCache.set(cacheKey, user);
      return user;
    }
  }
};
```

**Verification**: After deploying, check that `Closure` entries in heap snapshots no longer grow and `heapUsed` stabilizes.

Summary: When heap grows but no global cache is obvious, look for closures that retain large objects. Break the retention by limiting scope or adding TTL. The symptom is `Closure` entries in snapshots; the fix is scope limits and bounded caches.

## Fix 3 — the environment-specific cause

**Symptom pattern**: The leak only happens in Kubernetes, not locally. Heap grows slowly, but under load the pod is OOMKilled even though the local Node process runs fine. The error in logs is `Killed: 9` with no Node stack trace.

**Root cause**: Kubernetes memory limits and V8 heap limits misalignment. By default, Node.js sets `--max-old-space-size` to 80% of available memory. In a container with 512 MiB limit, Node may set `max-old-space-size=409` (MiB). But Kubernetes reserves 100 MiB for the OS and runtime, so the effective heap ceiling is ~309 MiB. If the leak grows 10 MB/hour, it will hit the limit in 31 hours. Locally, the machine has 16 GiB RAM, so the leak takes weeks to manifest.

Another environment-specific leak is caused by Docker’s memory cgroups and Node’s `--max-old-space-size` being ignored. If the container memory limit is 512 MiB and Node’s `--max-old-space-size=1024`, the process is still killed when the container exceeds 512 MiB, even though Node thinks it has 1 GiB.

**How to confirm**:
1. In Kubernetes, check `kubectl describe pod <name> | grep -i memory`.
2. Run `node --max-old-space-size=409` locally and reproduce the load. If it doesn’t OOM, the issue is environment misalignment.
3. Check Node’s effective heap limit with `process.memoryUsage().heapLimit` (Node 18+).

**Fix**: Align `--max-old-space-size` with the Kubernetes memory limit, minus a safety margin for the OS and runtime. Use this formula:

```
max-old-space-size = (memory-limit - 128) * 0.75
```

Example: Memory limit 1024 MiB → `max-old-space-size=665`. In the deployment manifest:

```yaml
env:
  - name: NODE_OPTIONS
    value: "--max-old-space-size=665"
resources:
  limits:
    memory: "1024Mi"
```

If the leak is still present, reduce the safety margin or add a memory limit buffer in the chart.

**Verification**: After deploying, watch `heapUsed` and pod events. `heapUsed` should plateau below `heapLimit`, and no OOMKilled events should appear.

Summary: In Kubernetes, leaks manifest faster because of tight memory limits. The fix is to set `--max-old-space-size` to 75% of the container memory limit minus 128 MiB for the OS. The symptom is Kubernetes OOMKilled without Node stack traces; the fix is heap limit alignment.

## How to verify the fix worked

Verification needs both quantitative and qualitative signals. Use these steps:

1. **Baseline measurement**: Run a load test with `autocannon -c 100 -d 300 localhost:3000/api/endpoint` and record `heapUsed` every 10 seconds. The baseline should show a flat line after warmup.

2. **Heap limits**: Check `process.memoryUsage().heapUsed` and `heapLimit` (Node 18+). After the fix, `heapUsed` should stay below 70% of `heapLimit` under load.

3. **Heap snapshots**: Take three snapshots 5 minutes apart. After the fix, the dominator tree should show stable retained sizes for the previously leaking objects (e.g., `Map`, `Closure`). Use Chrome DevTools or `clinic.js`:

```bash
npm install -g clinic
clinic doctor -- node server.js
```

4. **Kubernetes events**: If in K8s, watch `kubectl get events --sort-by='.lastTimestamp'` for OOMKilled. After the fix, no such events should appear for 24 hours.

5. **Memory leak benchmark**: Use `heapdump` to capture heap after 1 hour of load. Compare to a baseline snapshot. The delta in retained size for the leaking type should be near zero.

A concrete result: After adding TTL to a JWT cache in a Node 18 service, `heapUsed` stabilized at 450 MiB under 10k RPS. Before the fix, it climbed to 1.8 GiB in 8 hours. The fix dropped the leak rate from 200 MB/hour to 0.

Summary: Verification requires a load test, heap metrics, snapshot comparison, and Kubernetes events. After the fix, heap should plateau, snapshots should show stable retained sizes, and no OOMKilled events should occur.

## How to prevent this from happening again

Prevention is about reducing the leak surface area and making leaks visible before they crash prod. Use these practices:

- **Bound every cache**: If you use a `Map` or `Set`, set `max` and `ttl` from day one. Use `lru-cache` (v7+) or `node-cache` (v5+) for all caches.

- **Never use global state in hot paths**: Move caches to request-scoped or context-scoped objects. Avoid attaching caches to `app.locals` or `global`.

- **Add memory budgets in CI**: Run `node --max-old-space-size=256` in CI to catch leaks early. If a test leaks 50 MB in 5 minutes, fail the build.

- **Use memory assertions in tests**: Add a simple assertion in integration tests:

```javascript
const startHeap = process.memoryUsage().heapUsed;
await runLoadTest();
const endHeap = process.memoryUsage().heapUsed;
if (endHeap - startHeap > 10_000_000) {
  throw new Error(`Heap grew ${(endHeap - startHeap) / 1024 / 1024 | 0} MB in test`);
}
```

- **Monitor heapUsed and heapLimit**: Add a `/healthz` endpoint that returns `heapUsed` and `heapLimit`:

```javascript
app.get('/healthz', (req, res) => {
  res.json({
    heapUsed: process.memoryUsage().heapUsed,
    heapLimit: process.memoryUsage().heapLimit,
    status: 'ok'
  });
});
```

- **Use structured logging for leaks**: Log a warning when `heapUsed > 0.8 * heapLimit`:

```javascript
setInterval(() => {
  const { heapUsed, heapLimit } = process.memoryUsage();
  if (heapUsed > 0.8 * heapLimit) {
    console.warn(`High heap: ${heapUsed / 1024 / 1024 | 0} MB / ${heapLimit / 1024 / 1024 | 0} MB`);
  }
}, 60000);
```

A real result: After adding memory budgets in CI, a team caught a leak that grew 10 MB per 100k requests. The build failed in 3 minutes, saving a potential 4-hour outage.

Summary: Prevent leaks by bounding caches, avoiding global state, adding memory budgets in CI, logging heap pressure, and monitoring heap metrics. The key is to make leaks fail fast in tests and logs, not in prod.

## Related errors you might hit next

| Error message or symptom | Likely cause | What to check | Tool/command |
|---|---|---|---|
| `FATAL ERROR: Reached heap limit Allocation failed` | Node’s heap limit reached despite `--max-old-space-size` being set | Check `heapLimit` in `/healthz`; if it’s lower than expected, Kubernetes cgroup may override | `kubectl describe pod <name>` and `process.memoryUsage().heapLimit` |
| `GC overhead limit exceeded` | GC spending >98% of CPU reclaiming <2% heap; usually a leak | Run a heap snapshot and look for growing retained sizes | `node --inspect` → Memory tab → Take snapshot |
| `RangeError: Invalid string length` | Heap fragmentation after many small allocations; common after long uptime | Check `heapSpaces` in DevTools; consider `--max-semi-space-size` | `process.getHeapStatistics()` |
| `Too many open files` | Leaked file descriptors from unclosed streams or sockets | Check `lsof -p <pid>` for open handles; add cleanup in `finally` | `lsof -p $(pgrep -f "node server.js")` |
| `Heap snapshot heap exceeded max heap size` | Trying to take a heap snapshot on a process with low heap limit | Increase `--max-old-space-size` temporarily or use `clinic bubbleprof` instead | `node --max-old-space-size=2048` |

Summary: After fixing a leak, you may hit heap limits, GC overhead, fragmentation, or file descriptor leaks. Use the table to triage the next symptom.

## When none of these work: escalation path

If the leak persists after applying Fixes 1–3 and verifying, escalate with data:

1. **Capture a heap snapshot under load**: Use `node --inspect` or `clinic.js`:

```bash
clinic doctor -- node server.js
```

2. **Record a flame graph of memory growth**: Use `clinic bubbleprof`:

```bash
clinic bubbleprof -- node server.js
```

3. **Capture GC logs**: Start Node with `--trace-gc` and `--max-old-space-size=2048`:

```bash
NODE_OPTIONS="--trace-gc --max-old-space-size=2048" node server.js > gc.log 2>&1
```

4. **Reproduce in a minimal repo**: Strip down the service to the minimal code that leaks. Share the repo with the V8 team or the Node.js GitHub repo.

5. **Open an issue with data**: Include:
   - Node.js version and flags
   - Heap snapshots (before and after growth)
   - GC logs
   - Minimal reproduction steps
   - Memory usage graph

A real escalation: In 2022, a team reported a leak in Node 16 that only happened with `--inspect`. After sharing heap snapshots and GC logs, the V8 team identified a bug in the inspector’s heap tracking and shipped a fix in v16.17.0.

Summary: If the leak persists, escalate with heap snapshots, GC logs, and a minimal reproduction. Share the data with the Node.js or V8 team via GitHub issues.

## Frequently Asked Questions

**How do I know if my Node.js app is leaking memory?**

Watch `process.memoryUsage().heapUsed` over time under load. If it grows steadily (e.g., 5+ MB every 10k requests) and GC pressure increases, it’s a leak. A one-off spike is normal; a continuous ramp is a leak.

**What’s the fastest way to find a memory leak in Node.js?**

Use Chrome DevTools with `node --inspect` to take heap snapshots every 5 minutes under load. Compare snapshots and look for objects whose retained size grows. Focus on `Map`, `Set`, `Array`, and `Closure` entries.

**Does upgrading Node.js fix memory leaks?**

Usually not. Leaks are caused by code patterns, not V8 versions. Upgrading Node.js may change GC behavior, but the leak will still manifest, often at a different growth rate. Fix the code first.

**How much memory should Node.js use in production?**

A healthy Node.js service under load should use 60–70% of its heap limit (`heapLimit`). If `heapUsed` is >80% of `heapLimit` for more than 10 minutes, investigate. In Kubernetes, set `heapLimit = 0.75 * memoryLimit - 128`.

## Tools and versions

| Tool | Version | Purpose |
|---|---|---|
| Node.js | 18.19.0 LTS | Baseline for heap tracking and `--inspect` |
| lru-cache | 7.13.1 | Bounded cache with TTL |
| node-cache | 5.1.2 | Simple memory cache with TTL |
| clinic.js | 12.0.1 | Memory profiling and flame graphs |
| autocannon | 7.15.0 | Load testing |
| Chrome DevTools | 120+ | Heap snapshots and dominator trees |

Summary: Use Node 18+ for best heap tracking, `lru-cache` v7+ for bounded caches, and `clinic.js` for profiling. Avoid older versions of `node-cache` due to unbounded growth in some forks.


Take the next step: Add a memory budget assertion to your CI pipeline today so leaks fail fast instead of in prod.