# Node.js memory leak: find it in 10 min

A Node.js service runs for hours or days, RSS climbs, the container is OOM-killed, and the process restarts. Staging rarely reproduces it because staging does not carry the same traffic mix or stay up long enough. A single heap snapshot looks unremarkable — until two snapshots taken an hour apart show a constructor count that only goes up. The core problem is that a leak in a long-running service is a *rate*, not a state, and most teams look for it with the wrong instrument. This article is about matching the instrument to the failure mode before restarting the pod.

## Read the symptom before opening a profiler

Not every rising memory graph is a leak. Classify what you are seeing first; the three patterns below point to different causes and different fixes.

**Pattern A — sawtooth with a rising floor.** RSS climbs, GC brings it down partially, but the trough after each collection is higher than the last. This is the classic leak signature: something is retained that should be collectable, and the retained set grows monotonically.

**Pattern B — flat then cliff.** Memory is stable for hours, then jumps sharply and stays. This is usually a cache or buffer that crossed a size threshold, or a connection pool that grew in response to a burst. It may not be a leak at all — it may be an unbounded data structure that only reveals itself under a specific input.

**Pattern C — high but stable.** RSS sits at a level you dislike but does not grow. This is often V8 heap fragmentation or an external buffer (`Buffer`, `ArrayBuffer`, native addon) that the GC cannot see well. It is a tuning problem, not a leak.

A first check that costs nothing: log `process.memoryUsage()` on an interval (every 60 seconds is fine) and watch `heapUsed` versus `rss`.

```javascript
setInterval(() => {
  const m = process.memoryUsage();
  console.log(JSON.stringify({
    rss: Math.round(m.rss / 1024 / 1024),
    heapUsed: Math.round(m.heapUsed / 1024 / 1024),
    heapTotal: Math.round(m.heapTotal / 1024 / 1024),
    external: Math.round(m.external / 1024 / 1024),
  }));
}, 60_000).unref();
```

Run this for an hour under real traffic. The shape of those four numbers tells you which pattern you have before you spend time on a snapshot. If `heapUsed` is flat but `rss` climbs, the leak is outside the JS heap — native memory, Buffers, or a native addon. If `heapUsed` climbs, you are looking at retained JS objects and the heap snapshot is the right tool.

## The causes, in the order they commonly occur

### 1. Event listeners and subscriptions that outlive their owner

The most common leak in Node services is an `EventEmitter` listener added per request or per connection and never removed. Express middleware, WebSocket handlers, and message-queue consumers are the usual suspects. The symptom is a listener count that grows with request volume. You can confirm it without a profiler:

```javascript
// on the emitter in question, during an incident
console.log(emitter.listenerCount('data'));
```

If that number tracks your request count, you have found it. Node also emits a `MaxListenersExceededWarning` when a single emitter exceeds its limit (default 10), which is a strong hint even if the warning is buried in logs.

**Fix:** remove the listener in the same scope that added it, ideally in a `finally` block. For request-scoped work, prefer `once` over `on` when the handler should fire a single time. For long-lived subscriptions, tie the listener to an `AbortController` and remove it on abort.

### 2. Closures captured by timers and intervals

A `setInterval` that closes over a request object keeps that object alive for as long as the interval runs. If the interval is never cleared, every request that touched it is retained. This is subtle because the interval itself is small — it is the captured scope that is large.

**Confirm:** search for `setInterval` and check whether each one has a matching `clearInterval` on a shutdown path. In a heap snapshot, look for `Timeout` objects whose retained size is dominated by a closure scope you recognize.

**Fix:** clear intervals on shutdown and on the event that makes them irrelevant. `timer.unref()` stops the timer from keeping the process alive, but it does not clear the interval and does not release the captured scope.

### 3. Unbounded caches and maps

An in-memory cache is a leak with good intentions. A `Map` used as a cache grows until something evicts from it, and if nothing does, it grows until the process dies. The symptom is Pattern B: flat, then a cliff as the map crosses a threshold.

**Confirm:** log `map.size` on an interval. If it only goes up, you have your answer.

**Fix:** use an LRU with a maximum size, or a TTL cache with active expiry. Without a dependency, a `Map` plus a periodic sweep that deletes entries older than a timestamp is enough — but write the sweep, do not assume it exists.

### 4. Promises and async contexts that never settle

A promise that never resolves retains its closure and everything reachable from it. This shows up as a slow leak in services that do fan-out requests with no timeout. The fix is to put a timeout on every outbound call and to make sure the timeout path actually rejects rather than silently dropping.

**Confirm:** in a heap snapshot, look for a large number of `Promise` objects with retained sizes pointing at request payloads. If the count grows with uptime, this is likely.

**Fix:** wrap outbound calls with `AbortSignal.timeout(ms)` where the client supports it, and ensure the rejection is handled. An unhandled rejection swallowed by an empty `.catch(() => {})` is worse than a crash — it hides the leak.

### 5. Native memory and Buffers

Buffers allocated outside the V8 heap do not show up in `heapUsed`. If `rss` climbs while `heapUsed` is flat, look at `external` and `arrayBuffers` in `process.memoryUsage()`. Common causes are pooled Buffers that are never released, native addons with their own allocators, and stream pipelines where the readable side is paused and the writable side keeps buffering.

**Confirm:** compare `external` growth to `rss` growth. If they track each other, the leak is in Buffer or native territory.

**Fix:** for streams, respect backpressure — do not `pipe` into a slow consumer without checking `write()` return values, and do not buffer an entire response in memory when you can stream it.

## Taking a heap snapshot that is actually useful

A single snapshot tells you what is in memory. Two snapshots tell you what is *accumulating*. The difference is where the leak lives.

Take the first snapshot after the service has warmed up (for example, 10 minutes after deploy), then take a second one an hour later under similar load. In Chrome DevTools, load both and use the "Comparison" view. Sort by "# Delta" — the constructors with the largest positive delta are your candidates. Then expand the retainer paths for the top few and look for the frame in *your* code, not in a library.

On a server without a GUI, write snapshots to disk and pull them locally:

```javascript
const v8 = require('node:v8');

function snapshot(label) {
  const file = `/tmp/heap-${label}-${Date.now()}.heapsnapshot`;
  v8.writeHeapSnapshot(file);
  console.log('wrote', file);
}

// call snapshot('warm') once, snapshot('later') an hour later
```

Two practical warnings. First, `writeHeapSnapshot` blocks the event loop for the duration of the write — on a large heap this can be seconds. Do it on a canary instance, not on the whole fleet. Second, the snapshot file is roughly the size of the heap, so make sure `/tmp` has room and that old files are deleted.

The retainer path is the important part. It answers "why is this object still alive?" — and the answer is usually a global, a module-level variable, or a closure held by a long-lived object. If the retainer path runs through a library you do not control, the fix is usually to stop giving that library new objects, not to patch the library.

## When the heap looks fine but the container still dies

If `heapUsed` is stable and `rss` still climbs, you are outside V8. Three things to check, in order:

1. **Buffers and streams.** Log `external` and `arrayBuffers`. Growing `arrayBuffers` with a flat heap points at stream backpressure or a Buffer pool that is not being released.
2. **Native addons.** If you use a database driver or image library with a native component, its allocations are invisible to V8. Check the addon's issue tracker for known leaks and pin to a version that does not have them.
3. **V8 heap fragmentation.** If `heapUsed` is well below `heapTotal` but `rss` is high, the heap has fragmented. This is a tuning problem: adjusting `--max-old-space-size` or the semi-space size can help, but the real fix is usually to reduce allocation churn in hot paths.

A container memory limit set close to the heap limit is a common configuration mistake. If the container limit is 512 MB and `--max-old-space-size` is 400, that leaves 112 MB for the stack, native memory, and Buffers — which may not be enough. Leave headroom, and set the container limit above the sum of what the process is expected to use at peak.

## Prevention: make leaks visible before they page you

Three practices that cost little and catch most of the problems above.

**Log memory on a schedule and alert on slope, not level.** A service at 400 MB that has been flat for a week is fine. One at 200 MB that has climbed 20 MB a day is not. Alert on the derivative. Instrument by storing the `process.memoryUsage()` samples and computing a linear fit over the last 24 hours; alert when the slope exceeds a threshold derived from a known-good baseline rather than a number borrowed from an article.

**Cap every in-memory structure.** If you write `new Map()` in a service, the next line should be a comment explaining what bounds it. If there is no bound, it is a leak waiting for traffic.

**Test with a soak, not a spike.** Load tests that run for five minutes will not reveal a leak that needs an hour. A nightly soak against a staging environment with production-like traffic is the cheapest leak detector you can build.

A minimal soak check that can be added to CI:

```javascript
// after a 30-minute soak against staging, assert heap growth is bounded
const before = await getHeapUsed();
await runSoak({ durationMs: 30 * 60 * 1000 });
const after = await getHeapUsed();
const growthMb = (after - before) / 1024 / 1024;
if (growthMb > 50) {
  throw new Error(`heap grew ${growthMb.toFixed(1)} MB during soak`);
}
```

The threshold is illustrative — pick one based on the service's normal allocation pattern, not on a number from a blog post. To calibrate: run the soak three times against a known-good build, record the maximum observed growth, and set the threshold above it with margin.

## Worked example: classifying a leak from four numbers

Suppose a service logs the following samples (values are illustrative, rounded to MB):

| Minute | rss | heapUsed | heapTotal | external |
|---|---|---|---|---|
| 10 | 210 | 60 | 90 | 20 |
| 70 | 260 | 62 | 92 | 21 |
| 130 | 310 | 61 | 91 | 22 |
| 190 | 360 | 63 | 93 | 21 |

Reasoning step by step:

1. `heapUsed` moves between 60 and 63 MB across three hours. That is noise, not a trend. The JS heap is not the problem.
2. `rss` rises 210 → 360 MB, a 150 MB increase over 180 minutes, about 0.83 MB per minute. The growth is real and roughly linear.
3. `external` is flat at about 21 MB, so the growth is not in tracked Buffers either. That leaves native allocations not counted in `external` (some native addons and allocators), or fragmentation, or memory the OS attributes to the process for other reasons.
4. Next step is not a heap snapshot — it would show a flat heap and waste an hour. Instead, isolate: run the same build with the native addon disabled (or swap it for a pure-JS equivalent) and repeat the sample. If `rss` flattens, the addon is the source. If it still climbs, check for fragmentation by comparing `heapUsed` to `heapTotal` and by observing whether RSS drops after a forced GC (run the process with `--expose-gc` on a canary and call `global.gc()`).

The same table with `heapUsed` rising instead would send you straight to two snapshots and the delta view. The point of classification is to avoid spending the expensive instrument on the wrong target.

## Which tool for which budget

Heap snapshots and `process.memoryUsage()` are built in and free. Beyond that, the choice depends on what you are running.

| Situation | Reasonable choice | Notes |
|---|---|---|
| Single Node process, bootstrapped | `node --inspect` + Chrome DevTools, `v8.writeHeapSnapshot` | No cost, no agent, works on a small VM |
| Container fleet on a cloud provider | The provider's managed profiling service | Useful when you cannot SSH into every pod |
| Multi-language backend | An APM with continuous profiling | The value is correlation with traces, not the profiler alone |
| Existing enterprise contract | Whatever is already in the contract | Adding a vendor for memory profiling alone is rarely worth a procurement cycle |

The trap is buying an APM to solve a leak that has not been classified. A continuous profiler shows *where* CPU and allocation go; it does not tell you what is retained. For retention, a heap snapshot is required. Use the paid tool to find the hot path, then take a snapshot to find the retainer.

## Decision checklist

Before starting a debugging session, answer these in order:

- Is `heapUsed` trending up, or only `rss`? (Determines JS heap vs native.)
- Is the growth linear, stepped, or flat-then-cliff? (Pattern A, B, or C.)
- Does the growth correlate with request volume, uptime, or a specific input? (Narrows the suspect.)
- Which in-memory structures have no documented bound? (Candidates for Pattern B.)
- Can the leak be reproduced on a single canary with production-like traffic? (If not, fix observability first.)
- Is there a snapshot from before the last restart? (If not, capture one before the next.)

## Frequently Asked Questions

**Why does Node.js memory keep growing but heapUsed is stable?**

Because the growth is outside the V8 heap. `rss` includes native memory, Buffers, and the code itself. Check `external` and `arrayBuffers` in `process.memoryUsage()` — if they are growing, the leak is in Buffer or native territory, and a JS heap snapshot will not show it. The fix is usually in stream handling or a native addon, not in application objects.

**How many heap snapshots are needed to find a leak?**

Two, taken under similar load at different times. A single snapshot shows what is in memory; the comparison between two shows what is accumulating. Take the first after warm-up and the second an hour or more later, then sort by delta in DevTools. If the delta is small, a longer interval or heavier traffic may be needed to make the leak visible.

**What causes MaxListenersExceededWarning in production?**

A single `EventEmitter` has more than its configured limit of listeners (default 10). In a service, this almost always means a listener is being added per request or per connection without a matching removal. Check `emitter.listenerCount(eventName)` — if it tracks request volume, the leak is found. The fix is to remove the listener in the same scope that added it.

**When should the pod be restarted instead of debugged?**

When the service is degraded and the leak is not yet understood, restart to restore service — but capture `process.memoryUsage()` and, if possible, a heap snapshot before the restart. A restart without a snapshot destroys the evidence. A scheduled restart that masks a leak is a signal to schedule the debugging, not a fix.

## The next 30 minutes

Add the `process.memoryUsage()` interval logger shown above to the service's main entry file, deploy it to one canary instance, and let it run for an hour under real traffic. When you return, compare `heapUsed` and `rss`: that single comparison tells you whether you are hunting a JS retention leak or a native-memory leak, and which section above to read first.
