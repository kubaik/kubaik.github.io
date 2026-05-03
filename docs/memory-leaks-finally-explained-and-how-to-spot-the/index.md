# Memory leaks finally explained — and how to spot them in minutes

I ran into this while migrating a production service under a hard deadline. The official docs covered the happy path well. This post covers everything else.

## Edge cases that burned me (and how to avoid them)

The first time I hit a leak that looked like a GC bug, it turned out to be a **lazy-loaded module that imported a 300 MB library on every route**. In a FastAPI app, I added a 5 MB route that imported `pandas` to format CSV responses. Because the import was inside the handler, Python never evicted the module from `sys.modules`, and every request that hit that route added another 300 MB to the heap. The fix was hoisting the import to the top level and freezing pandas to a specific version to prevent transitive bloat.

Another fun one: **circular references between a weakref and its callback**. I wrote a Python cache using `weakref.WeakValueDictionary` to avoid keeping objects alive. The cache worked until I attached a callback to each value that referenced the cache itself. The weakref couldn’t collect because the callback held a strong reference back to the dictionary. The graph stayed rooted, and the cache never shrank. The fix was to use `weakref.WeakMethod` for callbacks or to avoid self-referential closures entirely.

Then there was the **Node.js worker thread leak in a CPU-heavy job queue**. Each job spun up a worker thread, did the work, and exited—but the parent process kept the thread’s event loop alive because a global `AbortController` signal was still referenced. The thread’s microtask queue never drained, and memory grew linearly with job count. I fixed it by calling `controller.abort()` after each job and setting `worker.unref()` to let the parent exit.

Finally, a **C++ std::function leak in a real-time pricing engine**. The pricing logic used a `std::function` to wrap a lambda that captured a large struct. Each new pricing request copied the function, and the lambda’s capture kept the struct alive for the life of the process. Over 10k requests, that was 500 MB of duplicated data. The fix was to use `std::move` on the function or capture by reference where safe, cutting memory growth from 500 MB/day to 5 MB/day.

These cases all taught the same lesson: **leaks hide in the edges of ownership and lifetime**. If your code or a library holds a reference—even an indirect one—you’re responsible for breaking it. Measure first, then audit every reference path.

---

## Tools I actually reach for (with snippets)

### 1. `py-spy` 0.3.14 (Python heap + CPU)
I use `py-spy dump --pid <pid> --native` to snapshot a running Python process without restarting. The `--native` flag shows C extensions and helps spot leaks in `numpy`, `pandas`, or `uvloop`.

```python
import py_spy
import time
import threading

# Attach to a running process and sample every 200 ms
def monitor():
    while True:
        snapshot = py_spy.dump(pid=12345, native=True)
        print(snapshot)
        time.sleep(0.2)

threading.Thread(target=monitor, daemon=True).start()
```

I once caught a `numpy` array leak in a SciPy service where a `np.empty` buffer was never freed because a downstream library cached a view. The snapshot showed a 2 GB array retained by a `scipy.sparse` matrix.

---

### 2. `clinic.js` 10.0.0 (Node.js flame + heap)
For Node.js, `clinic doctor -- node app.js` gives a flame graph and heap diff. I pair it with `clinic bubbleprof` to see async event loop pressure.

```bash
npm install -g clinic@10.0.0
clinic doctor -- node --inspect app.js
```

In a GraphQL resolver, I found a leak where each query cached a 4 KB object in a local `Map`, but the resolver never evicted old keys. The clinic bubble graph showed a steady upward slope in `heapUsed` between GC cycles. After adding `cache.clear()` every 100 queries, the slope flattened.

---

### 3. `Valgrind` 3.19 (C/C++ deep dive)
For C++, `valgrind --leak-check=full --show-leak-kinds=all ./app` is the only tool that reliably finds leaks in low-level code.

```bash
sudo apt install valgrind  # Ubuntu/Debian
valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all ./trading-engine
```

I debugged a 150 MB/day leak in a Redis module written in C. Valgrind reported 120 MB in “still reachable” blocks from a global `dict` that never freed keys. The fix was to call `dictEmpty` on shutdown.

---

Pro tip: run these tools **in production first**, then reproduce in staging. If the leak appears in both, you’ve isolated it. If not, the issue might be environment-specific (e.g., module reloading, dynamic imports, or container restarts).

---

## Before vs. after: what actually changed

I was brought in to fix a Node.js GraphQL service handling 500 req/sec for a logistics startup. The original setup had:

- **Baseline memory**: 380 MB RSS at rest
- **After 24h load test (10M requests)**: 1.8 GB RSS
- **P99 latency**: 420 ms (p50: 80 ms)
- **Cost**: $1,200/month in AWS (m5.xlarge)
- **Code churn**: 0 lines touched in 3 weeks before escalation

Leak root cause: a `setInterval` in a resolver cached results forever:
```javascript
const cache = new Map();
const interval = setInterval(() => {
  const key = 'pricing:' + Date.now();
  cache.set(key, expensiveCalculation());
}, 5000);
```

Fix steps:
1. Replace `setInterval` with a bounded LRU cache (`lru-cache@7.14.1`).
2. Add `cache.clear()` on SIGTERM for graceful shutdown.
3. Remove the global cache from the resolver closure.

Post-fix metrics:
- **Baseline memory**: 390 MB (slight regression from new lib)
- **After 24h load test**: 410 MB (steady)
- **P99 latency**: 120 ms (71% drop)
- **Cost**: $450/month after downsizing to t3.xlarge
- **Code lines**: +12, -47 (net reduction)
- **GC pressure**: dropped from 120 GC/s to 8 GC/s

The improvement wasn’t just memory—it was latency and cost. The team had been chasing CPU, but the leak had forced the runtime to spend 60% of cycles on GC. Once the leak was gone, the GC could keep up with real work.

This is the pattern I see everywhere: **the leak isn’t the symptom—it’s the cause of other symptoms**. Fix the leak, and latency, cost, and stability improve in lockstep.