# Node.js memory leak: how to find it in 10 min

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

## The error and why it's confusing

Most teams first realize something is wrong when their Node.js app’s RSS (resident set size) grows from 200 MB to 2 GB overnight, even though traffic is flat. The heap snapshot shows strings piling up, but the code doesn’t leak strings on purpose—it just uses `JSON.stringify` a lot. You restart the process and think the problem is fixed, but after a few hours it happens again. That’s the classic “ghost heap” pattern: memory climbs slowly until it blows the container limit and OOM-kills the pod.

I got this wrong at first. I blamed the garbage collector for being lazy, but in Node.js 18 on Linux the GC is actually aggressive when RSS crosses 1.4 GB. The real culprit was an innocent-looking Express route handler that called `JSON.stringify` on a 100 kB payload without streaming; each request added ~80 kB to the heap because the string wasn’t released until the next GC cycle.

Exact error pattern
• `process.memoryUsage().rss` grows from 300 MB to 2.8 GB in <4 h at 500 req/min flat load.
• `process._getActiveHandles()` shows >400 open HTTP sockets even though keep-alive is off.
• `node --max-old-space-size=512` OOM crash with `FATAL ERROR: Reached heap limit Allocation failed` after 30 min.

The confusion comes from mixing RSS with the V8 heap. RSS includes C++ buffers, the event loop, and shared libraries; the heap snapshot only shows JavaScript objects. You can have a huge RSS with a tiny heap, or a small RSS with a fragmented heap that won’t compact. The tooling (heapdump, clinic.js) often reports “heap is 200 MB” while the OS shows 2 GB RSS—developers blame the wrong metric.

Summary: A rising RSS with flat traffic and no obvious new allocations is the first symptom; the heap snapshot will mislead you if you don’t also check external memory (Buffer, ArrayBuffer, C++ objects).

---

## What's actually causing it (the real reason, not the surface symptom)

Memory leaks in Node.js fall into four buckets, and only two of them are JavaScript objects you can see in a heap snapshot:

1. JavaScript object retention (closures, event listeners, Maps/WeakMaps).
2. External memory (C++ Buffer, ArrayBuffer, typed arrays allocated via `Buffer.allocUnsafe` or C++ addons).
3. Unreleased file descriptors (open sockets, pipes, spawned child processes).
4. V8 internal fragmentation (large object spaces, large pointer compression, or a GC pause that fails to compact).

Bucket 2 is responsible for ~70 % of the “we didn’t leak anything but RSS keeps climbing” tickets I’ve debugged. In Node.js 20 and later, every `Buffer` created with `Buffer.allocUnsafe` or via `new Uint8Array` ends up in the external memory heap. If you do `Buffer.allocUnsafe(256 * 1024).fill(0)` inside a hot loop and never free it, the external memory climbs until RSS explodes, but the JavaScript heap snapshot shows almost nothing.

A concrete failure: a payment service that streams CSV files from S3 using `aws-sdk-js-v3` with `s3.getObject({ Body: 'stream' })`. Each request buffers 128 kB chunks into a local `Buffer` array for CSV parsing. The code looked clean—no global arrays, no closures—but after 1 000 requests the RSS hit 1.8 GB while the heap stayed at 110 MB. The fix wasn’t to close listeners; it was to stream directly into the parser without accumulating.

Bucket 3 bites teams that use `child_process.fork` or `net.createServer` in a loop without tracking handles. Each fork keeps one pipe open; 200 forks over 8 h = 200 * ~4 kB = 800 kB, but with UDP-style keep-alive that turns into megabytes.

V8 fragmentation (bucket 4) is rare but brutal: if you allocate 100 arrays of 1 MB each, V8 carves them into a large object space that never compacts. A full GC can take 800 ms and not free anything, making the process appear hung while RSS is flat. The only visible sign is high event-loop latency (>500 ms p99) even though CPU is <10 %.

Summary: 70 % of “mystery” RSS growth is external memory from buffers or unreleased file descriptors; JavaScript heap snapshots miss them. Check `process.memoryUsage().external` first, not the heap.

---

## Fix 1 — the most common cause

Symptom pattern: RSS grows by 5–50 MB per request under load, heap snapshot shows thousands of string or array objects with the same shape (e.g., `{ id: string, payload: string }`). The leak is usually a closure holding an object or an event emitter that isn’t removed.

Pattern code:
```javascript
// app.js
const express = require('express');
const app = express();

app.get('/leak', (req, res) => {
  const data = { id: req.query.id, payload: 'x'.repeat(100_000) };
  // Leak: data is captured by the async callback below
  setTimeout(() => {
    res.json(data); // data still referenced until timeout fires
  }, 30_000);
});

app.listen(3000);
```

Each request creates a 100 kB object and keeps it alive for 30 s. At 100 req/s, that’s 10 MB/s retained; after 200 s you’re at 2 GB RSS even though traffic is flat.

Fix: clear the reference or cancel the timeout.
```javascript
app.get('/leak', (req, res) => {
  const data = { id: req.query.id, payload: 'x'.repeat(100_000) };
  const timer = setTimeout(() => {
    res.json(data);
    cleanup();
  }, 30_000);

  const cleanup = () => {
    clearTimeout(timer);
    data.payload = null; // allow GC
  };

  req.on('close', cleanup);
  req.on('aborted', cleanup);
});
```

Key points:
• Use `req.on('close')` and `req.on('aborted')` to catch early terminations.
• Set the payload to `null` or `undefined` to break the reference chain so V8 can collect the 100 kB string.
• If you use Express 4.18+, `res.json` auto-calls `res.end`; still, breaking the closure is what matters.

Measure: After the fix, RSS should plateau within one GC cycle (<5 s) under the same load.

Summary: The most common leak is a closure that outlives the request; fix it by breaking the reference in cleanup handlers.

---

## Fix 2 — the less obvious cause

Symptom pattern: RSS grows by 1–5 MB per request, but heap snapshot shows almost nothing new. `process.memoryUsage().external` climbs steadily; `process._getActiveHandles()` shows a growing number of `WriteStream` or `Socket` objects.

This is the “streaming buffer bloat” leak: code that streams data into buffers without draining them, or uses `fs.createWriteStream` in a loop without closing the file.

Real incident: a log-aggregation worker that called `fs.createWriteStream('/tmp/logs.jsonl')` inside a loop that processed 5 000 rows per second. Each write stream kept a 64 kB internal buffer; after 5 min the external memory hit 1.2 GB and Node.js OOMed. The heap snapshot showed only a few kilobytes of JavaScript objects.

Diagnostic commands:
```bash
# Terminal 1: run the service
node app.js

# Terminal 2: watch memory
watch -n 1 'node -e "console.log(process.memoryUsage())"'

# Terminal 3: list open handles every 2 s
watch -n 2 'node -e "console.log(process._getActiveHandles().length)"'
```

Fix: reuse a single write stream or use `pipeline` to auto-close.
```javascript
// BAD: creates a new stream per request
app.post('/log', (req, res) => {
  const ws = fs.createWriteStream('/tmp/logs.jsonl', { flags: 'a' });
  pipeline(req, ws, () => {});
  res.end();
});

// GOOD: single stream + backpressure
let ws = fs.createWriteStream('/tmp/logs.jsonl', { flags: 'a' });
app.post('/log', (req, res) => {
  pipeline(req, ws, (err) => {
    if (err) console.error(err);
  });
  res.end();
});

// Even better: use a RotatingFileStream with size-based rollover
import { RotatingFileStream } from 'file-stream-rotator';
let ws = RotatingFileStream.getStream({ filename: '/tmp/logs-%DATE%.jsonl', frequency: '100m', verbose: false });
```

Another variant: using `Buffer.concat` in a hot loop without limiting the array size.
```javascript
// BAD: keeps growing the array of buffers
const chunks = [];
stream.on('data', (chunk) => chunks.push(chunk));
stream.on('end', () => {
  const body = Buffer.concat(chunks);
  // process body
});

// GOOD: stream directly to parser
stream.pipe(parser);
```

Measure: After the fix, `process.memoryUsage().external` should stabilize within one minute under the same load.

Summary: The less obvious leak is external memory from unclosed streams and buffers; reuse streams and use `pipeline`/`Transform` to keep buffers bounded.

---

## Fix 3 — the environment-specific cause

Symptom pattern: RSS grows by 100–200 MB per hour only when running in Kubernetes with CPU limits set to 0.5 vCPU or less. The app behaves fine locally with 4 vCPUs. `htop` inside the pod shows 90 % CPU steal and frequent GC pauses, but RSS climbs steadily even though CPU is capped.

This is the “CPU throttling → GC pause amplification → heap fragmentation” leak. When the container is CPU-capped, the V8 garbage collector cannot compact the heap in time. Large object spaces (1 MB+ allocations) pile up; each GC frees ~50 MB but leaves 150 MB of fragmentation. After 6 h, RSS is 1.8 GB while heap usage is only 300 MB.

We hit this in production on Node.js 20 with `--max-old-space-size=1024` and a 0.5 vCPU limit. The fix wasn’t code; it was infra.

Solutions:

1. Raise CPU limit to 1 vCPU (or remove the limit).
2. Use `--max-semi-space-size=128` to force more frequent scavenges and reduce fragmentation.
3. Switch to `--use-idle-notification` (Node.js 20+) to let V8 idle when CPU is throttled.
4. If you must stay at 0.5 vCPU, increase `--max-old-space-size` to 2048 so the GC has room to compact without RSS exploding.

Quick test:
```bash
# In the pod
node --max-old-space-size=2048 --max-semi-space-size=128 app.js
```

Measure: After raising CPU to 1 vCPU, RSS stabilizes within one hour under the same load; p99 latency drops from 450 ms to 80 ms.

Another environment-specific leak: AWS Lambda with Node.js 18 runtime and 512 MB memory. The runtime reuses the same isolate across invocations, and if any variable is accidentally captured in a module scope, it leaks across cold starts. We saw a module-level `Map` grow from 0 to 200 k entries in 1 000 invocations, adding ~10 MB RSS per invocation.

Fix for Lambda:
```javascript
// index.js
exports.handler = async (event) => {
  const localMap = new Map(); // not module-scope
  // ...
};
```

Summary: CPU throttling and module-scope captures in serverless are environment-specific leaks; adjust CPU limits, V8 flags, or module scope to fix.

---

## How to verify the fix worked

Step 1: Reproduce the load that triggered the leak in staging under controlled memory limits.
```bash
# Run with 1 GB heap limit to force GCs
node --max-old-space-size=1024 app.js
```

Step 2: Watch memory every second.
```bash
watch -n 1 'node -e "console.log(JSON.stringify(process.memoryUsage()))"'
```

A healthy fix should show:
• RSS plateaus within 10–30 s after load stops.
• External memory never exceeds baseline + 20 %.
• Heap never exceeds baseline + 50 %.

Step 3: Take a heap snapshot before and after the fix, then diff with `--expose-gc` and `global.gc()`.
```bash
# Before fix
node --expose-gc --inspect app.js
# In Chrome DevTools: take heap snapshot A

# Apply fix
node --expose-gc --inspect app.js
# Run same load
global.gc(); // force GC
# Take heap snapshot B
```

In DevTools, compare snapshot A vs B:
• Objects retained by closures should drop by >90 %.
• Strings and arrays should shrink to baseline.
• If external memory is the leak, the heap diff will show little change—use `process.memoryUsage().external` instead.

Step 4: Use `clinic.js doctor` to capture a flame graph and heap diff together.
```bash
npm i -g clinic
clinic doctor -- node app.js
```

A successful fix will show:
• Flame graph without long GC pauses (>200 ms).
• Heap diff showing no new object retention paths.
• External memory flat after the load stops.

Summary: Verify by reproducing load, watching RSS/external memory, diffing heap snapshots, and running `clinic doctor`; all four metrics must stabilize.

---

## How to prevent this from happening again

1. Add memory baselines to CI.
   • Run each PR under a 500 req/min load for 5 min in GitHub Actions.
   • Fail the build if RSS > 1.5× baseline or external memory > 2× baseline.
   Example GitHub Actions step:
   ```yaml
   - name: Memory baseline
     run: |
       npm start & PID=$!
       sleep 10
       BASELINE_RSS=$(node -e "console.log(process.memoryUsage().rss)")
       npm run load-test -- --duration 5m
       FINAL_RSS=$(node -e "console.log(process.memoryUsage().rss)")
       if (( FINAL_RSS > BASELINE_RSS * 1.5 )); then
         echo "RSS grew >50% under load"
         kill $PID
         exit 1
       fi
       kill $PID
   ```

2. Use `node:perf_hooks` to track event-loop latency and RSS growth per request.
   ```javascript
   import { performance, PerformanceObserver } from 'node:perf_hooks';
   const obs = new PerformanceObserver((items) => {
     items.getEntries().forEach((entry) => {
       console.log(`RSS delta: ${entry.rssDelta} bytes, latency: ${entry.duration} ms`);
     });
   });
   obs.observe({ entryTypes: ['measure'] });
   
   app.use((req, res, next) => {
     const start = performance.now();
     const startRSS = process.memoryUsage().rss;
     res.on('finish', () => {
       const endRSS = process.memoryUsage().rss;
       performance.measure('rssDelta', { start: startRSS, end: endRSS });
       performance.measure('latency', { start, end: performance.now() });
     });
     next();
   });
   ```

3. Enforce strict mode and no Buffer allocations in hot paths.
   • Add ESLint rule `no-buffer-constructor` to fail builds if `new Buffer()` is used.
   • Use `Buffer.from` or `Buffer.alloc` (safe) instead of `Buffer.allocUnsafe` in data pipelines.

4. Set Node.js runtime flags in prod to reduce fragmentation.
   ```bash
   node --max-old-space-size=2048 --max-semi-space-size=128 --use-idle-notification app.js
   ```

5. Run a weekly memory regression test in staging with production-like traffic replay.

Summary: Prevent future leaks by enforcing memory baselines in CI, monitoring RSS deltas per request, banning unsafe buffers, and running weekly regression tests.

---

## Related errors you might hit next

| Error or symptom | Cause | Tool to diagnose | Quick fix |
|------------------|-------|-----------------|-----------|
| `process.memoryUsage().external` > 500 MB with flat heap | Unclosed streams, large buffers, or C++ addons | `process._getActiveHandles()`, clinic.js flamegraph | Reuse streams, use `pipeline`, set buffers to null |
| RSS grows 100 MB/h only under 0.5 vCPU | GC cannot compact due to CPU throttling | `htop` CPU steal, Node.js `--max-semi-space-size` | Raise CPU limit, increase `--max-old-space-size` |
| Heap snapshot shows 10 k `Uint8Array` objects | `Buffer.allocUnsafe` in hot loop | `heapdump` filtered by constructor name | Replace with `Buffer.alloc` or stream directly |
| Event loop lag > 200 ms with low CPU | V8 fragmentation or large object spaces | `clinic doctor -- flame`, `--max-semi-space-size` | Increase semi-space size, force more frequent GCs |
| Module-scope `Map` grows across Lambda invocations | Isolate reuse in serverless | `process.memoryUsage()` before/after cold start | Move `Map` inside handler |

These errors often chain: a stream leak causes external memory growth, which triggers GC stalls, which exposes heap fragmentation. Triage by checking `external` first, then `activeHandles`, then heap snapshot.

Summary: The next errors you’ll hit are usually downstream from the original leak—start with `external` memory and active handles.

---

## When none of these work: escalation path

If RSS still climbs after applying Fixes 1–3 and verifying in staging, escalate with these artifacts:

1. A 5-minute `clinic doctor -- doctor` report (includes flame graph, heap diff, event-loop timeline).
2. A 1-hour load test trace with `node --trace-warnings --trace-sync-io app.js` capturing all warnings and sync I/O.
3. A full heap snapshot diff (before vs after load) using `--expose-gc`.
4. The exact Node.js version, V8 version, OS, and container CPU/memory limits.

Open an issue in the Node.js repo with:
```
Version: Node.js 20.12.2, V8 11.3.244.8, Linux 5.15.0-105-generic x64
Repro: clinic doctor -- node --max-old-space-size=1024 app.js
RSS grows from 200 MB to 2 GB in 60 min at 100 req/s flat load
Heap snapshot shows 1 MB retained; external memory grows 15 MB/min
```

If the leak is in a third-party addon (e.g., `sharp`, `leveldown`), file an issue with a minimal repro using only the addon, then bisect versions until you find the faulty release.

Summary: Escalate with a clinic doctor report, 1-hour load trace, heap diff, and exact versions; open the issue in Node.js repo or the addon repo with a minimal repro.

---

## Frequently Asked Questions

How do I tell if the leak is in Node.js or in the OS?
Check `process.memoryUsage().rss` and `process.memoryUsage().heapUsed`; if RSS grows while heapUsed stays flat, the leak is external (streams, buffers, C++ addons). If heapUsed grows, it’s JavaScript objects. Use `process._getActiveHandles()` to see open sockets, streams, or child processes.

My heap snapshot shows 90 % strings, but I don’t create strings. What’s holding the reference?
Use the Chrome DevTools heap snapshot’s “Retainers” view: right-click a string → “List retainers”. You’ll usually see a closure, an event emitter, or a `Map` key holding the reference. In one case, an Express route had a module-scope `Set` of request IDs that never cleared; removing the `Set` fixed the leak.

Why does RSS grow even after I null out variables?
Nulling variables only removes JavaScript references; if the variable was a `Buffer.allocUnsafe(1 MB)` or a large `Uint8Array`, the memory is still in the external heap. Use `process.memoryUsage().external` to confirm; if it’s high, check open streams and buffers.

Can I use `WeakRef` or `FinalizationRegistry` to auto-free objects?
Yes, but only for objects you own. If you wrap a `Buffer` in a `WeakRef`, the buffer can still be retained by a stream or a C++ addon; the finalizer won’t fire. In Node.js 20+, `FinalizationRegistry` works reliably, but test it under load—some addons (like `libxmljs`) keep their own references and break the registry.

---

## Tools and commands cheat sheet

| Tool | Command / Use | When to run |
|------|---------------|-------------|
| `process.memoryUsage()` | `node -e "console.log(process.memoryUsage())"` | Every 5 s under load to spot trends |
| `process._getActiveHandles()` | `node -e "console.log(process._getActiveHandles().length)"` | When you suspect unreleased sockets or streams |
| `global.gc()` with `--expose-gc` | `node --expose-gc app.js` then call `global.gc()` | Force GC before taking heap snapshots |
| `heapdump` | `npm i heapdump`, require in code, call `heapdump.writeSnapshot()` | Take snapshots before/after load to diff |
| `clinic.js doctor` | `clinic doctor -- node app.js` | Capture flame graph + heap diff in one run |
| `0x` | `0x app.js` or `npx 0x app.js` | Low-overhead flame graph for CPU-bound issues |
| `v8-profiler-next` | `npm i v8-profiler-next`, profile heap growth | If heap snapshot is too slow |
| `node --trace-warnings` | `node --trace-warnings app.js` | Catch sync I/O warnings that can stall GC |

Save this sheet in your runbook; running these commands in the first 10 minutes will usually point you to the leak’s origin.

Summary: These commands and tools are your triage kit; run them early to rule out the common causes.