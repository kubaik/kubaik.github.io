# Virtual threads: what actually changes

## The gap between the tutorial and the load test

Most introductions to virtual threads stop where the interesting part starts. They explain that virtual threads are cheap, show a `Thread.ofVirtual().start(...)` snippet, and imply that a service will get faster. In practice, the first thing a team notices after switching is often that nothing changed, or that latency got worse. Both outcomes are predictable once you understand what virtual threads do and do not remove.

Java virtual threads (JEP 444, final in JDK 21) are the first change to Java's threading model that genuinely alters how blocking network services are written. They look like threads, behave like threads in a debugger, and yet the rules for using them well differ from the rules for platform threads. This article covers what changes, what does not, and where the sharp edges are when they meet databases, HTTP APIs, and message brokers.

The decision this article supports: whether virtual threads will simplify a given service, or merely move its bottleneck from threads to connections.

## The one-sentence version

Virtual threads make thread-per-request cheap again, but they do not make downstream dependencies cheaper. If a service is limited by database connections, HTTP client pools, or CPU, virtual threads will not increase throughput; they will change where the queue forms.

Everything below is about the consequences of that sentence.

## What virtual threads actually are, and are not

A virtual thread is a Java thread scheduled by the JDK, not by the operating system. The JDK runs many virtual threads on a small pool of carrier threads, by default sized to the number of available processors; the `jdk.virtualThreadScheduler.parallelism` system property can change that. When a virtual thread blocks on a supported operation, the JDK unmounts it from its carrier, freeing the carrier to run another virtual thread. When the operation completes, the virtual thread is scheduled again.

That unmounting is the key mechanism. It only works when the blocking call is one the JDK knows how to handle. The list is wide but not universal: `Socket` I/O, `java.nio` channels, `Thread.sleep`, `Object.wait`, most `java.util.concurrent` locks and synchronizers, and `java.net.http.HttpClient` all cooperate. Blocking on `synchronized` pinned the carrier thread in JDK 21 and 22; JDK 24 (JEP 491) removed that limitation, so a virtual thread blocked inside a `synchronized` block no longer holds its carrier. On JDK 21 or 22 this matters a great deal and is the most common source of "virtual threads made things worse" reports.

What virtual threads are not:

- They are not a replacement for reactive programming in every case. They replace the *thread pool* part of the problem, not the *backpressure* part.
- They are not faster per request. A virtual thread doing the same work as a platform thread takes the same time.
- They are not a thread pool. Virtual threads should not be pooled. Create one per task; they are cheap (a few hundred bytes of heap initially, not a megabyte of stack).
- They do not make `ThreadLocal` free. Each virtual thread still has its own thread-locals, and a service with thousands of virtual threads and a fat `ThreadLocal` cache can use a lot of memory. `ScopedValue` was added as a preview in JDK 21 and finalized later, specifically to address per-thread context; `ThreadLocal` still works and still costs.

## The mental model: threads are cheap, connections are not

A useful analogy is a restaurant. Platform threads are tables: there is a fixed number, and when all tables are full, new customers wait outside. Virtual threads are effectively unlimited tables, so nobody waits at the door. But the kitchen — the database, the downstream API — still has a fixed number of burners. Adding tables does not add burners. It just means more people are seated, staring at the kitchen, wondering why their food is late.

This is why the first thing to check when adopting virtual threads is not the thread count. It is every pool between the service and the work:

- The JDBC connection pool. For HikariCP, the documented default `maximumPoolSize` is 10.
- The HTTP client connection pool. `java.net.http.HttpClient` has its own limits, and so does every HTTP client library.
- The Redis client pool (Lettuce, Jedis).
- Any semaphore or bulkhead added to protect a downstream.

If a service had 200 platform threads and a Hikari pool of 20, it was already serializing on the pool. Virtual threads do not change that. They make the serialization point more obvious, because now there may be 10,000 virtual threads all waiting on the same 20 connections.

## Worked example: a blocking HTTP service, before and after

Suppose a service handles requests by calling two downstream HTTP APIs and one database query. Each downstream call takes, illustratively, 50 ms; the database query takes 20 ms. The service is CPU-light. Assume a 4-core box.

**Before (platform threads, fixed pool of 200):**

```java
ExecutorService pool = Executors.newFixedThreadPool(200);
```

Each request occupies a platform thread for roughly 120 ms of wall time, almost all of it waiting. With 200 threads, the theoretical maximum throughput is 200 / 0.120 s ≈ 1,666 requests/s, assuming the downstreams can keep up. If they cannot, the pool fills and requests queue. The queue is invisible until latency spikes.

**After (virtual threads, one per request):**

```java
try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
    server.setExecutor(executor);
    server.start();
}
```

Now the thread count is not the limit. The limit is whichever downstream saturates first. If the Hikari pool is 10 and the database query takes 20 ms, the database caps throughput at 10 / 0.020 s = 500 requests/s, regardless of how many virtual threads exist. The HTTP client pool might cap higher or lower. The first saturated dependency sets the ceiling.

This is why "we switched to virtual threads and throughput didn't change" is a common report. The thread pool was never the bottleneck; it was just the most visible one. Virtual threads moved the queue from the thread pool to the connection pool, where it is harder to see.

**A more interesting case:** if the service was previously limited to 200 concurrent requests and each request blocked on a slow downstream (say 2 seconds), it was capped at 200 / 2 s = 100 requests/s. With virtual threads, thousands of requests can be in flight, and throughput is limited by the downstream's capacity, which might be 500 requests/s. That is a real improvement, but only because the thread pool was the binding constraint. If the downstream was already the constraint, nothing changes except the shape of the queue.

## What changes in the code, and what does not

**What does not change:** blocking code stays blocking. That is the point. There is no need to rewrite a service as reactive or async. A method that calls `jdbcTemplate.query(...)` and then `httpClient.send(...)` still works, and can now run on a virtual thread without consuming a platform thread for the duration.

**What changes:**

1. **Thread pools become per-task executors.** Replace `Executors.newFixedThreadPool(n)` with `Executors.newVirtualThreadPerTaskExecutor()` for request handling. Do not pool virtual threads manually; the executor creates a new one per task.

2. **Thread-local caches need rethinking.** A `ThreadLocal<byte[]>` buffer of 10 KB, multiplied by 10,000 virtual threads, is roughly 100 MB of heap. On 200 platform threads it was roughly 2 MB. Audit thread-locals before scaling concurrency.

3. **Pinning becomes a debugging concern.** On JDK 21 and 22, a virtual thread that blocks inside a `synchronized` block pins its carrier. If enough virtual threads pin, the carrier pool can starve and latency spikes appear that look like thread starvation. The JFR event `jdk.VirtualThreadPinned` (with a configurable threshold) is how to find these. On JDK 24+, this specific cause is gone, but native code that blocks (some JNI calls, some file I/O on older JDKs) can still pin.

4. **Thread-local-based frameworks need checking.** Some older frameworks (logging, transaction management, security contexts) rely on thread-locals and assume a bounded number of threads. Most modern versions are fine, but ones that cache per-thread objects are worth a look.

5. **Structured concurrency becomes natural.** `StructuredTaskScope` (preview in JDK 21, still preview in later releases) pairs well with virtual threads: fork a subtask per downstream call, join, handle failures together. It is not required to use virtual threads, but it is the idiomatic way to fan out.

## The pinning trap, specifically

Pinning is the failure mode most likely to bite a team moving to virtual threads on JDK 21. The mechanism: a virtual thread running inside a `synchronized` block cannot unmount when it blocks. It holds its carrier thread until the block exits. If there are, say, 4 carrier threads and 100 virtual threads all blocked inside `synchronized` on a slow I/O call, the scheduler has no free carriers and everything stalls.

A typical shape that triggers this:

```java
public synchronized String fetch(String key) throws IOException {
    // blocking I/O inside a synchronized method
    return httpClient.send(request, BodyHandlers.ofString()).body();
}
```

On JDK 21, every virtual thread calling `fetch` pins a carrier for the duration of the HTTP call. With enough concurrency, the carrier pool is exhausted and unrelated virtual threads stop making progress.

The fix is not to remove `synchronized` everywhere; it is to avoid blocking I/O inside it. Use `ReentrantLock` (which does not pin) or restructure so the lock is held only around in-memory state. On JDK 24+, the `synchronized` case is fixed, but the general rule still applies to native blocking calls.

To find pinning, enable JFR with the `jdk.VirtualThreadPinned` event and a threshold, for example:

```bash
java -XX:StartFlightRecording:filename=recording.jfr \
     -Djdk.tracePinnedThreads=full \
     -jar service.jar
```

`jdk.tracePinnedThreads` prints stack traces when a virtual thread pins; it is a diagnostic, not something to leave on in production. The JFR event is the production-safe option.

## How to measure whether it helped

Claims about virtual threads are only meaningful against a measurement of the specific service. The measurement does not require a special toolchain; it requires comparing the same load before and after, with the bottleneck visible.

**What to instrument:**

- In-flight requests, sampled continuously.
- Request-handling thread pool: active threads, queued tasks, and pool size (platform-thread baseline only).
- Each connection pool: active connections, idle connections, and threads waiting for a connection. For HikariCP, the relevant metrics include active connections and pending thread acquisition counts.
- Downstream latency (p50, p99) per dependency.
- CPU utilization and GC pause time, to catch thread-local memory pressure.

**What to run:** a fixed-rate load test at a concurrency level that previously saturated the thread pool, against an environment with the same downstream capacity. Run it once on the platform-thread executor and once on the virtual-thread executor, with the same pool sizes. Then compare throughput at a fixed error rate, and p99 latency.

**What to compare:**

- If throughput rises and the connection-pool wait time stays low, the thread pool was the binding constraint.
- If throughput is flat and connection-pool wait time rises, the downstream was already the constraint and virtual threads only lengthened the queue.
- If p99 latency rises sharply with low CPU and no connection-pool saturation, suspect pinning on JDK 21/22; check the JFR pinning events.

An illustrative reading: suppose the before-run shows a 200-thread pool fully occupied and a Hikari pool of 10 with near-zero wait time. After the switch, the pool metric disappears, Hikari wait time climbs to hundreds of milliseconds, and throughput is unchanged. That pattern says the database was always the ceiling and the platform-thread pool was masking it.

## Common misreadings

**"Virtual threads make the service faster."** They make it cheaper to have many concurrent requests. Per-request latency is unchanged unless the thread pool was the bottleneck. If p99 is dominated by a slow downstream, virtual threads will not fix it.

**"The reactive stack should be replaced with virtual threads."** Sometimes yes, sometimes no. Virtual threads remove the need for callback-based async in many services, which is a real simplification. But reactive frameworks also provide backpressure, which virtual threads do not. If a service needs to signal a producer to slow down, it still needs a mechanism for that (bounded queues, semaphores, reactive streams). Virtual threads make the code look synchronous; they do not make the problem synchronous.

**"Virtual threads need pooling."** No. Pooling defeats the purpose. Create one per task and let the JDK schedule them.

**"Virtual threads are only for new code."** They work with existing blocking code, which is the main appeal. Migration is mostly about auditing thread-locals and pinning, not rewriting business logic.

**"`Thread.sleep` and `Object.wait` are the problem."** They are supported and unmount correctly. The problem cases are pinning (synchronized on older JDKs, native blocking) and anything that holds a carrier while waiting.

## A decision framework for a service

Before flipping the executor, answer these in order:

1. **Is the thread pool currently the bottleneck?** Measure in-flight requests versus pool size. If the pool is rarely full, virtual threads will not help throughput. They may still simplify code, which is a valid reason.

2. **What is the first downstream to saturate?** Check connection pool sizes and timeouts for JDBC, HTTP, and cache clients. The smallest of these sets the ceiling. Virtual threads do not raise it.

3. **Are there blocking calls inside `synchronized` on JDK 21/22?** If yes, either upgrade to JDK 24+ or refactor to `ReentrantLock`. Run with JFR pinning events in staging to confirm.

4. **Are there fat thread-locals?** Estimate heap: thread-local size × expected concurrent virtual threads. If that number is uncomfortable, switch to `ScopedValue` where possible or shrink the cached object.

5. **Does anything rely on a bounded thread count?** Some rate limiters, metrics, and frameworks assume a fixed pool. Check them.

If the answers are "pool not the bottleneck, downstream is," virtual threads are a code-simplicity change, not a throughput change. Adopt them for readability, but do not expect a capacity win.

## Quick reference

| Concern | Platform threads | Virtual threads |
|---|---|---|
| Cost per thread | ~1 MB stack by default | A few hundred bytes, grows as needed |
| Pooling | Required | Not recommended; create per task |
| Blocking I/O | Consumes a thread | Unmounts, frees carrier (if supported) |
| `synchronized` + blocking | Fine | Pins carrier on JDK 21/22; fixed in JDK 24 |
| Thread-locals | Bounded by pool size | Multiply by concurrent virtual threads |
| Backpressure | Implicit via pool size | Must be explicit (semaphores, bounded queues) |
| Debugging | Thread dumps show all | Thread dumps show virtual threads; JFR for pinning |

## Frequently Asked Questions

**Do virtual threads work with Spring Boot and an embedded servlet container?**

On recent versions, yes. Spring Boot 3.2 added a configuration property (`spring.threads.virtual.enabled=true`) that switches the embedded container's request executor to virtual threads, and recent servlet containers support virtual threads as an executor. The caveats are the same as anywhere: check the JDBC pool size, audit thread-locals, and watch for pinning on JDK 21/22. Spring's transaction and security contexts use thread-locals, but they are per-request and bounded, so the multiplication concern is usually manageable.

**Should a maximum be set on virtual threads?**

Not at the thread level. Virtual threads are cheap enough that limiting them is the wrong control. What should be limited is the thing they contend for: database connections, HTTP connections, and any downstream with a quota. Use a `Semaphore` or a bounded pool around the downstream call, not around the thread. This is the same discipline that applies with platform threads, just made explicit.

**Why did latency get worse after switching?**

The most common causes, in order: (1) pinning on JDK 21/22 from `synchronized` around blocking calls, which starves carriers; (2) a connection pool that was already saturated, now with a much longer queue of waiters; (3) thread-local memory pressure causing GC pauses; (4) a framework that assumed a bounded thread count and now sees thousands. JFR with `jdk.VirtualThreadPinned` and a thread dump will usually distinguish these quickly.

**Can virtual threads be used with `CompletableFuture`?**

Yes, but be deliberate. `CompletableFuture.supplyAsync` without an executor uses the common ForkJoinPool, which is sized to CPU cores and is not a virtual-thread executor. Pass `Executors.newVirtualThreadPerTaskExecutor()` explicitly to run the task on a virtual thread. Mixing virtual threads with callback-based async is legal but tends to produce code that is neither simple nor fast; prefer structured concurrency or plain blocking calls on virtual threads.

## The next 30 minutes

Open the configuration for the JDBC connection pool (for HikariCP, look for `maximumPoolSize`; for Spring Boot, `spring.datasource.hikari.maximum-pool-size`) and write down the number. Compare it to the current request-handling thread pool size. If the connection pool is smaller, that number, not the thread count, is the real concurrency ceiling, and it is the first thing to reason about before adopting virtual threads.
