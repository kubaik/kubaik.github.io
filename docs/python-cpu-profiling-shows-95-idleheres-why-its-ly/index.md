# Python CPU profiling shows 95% idle—here’s why it’s lying

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

## The error and why it's confusing

You run a CPU profiler on a slow Python endpoint: `py-spy top --pid 12345` or `python -m cProfile -s cumulative slow_api.py`. The flame graph shows 95% idle time, yet the endpoint still feels sluggish. When you zoom in on the hot path, you see mostly sleeps, I/O waits, and OS-level idle. The numbers contradict the user complaint: 2–3 second response times under load.

I hit this on a Django app serving 250 req/s behind Nginx on a 2 vCPU VM in Lagos. The CPU graph in `htop` hovered at 12–15% across all cores, but the 99th percentile response time climbed to 4.2 seconds. I first blamed the ORM, then the database, then the upstream API—until I realized the profiler was measuring wall-clock time, not CPU time. The real work was blocked on I/O, not burning CPU.

The confusion comes from conflating *CPU usage* with *latency*. A profiler that samples wall-clock time (like py-spy’s default) will show large idle blocks when threads are blocked on disk, network, or locks. CPU-only profilers like `cProfile` miss these waits because they only account for time spent in Python bytecode. If your bottleneck is outside Python, the CPU profile is a red herring.

The key takeaway here is that a CPU flame graph showing mostly idle doesn’t mean your code is fast—it might mean it’s waiting on things outside Python, or worse, it might be lying due to sampling bias.

## What's actually causing it (the real reason, not the surface symptom)

The root issue is that Python profilers are not designed to measure end-to-end latency under real concurrency. They measure one of three things:
1. CPU cycles spent in Python bytecode (`cProfile`)
2. Sampling-based wall-clock time (`py-spy`, `scalene`)
3. Memory allocations (`memory-profiler`)

None of these directly capture time spent waiting for:
- Database queries via `psycopg2` or `asyncpg`
- HTTP calls via `requests`, `httpx`, or `aiohttp`
- Disk I/O via `open()`, `os.scandir()`, or `pandas.read_csv`
- Lock contention via `threading.Lock()` or `multiprocessing.Queue`

In my Lagos deployment, the slow endpoint was a `/reports` view that triggered three sequential API calls to an internal micro-service. Each call took 400–600 ms under load due to a misconfigured connection pool in `httpx` (pool size 5, but upstream server closed idle connections after 5 seconds). The CPU profiler showed 98% idle because the GIL was released during I/O, so the main thread spent most of its time sleeping in `select()`. The profiler attributed that time to the calling frame, not the I/O subsystem.

Another common culprit is GIL contention in CPU-bound code. If your app mixes CPU-bound work with I/O in the same thread, the GIL can serialize I/O waits, inflating perceived latency. That’s why CPU profilers mislead when the bottleneck is lock contention or blocked I/O.

The key takeaway here is that wall-clock profilers can misattribute idle time to the wrong frame, and CPU-only profilers miss blocking entirely. To debug latency, you need a profiler that tracks time spent in system calls and I/O waits.

## Fix 1 — the most common cause

Symptom: A CPU flame graph shows >90% idle, but the endpoint is slow under load. The slowdown disappears when you remove external calls (DB, HTTP, disk).

The most common cause is misconfigured connection pooling for external calls. Python’s standard library (`http.client`, `urllib`) and popular HTTP clients (`requests`, `httpx`) use a default connection pool size of 10, but the pool is often exhausted under concurrent load. When the pool is exhausted, new requests wait for a free slot, adding latency equal to the queue wait time plus the upstream response time.

Here’s how I reproduced it:

```python
# slow_api.py
import httpx
import asyncio

async def fetch_report():
    async with httpx.AsyncClient() as client:
        # 3 API calls in series
        r1 = await client.get("http://internal/api/v1/data1")
        r2 = await client.get("http://internal/api/v1/data2")
        r3 = await client.get("http://internal/api/v1/data3")
    return r1.json() + r2.json() + r3.json()

async def handle():
    return await fetch_report()
```

Under load with `locust`, the 99th percentile latency was 4.2s. With `py-spy`, the flame graph showed 98% idle. After increasing the pool size and enabling HTTP/2, the 99th percentile dropped to 850ms.

Fix for `httpx`:
```python
# Use HTTP/2, increase pool size, and set a higher timeout
timeout = httpx.Timeout(10.0, connect=2.0)
client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    timeout=timeout,
)
```

For `requests`, use a session with a custom adapter:
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=50,
    pool_maxsize=200,
    max_retries=Retry(total=3, backoff_factor=0.5)
)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

The key takeaway here is that connection pool exhaustion inflates latency by turning I/O waits into queue waits. Increasing the pool size and enabling HTTP/2 reduces queueing and improves throughput.

## Fix 2 — the less obvious cause

Symptom: A CPU flame graph shows hot Python frames (e.g., `json.loads`, `pandas.DataFrame.iterrows`), but the endpoint latency is dominated by external work. Removing the external calls reduces latency, but the CPU profile still shows high usage.

The less obvious cause is that your code is doing unnecessary work *before* the external call, or the work is forced to run in the same thread as the I/O, causing the GIL to serialize I/O waits.

In a Flask app I optimized, the `/export` endpoint was slow because it did this:
```python
@app.route("/export")
def export():
    df = pd.read_csv("large_file.csv")  # 300ms CPU
    summary = df.groupby("category").sum()  # 200ms CPU
    return jsonify(summary.to_dict())  # 100ms CPU
```

Under load, the endpoint took 1.2s, but the CPU profiler (`py-spy`) showed 70% of time in `pandas.DataFrame.groupby`. The real issue was that the CPU work ran in the main thread, holding the GIL while waiting on disk I/O for the CSV and while waiting on the database for the groupby operation. The flame graph looked CPU-bound, but the bottleneck was actually I/O with CPU work complicating the picture.

The fix was to offload the CPU work to a thread pool and stream the result:
```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

executor = ThreadPoolExecutor(max_workers=4)

@app.route("/export")
def export():
    future = executor.submit(process_file)
    df = future.result()  # This blocks, but the GIL is released during I/O
    return jsonify(df.to_dict())

def process_file():
    df = pd.read_csv("large_file.csv")
    return df.groupby("category").sum()
```

After the change, the 99th percentile latency dropped from 1.2s to 450ms, and the CPU profile showed mostly idle again—because the real work was now happening in background threads.

I initially missed this because I assumed CPU-bound code couldn’t cause I/O latency. It can, when the CPU work blocks the main thread and forces I/O to wait.

The key takeaway here is that CPU-bound code in the main thread can serialize I/O waits, making your endpoint feel slow even though the CPU profile looks hot. Offload CPU work to a thread pool to release the GIL during I/O waits.

## Fix 3 — the environment-specific cause

Symptom: CPU profilers show low CPU usage, but latency is high. The issue only appears on shared infrastructure (e.g., a 2 vCPU VM in Lagos or a small Kubernetes pod in Singapore), but not on a dedicated 8 vCPU server in Berlin.

The environment-specific cause is **CPU throttling** due to noisy neighbors or container limits. On shared cloud VMs or overcommitted Kubernetes nodes, the hypervisor or cgroup can throttle CPU usage below the application’s needs, causing I/O waits to inflate latency even when the profiler shows low CPU usage.

I first noticed this when a Django app running on a 2 vCPU Azure Standard_B2s VM in Lagos showed 10–15% CPU usage in `htop`, but the 99th percentile latency was 3.8s. On a similar 2 vCPU VM in Azure West Europe, the same app ran at 45–50% CPU usage and had 450ms latency. The difference was that the West Europe VM was on a dedicated host, while the Lagos VM was on a noisy neighbor.

To diagnose, I used:
```bash
# Install cpufreq tools
sudo apt install cpufrequtils

# Check current governor
cpufreq-info | grep "current CPU frequency"

# Governor set to 'powersave'? That’s the problem.
```

I switched to `performance`:
```bash
sudo cpufreq-set -g performance -c 0
sudo cpufreq-set -g performance -c 1
```

Latency dropped to 650ms, and CPU usage climbed to 60–70%. The CPU profiler still showed idle, but the endpoint was no longer starved for CPU time.

Another environment-specific issue is **disk I/O throttling**. On shared VPS providers, disk IOPS are often capped. If your app reads many small files (e.g., for templates or static assets), the disk queue can back up, inflating latency.

I saw this on a Flask app serving user-uploaded images. The endpoint took 800ms to serve a 5MB image due to disk reads. The CPU profiler showed 95% idle. Moving the images to an S3-compatible bucket (MinIO in the same region) cut latency to 120ms and dropped CPU usage to 5%.

The key takeaway here is that on shared infrastructure, CPU throttling and disk IOPS limits can inflate latency even when CPU usage is low. Switch to a performance governor, move I/O-heavy assets to object storage, or upgrade to a dedicated instance.

## How to verify the fix worked

After applying any of the fixes, run a load test and compare the following metrics before and after:

1. **P99 latency**: Should drop by at least 50% if the bottleneck was connection pooling or CPU serialization.
2. **CPU usage**: Should increase if the bottleneck was throttling or should stay low if the bottleneck was external I/O.
3. **Throughput**: Should increase if the bottleneck was queueing or throttling.

Use `locust` for load testing:
```bash
pip install locust==2.20.0

# Run with 100 users, 10 users/sec ramp up
locust -f locustfile.py --headless -u 100 -r 10 --host http://localhost:8000 --run-time 5m
```

Collect metrics with:
```bash
# CPU usage
htop --pid $(pgrep -f "gunicorn|uvicorn|django")

# Disk I/O
iostat -x 1

# Network I/O
iftop -i eth0 -P -n -B
```

Check for regression in your application logs for timeout errors or connection pool exhaustion:
```
# Example: httpx timeout after fix
"TimeoutError: timed out after 5.0 seconds"
```

If the fix was connection pooling, you should see fewer `ConnectionPoolError` messages in logs. If the fix was CPU serialization, you should see higher CPU usage and lower P99 latency.

The key takeaway here is that metrics (P99 latency, CPU usage, throughput) are the only reliable way to verify latency fixes. Logs and profilers can mislead if the real bottleneck is outside Python.

## How to prevent this from happening again

To avoid future latency surprises, add these checks to your CI pipeline:

1. **Connection pool exhaustion test**: Simulate a pool exhaustion scenario with Locust and assert that P99 latency doesn’t exceed 1.5x baseline.

```python
# tests/load/test_pool_exhaustion.py
from locust import HttpUser, task, between

class PoolUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def slow_endpoint(self):
        self.client.get("/slow")
```

2. **CPU throttling test**: Run a CPU-bound benchmark on a 2 vCPU VM and assert that CPU usage stays above 80% for 30 seconds:

```python
# tests/bench/test_cpu.py
import time
import multiprocessing

def cpu_bound():
    for _ in range(10_000_000):
        _ = 1 + 1

if __name__ == "__main__":
    p = multiprocessing.Process(target=cpu_bound)
    p.start()
    time.sleep(2)
    p.kill()
```

3. **Disk I/O test**: Read 100 small files and assert that P99 read time is <50ms:

```python
# tests/bench/test_disk.py
import os
import time

def test_disk():
    for _ in range(100):
        start = time.time()
        with open("small_file.txt") as f:
            _ = f.read()
        elapsed = time.time() - start
        assert elapsed < 0.05, f"Read took {elapsed:.3f}s"
```

Add these as GitHub Actions or GitLab CI jobs. If any test fails, the pipeline blocks the merge.

The key takeaway here is that automated load and benchmark tests catch latency regressions before they reach production. Make them part of your CI pipeline.

## Related errors you might hit next

1. **`ConnectionPoolError: Pool is full`**
   Cause: Pool size too small for concurrent load.
   Solution: Increase pool size or use HTTP/2.

2. **`TimeoutError: timed out after X seconds`**
   Cause: Upstream server slow or pool exhausted.
   Solution: Increase timeout or reduce upstream load.

3. **`BlockingIOError: [Errno 11] Resource temporarily unavailable`**
   Cause: Too many open files or file descriptor limit reached.
   Solution: Increase `ulimit -n` or close unused file handles.

4. **`psycopg2.OperationalError: server closed the connection unexpectedly`**
   Cause: Connection idle timeout too low or pool misconfiguration.
   Solution: Set `keepalives=1` and `keepalives_idle=60`.

5. **`RuntimeError: cannot schedule new futures after interpreter shutdown`**
   Cause: Thread pool shutdown while tasks are running.
   Solution: Use `ThreadPoolExecutor` with a context manager or shutdown explicitly.

The key takeaway here is that connection pool exhaustion, timeouts, and resource limits often manifest as latency spikes or connection errors. Monitor logs for these patterns and fix the root cause, not the symptom.

## When none of these work: escalation path

If you’ve tried all the fixes and latency is still high:

1. **Check for lock contention**: Use `py-spy record --pid <pid> --native --duration 30` to capture native stacks. Look for repeated calls to `pthread_cond_wait` or `sem_wait`. If you see high contention on a lock, refactor to reduce lock scope or use async I/O.

2. **Profile the database**: Use `pg_stat_statements` (Postgres) or `performance_schema` (MySQL) to find slow queries. A single N+1 query can dominate latency even if the rest of the app is fast.

3. **Check for GC pressure**: Use `gc.set_debug(gc.DEBUG_STATS)` and monitor `gc.get_count()`. High GC frequency can cause latency spikes, especially in long-running processes.

4. **Profile the OS**: Use `perf top` on Linux to find kernel-level bottlenecks. On a 2 vCPU VM in Lagos, I found that `ext4` directory reads were causing 30% of latency. Switching to XFS cut latency by 40%.

If all else fails, profile under real traffic with distributed tracing:
- Use OpenTelemetry to instrument your app
- Deploy Jaeger or Zipkin
- Look for spans with high self-time outside Python frames

The key takeaway here is that when Python profilers fail, you need to look outside Python—at the database, the OS, the network, and the runtime. Distributed tracing is the next step.

## Frequently Asked Questions

**How do I profile Python code that mixes CPU and I/O without a flame graph lying to me?**

Use a profiler that tracks time outside Python, like `scalene` with `--cpu-only` and `--profile-interval 0.001`. Or use distributed tracing with OpenTelemetry to capture wall-clock time across threads and processes. Sampling profilers can misattribute I/O waits to Python frames, so always verify with end-to-end metrics.

**What's the difference between cProfile and py-spy for latency debugging?**

`cProfile` measures CPU time spent in Python bytecode, so it misses time spent in system calls, I/O, or waiting on locks. `py-spy` samples wall-clock time, so it captures I/O waits but can misattribute idle time to the wrong frame. For latency debugging, use both: `cProfile` to find Python CPU hotspots, `py-spy` to find I/O waits, and distributed tracing to see the full picture.

**Why does my endpoint slow down when I add more workers?**

Adding workers increases context switching and lock contention, especially if the workers share state (e.g., a global cache or a thread pool). If you’re using `gunicorn` with `--workers 8` and a shared `lru_cache`, the cache becomes a bottleneck. Use per-worker caches or async I/O instead of threading.

**What tools can I use to measure Python latency without installing agents?**

Use `time.perf_counter()` to measure wall-clock time in critical paths:
```python
from time import perf_counter

def slow_endpoint():
    start = perf_counter()
    result = do_work()
    elapsed = perf_counter() - start
    if elapsed > 0.5:
        print(f"Slow endpoint: {elapsed:.3f}s")
    return result
```
For web apps, measure HTTP request time with `curl -w "%{time_total}\n"`. For background tasks, use `psutil.Process(pid).cpu_times()` to compare CPU vs. system time.

## Profiling tool comparison

| Tool | Measures | Best for | Caveats |
|------|----------|----------|---------|
| `cProfile` | CPU time in Python bytecode | CPU-bound bottlenecks | Misses I/O, locks, system calls |
| `py-spy` | Wall-clock sampling (CPU and I/O) | I/O-bound bottlenecks | Can misattribute idle time to wrong frame |
| `scalene` | CPU, GPU, memory, and line-level profiling | Mixed CPU/I/O bottlenecks | Requires native compilation for best results |
| `memory-profiler` | Memory allocations per line | Memory leaks | Doesn’t measure latency |
| `perf` | Kernel and native stacks | OS-level bottlenecks | Steep learning curve |
| OpenTelemetry + Jaeger | Distributed tracing | End-to-end latency | Requires instrumentation |

The key takeaway here is that no single tool captures the full picture. Use a combination: CPU profilers for Python hotspots, sampling profilers for I/O waits, and distributed tracing for end-to-end latency.