# Unlock Python 3.13's free-threaded GIL for web apps

Free-threaded Python builds are the headline change in recent CPython releases: a build where the global interpreter lock can be disabled at runtime, allowing threads to execute Python bytecode in parallel. The tutorials tend to stop at a micro-benchmark. This article covers what happens when you point a real web service at it: which failures appear, which ones are silent, and how to measure whether the switch was worth it.

## What "free-threaded" does and does not change

The documented behaviour is narrower than the marketing suggests:

- The GIL is removed only in a special build of the interpreter, configured with `--disable-gil` at compile time.
- In that build, the GIL can be re-enabled at runtime with the environment variable `PYTHON_GIL=1`, or disabled with `PYTHON_GIL=0`. The default is determined by the build.
- `sys._is_gil_enabled()` reports the current state of the running interpreter.
- Only the interpreter itself is affected. Every C extension, every compiled `.so` or `.pyd` module, and every library that ships prebuilt binaries was compiled against a particular ABI. Extensions built for the GIL-enabled ABI either fail to import, crash, or silently re-enable the GIL for the whole process.

That last point is the one that turns a one-line environment change into a multi-day migration. It is also the point that most quick-start guides omit, because their examples use only the standard library and pure-Python packages.

The realistic expectation is therefore:

- **Pure-Python I/O-bound workloads** can gain throughput, because threads that were previously serialized on the GIL can now overlap.
- **CPU-bound Python workloads** gain little or nothing. Removing the GIL does not give you more cores; the interpreter still executes bytecode on whatever cores are available, and contention moves from the GIL to other shared resources.
- **Anything with a compiled dependency** is gated on that dependency having a free-threaded build.

## Prerequisites

You need:

- Linux x86_64 or arm64 with sudo access, to install a custom interpreter build.
- A C toolchain and the usual Python build dependencies (`build-essential`, `libssl-dev`, `zlib1g-dev`, `libffi-dev`, `libsqlite3-dev`, `liblzma-dev`, and so on).
- A local Redis and PostgreSQL if you want to reproduce the example endpoints below, or equivalents you already run.

Two build flags are worth knowing about:

- `--disable-gil` produces the free-threaded interpreter.
- `--with-pydebug` produces a debug build. Debug builds are substantially slower and are not representative of production performance; avoid them when measuring.

Build and install:

```bash
PYTHON_VERSION=3.13.0
wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz
tar -xf Python-${PYTHON_VERSION}.tar.xz
cd Python-${PYTHON_VERSION}

./configure --disable-gil --enable-optimizations --prefix=/usr/local/python3.13ft
make -j"$(nproc)"
sudo make altinstall
```

Verify the state of the interpreter:

```bash
/usr/local/python3.13ft/bin/python3.13 -c "import sys; print(sys._is_gil_enabled())"
```

If that prints `False`, the build is free-threaded and the GIL is currently disabled. If it prints `True` on a free-threaded build, something in the startup path re-enabled it — usually an imported extension.

Create a virtual environment and install the stack:

```bash
/usr/local/python3.13ft/bin/python3.13 -m venv ft-env
source ft-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install fastapi gunicorn uvicorn redis asyncpg psutil
```

Pin exact versions in your own project. The packages above are listed without pins deliberately: the versions that work change faster than this article will, and a stale pin is worse than no pin when the whole point is ABI compatibility.

## A minimal service to test against

The service below has three endpoints that isolate the three cases you care about: pure CPU, pure I/O, and a mix.

```python
import asyncio
import time
from contextlib import asynccontextmanager

import asyncpg
import psutil
import redis.asyncio as redis
from fastapi import FastAPI

redis_pool = None
pg_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool, pg_pool
    redis_pool = redis.Redis(host="127.0.0.1", port=6379, db=0)
    pg_pool = await asyncpg.create_pool(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        password="postgres",
        database="postgres",
        min_size=2,
        max_size=10,
    )
    yield
    await redis_pool.aclose()
    await pg_pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/sync")
async def cpu_bound():
    start = time.perf_counter()
    _ = sum(i * i for i in range(1_000_000))
    elapsed = time.perf_counter() - start
    return {
        "cpu_ms": int(elapsed * 1000),
        "threads": psutil.Process().num_threads(),
    }


@app.get("/io")
async def io_bound():
    pong = await redis_pool.ping()
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT 1 AS one")
    return {"redis_pong": pong, "pg_rows": len(rows)}


@app.get("/mixed")
async def mixed():
    start = time.perf_counter()
    _ = sum(i * i for i in range(500_000))
    pong = await redis_pool.ping()
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT 1 AS one")
    return {
        "cpu_ms": int((time.perf_counter() - start) * 1000),
        "redis_pong": pong,
        "pg_rows": len(rows),
    }
```

A note on the `/sync` endpoint: it is declared `async def`, so the CPU loop runs on the event loop thread and blocks it. That is intentional here — it makes the CPU cost visible — but it is a bug in a real service. The correct form is a plain `def` endpoint, which FastAPI runs in a threadpool, or an explicit `await asyncio.to_thread(...)`. The behaviour of `asyncio.to_thread` is the interesting part in a free-threaded build: the work genuinely runs on another core instead of contending for the GIL.

Gunicorn configuration:

```python
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
```

Local dependencies:

```bash
docker run -d --name redis7 -p 6379:6379 redis:7-alpine
docker run -d --name pg16 -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16-alpine
```

Run the service in each mode:

```bash
PYTHON_GIL=1 gunicorn -c gunicorn.conf.py app:app   # GIL re-enabled
PYTHON_GIL=0 gunicorn -c gunicorn.conf.py app:app   # GIL disabled
```

`PYTHON_GIL` is a process-level switch. It is read once at interpreter startup. Every gunicorn worker inherits the same state, so you cannot mix GIL-on and GIL-off workers inside one process tree; use separate deployments if you want a canary.

## How to measure it yourself

Do not trust throughput numbers from someone else's hardware, including the ones you would have found in an earlier draft of this article. The right way to decide is to measure on your own workload. Here is what to instrument.

**Load generator.** Locust, k6, or `wrk` all work. A minimal Locust file:

```python
from locust import HttpUser, task, between


class ApiUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def io(self):
        self.client.get("/io")

    @task(1)
    def mixed(self):
        self.client.get("/mixed")

    @task(1)
    def sync(self):
        self.client.get("/sync")
```

Run it headless against each mode in turn, with the same user count, spawn rate, and duration:

```bash
locust -f locustfile.py --headless -u 200 -r 20 --run-time 2m \
  --host http://localhost:8000 --html=report-gil-on.html
```

**What to record per run:**

| Metric | Where it comes from | Why it matters |
|---|---|---|
| Requests per second per endpoint | Load generator summary | The headline throughput number |
| Median and P99 latency | Load generator summary | Tail latency is what users feel |
| CPU utilisation per core | `pidstat -u 1`, `mpstat -P ALL 1` | Tells you whether you are now CPU-bound |
| Resident memory | `ps -o rss= -p <pid>`, or `docker stats` | Free-threading can increase memory; confirm it does not |
| Thread count | `psutil.Process().num_threads()` | Confirms threads are actually being created |
| GIL state at runtime | `sys._is_gil_enabled()` in a health endpoint | Catches silent re-enablement |

**What to compare.** Run the GIL-on build first and treat it as the baseline. Then run the identical build with `PYTHON_GIL=0`. Same machine, same kernel, same load profile, same database state. Any difference you see is the effect of the GIL switch, assuming nothing else changed.

**What a result looks like.** If `/io` is dominated by network round-trips to Redis and PostgreSQL, the event loop is mostly waiting, and the GIL is not the bottleneck. In that case you will see little or no throughput change, and the correct conclusion is that free-threading buys you nothing here. If `/io` is dominated by Python-level work between I/O calls — serialisation, response construction, small computations — then removing the GIL lets those sections overlap across threads, and throughput can rise. The size of the rise depends entirely on the ratio of Python execution time to waiting time in your handler. That ratio is specific to your code; nobody else's benchmark can tell you what it is.

**A worked estimate.** Suppose a handler spends 2 ms per request in Python and 8 ms waiting on I/O, and you run 4 threads. With the GIL, the Python portions serialise: 4 requests require 4 × 2 ms = 8 ms of interpreter time. Without the GIL, the Python portions overlap across 4 cores, so the same 4 requests require 2 ms of wall-clock interpreter time. The I/O portions overlap in both cases. Total wall-clock for 4 requests drops from roughly 8 ms + 8 ms = 16 ms to 2 ms + 8 ms = 10 ms, a 1.6× improvement. Change the split to 8 ms Python and 2 ms I/O and the same arithmetic gives 32 ms + 2 ms = 34 ms versus 8 ms + 2 ms = 10 ms, a 3.4× improvement. Change it to 1 ms Python and 9 ms I/O and you get 4 ms + 9 ms = 13 ms versus 1 ms + 9 ms = 10 ms, a 1.3× improvement. These figures are illustrative, not measured: the point is that the gain is a function of your Python-to-I/O ratio, which you can estimate from a profile before you run any load test.

## Failure modes to expect

### C extensions that were not built for free-threading

This is the dominant failure. Symptoms range from an `ImportError` to a hard `SIGSEGV` to a `Fatal Python error: _PyThreadState_Get: no current thread` message. The worst case is silent: the extension re-enables the GIL for the process, and your free-threaded build behaves exactly like the normal one with no warning.

Detection: add a health endpoint that returns `sys._is_gil_enabled()`, and check it after startup with all your production imports loaded. If it reports `True` on a free-threaded build, an extension re-enabled the GIL.

Remedy: use a version of the package that ships a free-threaded wheel, or build it from source against the free-threaded interpreter. If neither is possible, isolate the package in a subprocess or a separate service.

### Shared mutable state

With the GIL, a surprising amount of code is accidentally correct because bytecode operations on built-in containers are effectively atomic at the interpreter level. Remove the GIL and those assumptions break. A module-level counter incremented from multiple threads is the canonical example:

```python
from threading import Lock, get_ident

request_counters = {}
counter_lock = Lock()


def record_request():
    ident = get_ident()
    with counter_lock:
        request_counters[ident] = request_counters.get(ident, 0) + 1
```

The lock is not optional. It is also not sufficient on its own: any invariant that spans more than one statement needs to be protected as a unit. Audit module-level caches, lazily initialised singletons, and anything that reads-then-writes a shared structure.

Note that asyncio code running on a single event loop is unaffected by this class of bug, because coroutines interleave only at `await` points. The bugs appear when you introduce real threads — a threadpool for `def` endpoints, `asyncio.to_thread`, or a background worker.

### Connection pool sizing

Pools that were sized for a GIL-serialised workload are frequently too small once requests actually run concurrently. The symptom is a timeout waiting to acquire a connection, not a crash, so it shows up as latency rather than errors until the timeout expires. Size the pool against the number of concurrent requests you expect, not the number of workers, and set an explicit acquire timeout so the failure is a clear error rather than a hang.

### Middleware that monkey-patches thread state

Instrumentation and error-reporting libraries sometimes patch thread-local storage or assume a single interpreter-wide lock. Symptoms include missing spans, duplicated spans, and occasional crashes inside the instrumentation rather than your code. Test with instrumentation enabled and disabled; if the free-threaded build only misbehaves with instrumentation on, that is your culprit.

### Silent serialisation

The most expensive failure is the one that produces no error. A single GIL-bound extension anywhere in the process can serialise everything, and the only evidence is that your free-threaded build performs identically to the normal one. Always include the GIL-state check in your deployment's smoke test.

## Observability

Whatever you use for tracing and metrics, the key additions for a free-threaded deployment are:

- A gauge for `sys._is_gil_enabled()`, exported at startup and on every scrape.
- A gauge for thread count and a histogram for connection-pool wait time.
- Per-endpoint latency histograms, so you can see whether the gain is concentrated in I/O-heavy routes.

If you use OpenTelemetry, the FastAPI, asyncpg, and Redis instrumentors each need to be verified against a free-threaded interpreter before you rely on them; a broken instrumentor should not take down the service, so wrap initialisation in a try/except and log the failure.

## A decision checklist

Before switching a service to a free-threaded interpreter, answer these:

1. **Is the workload I/O-bound with significant Python-level work between I/O calls?** If the handler is almost entirely waiting, the GIL was never the bottleneck and the switch will not help.
2. **Does every compiled dependency in the process have a free-threaded build?** If not, can it be isolated?
3. **Does the process create threads at all?** If everything runs on a single event loop, free-threading changes nothing.
4. **Is there shared mutable state outside the event loop?** If yes, it needs locking before the switch, not after.
5. **Are the connection pools sized for real concurrency?** If they were sized for a serialised workload, they are too small.
6. **Can you verify the GIL state at runtime in production?** If not, you cannot tell a successful migration from a silent failure.
7. **Do you have a rollback path?** `PYTHON_GIL=1` on the same build is the cheapest one, but only if the build itself is otherwise identical.

If any answer is "no" or "don't know", resolve that before deploying.

## FAQ

**Does free-threading break asyncio?**
No. asyncio schedules coroutines on a single thread and is unaffected. The change is that work dispatched to real threads — via `asyncio.to_thread` or a threadpool — now runs genuinely in parallel with the event loop instead of contending for the GIL.

**Can I mix GIL-on and GIL-off workers in one process?**
No. `PYTHON_GIL` is read once at interpreter startup and applies to the whole process. Use separate processes or separate deployments.

**How do I tell whether a package is free-thread-safe?**
Check whether a free-threaded wheel exists for your platform and interpreter version. If there is no such wheel and the package contains compiled code, assume it is not safe and plan to build it yourself or isolate it. Pure-Python packages are generally fine, but they can still hold state that assumed GIL-protected atomicity.

**Will this help a CPU-bound service?**
Not meaningfully. Removing the GIL does not add cores. It removes a serialisation point, which helps when threads are waiting on I/O and could otherwise be doing Python work. A CPU-bound service is already limited by core count.

**What about ARM?**
There is no general rule. The gain depends on the same Python-to-I/O ratio as on x86_64, plus whatever differences the platform has in memory model and scheduling. Measure it; do not assume the x86_64 result transfers.

## Your next 30 minutes

Pick one endpoint in your service that is I/O-bound and does a non-trivial amount of Python work per request. Add a temporary route that returns `sys._is_gil_enabled()` and `psutil.Process().num_threads()`. Deploy that build with `PYTHON_GIL=0` to a single canary instance behind your load balancer, hit the route, and confirm the GIL is actually off with all your production imports loaded. If it reports `True`, you have found a GIL-bound extension before it cost you a migration. If it reports `False`, run your existing load test against the canary and the baseline for the same duration and compare P99 latency on that one endpoint. That single comparison tells you more than any published benchmark.
