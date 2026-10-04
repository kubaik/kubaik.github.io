# A Solo-Project Deploy Pipeline That Stays Under $20/Month

Most deployment tutorials show the happy path: a workflow file, a `git push`, a green checkmark. What they rarely show is the accounting — which line items actually bill, which defaults quietly cost money, and how to verify any of it on your own project.

This article walks through a minimal pipeline for a single Python web service: GitHub Actions for test and build, a container host for runtime, one YAML file, one TOML file. The goal is a pipeline whose monthly cost can be estimated from documented prices and whose behavior can be measured with commands you run yourself.

## What you'll build and what you need

Prerequisites:

1. A GitHub account.
2. An account with a container-hosting platform that can run a Docker image (Fly.io is used as the example here; the same structure applies to any host that accepts an image and exposes an HTTP service).
3. Python 3.11 or newer, plus `pipx` if you want isolated CLI installs.
4. A single Python web service (Flask or FastAPI) with a handful of dependencies.

The end state:

- A GitHub Actions workflow that runs tests on every push to `main` and deploys on success.
- A container image built from a multi-stage Dockerfile, kept small by copying only installed packages into the runtime layer.
- Automatic HTTPS at the host, with a health check that matches the app's real readiness endpoint.
- A rollback path that consists of re-deploying a previous image tag.
- A cost model you can compute from published prices, not from a screenshot.

If your service is JavaScript, Go, Rust, or .NET, the principles carry over. Swap the Dockerfile steps and the test command; the workflow shape and the host configuration stay the same.

### A note on the numbers in this article

Any cost figure here is derived from stated assumptions (hours per month, requests per day, average response size) and shown as arithmetic. Any latency or memory figure is something you should reproduce on your own machine and host, because it depends on your image, your region, and your traffic. Treat every number as a hypothesis to verify, not a benchmark.

## Step 1 — Set up the project

Create a directory and a virtual environment:

```bash
mkdir solo-pipeline && cd solo-pipeline
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install --upgrade pip setuptools
pip install fastapi uvicorn gunicorn python-dotenv httpx pytest pytest-asyncio
```

Pin versions in `requirements.txt`. Pinning matters because an unpinned dependency can change your image size and your cold start without any code change:

```text
fastapi==0.115.2
uvicorn==0.32.0
gunicorn==21.2.0
python-dotenv==1.0.1
httpx==0.27.0
pytest==8.3.2
pytest-asyncio==0.23.8
```

If you need browser tests, install Playwright separately — it should not live in the production image:

```bash
pip install playwright
playwright install
```

Create a minimal FastAPI app in `app/main.py`:

```python
from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "region": os.getenv("FLY_REGION", "local")}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

Add a test in `tests/test_main.py`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

Install the host CLI and authenticate. The install method varies by platform; on Linux and macOS the vendor script is the standard path:

```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

Create the app without deploying yet:

```bash
fly launch --name solo-app --image none --no-deploy
```

This writes a `fly.toml`. Replace its contents with the following. Note the health check path — the default is `/`, and if your app only answers on `/health`, the health check will fail even though the service is fine:

```toml
app = "solo-app"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = false
  auto_start_machines = true
  min_machines_running = 1
  processes = ["app"]

[[vm]]
  memory = "256mb"
  cpu_kind = "shared"
  cpus = 1

[[http_checks]]
  path = "/health"
  interval = "30s"
  timeout = "5s"
```

Before writing the Dockerfile, run a local build to confirm the toolchain works:

```bash
docker build -t solo-app:local .
docker run --rm -p 8080:8080 solo-app:local
```

Visit `http://localhost:8080` and `http://localhost:8080/health`. If you see JSON, the app is wired correctly. Stop the container with Ctrl+C.

One failure mode worth knowing: Docker ignores files whose names end in a trailing dot, so a file accidentally named `.dockerignore.` is not read, and every file in your working directory — including `.venv` and `.git` — gets copied into the build context. If your builds suddenly take minutes instead of seconds, check the filename first.

## Step 2 — Build the image and the workflow

Create a multi-stage Dockerfile in the project root:

```dockerfile
# ---- base builder ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ---- runtime ----
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgcc-s1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY app/ ./app/

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

Note what is not in this Dockerfile: no `COPY .env .`. Baking secrets into an image means they end up in the layer history and in every registry copy. The `.dockerignore` below excludes `.env` for the same reason — pass secrets at runtime through your host's secret mechanism instead:

```text
.git
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.env
.env.local
.DS_Store
```

Build locally and inspect the result:

```bash
docker build -t solo-app:local .
docker images | grep solo-app
```

You'll see the image size in the output. A `python:3.11-slim-bookworm` base plus FastAPI and a handful of dependencies typically lands somewhere in the low hundreds of megabytes; the exact figure depends on your dependency tree. What matters is that you record it, then re-check it after every dependency change. A single heavy transitive dependency can add tens of megabytes without any visible change to your `requirements.txt` top level.

To find which layer is responsible, use:

```bash
docker history solo-app:local
```

Next, create the workflow at `.github/workflows/deploy.yml`. The `test` job runs first; `build-and-deploy` only runs if `test` succeeds:

```yaml
name: Deploy to Fly.io

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions@v1
        with:
          args: "deploy --remote-only"
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

Generate an API token and store it as a repository secret named `FLY_API_TOKEN`:

```bash
flyctl auth token
```

Commit and tag the first release:

```bash
git add .
git commit -m "init pipeline"
git tag -a v0.1.0 -m "first release"
git push origin main --follow-tags
```

Watch the Actions tab. The first run builds every layer from scratch; subsequent runs reuse cached layers where the inputs are unchanged. To see the difference, compare the "Build" step duration on the first run and on the second. If the second run is not meaningfully faster, your layer ordering is wrong — anything that changes on every commit (application code) should be copied after anything that rarely changes (dependency manifests).

## Step 3 — Handle the failure modes that actually occur

The four issues below are the ones most likely to bite a small service in its first weeks. Each has a concrete fix and a way to verify it.

**1. Health checks timing out.** If your health check interval is longer than your host's patience, or your timeout is longer than your app's worst-case response, the host will kill a healthy machine. Tighten both:

```toml
[[http_checks]]
  path = "/health"
  interval = "15s"
  timeout = "3s"
```

Verify by watching logs during a deploy: `fly logs` should show the check passing on the first or second attempt, not flapping.

**2. Workers killed on SIGTERM.** On redeploy, the host sends SIGTERM and waits. If Gunicorn's graceful timeout is shorter than your longest in-flight request, connections get dropped. Set it explicitly:

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--graceful-timeout", "30", "app.main:app"]
```

**3. Memory growth from per-request clients.** The classic version of this bug is creating an `httpx.AsyncClient` inside a request handler and never closing it. Each instance holds a connection pool; over thousands of requests, memory climbs and never comes back. The fix is a module-level client created once and closed on shutdown:

```python
# app/http_client.py
import httpx

_client: httpx.AsyncClient | None = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=5.0)
    return _client

async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
```

Wire the shutdown into the app's lifespan so it runs exactly once:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.http_client import close_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()

app = FastAPI(lifespan=lifespan)
```

A Gunicorn `post_fork` hook that just logs the worker PID is also useful — it tells you whether workers are being recycled unexpectedly:

```python
# app/gunicorn.py
def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
```

If you use a custom worker class, point Gunicorn at it via `--worker-class app.gunicorn.UvicornWorkerWithCleanup`, but only after the module actually defines that class. A `--worker-class` pointing at a missing attribute fails at boot, and the error message is not always obvious.

**4. Region failover that doesn't fail over.** A single-region deployment has no failover, no matter what the dashboard implies. If you want a second region, declare it and verify it:

```toml
[[vm]]
  memory = "256mb"
  cpu_kind = "shared"
  cpus = 1
  regions = ["iad", "ord"]
```

After deploying, confirm both machines exist with `fly status`, then deliberately stop the primary and time how long until the service answers again. That elapsed time is your real failover number; there is no way to know it without testing it.

### Measuring memory under load

Add a small load script that runs against a local container:

```python
# tests/load.py
import asyncio
import httpx

async def hit_endpoint(url: str, count: int = 1000):
    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = [client.get(url) for _ in range(count)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    url = "http://localhost:8080/"
    asyncio.run(hit_endpoint(url))
```

Run it, and while it runs, sample memory from another terminal:

```bash
docker stats --no-stream
```

Run the load script twice and compare. Flat memory across runs means no leak; a rising baseline means you have one. Note that `pytest tests/load.py` will not execute this script the way you might expect — it has no test functions, so pytest collects nothing. Run it directly with `python tests/load.py`, or wrap the logic in an `async def test_...` function if you want it under pytest.

## Step 4 — Observability, tests, and the cost model

Observability for a solo service can be three things:

- Host logs (`fly logs`) for deploy-time and crash-time events.
- A `/metrics` endpoint for request counts and latency, if you want graphs.
- An error reporter for exceptions, so you see them without tailing logs.

Add metrics to the app. The `prometheus-client` library and `starlette-exporter` middleware are the common pairing:

```python
from fastapi import FastAPI
from prometheus_client import Counter, Gauge
from starlette_exporter import PrometheusMiddleware, handle_metrics

app = FastAPI()

REQUEST_COUNT = Counter("app_requests_total", "Total HTTP Requests", ["method", "endpoint"])
REQUEST_LATENCY = Gauge("app_request_latency_seconds", "Request latency", ["method", "endpoint"])

@app.middleware("http")
async def metrics_middleware(request, call_next):
    from time import time
    start = time()
    response = await call_next(request)
    latency = time() - start
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).set(latency)
    return response

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
```

Add the two dependencies to `requirements.txt`:

```text
prometheus-client==0.19.0
starlette-exporter==0.22.0
```

Be aware that `starlette_exporter` already registers its own request counter and latency histogram. Adding your own middleware on top means you'll have two sets of metrics for the same traffic, which is fine for a solo project but confusing if you later wonder why the numbers don't match. Pick one: either use the exporter's built-in metrics, or write your own middleware and skip the exporter.

Add an error reporter. The initialization is a few lines:

```python
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    integrations=[FastApiIntegration()],
)
```

Add an endpoint that fails on purpose, so you can confirm the integration works before you need it:

```python
@app.get("/boom")
async def boom():
    raise RuntimeError("intentional error for sentry")
```

Set the DSN as a host secret rather than in the image, then deploy and hit the endpoint:

```bash
flyctl secrets set SENTRY_DSN=<your-dsn>
flyctl deploy
curl https://solo-app.fly.dev/boom
fly logs | grep RuntimeError
```

A common trap: FastAPI's exception handlers can swallow an exception before the reporter sees it. If nothing arrives in the dashboard, check that the handler you wrote re-raises, or remove the handler and let the framework's default behavior propagate.

Add a test that the metrics endpoint responds:

```python
import httpx

def test_metrics():
    response = httpx.get("http://localhost:8080/metrics")
    assert response.status_code == 200
    assert b"app_requests_total" in response.content
```

This test requires a running server on port 8080, so it belongs in a separate job or a step that starts the container first. Running it in the same job as unit tests will fail with a connection error.

### The cost model

Costs are arithmetic. State your assumptions, then multiply.

Assumptions for this example:

- One shared-CPU machine running continuously: 730 hours per month.
- Published price of $0.0019 per hour for that machine class.
- 12,000 requests per month at 140 KB average response size.
- 150 CI minutes per month at $0.008 per minute.
- A domain at $9 per year.

The arithmetic:

- Machine: 730 × $0.0019 = $1.39 per month.
- Outbound: 12,000 × 140 KB = 1,680,000 KB ≈ 1.68 GB, which is under a 3 GB free allowance, so $0.
- CI: 150 × $0.008 = $1.20 per month.
- Domain: $9 ÷ 12 = $0.75 per month.

Total: $1.39 + $1.20 + $0.75 = $3.34 per month, well under the $20 target.

The point of writing it out this way is that you can change one assumption and see the effect. If traffic grows to 100,000 requests per month, outbound becomes about 14 GB, which is 11 GB over the 3 GB allowance. At $0.05 per GB, that is $0.55 — still small. The line item that grows fastest for a small project is usually CI minutes, not bandwidth.

Verify your own numbers rather than trusting these. The host's billing page shows actual usage; the CI provider's usage page shows actual minutes. Compare them monthly.

### Measuring what matters

Three things are worth measuring on your own deployment, because no article can measure them for you:

- **Cold start.** After a deploy, time the first request with `curl -w "%{time_total}\n" -o /dev/null -s https://your-app/`. Repeat after the machine has been idle to see whether `auto_stop_machines` is affecting you.
- **Rollback time.** Deploy a known-good tag, then deploy a deliberately broken one, then roll back. Time the whole cycle. If it takes longer than a coffee break, your rollback procedure is not real.
- **Memory under load.** Run the load script and sample `docker stats` as described above. Record the steady-state number so you notice when it changes.

## Common questions

### How do I add a database?

Use a managed Postgres from the same host so the connection stays on the private network:

```bash
fly postgres create --name solo-db
fly postgres attach solo-db -a solo-app
```

The attach command sets a `DATABASE_URL` secret on the app. Do not hardcode the URL in `fly.toml` — the file is committed, and the credential would be too. Read it from the environment in your app instead.

### Can I use this for a Node app?

Yes. Replace the Dockerfile with a Node multi-stage build, change `actions/setup-python` to `actions/setup-node`, and swap `pytest` for your test runner. The host configuration and the workflow structure are unchanged. The image size and cold start will differ; measure them the same way.

### What if outbound bandwidth exceeds the free allowance?

Two options. First, compute the overage: if you serve 140 KB per response and exceed a 3 GB allowance, that is roughly 22,000 requests before you start paying. At $0.05 per GB, doubling that traffic costs a few dollars a month. Second, put a CDN in front of the host. A CDN with a generous free egress tier absorbs most static and cacheable traffic, so the origin only sees cache misses. The trade-off is an extra DNS and cache-invalidation surface to manage — worth it for a content-heavy site, less so for a JSON API where nothing is cacheable.

### How do I set up a custom domain?

```bash
fly certs create solo-app.com
fly certs status
```

Certificate issuance typically completes within a few minutes. Point the domain's A record at the host's anycast addresses, which you can list with `fly ips list`. Before switching a live domain, lower the TTL on the existing record so the change propagates quickly, and keep the old record in place until the new one resolves from a few different networks.

### Does CI caching help?

For public repositories, GitHub Actions cache is free. For private repositories it is billed by storage. Caching `~/.cache/pip` saves the time spent downloading wheels, which for a small dependency set is a few seconds per build. Add it if your builds are slow for that reason; skip it if they are slow because of something else, like a large base image pull.

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

## Your next 30 minutes

Write a script that prints your current monthly estimate from the provider APIs, and run it now. Start with the two figures that dominate a small project's bill — machine hours and CI minutes — and add line items as you discover them. A rough version that prints two numbers and their sum is enough to catch a runaway before the invoice does.

Then deploy the app from Step 1 with the health check path corrected, hit `/health` once, and time the response. You now have a baseline. Everything else in this article is an adjustment to that baseline, and every number in it is something you can check.
