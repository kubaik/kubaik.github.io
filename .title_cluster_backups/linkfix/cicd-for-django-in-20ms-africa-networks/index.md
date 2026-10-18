# CI/CD for Django in 20ms Africa networks

Teams running Django on small cloud VMs in African regions inherit a set of constraints that most CI/CD guides ignore: 1–2 vCPU runners, 2–4 GB RAM, links whose round-trip time to the nearest registry mirror can sit in the hundreds of milliseconds, and egress billed per gigabyte. A pipeline that is merely "slow" on a fat pipe becomes a repeated failure on a thin one. This article walks through a pipeline designed around those constraints, explains why each piece exists, and shows how to measure whether it is working.

The teaching order below is: environment and caching, image construction, the CI workflow, deployment and migrations, failure modes, observability, then a decision checklist.

## The constraints that drive every design choice

Four properties of the environment determine almost every decision that follows.

**CPU and memory.** A 1 vCPU / 2 GB runner spends a large fraction of its time in `apt-get update`, dependency resolution and compilation. Anything that avoids recompiling is worth more here than on a 16-core machine.

**Round-trip time.** Every network round trip costs the RTT, not the bandwidth. A registry push that is 200 ms away is not slow because of throughput; it is slow because of the number of round trips. Co-locating the registry with the runner is the single highest-leverage change.

**Metered egress.** Pulling a base image on every build is the most common source of avoidable egress. Pinning digests and reusing a local cache turns a per-build cost into a per-change cost.

**Memory headroom at runtime.** The same 2 GB VM often hosts the application. Gunicorn worker counts chosen for a 4 GB box will trigger the OOM killer. Worker sizing must be derived from measured RSS, not copied from a default.

## Step 1 — prepare the runner and a local registry

A self-hosted runner is the norm here, because hosted runners are typically far from the target region and re-download toolchains on every job.

Install a runner (the version and URL pattern below are the documented GitHub Actions runner release layout):

```bash
sudo apt update && sudo apt install -y curl jq
RUNNER_VERSION=2.316.0
curl -sL https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz | tar xz
./config.sh --url https://github.com/your-org/your-repo --token YOUR_TOKEN
sudo ./svc.sh install
sudo ./svc.sh start
```

Cap the Node heap used by the runner so it cannot starve the build:

```ini
RUNNER_TOOL_CACHE=/opt/hostedtoolcache
NODE_OPTIONS="--max-old-space-size=512"
```

Configure Docker's storage driver explicitly. On modern kernels `overlay2` is the correct choice; `vfs` is a fallback for filesystems that cannot support overlay mounts, and it copies every layer, which is expensive in both disk and time.

```json
{
  "storage-driver": "overlay2"
}
```

A registry mirror is only useful if it is actually closer than the upstream. A public mirror URL is not necessarily in your region, and configuring one that is further away than the origin makes pulls slower. Run a registry on the same host or in the same region instead:

```bash
docker run -d --name registry -p 5000:5000 \
  -v /opt/registry:/var/lib/registry \
  -e REGISTRY_STORAGE_DELETE_ENABLED=true \
  registry:2.8.3
```

Point the runner at it by hostname so pushes avoid a DNS lookup:

```
127.0.0.1   registry.local
```

Warm the cache once by pulling the base image you will use, and record its digest:

```bash
docker pull python:3.11-slim-bookworm
docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim-bookworm
```

That digest is what goes into the Dockerfile. Mutable tags such as `3.11-slim-bookworm` are re-published upstream; pinning the digest makes the layer cacheable across builds and makes the pipeline reproducible.

## Step 2 — build a small runtime image

A naive single-stage build bakes `gcc`, `python3-dev` and `libpq-dev` into the shipped image. Multi-stage builds separate the compile toolchain from the runtime.

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm@sha256:<digest> AS builder
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev libpq-dev && \
    pip install --user --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

FROM python:3.11-slim-bookworm@sha256:<digest> AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY --from=builder /root/.local /root/.local
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    find /root/.local -type d -exec chmod 755 {} +
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000
CMD ["gunicorn", "project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2"]
```

Three details matter. First, `libpq-dev` belongs in the builder stage only; the runtime needs `libpq5` alone. Second, `PATH` must include `/root/.local/bin`, otherwise `gunicorn` will not be found. Third, `.dockerignore` should exclude `.git`, `node_modules`, `*.sqlite3` and any local virtualenv, or `COPY . .` will bloat the context and invalidate the layer cache on every commit.

Build with a persistent local cache so apt and pip layers survive between runs:

```bash
docker build \
  --cache-from type=local,src=/var/cache/buildkit \
  --cache-to type=local,dest=/var/cache/buildkit \
  -t registry.local/django-app:latest .
```

**How to measure the result.** Compare the image size and the build time directly:

```bash
docker images registry.local/django-app --format '{{.Size}}'
time docker build --no-cache -t registry.local/django-app:test .
```

Run the second command once with and once without the cache mounts and record the wall-clock difference. That difference, multiplied by your number of builds per day, is the value of the cache. Do not estimate it; measure it.

## Step 3 — the CI workflow

A two-job workflow keeps tests and image publication separate so a failing test never produces a pushed image.

```yaml
name: cicd
on: [push]
jobs:
  test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: pip-${{ hashFiles('requirements.txt') }}
          restore-keys: pip-
      - name: install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: run tests
        run: pytest --cov=project --cov-fail-under=80

  build-and-push:
    needs: test
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: login to registry
        run: echo "${{ secrets.REGISTRY_PASSWORD }}" | docker login registry.local -u "${{ secrets.REGISTRY_USER }}" --password-stdin
      - name: build image
        run: |
          docker build --cache-from registry.local/django-app:latest \
            -t registry.local/django-app:${{ github.sha }} .
          docker tag registry.local/django-app:${{ github.sha }} registry.local/django-app:latest
      - name: push image
        run: |
          docker push registry.local/django-app:${{ github.sha }}
          docker push registry.local/django-app:latest
```

Two notes on correctness. `actions/cache@v4` is the current major version; older references to `v3` should be updated. And the `retry:` and `delay:` keys shown in some published examples are **not** valid GitHub Actions step keys — they are silently ignored. Retries must be written explicitly:

```yaml
      - name: push image with retry
        run: |
          for i in 1 2 3; do
            docker push registry.local/django-app:${{ github.sha }} && break
            echo "push attempt $i failed, retrying"
            sleep 5
          done
```

Image signing is a separate concern. If your threat model includes untrusted runners or a shared registry, signing with a keyless OIDC-based signer (the class of tooling that implements the Sigstore signing flow) is the standard approach. It is not free: signing adds a network round trip and requires the runner to reach the transparency log, which on a high-RTT link can dominate the push. Measure it before adopting it, and skip it if your runners and registry are both private and single-tenant.

## Step 4 — deployment, health checks and migrations

For a single-VM deployment, a systemd unit that manages a Podman container is simpler and more debuggable than a full orchestrator.

```ini
[Unit]
Description=Django app
After=network-online.target
Wants=network-online.target

[Service]
ExecStartPre=/usr/bin/podman pull registry.local/django-app:latest
ExecStart=/usr/bin/podman run --rm --name django-app --env-file /etc/django/env \
  -p 8000:8000 registry.local/django-app:latest
ExecStop=/usr/bin/podman stop django-app
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Note that `podman start` only works on a container that already exists and is stopped; for a rollout you want `podman run` with `--rm`, or an explicit `podman rm -f` before starting. The unit above pulls the image as a pre-step so the pull failure is visible in `systemctl status` rather than inside the container start.

**Migrations.** Running `manage.py migrate` inside the serving container is a common failure mode: if the container is restarted mid-migration, the schema can be left half-applied. Run migrations as a separate, single-shot step before the new container starts:

```yaml
      - name: run migrations
        run: |
          docker run --rm --env-file /etc/django/env \
            registry.local/django-app:${{ github.sha }} \
            python manage.py migrate --noinput
```

For zero-downtime rollouts, migrations must be backward compatible with the previous release: add columns as nullable, deploy code that writes both old and new shapes, backfill, then remove the old column in a later release. A migration that drops a column in the same deploy that stops using it will break the old container during the overlap window.

**Health checks.** A readiness probe should hit an endpoint that actually verifies dependencies, not just that the process is listening.

```python
# project/views.py
from django.db import connection
from django.http import JsonResponse

def healthz(request):
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        return JsonResponse({"status": "unhealthy"}, status=503)
    return JsonResponse({"status": "ok"})
```

Then poll it after the rollout, with a bounded budget:

```bash
for i in $(seq 1 10); do
  if curl -fsS --max-time 5 http://127.0.0.1:8000/healthz; then exit 0; fi
  sleep 3
done
exit 1
```

An unbounded `curl` with no `--max-time` will hang for the OS TCP timeout on a high-RTT link, which turns a fast failure into a stalled pipeline.

## Failure modes and how to diagnose each

**apt lock contention.** Two builds on the same runner can collide on `/var/lib/apt/lists/lock`. Serialize builds with a concurrency group, or give each job its own cache directory.

**Cache misses after a base-image change.** Pinning a digest freezes the base image; you then need a deliberate process to update it. Schedule a weekly job that pulls the tag, resolves the new digest, and opens a pull request. Without this, the digest silently ages and misses security updates.

**Registry push timeouts.** Distinguish throughput from round trips. Run `docker push` with `time` and compare against a `curl -w '%{time_total}'` to the registry's `/v2/` endpoint. If the push is slow but the endpoint responds quickly, the problem is layer count or size, not latency.

**OOM kills during the build.** `pip install` of a large dependency tree can exceed 2 GB. Add swap, or build the wheel on a larger machine and copy the wheel into the build context.

**Egress spikes.** Instrument by reading the host's network counters before and after each build:

```bash
cat /proc/net/dev
```

Diff the transmit column for the interface carrying registry traffic. That gives you bytes per build without any external service. If the number is close to the size of your base image, the cache is not being reused.

**Gunicorn memory growth.** Measure RSS per worker rather than guessing:

```bash
ps -o rss= -C gunicorn | awk '{sum+=$1} END {print sum/1024 " MB"}'
```

A common working rule on a 2 GB VM is two sync workers with two threads, leaving headroom for the OS, the database client and any background task. Confirm against the measurement above; if total RSS approaches the VM's RAM, reduce workers rather than adding swap.

## Observability that answers real questions

Instrument three things and no more at first: build duration, bytes transferred per build, and rollout outcome.

For build duration and egress, a small script that wraps the build and writes a line to a file is enough:

```bash
START=$(date +%s)
docker build -t registry.local/django-app:$GIT_SHA .
END=$(date +%s)
echo "build_seconds $((END-START)) sha=$GIT_SHA" >> /var/log/cicd/metrics.log
```

For application-level metrics, the Django Prometheus client exposes a `/metrics` endpoint. Add the middleware and app, and ensure the multiprocess directory is set so that metrics from multiple Gunicorn workers are aggregated correctly:

```python
MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    # ... your middleware ...
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]
INSTALLED_APPS = [
    # ... your apps ...
    'django_prometheus',
]
```

```ini
[Service]
Environment=PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

Without `PROMETHEUS_MULTIPROC_DIR`, each worker keeps its own counters and the scraped values are whichever worker answered the scrape — a subtle source of misleading dashboards.

A smoke test that runs against the deployed service, not the test client, catches configuration errors that unit tests cannot:

```python
# tests/test_smoke.py
import os
import requests

def test_health_endpoint_live():
    base = os.environ["DEPLOY_BASE_URL"]
    resp = requests.get(f"{base}/healthz", timeout=5)
    assert resp.status_code == 200
```

Note that Django's test `client.get()` does not accept a `timeout` argument; timeouts belong on the real HTTP call, which is why the smoke test uses `requests` against a live URL.

## Choosing a runner: a checklist rather than a benchmark

Published benchmark tables are worthless for your situation because the variables — RTT, registry location, image size, dependency tree — differ per project. Instead, decide with this checklist:

1. **Is the runner in the same region as the deployment target?** If not, every artifact crosses a long-haul link twice.
2. **Is there a registry on the same host or in the same region?** If not, you are paying egress on every build.
3. **Is the base image pinned by digest and cached?** If not, the first build of the day pays full price.
4. **Are builds serialized?** Concurrent builds on a small runner contend for CPU, disk and apt locks.
5. **Is the rollout bounded?** Every network operation in the pipeline needs a timeout.
6. **Can you measure bytes and seconds per build?** If not, you cannot tell whether any change helped.

A worked sizing example makes the arithmetic concrete. Suppose a base image is 50 MB compressed, your dependency layer is 60 MB, and your application layer is 5 MB. A cold build transfers roughly 115 MB; a warm build with a local registry transfers the application layer only, about 5 MB. At a hypothetical egress price of $0.10/GB (illustrative, not a quoted rate), the difference is about $0.011 per build. At 40 builds a day that is roughly $0.44 a day, or about $160 a year. The larger saving is usually time: if the pull takes 45 seconds cold and 8 seconds warm, the warm path saves 37 seconds per build, which is 25 minutes a day at 40 builds. Substitute your own measured numbers; the point is that the calculation is trivial once you have the two measurements.

## A 30-minute action

Open your CI workflow and find every network operation that has no timeout: `docker pull`, `docker push`, `pip install`, `curl` in a health check. Add an explicit bound to each — `--max-time` for curl, `--timeout` for pip, and a retry loop with a fixed sleep for pushes. Then run one build and record the wall-clock time and the bytes transferred from `/proc/net/dev`. Those two numbers are the baseline against which every later change is judged.
