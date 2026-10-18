# Designing a Low-Cost Internal Developer Platform

Most guides to internal developer platforms assume a clean environment, a patient timeline, and a budget for managed tooling. Production gives teams none of those. This article covers the constraints that actually shape an IDP for a distributed team, the three approaches that usually fail first, one approach that tends to hold up, and — more importantly — how to measure each claim on your own infrastructure rather than trusting someone else's numbers.

## The constraints that shape the design

A typical situation: a startup with 35 engineers spread across three cities, shipping a B2B payments product that handles heavy traffic during peak hours. The monolith runs on Node 20 LTS and a PostgreSQL 15 cluster in a single AWS region. Engineers wait 15–20 minutes for a fresh staging environment per branch. Local dev needs 32 GB RAM and 20 minutes of `docker-compose up`. Environment variables live in three places and drift constantly, so staging is only roughly similar to production.

A recurring failure mode is a single mis-set `REACT_APP_API_BASE_URL` pointing at prod instead of staging, which can consume days of debugging. That class of incident is what an IDP should prevent.

The design constraints that matter most:

1. **Mobile-first engineers on unreliable connections.** Engineers on 3G or 4G who drop to 2G at peak hours. A platform that assumes fibre latency will be abandoned.
2. **Intermittent-connection-tolerant CI/CD.** If CI runners are 120 ms away from the farthest developer, every `git push` feels slow regardless of how fast the build itself is.
3. **Near-zero budget for commercial IDP tools.** Runway is finite and G&A must stay small.

A useful latency bar: under 300 ms round-trip for any API call from a phone on 3G, assuming an 800 ms TCP handshake. Slower than that and engineers stop using the platform. This number is a design target, not a measured result — measure your own.

## What teams try first and why it usually fails

### Option A: Self-hosted GitLab plus Kubernetes

A common first attempt is GitLab Runner on Kubernetes, with GitLab on AWS EKS. The cluster has a fixed monthly cost before a single pipeline runs. Each pipeline pulls many Docker layers totalling gigabytes per job; cache miss rates on first build are high, so every push triggers a full rebuild. The `docker build` step can take 8 minutes on a small runner. BuildKit multi-stage caching helps, but when the base image includes Node, Python, and Chromium for Playwright tests, the cache footprint remains large. Engineers far from the runner region see high ping times, so each push feels like a 3G page load.

- Latency: 450–600 ms per API call from a phone on 3G
- Cost: fixed cluster spend before any work happens
- Pain: 10–12 minutes waiting for a staging environment to become healthy

### Option B: Terraform plus Helm, no platform layer

The next attempt is usually a large amount of Terraform and Helm to spin up ephemeral namespaces. The plan works on a dev laptop in a few minutes, but in CI it takes far longer because runners pull a large base image over a constrained link. Engineers try to run the same Helm chart locally with `k3d`, but on macOS that requires Docker Desktop with significant RAM and still hits latency spikes when pulling images.

The wall comes when injecting environment variables: dozens of secrets across staging and prod. The Helm `--set-file` approach requires base64 encoding per environment, so the result is a long list of command-line arguments with no way to diff them. A single accidental prod value leak is enough to cause a multi-hour outage during a payment spike.

### Option C: A developer portal with a database backend

A unified developer portal looks promising. Deployed on a managed host with a small managed PostgreSQL instance, the first surprise is that the portal expects every plugin to have its own Node backend. Twenty plugins mean twenty separate Node services, each with its own health check and port. Health-check timeouts pile up quickly. The frontend bundle is large; on a 2G connection it takes many seconds to load. Engineers stop using the portal after two tries.

- Latency: over a second per page load on 2G
- Cost: the portal alone, before any environment provisioning
- Pain: no staging environment provisioning, just a catalog

## The approach that tends to hold up

Skip the Kubernetes cluster you can't afford. Build a lightweight IDP around three pillars:

1. Ephemeral environments via CI environments plus an infrastructure-as-code tool with a real programming language SDK
2. A connection-first developer portal that works offline
3. Secrets-as-code that keeps prod safe while letting engineers run prod-like locally

The key insight: most engineers don't need a full Kubernetes cluster. They need a process that gives them a staging URL and an API collection that looks like prod. A Python-SDK infrastructure tool beats HCL here because you can express both the infrastructure and the derived environment variables in one program, rather than maintaining two artifacts that drift.

The stack in this design:

- The infrastructure program provisions a managed Postgres cluster, a managed Redis cluster, a load balancer, and a set of container services.
- The whole stack is torn down when the CI environment is deleted, on a short TTL.
- The pipeline builds a Docker image once per commit SHA and pushes it to a registry. The task definition is parameterized by CI environment variables, so each branch gets its own URL with the same secret shape as prod.
- The portal is a statically generated site on an edge network, with incremental static regeneration. Pages are generated at build time from infrastructure outputs, so portal latency from a distant city is low. A Service Worker caches the portal for offline use — engineers can open it on the bus and still see their staging URLs.
- Secrets live in a managed secrets store. A small function fetches them at runtime and injects them into the container task. Secrets rotate on a schedule, and the function caches them for a few minutes to avoid cold-start latency spikes.

The next sections cover implementation details and, crucially, how to measure each of these claims.

## Implementation details

### 1. Ephemeral environments with CI environments and an IaC program

Create a CI workflow that runs on `push` to any non-main branch. The job uses a CI environment name derived from the pull request number:

```yaml
# .github/workflows/ephemeral-env.yml
name: ephemeral-env
on:
  push:
    branches-ignore: [main]
jobs:
  deploy:
    runs-on: ubuntu-22.04
    environment:
      name: pr-${{ github.event.number }}
      url: https://pr-${{ github.event.number }}.staging.example.com
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pulumi==3.89.0 pulumi-aws==6.48.0
      - run: pulumi up --yes --stack pr-${{ github.event.number }}
        env:
          PULUMI_ACCESS_TOKEN: ${{ secrets.PULUMI_ACCESS_TOKEN }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

The infrastructure program (`infra/__main__.py`) is a few hundred lines. The skeleton below shows the shape; the container definitions are abbreviated because the exact JSON depends on your app:

```python
# infra/__main__.py
import pulumi
from pulumi_aws import ec2, ecs, elbv2, rds, elasticache, iam, lambda_

config = pulumi.Config()
sha = config.require("sha")
env_name = config.require("env_name")

# 1. Managed Postgres, small instance class, gp3 storage
postgres = rds.Instance(
    f"postgres-{env_name}",
    allocated_storage=20,
    engine="postgres",
    engine_version="15.6",
    instance_class="db.t4g.micro",
    username="admin",
    password=config.require_secret("db_password"),
    skip_final_snapshot=True,
)

# 2. Managed Redis, small node type
redis = elasticache.Cluster(
    f"redis-{env_name}",
    engine="redis",
    node_type="cache.t4g.small",
    num_cache_nodes=1,
    parameter_group_name="default.redis7",
)

# 3. Container cluster
cluster = ecs.Cluster(f"cluster-{env_name}")

# 4. Secrets function (small memory footprint)
secrets_lambda = lambda_.Function(
    f"secrets-{env_name}",
    runtime="python3.11",
    handler="lambda_function.handler",
    code=pulumi.AssetArchive({
        ".": pulumi.FileArchive("lambda_code/")
    }),
    memory_size=128,
    timeout=3,
    environment={
        "variables": {
            "SECRET_ARN": config.require("secret_arn"),
        }
    },
)

# 5. Task definition with secrets injected
# execution_role_arn, task role, container definitions, and log config
# are omitted here; they follow the standard ECS pattern.
task_def = ecs.TaskDefinition(
    f"task-{env_name}",
    family=f"app-{env_name}",
    network_mode="awsvpc",
    requires_compatibilities=["FARGATE"],
    cpu="1024",
    memory="2048",
    execution_role_arn=execution_role.arn,
    container_definitions=pulumi.Output.all(
        postgres.endpoint,
        redis.cache_nodes[0].address,
        secrets_lambda.arn,
    ).apply(lambda args: "[ /* container definitions */ ]"),
)

# 6. Service and load balancer target group
service = ecs.Service(
    f"service-{env_name}",
    cluster=cluster.arn,
    task_definition=task_def.arn,
    desired_count=1,
    network_configuration={
        "subnets": subnet_ids,
        "security_groups": [security_group.id],
    },
    load_balancers=[{
        "target_group_arn": target_group.arn,
        "container_name": "app",
        "container_port": 3000,
    }],
)

pulumi.export("url", f"https://{env_name}.staging.example.com")
```

Set a short TTL in the CI environment settings so the stack is destroyed automatically. The IaC tool tags resources with the stack name, so Cost Explorer can show spend per environment.

### 2. Developer portal: static generation plus a Service Worker

The portal is a statically generated site built with `next build` and deployed to an edge network. Incremental static regeneration keeps it in sync with infrastructure outputs on a short interval.

```javascript
// pages/index.js
import { getEnvironments } from '../lib/pulumi-exports';

export async function getStaticProps() {
  const envs = await getEnvironments();
  return { props: { envs }, revalidate: 300 };
}

function Portal({ envs }) {
  return (
    <ul>
      {envs.map(env => (
        <li key={env.name}>
          <a href={env.url}>{env.name}</a>
          <span>{env.status}</span>
        </li>
      ))}
    </ul>
  );
}

export default Portal;
```

Add a Service Worker (`public/sw.js`) that caches the portal shell for offline use. Note that precaching hashed build assets with a wildcard is not possible — enumerate the actual files at build time or use a stale-while-revalidate strategy for the shell only:

```javascript
// public/sw.js
const CACHE = 'portal-v1';

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll([
      '/',
      '/manifest.json'
    ]))
  );
});

self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
```

### 3. Secrets-as-code with a managed secrets store and a function

Write a small Python Lambda that fetches secrets at runtime and returns them. The container task references the secret by ARN via the `secrets` field in the container definition, so the plaintext never appears in the task definition.

```python
# lambda_code/lambda_function.py
import os
import json
import boto3
from datetime import datetime, timedelta, timezone

secrets_client = boto3.client('secretsmanager')
cache = {}

def handler(event, context):
    secret_arn = os.environ['SECRET_ARN']
    now = datetime.now(timezone.utc)
    entry = cache.get(secret_arn)
    if entry is None or (now - entry['ts']) > timedelta(minutes=5):
        secret = secrets_client.get_secret_value(SecretId=secret_arn)
        cache[secret_arn] = {
            'value': json.loads(secret['SecretString']),
            'ts': now,
        }
    return cache[secret_arn]['value']
```

Rotate secrets on a schedule via the managed secrets store. The function's small memory footprint keeps cold starts low, but the only way to know your actual cold-start latency is to measure it — see the next section.

## How to measure this yourself

Every number in this design should come from your own measurements. Here is what to instrument and how.

### Staging spin-up time

Instrument the CI job to record timestamps at three points: job start, `pulumi up` start, and the first successful health check on the new URL. Emit them as a CI annotation or write them to a metrics store. Compare the p50 and p95 across the last 50 runs. The metric that matters is time from `git push` to a green health check, not the duration of any single step.

### API p95 latency from a distant city

Use a synthetic monitoring tool that runs from a node in your farthest region, or a simple `curl` loop from a phone with a stopwatch. Measure p50 and p95 over at least 100 requests. Do not trust a single sample. To get a real 3G profile, throttle the connection: on Android, use the developer options to set network type; on iOS, use Network Link Conditioner on a tethered Mac.

### Portal page load

Use a web performance tool that supports location and connection throttling. Record Time to First Byte, First Contentful Paint, and Largest Contentful Paint from a location near your farthest developer. Compare against the same page served without a Service Worker (use a private window or clear the cache).

### Monthly infrastructure cost

Tag every resource with the environment name and the stack name. Use the cloud provider's cost explorer to group by tag. Compare against the sum of the line items below, which are illustrative at 2026 list prices in a European region:

| Line item | Illustrative monthly cost |
|---|---|
| CI runner minutes | $120 |
| IaC SaaS | $50 |
| Managed Postgres (small) | $35 |
| Managed Redis (small) | $15 |
| Load balancer | $17 |
| Container service (1 vCPU, 2 GB) | $160 |
| Secrets rotation function | $5 |
| Edge network for portal | $10 |
| **Total** | **$412** |

These figures are illustrative. Your actual spend depends on region, instance sizes, and how many environments run concurrently. The point of the table is the shape of the cost, not the total.

### Cache miss rate

Instrument your CI to log whether the Docker layer cache was hit. Most CI systems expose this in the build log. Compute `misses / total_builds` over a rolling window. A single image per SHA typically reduces misses, but the only way to know by how much is to measure before and after.

### Secrets leak incidents

Track these as a count over time, not a percentage. A single incident is a data point; the metric that matters is whether the count is zero over the last N months.

## Failure modes to plan for

**The function that fetches secrets times out.** Set the function timeout to a few seconds and have the container task wait longer than that before failing. If the function times out, the task fails and the CI job marks the environment as failed. Log the timeout to your monitoring system and alert. In practice, timeouts are rare if the function caches secrets for a few minutes, but they do happen during cold starts under load.

**Database migrations race with app startup.** Run migrations in a sidecar container in the same task, with a retry loop that waits for the database to be ready. Give the migration container a distinct name and a `dependsOn` relationship so the app container starts only after migrations complete. Disable migrations in prod-like environments where you don't want them running automatically.

**Staging environments can read prod secrets.** Use a policy that prevents the staging function from accessing prod secret ARNs. The staging stack should reference a different secret ARN that contains only staging values. Use the IaC tool's `protect` flag to prevent accidental deletion of secret ARNs in staging stacks.

**Load balancer limits.** Each environment creates a load balancer, and the cloud provider has a soft limit per region. You will hit it around 20–25 environments. Options: request a limit increase, or reuse a single load balancer with path-based or host-based routing. Path-based routing can reduce the count significantly but adds routing complexity.

**Data egress from webhooks.** Payment webhooks call back to ephemeral environments. Each webhook triggers a small amount of egress. At a few thousand webhooks per day, the egress bill is small but non-zero. Moving the webhook handler to a dedicated function that forwards to a chat channel can cut egress substantially.

**Idle environments.** Teams commonly discover that a meaningful fraction of environments are idle for hours. A scheduled job that deletes environments after a period of no traffic (using load balancer access logs) cuts idle spend without hurting developer experience. Measure your own idle rate before assuming a number.

## A decision checklist

Before committing to this design, answer these:

- Can you measure staging spin-up time today? If not, that's step one.
- Do you know your p95 API latency from your farthest developer's location? If not, measure it before optimizing anything.
- Is your CI runner region close to your developers, or close to your infrastructure? These are different optimizations.
- Do you have a policy that prevents staging from reading prod secrets? If not, that's a higher priority than any latency work.
- Do you know your current monthly spend broken down by environment? If not, tagging is the prerequisite.
- Can your team operate a managed secrets store and a small function, or would that be a new operational burden? Be honest.

## Next step (do this in the next 30 minutes)

Open your terminal and measure one number: the time from `git push` to a green staging health check on your current setup. If you don't have staging, measure the time from `git push` to a green CI run.

```bash
# Rough timing: record the current time, push, then poll your staging URL
date -u +"%Y-%m-%dT%H:%M:%SZ"
git commit --allow-empty -m "timing probe" && git push
# Then poll the staging health endpoint until it returns 200
while ! curl -sf https://staging.example.com/health > /dev/null; do sleep 5; done
date -u +"%Y-%m-%dT%H:%M:%SZ"
```

Write the two timestamps down. That number is your baseline. Every design decision in this article should be justified by whether it moves that number, and you now have the only measurement that matters.
