# Internal platform adoption: one metric kills velocity

## The adoption layer is a latency problem

Internal developer platforms usually fail for a mundane reason rather than an exotic one. The tooling works, the golden paths exist, the catalog is populated, and adoption still stalls. The recurring first-order cause is feedback latency: the delay between a developer's action and the platform's response that matters to their immediate workflow.

Security policy friction, RBAC complexity, and Kubernetes networking are real pain points, but they are second-order. A developer who types `kubectl apply` instead of clicking a dashboard button is usually not making a security argument. They are making a latency argument: the platform has not answered "how long will this take?" quickly enough to be worth the detour.

This article describes a concrete implementation: a thin backend service that records rollout timing per deployment, caches it, and exposes it so a catalog card can show it. It also covers the measurement approach so you can decide for yourself whether the metric moves adoption in your organization, rather than trusting a number from someone else's context.

## What feedback latency means and how to measure it

Define feedback latency precisely before instrumenting anything. A useful working definition:

**feedback latency = timestamp(platform surfaces the answer) − timestamp(developer action that triggered the question)**

For a deployment, the developer action is a push or a `kubectl set image`. The answer they want is "the rollout will take N seconds" or "the rollout took N seconds." The platform's response is the moment that number appears in their UI.

To measure it you need three timestamps:

1. The moment the triggering action occurred (commit time, image tag change, or an explicit annotation).
2. The moment the rollout actually started (not when the Deployment object was created).
3. The moment the number rendered in the developer's interface.

Timestamps 1 and 2 are usually conflated, and that conflation is the most common source of misleading numbers. Kubernetes `metadata.creationTimestamp` marks when the Deployment object was created. If a team updates an existing Deployment's image, the object is not recreated, so `creationTimestamp` can be hours or days stale. Recording latency from `creationTimestamp` produces a number that is systematically wrong in the direction of looking worse than reality.

**How to measure it properly:** instrument the API endpoint that serves the latency value and record the delta between the request arrival and the response send. Pair that with a client-side timing event when the value renders. Compare the two. If the server responds in 40 ms and the badge appears 4 seconds later, your problem is the frontend, not the backend.

## Prerequisites

- A Kubernetes cluster (any distribution) with at least a few Deployments.
- A Backstage instance on a current Node LTS release, with the Kubernetes plugin enabled.
- Redis for caching timing data.
- Prometheus and Grafana for observability.
- A source repository with CI you can modify to emit an annotation.

## Step 1 — environment setup

Scaffold a Backstage app:

```bash
npx @backstage/create-app@latest --name idp-latency-demo
cd idp-latency-demo
```

Install the Kubernetes plugin for the backend:

```bash
yarn add --cwd packages/backend @backstage/plugin-kubernetes @kubernetes/client-node
```

Configure the plugin to reach your cluster. In `app-config.yaml`:

```yaml
kubernetes:
  serviceLocatorMethod:
    type: multiTenant
  clusterLocatorMethods:
    - type: config
      clusters:
        - name: prod
          url: https://<CLUSTER-URL>
          authProvider: serviceAccount
          caData: <base64-encoded-ca>
          serviceAccountToken: ${K8S_SA_TOKEN}
```

Note on authentication: the exact `authProvider` values available depend on the plugin version and your cluster's auth setup. Managed clusters often require short-lived tokens rather than a static service account token. If your token expires, the plugin typically returns empty catalog cards rather than an obvious error, which is itself a confusing failure mode. Check the backend logs for Kubernetes client errors before assuming the catalog is empty for a real reason.

Run Redis locally for development:

```bash
docker run --rm -p 6379:6379 redis:7-alpine
```

Install a Redis client in the backend:

```bash
yarn add --cwd packages/backend redis
```

## Step 2 — record rollout timing correctly

The core problem is that you need a timestamp that marks the start of a rollout, not the creation of the Deployment object. The reliable approach is to have your CI pipeline write an annotation immediately before it changes the image.

```yaml
# CI snippet: stamp the rollout start, then update the image
- name: Deploy to staging
  run: |
    kubectl annotate deployment/myapp \
      platform.example.com/rollout-started-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --overwrite
    kubectl set image deployment/myapp myapp=myapp:${GIT_SHA}
```

Using a custom annotation namespace (`platform.example.com/...`) avoids colliding with Kubernetes' own `deployment.kubernetes.io/revision`, which is an integer revision counter, not a timestamp. Treating the revision as a timestamp is a common bug and produces nonsense latency values.

Now build the collector. Create `packages/backend/src/plugins/latency-collector.ts`:

```typescript
import { createRouter } from '@backstage/backend-common';
import express from 'express';
import type { RedisClientType } from 'redis';
import { KubernetesClient } from '@kubernetes/client-node';

const STARTED_AT_ANNOTATION = 'platform.example.com/rollout-started-at';
const TTL_SECONDS = 3600;

export const createLatencyCollectorRouter = async (options: {
  redisClient: RedisClientType;
  k8sClient: KubernetesClient;
}) => {
  const { redisClient, k8sClient } = options;
  const router = express.Router();
  const watch = new k8sClient.Watch();

  watch.watch(
    '/apis/apps/v1/deployments',
    {},
    async (type, obj: any) => {
      if (type !== 'MODIFIED' && type !== 'ADDED') return;

      const ns = obj.metadata?.namespace;
      const name = obj.metadata?.name;
      const startedAt = obj.metadata?.annotations?.[STARTED_AT_ANNOTATION];
      if (!ns || !name || !startedAt) return;

      const conditions: any[] = obj.status?.conditions ?? [];
      const available = conditions.find(
        (c) => c.type === 'Available' && c.status === 'True',
      );
      if (!available) return;

      const latencyMs = Date.now() - new Date(startedAt).getTime();
      if (Number.isNaN(latencyMs) || latencyMs < 0) return;

      await redisClient.setEx(`deploy:${ns}:${name}`, TTL_SECONDS, String(latencyMs));
    },
    (err) => console.error('deployment watch error', err),
  );

  router.get('/latency/:ns/:name', async (req, res) => {
    const { ns, name } = req.params;
    try {
      const latency = await redisClient.get(`deploy:${ns}:${name}`);
      if (latency) {
        return res.json({ ns, name, latencyMs: Number(latency), source: 'cache' });
      }
    } catch (err) {
      console.error('redis read failed', err);
    }
    return res.json({ ns, name, latencyMs: null, source: 'none' });
  });

  return router;
};
```

Two things to note about the watch. First, the watch callback is async, and the watcher will not await it; unhandled promise rejections from Redis writes must be caught inside the callback or they will surface as process-level errors. Second, `Available: True` is not the same as "rollout complete" for every strategy. For a rolling update with `maxUnavailable` greater than zero, `Available` can be true while old pods are still terminating. If you need strict completion, check `updatedReplicas === spec.replicas` and `unavailableReplicas === 0` as well.

Register the router in `packages/backend/src/index.ts`:

```typescript
import { createClient } from 'redis';
import { KubernetesClient } from '@kubernetes/client-node';
import { createLatencyCollectorRouter } from './plugins/latency-collector';

async function main() {
  const redis = createClient({ url: process.env.REDIS_URL ?? 'redis://localhost:6379' });
  redis.on('error', (e) => console.error('redis error', e));
  await redis.connect();

  const k8sClient = new KubernetesClient();
  const latencyRouter = await createLatencyCollectorRouter({
    redisClient: redis,
    k8sClient,
  });
  apiRouter.use('/latency', latencyRouter);
}
```

## Step 3 — backfill and edge cases

For existing Deployments with no `rollout-started-at` annotation, there is no honest latency figure to compute. You have two options: skip them, or compute a value from `status.conditions[Available].lastTransitionTime` and label it clearly as an estimate. Skipping is usually better, because a wrong number is worse than a missing one when the whole point is developer trust.

```typescript
// scripts/backfill.ts
import { createClient } from 'redis';
import { KubernetesClient } from '@kubernetes/client-node';

const STARTED_AT_ANNOTATION = 'platform.example.com/rollout-started-at';

async function backfill() {
  const redis = createClient({ url: 'redis://localhost:6379' });
  await redis.connect();

  const k8s = new KubernetesClient();
  const list = await k8s.listDeploymentForAllNamespaces();
  let recorded = 0;
  let skipped = 0;

  for (const d of list.items) {
    const ns = d.metadata?.namespace;
    const name = d.metadata?.name;
    const startedAt = d.metadata?.annotations?.[STARTED_AT_ANNOTATION];
    if (!ns || !name || !startedAt) {
      skipped++;
      continue;
    }
    const latencyMs = Date.now() - new Date(startedAt).getTime();
    if (Number.isNaN(latencyMs) || latencyMs < 0) {
      skipped++;
      continue;
    }
    await redis.setEx(`deploy:${ns}:${name}`, 3600, String(latencyMs));
    recorded++;
  }

  console.log({ recorded, skipped });
  await redis.quit();
}

backfill().catch((e) => {
  console.error(e);
  process.exit(1);
});
```

Run it once with your TypeScript runner of choice, for example `npx tsx scripts/backfill.ts`.

### Failure mode: rollback leaves a stale value

After a rollback, the Deployment object still exists and the cached latency still describes the previous rollout. Delete the key when the annotation changes:

```typescript
// inside the watch callback, before computing latency
const lastSeenKey = `seen:${ns}:${name}`;
const lastSeen = await redisClient.get(lastSeenKey);
if (lastSeen && lastSeen !== startedAt) {
  await redisClient.del(`deploy:${ns}:${name}`);
}
await redisClient.setEx(lastSeenKey, TTL_SECONDS, startedAt);
```

### Failure mode: image pull stalls

If the image registry is slow or the image is missing, the rollout stalls in a way the Deployment conditions may not reflect for minutes. Watch pod events and record the first container start time as a separate signal:

```typescript
const podWatch = new k8sClient.Watch();
podWatch.watch(
  '/api/v1/pods',
  {},
  async (type, obj: any) => {
    if (type !== 'MODIFIED' && type !== 'ADDED') return;
    const ns = obj.metadata?.namespace;
    const name = obj.metadata?.labels?.['app.kubernetes.io/name'];
    const started = obj.status?.containerStatuses?.[0]?.state?.running?.startedAt;
    if (ns && name && started) {
      await redisClient.setEx(`pod-start:${ns}:${name}`, TTL_SECONDS, started);
    }
  },
  (err) => console.error('pod watch error', err),
);
```

### Failure mode: Redis unavailable

If Redis is down, the endpoint should degrade rather than fail. Return `latencyMs: null` and let the UI render "n/a". Do not silently substitute a Prometheus value under the same field name, because the two numbers have different semantics (one is measured from your own annotation, the other from whatever the exporter observes). If you do fall back, include a `source` field so the UI can label it.

```typescript
router.get('/latency/:ns/:name', async (req, res) => {
  const { ns, name } = req.params;
  try {
    const latency = await redisClient.get(`deploy:${ns}:${name}`);
    if (latency) {
      return res.json({ ns, name, latencyMs: Number(latency), source: 'cache' });
    }
  } catch {
    return res.json({ ns, name, latencyMs: null, source: 'unavailable' });
  }
  return res.json({ ns, name, latencyMs: null, source: 'none' });
});
```

### Failure mode: unbounded key growth

Short-lived experiment services create keys like `deploy:experiment-12345:api` that are never read again. The 3600-second TTL bounds each key's lifetime, but if your cluster creates thousands of Deployments per hour, the steady-state key count is still `creation_rate × TTL`. Compute it: at 100 new Deployments per hour and a 3600-second TTL, steady state is roughly 100 keys. At 10,000 per hour it is 10,000 keys, and each key plus its value is on the order of a few hundred bytes. Set `maxmemory` and `maxmemory-policy allkeys-lru` so the cache evicts rather than growing until Redis rejects writes.

## Step 4 — observability and tests

Three signals are worth tracking:

1. **Rollout latency distribution** — p50, p95, p99 per service over a rolling window. This is the number developers see, so its shape matters more than its mean.
2. **Cache hit ratio** — `redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)`. A low ratio means the TTL is too short relative to how often cards are viewed.
3. **Endpoint response time** — histogram of the `/latency` handler duration. This is the platform's own feedback latency, and it should be well under the UI's render budget.

An alert on cache miss ratio:

```yaml
- alert: LatencyCacheMissSpike
  expr: rate(redis_keyspace_misses_total[5m]) / (rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])) > 0.25
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Latency cache miss ratio above 25%"
```

A unit test for the router, using a mocked Redis client rather than a live one:

```typescript
import express from 'express';
import request from 'supertest';
import { createLatencyCollectorRouter } from '../plugins/latency-collector';

test('returns cached latency', async () => {
  const redis = {
    get: async () => '1200',
    setEx: async () => 'OK',
    del: async () => 1,
  };
  const k8sClient = { Watch: class { watch() {} } };

  const router = await createLatencyCollectorRouter({
    redisClient: redis as any,
    k8sClient: k8sClient as any,
  });

  const app = express().use(router);
  const res = await request(app).get('/latency/default/api');

  expect(res.status).toBe(200);
  expect(res.body.latencyMs).toBe(1200);
  expect(res.body.source).toBe('cache');
});

test('returns null when cache misses', async () => {
  const redis = { get: async () => null, setEx: async () => 'OK', del: async () => 1 };
  const k8sClient = { Watch: class { watch() {} } };

  const router = await createLatencyCollectorRouter({
    redisClient: redis as any,
    k8sClient: k8sClient as any,
  });

  const app = express().use(router);
  const res = await request(app).get('/latency/default/api');

  expect(res.status).toBe(200);
  expect(res.body.latencyMs).toBeNull();
});
```

Mocking the Kubernetes client at the SDK boundary is fragile because the SDK's shape changes between versions. Mocking it at the HTTP boundary — intercepting the watch request — is more stable but requires more setup. For a unit test of the router, injecting a minimal fake is usually sufficient.

## How to tell whether this actually helps

Do not assume that surfacing latency improves adoption. Measure it. The measurement is straightforward if you plan it before shipping:

**Instrument:**
- A client-side event when the latency badge renders, including the service name and the value shown.
- A client-side event when a developer clicks through to the platform from the catalog.
- A count of platform-initiated deploys versus direct `kubectl` deploys, if you can distinguish them (a CI annotation makes this possible).

**Compare:**
- Adoption rate for services whose cards show latency versus services whose cards do not. If you roll out the badge gradually, you get a natural comparison group.
- Time-to-first-deploy for new services before and after the badge appears.
- The ratio of platform deploys to direct deploys over the same window.

**Watch for confounds:**
- Teams with mature CI will show higher platform adoption regardless of the badge.
- Services with frequent deploys generate more latency data, so their cards are more likely to show a value, which biases any comparison that does not control for deploy frequency.

A useful sanity check is to look at whether developers who see the badge change their behavior. If the badge renders and nothing changes, the metric is decorative.

## A decision checklist

Before building this, confirm each of these:

- [ ] You can produce a trustworthy rollout-start timestamp. If your CI cannot annotate before the image change, stop here — the numbers will be wrong.
- [ ] You have a place to display the value that developers already look at. A new dashboard nobody opens will not change behavior.
- [ ] You have a way to measure the before-and-after. Without a comparison group, you cannot distinguish a real effect from a seasonal one.
- [ ] You have an owner for the cache. Redis without `maxmemory` set is a future incident.
- [ ] You have decided what "n/a" means and how it renders. A missing number is fine; a wrong number is not.

## FAQ

**How do you handle multi-cluster deployments?**
Include the cluster in the cache key (`deploy:${cluster}:${ns}:${name}`) and aggregate at read time. Do not average across clusters unless the clusters are comparable; a staging cluster and a production cluster have different rollout characteristics.

**What if the team uses Helm instead of raw manifests?**
The annotation approach still works. Stamp the annotation on the Deployment object via a Helm hook or a post-renderer, then read it the same way. Reading Helm release history is an alternative, but it introduces a second source of truth for timing.

**Can this feed a scorecard instead of a catalog card?**
Yes, if your scorecard system reads entity annotations. Store the latency in an annotation and reference it in the scorecard rule. The same caveat applies: a scorecard that shows a stale number is worse than one that shows nothing.

**What if Redis is down during a rollout?**
The endpoint returns `latencyMs: null` with `source: "unavailable"`. The UI should render "n/a" rather than a zero, because zero reads as "instant" and is actively misleading.

## Take the next 30 minutes

Pick one Deployment in a non-production cluster and prove the whole path end to end:

1. Annotate it with a rollout-start timestamp:
```bash
kubectl annotate deployment/myapp \
  platform.example.com/rollout-started-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --overwrite
```
2. Query the endpoint your collector exposes and confirm it returns a number:
```bash
curl -s http://localhost:7007/api/latency/default/myapp
```
3. Time the round trip:
```bash
curl -s -o /dev/null -w '%{time_total}\n' http://localhost:7007/api/latency/default/myapp
```

If the endpoint returns `null`, the annotation is missing or the watch is not running. If the round trip exceeds roughly 200 ms, the problem is in your serving path, not in Kubernetes. That single number — the response time of the endpoint that answers "how long will this take?" — is the one to track first.
